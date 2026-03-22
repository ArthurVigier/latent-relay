"""
ERIS v5 — Injector
===================
Surgical modification of hidden states stored inside a :class:`LatentThought`.

Phase 0 scope
--------------
The injection targets ``thought.hidden_embedding`` — the last-layer, last-token
hidden state that seeds the next latent rollout.  The KV-cache is NOT modified
directly (that is Phase 2 work).

To propagate an injection into generated text, the caller should follow up with
a new ``engine.think(session_id, "", n_steps=0, inherit_from=[injected_handle])``
to push the modified hidden state back into a KV-cache entry.

Operations
----------
- ``"add"``     : h_new = h_old + scale * v
                  Additive perturbation.  Changes both norm and direction.
- ``"steer"``   : h_new = h_old + scale * v, then renormalized to ‖h_old‖.
                  Steering-vector style: changes direction only, preserves norm.
- ``"replace"`` : h_new = v (vector is used directly, scale is ignored).
                  Full substitution of the hidden state.

Layer / position targeting
--------------------------
- ``layer=-1, position=-1`` (defaults) → modifies ``hidden_embedding`` in-place
  (last transformer layer, last sequence position).
- ``layer=N, position=P`` → modifies the [P, :] row of
  ``thought.layer_hidden_states["layer_N"]`` if that key exists, AND updates
  ``hidden_embedding`` if layer is the last layer.  If the key does not exist,
  raises :class:`InjectionError`.

All tensors involved are on CPU (layer_hidden_states and hidden_embedding are
stored on CPU).  The incoming vector is accepted on any device and moved to CPU.

Thread safety
-------------
The injector modifies the thought in-place under a caller-supplied lock — the
engine's ``self._lock`` should be held for the duration of the call if the
thought could be accessed concurrently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ── Errors ────────────────────────────────────────────────────────────────────

class InjectionError(ValueError):
    """Raised when an injection cannot be performed (bad params, missing data)."""


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class InjectionResult:
    """
    Metrics describing what changed after an injection.

    Attributes:
        status:        Always ``"injected"`` on success.
        operation:     The operation that was applied.
        old_norm:      L2 norm of the hidden state before injection.
        new_norm:      L2 norm of the hidden state after injection.
        cosine_shift:  ``1 − cosine_similarity(h_old, h_new)``.
                       0.0 = no directional change, 2.0 = full reversal.
    """
    status: str
    operation: str
    old_norm: float
    new_norm: float
    cosine_shift: float

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "operation": self.operation,
            "old_norm": round(self.old_norm, 4),
            "new_norm": round(self.new_norm, 4),
            "cosine_shift": round(self.cosine_shift, 4),
        }


# ── Core function ─────────────────────────────────────────────────────────────

def inject(
    thought,                          # LatentThought — imported lazily to avoid circular
    vector: torch.Tensor,
    *,
    operation: str = "add",
    layer: int = -1,
    position: int = -1,
    scale: float = 1.0,
) -> InjectionResult:
    """
    Inject a vector into the hidden state of a stored :class:`LatentThought`.

    Args:
        thought:   The :class:`~engine.LatentThought` to modify in-place.
        vector:    The injection vector.  Shape must be ``[hidden_dim]`` or
                   ``[1, hidden_dim]``.  Accepted on any device; moved to CPU.
        operation: ``"add"``, ``"steer"``, or ``"replace"``.
        layer:     Target layer index (``-1`` = last).  Must match a key in
                   ``thought.layer_hidden_states`` unless it is ``-1``.
        position:  Target sequence position (``-1`` = last token).
        scale:     Scaling factor applied to ``vector`` before injection
                   (ignored for ``"replace"``).

    Returns:
        :class:`InjectionResult` with before/after metrics.

    Raises:
        :class:`InjectionError`: On invalid operation, missing layer key, or
                                 shape mismatch.

    Example — steer the last thought in a session, then re-think::

        result = inject(thought, steering_vec, operation="steer", scale=0.5)
        engine.think(session_id, "", n_steps=0, inherit_from=[thought.handle])
    """
    if operation not in {"add", "steer", "replace"}:
        raise InjectionError(
            f"Unknown operation '{operation}'. Expected 'add', 'steer', or 'replace'."
        )

    # Normalise vector → 1-D float32 CPU
    v = vector.detach().cpu().float().reshape(-1)  # [hidden_dim]

    # ── Resolve the target tensor ─────────────────────────────────────────────
    target, is_embedding = _resolve_target(thought, layer, position, v.shape[0])

    # ── Snapshot before ───────────────────────────────────────────────────────
    old = target.clone()
    old_norm = float(old.norm().item())

    # ── Apply operation ───────────────────────────────────────────────────────
    new = _apply_operation(old, v, operation=operation, scale=scale)

    # ── Write back in-place ───────────────────────────────────────────────────
    target.copy_(new)

    # If we modified a specific position in layer_hidden_states, and that
    # layer is the last one, also update hidden_embedding to stay consistent.
    if not is_embedding and layer == -1:
        _sync_hidden_embedding(thought, new)

    # ── Compute metrics ───────────────────────────────────────────────────────
    new_norm = float(new.norm().item())
    cosine_shift = _cosine_shift(old, new)

    return InjectionResult(
        status="injected",
        operation=operation,
        old_norm=old_norm,
        new_norm=new_norm,
        cosine_shift=cosine_shift,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_target(
    thought, layer: int, position: int, expected_dim: int
) -> tuple[torch.Tensor, bool]:
    """
    Locate the 1-D target slice [hidden_dim] inside the thought.

    Returns (target_slice, is_embedding) where is_embedding=True means we are
    operating directly on thought.hidden_embedding.

    The returned tensor is a VIEW — writing to it modifies the parent tensor.
    """
    # Fast path: layer=-1, position=-1 → hidden_embedding
    if layer == -1 and position == -1:
        if thought.hidden_embedding is None:
            raise InjectionError(
                "thought.hidden_embedding is None — the thought has no hidden state."
            )
        h = thought.hidden_embedding.detach().cpu().float()
        # hidden_embedding may be [1, hidden_dim] or [hidden_dim]
        h = h.reshape(-1)
        if h.shape[0] != expected_dim:
            raise InjectionError(
                f"Vector dim {expected_dim} does not match hidden_embedding dim {h.shape[0]}."
            )
        # Store a mutable 1-D view; we'll write back via thought directly
        thought.hidden_embedding = h.unsqueeze(0)   # normalise to [1, hidden_dim]
        return thought.hidden_embedding[0], True

    # Layer-specific path: operate on layer_hidden_states
    if thought.layer_hidden_states is None:
        raise InjectionError(
            "thought.layer_hidden_states is None — call encode() to populate it, "
            "or use layer=-1 to target hidden_embedding."
        )

    key = "last" if layer == -1 else f"layer_{layer}"
    if key not in thought.layer_hidden_states:
        available = list(thought.layer_hidden_states.keys())
        raise InjectionError(
            f"Layer key '{key}' not found in layer_hidden_states. "
            f"Available: {available}"
        )

    hs = thought.layer_hidden_states[key]  # [seq_len, hidden_dim]
    seq_len = hs.shape[0]

    # Resolve position
    abs_pos = position if position >= 0 else seq_len + position
    if not (0 <= abs_pos < seq_len):
        raise InjectionError(
            f"Position {position} out of range for seq_len={seq_len}."
        )
    if hs.shape[1] != expected_dim:
        raise InjectionError(
            f"Vector dim {expected_dim} does not match layer '{key}' hidden_dim {hs.shape[1]}."
        )

    return hs[abs_pos], False   # view into the row


def _apply_operation(
    h: torch.Tensor,
    v: torch.Tensor,
    operation: str,
    scale: float,
) -> torch.Tensor:
    """
    Apply the injection operation.

    All tensors are float32, 1-D, on CPU.  Returns a NEW tensor (not in-place).
    """
    if operation == "add":
        return h + scale * v

    elif operation == "steer":
        # Add perturbation, then rescale to the original norm.
        original_norm = h.norm()
        steered = h + scale * v
        steered_norm = steered.norm().clamp_min(1e-8)
        return steered * (original_norm / steered_norm)

    elif operation == "replace":
        # Ignore scale — use v directly.
        return v.clone()

    else:
        raise InjectionError(f"Unknown operation '{operation}'.")  # unreachable


def _sync_hidden_embedding(thought, new_slice: torch.Tensor) -> None:
    """
    After modifying the last-layer row in layer_hidden_states, push the updated
    slice back into hidden_embedding for consistency.
    """
    if thought.hidden_embedding is not None:
        thought.hidden_embedding = new_slice.detach().unsqueeze(0)


def _cosine_shift(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Cosine shift = 1 − cosine_similarity(a, b).
    Range [0, 2].  0 = identical direction, 1 = orthogonal, 2 = opposite.
    """
    na = a.norm()
    nb = b.norm()
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cos_sim = float(torch.dot(a / na, b / nb).clamp(-1.0, 1.0).item())
    return 1.0 - cos_sim
