"""
ERIS v5 — Trajectory Tracker
==============================
Records the hidden state at each step of a latent rollout and computes
per-step metrics (norm, cosine displacement from z_0, optional Â-hat score).

Design constraints:
  - Hidden states are copied to CPU immediately after each step to avoid
    accumulating GPU memory (60 steps × ~14 KB/step = ~840 KB per trajectory).
  - The tracker is intentionally thin: it does NOT load any model or analyzer.
    Callers pass in pre-computed values (norm, a_hat) so the tracker stays
    decoupled from heavy dependencies.
  - All public methods are thread-safe via a simple list append (GIL is
    sufficient here; trajectories are per-request, not shared).

Usage:
    tracker = TrajectoryTracker(z0=initial_hidden)

    for step in range(n_steps):
        # ... latent rollout step ...
        tracker.record(step=step, hidden=last_hidden, a_hat_score=0.72)

    result = tracker.to_dict()   # serialisable, ready for JSON response
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ── Per-step snapshot ──────────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """
    Metrics captured at a single latent rollout step.

    All tensor values have already been moved to CPU and converted to Python
    scalars — this object is lightweight and JSON-serialisable.
    """

    step: int

    # L2 norm of the hidden state at this step.
    hidden_norm: float

    # Cosine distance from z_0 (the hidden state before the first rollout step).
    # distance = 1 − cosine_similarity(z_0, z_k).
    # 0.0 at step 0 by definition.
    displacement: float

    # Â-hat agentivity score at this step, or None if the probe is not loaded.
    a_hat: Optional[float] = None

    def to_dict(self) -> Dict:
        d: Dict = {
            "step": self.step,
            "hidden_norm": round(self.hidden_norm, 4),
            "displacement": round(self.displacement, 4),
        }
        if self.a_hat is not None:
            d["a_hat"] = round(self.a_hat, 4)
        return d


# ── Tracker ────────────────────────────────────────────────────────────────────

class TrajectoryTracker:
    """
    Accumulates per-step metrics during a latent rollout.

    Args:
        z0: The initial hidden state (shape ``[1, hidden_dim]`` or
            ``[hidden_dim]``), used as the displacement reference.
            Copied to CPU and float32 on construction.
    """

    def __init__(self, z0: torch.Tensor) -> None:
        # Normalised reference vector on CPU (float32) for cosine distance.
        z0_cpu = z0.detach().cpu().float().reshape(-1)
        self._z0_norm: torch.Tensor = F.normalize(z0_cpu, dim=0)

        self._steps: List[TrajectoryStep] = []

    # ── Recording ──────────────────────────────────────────────────────────────

    def record(
        self,
        step: int,
        hidden: torch.Tensor,
        *,
        a_hat_score: Optional[float] = None,
    ) -> TrajectoryStep:
        """
        Record metrics for one rollout step.

        Args:
            step: Zero-based step index.
            hidden: Hidden state tensor from the model at this step
                    (shape ``[1, hidden_dim]`` or ``[hidden_dim]``).
                    Moved to CPU internally — the GPU tensor is not retained.
            a_hat_score: Pre-computed Â-hat score, or None.

        Returns:
            The ``TrajectoryStep`` that was appended.
        """
        h_cpu = hidden.detach().cpu().float().reshape(-1)

        norm: float = h_cpu.norm().item()
        displacement: float = _cosine_distance(self._z0_norm, h_cpu)

        snap = TrajectoryStep(
            step=step,
            hidden_norm=norm,
            displacement=displacement,
            a_hat=a_hat_score,
        )
        self._steps.append(snap)
        return snap

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def steps(self) -> List[TrajectoryStep]:
        """All recorded steps, in order."""
        return list(self._steps)

    @property
    def total_displacement(self) -> float:
        """Cosine distance between z_0 and the last recorded hidden state."""
        if not self._steps:
            return 0.0
        return self._steps[-1].displacement

    @property
    def max_a_hat(self) -> Optional[float]:
        """Peak Â-hat score across all steps, or None if never recorded."""
        scores = [s.a_hat for s in self._steps if s.a_hat is not None]
        return max(scores) if scores else None

    @property
    def steps_to_convergence(self) -> Optional[int]:
        """
        Estimate the step at which displacement stopped growing significantly.

        Uses a simple heuristic: the last step where the per-step increase
        in displacement exceeded 1 % of the total displacement.
        Returns None if fewer than 2 steps were recorded or total_displacement
        is effectively zero.
        """
        if len(self._steps) < 2:
            return None
        total = self.total_displacement
        if total < 1e-6:
            return None

        threshold = 0.01 * total
        last_significant = 0
        for i in range(1, len(self._steps)):
            delta = self._steps[i].displacement - self._steps[i - 1].displacement
            if delta > threshold:
                last_significant = i
        return last_significant

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_list(self) -> List[Dict]:
        """Return the trajectory as a list of dicts (for JSON responses)."""
        return [s.to_dict() for s in self._steps]

    def summary(self) -> Dict:
        """
        Return a compact summary dict for the ``trajectory_summary`` field
        in /v1/bridge and /v1/latent_think responses.
        """
        return {
            "total_displacement": round(self.total_displacement, 4),
            "max_a_hat": (
                round(self.max_a_hat, 4) if self.max_a_hat is not None else None
            ),
            "steps_to_convergence": self.steps_to_convergence,
            "n_steps": len(self._steps),
        }

    def to_dict(self) -> Dict:
        """Full serialised trajectory including per-step list and summary."""
        return {
            "trajectory": self.to_list(),
            **self.summary(),
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cosine_distance(a_normed: torch.Tensor, b: torch.Tensor) -> float:
    """
    Cosine distance between a pre-normalised vector ``a`` and vector ``b``.

    distance = 1 − cosine_similarity(a, b)
    Clamped to [0, 2] (the theoretical range for cosine distance).
    """
    b_norm = b.norm()
    if b_norm < 1e-8:
        return 0.0
    b_normed = b / b_norm
    cosine_sim: float = torch.dot(a_normed, b_normed).clamp(-1.0, 1.0).item()
    return float(1.0 - cosine_sim)
