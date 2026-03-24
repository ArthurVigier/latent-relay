"""
ERIS v5 — LatentProbe
======================

Pure representation tool. No dialogue. No generation. No web access.

The zombie model is called here as a **transformation function**:
    input_text → hidden activations (numpy)

This is the only interface the zombie exposes in the new ERIS paradigm.
Claude calls LatentProbe.probe() when DriftDetector signals that its own
latent state has diverged from the reference — it uses the zombie's
activations as an external reference frame to recalibrate, not as a
conversational partner.

Design constraints (hard):
  - max_new_tokens=0 is structural: forward pass only, no generate() call
  - enable_thinking=False on models that support it
  - No web access, no tools, no session state
  - Logging: layer index, output shape, timestamp only

Usage::

    probe = LatentProbe("Qwen/Qwen3-32B", layers=[9, 18, 27], device="cuda")
    activations = probe.probe("What is the halting problem?", pooling="last_token")
    # activations = {9: np.ndarray[2560], 18: np.ndarray[5120], 27: np.ndarray[5120]}

    batch = probe.probe_batch(["text A", "text B"], layers=[9, 18])
    # batch = [{9: arr, 18: arr}, {9: arr, 18: arr}]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from eris.interfaces import ProbeModel

log = logging.getLogger("eris.probe")


# ── Exceptions ────────────────────────────────────────────────────────────────

class ProbeError(RuntimeError):
    """Raised when the probe cannot extract activations safely."""


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """
    Activations extracted from one forward pass.

    Attributes:
        activations:  {layer_idx: np.ndarray[hidden_dim]} — one vector per
                      requested layer, pooled according to the pooling method.
        input_tokens: Number of input tokens processed.
        elapsed_s:    Wall-clock time for the forward pass.
    """
    activations: dict[int, np.ndarray]
    input_tokens: int
    elapsed_s: float


# ── LatentProbe ───────────────────────────────────────────────────────────────

class LatentProbe(ProbeModel):
    """
    Pure representation tool. Input → activations. No text output.

    Implements ProbeModel by delegating to HFProbe.  This keeps backward
    compatibility (callers that imported LatentProbe directly still work)
    while the actual logic lives in eris/backends/probes/hf_probe.py.

    Args:
        model_id: HuggingFace model name or local path.
        layers:   List of layer indices to extract.  Negative indices are
                  resolved against the actual number of layers at load time.
                  Layer -1 = last hidden layer.
        device:   Torch device string.  Defaults to "cuda" if available,
                  falls back to "cpu".
        dtype:    Torch dtype for model weights.  Defaults to bfloat16 on
                  CUDA, float32 on CPU.
    """

    def __init__(
        self,
        model_id: str,
        layers: list[int],
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        from eris.backends.probes.hf_probe import HFProbe
        self._backend = HFProbe(model_id=model_id, layers=layers, device=device, dtype=dtype)

        # Expose attributes callers depend on.
        self.model_id = self._backend.model_id
        self.device   = self._backend.device
        self.dtype    = self._backend.dtype
        self.layers   = self._backend.layers
        self.n_layers = self._backend.n_layers

    # ── ProbeModel interface (delegated to HFProbe) ───────────────────────────

    def probe(
        self,
        input_text: str,
        layers: Optional[list[int]] = None,
        pooling: str = "last_token",
        centered: bool = True,
    ) -> dict[int, np.ndarray]:
        """Extract hidden-state activations for one input."""
        layers = layers if layers is not None else self.layers
        return self._backend.probe(input_text, layers=layers, pooling=pooling, centered=centered)

    def probe_batch(
        self,
        inputs: list[str],
        layers: Optional[list[int]] = None,
        pooling: str = "last_token",
        centered: bool = True,
    ) -> list[dict[int, np.ndarray]]:
        """Extract activations for a batch of inputs."""
        layers = layers if layers is not None else self.layers
        return self._backend.probe_batch(inputs, layers=layers, pooling=pooling, centered=centered)

    def steer(
        self,
        input_text: str,
        direction: np.ndarray,
        alpha: float,
        layers: Optional[list[int]] = None,
        mode: str = "add",
    ) -> dict[int, np.ndarray]:
        """Apply a steering vector and return steered activations."""
        layers = layers if layers is not None else self.layers
        return self._backend.steer(input_text, direction=direction, alpha=alpha, layers=layers, mode=mode)

    def steer_batch(
        self,
        inputs: list[str],
        direction: np.ndarray,
        alpha: float,
        layers: Optional[list[int]] = None,
        mode: str = "add",
    ) -> list[dict[int, np.ndarray]]:
        """Apply steering to a batch of inputs."""
        layers = layers if layers is not None else self.layers
        return self._backend.steer_batch(inputs, direction=direction, alpha=alpha, layers=layers, mode=mode)

    def save_direction(self, name: str, vector: np.ndarray) -> None:
        """Save a named steering direction to the in-memory library."""
        self._backend.save_direction(name, vector)

    def load_direction(self, name: str) -> np.ndarray:
        """Load a named steering direction from the library."""
        return self._backend.load_direction(name)

    def list_directions(self) -> list[str]:
        """Return the names of all saved steering directions."""
        return self._backend.list_directions()

    def __repr__(self) -> str:
        return (
            f"LatentProbe(model={self.model_id!r}, "
            f"layers={self.layers}, device={self.device!r})"
        )

    # ── Legacy shim: probe_batch with layers override ─────────────────────────
    # Old callers passed layers as a keyword to probe_batch; keep working.

    def _forward(self, texts, pooling, centered):
        """Internal forward — delegates to backend. Legacy callers only."""
        return [
            ProbeResult(
                activations=acts,
                input_tokens=0,
                elapsed_s=0.0,
            )
            for acts in self._backend.probe_batch(
                texts, layers=self.layers, pooling=pooling, centered=centered
            )
        ]


