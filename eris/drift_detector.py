"""
ERIS v5 — DriftDetector
========================

Measures latent-state divergence between a reference activation snapshot
and a sequence of subsequent activations.  Triggers a probe consultation
when divergence exceeds the configured threshold.

Role in the new ERIS paradigm:
  Claude extracts activations at regular checkpoints during reasoning.
  DriftDetector compares them to the initial (reference) state and decides
  whether the reasoning trajectory has drifted far enough to warrant a
  consultation with LatentProbe.  It does not modify Claude's output —
  it only signals when to ask for an external reference frame.

Metrics implemented:
  1. Cosine distance per layer  — direction change from reference
  2. L2 distance per layer      — magnitude change from reference
  3. Combined drift_score       — weighted average of cosine distances,
                                  smoothed over a moving window of `window`
                                  recent steps to suppress transient spikes
  4. LLC Score (simplified)     — KL-divergence between top-k activation
                                  rank distributions (reference vs current)
                                  — signals feature-composition shift, not
                                  just directional drift

Usage::

    detector = DriftDetector(threshold=0.3, window=5)
    detector.register_reference(probe.probe(problem_statement))

    for step, reasoning_chunk in enumerate(reasoning_steps):
        acts = probe.probe(reasoning_chunk)
        report = detector.compute_drift(acts, step=step + 1)
        if report.should_consult_probe:
            # pass activations + report to Claude as context
            ...
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("eris.drift")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    """
    Output of a single DriftDetector.compute_drift() call.

    Attributes:
        step:                 The reasoning step this report belongs to.
        drift_score:          Scalar in [0, 1] — higher means more drift.
                              Smoothed over the configured window.
        raw_drift_score:      Unsmoothed drift_score for this step only.
        cosine_distances:     {layer_idx: float} — cosine distance from
                              reference for each layer.
        l2_distances:         {layer_idx: float} — L2 distance from reference.
        llc_scores:           {layer_idx: float} — simplified LLC score
                              (KL-div of top-k rank distribution).
        layers_affected:      Layers ranked by cosine distance (highest first).
        should_consult_probe: True if drift_score > threshold.
        threshold:            The threshold in effect at this step.
    """
    step: int
    drift_score: float
    raw_drift_score: float
    cosine_distances: dict[int, float]
    l2_distances: dict[int, float]
    llc_scores: dict[int, float]
    layers_affected: list[int]
    should_consult_probe: bool
    threshold: float


# ── DriftDetector ─────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Stateful detector that tracks latent-state divergence over a reasoning run.

    Args:
        threshold:  Drift score above which should_consult_probe is True.
                    0.3 is a reasonable starting point — tune based on
                    test_0 correlation results.
        window:     Number of recent steps to average for the smoothed
                    drift_score.  Suppresses single-step spikes.
        llc_k:      Number of top features used for LLC score computation.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        window: int = 5,
        llc_k: int = 32,
    ) -> None:
        self.threshold  = threshold
        self.window     = window
        self.llc_k      = llc_k

        self._reference: Optional[dict[int, np.ndarray]] = None
        self._history: deque[float] = deque(maxlen=window)
        self._step_log: list[DriftReport] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def register_reference(
        self,
        activations: dict[int, np.ndarray],
        step: int = 0,
    ) -> None:
        """
        Store the initial latent state as the reference for all future comparisons.

        Call this once at the start of a reasoning run, with the activations
        from the problem statement or the first reasoning step.

        Args:
            activations: {layer_idx: np.ndarray[hidden_dim]}
            step:        The step number of the reference (informational).
        """
        self._reference = {
            k: v.copy().astype(np.float32) for k, v in activations.items()
        }
        self._history.clear()
        self._step_log.clear()
        log.info(
            "Reference registered at step=%d, layers=%s",
            step, sorted(activations.keys()),
        )

    def compute_drift(
        self,
        activations: dict[int, np.ndarray],
        step: int,
    ) -> DriftReport:
        """
        Compare current activations to the reference and return a DriftReport.

        Args:
            activations: {layer_idx: np.ndarray[hidden_dim]}
            step:        Current reasoning step number (used for logging).

        Returns:
            DriftReport with all metrics populated.

        Raises:
            RuntimeError: If register_reference() has not been called yet.
        """
        if self._reference is None:
            raise RuntimeError(
                "Call register_reference() before compute_drift()."
            )

        # Only compare layers present in both reference and current.
        common_layers = sorted(
            set(self._reference.keys()) & set(activations.keys())
        )
        if not common_layers:
            raise RuntimeError(
                "No layer overlap between reference and current activations."
            )

        cosine_dists: dict[int, float] = {}
        l2_dists: dict[int, float]     = {}
        llc_scores: dict[int, float]   = {}

        for layer in common_layers:
            ref = self._reference[layer].astype(np.float32)
            cur = activations[layer].astype(np.float32)

            cosine_dists[layer] = float(_cosine_distance(ref, cur))
            l2_dists[layer]     = float(np.linalg.norm(cur - ref))
            llc_scores[layer]   = float(_llc_score(ref, cur, k=self.llc_k))

        # Raw drift = mean cosine distance across layers (primary signal).
        raw_score = float(np.mean(list(cosine_dists.values())))
        self._history.append(raw_score)

        # Smoothed drift = mean of recent `window` raw scores.
        smoothed = float(np.mean(self._history))

        # Layers most affected = ranked by cosine distance.
        layers_affected = sorted(
            common_layers,
            key=lambda l: cosine_dists[l],
            reverse=True,
        )

        report = DriftReport(
            step=step,
            drift_score=round(smoothed, 5),
            raw_drift_score=round(raw_score, 5),
            cosine_distances={k: round(v, 5) for k, v in cosine_dists.items()},
            l2_distances={k: round(v, 4) for k, v in l2_dists.items()},
            llc_scores={k: round(v, 5) for k, v in llc_scores.items()},
            layers_affected=layers_affected,
            should_consult_probe=smoothed > self.threshold,
            threshold=self.threshold,
        )
        self._step_log.append(report)

        log.info(
            "step=%d drift=%.4f (raw=%.4f) consult=%s layers=%s",
            step, smoothed, raw_score,
            report.should_consult_probe, layers_affected[:3],
        )
        return report

    def reset(self) -> None:
        """Reset all state for a new reasoning run."""
        self._reference = None
        self._history.clear()
        self._step_log.clear()
        log.debug("DriftDetector reset.")

    @property
    def history(self) -> list[DriftReport]:
        """All DriftReport objects recorded since last reset/reference."""
        return list(self._step_log)

    @property
    def max_drift(self) -> Optional[float]:
        """Peak smoothed drift_score across all recorded steps."""
        if not self._step_log:
            return None
        return max(r.drift_score for r in self._step_log)

    def summary(self) -> str:
        """Human-readable summary of the drift history."""
        if not self._step_log:
            return "DriftDetector: no steps recorded."
        consults = sum(1 for r in self._step_log if r.should_consult_probe)
        lines = [
            f"DriftDetector: {len(self._step_log)} steps, "
            f"{consults} probe consultations triggered",
            f"  threshold={self.threshold}, window={self.window}",
            f"  max_drift={self.max_drift:.4f}",
            "  step | raw   | smoothed | consult",
        ]
        for r in self._step_log:
            flag = "!" if r.should_consult_probe else " "
            lines.append(
                f"  {r.step:4d} | {r.raw_drift_score:.4f} | "
                f"{r.drift_score:.4f}   | {flag}"
            )
        return "\n".join(lines)


# ── Metric helpers (pure functions) ──────────────────────────────────────────

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance in [0, 1].  0 = identical direction, 1 = orthogonal.
    Clipped to [0, 1] to guard against floating-point overflow.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    sim = np.dot(a, b) / (na * nb)
    sim = float(np.clip(sim, -1.0, 1.0))
    return (1.0 - sim) / 2.0  # map [-1,1] → [1,0]


def _llc_score(ref: np.ndarray, cur: np.ndarray, k: int = 32) -> float:
    """
    Simplified LLC (Local Latent Composition) score.

    Approximates whether the *which* features are active has changed, not
    just the direction.  Computes the KL-divergence between the softmax
    distributions over the top-k absolute activation values in ref and cur.

    Returns a value ≥ 0.  Values near 0 mean the same features dominate;
    higher values mean the composition has shifted.
    """
    dim = min(len(ref), len(cur), k)
    if dim < 2:
        return 0.0

    # Top-k indices by absolute magnitude.
    top_ref = np.argsort(np.abs(ref))[-dim:]
    top_cur = np.argsort(np.abs(cur))[-dim:]

    # Build probability distributions over the full dimension using only
    # the union of top-k indices, softmax-normalised.
    all_idx = list(set(top_ref.tolist()) | set(top_cur.tolist()))
    p = np.abs(ref[all_idx]).astype(np.float64) + 1e-10
    q = np.abs(cur[all_idx]).astype(np.float64) + 1e-10
    p /= p.sum()
    q /= q.sum()

    # Symmetric KL: (KL(p||q) + KL(q||p)) / 2
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return (kl_pq + kl_qp) / 2.0
