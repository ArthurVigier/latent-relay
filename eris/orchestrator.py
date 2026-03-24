"""
ERIS v5 — ERISOrchestrator
============================

Claude as an interpreter of external latent spaces.

In the new ERIS paradigm, Claude is the primary reasoner.  The zombie model
is a pure representation tool (LatentProbe).  The orchestrator connects them:

1.  Claude reasons step by step on the problem.
2.  Every `checkpoint_every` steps, the orchestrator extracts hidden
    activations from the zombie for the current reasoning state.
3.  DriftDetector decides whether the zombie's representation has diverged
    enough from the reference to warrant a recalibration signal.
4.  If yes, the orchestrator calls LatentProbe, formats the activations into
    a structured description, and passes it to Claude as an observation note.
5.  Claude reads the note and produces a "recalibration step" before continuing.
6.  Full log of every consultation and its measured impact is returned.

The zombie never speaks.  It has no opinion.  It is a mirror.

Usage::

    from eris.probe import LatentProbe
    from eris.drift_detector import DriftDetector
    from eris.orchestrator import ERISOrchestrator
    from eris.backends.orchestrators.claude_orchestrator import ClaudeOrchestrator

    probe    = LatentProbe("Qwen/Qwen3-14B", layers=[9, 18], device="cuda")
    detector = DriftDetector(threshold=0.3, window=5)
    llm      = ClaudeOrchestrator()

    orch = ERISOrchestrator(probe, detector, llm)
    result = orch.run("Prove that √2 is irrational.", max_steps=15)
    print(result.final_answer)
    print(result.consultations)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from eris.interfaces import OrchestratorLLM, ReasoningStep

log = logging.getLogger("eris.orchestrator")

_RECALIBRATION_PROMPT = """\
[Latent Probe Observation]

The following is a structural observation about the current reasoning trajectory,
derived from the internal representation of an external model processing the same input.
This is not a correction — it is a reference frame.  Use it if you find it informative,
ignore it if you do not.

{observation}

[End Observation]

Continue reasoning from where you left off.
"""


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class ConsultationRecord:
    """Record of a single probe consultation."""
    step: int
    drift_score: float
    drift_layers_affected: list[int]
    observation_text: str           # what was passed to Claude
    claude_response_preview: str    # first 200 chars of Claude's next response
    elapsed_s: float


@dataclass
class OrchestratorResult:
    """Full output of one ERISOrchestrator.run() call."""
    problem: str
    final_answer: str                    # Claude's final answer text
    n_steps: int                         # total reasoning steps taken
    n_consultations: int                 # probe consultations triggered
    consultations: list[ConsultationRecord]
    reasoning_log: list[dict]            # {step, text, drift_score, consulted}
    max_drift: Optional[float]
    elapsed_s: float


# ── Orchestrator ───────────────────────────────────────────────────────────────

class ERISOrchestrator:
    """
    Connects an OrchestratorLLM (primary reasoner) with a ProbeModel (representation tool).

    Args:
        probe:           Any ProbeModel implementation (LatentProbe, HFProbe, etc.).
        drift_detector:  DriftDetector instance.  Reset at the start of each run().
        llm:             Any OrchestratorLLM implementation (ClaudeOrchestrator, etc.).
        probe_layers:    Which layers to extract at each checkpoint.
                         Defaults to the probe's configured layers.
    """

    def __init__(
        self,
        probe,                        # ProbeModel
        drift_detector,               # DriftDetector
        llm: OrchestratorLLM,
        probe_layers: Optional[list[int]] = None,
    ) -> None:
        self.probe    = probe
        self.detector = drift_detector
        self.llm      = llm
        self.layers   = probe_layers or getattr(probe, "layers", [-1])

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        problem: str,
        *,
        max_steps: int = 20,
        checkpoint_every: int = 3,
        pooling: str = "last_token",
        max_tokens_per_step: int = 512,
    ) -> OrchestratorResult:
        """
        Orchestrate a full reasoning session with probe-based recalibration.

        Args:
            problem:            The problem for Claude to reason about.
            max_steps:          Maximum reasoning steps before forcing a conclusion.
            checkpoint_every:   Extract probe activations every N steps.
            pooling:            Pooling strategy passed to LatentProbe.probe().
            max_tokens_per_step: Token budget per Claude step.

        Returns:
            OrchestratorResult with full log of reasoning and consultations.
        """
        t0 = time.time()
        self.detector.reset()

        # Register reference activations on the problem statement.
        ref_acts = self.probe.probe(problem, layers=self.layers, pooling=pooling)
        self.detector.register_reference(ref_acts, step=0)
        log.info("Reference registered. Starting reasoning loop (max_steps=%d).", max_steps)

        history: list[ReasoningStep] = []
        reasoning_log: list[dict] = []
        consultations: list[ConsultationRecord] = []
        final_answer = ""
        recalibration_context: Optional[str] = None

        for step in range(1, max_steps + 1):
            # ── LLM reasoning step ────────────────────────────────────────────
            step_t0 = time.time()
            try:
                reasoning_step = self.llm.reason_step(
                    problem=problem,
                    history=history,
                    recalibration_context=recalibration_context,
                )
                recalibration_context = None  # consumed
            except Exception as e:
                log.error("LLM reason_step error at step %d: %s", step, e)
                break

            step_text = reasoning_step.content
            history.append(reasoning_step)
            log.info("Step %d: %d chars, %.2fs", step, len(step_text), time.time() - step_t0)

            # Check for final answer.
            if "[Final Answer]" in step_text or step == max_steps:
                final_answer = step_text
                reasoning_log.append({
                    "step": step, "text": step_text,
                    "drift_score": None, "consulted": False,
                })
                break

            # ── Drift checkpoint ──────────────────────────────────────────────
            drift_score = None
            consulted   = False

            if step % checkpoint_every == 0:
                cur_acts = self.probe.probe(step_text, layers=self.layers, pooling=pooling)
                report   = self.detector.compute_drift(cur_acts, step=step)
                drift_score = report.drift_score

                if report.should_consult_probe:
                    consulted = True
                    ct0 = time.time()
                    log.info(
                        "Probe consultation triggered: drift=%.4f > threshold=%.4f",
                        report.drift_score, report.threshold,
                    )

                    # Re-probe on the full accumulated context for richer signal.
                    context_text = "\n".join(s.content for s in history)
                    full_acts = self.probe.probe(
                        context_text[-4096:], layers=self.layers, pooling=pooling
                    )
                    full_report = self.detector.compute_drift(full_acts, step=step)

                    observation = self._format_activations_for_claude(full_acts, full_report)
                    recal_note = self.llm.interpret_activations(
                        activations_description=observation,
                        drift_report=full_report,
                        problem_context=context_text[-2000:],
                    )
                    recalibration_context = _RECALIBRATION_PROMPT.format(
                        observation=recal_note.content
                    )

                    consultations.append(ConsultationRecord(
                        step=step,
                        drift_score=report.drift_score,
                        drift_layers_affected=report.layers_affected[:3],
                        observation_text=observation,
                        claude_response_preview=recal_note.content[:200],
                        elapsed_s=round(time.time() - ct0, 3),
                    ))

            reasoning_log.append({
                "step": step,
                "text": step_text[:300],
                "drift_score": round(drift_score, 5) if drift_score is not None else None,
                "consulted": consulted,
            })

        return OrchestratorResult(
            problem=problem,
            final_answer=final_answer,
            n_steps=len(reasoning_log),
            n_consultations=len(consultations),
            consultations=consultations,
            reasoning_log=reasoning_log,
            max_drift=self.detector.max_drift,
            elapsed_s=round(time.time() - t0, 2),
        )

    # ── Activation formatting ─────────────────────────────────────────────────

    def _format_activations_for_claude(
        self,
        activations: dict[int, np.ndarray],
        drift_report,
    ) -> str:
        """
        Convert numpy activations into a structured description for Claude.

        No raw numbers — Claude reasons about shape, norm, direction, and
        the PCA principal components.  The goal is to give Claude a
        structural description of how the zombie is representing this
        content, not a data dump.

        Args:
            activations:  {layer_idx: np.ndarray[hidden_dim]}
            drift_report: DriftReport from DriftDetector.

        Returns:
            A formatted multi-line string suitable for use in a Claude prompt.
        """
        lines = [
            f"Drift score: {drift_report.drift_score:.4f} "
            f"(threshold: {drift_report.threshold:.4f})",
            f"Layers with highest drift: {drift_report.layers_affected[:3]}",
            "",
            "Per-layer activation summary:",
        ]

        for layer_idx, vec in sorted(activations.items()):
            vec = vec.astype(np.float32)
            norm    = float(np.linalg.norm(vec))
            std     = float(np.std(vec))
            # Top-3 PCA directions (power iteration approximation).
            pca_info = _top_pca_components(vec, n=3)
            cosine_d = drift_report.cosine_distances.get(layer_idx, "n/a")
            llc      = drift_report.llc_scores.get(layer_idx, "n/a")

            lines.append(f"  Layer {layer_idx}:")
            lines.append(f"    dim={len(vec)}, norm={norm:.3f}, std={std:.4f}")
            if isinstance(cosine_d, float):
                lines.append(f"    cosine_distance_from_reference={cosine_d:.4f}")
            if isinstance(llc, float):
                lines.append(f"    llc_score={llc:.4f} (feature composition shift)")
            lines.append(f"    top-3 principal components (variance %):")
            for i, (variance_pct, n_positive) in enumerate(pca_info):
                lines.append(
                    f"      PC{i+1}: {variance_pct:.1f}% variance, "
                    f"{n_positive}/{len(vec)} positive dimensions"
                )

        return "\n".join(lines)


# ── PCA helper ────────────────────────────────────────────────────────────────

def _top_pca_components(
    vec: np.ndarray, n: int = 3
) -> list[tuple[float, int]]:
    """
    Approximate the top-n PCA components of a single vector using a
    simple rank-1 decomposition (outer product approximation).

    For a 1D vector the "PCA" is trivially the vector itself.  We split it
    into n non-overlapping windows and report variance + sign info per window.
    This is an approximation intended for prompt readability, not accuracy.

    Returns: list of (variance_pct, n_positive_dims)
    """
    dim = len(vec)
    if dim == 0:
        return []
    chunk = max(1, dim // n)
    result = []
    total_var = float(np.var(vec)) * dim + 1e-10

    for i in range(n):
        start = i * chunk
        end   = (i + 1) * chunk if i < n - 1 else dim
        segment = vec[start:end]
        var_pct = float(np.var(segment) * len(segment) / total_var * 100)
        n_pos   = int(np.sum(segment > 0))
        result.append((round(var_pct, 1), n_pos))

    return result
