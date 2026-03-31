from __future__ import annotations

import numpy as np

from eris.drift_detector import DriftDetector
from eris.interfaces import OrchestratorLLM, ReasoningStep
from eris.orchestrator import ERISOrchestrator
from eris.sae_probe import ProbeOutput


class _FakeProbe:
    def __init__(self):
        self.layers = [20]

    def probe(self, *_args, **_kwargs):
        raise NotImplementedError


class _FakeLLM(OrchestratorLLM):
    def reason_step(self, problem: str, history: list[ReasoningStep], recalibration_context=None) -> ReasoningStep:
        return ReasoningStep(content="stub", step_idx=len(history) + 1, uncertainty=0.1)

    def interpret_activations(self, activations_description: str, drift_report, problem_context: str):
        raise NotImplementedError

    def estimate_uncertainty(self, reasoning_step: ReasoningStep) -> float:
        return reasoning_step.uncertainty


def _probe_output(layer: int, active: list[int], labels: list[str | None], raw: list[float]) -> ProbeOutput:
    return ProbeOutput(
        layer=layer,
        active_feature_indices=active,
        active_feature_values=[1.0] * len(active),
        all_active_indices=active,
        n_active=len(active),
        raw_activations=np.asarray(raw, dtype=np.float32),
        elapsed_s=0.0,
        active_feature_labels=labels,
    )


def test_drift_summary_and_orchestrator_include_feature_labels_when_available():
    detector = DriftDetector(threshold=0.2, window=1, jaccard_weight=1.0, cosine_weight=0.0)
    detector.register_reference(
        {
            20: _probe_output(20, [1, 2], ["proof_state", "algebraic_manipulation"], [1.0, 0.0]),
        }
    )

    report = detector.compute_drift(
        {
            20: _probe_output(20, [2, 7], ["algebraic_manipulation", "modular_reasoning"], [0.0, 1.0]),
        },
        step=3,
    )

    assert "proof_state" in report.summary
    assert "modular_reasoning" in report.summary

    orch = ERISOrchestrator(_FakeProbe(), detector, _FakeLLM())
    text = orch._format_drift_for_claude(report)

    assert "proof_state" in text
    assert "modular_reasoning" in text
