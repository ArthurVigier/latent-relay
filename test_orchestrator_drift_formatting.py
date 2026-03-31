from __future__ import annotations

import numpy as np

from eris.drift_detector import DriftDetector, DriftReport
from eris.interfaces import OrchestratorLLM, ReasoningStep
from eris.orchestrator import ERISOrchestrator
from eris.sae_probe import ProbeOutput


class _FakeProbe:
    def __init__(self):
        self.layers = [10, 20, 30]

    def probe(self, *_args, **_kwargs):
        raise NotImplementedError


class _FakeLLM(OrchestratorLLM):
    def reason_step(self, problem: str, history: list[ReasoningStep], recalibration_context=None) -> ReasoningStep:
        return ReasoningStep(content="stub", step_idx=len(history) + 1, uncertainty=0.1)

    def interpret_activations(self, activations_description: str, drift_report, problem_context: str):
        raise NotImplementedError

    def estimate_uncertainty(self, reasoning_step: ReasoningStep) -> float:
        return reasoning_step.uncertainty


def _probe_output(layer: int, active: list[int], raw: list[float]) -> ProbeOutput:
    return ProbeOutput(
        layer=layer,
        active_feature_indices=active[:3],
        active_feature_values=[1.0] * min(len(active), 3),
        all_active_indices=active,
        n_active=len(active),
        raw_activations=np.asarray(raw, dtype=np.float32),
        elapsed_s=0.0,
    )


def _state(layer_map: dict[int, tuple[list[int], list[float]]]) -> dict[int, ProbeOutput]:
    return {
        layer: _probe_output(layer, active=active, raw=raw)
        for layer, (active, raw) in layer_map.items()
    }


def test_format_drift_for_claude_includes_severity_mode_and_layer_scores():
    detector = DriftDetector(
        threshold=0.3,
        window=1,
        jaccard_weight=0.7,
        cosine_weight=0.3,
        layer_weights={10: 1.0, 20: 2.0, 30: 1.0},
        comparison_mode="previous",
    )
    detector.register_reference(
        _state({
            10: ([1, 2, 3], [1.0, 0.0, 0.0]),
            20: ([10, 11, 12], [0.0, 1.0, 0.0]),
            30: ([20, 21, 22], [0.0, 0.0, 1.0]),
        })
    )
    report = detector.compute_drift(
        _state({
            10: ([1, 2, 4], [0.9, 0.1, 0.0]),
            20: ([90, 91, 92], [0.0, 0.1, 0.9]),
            30: ([20, 21, 22], [0.0, 0.0, 1.0]),
        }),
        step=2,
    )

    orch = ERISOrchestrator(_FakeProbe(), detector, _FakeLLM())
    text = orch._format_drift_for_claude(report)

    assert "Severity" in text
    assert "Comparison mode" in text
    assert "Top drifting layers" in text
    assert "score=" in text
    assert "Couche 20" in text


def test_format_drift_for_claude_uses_report_layer_ranking_order():
    report = DriftReport(
        step=5,
        drift_score=0.42,
        raw_drift_score=0.5,
        should_consult_probe=True,
        threshold=0.3,
        comparison_mode="reference",
        severity="medium",
        features_lost={10: [1], 20: [2, 3], 30: []},
        features_gained={10: [4], 20: [9], 30: []},
        cosine_distances={10: 0.2, 20: 0.7, 30: 0.1},
        jaccard_distances={10: 0.3, 20: 0.8, 30: 0.0},
        layer_scores={10: 0.27, 20: 0.77, 30: 0.03},
        layers_ranked=[20, 10, 30],
        n_active_per_layer={10: 3, 20: 4, 30: 2},
        n_layers_evaluated=3,
        summary="demo",
    )
    orch = ERISOrchestrator(_FakeProbe(), DriftDetector(), _FakeLLM())

    text = orch._format_drift_for_claude(report)

    pos20 = text.index("Couche 20")
    pos10 = text.index("Couche 10")
    pos30 = text.index("Couche 30")
    assert pos20 < pos10 < pos30
