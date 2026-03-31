from __future__ import annotations

import numpy as np
import pytest

from eris.drift_detector import DriftDetector
from eris.sae_probe import ProbeOutput


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


def test_layer_weights_prioritize_weighted_layers():
    detector = DriftDetector(
        threshold=0.1,
        window=1,
        jaccard_weight=1.0,
        cosine_weight=0.0,
        layer_weights={1: 3.0, 2: 1.0},
    )
    detector.register_reference(
        _state({
            1: ([1, 2, 3], [1.0, 0.0, 0.0]),
            2: ([10, 11], [1.0, 0.0, 0.0]),
        })
    )

    report = detector.compute_drift(
        _state({
            1: ([100, 101, 102], [0.0, 1.0, 0.0]),
            2: ([10, 11], [1.0, 0.0, 0.0]),
        }),
        step=1,
    )

    assert report.raw_drift_score == pytest.approx(0.75)
    assert report.layers_ranked == [1, 2]
    assert report.severity == "medium"


def test_previous_comparison_mode_updates_baseline_after_each_step():
    detector = DriftDetector(
        threshold=0.1,
        window=1,
        jaccard_weight=1.0,
        cosine_weight=0.0,
        comparison_mode="previous",
    )
    detector.register_reference(_state({1: ([1, 2], [1.0, 0.0, 0.0])}))

    first = detector.compute_drift(_state({1: ([2, 3], [0.0, 1.0, 0.0])}), step=1)
    second = detector.compute_drift(_state({1: ([2, 3], [0.0, 1.0, 0.0])}), step=2)

    assert first.raw_drift_score == pytest.approx(2 / 3)
    assert second.raw_drift_score == pytest.approx(0.0)
    assert second.features_lost[1] == []
    assert second.features_gained[1] == []
    assert second.severity == "stable"


def test_report_to_dict_exposes_serializable_summary_fields():
    detector = DriftDetector(threshold=0.2, window=2)
    detector.register_reference(_state({1: ([1, 2, 3], [1.0, 0.0, 0.0])}))

    report = detector.compute_drift(
        _state({1: ([1, 4, 5], [0.0, 1.0, 0.0])}),
        step=3,
    )

    payload = report.to_dict()

    assert payload["step"] == 3
    assert payload["severity"] in {"low", "medium", "high"}
    assert payload["layers_ranked"] == [1]
    assert payload["n_layers_evaluated"] == 1
    assert isinstance(payload["summary"], str)


def test_summary_mentions_ranked_layers_and_severity():
    detector = DriftDetector(
        threshold=0.3,
        window=1,
        jaccard_weight=1.0,
        cosine_weight=0.0,
        layer_weights={1: 1.0, 2: 2.0},
    )
    detector.register_reference(
        _state({
            1: ([1, 2], [1.0, 0.0, 0.0]),
            2: ([10, 20], [1.0, 0.0, 0.0]),
        })
    )

    report = detector.compute_drift(
        _state({
            1: ([1, 2], [1.0, 0.0, 0.0]),
            2: ([99, 100], [0.0, 1.0, 0.0]),
        }),
        step=4,
    )

    assert "severity=" in report.summary
    assert "top_layers=[2, 1]" in report.summary
