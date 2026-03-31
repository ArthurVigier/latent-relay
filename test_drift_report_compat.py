from __future__ import annotations

import pytest

from eris.drift_detector import DriftReport as V2DriftReport
from eris.interfaces import DriftReport as V1DriftReport


def test_v2_report_exposes_v1_layers_affected_alias():
    report = V2DriftReport(
        step=2,
        drift_score=0.4,
        raw_drift_score=0.5,
        should_consult_probe=True,
        threshold=0.3,
        layers_ranked=[20, 10, 30],
        layer_scores={20: 0.8, 10: 0.3, 30: 0.1},
    )

    assert report.layers_affected == [20, 10, 30]


def test_v1_report_exposes_v2_ranked_layers_and_defaults():
    report = V1DriftReport(
        step=4,
        drift_score=0.22,
        raw_drift_score=0.28,
        cosine_distances={9: 0.2, 18: 0.6},
        l2_distances={9: 1.2, 18: 2.4},
        llc_scores={9: 0.1, 18: 0.2},
        layers_affected=[18, 9],
        should_consult_probe=False,
        threshold=0.35,
    )

    assert report.layers_ranked == [18, 9]
    assert report.comparison_mode == "reference"
    assert report.layer_scores == {}
    assert report.features_lost == {}
    assert report.features_gained == {}
    assert report.n_layers_evaluated == 2
    assert report.severity in {"stable", "low", "medium", "high"}


def test_v1_report_to_dict_includes_unified_fields():
    report = V1DriftReport(
        step=1,
        drift_score=0.75,
        raw_drift_score=0.8,
        cosine_distances={3: 0.7},
        l2_distances={3: 1.9},
        llc_scores={3: 0.4},
        layers_affected=[3],
        should_consult_probe=True,
        threshold=0.5,
    )

    payload = report.to_dict()

    assert payload["layers_ranked"] == [3]
    assert payload["layers_affected"] == [3]
    assert payload["comparison_mode"] == "reference"
    assert payload["n_layers_evaluated"] == 1
    assert payload["severity"] == "medium"
