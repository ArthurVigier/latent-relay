from __future__ import annotations

from eris.drift_detector import DriftReport as V2DriftReport
from eris.interfaces import DriftReport as V1DriftReport
from eris.backends.orchestrators.common import build_interpretation_prompt


def test_build_interpretation_prompt_supports_v2_report_fields():
    report = V2DriftReport(
        step=7,
        drift_score=0.61,
        raw_drift_score=0.67,
        should_consult_probe=True,
        threshold=0.35,
        comparison_mode="previous",
        severity="medium",
        layers_ranked=[20, 10],
    )

    prompt = build_interpretation_prompt(
        problem_context="context",
        activations_description="observation",
        drift_report=report,
    )

    assert "Drift score: 0.6100 (threshold: 0.3500)" in prompt
    assert "Layers most affected: [20, 10]" in prompt
    assert "Comparison mode: previous" in prompt
    assert "Severity: medium" in prompt
    assert "Activation geometry observation:\nobservation" in prompt


def test_build_interpretation_prompt_supports_v1_report_compatibility_fields():
    report = V1DriftReport(
        step=2,
        drift_score=0.18,
        raw_drift_score=0.22,
        cosine_distances={9: 0.2},
        l2_distances={9: 1.0},
        llc_scores={9: 0.1},
        layers_affected=[9],
        should_consult_probe=False,
        threshold=0.30,
    )

    prompt = build_interpretation_prompt(
        problem_context="context",
        activations_description="observation",
        drift_report=report,
    )

    assert "Layers most affected: [9]" in prompt
    assert "Comparison mode: reference" in prompt
    assert "Severity: low" in prompt
