"""
ERIS v5 — Kill Criteria
========================

Explicit stop/pivot conditions for each experiment in the drift detection
pipeline.  Every test must check its result against the relevant criterion
before proceeding to the next test.

If a kill criterion fires, do not proceed.  The entire direction is invalid
until the underlying hypothesis is fixed.

Usage::

    from eris.experiments.drift_detection.kill_criteria import KILL_CRITERIA, check_criterion

    result = run_test_0(...)
    passed, message = check_criterion("test_0", result["spearman_rho"])
    if not passed:
        print(f"KILL: {message}")
        sys.exit(1)
"""

from __future__ import annotations

KILL_CRITERIA: dict[str, dict] = {
    "test_0": {
        "metric":      "spearman_rho_drift_vs_error",
        "threshold":   0.35,
        "direction":   "greater",
        "description": (
            "Latent drift must predict final answer errors. "
            "Measured as Spearman ρ between final drift_score and binary "
            "error (0=correct, 1=wrong) on 20 AIME problems."
        ),
        "action_if_failed": (
            "STOP — reframe entirely invalid. "
            "Drift in this model/layer does not correlate with reasoning errors. "
            "Before retrying: (1) check layer selection, (2) try a harder dataset, "
            "(3) consider that the model may not exhibit meaningful latent drift."
        ),
        "action_if_passed": "Proceed to test_1 (probe divergence detection).",
        "estimated_cost":   "$0 — no API calls, runs on local GPU",
        "estimated_time":   "~2h on A100 (20 problems × 8 reasoning steps)",
    },
    "test_1": {
        "metric":      "probe_divergence_auc",
        "threshold":   0.60,
        "direction":   "greater",
        "description": (
            "Zombie probe divergence must detect drift better than a naive baseline. "
            "AUC of probe_divergence_score as a binary classifier for "
            "correct vs wrong answers."
        ),
        "action_if_failed": (
            "STOP — probe does not add signal beyond drift detector alone. "
            "Zombie reference frame does not distinguish error types. "
            "Consider: different zombie model, different layers, different pooling."
        ),
        "action_if_passed": "Proceed to test_2 (active intervention).",
        "estimated_cost":   "~$5 (Claude API for 20 problems)",
        "estimated_time":   "~3h",
    },
    "test_2": {
        "metric":      "accuracy_delta_with_intervention",
        "threshold":   0.05,
        "direction":   "greater",
        "description": (
            "Active probe consultation must improve final answer accuracy by "
            "at least 5 percentage points vs no-intervention baseline."
        ),
        "action_if_failed": (
            "PIVOT — drift detection and probe signal are real, but the "
            "intervention mechanism (how Claude uses the activations) is not "
            "effective yet.  Do not abandon the direction — redesign the "
            "_format_activations_for_claude() prompt."
        ),
        "action_if_passed": "Phase 2 validated. Proceed to full eval suite.",
        "estimated_cost":   "~$20 (Claude API with active consultation loop)",
        "estimated_time":   "~4h",
    },
}


def check_criterion(test_id: str, metric_value: float) -> tuple[bool, str]:
    """
    Check a metric value against the kill criterion for a test.

    Args:
        test_id:      One of "test_0", "test_1", "test_2".
        metric_value: The measured value of the criterion metric.

    Returns:
        (passed: bool, message: str)
    """
    if test_id not in KILL_CRITERIA:
        raise KeyError(f"Unknown test_id: {test_id!r}. Valid: {list(KILL_CRITERIA)}")

    c = KILL_CRITERIA[test_id]
    threshold  = c["threshold"]
    direction  = c["direction"]

    if direction == "greater":
        passed = metric_value > threshold
    elif direction == "less":
        passed = metric_value < threshold
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    if passed:
        message = (
            f"PASS — {test_id}: {c['metric']}={metric_value:.4f} "
            f"{'>' if direction == 'greater' else '<'} {threshold}. "
            f"{c['action_if_passed']}"
        )
    else:
        message = (
            f"KILL — {test_id}: {c['metric']}={metric_value:.4f} "
            f"NOT {'>' if direction == 'greater' else '<'} {threshold}. "
            f"{c['action_if_failed']}"
        )

    return passed, message
