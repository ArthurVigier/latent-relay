"""
eris/experiments/multi_agent/kill_criteria_multi_agent.py
===========================================================

Kill-gate criteria for multi-agent experiments.

Tests:
    MA-0  Isolation baseline — do agents in ISOLATED mode diverge less than
          in SHARED_MEDIUM?  (i.e. does sharing activations meaningfully change
          reasoning trajectories?)
          Criterion: mean cosine distance between same-step activations of
          two ISOLATED agents < 0.15  (baseline, should be non-trivial)

    MA-1  Shared medium benefit — does SHARED_MEDIUM mode improve accuracy
          over ISOLATED on a held-out problem set?
          Criterion: accuracy delta ≥ 3pp  (SHARED_MEDIUM − ISOLATED)

    MA-2  Collaborative convergence — does COLLABORATIVE steering cause agents
          to converge on a shared reasoning trajectory?
          Criterion: mean pairwise cosine similarity of final-step activations
          ≥ 0.70 after collaborative steering.
"""

from __future__ import annotations

KILL_CRITERIA_MA: dict[str, dict] = {
    "ma_0": {
        "description": "ISOLATED baseline: inter-agent activation divergence",
        "metric":      "mean_pairwise_cosine_distance",
        "threshold":   0.15,
        "direction":   "lt",   # pass if metric < threshold
        "unit":        "cosine distance",
    },
    "ma_1": {
        "description": "SHARED_MEDIUM accuracy gain over ISOLATED",
        "metric":      "accuracy_delta_pp",
        "threshold":   3.0,
        "direction":   "gt",   # pass if metric > threshold
        "unit":        "percentage points",
    },
    "ma_2": {
        "description": "COLLABORATIVE convergence: pairwise cosine similarity of final activations",
        "metric":      "mean_pairwise_cosine_similarity",
        "threshold":   0.70,
        "direction":   "gt",
        "unit":        "cosine similarity",
    },
}


def check_criterion(test_id: str, value: float) -> tuple[bool, str]:
    """
    Check whether a measured value passes the kill-gate criterion.

    Args:
        test_id: One of "ma_0", "ma_1", "ma_2".
        value:   The measured metric value.

    Returns:
        (passed: bool, message: str)
    """
    if test_id not in KILL_CRITERIA_MA:
        raise KeyError(f"Unknown test_id: {test_id!r}. Valid: {list(KILL_CRITERIA_MA)}")

    c = KILL_CRITERIA_MA[test_id]
    if c["direction"] == "gt":
        passed = value > c["threshold"]
    else:
        passed = value < c["threshold"]

    status = "PASS" if passed else "FAIL (KILL)"
    msg = (
        f"[{test_id.upper()}] {c['description']}\n"
        f"  {c['metric']} = {value:.4f} {c['unit']}\n"
        f"  threshold = {c['threshold']} ({c['direction']})\n"
        f"  → {status}"
    )
    return passed, msg
