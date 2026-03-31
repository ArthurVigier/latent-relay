from __future__ import annotations

from eris.interfaces import DriftReport


def build_interpretation_prompt(
    *,
    problem_context: str,
    activations_description: str,
    drift_report: DriftReport,
) -> str:
    layers = getattr(drift_report, "layers_ranked", getattr(drift_report, "layers_affected", []))
    comparison_mode = getattr(drift_report, "comparison_mode", "reference")
    severity = getattr(drift_report, "severity", "unknown")
    return (
        f"Problem context (last 2000 chars):\n{problem_context[-2000:]}\n\n"
        f"Drift score: {drift_report.drift_score:.4f} "
        f"(threshold: {drift_report.threshold:.4f})\n"
        f"Layers most affected: {layers[:3]}\n"
        f"Comparison mode: {comparison_mode}\n"
        f"Severity: {severity}\n\n"
        f"Activation geometry observation:\n{activations_description}\n\n"
        "Produce a brief recalibration note (2-4 sentences). "
        "Focus on what this suggests about the current reasoning trajectory."
    )
