from __future__ import annotations

from eris.interfaces import DriftReport


def _format_named_concepts(drift_report: DriftReport) -> str | None:
    feature_labels = getattr(drift_report, "feature_labels", {}) or {}
    features_lost = getattr(drift_report, "features_lost", {}) or {}
    features_gained = getattr(drift_report, "features_gained", {}) or {}

    lost_refs: list[str] = []
    gained_refs: list[str] = []

    for layer, indices in features_lost.items():
        label_map = feature_labels.get(layer, {})
        for idx in indices[:4]:
            label = label_map.get(idx)
            if label:
                lost_refs.append(f"{layer}:{idx}:{label}")

    for layer, indices in features_gained.items():
        label_map = feature_labels.get(layer, {})
        for idx in indices[:4]:
            label = label_map.get(idx)
            if label:
                gained_refs.append(f"{layer}:{idx}:{label}")

    if not lost_refs and not gained_refs:
        return None

    return f"Named concepts in drift: lost={lost_refs} gained={gained_refs}"


def build_interpretation_prompt(
    *,
    problem_context: str,
    activations_description: str,
    drift_report: DriftReport,
) -> str:
    layers = getattr(drift_report, "layers_ranked", getattr(drift_report, "layers_affected", []))
    comparison_mode = getattr(drift_report, "comparison_mode", "reference")
    severity = getattr(drift_report, "severity", "unknown")
    named_concepts = _format_named_concepts(drift_report)
    concept_block = f"{named_concepts}\n\n" if named_concepts else ""
    return (
        f"Problem context (last 2000 chars):\n{problem_context[-2000:]}\n\n"
        f"Drift score: {drift_report.drift_score:.4f} "
        f"(threshold: {drift_report.threshold:.4f})\n"
        f"Layers most affected: {layers[:3]}\n"
        f"Comparison mode: {comparison_mode}\n"
        f"Severity: {severity}\n\n"
        f"{concept_block}"
        f"Activation geometry observation:\n{activations_description}\n\n"
        "Produce a brief recalibration note (2-4 sentences). "
        "Focus on what this suggests about the current reasoning trajectory."
    )
