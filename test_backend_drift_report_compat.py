from __future__ import annotations

from eris.drift_detector import DriftReport as V2DriftReport
from eris.interfaces import DriftReport as V1DriftReport
from eris.backends.orchestrators.claude_orchestrator import ClaudeOrchestrator
from eris.backends.orchestrators.gemini_orchestrator import GeminiOrchestrator
from eris.backends.orchestrators.openrouter_orchestrator import OpenRouterOrchestrator


def _make_v2_report() -> V2DriftReport:
    return V2DriftReport(
        step=3,
        drift_score=0.61,
        raw_drift_score=0.66,
        should_consult_probe=True,
        threshold=0.35,
        comparison_mode="previous",
        severity="medium",
        layer_scores={20: 0.8, 10: 0.3},
        layers_ranked=[20, 10],
        feature_labels={20: {1: "proof_state", 9: "modular_reasoning"}},
        features_lost={20: [1, 2], 10: [3]},
        features_gained={20: [9], 10: [8]},
    )


def _make_v1_report() -> V1DriftReport:
    return V1DriftReport(
        step=2,
        drift_score=0.18,
        raw_drift_score=0.2,
        cosine_distances={9: 0.2, 18: 0.4},
        l2_distances={9: 1.0, 18: 1.5},
        llc_scores={9: 0.1, 18: 0.2},
        layers_affected=[18, 9],
        should_consult_probe=False,
        threshold=0.3,
    )


def _capture_call(instance, attr_name: str = "captured"):
    def fake_call(messages, system, max_tokens=None):
        setattr(instance, attr_name, {
            "messages": messages,
            "system": system,
            "max_tokens": max_tokens,
        })
        return "ok"
    instance._call = fake_call


def test_claude_interpret_activations_uses_unified_v2_fields():
    orch = ClaudeOrchestrator.__new__(ClaudeOrchestrator)
    _capture_call(orch)

    note = orch.interpret_activations("obs", _make_v2_report(), "ctx")
    prompt = orch.captured["messages"][0]["content"]

    assert note.content == "ok"
    assert "Layers most affected: [20, 10]" in prompt
    assert "Comparison mode: previous" in prompt
    assert "Severity: medium" in prompt
    assert "proof_state" in prompt
    assert "modular_reasoning" in prompt


def test_gemini_interpret_activations_works_with_v1_compat_fields():
    orch = GeminiOrchestrator.__new__(GeminiOrchestrator)
    _capture_call(orch)

    note = orch.interpret_activations("obs", _make_v1_report(), "ctx")
    prompt = orch.captured["messages"][0]["parts"][0]

    assert note.content == "ok"
    assert "Layers most affected: [18, 9]" in prompt
    assert "Comparison mode: reference" in prompt
    assert "Severity: low" in prompt


def test_openrouter_interpret_activations_uses_unified_fields():
    orch = OpenRouterOrchestrator.__new__(OpenRouterOrchestrator)
    _capture_call(orch)

    note = orch.interpret_activations("obs", _make_v2_report(), "ctx")
    prompt = orch.captured["messages"][0]["content"]

    assert note.content == "ok"
    assert "Layers most affected: [20, 10]" in prompt
    assert "Comparison mode: previous" in prompt
    assert "Severity: medium" in prompt
    assert "proof_state" in prompt
    assert "modular_reasoning" in prompt
