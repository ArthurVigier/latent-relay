"""
eris/backends/orchestrators/gemini_orchestrator.py
===================================================

Stub — GeminiOrchestrator is not yet implemented.

To implement:
  1. pip install google-generativeai
  2. Inherit OrchestratorLLM
  3. Implement reason_step(), interpret_activations(), estimate_uncertainty()
     following the same pattern as ClaudeOrchestrator.
"""

from eris.interfaces import OrchestratorLLM


class GeminiOrchestrator(OrchestratorLLM):
    """OrchestratorLLM backed by Google Gemini. Not yet implemented."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GeminiOrchestrator is a stub. "
            "Implement it by following eris/backends/orchestrators/claude_orchestrator.py."
        )

    def reason_step(self, problem, history, recalibration_context=None):
        raise NotImplementedError

    def interpret_activations(self, activations_description, drift_report, problem_context):
        raise NotImplementedError

    def estimate_uncertainty(self, reasoning_step):
        raise NotImplementedError
