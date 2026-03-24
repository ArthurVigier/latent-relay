"""
eris/backends/orchestrators/openai_orchestrator.py
===================================================

Stub — OpenAIOrchestrator is not yet implemented.

To implement:
  1. pip install openai
  2. Inherit OrchestratorLLM
  3. Implement reason_step(), interpret_activations(), estimate_uncertainty()
     following the same pattern as ClaudeOrchestrator.
"""

from eris.interfaces import OrchestratorLLM


class OpenAIOrchestrator(OrchestratorLLM):
    """OrchestratorLLM backed by the OpenAI API. Not yet implemented."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "OpenAIOrchestrator is a stub. "
            "Implement it by following eris/backends/orchestrators/claude_orchestrator.py."
        )

    def reason_step(self, problem, history, recalibration_context=None):
        raise NotImplementedError

    def interpret_activations(self, activations_description, drift_report, problem_context):
        raise NotImplementedError

    def estimate_uncertainty(self, reasoning_step):
        raise NotImplementedError
