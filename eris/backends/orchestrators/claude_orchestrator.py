"""
eris/backends/orchestrators/claude_orchestrator.py
===================================================

OrchestratorLLM implementation for the Anthropic Claude API.

Uses claude-sonnet-4-6 by default (configurable).
All API calls include retry logic (3 attempts, exponential backoff).

uncertainty estimation: self-ask via a separate follow-up message.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from eris.interfaces import (
    DriftReport,
    OrchestratorLLM,
    RecalibrationNote,
    ReasoningStep,
)

log = logging.getLogger("eris.claude_orchestrator")

_DEFAULT_MODEL   = "claude-sonnet-4-6"
_MAX_RETRIES     = 3
_RETRY_BASE_WAIT = 1.0  # seconds, doubled each attempt

_SYSTEM_REASONING = """\
You are a careful reasoner. Work through the problem step by step.
Label each step [Step N]. When done, write [Final Answer] followed by your answer.
"""

_SYSTEM_INTERPRET = """\
You are analysing the internal representation geometry of an external language model
that processed the same problem you are reasoning about.
Interpret the structural observation and produce a brief recalibration note.
Focus on what the observation suggests about the current reasoning trajectory —
not on the numbers themselves.
"""

_UNCERTAINTY_PROMPT = (
    "On a scale from 0.0 to 1.0, how uncertain are you about the reasoning "
    "in your last response? Reply with a single float and nothing else."
)


class ClaudeOrchestrator(OrchestratorLLM):
    """
    OrchestratorLLM backed by the Anthropic Claude API.

    Args:
        model:                 Claude model ID.
        max_tokens:            Token budget per step.
        uncertainty_threshold: Not used internally — exposed for external callers.
        api_key:               Anthropic API key.  If None, reads ANTHROPIC_API_KEY.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 2048,
        uncertainty_threshold: float = 0.6,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        self.model                = model
        self.max_tokens           = max_tokens
        self.uncertainty_threshold = uncertainty_threshold
        self._client              = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # ── OrchestratorLLM interface ─────────────────────────────────────────────

    def reason_step(
        self,
        problem: str,
        history: list[ReasoningStep],
        recalibration_context: Optional[str] = None,
    ) -> ReasoningStep:
        messages = self._build_messages(problem, history, recalibration_context)
        text = self._call(messages, system=_SYSTEM_REASONING)
        step_idx = len(history) + 1
        # Defer uncertainty estimation — caller can invoke estimate_uncertainty separately.
        return ReasoningStep(
            content=text,
            step_idx=step_idx,
            uncertainty=0.0,
            metadata={"model": self.model},
        )

    def interpret_activations(
        self,
        activations_description: str,
        drift_report: DriftReport,
        problem_context: str,
    ) -> RecalibrationNote:
        prompt = (
            f"Problem context (last 2000 chars):\n{problem_context[-2000:]}\n\n"
            f"Drift score: {drift_report.drift_score:.4f} "
            f"(threshold: {drift_report.threshold:.4f})\n"
            f"Layers most affected: {drift_report.layers_affected[:3]}\n\n"
            f"Activation geometry observation:\n{activations_description}\n\n"
            "Produce a brief recalibration note (2-4 sentences). "
            "Focus on what this suggests about the current reasoning trajectory."
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._call(messages, system=_SYSTEM_INTERPRET)
        return RecalibrationNote(
            content=text,
            suggested_steering_direction=None,
            suggested_alpha=0.0,
            confidence=min(1.0, drift_report.drift_score * 2),
        )

    def estimate_uncertainty(self, reasoning_step: ReasoningStep) -> float:
        messages = [
            {"role": "user",      "content": f"Previous reasoning:\n{reasoning_step.content[:1000]}"},
            {"role": "assistant", "content": reasoning_step.content[:1000]},
            {"role": "user",      "content": _UNCERTAINTY_PROMPT},
        ]
        try:
            raw = self._call(messages, system="", max_tokens=10)
            m = re.search(r"[01]?\.\d+", raw)
            if m:
                return float(max(0.0, min(1.0, float(m.group()))))
        except Exception as e:
            log.warning("Uncertainty estimation failed: %s", e)
        return 0.5

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        problem: str,
        history: list[ReasoningStep],
        recalibration_context: Optional[str],
    ) -> list[dict]:
        messages: list[dict] =[{"role": "user", "content": problem}]

        for step in history:
            messages.append({"role": "assistant", "content": step.content})

        if recalibration_context:
            # Approche Canonique : Le coordinateur injecte l'observation de l'espace latent.
            messages.append({
                "role": "user",
                "content": f"[System Observation - Latent Environment]\n{recalibration_context}"
            })
        elif len(history) > 0:
            # Si on est dans une boucle (Etape > 1) sans observation,
            # il FAUT relancer Claude avec un prompt user pour éviter l'erreur 400.
            messages.append({
                "role": "user",
                "content": "Please continue your reasoning step by step. If you are done, conclude with[Final Answer]."
            })

        return messages

    def _call(
        self,
        messages: list[dict],
        system: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Anthropic API call with retry and exponential backoff."""
        mt = max_tokens or self.max_tokens
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict = dict(
                    model=self.model,
                    messages=messages,
                    max_tokens=mt,
                )
                if system:
                    kwargs["system"] = system
                resp = self._client.messages.create(**kwargs)
                return resp.content[0].text
            except Exception as e:
                last_exc = e
                wait = _RETRY_BASE_WAIT * (2 ** attempt)
                log.warning("API call failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1, _MAX_RETRIES, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"Claude API failed after {_MAX_RETRIES} attempts: {last_exc}")
