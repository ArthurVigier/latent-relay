"""
eris/backends/orchestrators/gemini_orchestrator.py
===================================================

OrchestratorLLM implementation for Google Gemini.

Uses gemini-2.5-pro by default.
All API calls include retry logic (3 attempts, exponential backoff).
"""

from __future__ import annotations

import logging
import re
import time
import os
from typing import Optional

from eris.interfaces import (
    DriftReport,
    OrchestratorLLM,
    RecalibrationNote,
    ReasoningStep,
)

log = logging.getLogger("eris.gemini_orchestrator")

_DEFAULT_MODEL   = "gemini-2.5-pro"
_MAX_RETRIES     = 3
_RETRY_BASE_WAIT = 1.0  # seconds, doubled each attempt

_SYSTEM_REASONING = """\
You are a careful reasoner. Work through the problem step by step.
Label each step [Step N]. When done, write[Final Answer] followed by your answer.
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


class GeminiOrchestrator(OrchestratorLLM):
    """
    OrchestratorLLM backed by the Google Gemini API.

    Args:
        model:                 Gemini model ID (e.g., "gemini-2.5-pro").
        max_tokens:            Token budget per step.
        uncertainty_threshold: Threshold to trigger a probe consultation.
        api_key:               Gemini API key. If None, reads GEMINI_API_KEY.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 2048,
        uncertainty_threshold: float = 0.6,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package required: pip install google-generativeai")

        self.model = model
        self.max_tokens = max_tokens
        self.uncertainty_threshold = uncertainty_threshold

        # Setup API Key
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            log.warning("GEMINI_API_KEY not found in environment variables.")

        genai.configure(api_key=key)
        self._genai = genai

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

        return ReasoningStep(
            content=text,
            step_idx=step_idx,
            uncertainty=0.0, # Computed externally if needed
            metadata={"model": self.model},
        )

    def interpret_activations(
        self,
        activations_description: str,
        drift_report: DriftReport,
        problem_context: str,
    ) -> RecalibrationNote:
        layers = getattr(drift_report, "layers_ranked", getattr(drift_report, "layers_affected", []))
        comparison_mode = getattr(drift_report, "comparison_mode", "reference")
        severity = getattr(drift_report, "severity", "unknown")
        prompt = (
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
        messages = [{"role": "user", "parts": [prompt]}]
        text = self._call(messages, system=_SYSTEM_INTERPRET)

        return RecalibrationNote(
            content=text,
            suggested_steering_direction=None,
            suggested_alpha=0.0,
            confidence=min(1.0, drift_report.drift_score * 2),
        )

    def estimate_uncertainty(self, reasoning_step: ReasoningStep) -> float:
        messages = [
            {"role": "user",  "parts":[f"Previous reasoning:\n{reasoning_step.content[:1000]}"]},
            {"role": "model", "parts": [reasoning_step.content[:1000]]},
            {"role": "user",  "parts": [_UNCERTAINTY_PROMPT]},
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
        messages: list[dict] =[{"role": "user", "parts": [problem]}]
        for step in history:
            messages.append({"role": "model", "parts": [step.content]})
        if recalibration_context:
            messages.append({"role": "user", "parts": [recalibration_context]})
        return messages

    def _call(
        self,
        messages: list[dict],
        system: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Gemini API call with retry and exponential backoff."""
        mt = max_tokens or self.max_tokens
        last_exc = None

        for attempt in range(_MAX_RETRIES):
            try:
                # Gemini takes system instructions at the model initialization level
                model_obj = self._genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system if system else None
                )

                response = model_obj.generate_content(
                    messages,
                    generation_config=self._genai.types.GenerationConfig(
                        max_output_tokens=mt
                    )
                )
                return response.text
            except Exception as e:
                last_exc = e
                wait = _RETRY_BASE_WAIT * (2 ** attempt)
                log.warning("Gemini API call failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1, _MAX_RETRIES, e, wait)
                time.sleep(wait)

        raise RuntimeError(f"Gemini API failed after {_MAX_RETRIES} attempts: {last_exc}")
