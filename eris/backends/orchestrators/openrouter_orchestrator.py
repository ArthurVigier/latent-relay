"""
eris/backends/orchestrators/openrouter_orchestrator.py
=======================================================

OrchestratorLLM implementation backed by OpenRouter.

OpenRouter provides a unified OpenAI-compatible API that routes to hundreds
of models: Anthropic, Google, Meta, Mistral, Cohere, local Ollama, etc.
Any model slug listed at https://openrouter.ai/models works here — just
pass it as `model`.

Examples::

    # Claude via OpenRouter
    llm = OpenRouterOrchestrator("anthropic/claude-opus-4")

    # Gemini via OpenRouter
    llm = OpenRouterOrchestrator("google/gemini-2.5-pro")

    # Llama 3.3 70B (free tier)
    llm = OpenRouterOrchestrator("meta-llama/llama-3.3-70b-instruct:free")

    # Auto-route: cheapest model that meets capability requirements
    llm = OpenRouterOrchestrator("openrouter/auto")

    # Discover all available models at runtime
    models = OpenRouterOrchestrator.list_models(api_key="...")

OpenRouter-specific features exposed:
    provider_order:   list of provider names to try in order
                      e.g. ["Anthropic", "AWS Bedrock"] for Claude
    require_params:   enforce model capabilities
                      e.g. {"max_tokens": {"gte": 32000}}
    transforms:       ["middle-out"] — compress prompts that exceed context
    fallbacks:        list of model slugs to try if primary fails
    extra_headers:    arbitrary HTTP headers forwarded to OpenRouter
                      (HTTP-Referer, X-Title, etc.)

API key: reads OPENROUTER_API_KEY from environment if not passed explicitly.
Retry logic: 3 attempts with exponential backoff, same as other orchestrators.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional

from eris.interfaces import (
    DriftReport,
    OrchestratorLLM,
    RecalibrationNote,
    ReasoningStep,
)
from eris.backends.orchestrators.common import build_interpretation_prompt

log = logging.getLogger("eris.openrouter_orchestrator")

_BASE_URL        = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL   = "openrouter/auto"  # cheapest capable model by default
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


class OpenRouterOrchestrator(OrchestratorLLM):
    """
    OrchestratorLLM backed by OpenRouter.

    Accepts any model slug from https://openrouter.ai/models.
    Uses the OpenAI-compatible Chat Completions endpoint.

    Args:
        model:            OpenRouter model slug.
                          Default: "openrouter/auto" (cheapest capable model).
        max_tokens:       Token budget per step.
        api_key:          OpenRouter API key. Falls back to OPENROUTER_API_KEY.
        site_url:         Your site URL — sent as HTTP-Referer to OpenRouter.
                          Helps with rate limits on free-tier models.
        site_name:        Your app name — sent as X-Title to OpenRouter.
        provider_order:   Preferred providers for this model.
                          e.g. ["Anthropic", "AWS Bedrock"]
        fallbacks:        Model slugs to try if the primary model fails.
        transforms:       OpenRouter transforms. ["middle-out"] compresses
                          prompts that exceed the context window.
        require_params:   Enforce model capability requirements.
                          e.g. {"max_tokens": {"gte": 32000}}
        extra_headers:    Additional HTTP headers forwarded to OpenRouter.

    Examples::

        # Minimal
        llm = OpenRouterOrchestrator("meta-llama/llama-3.3-70b-instruct")

        # With provider preferences and fallbacks
        llm = OpenRouterOrchestrator(
            model="anthropic/claude-opus-4",
            provider_order=["Anthropic", "AWS Bedrock"],
            fallbacks=["anthropic/claude-sonnet-4-5", "openrouter/auto"],
        )

        # With automatic prompt compression
        llm = OpenRouterOrchestrator(
            model="google/gemini-2.5-pro",
            transforms=["middle-out"],
        )
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        site_url: str = "https://github.com/ArthurVigier/latent-relay",
        site_name: str = "latent-relay / ERIS",
        provider_order: Optional[list[str]] = None,
        fallbacks: Optional[list[str]] = None,
        transforms: Optional[list[str]] = None,
        require_params: Optional[dict] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            log.warning(
                "OPENROUTER_API_KEY not set. Most OpenRouter models require authentication."
            )

        self.model      = model
        self.max_tokens = max_tokens

        # Build default headers — OpenRouter recommends both for better routing.
        headers: dict[str, str] = {
            "HTTP-Referer": site_url,
            "X-Title":      site_name,
        }
        if extra_headers:
            headers.update(extra_headers)

        self._client = OpenAI(
            api_key=resolved_key or "no-key",
            base_url=_BASE_URL,
            default_headers=headers,
        )

        # OpenRouter-specific routing params — injected into `extra_body` per call.
        self._routing: dict[str, Any] = {}
        if provider_order:
            self._routing["provider"] = {"order": provider_order}
        if fallbacks:
            self._routing["fallbacks"] = fallbacks
        if transforms:
            self._routing["transforms"] = transforms
        if require_params:
            self._routing["route"] = {"require_params": require_params}

        log.info(
            "OpenRouterOrchestrator ready: model=%s routing=%s",
            model, self._routing or "default",
        )

    # ── OrchestratorLLM interface ─────────────────────────────────────────────

    def reason_step(
        self,
        problem: str,
        history: list[ReasoningStep],
        recalibration_context: Optional[str] = None,
    ) -> ReasoningStep:
        messages = self._build_messages(problem, history, recalibration_context)
        text = self._call(messages, system=_SYSTEM_REASONING)
        return ReasoningStep(
            content=text,
            step_idx=len(history) + 1,
            uncertainty=0.0,
            metadata={"model": self.model, "backend": "openrouter"},
        )

    def interpret_activations(
        self,
        activations_description: str,
        drift_report: DriftReport,
        problem_context: str,
    ) -> RecalibrationNote:
        prompt = build_interpretation_prompt(
            problem_context=problem_context,
            activations_description=activations_description,
            drift_report=drift_report,
        )
        text = self._call([{"role": "user", "content": prompt}], system=_SYSTEM_INTERPRET)
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

    # ── Discovery ─────────────────────────────────────────────────────────────

    @staticmethod
    def list_models(
        api_key: Optional[str] = None,
        *,
        filter_free: bool = False,
        filter_context_gte: Optional[int] = None,
    ) -> list[dict]:
        """
        Fetch all models currently available on OpenRouter.

        Returns a list of dicts with keys: id, name, context_length,
        pricing (prompt/completion per token), top_provider.

        Args:
            api_key:            OpenRouter API key. Falls back to OPENROUTER_API_KEY.
            filter_free:        If True, return only free-tier models.
            filter_context_gte: If set, return only models with context_length
                                >= this value.

        Example::

            models = OpenRouterOrchestrator.list_models(filter_free=True)
            for m in models:
                print(m["id"], m["context_length"])
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for list_models: pip install httpx")

        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}

        resp = httpx.get(f"{_BASE_URL}/models", headers=headers, timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])

        if filter_free:
            models = [
                m for m in models
                if str(m.get("pricing", {}).get("prompt", "1")) == "0"
            ]
        if filter_context_gte is not None:
            models = [
                m for m in models
                if (m.get("context_length") or 0) >= filter_context_gte
            ]

        return models

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        problem: str,
        history: list[ReasoningStep],
        recalibration_context: Optional[str],
    ) -> list[dict]:
        messages: list[dict] = [{"role": "user", "content": problem}]
        for step in history:
            messages.append({"role": "assistant", "content": step.content})
        if recalibration_context:
            messages.append({"role": "user", "content": recalibration_context})
        return messages

    def _call(
        self,
        messages: list[dict],
        system: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """OpenRouter API call with retry and exponential backoff."""
        mt = max_tokens or self.max_tokens
        last_exc = None

        # Prepend system message if provided (OpenAI-style).
        full_messages = (
            [{"role": "system", "content": system}] + messages
            if system
            else list(messages)
        )

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=mt,
                )
                if self._routing:
                    kwargs["extra_body"] = self._routing

                resp = self._client.chat.completions.create(**kwargs)

                # Surface OpenRouter usage/model info at debug level.
                if hasattr(resp, "model") and resp.model:
                    log.debug("OpenRouter routed to: %s", resp.model)

                return resp.choices[0].message.content or ""

            except Exception as e:
                last_exc = e
                wait = _RETRY_BASE_WAIT * (2 ** attempt)
                log.warning(
                    "OpenRouter call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"OpenRouter API failed after {_MAX_RETRIES} attempts "
            f"(model={self.model}): {last_exc}"
        )

    def __repr__(self) -> str:
        return f"OpenRouterOrchestrator(model={self.model!r}, routing={self._routing or 'default'})"
