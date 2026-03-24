"""
eris/factory.py
===============

Instantiate ERIS backends from a configuration dictionary.

The factory reads the ``backends`` section of eris_config.yaml (or an
equivalent dict) and returns the appropriate concrete implementation of
OrchestratorLLM and ProbeModel.

Usage::

    from eris.factory import create_orchestrator, create_probe, create_coordinator

    llm   = create_orchestrator(cfg["backends"]["orchestrator"])
    probe = create_probe(cfg["backends"]["probe"])
    coord = create_coordinator(cfg.get("multi_agent", {}), probe)

Supported orchestrator backends:
    - "claude"       → ClaudeOrchestrator
    - "gemini"       → GeminiOrchestrator
    - "openai"       → OpenAIOrchestrator  (stub — raises NotImplementedError)
    - "openrouter"   → OpenRouterOrchestrator  (any model slug from openrouter.ai/models)

Supported probe backends:
    - "hf"    → HFProbe (locally-loaded HuggingFace model)
    - "vllm"  → VLLMProbe (stub — raises NotImplementedError)
"""

from __future__ import annotations

import logging
from typing import Optional

from eris.interfaces import OrchestratorLLM, ProbeModel

log = logging.getLogger("eris.factory")


def create_orchestrator(cfg: dict) -> OrchestratorLLM:
    """
    Instantiate an OrchestratorLLM from a backend config dict.

    Expected cfg keys (all optional unless noted):
        backend:    "claude" | "gemini" | "openai"  (default: "claude")
        model:      model ID string
        max_tokens: int
        api_key:    API key string (falls back to environment variable)

    Example::

        llm = create_orchestrator({"backend": "claude", "model": "claude-sonnet-4-6"})
    """
    backend = cfg.get("backend", "claude").lower()
    log.info("Creating orchestrator: backend=%s", backend)

    if backend == "claude":
        from eris.backends.orchestrators.claude_orchestrator import ClaudeOrchestrator
        kwargs = {}
        if "model"      in cfg: kwargs["model"]      = cfg["model"]
        if "max_tokens" in cfg: kwargs["max_tokens"]  = cfg["max_tokens"]
        if "api_key"    in cfg: kwargs["api_key"]     = cfg["api_key"]
        return ClaudeOrchestrator(**kwargs)

    if backend == "gemini":
        from eris.backends.orchestrators.gemini_orchestrator import GeminiOrchestrator
        return GeminiOrchestrator(**cfg)

    if backend == "openai":
        from eris.backends.orchestrators.openai_orchestrator import OpenAIOrchestrator
        return OpenAIOrchestrator(**cfg)

    if backend == "openrouter":
        from eris.backends.orchestrators.openrouter_orchestrator import OpenRouterOrchestrator
        kwargs = {}
        for key in (
            "model", "max_tokens", "api_key", "site_url", "site_name",
            "provider_order", "fallbacks", "transforms", "require_params", "extra_headers",
        ):
            if key in cfg:
                kwargs[key] = cfg[key]
        return OpenRouterOrchestrator(**kwargs)

    raise ValueError(
        f"Unknown orchestrator backend: {backend!r}. "
        "Supported: 'claude', 'gemini', 'openai', 'openrouter'."
    )


def create_probe(cfg: dict) -> ProbeModel:
    """
    Instantiate a ProbeModel from a backend config dict.

    Expected cfg keys (all optional unless noted):
        backend:   "hf" | "vllm"  (default: "hf")
        model_id:  HuggingFace model name or path  (required)
        layers:    list[int]  (required)
        device:    "cuda" | "cpu" | "cuda:0" etc.
        dtype:     "bfloat16" | "float32" | "float16"

    Example::

        probe = create_probe({
            "backend": "hf",
            "model_id": "Qwen/Qwen3-14B",
            "layers": [9, 18],
            "device": "cuda",
        })
    """
    backend = cfg.get("backend", "hf").lower()
    log.info("Creating probe: backend=%s model=%s", backend, cfg.get("model_id", "?"))

    if backend == "hf":
        from eris.backends.probes.hf_probe import HFProbe
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
            "float16":  torch.float16,
        }
        kwargs: dict = {
            "model_id": cfg["model_id"],
            "layers":   cfg["layers"],
        }
        if "device" in cfg:
            kwargs["device"] = cfg["device"]
        if "dtype" in cfg and cfg["dtype"] in dtype_map:
            kwargs["dtype"] = dtype_map[cfg["dtype"]]
        return HFProbe(**kwargs)

    if backend == "vllm":
        from eris.backends.probes.vllm_probe import VLLMProbe
        return VLLMProbe(**cfg)

    raise ValueError(
        f"Unknown probe backend: {backend!r}. Supported: 'hf', 'vllm'."
    )


def create_coordinator(cfg: dict, probe: ProbeModel):
    """
    Instantiate a MultiAgentCoordinator from a config dict.

    Args:
        cfg:   The ``multi_agent`` section of eris_config.yaml.
        probe: A shared ProbeModel instance.

    Returns:
        MultiAgentCoordinator instance.
    """
    from eris.multi_agent import MultiAgentCoordinator, CoordinationMode

    mode_str = cfg.get("mode", "isolated").upper()
    try:
        mode = CoordinationMode[mode_str]
    except KeyError:
        raise ValueError(
            f"Unknown multi_agent mode: {mode_str!r}. "
            f"Supported: {[m.name for m in CoordinationMode]}"
        )

    n_agents = cfg.get("n_agents", 2)
    log.info("Creating coordinator: mode=%s n_agents=%d", mode.name, n_agents)
    return MultiAgentCoordinator(probe=probe, mode=mode, n_agents=n_agents)
