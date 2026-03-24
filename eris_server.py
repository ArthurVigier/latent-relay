"""
ERIS v5 Server — FastAPI
=========================
Superset of server.py: imports all existing endpoints unchanged and adds the
five ERIS v5 endpoints.

New endpoints
-------------
    POST /v1/encode        → encode text, return hidden states per layer
    POST /v1/analyze       → run MI analyses on a stored thought's hidden states
    POST /v1/latent_think  → extended /think with trajectory + perturbation
    POST /v1/inject        → surgical hidden-state injection
    POST /v1/bridge        → full Claude → Zombie → Claude pipeline

Existing endpoints (unchanged)
-------------------------------
    POST   /sessions
    GET    /sessions
    DELETE /sessions/{id}
    POST   /think
    POST   /collaborate
    GET    /thoughts/{session_id}/{handle}
    GET    /health

Usage
-----
    python eris_server.py --model Qwen/Qwen3-14B --port 8001
    # or
    uvicorn eris_server:app --host 0.0.0.0 --port 8001

Environment variables
---------------------
    LATENT_MODEL   — HuggingFace model name (read by server.py)
    LATENT_DEVICE  — torch device, e.g. "cuda:0" (read by server.py)
    ERIS_CONFIG    — path to eris_config.yaml (optional, defaults to configs/)
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Import existing app and engine reference ──────────────────────────────────
# server.py registers /sessions, /think, /collaborate, /thoughts, /health on
# startup.  We extend the same `app` object so both sets of endpoints are
# served from one process.
import server                        # registers startup event + existing routes
from server import app               # the FastAPI instance we extend

from eris.analyzers import AnalyzerRegistry
from eris.bridge import ERISBridge
from eris.config import ERISConfig
from eris.injector import InjectionError, inject

# ── ERIS globals (populated in eris_startup) ──────────────────────────────────
_eris_cfg:      Optional[ERISConfig]      = None
_eris_registry: Optional[AnalyzerRegistry] = None
_eris_bridge:   Optional[ERISBridge]       = None

# ── Response-size guard (CWE-400) ─────────────────────────────────────────────
# Default: 2 GB — sized for multi-layer hidden states on DeepSeek R1 / Qwen3-32B.
# Updated at startup once the ERIS config is loaded.
_max_payload_bytes: int = 2_147_483_648        # 2 GB
_max_payload_warn:  int = 524_288_000          # 500 MB — log warning above this

import logging as _logging
_eris_log = _logging.getLogger("eris")

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as _StarletteResponse


class _ResponseSizeMiddleware(BaseHTTPMiddleware):
    """Return HTTP 413 with a helpful body when a /v1/ response exceeds the limit."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Only enforce on ERIS endpoints — leave /health, /sessions, etc. alone.
        if not request.url.path.startswith("/v1/"):
            return response

        # Consume the response body so we can measure it.
        body_parts = []
        async for chunk in response.body_iterator:
            body_parts.append(chunk if isinstance(chunk, bytes) else chunk.encode())
        body = b"".join(body_parts)
        size = len(body)

        if size > _max_payload_bytes:
            _eris_log.error(
                "Response too large: %s bytes (limit: %s) for %s. "
                "Hint: use fewer return_layers, shorter text, or compact=true.",
                f"{size:,}", f"{_max_payload_bytes:,}", request.url.path,
            )
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"Response too large: {size:,} bytes "
                              f"(limit: {_max_payload_bytes:,} bytes).",
                    "hint": "Use fewer return_layers, shorter input text, "
                            "or set compact=true.",
                    "response_bytes": size,
                    "limit_bytes": _max_payload_bytes,
                },
            )

        if size > _max_payload_warn:
            _eris_log.warning(
                "Large response: %s bytes for %s (warning threshold: %s)",
                f"{size:,}", request.url.path, f"{_max_payload_warn:,}",
            )

        return _StarletteResponse(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


app.add_middleware(_ResponseSizeMiddleware)


@app.on_event("startup")
async def eris_startup() -> None:
    """
    Second startup hook (runs after server.py's engine load).
    Initialises the ERIS config, analyzer registry, and bridge.
    """
    global _eris_cfg, _eris_registry, _eris_bridge, _max_payload_bytes, _max_payload_warn

    config_path = os.environ.get("ERIS_CONFIG")
    _eris_cfg = ERISConfig.load(config_path)
    _max_payload_bytes = _eris_cfg.server.max_payload_bytes
    _max_payload_warn = getattr(
        _eris_cfg.server, "max_payload_warning_bytes", _max_payload_bytes // 4
    )
    print(_eris_cfg.summary())

    _eris_registry = AnalyzerRegistry.from_config(_eris_cfg)
    # server.engine is set by this point (server startup runs first).
    _eris_bridge = ERISBridge(server.engine, _eris_registry)
    print("[ERIS] Bridge ready.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_engine():
    if server.engine is None:
        raise HTTPException(503, "Engine not initialised yet.")
    return server.engine


def _require_bridge() -> ERISBridge:
    if _eris_bridge is None:
        raise HTTPException(503, "ERIS bridge not initialised yet.")
    return _eris_bridge


def _get_thought_hidden(session_id: str, handle: str) -> torch.Tensor:
    """
    Retrieve the best available hidden-state tensor (CPU float32) from a thought.

    Preference order:
    1. ``layer_hidden_states["last"]``  — full sequence, last layer (from encode)
    2. ``hidden_embedding``             — last token only (always available)

    Raises HTTP 404 if the thought or session does not exist.
    Raises HTTP 422 if the thought has no hidden state at all.
    """
    eng = _require_engine()
    with eng._lock:
        session = eng._sessions.get(session_id)
        if session is None:
            raise HTTPException(404, f"Session '{session_id}' not found.")
        thought = session.thoughts.get(handle)
        if thought is None:
            raise HTTPException(404, f"Thought '{handle}' not found in session '{session_id}'.")

        if thought.layer_hidden_states and "last" in thought.layer_hidden_states:
            return thought.layer_hidden_states["last"].detach().cpu().float()

        if thought.hidden_embedding is not None:
            return thought.hidden_embedding.detach().cpu().float().reshape(1, -1)

    raise HTTPException(422, f"Thought '{handle}' has no hidden state (encode or think first).")


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/encode
# ══════════════════════════════════════════════════════════════════════════════

class EncodeRequest(BaseModel):
    text: str = Field(..., description="Text to encode.")
    return_layers: List[int] = Field(
        [-1], description="Layer indices to return. -1 = last layer."
    )
    return_attention: bool = Field(False, description="Also return attention weights.")
    session_id: Optional[str] = Field(
        None, description="If provided, store the result in this session."
    )
    compact: bool = Field(
        True,
        description=(
            "If true (default), hidden states are base64-encoded float32 bytes. "
            "If false, returned as nested float lists (human-readable, ~4× larger)."
        ),
    )


class EncodeResponse(BaseModel):
    handle: Optional[str]
    hidden_states: Dict[str, Any]  # layer_key → base64 str or list[list[float]]
    tokens: List[str]
    seq_len: int
    hidden_dim: int


@app.post("/v1/encode", response_model=EncodeResponse, tags=["ERIS v5"])
async def v1_encode(req: EncodeRequest):
    """
    Encode text and expose per-layer hidden states.

    Performs a single forward pass and returns the hidden states for the
    requested transformer layers.  Hidden states are serialised as base64
    float32 bytes by default (``compact=true``) or as nested float lists
    (``compact=false``).

    Use the helper ``eris_client.decode_hidden_states(response)`` to
    deserialise base64 payloads back to numpy arrays.

    Example::

        curl -X POST http://localhost:8001/v1/encode \\
          -H 'Content-Type: application/json' \\
          -d '{"text": "Hello world", "return_layers": [15, -1], "compact": false}'
    """
    eng = _require_engine()
    try:
        result = eng.encode(
            req.text,
            return_layers=req.return_layers,
            return_attention=req.return_attention,
            session_id=req.session_id,
            compact=req.compact,
        )
        return EncodeResponse(**result)
    except ValueError as e:
        raise HTTPException(404, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/analyze
# ══════════════════════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    handle: str = Field(..., description="Thought handle (enc_* or t_*).")
    session_id: str = Field(..., description="Session that owns the thought.")
    analyses: List[str] = Field(
        ["sae_features", "a_hat", "cosine_map", "pca_3d", "token_norms"],
        description=(
            "Analyses to run. Valid values: sae_features, a_hat, cosine_map, "
            "pca_3d, token_norms."
        ),
    )


@app.post("/v1/analyze", tags=["ERIS v5"])
async def v1_analyze(req: AnalyzeRequest):
    """
    Run MI analyses on the hidden states of a stored thought.

    Uses Pattern B: the hidden state is fetched from CPU storage and
    temporarily re-uploaded to GPU for SAE / Â-hat inference.  Analyzers
    that are not configured (SAE path missing, etc.) return null for their
    key without raising.

    Example::

        curl -X POST http://localhost:8001/v1/analyze \\
          -H 'Content-Type: application/json' \\
          -d '{"handle": "enc_abc123", "session_id": "...",
               "analyses": ["sae_features", "a_hat", "token_norms"]}'
    """
    eng      = _require_engine()
    registry = _eris_registry
    if registry is None:
        raise HTTPException(503, "Analyzer registry not initialised.")

    hidden = _get_thought_hidden(req.session_id, req.handle)

    result = registry.run(
        hidden,
        req.analyses,
        inference_device=eng.device,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/latent_think  (extended /think with trajectory + perturbation)
# ══════════════════════════════════════════════════════════════════════════════

class LatentThinkRequest(BaseModel):
    session_id:  str              = Field(...)
    prompt:      str              = Field(...)
    n_steps:     int              = Field(60, ge=0, le=500)
    role:        str              = Field("general")
    inherit_from: Optional[List[str]] = Field(None)
    # ERIS extensions
    return_trajectory: bool       = Field(False)
    trajectory_analyses: List[str] = Field(
        [],
        description="Analyses to run at each trajectory step. Only 'a_hat' is "
                    "cheap enough for inline use; 'sae_features' at every step "
                    "is expensive and should be avoided.",
    )
    perturbation: Optional[List[float]] = Field(
        None,
        description=(
            "Additive perturbation vector (float32, length = hidden_dim). "
            "Applied once to the initial hidden state before the rollout starts."
        ),
    )


class LatentThinkResponse(BaseModel):
    handle:        str
    session_id:    str
    role:          str
    n_steps:       int
    n_positions:   int
    elapsed_s:     float
    hidden_norm:   float
    trajectory:    Optional[List[Dict[str, Any]]] = None
    total_displacement: Optional[float] = None


@app.post("/v1/latent_think", response_model=LatentThinkResponse, tags=["ERIS v5"])
async def v1_latent_think(req: LatentThinkRequest):
    """
    Extended latent reasoning endpoint with optional trajectory tracking.

    Identical to ``POST /think`` but adds:
    - ``return_trajectory``: record hidden-state metrics at each rollout step.
    - ``trajectory_analyses``: run lightweight analyzers inline during the rollout.
      Use ``["a_hat"]`` only — SAE at every step is prohibitively expensive.
    - ``perturbation``: steer the starting hidden state before the rollout.

    Example::

        curl -X POST http://localhost:8001/v1/latent_think \\
          -H 'Content-Type: application/json' \\
          -d '{
            "session_id": "...", "prompt": "...", "n_steps": 60,
            "return_trajectory": true, "trajectory_analyses": ["a_hat"]
          }'
    """
    eng = _require_engine()

    # Build a_hat callback if requested and available
    a_hat_fn = None
    if "a_hat" in req.trajectory_analyses and _eris_registry is not None:
        if _eris_registry.a_hat is not None:
            a_hat_fn = _eris_registry.a_hat.as_callback()

    # Convert perturbation list → tensor
    pert: Optional[torch.Tensor] = None
    if req.perturbation is not None:
        pert = torch.tensor(req.perturbation, dtype=torch.float32)

    try:
        result = eng.think(
            session_id=req.session_id,
            prompt=req.prompt,
            n_steps=req.n_steps,
            role=req.role,
            inherit_from=req.inherit_from,
            return_trajectory=req.return_trajectory,
            perturbation=pert,
            a_hat_fn=a_hat_fn,
        )
        return LatentThinkResponse(**result)
    except ValueError as e:
        raise HTTPException(404, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/inject
# ══════════════════════════════════════════════════════════════════════════════

class InjectRequest(BaseModel):
    handle:    str   = Field(..., description="Thought handle to modify.")
    session_id: str  = Field(..., description="Session that owns the thought.")
    operation: str   = Field(
        "add",
        description="'add': h + scale·v  |  'steer': add + renorm  |  'replace': v",
    )
    layer:     int   = Field(-1, description="Layer index (-1 = last).")
    position:  int   = Field(-1, description="Sequence position (-1 = last token).")
    vector:    List[float] = Field(..., description="Injection vector (float32).")
    scale:     float = Field(1.0, description="Scaling factor (ignored for 'replace').")


class InjectResponse(BaseModel):
    status:       str
    operation:    str
    old_norm:     float
    new_norm:     float
    cosine_shift: float


@app.post("/v1/inject", response_model=InjectResponse, tags=["ERIS v5"])
async def v1_inject(req: InjectRequest):
    """
    Surgical vector injection into a stored thought's hidden state.

    Modifies ``hidden_embedding`` (or a specific position in
    ``layer_hidden_states``) in-place.  The KV-cache is NOT modified —
    to propagate the injection into generated text, follow up with::

        POST /v1/latent_think  {session_id, prompt: "", n_steps: 0,
                                 inherit_from: [injected_handle]}

    Operations:
    - ``add``     — h_new = h_old + scale × v
    - ``steer``   — h_new = (h_old + scale × v) renormalised to ‖h_old‖
    - ``replace`` — h_new = v  (scale ignored)

    Example::

        curl -X POST http://localhost:8001/v1/inject \\
          -H 'Content-Type: application/json' \\
          -d '{
            "handle": "t_abc123", "session_id": "...",
            "operation": "steer", "layer": -1, "position": -1,
            "vector": [0.1, -0.2, ...], "scale": 0.5
          }'
    """
    eng = _require_engine()

    with eng._lock:
        session = eng._sessions.get(req.session_id)
        if session is None:
            raise HTTPException(404, f"Session '{req.session_id}' not found.")
        thought = session.thoughts.get(req.handle)
        if thought is None:
            raise HTTPException(404, f"Thought '{req.handle}' not found.")

        v = torch.tensor(req.vector, dtype=torch.float32)
        try:
            result = inject(
                thought,
                v,
                operation=req.operation,
                layer=req.layer,
                position=req.position,
                scale=req.scale,
            )
        except InjectionError as e:
            raise HTTPException(422, str(e))

    return InjectResponse(**result.to_dict())


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/bridge
# ══════════════════════════════════════════════════════════════════════════════

class BridgeRequest(BaseModel):
    claude_text:   str       = Field(..., description="Text input from Claude.")
    mode:          str       = Field(
        "passive",
        description="'passive': encode+analyze+[decode]  |  "
                    "'ruminate': full rollout  |  "
                    "'analyze_only': encode+analyze only",
    )
    n_steps:       int       = Field(60, ge=0, le=500,
                                     description="Rollout steps (ruminate only).")
    analyses:      List[str] = Field(
        ["sae_features", "a_hat", "token_norms"],
        description="Analyses to run on the final hidden state.",
    )
    decode_after:  bool      = Field(True, description="Generate text after the pipeline.")
    max_new_tokens: int      = Field(512, ge=1, le=4096)
    temperature:   float     = Field(0.6, ge=0.0, le=2.0)
    top_p:         float     = Field(0.95, ge=0.0, le=1.0)
    session_id:    Optional[str] = Field(
        None,
        description="Reuse an existing session for multi-turn conversations. "
                    "If None, a temporary session is created and destroyed.",
    )
    implicit_min_activation: float = Field(
        0.0, description="Minimum SAE activation for implicit feature detection."
    )
    implicit_match_mode: str = Field(
        "any",
        description="'any': feature present if any label word hits text surface. "
                    "'all': feature present only if all label words hit.",
    )


@app.post("/v1/bridge", tags=["ERIS v5"])
async def v1_bridge(req: BridgeRequest):
    """
    Full Claude → Zombie → Claude bridge pipeline.

    Orchestrates the complete ERIS v5 pipeline:

    - ``passive``      → encode → analyze → generate
    - ``analyze_only`` → encode → analyze (no generation)
    - ``ruminate``     → encode → analyze(pre) → latent rollout → analyze(post) → generate

    Returns ``enriched_text`` (zombie's response after rumination),
    ``analysis`` (SAE features, Â-hat score, implicit features, norms),
    and ``trajectory_summary`` (displacement, max_a_hat, convergence step).

    Implicit features are SAE activations present in the hidden space but
    absent from the surface text — concepts the zombie is "thinking about"
    without being told.

    Example::

        curl -X POST http://localhost:8001/v1/bridge \\
          -H 'Content-Type: application/json' \\
          -d '{
            "claude_text": "What are the risks of distributed locking?",
            "mode": "ruminate",
            "n_steps": 60,
            "analyses": ["sae_features", "a_hat", "token_norms"],
            "decode_after": true,
            "max_new_tokens": 256
          }'
    """
    bridge = _require_bridge()
    try:
        result = bridge.run(
            claude_text=req.claude_text,
            mode=req.mode,
            n_steps=req.n_steps,
            analyses=req.analyses,
            decode_after=req.decode_after,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            session_id=req.session_id,
            implicit_min_activation=req.implicit_min_activation,
            implicit_match_mode=req.implicit_match_mode,
        )
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(422, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/probe
# Pure activation extraction — no generation, no session state.
# ══════════════════════════════════════════════════════════════════════════════

class ProbeRequest(BaseModel):
    text:     str             = Field(..., description="Input text to encode.")
    layers:   List[int]       = Field(
        [-1], description="Layer indices to extract (-1 = last layer)."
    )
    pooling:  str             = Field(
        "last_token",
        description="Pooling strategy: 'last_token' (default) or 'mean'.",
    )
    centered: bool            = Field(
        True,
        description="Subtract per-layer sequence mean before pooling.",
    )


class ProbeResponse(BaseModel):
    activations: Dict[str, List[float]] = Field(
        ...,
        description="Layer index (as string) → activation vector.",
    )
    input_tokens: int
    elapsed_s: float
    model: str


@app.post("/v1/probe", response_model=ProbeResponse, tags=["ERIS v5"])
async def v1_probe(req: ProbeRequest):
    """
    Extract hidden-state activations for one input text.

    Pure representation extraction — no text generation, no session state.
    The zombie model is called with a forward pass only (no generate()).
    Returns {layer_idx: activation_vector} for each requested layer.

    This is the primary interface to the zombie in the new ERIS paradigm.
    Use DriftDetector + ERISOrchestrator to decide when to call this.
    """
    eng = _require_engine()

    try:
        # Use the engine's encode() infrastructure for consistency.
        # We request the specific layers and pool client-side.
        import base64

        result_activations: Dict[str, List[float]] = {}
        t0 = time.time()

        for layer_idx in req.layers:
            enc = eng.encode(req.text, return_layers=[layer_idx], session_id=None)
            hd = enc.get("hidden_dim", 0)
            hs = enc.get("hidden_states", {})

            for val in hs.values():
                if isinstance(val, str):
                    import numpy as _np
                    flat = _np.frombuffer(base64.b64decode(val), dtype=_np.float32)
                    mat  = flat.reshape(-1, hd)       # [seq, hidden_dim]
                elif isinstance(val, list):
                    import numpy as _np
                    mat = _np.array(val, dtype=_np.float32)
                    if mat.ndim == 1:
                        mat = mat.reshape(1, -1)
                else:
                    continue

                if req.centered:
                    mat = mat - mat.mean(axis=0, keepdims=True)

                if req.pooling == "last_token":
                    vec = mat[-1]
                elif req.pooling == "mean":
                    vec = mat.mean(axis=0)
                else:
                    raise HTTPException(422, f"Unknown pooling: {req.pooling!r}")

                result_activations[str(layer_idx)] = vec.tolist()
                break

        n_tokens = len(eng.tokenizer(req.text)["input_ids"])
        elapsed  = time.time() - t0
        model_name = getattr(eng, "model_name", "unknown")

        return ProbeResponse(
            activations=result_activations,
            input_tokens=n_tokens,
            elapsed_s=round(elapsed, 4),
            model=model_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        _eris_log.exception("v1_probe error")
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ERIS V2 — /v1/sae_probe  (Gemma Scope 2 SAE features)
# ══════════════════════════════════════════════════════════════════════════════

# SAEProbe est lazy-loaded à la première requête pour ne pas bloquer le startup
# si sae-lens n'est pas installé.
_sae_probe = None
_sae_probe_lock = __import__("threading").Lock()


class SAEProbeRequest(BaseModel):
    text:       str   = Field(..., description="Texte à encoder via SAE.")
    model_id:   str   = Field(
        "google/gemma-3-9b-it",
        description="Modèle zombie : 'google/gemma-3-9b-it' ou 'google/gemma-3-27b-it'.",
    )
    layers:     List[int] = Field(
        [10, 20, 30],
        description="Layers à sonder. Pour Gemma 3 9B : [10, 20, 30] recommandé.",
    )
    sae_width:  str   = Field("16k",    description="Largeur SAE : 16k, 64k, 256k, 1m.")
    l0:         str   = Field("medium", description="Sparsité SAE : small, medium, big.")
    top_k:      int   = Field(20, ge=1, le=500, description="Features top-K retournées.")
    device:     str   = Field("cuda",   description="Device torch.")


class SAEProbeResponse(BaseModel):
    layers: Dict[str, Any] = Field(
        description="Par layer : {active_feature_indices, active_feature_values, "
                    "n_active, n_all_active}.",
    )
    model_id:   str
    input_tokens: int
    elapsed_s:  float


@app.post("/v1/sae_probe", tags=["ERIS V2"])
async def v1_sae_probe(req: SAEProbeRequest):
    """
    Extraction de features SAE via Gemma Scope 2.

    Remplace /v1/encode pour les expériences de drift detection V2.
    Le modèle zombie (Gemma 3 9B ou 27B) effectue un forward pass et
    encode les activations via les SAEs Gemma Scope 2.

    Retourne les features SAE actives (indices + valeurs) pour chaque
    layer demandé — pas d'activations brutes dans la réponse.

    Les anciens endpoints (/v1/encode, /v1/bridge, etc.) restent intacts.

    Prérequis :
        pip install sae-lens
        pip install transformer-lens>=3.0.0b0

    Exemple ::

        curl -X POST http://localhost:8001/v1/sae_probe \\
          -H 'Content-Type: application/json' \\
          -d '{
            "text": "Find all integers n such that n^2 + 1 is divisible by 5.",
            "layers": [10, 20, 30],
            "top_k": 20
          }'
    """
    global _sae_probe

    try:
        # Lazy-load SAEProbe (bloquant à la première requête)
        with _sae_probe_lock:
            if _sae_probe is None or (
                _sae_probe.model_id   != req.model_id
                or _sae_probe.sae_width != req.sae_width
                or _sae_probe.l0        != req.l0
            ):
                _eris_log.info(
                    "Chargement SAEProbe: model=%s sae=%s/l0_%s layers=%s",
                    req.model_id, req.sae_width, req.l0, req.layers,
                )
                from eris.sae_probe import SAEProbe
                _sae_probe = SAEProbe(
                    model_id=req.model_id,
                    layers=req.layers,
                    sae_width=req.sae_width,
                    l0=req.l0,
                    device=req.device,
                )

        # Probe
        t0 = time.time()
        probe_out = _sae_probe.probe(req.text, top_k=req.top_k)
        elapsed   = round(time.time() - t0, 4)

        # Sérialiser
        layers_resp: dict[str, Any] = {}
        for layer_idx, out in probe_out.items():
            layers_resp[str(layer_idx)] = {
                "active_feature_indices": out.active_feature_indices,
                "active_feature_values":  out.active_feature_values,
                "n_active":               out.n_active,
                "n_all_active":           len(out.all_active_indices),
            }

        # Compter les tokens via le tokenizer du SAEProbe
        n_tokens = len(_sae_probe._tokenizer(req.text)["input_ids"])

        return SAEProbeResponse(
            layers=layers_resp,
            model_id=req.model_id,
            input_tokens=n_tokens,
            elapsed_s=elapsed,
        )

    except ImportError as e:
        raise HTTPException(
            503,
            detail=str(e) + " — installer : pip install sae-lens",
        )
    except HTTPException:
        raise
    except Exception as e:
        _eris_log.exception("v1_sae_probe error")
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERIS v5 Server")
    parser.add_argument("--model",  default="Qwen/Qwen3-14B",
                        help="HuggingFace model name or local path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port",   type=int, default=8001)
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--config", default=None,
                        help="Path to eris_config.yaml (default: configs/eris_config.yaml)")
    args = parser.parse_args()

    os.environ["LATENT_MODEL"]  = args.model
    os.environ["LATENT_DEVICE"] = args.device
    if args.config:
        os.environ["ERIS_CONFIG"] = args.config

    uvicorn.run(app, host=args.host, port=args.port)
