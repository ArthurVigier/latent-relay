"""
Latent Relay — OpenAI-Compatible Proxy
========================================
Wraps the LatentRelayEngine behind a standard OpenAI /v1/chat/completions
endpoint, so OpenClaw and OpenClaw-RL can use it as a drop-in provider.

Accepts ANY OpenAI-compatible request body (tools, tool_choice, stream, etc.)
without strict validation. Extracts only the fields needed for latent reasoning.

Usage:
    LATENT_MODEL=Qwen/Qwen3-8B python openai_proxy.py --port 30000
    # Then point OpenClaw at http://<HOST_IP>:30000/v1
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from engine import LatentRelayEngine


# ── Text cleanup ──

# Patterns to strip from generated text
STRIP_PATTERNS = [
    r'(?:\s*NO_REPLY\s*)+$',           # Trailing NO_REPLY tokens
    r'(?:\s*Assistant\s*)+$',           # Trailing "Assistant" repetitions
    r'(?:\s*<\|im_end\|>\s*)+',        # Leaked special tokens
    r'(?:\s*<\|endoftext\|>\s*)+',      # Leaked EOS tokens
]

def clean_generated_text(text: str) -> str:
    """Strip model artifacts from generated text."""
    for pattern in STRIP_PATTERNS:
        text = re.sub(pattern, '', text)
    return text.strip()


# ── Latent reasoning pipeline ──

PIPELINE_CONFIGS = {
    "single": [("solver", 60)],
    "planner-critic": [("planner", 40), ("critic", 30)],
    "full": [("planner", 40), ("critic", 30), ("refiner", 20)],
}
DEFAULT_PIPELINE = "planner-critic"


def build_prompt_from_messages(messages: list) -> str:
    """Convert OpenAI-format messages to a single text prompt."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Handle content arrays (vision messages, etc.)
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def run_latent_pipeline(engine, session_id, prompt, pipeline, latent_steps,
                        max_new_tokens, temperature, top_p):
    """Run multi-agent latent reasoning pipeline and return final text."""
    stages = PIPELINE_CONFIGS.get(pipeline, PIPELINE_CONFIGS[DEFAULT_PIPELINE])
    handles = []
    for role, default_steps in stages:
        n_steps = latent_steps if latent_steps is not None else default_steps
        inherit_from = [handles[-1]] if handles else None
        result = engine.think(
            session_id=session_id,
            prompt=prompt,
            n_steps=n_steps,
            role=role,
            inherit_from=inherit_from,
        )
        handles.append(result["handle"])

    collab_result = engine.collaborate(
        session_id=session_id,
        handles=handles,
        final_prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return {
        "text": clean_generated_text(collab_result["text"]),
        "tokens_generated": collab_result["tokens_generated"],
        "elapsed_s": collab_result["elapsed_s"],
    }


# ── FastAPI app ──

app = FastAPI(
    title="Latent Relay — OpenAI-Compatible Proxy",
    description="Wraps LatentMAS latent reasoning behind a standard OpenAI API.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[LatentRelayEngine] = None
_model_name: str = "latent-relay"
_model_id: str = "latent-relay"


@app.on_event("startup")
async def startup():
    global engine, _model_name, _model_id
    model_name = os.environ.get("LATENT_MODEL", "Qwen/Qwen3-4B")
    device = os.environ.get("LATENT_DEVICE", "cuda:0")
    _model_name = model_name
    _model_id = model_name.split("/")[-1].lower()
    engine = LatentRelayEngine(model_name=model_name, device=device)


# -- OpenAI-compatible endpoints --

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "latent-relay",
        }],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    return {
        "id": _model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "latent-relay",
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Accept ANY OpenAI-compatible request body without strict validation.
    Extract only the fields we need, ignore everything else
    (tools, tool_choice, response_format, etc).
    """
    if engine is None:
        raise HTTPException(503, "Model not loaded yet")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    # Extract fields we care about, ignore the rest
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(400, "messages is required")

    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.95))
    max_tokens = int(body.get("max_tokens", 1024))
    stream = body.get("stream", False)

    # Latent Relay extensions (ignored by standard clients)
    latent_steps = body.get("latent_steps")
    latent_pipeline = body.get("latent_pipeline", DEFAULT_PIPELINE)

    # Create a temporary session
    session_id = engine.create_session()

    try:
        prompt = build_prompt_from_messages(messages)

        result = run_latent_pipeline(
            engine=engine,
            session_id=session_id,
            prompt=prompt,
            pipeline=latent_pipeline,
            latent_steps=latent_steps,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_tokens = len(prompt.split()) * 2  # rough estimate
        completion_tokens = result["tokens_generated"]

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if stream:
            # Return as SSE stream (single chunk + DONE)
            async def stream_response():
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": _model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": result["text"]},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
            )

        # Non-streaming response
        return JSONResponse(content={
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": _model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })

    finally:
        engine.delete_session(session_id)


# -- Health / info endpoints --

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _model_name,
        "openai_compat": True,
        "pipelines": list(PIPELINE_CONFIGS.keys()),
        "default_pipeline": DEFAULT_PIPELINE,
    }


@app.get("/v1/latent/pipelines")
async def get_pipelines():
    return {
        "pipelines": {
            name: [{"role": r, "default_steps": s} for r, s in stages]
            for name, stages in PIPELINE_CONFIGS.items()
        },
        "default": DEFAULT_PIPELINE,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Relay OpenAI Proxy")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--pipeline", default="planner-critic",
                        choices=list(PIPELINE_CONFIGS.keys()))
    args = parser.parse_args()

    os.environ["LATENT_MODEL"] = args.model
    os.environ["LATENT_DEVICE"] = args.device
    DEFAULT_PIPELINE = args.pipeline

    uvicorn.run(app, host=args.host, port=args.port)
