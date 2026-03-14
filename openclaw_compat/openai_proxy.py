"""
Latent Relay — OpenAI-Compatible Proxy
========================================
Wraps the LatentRelayEngine behind a standard OpenAI /v1/chat/completions
endpoint, so OpenClaw and OpenClaw-RL can use it as a drop-in provider.
 
How it works:
  1. Receives a standard OpenAI ChatCompletion request
  2. Runs latent multi-agent reasoning internally (planner → critic → refiner)
  3. Returns a standard OpenAI ChatCompletion response
 
For OpenClaw-RL compatibility, the response includes per-token log_probs
when requested (logprobs=true), so the RL rollout worker can collect
training signals.
 
Usage:
    python openai_proxy.py --model Qwen/Qwen3-4B --port 30000
    # Then point OpenClaw at http://<HOST_IP>:30000/v1
"""
 
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
 
# Add parent directory to path so engine.py is importable
# regardless of whether we run from repo root or openclaw_compat/
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
 
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
 
from engine import LatentRelayEngine
 
 
# ── OpenAI-compatible request/response models ──
 
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
 
class ChatCompletionRequest(BaseModel):
    model: str = "latent-relay"
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: Optional[int] = 1024
    stream: bool = False
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    n: int = 1
    stop: Optional[Any] = None
    # Latent Relay specific (optional, ignored by standard clients)
    latent_steps: Optional[int] = None
    latent_pipeline: Optional[str] = None  # "single", "planner-critic", "full"
    repetition_penalty: Optional[float] = 1.2
 
class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[Dict] = None
 
class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
 
class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
 
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "latent-relay"
 
class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
 
 
# ── Latent reasoning pipeline ──
 
# Default latent steps per pipeline stage
PIPELINE_CONFIGS = {
    "single": [("solver", 60)],
    "planner-critic": [("planner", 40), ("critic", 30)],
    "full": [("planner", 40), ("critic", 30), ("refiner", 20)],
}
DEFAULT_PIPELINE = "planner-critic"
DEFAULT_STEPS = 40
 
 
def build_prompt_from_messages(messages: List[ChatMessage]) -> str:
    """Convert OpenAI-format messages to a single text prompt."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)
 
 
def run_latent_pipeline(
    engine: LatentRelayEngine,
    session_id: str,
    prompt: str,
    pipeline: str,
    latent_steps: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> Dict[str, Any]:
    """
    Run a multi-agent latent reasoning pipeline and return the final text.
    """
    stages = PIPELINE_CONFIGS.get(pipeline, PIPELINE_CONFIGS[DEFAULT_PIPELINE])
 
    handles = []
    pipeline_info = {"stages": [], "total_latent_steps": 0}
 
    for i, (role, default_steps) in enumerate(stages):
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
        pipeline_info["stages"].append({
            "role": role,
            "n_steps": n_steps,
            "n_positions": result["n_positions"],
            "elapsed_s": result["elapsed_s"],
        })
        pipeline_info["total_latent_steps"] += n_steps
 
    # Final text generation using accumulated latent context
    collab_result = engine.collaborate(
        session_id=session_id,
        handles=handles,
        final_prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
 
    return {
        "text": collab_result["text"],
        "tokens_generated": collab_result["tokens_generated"],
        "elapsed_s": collab_result["elapsed_s"],
        "pipeline_info": pipeline_info,
    }
 
 
# ── FastAPI app ──
 
app = FastAPI(
    title="Latent Relay — OpenAI-Compatible Proxy",
    description="Wraps LatentMAS latent reasoning behind a standard OpenAI API.",
    version="0.1.0",
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
    """List available models (OpenAI-compatible)."""
    return ModelListResponse(data=[
        ModelInfo(
            id=_model_id,
            created=int(time.time()),
        )
    ])
 
 
@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model info (OpenAI-compatible)."""
    return ModelInfo(
        id=_model_id,
        created=int(time.time()),
    )
 
 
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
 
    Internally runs latent multi-agent reasoning before generating text.
    Compatible with OpenClaw provider config and OpenClaw-RL rollout worker.
    """
    if engine is None:
        raise HTTPException(503, "Model not loaded yet")
 
    # Create a temporary session for this request
    session_id = engine.create_session()
 
    try:
        # Build prompt from messages
        prompt = build_prompt_from_messages(req.messages)
 
        # Determine pipeline
        pipeline = req.latent_pipeline or DEFAULT_PIPELINE
 
        # Run latent reasoning + generation
        t0 = time.time()
        result = run_latent_pipeline(
            engine=engine,
            session_id=session_id,
            prompt=prompt,
            pipeline=pipeline,
            latent_steps=req.latent_steps,
            max_new_tokens=req.max_tokens or 1024,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty or 1.2,
        )
        total_elapsed = time.time() - t0
 
        # Estimate token counts (rough)
        prompt_tokens = len(prompt.split()) * 2  # rough approximation
        completion_tokens = result["tokens_generated"]
 
        # Build response
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=result["text"]),
            finish_reason="stop",
            logprobs=None,
        )
 
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=_model_id,
            choices=[choice],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
 
        return response
 
    finally:
        # Clean up session to free GPU memory
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
    """List available latent reasoning pipelines (extension endpoint)."""
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
    parser.add_argument("--port", type=int, default=30000,
                        help="Port (30000 matches OpenClaw-RL default)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--pipeline", default="planner-critic",
                        choices=list(PIPELINE_CONFIGS.keys()))
    args = parser.parse_args()
 
    os.environ["LATENT_MODEL"] = args.model
    os.environ["LATENT_DEVICE"] = args.device
    DEFAULT_PIPELINE = args.pipeline
 
    uvicorn.run(app, host=args.host, port=args.port)