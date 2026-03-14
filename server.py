"""
Latent Relay Server — FastAPI
==============================
REST API wrapping the LatentRelayEngine.

Endpoints:
    POST /sessions           → create session
    GET  /sessions           → list sessions
    DELETE /sessions/{id}    → delete session
    POST /think              → run latent reasoning, return handle
    POST /collaborate        → combine thoughts, generate text
    GET  /thoughts/{handle}  → get thought metadata
    GET  /health             → health check

Usage:
    python server.py --model Qwen/Qwen3-4B --port 8000
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import argparse
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from engine import LatentRelayEngine

# ── Request/Response models ──

class CreateSessionResponse(BaseModel):
    session_id: str
    model: str

class ThinkRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    prompt: str = Field(..., description="Text prompt for the agent to reason about")
    n_steps: int = Field(60, ge=0, le=200, description="Number of latent rollout steps (0=encode only)")
    role: str = Field("general", description="Agent role: planner, critic, refiner, solver, general")
    inherit_from: Optional[List[str]] = Field(None, description="Thought handles to inherit KV-cache from")

class ThinkResponse(BaseModel):
    handle: str
    session_id: str
    role: str
    n_steps: int
    n_positions: int
    elapsed_s: float
    hidden_norm: float

class CollaborateRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    handles: List[str] = Field(..., description="Ordered list of thought handles to use as context")
    final_prompt: str = Field(..., description="Text prompt for final text generation")
    max_new_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)

class CollaborateResponse(BaseModel):
    text: str
    tokens_generated: int
    handles_used: List[str]
    elapsed_s: float

class ThoughtInfo(BaseModel):
    handle: str
    session_id: str
    role: str
    n_positions: int
    created_at: float
    metadata: dict

# ── App ──

app = FastAPI(
    title="Latent Relay Server",
    description="Exposes LatentMAS latent-space multi-agent reasoning as a service. "
                "Agents think in latent space (no tokens generated), exchange KV-cache handles, "
                "and only produce text at the final collaboration step.",
    version="0.1.0",
)

engine: Optional[LatentRelayEngine] = None


@app.on_event("startup")
async def startup():
    global engine
    model_name = os.environ.get("LATENT_MODEL", "Qwen/Qwen3-4B")
    device = os.environ.get("LATENT_DEVICE", "cuda:0")
    engine = LatentRelayEngine(model_name=model_name, device=device)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": engine.model_name if engine else None,
        "device": engine.device if engine else None,
    }


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session():
    """Create a new session for latent reasoning."""
    sid = engine.create_session()
    return CreateSessionResponse(session_id=sid, model=engine.model_name)


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return engine.list_sessions()


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free GPU memory."""
    ok = engine.delete_session(session_id)
    if not ok:
        raise HTTPException(404, f"Session {session_id} not found")
    return {"deleted": session_id}


@app.post("/think", response_model=ThinkResponse)
async def think(req: ThinkRequest):
    """
    Run latent reasoning on a prompt.

    The agent processes the prompt through the model, then performs n_steps of
    latent rollout (h → W_a → forward → h'). No text is generated. The result
    is stored as a KV-cache handle that can be passed to other agents or to
    the /collaborate endpoint.
    """
    try:
        result = engine.think(
            session_id=req.session_id,
            prompt=req.prompt,
            n_steps=req.n_steps,
            role=req.role,
            inherit_from=req.inherit_from,
        )
        return ThinkResponse(**result)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/collaborate", response_model=CollaborateResponse)
async def collaborate(req: CollaborateRequest):
    """
    Combine latent thoughts from multiple agents and generate a text answer.

    Takes an ordered list of thought handles (from /think calls) and a final
    prompt. The accumulated KV-cache context from the thoughts is used as a
    prefix for text generation.
    """
    try:
        result = engine.collaborate(
            session_id=req.session_id,
            handles=req.handles,
            final_prompt=req.final_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return CollaborateResponse(**result)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/thoughts/{session_id}/{handle}", response_model=ThoughtInfo)
async def get_thought(session_id: str, handle: str):
    """Get metadata about a stored latent thought."""
    info = engine.get_thought_info(session_id, handle)
    if info is None:
        raise HTTPException(404, f"Thought {handle} not found in session {session_id}")
    return ThoughtInfo(**info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Relay Server")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    os.environ["LATENT_MODEL"] = args.model
    os.environ["LATENT_DEVICE"] = args.device

    uvicorn.run(app, host=args.host, port=args.port)
