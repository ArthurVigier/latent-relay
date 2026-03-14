"""
Latent Relay MCP Server
========================
Exposes LatentMAS latent-space collaboration as MCP tools.

Any MCP-compatible agent (Claude, GPT, custom) can discover and use:
  - latent_create_session  → initialize a reasoning session
  - latent_think           → run latent reasoning, get a handle
  - latent_collaborate     → combine thoughts into a text answer
  - latent_thought_info    → inspect a stored thought
  - latent_delete_session  → free resources

Usage:
    # Start MCP server (stdio transport for local use)
    python mcp_server.py --model Qwen/Qwen3-4B

    # Or with environment variable
    LATENT_MODEL=Qwen/Qwen3-4B python mcp_server.py
"""

import os
import json
from typing import Optional, List

from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

from engine import LatentRelayEngine

# ── Initialize ──

mcp = FastMCP("latent_relay_mcp")

engine: Optional[LatentRelayEngine] = None


def get_engine() -> LatentRelayEngine:
    global engine
    if engine is None:
        model = os.environ.get("LATENT_MODEL", "Qwen/Qwen3-4B")
        device = os.environ.get("LATENT_DEVICE", "cuda:0")
        engine = LatentRelayEngine(model_name=model, device=device)
    return engine


# ── Input Models ──

class CreateSessionInput(BaseModel):
    """No parameters needed — creates a new session on the loaded model."""
    model_config = ConfigDict(extra="forbid")


class ThinkInput(BaseModel):
    """Input for latent reasoning."""
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Session ID from latent_create_session")
    prompt: str = Field(..., description="Text prompt for the agent to reason about. Can be a question, instruction, or sub-task.")
    n_steps: int = Field(
        default=60, ge=0, le=200,
        description="Number of latent rollout steps. 0=encode only, 40-80=recommended for reasoning. More steps = deeper thinking but diminishing returns past ~80."
    )
    role: str = Field(
        default="general",
        description="Agent role label. Helps organize multi-agent workflows. Common roles: planner, critic, refiner, solver, general."
    )
    inherit_from: Optional[List[str]] = Field(
        default=None,
        description="List of thought handles from previous latent_think calls. The agent inherits their accumulated context via KV-cache."
    )


class CollaborateInput(BaseModel):
    """Input for combining thoughts into text."""
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Session ID")
    handles: List[str] = Field(
        ..., min_length=1,
        description="Ordered list of thought handles from latent_think calls. The last handle's full context is used for generation."
    )
    final_prompt: str = Field(
        ...,
        description="Text prompt for the final answer. The model generates text conditioned on all the latent reasoning context."
    )
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)


class ThoughtInfoInput(BaseModel):
    """Input for querying thought metadata."""
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Session ID")
    handle: str = Field(..., description="Thought handle from a latent_think call")


class DeleteSessionInput(BaseModel):
    """Input for deleting a session."""
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Session ID to delete")


# ── Tools ──

@mcp.tool(
    name="latent_create_session",
    annotations={
        "title": "Create Latent Reasoning Session",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def latent_create_session(params: CreateSessionInput) -> str:
    """Create a new session for latent multi-agent reasoning.

    A session holds the model state and stores latent thoughts (KV-cache handles).
    Create one session per task. Multiple agents can think within the same session
    and share context through thought handles.

    Returns a session_id to use with latent_think and latent_collaborate.
    """
    eng = get_engine()
    sid = eng.create_session()
    return json.dumps({
        "session_id": sid,
        "model": eng.model_name,
        "instructions": "Use this session_id with latent_think and latent_collaborate."
    })


@mcp.tool(
    name="latent_think",
    annotations={
        "title": "Run Latent Reasoning",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def latent_think(params: ThinkInput) -> str:
    """Run latent-space reasoning on a prompt — no text output generated.

    The agent processes the prompt through the model, then performs n_steps of
    latent rollout where the hidden state is iteratively refined through the
    model's layers. This is equivalent to "silent thinking" — the model reasons
    without producing any tokens.

    The result is stored as a handle pointing to the accumulated KV-cache.
    Pass this handle to other latent_think calls (via inherit_from) to chain
    agents, or to latent_collaborate to generate a final text answer.

    Typical workflow:
      1. latent_think(prompt="Plan how to solve X", role="planner", n_steps=60)
      2. latent_think(prompt="Critique the plan", role="critic", n_steps=40, inherit_from=[handle_1])
      3. latent_think(prompt="Refine the plan", role="refiner", n_steps=60, inherit_from=[handle_2])
      4. latent_collaborate(handles=[handle_3], final_prompt="Solve X and give the answer")
    """
    eng = get_engine()
    try:
        result = eng.think(
            session_id=params.session_id,
            prompt=params.prompt,
            n_steps=params.n_steps,
            role=params.role,
            inherit_from=params.inherit_from,
        )
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="latent_collaborate",
    annotations={
        "title": "Generate Text from Latent Thoughts",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def latent_collaborate(params: CollaborateInput) -> str:
    """Combine latent thoughts from multiple agents and generate a text answer.

    Takes thought handles from previous latent_think calls and uses their
    accumulated reasoning context (KV-cache) to generate a text response.
    This is the only step that produces actual text output.

    The latent context contains rich, continuous reasoning that would have
    required hundreds or thousands of text tokens to express. The final
    generation benefits from this dense context while only generating
    the concise answer.
    """
    eng = get_engine()
    try:
        result = eng.collaborate(
            session_id=params.session_id,
            handles=params.handles,
            final_prompt=params.final_prompt,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
        )
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="latent_thought_info",
    annotations={
        "title": "Get Thought Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def latent_thought_info(params: ThoughtInfoInput) -> str:
    """Get metadata about a stored latent thought.

    Returns the role, number of KV-cache positions, creation time,
    and other metadata. Useful for inspecting the reasoning chain.
    """
    eng = get_engine()
    info = eng.get_thought_info(params.session_id, params.handle)
    if info is None:
        return json.dumps({"error": f"Thought {params.handle} not found"})
    return json.dumps(info, default=str)


@mcp.tool(
    name="latent_delete_session",
    annotations={
        "title": "Delete Session",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def latent_delete_session(params: DeleteSessionInput) -> str:
    """Delete a session and free GPU memory.

    All thought handles in the session become invalid. Use when done
    with a task to free resources.
    """
    eng = get_engine()
    ok = eng.delete_session(params.session_id)
    if not ok:
        return json.dumps({"error": f"Session {params.session_id} not found"})
    return json.dumps({"deleted": params.session_id, "status": "ok"})


if __name__ == "__main__":
    import sys
    model = os.environ.get("LATENT_MODEL", "Qwen/Qwen3-4B")
    device = os.environ.get("LATENT_DEVICE", "cuda:0")
    print(f"[MCP] Starting Latent Relay MCP Server")
    print(f"[MCP] Model: {model} | Device: {device}")
    print(f"[MCP] Transport: stdio")
    mcp.run(transport="stdio")
