"""
ERIS v5 Client
===============
Python client for the ERIS server (eris_server.py) and the full
Claude → Zombie → Claude conversation loop.

Two main classes
-----------------
:class:`ERISClient`
    Thin synchronous HTTP wrapper around all ``/v1/`` endpoints.
    No Anthropic dependency — usable standalone for direct server calls.

:class:`ClaudeZombieBridge`
    High-level conversation loop that wires Claude (Anthropic SDK) to the
    zombie model via :class:`ERISClient`.  Each turn:

      1. Sends user message → Claude → ``claude_text``
      2. Pipes ``claude_text`` through the ERIS bridge (encode / ruminate / decode)
      3. Returns a :class:`BridgeTurnResult` with both texts + analysis

Helpers
-------
:func:`decode_hidden_states`
    Deserialise base64-encoded float32 hidden states from a ``/v1/encode``
    response back to numpy arrays.

Usage — client only::

    from eris_client import ERISClient

    client = ERISClient("http://localhost:8001")
    with client.session() as sid:
        enc = client.encode("Hello world", session_id=sid)
        hs  = decode_hidden_states(enc)          # {"last": np.ndarray[seq_len, 3584]}
        ana = client.analyze(enc["handle"], sid, analyses=["token_norms"])

Usage — full loop::

    from eris_client import ClaudeZombieBridge

    bridge = ClaudeZombieBridge(
        anthropic_api_key="sk-ant-...",
        eris_base_url="http://localhost:8001",
        bridge_mode="ruminate",
        n_steps=60,
    )
    turn = bridge.chat("What are the risks of distributed locking?")
    print(turn.claude_text)
    print(turn.enriched_text)
    print(turn.analysis["implicit_features"])
"""

from __future__ import annotations

import base64
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

import httpx
import numpy as np


# ── decode_hidden_states ──────────────────────────────────────────────────────

def decode_hidden_states(response: Dict) -> Dict[str, np.ndarray]:
    """
    Deserialise hidden states from a ``POST /v1/encode`` response.

    Handles both compact (base64 float32 bytes) and verbose (nested float lists)
    formats transparently.

    Args:
        response: The JSON dict returned by ``/v1/encode``.

    Returns:
        Dict mapping layer key (e.g. ``"last"``, ``"layer_15"``) to a float32
        numpy array of shape ``[seq_len, hidden_dim]``.

    Example::

        enc = client.encode("Hello world", return_layers=[-1, 15], compact=True)
        hs  = decode_hidden_states(enc)
        print(hs["last"].shape)    # (4, 3584)
        print(hs["layer_15"].mean())
    """
    seq_len    = response["seq_len"]
    hidden_dim = response["hidden_dim"]
    result: Dict[str, np.ndarray] = {}

    for key, payload in response.get("hidden_states", {}).items():
        if isinstance(payload, str):
            # base64-encoded float32 bytes
            raw = base64.b64decode(payload)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(seq_len, hidden_dim)
            result[key] = arr.copy()   # copy: frombuffer is read-only
        else:
            # Nested list of floats
            result[key] = np.array(payload, dtype=np.float32)

    return result


# ── ERISClient ────────────────────────────────────────────────────────────────

class ERISClient:
    """
    Synchronous HTTP client for all ERIS server endpoints.

    Uses ``httpx`` for transport.  All methods raise ``httpx.HTTPStatusError``
    on 4xx/5xx responses.

    Args:
        base_url: Base URL of the ERIS server (default ``http://localhost:8001``).
        timeout:  Request timeout in seconds (default 120 — latent rollout is slow).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 120.0,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> Dict:
        """GET /health — server status and loaded model."""
        return self._get("/health")

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(self) -> str:
        """Create a new session. Returns the session_id string."""
        return self._post("/sessions")["session_id"]

    def list_sessions(self) -> List[Dict]:
        """List all active sessions."""
        return self._get("/sessions")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and free GPU memory. Returns True on success."""
        resp = self._delete(f"/sessions/{session_id}")
        return "deleted" in resp

    @contextmanager
    def session(self) -> Generator[str, None, None]:
        """
        Context manager that creates a temporary session and deletes it on exit.

        Usage::

            with client.session() as sid:
                enc = client.encode("Hello", session_id=sid)
        """
        sid = self.create_session()
        try:
            yield sid
        finally:
            self.delete_session(sid)

    # ── ERIS endpoints ────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        *,
        return_layers: List[int] = [-1],
        return_attention: bool = False,
        session_id: Optional[str] = None,
        compact: bool = True,
    ) -> Dict:
        """
        POST /v1/encode — encode text and return hidden states.

        Args:
            text:            Input text.
            return_layers:   Layer indices to return (``-1`` = last).
            return_attention: Also return attention weights.
            session_id:      Store the result in this session.
            compact:         True → base64 bytes.  False → nested float lists.

        Returns:
            Dict with ``handle``, ``hidden_states``, ``tokens``,
            ``seq_len``, ``hidden_dim``.

        Example::

            enc = client.encode("Hello", return_layers=[-1], compact=True)
            hs  = decode_hidden_states(enc)   # {"last": np.ndarray}
        """
        return self._post("/v1/encode", {
            "text":             text,
            "return_layers":    return_layers,
            "return_attention": return_attention,
            "session_id":       session_id,
            "compact":          compact,
        })

    def analyze(
        self,
        handle: str,
        session_id: str,
        analyses: List[str] = ("sae_features", "a_hat", "cosine_map", "pca_3d", "token_norms"),
    ) -> Dict:
        """
        POST /v1/analyze — run MI analyses on a stored thought.

        Args:
            handle:     Thought handle (``enc_*`` or ``t_*``).
            session_id: Session that owns the thought.
            analyses:   Analyses to run.

        Returns:
            Dict with one key per analysis.  Unavailable analyzers → None.
        """
        return self._post("/v1/analyze", {
            "handle":     handle,
            "session_id": session_id,
            "analyses":   list(analyses),
        })

    def latent_think(
        self,
        session_id: str,
        prompt: str,
        *,
        n_steps: int = 60,
        role: str = "general",
        inherit_from: Optional[List[str]] = None,
        return_trajectory: bool = False,
        trajectory_analyses: List[str] = (),
        perturbation: Optional[List[float]] = None,
    ) -> Dict:
        """
        POST /v1/latent_think — extended latent reasoning with trajectory tracking.

        Args:
            session_id:          Active session.
            prompt:              Input text.
            n_steps:             Rollout steps.
            role:                Agent role label.
            inherit_from:        KV-cache handles to inherit.
            return_trajectory:   Include per-step metrics in the response.
            trajectory_analyses: Analyses to run inline (keep to ``["a_hat"]``).
            perturbation:        Additive vector applied before the rollout.

        Returns:
            Dict with ``handle``, ``n_steps``, ``elapsed_s``, ``hidden_norm``,
            and optionally ``trajectory``, ``total_displacement``.
        """
        return self._post("/v1/latent_think", {
            "session_id":          session_id,
            "prompt":              prompt,
            "n_steps":             n_steps,
            "role":                role,
            "inherit_from":        inherit_from,
            "return_trajectory":   return_trajectory,
            "trajectory_analyses": list(trajectory_analyses),
            "perturbation":        perturbation,
        })

    def inject(
        self,
        handle: str,
        session_id: str,
        vector: Union[List[float], np.ndarray],
        *,
        operation: str = "add",
        layer: int = -1,
        position: int = -1,
        scale: float = 1.0,
    ) -> Dict:
        """
        POST /v1/inject — inject a vector into a stored thought's hidden state.

        Args:
            handle:     Thought to modify.
            session_id: Session that owns the thought.
            vector:     Injection vector (length = hidden_dim).
            operation:  ``"add"``, ``"steer"``, or ``"replace"``.
            layer:      Target layer (``-1`` = last).
            position:   Target sequence position (``-1`` = last token).
            scale:      Scaling factor (ignored for ``"replace"``).

        Returns:
            Dict with ``status``, ``operation``, ``old_norm``, ``new_norm``,
            ``cosine_shift``.

        Example — steer then re-think to propagate::

            client.inject(handle, sid, steering_vec, operation="steer", scale=0.5)
            client.latent_think(sid, "", n_steps=0, inherit_from=[handle])
        """
        v = vector.tolist() if isinstance(vector, np.ndarray) else list(vector)
        return self._post("/v1/inject", {
            "handle":     handle,
            "session_id": session_id,
            "operation":  operation,
            "layer":      layer,
            "position":   position,
            "vector":     v,
            "scale":      scale,
        })

    def bridge(
        self,
        claude_text: str,
        *,
        mode: str = "passive",
        n_steps: int = 60,
        analyses: List[str] = ("sae_features", "a_hat", "token_norms"),
        decode_after: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        session_id: Optional[str] = None,
        implicit_min_activation: float = 0.0,
        implicit_match_mode: str = "any",
    ) -> Dict:
        """
        POST /v1/bridge — full Claude → Zombie → Claude pipeline.

        Args:
            claude_text:             Text to pipe through the zombie.
            mode:                    ``"passive"``, ``"ruminate"``, or
                                     ``"analyze_only"``.
            n_steps:                 Rollout steps (``ruminate`` only).
            analyses:                Analyses to run on the final hidden state.
            decode_after:            Generate text after the pipeline.
            max_new_tokens:          Token budget for generation.
            temperature:             Sampling temperature.
            top_p:                   Nucleus sampling p.
            session_id:              Reuse an existing session (None = temporary).
            implicit_min_activation: Min SAE activation for implicit detection.
            implicit_match_mode:     ``"any"`` or ``"all"``.

        Returns:
            Dict with ``enriched_text``, ``analysis`` (including
            ``implicit_features``), ``trajectory_summary``, ``tokens``,
            ``elapsed_s``, ``handles``.
        """
        return self._post("/v1/bridge", {
            "claude_text":              claude_text,
            "mode":                     mode,
            "n_steps":                  n_steps,
            "analyses":                 list(analyses),
            "decode_after":             decode_after,
            "max_new_tokens":           max_new_tokens,
            "temperature":              temperature,
            "top_p":                    top_p,
            "session_id":               session_id,
            "implicit_min_activation":  implicit_min_activation,
            "implicit_match_mode":      implicit_match_mode,
        })

    # ── Low-level transport ───────────────────────────────────────────────────

    def _post(self, path: str, body: Optional[Dict] = None) -> Dict:
        resp = self._client.post(path, json=body or {})
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> Any:
        resp = self._client.get(path)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> Dict:
        resp = self._client.delete(path)
        resp.raise_for_status()
        return resp.json()


# ── BridgeTurnResult ──────────────────────────────────────────────────────────

@dataclass
class BridgeTurnResult:
    """
    Result of a single Claude ↔ Zombie turn.

    Attributes:
        user_message:       The original user input.
        claude_text:        Claude's raw response.
        enriched_text:      Zombie's response after rumination, or None.
        analysis:           SAE features, Â-hat score, implicit features, norms.
        trajectory_summary: Rollout summary (mode=``ruminate`` only).
        pre_analysis:       Hidden state analysis before the rollout, or None.
        tokens:             Subword tokens of ``claude_text``.
        elapsed_s:          Total wall-clock time for this turn.
        handles:            Internal thought handles created during the run.
    """
    user_message:       str
    claude_text:        str
    enriched_text:      Optional[str]
    analysis:           Dict[str, Any]
    trajectory_summary: Optional[Dict]
    pre_analysis:       Optional[Dict]
    tokens:             List[str]
    elapsed_s:          float
    handles:            Dict[str, str] = field(default_factory=dict)

    @property
    def implicit_features(self) -> List[Dict]:
        """Shortcut: implicit features from the analysis dict."""
        return self.analysis.get("implicit_features", [])

    def summary(self) -> str:
        """One-line human-readable summary of the turn."""
        parts = [f"claude={len(self.claude_text)}ch"]
        if self.enriched_text:
            parts.append(f"zombie={len(self.enriched_text)}ch")
        if self.trajectory_summary:
            parts.append(
                f"disp={self.trajectory_summary.get('total_displacement', '?'):.3f}"
            )
        n_impl = len(self.implicit_features)
        if n_impl:
            parts.append(f"implicit={n_impl}")
        return f"Turn({', '.join(parts)}, {self.elapsed_s:.1f}s)"


# ── ClaudeZombieBridge ────────────────────────────────────────────────────────

class ClaudeZombieBridge:
    """
    High-level Claude → Zombie → Claude conversation loop.

    Each call to :meth:`chat` sends a user message to Claude, pipes Claude's
    response through the ERIS zombie bridge, and returns a
    :class:`BridgeTurnResult` containing both texts plus the latent analysis.

    The zombie's ``enriched_text`` is NOT automatically fed back to Claude —
    it is available in the result for the caller to decide how to use it
    (e.g. append to the conversation, display to Arthur, or discard).

    Args:
        anthropic_api_key: Anthropic API key.
        eris_base_url:     ERIS server URL.
        model:             Claude model ID (default ``claude-sonnet-4-6``).
        bridge_mode:       ERIS bridge mode (``"passive"``, ``"ruminate"``,
                           ``"analyze_only"``).
        n_steps:           Latent rollout steps when ``bridge_mode="ruminate"``.
        analyses:          Analyses to run on every turn.
        max_new_tokens:    Token budget for zombie text generation.
        system:            System prompt for Claude (optional).
        eris_timeout:      ERIS server HTTP timeout in seconds.
    """

    def __init__(
        self,
        anthropic_api_key: str,
        eris_base_url: str = "http://localhost:8001",
        *,
        model: str = "claude-sonnet-4-6",
        bridge_mode: str = "passive",
        n_steps: int = 60,
        analyses: List[str] = ("sae_features", "a_hat", "token_norms"),
        max_new_tokens: int = 512,
        system: Optional[str] = None,
        eris_timeout: float = 180.0,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeZombieBridge. "
                "Install with: pip install anthropic"
            ) from exc

        self._claude     = _anthropic.Anthropic(api_key=anthropic_api_key)
        self._eris       = ERISClient(eris_base_url, timeout=eris_timeout)
        self.model       = model
        self.bridge_mode = bridge_mode
        self.n_steps     = n_steps
        self.analyses    = list(analyses)
        self.max_new_tokens = max_new_tokens
        self.system      = system

        # Conversation history (Claude Messages API format)
        self._history: List[Dict[str, str]] = []

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    @property
    def history(self) -> List[Dict[str, str]]:
        """Read-only view of the conversation history."""
        return list(self._history)

    def chat(
        self,
        user_message: str,
        *,
        system: Optional[str] = None,
        bridge_mode: Optional[str] = None,
        n_steps: Optional[int] = None,
    ) -> BridgeTurnResult:
        """
        Send a user message, get Claude's response, bridge through the zombie.

        Args:
            user_message: The user's input text.
            system:       Per-turn system prompt override.
            bridge_mode:  Per-turn bridge mode override.
            n_steps:      Per-turn rollout steps override.

        Returns:
            :class:`BridgeTurnResult` with Claude + zombie outputs and analysis.

        Example::

            turn = bridge.chat("Explain distributed locking risks.")
            print(turn.claude_text)
            print("Implicit concepts:", turn.implicit_features)

            # Optionally feed zombie output back as context
            if turn.enriched_text:
                bridge._history.append({
                    "role": "assistant",
                    "content": turn.enriched_text,
                })
        """
        t0 = time.time()

        # ── Step 1: Ask Claude ────────────────────────────────────────────────
        self._history.append({"role": "user", "content": user_message})

        kwargs: Dict[str, Any] = {
            "model":      self.model,
            "max_tokens": 4096,
            "messages":   list(self._history),
            "timeout":    120.0,
        }
        sys_prompt = system or self.system
        if sys_prompt:
            kwargs["system"] = sys_prompt

        claude_response = self._claude.messages.create(**kwargs)
        claude_text: str = claude_response.content[0].text

        self._history.append({"role": "assistant", "content": claude_text})

        # ── Step 2: Bridge through zombie ────────────────────────────────────
        mode   = bridge_mode or self.bridge_mode
        steps  = n_steps    if n_steps is not None else self.n_steps

        bridge_result = self._eris.bridge(
            claude_text,
            mode=mode,
            n_steps=steps,
            analyses=self.analyses,
            decode_after=(mode != "analyze_only"),
            max_new_tokens=self.max_new_tokens,
        )

        elapsed = time.time() - t0

        return BridgeTurnResult(
            user_message=user_message,
            claude_text=claude_text,
            enriched_text=bridge_result.get("enriched_text"),
            analysis=bridge_result.get("analysis", {}),
            trajectory_summary=bridge_result.get("trajectory_summary"),
            pre_analysis=bridge_result.get("pre_analysis"),
            tokens=bridge_result.get("tokens", []),
            elapsed_s=round(elapsed, 3),
            handles=bridge_result.get("handles", {}),
        )

    def close(self) -> None:
        """Close the underlying ERIS HTTP client."""
        self._eris.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
