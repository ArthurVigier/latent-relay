"""
Latent Relay Engine
====================
Stateful wrapper around LatentMAS model operations.
Manages sessions, KV-cache storage, W_a computation, and latent rollout.

This is the core engine — no HTTP/MCP dependencies. Can be used by:
  - The FastAPI server (server.py)
  - The MCP tool layer (mcp_server.py)
  - The OpenAI-compatible proxy (openclaw_compat/openai_proxy.py)
  - Direct Python usage
"""

import base64
import time
import uuid
import numpy as np
import torch
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


@dataclass
class LatentThought:
    """A stored latent reasoning result."""
    handle: str
    session_id: str
    n_positions: int
    role: str
    created_at: float
    kv_cache: Optional[Tuple] = field(repr=False, default=None)
    hidden_embedding: Optional[torch.Tensor] = field(repr=False, default=None)
    # Per-layer hidden states stored on CPU (populated by encode()).
    # Keys: "layer_N" or "last".  Values: float32 tensors [seq_len, hidden_dim].
    layer_hidden_states: Optional[Dict[str, torch.Tensor]] = field(repr=False, default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A model session with precomputed W_a."""
    session_id: str
    model_name: str
    device: str
    created_at: float
    thoughts: Dict[str, LatentThought] = field(default_factory=dict)


class LatentRelayEngine:
    """
    Core engine for latent-space multi-agent collaboration.

    Provides three operations:
      1. create_session() — load model, compute W_a
      2. think() — run latent rollout, store KV-cache, return handle
      3. collaborate() — combine latent thoughts, generate text answer
    """

    def __init__(self, model_name: str, device: str = "cuda:0",
                 trust_remote_code: bool = True):
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code

        self._lock = threading.Lock()
        self._sessions: Dict[str, Session] = {}

        # Load model and tokenizer once
        print(f"[Engine] Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=trust_remote_code,
        ).to(device).eval()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        # Detect hybrid architecture (Qwen3.5 GDN + full attention)
        _text_cfg = getattr(self.model.config, 'text_config', self.model.config)
        _layer_types = getattr(_text_cfg, 'layer_types', None)
        if _layer_types:
            _n_full = sum(1 for t in _layer_types if t == 'full_attention')
            print(f"[Engine] Hybrid architecture: {len(_layer_types)} layers "
                  f"({_n_full} full attention, {len(_layer_types) - _n_full} linear)")
        _tied = getattr(_text_cfg, 'tie_word_embeddings', False)
        if _tied:
            print("[Engine] Tied embeddings detected — empirical W_a will be used")

        # Precompute W_a (auto-detects degenerate static solution → empirical fallback)
        self._wa_matrix, self._target_norm = self._compute_wa()
        print(f"[Engine] W_a ready: {list(self._wa_matrix.shape)}")
        print(f"[Engine] Ready.")

    def _compute_wa(self, lambda_reg: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute alignment matrix W_a = (W_out^T W_out + lambda*I)^{-1} W_out^T W_in.

        Falls back to empirical W_a when the static solution is near-identity
        (happens when input/output embeddings are numerically identical even
        without tie_word_embeddings=True, as observed on Qwen3 and Qwen3.5).
        """
        input_emb  = self.model.get_input_embeddings().weight.detach().float().to(self.device)
        output_emb = self.model.get_output_embeddings().weight.detach().float().to(self.device)

        gram = output_emb.T @ output_emb
        gram += lambda_reg * torch.eye(gram.shape[0], device=self.device, dtype=gram.dtype)
        rhs  = output_emb.T @ input_emb
        wa   = torch.linalg.solve(gram, rhs)

        # Detect degenerate W_a — near-identity means the static projection
        # carries no information.  Switch to an empirical estimate learned from
        # actual (h_t, e_{t+1}) pairs so the realignment is genuinely useful.
        I    = torch.eye(wa.shape[0], device=self.device, dtype=wa.dtype)
        diff = (wa - I).norm().item()
        if diff < 0.5:
            print(f"[Engine] W_a static ≈ I (‖W_a−I‖={diff:.4f}) — computing empirical W_a")
            return self._compute_wa_empirical(lambda_reg=lambda_reg)

        target_norm = input_emb.norm(dim=1).mean()
        return wa, target_norm

    def _compute_wa_empirical(
        self, n_samples: int = 256, lambda_reg: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Learn W_a from real forward passes: W_a ≈ argmin ‖H W − E‖²

        H[i] = last hidden state at token position t
        E[i] = input embedding of token at position t+1

        This captures the true hidden→embedding projection even when the
        static embedding matrices are numerically identical (Qwen3, Qwen3.5).
        """
        import random

        emb        = self.model.get_input_embeddings().weight.detach().float().to(self.device)
        vocab_size = emb.shape[0]
        seq_len    = 9                              # 9 tokens → 8 (h_t, e_{t+1}) pairs
        n_seqs     = max(1, n_samples // (seq_len - 1))

        print(f"[Engine] Sampling {n_seqs} sequences for empirical W_a...")

        H_list, E_list = [], []
        for _ in range(n_seqs):
            token_ids = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
            ids       = torch.tensor([token_ids], device=self.device)

            with torch.no_grad():
                out = self.model(ids, output_hidden_states=True, use_cache=False)

            hs = out.hidden_states[-1][0].float()   # [seq_len, hidden_dim]
            for t in range(seq_len - 1):
                H_list.append(hs[t])
                E_list.append(emb[token_ids[t + 1]])

        H = torch.stack(H_list)  # [N, hidden_dim]
        E = torch.stack(E_list)  # [N, hidden_dim]

        gram = H.T @ H + lambda_reg * torch.eye(H.shape[1], device=self.device, dtype=H.dtype)
        rhs  = H.T @ E
        wa   = torch.linalg.solve(gram, rhs)

        I    = torch.eye(wa.shape[0], device=self.device, dtype=wa.dtype)
        diff = (wa - I).norm().item()
        print(f"[Engine] Empirical W_a ready: shape={list(wa.shape)}, ‖W_a−I‖={diff:.4f}")

        target_norm = E.norm(dim=1).mean()
        return wa, target_norm

    def _apply_realignment(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply W_a realignment: project hidden state to input embedding space."""
        h = hidden.float()
        aligned = h @ self._wa_matrix
        norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (self._target_norm / norm)
        return aligned.to(hidden.dtype)

    @staticmethod
    def _past_length(past_kv) -> int:
        """Get sequence length from past_key_values.

        Handles DynamicCache, Qwen3_5DynamicCache (hybrid GDN + full attention
        where some key_cache entries are None), and legacy tuple format.
        """
        if past_kv is None:
            return 0
        # Prefer get_seq_length() — works for all Cache subclasses including
        # hybrid models (Qwen3.5) where key_cache entries may be None.
        if hasattr(past_kv, 'get_seq_length'):
            return past_kv.get_seq_length()
        # DynamicCache without get_seq_length: find first non-None key tensor.
        if hasattr(past_kv, 'key_cache'):
            for k in past_kv.key_cache:
                if k is not None:
                    return k.shape[-2]
            return 0
        # Legacy tuple format: ((k, v), (k, v), ...)
        if isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
            return past_kv[0][0].shape[-2]
        return 0

    # ──────────────────────────────────────────────
    # Hidden-state helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _resolve_layer_key(layer_idx: int, n_layers: int) -> Tuple[int, str]:
        """
        Convert a layer index (possibly -1) to an absolute index and a string key.

        ``outputs.hidden_states`` has shape (n_layers + 1,) — index 0 is the
        embedding layer, indices 1..n_layers are transformer layers.
        -1 maps to the final transformer layer.
        """
        total = n_layers + 1  # embedding layer + transformer layers
        if layer_idx == -1:
            return total - 1, "last"
        if not (0 <= layer_idx < total):
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {total - 1}] "
                f"(model has {n_layers} transformer layers)"
            )
        return layer_idx, f"layer_{layer_idx}"

    @staticmethod
    def _tensor_to_payload(t: torch.Tensor, compact: bool = True) -> Union[str, list]:
        """
        Serialise a hidden-state tensor for JSON transport.

        Args:
            t: Tensor of shape [seq_len, hidden_dim], any dtype.
            compact: If True (default), return a base64-encoded string of the
                     raw float32 bytes.  If False, return a nested Python list
                     of floats (human-readable but ~4× larger).

        Returns:
            base64 string or nested list of floats.
        """
        arr = t.detach().cpu().float().numpy()  # → float32 numpy array
        if compact:
            return base64.b64encode(arr.tobytes()).decode("ascii")
        return arr.tolist()

    @staticmethod
    def _payload_to_tensor(payload: Union[str, list], shape: Tuple[int, int]) -> torch.Tensor:
        """
        Deserialise a hidden-state payload back to a float32 tensor.

        Args:
            payload: base64 string or nested list (as returned by _tensor_to_payload).
            shape: (seq_len, hidden_dim) — required when payload is a base64 string.
        """
        if isinstance(payload, str):
            arr = np.frombuffer(base64.b64decode(payload), dtype=np.float32).reshape(shape)
            return torch.from_numpy(arr.copy())
        return torch.tensor(payload, dtype=torch.float32)

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def create_session(self) -> str:
        """Create a new session. Returns session_id."""
        sid = str(uuid.uuid4())[:12]
        session = Session(
            session_id=sid,
            model_name=self.model_name,
            device=self.device,
            created_at=time.time(),
        )
        with self._lock:
            self._sessions[sid] = session
        return sid

    def list_sessions(self) -> List[Dict]:
        """List active sessions."""
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "model": s.model_name,
                    "n_thoughts": len(s.thoughts),
                    "created_at": s.created_at,
                }
                for s in self._sessions.values()
            ]

    @torch.no_grad()
    def encode(
        self,
        text: str,
        *,
        return_layers: Optional[List[int]] = None,
        return_attention: bool = False,
        session_id: Optional[str] = None,
        compact: bool = True,
    ) -> Dict:
        """
        Encode text and expose hidden states for the requested layers.

        Performs a single forward pass with ``output_hidden_states=True`` and
        returns the hidden states for the requested transformer layers.  The
        result is optionally stored as a :class:`LatentThought` if a
        ``session_id`` is provided.

        Args:
            text: Input text to encode.
            return_layers: Layer indices to return.  Use -1 for the last layer.
                           Defaults to ``[-1]`` (last layer only).
            return_attention: If True, also return per-layer attention weights
                              (``output_attentions=True``).  Increases latency.
            session_id: If given, store the result in this session and return a
                        handle.  The session must already exist.
            compact: If True (default), hidden states are base64-encoded float32
                     bytes.  If False, they are returned as nested float lists
                     (human-readable but ~4× larger).

        Returns:
            Dict with keys:
              - ``handle`` (str | None): storage handle, None if no session_id
              - ``hidden_states`` (dict): layer_key → base64 str or list
              - ``tokens`` (list[str]): token strings from the tokenizer
              - ``seq_len`` (int)
              - ``hidden_dim`` (int)

        Example curl::

            curl -X POST http://localhost:8001/v1/encode \\
              -H 'Content-Type: application/json' \\
              -d '{"text": "Hello world", "return_layers": [15, -1]}'
        """
        if return_layers is None:
            return_layers = [-1]

        # Tokenise
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        seq_len: int = input_ids.shape[-1]

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=return_attention,
            return_dict=True,
        )

        # ``outputs.hidden_states`` is a tuple of length n_layers + 1.
        # Index 0 is the embedding layer; indices 1..n_layers are transformer layers.
        all_hs = outputs.hidden_states  # tuple of [1, seq_len, hidden_dim]
        n_transformer_layers: int = len(all_hs) - 1
        hidden_dim: int = all_hs[-1].shape[-1]

        # Extract requested layers — deduplicate while preserving order
        seen: set = set()
        unique_layers = [l for l in return_layers if not (l in seen or seen.add(l))]

        layer_hidden_states: Dict[str, torch.Tensor] = {}
        hidden_states_payload: Dict[str, Union[str, list]] = {}

        for layer_idx in unique_layers:
            abs_idx, key = self._resolve_layer_key(layer_idx, n_transformer_layers)
            hs_cpu = all_hs[abs_idx][0].detach().cpu().float()  # [seq_len, hidden_dim]
            layer_hidden_states[key] = hs_cpu
            hidden_states_payload[key] = self._tensor_to_payload(hs_cpu, compact=compact)

        # Token strings (skip special if desired — expose raw for transparency)
        tokens: List[str] = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].tolist()
        )

        # Optionally store in session
        handle: Optional[str] = None
        if session_id is not None:
            with self._lock:
                session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

            handle = f"enc_{session_id}_{uuid.uuid4().hex[:8]}"
            thought = LatentThought(
                handle=handle,
                session_id=session_id,
                n_positions=self._past_length(outputs.past_key_values),
                role="encode",
                created_at=time.time(),
                kv_cache=outputs.past_key_values,
                hidden_embedding=all_hs[-1][0, -1, :].detach(),
                layer_hidden_states=layer_hidden_states,
                metadata={
                    "source": "encode",
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                    "layers_stored": list(layer_hidden_states.keys()),
                },
            )
            with self._lock:
                session.thoughts[handle] = thought

        return {
            "handle": handle,
            "hidden_states": hidden_states_payload,
            "tokens": tokens,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
        }

    @torch.no_grad()
    def think(
        self,
        session_id: str,
        prompt: str,
        *,
        n_steps: int = 60,
        role: str = "general",
        inherit_from: Optional[List[str]] = None,
        # ── ERIS extensions ──────────────────────────────────────────────────
        return_trajectory: bool = False,
        perturbation: Optional[torch.Tensor] = None,
        a_hat_fn: Optional[Callable[[torch.Tensor], float]] = None,
    ) -> Dict:
        """
        Run latent reasoning on a prompt.

        Args:
            session_id: Active session ID.
            prompt: Text prompt for the agent.
            n_steps: Number of latent rollout steps (0 = just encode).
            role: Agent role label (planner, critic, refiner, etc.).
            inherit_from: List of thought handles to inherit KV-cache from.
            return_trajectory: If True, include per-step metrics in the response
                               (``trajectory`` list + ``total_displacement``).
            perturbation: Optional float32 tensor of shape ``[hidden_dim]`` or
                          ``[1, hidden_dim]``.  Added to the initial hidden state
                          before the first rollout step (one-shot steering).
            a_hat_fn: Optional callable ``(hidden: Tensor) -> float`` that returns
                      the Â-hat agentivity score for a hidden state.  When provided,
                      the score is recorded at each trajectory step.  Pass the
                      AHatAnalyzer's score method from eris/analyzers.py.

        Returns:
            Dict with handle, metadata, timing info.  When ``return_trajectory``
            is True, also includes ``trajectory`` (list of per-step dicts) and
            ``total_displacement`` (float).

        Example curl::

            curl -X POST http://localhost:8001/think \\
              -H 'Content-Type: application/json' \\
              -d '{"session_id": "...", "prompt": "hello", "n_steps": 10,
                   "return_trajectory": true}'
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

        t0 = time.time()

        # Build inherited KV-cache
        past_kv = None
        if inherit_from:
            for inh_handle in inherit_from:
                thought = session.thoughts.get(inh_handle)
                if thought and thought.kv_cache is not None:
                    past_kv = thought.kv_cache

        # Encode prompt
        encoded = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Extend attention mask for inherited past
        if past_kv is not None:
            past_len = self._past_length(past_kv)
            if past_len > 0:
                past_mask = torch.ones(
                    (1, past_len), dtype=attention_mask.dtype, device=self.device
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # Forward pass: encode the prompt
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]

        # Apply one-shot perturbation before the rollout starts
        if perturbation is not None:
            pert = perturbation.to(self.device, dtype=last_hidden.dtype)
            last_hidden = last_hidden + pert.reshape(1, -1)

        # Lazy-import TrajectoryTracker to avoid circular deps at module load
        tracker = None
        if return_trajectory:
            from eris.trajectory import TrajectoryTracker
            tracker = TrajectoryTracker(z0=last_hidden)

        # Latent rollout: h -> W_a -> forward -> h'
        for step in range(n_steps):
            latent_vec = self._apply_realignment(last_hidden)
            latent_embed = latent_vec.unsqueeze(1)  # [1, 1, d_h]

            past_len = self._past_length(past)
            latent_mask = torch.ones(
                (1, past_len + 1), dtype=torch.long, device=self.device
            )

            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            if tracker is not None:
                a_hat_score = a_hat_fn(last_hidden) if a_hat_fn is not None else None
                tracker.record(step=step, hidden=last_hidden, a_hat_score=a_hat_score)

        elapsed = time.time() - t0

        # Store the thought
        handle = f"t_{session_id}_{uuid.uuid4().hex[:8]}"
        thought = LatentThought(
            handle=handle,
            session_id=session_id,
            n_positions=self._past_length(past),
            role=role,
            created_at=time.time(),
            kv_cache=past,
            hidden_embedding=last_hidden.detach(),
            metadata={
                "n_steps": n_steps,
                "prompt_tokens": input_ids.shape[-1],
                "hidden_norm": last_hidden.norm().item(),
            },
        )

        with self._lock:
            session.thoughts[handle] = thought

        result: Dict = {
            "handle": handle,
            "session_id": session_id,
            "role": role,
            "n_steps": n_steps,
            "n_positions": thought.n_positions,
            "elapsed_s": round(elapsed, 3),
            "hidden_norm": round(last_hidden.norm().item(), 2),
        }

        if tracker is not None:
            result["trajectory"] = tracker.to_list()
            result["total_displacement"] = round(tracker.total_displacement, 4)

        return result

    @torch.no_grad()
    def collaborate(
        self,
        session_id: str,
        handles: List[str],
        final_prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        use_latent_context: bool = True,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
    ) -> Dict:
        """
        Combine latent thoughts and generate a text answer.

        Passes the accumulated KV-cache from the last handle into generate()
        so the latent reasoning context actually influences generation.
        Set use_latent_context=False to fall back to prompt-only generation.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

        t0 = time.time()

        # Find the accumulated KV-cache from the most recent handle
        past_kv = None
        if use_latent_context:
            for handle in reversed(handles):
                thought = session.thoughts.get(handle)
                if thought and thought.kv_cache is not None:
                    past_kv = thought.kv_cache
                    break

        encoded = self.tokenizer(
            final_prompt, return_tensors="pt", add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(self.device)

        generate_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Disable thinking mode on models that support it (Qwen3.x).
        # Avoids <think> meta-discourse in enriched output.
        if enable_thinking is not None:
            generate_kwargs["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            generate_kwargs["thinking_budget"] = thinking_budget

        if past_kv is not None:
            past_len = self._past_length(past_kv)
            attention_mask = torch.ones(
                (1, past_len + input_ids.shape[1]),
                dtype=torch.long,
                device=self.device,
            )
            generate_kwargs["past_key_values"] = past_kv
            generate_kwargs["attention_mask"] = attention_mask

        gen_outputs = self.model.generate(input_ids=input_ids, **generate_kwargs)

        # Decode only the new tokens
        prompt_len = input_ids.shape[-1]
        new_tokens = gen_outputs[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        elapsed = time.time() - t0

        return {
            "text": text,
            "tokens_generated": len(new_tokens),
            "handles_used": handles,
            "elapsed_s": round(elapsed, 3),
            "latent_context_used": past_kv is not None,
        }

    def get_thought_info(self, session_id: str, handle: str) -> Optional[Dict]:
        """Get metadata about a stored thought."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            thought = session.thoughts.get(handle)
            if thought is None:
                return None
        return {
            "handle": thought.handle,
            "session_id": thought.session_id,
            "role": thought.role,
            "n_positions": thought.n_positions,
            "created_at": thought.created_at,
            "metadata": thought.metadata,
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and free its GPU memory."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        for thought in session.thoughts.values():
            thought.kv_cache = None
            thought.hidden_embedding = None
        torch.cuda.empty_cache()
        return True