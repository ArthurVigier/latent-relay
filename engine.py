"""
Latent Relay Engine
====================
Stateful wrapper around LatentMAS model operations.
Manages sessions, KV-cache storage, W_a computation, and latent rollout.

This is the core engine — no HTTP/MCP dependencies. Can be used by:
  - The FastAPI server (server.py)
  - The MCP tool layer (mcp_server.py)
  - Direct Python usage
"""

import time
import uuid
import torch
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
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

        # Precompute W_a
        self._wa_matrix, self._target_norm = self._compute_wa()
        print(f"[Engine] W_a computed: {self._wa_matrix.shape}")
        print(f"[Engine] Ready.")

    def _compute_wa(self, lambda_reg: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute alignment matrix W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in"""
        input_emb = self.model.get_input_embeddings().weight.detach().float().to(self.device)
        output_emb = self.model.get_output_embeddings().weight.detach().float().to(self.device)

        gram = output_emb.T @ output_emb
        gram += lambda_reg * torch.eye(gram.shape[0], device=self.device, dtype=gram.dtype)
        rhs = output_emb.T @ input_emb
        wa = torch.linalg.solve(gram, rhs)
        target_norm = input_emb.norm(dim=1).mean()
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
        if not past_kv:
            return 0
        return past_kv[0][0].shape[-2]

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
    def think(
        self,
        session_id: str,
        prompt: str,
        *,
        n_steps: int = 60,
        role: str = "general",
        inherit_from: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run latent reasoning on a prompt.

        Args:
            session_id: Active session ID
            prompt: Text prompt for the agent
            n_steps: Number of latent rollout steps (0 = just encode, no latent thinking)
            role: Agent role label (planner, critic, refiner, etc.)
            inherit_from: List of thought handles to inherit KV-cache from

        Returns:
            Dict with handle, metadata, timing info
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

        t0 = time.time()

        # Build inherited KV-cache
        past_kv = None
        if inherit_from:
            for handle in inherit_from:
                thought = session.thoughts.get(handle)
                if thought and thought.kv_cache is not None:
                    past_kv = thought.kv_cache
                    # For chaining: use the last inherited thought's cache
                    # (sequential agent pattern)

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
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        # Latent rollout: h → W_a → forward → h'
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

        return {
            "handle": handle,
            "session_id": session_id,
            "role": role,
            "n_steps": n_steps,
            "n_positions": thought.n_positions,
            "elapsed_s": round(elapsed, 3),
            "hidden_norm": round(last_hidden.norm().item(), 2),
        }

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
    ) -> Dict:
        """
        Combine latent thoughts and generate a text answer.

        Uses the last handle's KV-cache as context prefix, then generates text.

        Args:
            session_id: Active session ID
            handles: Ordered list of thought handles to use as context
            final_prompt: Text prompt for the final generation
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Dict with generated text and metadata
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

        t0 = time.time()

        # Get the accumulated KV-cache from the last thought in the chain
        past_kv = None
        for h in handles:
            thought = session.thoughts.get(h)
            if thought and thought.kv_cache is not None:
                past_kv = thought.kv_cache

        # Encode final prompt
        encoded = self.tokenizer(
            final_prompt, return_tensors="pt", add_special_tokens=True
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        if past_kv is not None:
            past_len = self._past_length(past_kv)
            if past_len > 0:
                past_mask = torch.ones(
                    (1, past_len), dtype=attention_mask.dtype, device=self.device
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)


        # Generate text — let HF handle cache positions via attention_mask
        gen_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.2,
        )

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
        # Free KV-caches
        for thought in session.thoughts.values():
            thought.kv_cache = None
            thought.hidden_embedding = None
        torch.cuda.empty_cache()
        return True
