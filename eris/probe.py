"""
ERIS v5 — LatentProbe
======================

Pure representation tool. No dialogue. No generation. No web access.

The zombie model is called here as a **transformation function**:
    input_text → hidden activations (numpy)

This is the only interface the zombie exposes in the new ERIS paradigm.
Claude calls LatentProbe.probe() when DriftDetector signals that its own
latent state has diverged from the reference — it uses the zombie's
activations as an external reference frame to recalibrate, not as a
conversational partner.

Design constraints (hard):
  - max_new_tokens=0 is structural: forward pass only, no generate() call
  - enable_thinking=False on models that support it
  - No web access, no tools, no session state
  - Logging: layer index, output shape, timestamp only

Usage::

    probe = LatentProbe("Qwen/Qwen3-32B", layers=[9, 18, 27], device="cuda")
    activations = probe.probe("What is the halting problem?", pooling="last_token")
    # activations = {9: np.ndarray[2560], 18: np.ndarray[5120], 27: np.ndarray[5120]}

    batch = probe.probe_batch(["text A", "text B"], layers=[9, 18])
    # batch = [{9: arr, 18: arr}, {9: arr, 18: arr}]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("eris.probe")


# ── Exceptions ────────────────────────────────────────────────────────────────

class ProbeError(RuntimeError):
    """Raised when the probe cannot extract activations safely."""


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """
    Activations extracted from one forward pass.

    Attributes:
        activations:  {layer_idx: np.ndarray[hidden_dim]} — one vector per
                      requested layer, pooled according to the pooling method.
        input_tokens: Number of input tokens processed.
        elapsed_s:    Wall-clock time for the forward pass.
    """
    activations: dict[int, np.ndarray]
    input_tokens: int
    elapsed_s: float


# ── LatentProbe ───────────────────────────────────────────────────────────────

class LatentProbe:
    """
    Pure representation tool. Input → activations. No text output.

    The model is loaded once and reused across calls.  The forward pass
    uses model.__call__() directly — not model.generate() — so there is
    no code path that can produce text output.

    Args:
        model_id: HuggingFace model name or local path.
        layers:   List of layer indices to extract.  Negative indices are
                  resolved against the actual number of layers at load time.
                  Layer -1 = last hidden layer.
        device:   Torch device string.  Defaults to "cuda" if available,
                  falls back to "cpu".
        dtype:    Torch dtype for model weights.  Defaults to bfloat16 on
                  CUDA, float32 on CPU.
    """

    def __init__(
        self,
        model_id: str,
        layers: list[int],
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_id = model_id
        self._requested_layers = layers
        self.device = device if torch.cuda.is_available() else "cpu"

        if dtype is None:
            self.dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
        else:
            self.dtype = dtype

        log.info("Loading probe model: %s on %s (%s)", model_id, self.device, self.dtype)
        t0 = time.time()
        self._tokenizer, self._model, self.n_layers = self._load_model()
        self.layers = self._resolve_layers(self._requested_layers)
        log.info(
            "Probe ready: %s | %d layers | resolved=%s | %.1fs",
            model_id, self.n_layers, self.layers, time.time() - t0,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def probe(
        self,
        input_text: str,
        pooling: str = "last_token",
        centered: bool = True,
    ) -> dict[int, np.ndarray]:
        """
        Extract hidden-state activations for one input.

        Args:
            input_text: The text to encode.
            pooling:    "last_token" (default) or "mean".
            centered:   If True, subtract the per-layer mean across the
                        sequence before pooling.  Matches the M4 setup.

        Returns:
            {layer_idx: np.ndarray[hidden_dim]} — one array per requested layer.

        Raises:
            ProbeError: On tokenisation failure or empty output.
        """
        result = self._forward([input_text], pooling=pooling, centered=centered)
        return result[0].activations

    def probe_batch(
        self,
        inputs: list[str],
        layers: Optional[list[int]] = None,
        pooling: str = "last_token",
        centered: bool = True,
    ) -> list[dict[int, np.ndarray]]:
        """
        Extract hidden-state activations for a batch of inputs.

        Args:
            inputs:  List of input strings.
            layers:  Override layer list for this call.  If None, uses the
                     layers specified at construction.
            pooling: "last_token" or "mean".
            centered: Subtract per-layer sequence mean before pooling.

        Returns:
            List of {layer_idx: np.ndarray} — one dict per input.
        """
        if layers is not None:
            prev = self.layers
            self.layers = self._resolve_layers(layers)

        results = self._forward(inputs, pooling=pooling, centered=centered)

        if layers is not None:
            self.layers = prev

        return [r.activations for r in results]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self):
        """Load tokenizer and model.  Returns (tokenizer, model, n_layers)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        model.eval()

        # Disable thinking on models that support it (Qwen3.x).
        # This is belt-and-suspenders: we never call generate(), but if the
        # model has a generation_config with thinking enabled it can affect
        # the forward pass internals on some model versions.
        if hasattr(model, "generation_config"):
            gc = model.generation_config
            if hasattr(gc, "enable_thinking"):
                gc.enable_thinking = False
            if hasattr(gc, "thinking_budget"):
                gc.thinking_budget = 0

        # Count transformer layers robustly across architectures.
        n_layers = _count_layers(model)

        return tokenizer, model, n_layers

    def _resolve_layers(self, layers: list[int]) -> list[int]:
        """Resolve negative layer indices and validate range."""
        resolved = []
        for l in layers:
            r = l if l >= 0 else self.n_layers + l
            if not (0 <= r < self.n_layers):
                raise ProbeError(
                    f"Layer {l} (resolved={r}) out of range for model "
                    f"with {self.n_layers} layers."
                )
            resolved.append(r)
        return sorted(set(resolved))

    @torch.no_grad()
    def _forward(
        self,
        texts: list[str],
        pooling: str,
        centered: bool,
    ) -> list[ProbeResult]:
        """
        Run a forward pass and extract hidden states.

        This uses model.__call__() with output_hidden_states=True.
        There is no generate() call and no token sampling path.
        """
        if not texts:
            raise ProbeError("Empty input list.")

        t0 = time.time()

        # Tokenise (pad to same length for batching).
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        if input_ids.shape[1] == 0:
            raise ProbeError("Tokenisation produced empty sequence.")

        # Forward pass — hidden states only, no generation.
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states: tuple of (n_layers+1) tensors, each [batch, seq, dim]
        # Index 0 is the embedding layer; index k is layer k-1's output.
        hs = outputs.hidden_states  # tuple[(batch, seq, dim), ...]
        if hs is None:
            raise ProbeError("Model did not return hidden_states.")

        elapsed = time.time() - t0
        n_tokens = input_ids.shape[1]

        results: list[ProbeResult] = []
        for batch_idx in range(len(texts)):
            seq_mask = attention_mask[batch_idx].bool()  # [seq]
            activations: dict[int, np.ndarray] = {}

            for layer_idx in self.layers:
                # hs has n_layers+1 entries (embedding + each layer).
                # Layer k's output is at hs[k+1].
                h = hs[layer_idx + 1][batch_idx]   # [seq, hidden_dim]
                h = h[seq_mask]                     # strip padding

                if centered:
                    h = h - h.mean(dim=0, keepdim=True)

                if pooling == "last_token":
                    vec = h[-1]
                elif pooling == "mean":
                    vec = h.mean(dim=0)
                else:
                    raise ProbeError(f"Unknown pooling: '{pooling}'. Use 'last_token' or 'mean'.")

                arr = vec.float().cpu().numpy()
                activations[layer_idx] = arr
                log.debug(
                    "layer=%d shape=%s norm=%.3f",
                    layer_idx, arr.shape, float(np.linalg.norm(arr)),
                )

            results.append(ProbeResult(
                activations=activations,
                input_tokens=int(seq_mask.sum().item()),
                elapsed_s=round(elapsed / len(texts), 4),
            ))

        log.info(
            "probe: %d inputs, layers=%s, pooling=%s, %.3fs",
            len(texts), self.layers, pooling, elapsed,
        )
        return results

    def __repr__(self) -> str:
        return (
            f"LatentProbe(model={self.model_id!r}, "
            f"layers={self.layers}, device={self.device!r})"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_layers(model) -> int:
    """
    Count transformer blocks robustly across Qwen3/Qwen3.5/DeepSeek architectures.
    """
    # Standard: model.model.layers (Qwen3, Llama, Mistral, DeepSeek)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    # GPT-style: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    # Fallback: count from config
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            v = getattr(cfg, attr, None)
            if v is not None:
                return int(v)
    raise ProbeError(
        "Cannot determine layer count for this model. "
        "Check model.model.layers or model.config.num_hidden_layers."
    )
