"""
eris/backends/probes/hf_probe.py
=================================

HFProbe — ProbeModel implementation backed by HuggingFace Transformers.

This is the canonical implementation of the ProbeModel interface for
locally-loaded HuggingFace models (Qwen3, DeepSeek-R1, Llama, etc.).

LatentProbe (eris/probe.py) delegates to this class so that:
  - The forward-pass extraction logic lives in one place.
  - steer() and the steering library are available to all callers.
  - Other ProbeModel backends (VLLMProbe) can be swapped in without
    changing the orchestrator.

Design constraints (hard):
  - max_new_tokens=0 is structural: we call model() not model.generate()
  - enable_thinking=False on models that support it
  - Steering is applied via forward hooks on the post-attention layernorm
    of each requested layer — no modification to model weights
  - Steering library is in-memory only (dict[str, np.ndarray])
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from eris.interfaces import ProbeModel

log = logging.getLogger("eris.hf_probe")

_VALID_MODES   = {"add", "project_out", "replace"}
_MANIFEST_FILE = "manifest.json"


def _sanitize_filename(name: str) -> str:
    """Convert an arbitrary direction name into a safe filename stem."""
    safe = re.sub(r"[^\w\-]", "_", name).strip("_")
    return safe or "direction"


class HFProbe(ProbeModel):
    """
    ProbeModel backed by a locally-loaded HuggingFace causal LM.

    Args:
        model_id:  HuggingFace model name or local path.
        layers:    Layer indices to extract by default.  Negative indices
                   are resolved at load time.
        device:    Torch device string.  Falls back to "cpu" if CUDA unavailable.
        dtype:     Torch dtype.  Defaults to bfloat16 on CUDA, float32 on CPU.
    """

    def __init__(
        self,
        model_id: str,
        layers: list[int],
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        library_dir: str = "steering_library",
    ) -> None:
        self.model_id = model_id
        self._requested_layers = layers
        self.device = device if torch.cuda.is_available() else "cpu"
        self.library_dir = Path(library_dir)

        if dtype is None:
            self.dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
        else:
            self.dtype = dtype

        log.info("Loading HFProbe: %s on %s (%s)", model_id, self.device, self.dtype)
        t0 = time.time()
        self._tokenizer, self._model, self.n_layers = self._load_model()
        self.layers = self._resolve_layers(self._requested_layers)
        self._steering_library: dict[str, np.ndarray] = {}
        self._load_library_from_disk()
        log.info(
            "HFProbe ready: %s | %d layers | resolved=%s | %.1fs | directions=%d",
            model_id, self.n_layers, self.layers, time.time() - t0,
            len(self._steering_library),
        )

    # ── ProbeModel interface ────────────────────────────────────────────────────

    def probe(
        self,
        input_text: str,
        layers: list[int],
        pooling: str = "last_token",
        centered: bool = True,
    ) -> dict[int, np.ndarray]:
        """Extract hidden-state activations for one input."""
        resolved = self._resolve_layers(layers)
        results = self._forward([input_text], resolved, pooling=pooling, centered=centered)
        return results[0]

    def probe_batch(
        self,
        inputs: list[str],
        layers: list[int],
        pooling: str = "last_token",
        centered: bool = True,
    ) -> list[dict[int, np.ndarray]]:
        """Extract activations for a batch of inputs."""
        resolved = self._resolve_layers(layers)
        return self._forward(inputs, resolved, pooling=pooling, centered=centered)

    def steer(
        self,
        input_text: str,
        direction: np.ndarray,
        alpha: float,
        layers: list[int],
        mode: str = "add",
    ) -> dict[int, np.ndarray]:
        """
        Apply a steering vector and return steered activations.

        No text is generated.  The steering is applied via a forward hook
        on the post-attention layernorm of each requested layer.

        Modes:
            "add"         — activation += alpha * direction
            "project_out" — remove component along direction
            "replace"     — replace component with alpha * direction
        """
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown steering mode: {mode!r}. Use one of {_VALID_MODES}.")
        resolved = self._resolve_layers(layers)
        results = self._forward(
            [input_text], resolved,
            pooling="last_token", centered=False,
            steer_direction=direction, steer_alpha=alpha, steer_mode=mode,
        )
        return results[0]

    def steer_batch(
        self,
        inputs: list[str],
        direction: np.ndarray,
        alpha: float,
        layers: list[int],
        mode: str = "add",
    ) -> list[dict[int, np.ndarray]]:
        """Apply steering to a batch of inputs."""
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown steering mode: {mode!r}. Use one of {_VALID_MODES}.")
        resolved = self._resolve_layers(layers)
        return self._forward(
            inputs, resolved,
            pooling="last_token", centered=False,
            steer_direction=direction, steer_alpha=alpha, steer_mode=mode,
        )

    def save_direction(self, name: str, vector: np.ndarray) -> None:
        """
        Save a named steering direction to memory and to disk.

        Persists to ``{library_dir}/{sanitized_name}.npy`` and updates
        ``{library_dir}/manifest.json``.  Survives process restarts.
        """
        arr = np.array(vector, dtype=np.float32)
        self._steering_library[name] = arr
        self._save_direction_to_disk(name, arr)
        log.info("Saved steering direction: %r (dim=%d) → %s", name, len(arr), self.library_dir)

    def load_direction(self, name: str) -> np.ndarray:
        """
        Load a named steering direction from memory.

        All persisted directions are loaded automatically at startup, so
        this always reflects the on-disk state.

        Raises:
            KeyError: If the direction is not in the library.
        """
        if name not in self._steering_library:
            raise KeyError(
                f"Steering direction {name!r} not found. "
                f"Available: {list(self._steering_library.keys())}"
            )
        return self._steering_library[name].copy()

    def delete_direction(self, name: str) -> None:
        """
        Remove a steering direction from memory and from disk.

        Raises:
            KeyError: If the direction is not in the library.
        """
        if name not in self._steering_library:
            raise KeyError(f"Steering direction {name!r} not found.")
        del self._steering_library[name]
        self._delete_direction_from_disk(name)
        log.info("Deleted steering direction: %r", name)

    def list_directions(self) -> list[str]:
        """Return the names of all saved steering directions."""
        return list(self._steering_library.keys())

    # ── Persistence helpers ──────────────────────────────────────────────────

    def _manifest_path(self) -> Path:
        return self.library_dir / _MANIFEST_FILE

    def _read_manifest(self) -> dict[str, str]:
        """Return the manifest dict {name: filename} or {} if missing."""
        p = self._manifest_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Could not read manifest %s: %s", p, e)
            return {}

    def _write_manifest(self, manifest: dict[str, str]) -> None:
        self._manifest_path().write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_library_from_disk(self) -> None:
        """Load all persisted directions into memory at startup."""
        if not self.library_dir.exists():
            return
        manifest = self._read_manifest()
        loaded = 0
        for name, filename in manifest.items():
            npy_path = self.library_dir / filename
            if not npy_path.exists():
                log.warning("Direction %r listed in manifest but file missing: %s", name, npy_path)
                continue
            try:
                self._steering_library[name] = np.load(str(npy_path)).astype(np.float32)
                loaded += 1
            except Exception as e:
                log.warning("Failed to load direction %r from %s: %s", name, npy_path, e)
        if loaded:
            log.info("Loaded %d steering direction(s) from %s", loaded, self.library_dir)

    def _save_direction_to_disk(self, name: str, arr: np.ndarray) -> None:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._read_manifest()

        # Reuse existing filename if name already persisted, else create new.
        if name in manifest:
            filename = manifest[name]
        else:
            stem = _sanitize_filename(name)
            # Avoid collisions: append suffix if stem already used by another name.
            existing_files = set(manifest.values())
            filename = f"{stem}.npy"
            counter = 1
            while filename in existing_files:
                filename = f"{stem}_{counter}.npy"
                counter += 1

        np.save(str(self.library_dir / filename), arr)
        manifest[name] = filename
        self._write_manifest(manifest)

    def _delete_direction_from_disk(self, name: str) -> None:
        manifest = self._read_manifest()
        filename = manifest.pop(name, None)
        if filename:
            npy_path = self.library_dir / filename
            if npy_path.exists():
                npy_path.unlink()
        self._write_manifest(manifest)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _load_model(self):
        """Load tokenizer and model. Returns (tokenizer, model, n_layers)."""
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
        if hasattr(model, "generation_config"):
            gc = model.generation_config
            if hasattr(gc, "enable_thinking"):
                gc.enable_thinking = False
            if hasattr(gc, "thinking_budget"):
                gc.thinking_budget = 0

        n_layers = _count_layers(model)
        return tokenizer, model, n_layers

    def _resolve_layers(self, layers: list[int]) -> list[int]:
        """Resolve negative indices and validate range."""
        resolved = []
        for l in layers:
            r = l if l >= 0 else self.n_layers + l
            if not (0 <= r < self.n_layers):
                raise ValueError(
                    f"Layer {l} (resolved={r}) out of range for model "
                    f"with {self.n_layers} layers."
                )
            resolved.append(r)
        return sorted(set(resolved))

    @torch.no_grad()
    def _forward(
        self,
        texts: list[str],
        resolved_layers: list[int],
        pooling: str,
        centered: bool,
        steer_direction: Optional[np.ndarray] = None,
        steer_alpha: float = 0.0,
        steer_mode: str = "add",
    ) -> list[dict[int, np.ndarray]]:
        """
        Forward pass with optional steering hooks.

        Returns a list of {layer_idx: np.ndarray} — one per input.
        """
        if not texts:
            raise ValueError("Empty input list.")

        t0 = time.time()
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
            raise ValueError("Tokenisation produced empty sequence.")

        # Register steering hooks if requested.
        handles = []
        if steer_direction is not None:
            direction_t = torch.from_numpy(
                steer_direction.astype(np.float32)
            ).to(self.device)
            for layer_idx in resolved_layers:
                hook = _make_steering_hook(direction_t, steer_alpha, steer_mode)
                layer_module = _get_layer(self._model, layer_idx)
                handles.append(layer_module.register_forward_hook(hook))

        try:
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        finally:
            for h in handles:
                h.remove()

        hs = outputs.hidden_states
        if hs is None:
            raise RuntimeError("Model did not return hidden_states.")

        elapsed = time.time() - t0
        results: list[dict[int, np.ndarray]] = []

        for batch_idx in range(len(texts)):
            seq_mask = attention_mask[batch_idx].bool()
            activations: dict[int, np.ndarray] = {}

            for layer_idx in resolved_layers:
                h = hs[layer_idx + 1][batch_idx]  # [seq, dim]
                h = h[seq_mask]                    # strip padding

                if centered:
                    h = h - h.mean(dim=0, keepdim=True)

                if pooling == "last_token":
                    vec = h[-1]
                elif pooling == "mean":
                    vec = h.mean(dim=0)
                else:
                    raise ValueError(f"Unknown pooling: {pooling!r}")

                activations[layer_idx] = vec.float().cpu().numpy()

            results.append(activations)

        log.info(
            "forward: %d inputs, layers=%s, pooling=%s, steer=%s, %.3fs",
            len(texts), resolved_layers, pooling,
            steer_mode if steer_direction is not None else "none",
            elapsed,
        )
        return results

    def __repr__(self) -> str:
        return (
            f"HFProbe(model={self.model_id!r}, "
            f"layers={self.layers}, device={self.device!r})"
        )


# ── Steering hook factory ──────────────────────────────────────────────────────

def _make_steering_hook(direction: torch.Tensor, alpha: float, mode: str):
    """
    Returns a forward hook that modifies the hidden state in-place.

    The hook is applied to the post-attention layernorm output of each layer,
    meaning it operates on the normalised activations before the MLP.

    Modes:
        "add"         — h += alpha * direction
        "project_out" — h -= (h · d̂) * d̂  (remove component along direction)
        "replace"     — h = h - (h · d̂) * d̂ + alpha * d̂  (replace component)
    """
    d_norm = direction / (direction.norm() + 1e-8)

    def hook(module, input, output):
        # output may be a tuple (hidden_state, ...) depending on the layer.
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        if mode == "add":
            h = h + alpha * direction
        elif mode == "project_out":
            proj = (h * d_norm).sum(dim=-1, keepdim=True) * d_norm
            h = h - proj
        elif mode == "replace":
            proj = (h * d_norm).sum(dim=-1, keepdim=True) * d_norm
            h = h - proj + alpha * d_norm

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    return hook


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_layer(model, layer_idx: int):
    """
    Return the transformer block at layer_idx.

    Supports standard architectures (Qwen3, Llama, DeepSeek, Mistral).
    The hook is applied to the whole layer block (not just the norm) so
    that the output tensor is the block's full hidden-state output.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise RuntimeError(
        f"Cannot locate layer {layer_idx} in this model. "
        "Supported: model.model.layers (Qwen3/Llama/DeepSeek) or model.transformer.h (GPT-2)."
    )


def _count_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            v = getattr(cfg, attr, None)
            if v is not None:
                return int(v)
    raise RuntimeError(
        "Cannot determine layer count. "
        "Check model.model.layers or model.config.num_hidden_layers."
    )
