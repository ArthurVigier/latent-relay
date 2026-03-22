"""
ERIS v5 — Analyzers
====================
Pluggable analyzers for hidden state inspection.  Each analyzer is:

  - **Optional** — skips gracefully if the required model/data is not configured.
  - **Lazy-loaded** — weights are loaded on first call, not at import time.
  - **Stateless across requests** — analyze() / score() are pure functions once loaded.
  - **Thread-safe** — a per-analyzer lock protects the one-time load.

All public analyze/score methods receive a float32 CPU tensor of shape
``[seq_len, hidden_dim]`` (the output of ``engine._tensor_to_payload`` decoded,
or directly from ``LatentThought.layer_hidden_states``).

For single-token inputs (e.g. the last hidden state from think()), pass a
``[1, hidden_dim]`` tensor — analyzers handle both shapes consistently.

Available analyzers
-------------------
- :class:`SAEAnalyzer`       — top-k sparse feature activations (sae-lens format)
- :class:`AHatAnalyzer`      — Â-hat agentivity score (linear probe)
- :class:`CosineMapAnalyzer` — cosine similarity to reference concept vectors
- :class:`PCAAnalyzer`       — incremental PCA → 3D projection per token
- :class:`NormAnalyzer`      — L2 norm per token (trivial, always available)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseAnalyzer:
    """Shared scaffolding: lazy-load lock + availability gate."""

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._loaded: bool = False
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        """True if this analyzer has been successfully loaded."""
        return self._loaded

    def _ensure_loaded(self) -> bool:
        """
        Call _load() exactly once.  Returns True if the analyzer is ready.
        Thread-safe: concurrent callers block until the first load finishes.
        """
        if self._loaded:
            return True
        if self._load_error is not None:
            return False
        with self._load_lock:
            if self._loaded or self._load_error:
                return self._loaded
            try:
                self._load()
                self._loaded = True
            except Exception as exc:
                self._load_error = str(exc)
                print(f"[{type(self).__name__}] Load failed: {exc}")
        return self._loaded

    def _load(self) -> None:
        """Override in subclasses to load weights/models."""
        raise NotImplementedError


# ── SAEAnalyzer ───────────────────────────────────────────────────────────────

class SAEAnalyzer(BaseAnalyzer):
    """
    Sparse Autoencoder analysis (sae-lens format).

    Loads an encoder from a directory or ``.pt`` checkpoint and returns the
    top-k feature activations for a hidden state.

    Expected checkpoint keys (state dict or top-level dict):
      - ``W_enc``  : shape [hidden_dim, n_features]
      - ``b_enc``  : shape [n_features]
      - ``threshold`` (optional): per-feature JumpReLU threshold, shape [n_features]

    If the checkpoint does not contain these keys, loading fails gracefully and
    the analyzer is marked unavailable.

    Args:
        model_path: Path to a ``.pt`` file or a directory containing
                    ``sae_weights.pt`` / ``model.pt``.
        labels_path: Optional path to a JSON file mapping str(feature_idx) to
                     a human-readable label.
        top_k: Number of top features to return.
        device: Torch device for encoder weights.  CPU is fine for inference.
    """

    def __init__(
        self,
        model_path: Optional[str],
        labels_path: Optional[str] = None,
        top_k: int = 20,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.labels_path = labels_path
        self.top_k = top_k
        self.device = device

        # Populated by _load()
        self._W_enc: Optional[torch.Tensor] = None   # [hidden_dim, n_features]
        self._b_enc: Optional[torch.Tensor] = None   # [n_features]
        self._threshold: Optional[torch.Tensor] = None
        self._labels: Optional[Dict[int, str]] = None

    def _load(self) -> None:
        if self.model_path is None:
            raise ValueError("model_path not set")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"SAE path not found: {path}")

        # Locate the weights file
        if path.is_dir():
            candidates = ["sae_weights.pt", "model.pt", "weights.pt"]
            ckpt_path = next(
                (path / c for c in candidates if (path / c).exists()), None
            )
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"No weights file found in {path}. Expected one of {candidates}"
                )
        else:
            ckpt_path = path

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch

        # Support both raw state dicts and wrapped {"cfg": ..., "state_dict": ...}
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        if "W_enc" not in state:
            raise KeyError(
                f"SAE checkpoint missing 'W_enc'. Available keys: {list(state.keys())}"
            )

        self._W_enc = state["W_enc"].float().to(self.device)       # [d_h, n_feat]
        self._b_enc = state["b_enc"].float().to(self.device)       # [n_feat]
        self._threshold = state.get("threshold")
        if self._threshold is not None:
            self._threshold = self._threshold.float().to(self.device)

        # Optional labels
        if self.labels_path and Path(self.labels_path).exists():
            import json
            with open(self.labels_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._labels = {int(k): v for k, v in raw.items()}

    def analyze(self, hidden: torch.Tensor) -> Optional[Dict]:
        """
        Compute top-k feature activations for the mean-pooled hidden state.

        Args:
            hidden: Float32 tensor of shape ``[seq_len, hidden_dim]``.
                    Must already be on ``self.device`` — do NOT pass CPU tensors
                    when the SAE weights are on GPU.  Use Pattern A (inline during
                    rollout, while hidden is still on GPU) or Pattern B (re-upload
                    via ``AnalyzerRegistry.run(..., inference_device=...)``.

        Returns:
            Dict with ``top_k`` list of ``{index, activation, label}`` dicts,
            or None if the analyzer is not available.
        """
        if not self._ensure_loaded():
            return None

        # Keep tensor on self.device — never silently move GPU→CPU.
        h = hidden.float().to(self.device)
        h_mean = h.mean(dim=0)  # [hidden_dim]

        # Encode: features = activation_fn(h @ W_enc + b_enc)
        pre_act = h_mean @ self._W_enc + self._b_enc  # [n_features]

        if self._threshold is not None:
            # JumpReLU: feature active if pre_act > threshold
            activations = torch.where(
                pre_act > self._threshold,
                pre_act - self._threshold,
                torch.zeros_like(pre_act),
            )
        else:
            activations = F.relu(pre_act)

        # Top-k
        k = min(self.top_k, activations.shape[0])
        top_vals, top_idx = torch.topk(activations, k)

        features = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            features.append({
                "index": idx,
                "activation": round(val, 4),
                "label": self._labels.get(idx) if self._labels else None,
            })

        return {"top_k": features}


# ── AHatAnalyzer ─────────────────────────────────────────────────────────────

class AHatAnalyzer(BaseAnalyzer):
    """
    Â-hat agentivity probe.

    Loads a linear probe trained to predict the agentivity score of a hidden
    state.  Supports two checkpoint formats:

    1. **PyTorch state dict** with keys ``weight`` [1, hidden_dim] and
       ``bias`` [1] — the probe is ``sigmoid(h @ weight.T + bias)``.
    2. **sklearn** ``.pkl`` or ``.joblib`` file — called as
       ``probe.predict_proba(h_numpy)[..., 1]``.

    The score returned is a float in [0, 1].

    Args:
        model_path: Path to the probe checkpoint (``.pt``, ``.pkl``, ``.joblib``).
        layer: Hidden layer the probe was trained on (-1 = last).  Stored as
               metadata only — callers must pass the correct layer.
        device: Torch device for the weight tensor.
    """

    def __init__(
        self,
        model_path: Optional[str],
        layer: int = -1,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.layer = layer
        self.device = device

        self._weight: Optional[torch.Tensor] = None  # [1, hidden_dim]
        self._bias: Optional[torch.Tensor] = None    # [1]
        self._sklearn_probe = None

    def _load(self) -> None:
        if self.model_path is None:
            raise ValueError("model_path not set")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Â-hat probe not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".pt":
            ckpt = torch.load(path, map_location="cpu", weights_only=True)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if "weight" not in state:
                raise KeyError(
                    f"Probe checkpoint missing 'weight'. Keys: {list(state.keys())}"
                )
            self._weight = state["weight"].float().to(self.device)  # [1, d_h]
            self._bias = state.get("bias", torch.zeros(1)).float().to(self.device)

        elif suffix in {".pkl", ".joblib"}:
            try:
                import joblib
            except ImportError as exc:
                raise ImportError(
                    "The 'joblib' package is required to load .pkl/.joblib probe files. "
                    "Install with: pip install joblib"
                ) from exc
            self._sklearn_probe = joblib.load(path)  # nosemgrep: trailofbits.python.scikit-joblib-load

        else:
            raise ValueError(
                f"Unsupported probe format: {suffix}. Expected .pt, .pkl, or .joblib"
            )

    def score(self, hidden: torch.Tensor) -> Optional[float]:
        """
        Compute the Â-hat agentivity score for a hidden state.

        Args:
            hidden: Float32 tensor of shape ``[seq_len, hidden_dim]`` or
                    ``[1, hidden_dim]``.  Mean-pooled before scoring.
                    Must already be on ``self.device`` — do NOT pass CPU tensors
                    when probe weights are on GPU.

        Returns:
            Float in [0, 1], or None if the analyzer is not available.
        """
        if not self._ensure_loaded():
            return None

        # Keep on self.device — never silently move GPU→CPU.
        h = hidden.float().to(self.device).mean(dim=0)  # [hidden_dim]

        if self._sklearn_probe is not None:
            h_np = h.cpu().numpy().reshape(1, -1)
            prob = self._sklearn_probe.predict_proba(h_np)[0, 1]
            return float(prob)

        # Linear probe: sigmoid(h @ W^T + b)
        logit = (h @ self._weight.squeeze(0)) + self._bias.squeeze(0)
        return float(torch.sigmoid(logit).item())

    def as_callback(self):
        """
        Return a callable suitable for passing as ``a_hat_fn`` to
        ``LatentRelayEngine.think()``.

        The returned function accepts a ``[1, hidden_dim]`` tensor (last hidden
        state, still on GPU) and returns a float.
        """
        def _fn(hidden: torch.Tensor) -> Optional[float]:
            return self.score(hidden)
        return _fn


# ── CosineMapAnalyzer ─────────────────────────────────────────────────────────

class CosineMapAnalyzer(BaseAnalyzer):
    """
    Cosine similarity map against pre-computed reference concept vectors.

    Loads all ``.pt`` and ``.npy`` files from a directory.  Each file contains
    a vector of shape ``[hidden_dim]`` representing a concept.  The analyzer
    returns the cosine similarity between the mean-pooled hidden state and each
    concept vector.

    Args:
        vectors_dir: Directory containing concept vector files.
    """

    def __init__(self, vectors_dir: str) -> None:
        super().__init__()
        self.vectors_dir = vectors_dir
        self._concepts: Dict[str, torch.Tensor] = {}  # name → normalised [hidden_dim]

    def _load(self) -> None:
        p = Path(self.vectors_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Concept vectors directory not found: {p}")

        loaded = 0
        for f in sorted(p.iterdir()):
            if f.suffix == ".pt":
                vec = torch.load(f, map_location="cpu", weights_only=True)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
            elif f.suffix == ".npy":
                vec = torch.from_numpy(np.load(f))
            else:
                continue

            vec = vec.float().reshape(-1)
            norm = vec.norm()
            if norm < 1e-8:
                print(f"[CosineMapAnalyzer] Skipping zero vector: {f.name}")
                continue
            self._concepts[f.stem] = vec / norm
            loaded += 1

        if loaded == 0:
            raise ValueError(f"No valid concept vectors found in {p}")

    def analyze(self, hidden: torch.Tensor) -> Optional[Dict[str, float]]:
        """
        Compute cosine similarity between mean-pooled hidden state and each concept.

        Args:
            hidden: Float32 tensor of shape ``[seq_len, hidden_dim]``.

        Returns:
            Dict mapping concept name to cosine similarity in [-1, 1],
            or None if the analyzer is not available.
        """
        if not self._ensure_loaded():
            return None

        h_mean = hidden.float().mean(dim=0)
        h_norm = h_mean.norm()
        if h_norm < 1e-8:
            return {name: 0.0 for name in self._concepts}
        h_normed = h_mean / h_norm

        return {
            name: round(float(torch.dot(h_normed, vec).clamp(-1.0, 1.0).item()), 4)
            for name, vec in self._concepts.items()
        }


# ── PCAAnalyzer ───────────────────────────────────────────────────────────────

class PCAAnalyzer(BaseAnalyzer):
    """
    Incremental PCA for 3D projection of hidden states per token.

    Uses sklearn's ``IncrementalPCA`` which can be updated online (call
    :meth:`partial_fit`) or used with a pre-fitted model.

    The analyzer is **always available** (no external checkpoint needed) but
    requires at least ``n_components`` samples before it can transform.

    Args:
        n_components: Number of PCA dimensions (default 3).
        batch_size: Mini-batch size for partial_fit.
    """

    def __init__(self, n_components: int = 3, batch_size: int = 256) -> None:
        super().__init__()
        self.n_components = n_components
        self.batch_size = batch_size
        self._pca = None
        self._n_samples_seen: int = 0

    def _load(self) -> None:
        try:
            from sklearn.decomposition import IncrementalPCA
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for PCAAnalyzer. "
                "Install with: pip install scikit-learn"
            ) from exc
        self._pca = IncrementalPCA(
            n_components=self.n_components, batch_size=self.batch_size
        )

    def partial_fit(self, hidden: torch.Tensor) -> None:
        """
        Update the PCA model with new hidden states.

        Args:
            hidden: Float32 tensor of shape ``[seq_len, hidden_dim]``.
        """
        if not self._ensure_loaded():
            return
        X = hidden.detach().cpu().float().numpy()
        if X.shape[0] < self.n_components:
            return  # Not enough samples for this batch
        self._pca.partial_fit(X)
        self._n_samples_seen += X.shape[0]

    def analyze(self, hidden: torch.Tensor) -> Optional[List[List[float]]]:
        """
        Project hidden states to 3D using the fitted PCA model.

        Args:
            hidden: Float32 tensor of shape ``[seq_len, hidden_dim]``.

        Returns:
            List of ``[x, y, z]`` coordinates (one per token), or None if the
            PCA has not been fitted yet (need > n_components samples total).
        """
        if not self._ensure_loaded():
            return None
        if self._n_samples_seen < self.n_components:
            return None  # Not fitted yet

        X = hidden.detach().cpu().float().numpy()
        projected = self._pca.transform(X)  # [seq_len, 3]
        return [[round(float(v), 4) for v in row] for row in projected.tolist()]


# ── NormAnalyzer ──────────────────────────────────────────────────────────────

class NormAnalyzer(BaseAnalyzer):
    """
    Per-token L2 norm.  Always available — no external dependencies.

    Useful as a cheap proxy for "how much information" is encoded at each
    token position.
    """

    def _load(self) -> None:
        pass  # Nothing to load

    def analyze(self, hidden: torch.Tensor) -> List[float]:
        """
        Compute L2 norm for each token's hidden state.

        Args:
            hidden: Float32 tensor of shape ``[seq_len, hidden_dim]``.

        Returns:
            List of floats, one per token.
        """
        self._ensure_loaded()
        norms = hidden.float().norm(dim=-1)  # [seq_len]
        return [round(float(v), 4) for v in norms.tolist()]


# ── AnalyzerRegistry ──────────────────────────────────────────────────────────

class AnalyzerRegistry:
    """
    Holds all analyzer instances and dispatches ``/v1/analyze`` requests.

    Analyzers are registered at startup from an :class:`~eris.config.ERISConfig`.
    Missing or unavailable analyzers are silently skipped.

    Usage::

        from eris.config import ERISConfig
        from eris.analyzers import AnalyzerRegistry

        cfg = ERISConfig.load()
        registry = AnalyzerRegistry.from_config(cfg)

        hidden = thought.layer_hidden_states["last"]   # [seq_len, hidden_dim]
        results = registry.run(hidden, analyses=["sae_features", "a_hat", "token_norms"])
    """

    def __init__(
        self,
        *,
        sae: Optional[SAEAnalyzer] = None,
        a_hat: Optional[AHatAnalyzer] = None,
        cosine_map: Optional[CosineMapAnalyzer] = None,
        pca: Optional[PCAAnalyzer] = None,
        norm: Optional[NormAnalyzer] = None,
    ) -> None:
        self.sae = sae
        self.a_hat = a_hat
        self.cosine_map = cosine_map
        self.pca = pca  # None → pca_3d returns None until fitted
        self.norm = norm or NormAnalyzer()

    @classmethod
    def from_config(cls, cfg) -> "AnalyzerRegistry":
        """
        Build a registry from an :class:`~eris.config.ERISConfig`.

        SAEAnalyzer and AHatAnalyzer receive ``cfg.device`` so their weights
        live on the same device as the zombie model.  This ensures inference
        runs on GPU (Pattern A/B) without silent CPU fallbacks.

        Args:
            cfg: ERISConfig instance.

        Returns:
            AnalyzerRegistry with all configured analyzers.
        """
        sae = (
            SAEAnalyzer(
                model_path=cfg.sae.model_path,
                labels_path=cfg.sae.labels_path,
                top_k=cfg.sae.top_k,
                device=cfg.device,          # same device as zombie model
            )
            if cfg.sae.model_path is not None
            else None
        )
        a_hat = (
            AHatAnalyzer(
                model_path=cfg.a_hat.model_path,
                layer=cfg.a_hat.layer,
                device=cfg.device,          # same device as zombie model
            )
            if cfg.a_hat.model_path is not None
            else None
        )
        cosine_map = (
            CosineMapAnalyzer(vectors_dir=cfg.concept_vectors.vectors_dir)
            if cfg.concept_vectors.available
            else None
        )
        return cls(
            sae=sae,
            a_hat=a_hat,
            cosine_map=cosine_map,
            pca=PCAAnalyzer(),
            norm=NormAnalyzer(),
        )

    def run(
        self,
        hidden: torch.Tensor,
        analyses: List[str],
        *,
        inference_device: Optional[str] = None,
    ) -> Dict:
        """
        Run the requested analyses on a hidden state tensor.

        **Pattern A (inline, during rollout):** ``hidden`` is already on GPU.
        Do not pass ``inference_device`` — the tensor is used as-is.

        **Pattern B (post-hoc, from stored CPU snapshot):** ``hidden`` is on CPU
        (from ``LatentThought.layer_hidden_states``).  Pass
        ``inference_device="cuda:0"`` (or the engine's device) so SAE / Â-hat
        weights receive a GPU tensor.  The temporary GPU copy is freed after
        each GPU-bound analysis.  CPU-only analyzers (cosine_map, pca_3d,
        token_norms) always receive the original CPU tensor.

        Args:
            hidden: Float32 tensor of shape ``[seq_len, hidden_dim]``.
            analyses: List of analysis names.  Valid values:
                      ``"sae_features"``, ``"a_hat"``, ``"cosine_map"``,
                      ``"pca_3d"``, ``"token_norms"``.
            inference_device: Device string for GPU-bound analyzers (Pattern B).
                              None means use ``hidden`` as-is (Pattern A).

        Returns:
            Dict with one key per requested analysis.  Unavailable analyzers
            return None for their key without raising.
        """
        result: Dict = {}

        # Normalise to 2-D — bridge passes [hidden_dim] (last token only),
        # encode passes [seq_len, hidden_dim].  All analyzers expect 2-D.
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)  # [1, hidden_dim]

        # GPU-bound analyses: upload once, run, then explicitly free.
        needs_gpu = any(n in analyses for n in ("sae_features", "a_hat"))
        hidden_gpu: Optional[torch.Tensor] = None
        if needs_gpu and inference_device is not None:
            hidden_gpu = hidden.to(inference_device)

        for name in analyses:
            if name == "sae_features":
                h = hidden_gpu if hidden_gpu is not None else hidden
                result["sae_features"] = (
                    self.sae.analyze(h) if self.sae is not None else None
                )

            elif name == "a_hat":
                h = hidden_gpu if hidden_gpu is not None else hidden
                score = (
                    self.a_hat.score(h) if self.a_hat is not None else None
                )
                result["a_hat_score"] = (
                    round(score, 4) if score is not None else None
                )

            elif name == "cosine_map":
                # CosineMap is CPU-only (dot products on normalised float32 vecs)
                result["cosine_map"] = (
                    self.cosine_map.analyze(hidden.cpu())
                    if self.cosine_map is not None
                    else None
                )

            elif name == "pca_3d":
                # PCA is CPU-only (sklearn)
                result["pca_3d"] = (
                    self.pca.analyze(hidden.cpu()) if self.pca is not None else None
                )

            elif name == "token_norms":
                result["token_norms"] = self.norm.analyze(hidden.cpu())

            else:
                result[name] = None  # Unknown analysis → null, no crash

        # Release temporary GPU copy immediately — don't hold VRAM between calls.
        if hidden_gpu is not None:
            del hidden_gpu
            if inference_device is not None and "cuda" in inference_device:
                torch.cuda.empty_cache()

        return result
