"""
ERIS v5 — Configuration
========================
Parses and validates the ERIS config file (configs/eris_config.yaml).
All paths to optional components (SAE, Â-hat probe, concept vectors)
are resolved here. Missing components degrade gracefully — they are
marked as unavailable rather than crashing.

Usage:
    from eris.config import ERISConfig
    cfg = ERISConfig.load()           # loads from default path
    cfg = ERISConfig.load("my.yaml")  # loads from custom path
    cfg = ERISConfig.default()        # in-memory defaults, no file needed
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ── Default path for the config file ──────────────────────────────────────────
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "eris_config.yaml"


@dataclass
class SAEConfig:
    """Configuration for a Sparse Autoencoder (sae-lens format)."""

    # Path to the SAE weights directory or .pt file.
    # If None or the path does not exist, SAEAnalyzer is disabled.
    model_path: Optional[str] = None

    # Which hidden layer the SAE was trained on (-1 = last).
    layer: int = -1

    # Path to a JSON file mapping feature index → human-readable label.
    # e.g. {"4521": "code_architecture", ...}
    # Optional — if missing, labels are reported as None.
    labels_path: Optional[str] = None

    # Number of top features to return per analysis.
    top_k: int = 20

    @property
    def available(self) -> bool:
        """True if the SAE weights file/directory exists on disk."""
        if self.model_path is None:
            return False
        return Path(self.model_path).exists()


@dataclass
class AHatConfig:
    """Configuration for the Â-hat agentivity probe (a-hat-optimizer format)."""

    # Path to the probe checkpoint (.pt file produced by a-hat-optimizer).
    # If None or the path does not exist, AHatAnalyzer is disabled.
    model_path: Optional[str] = None

    # Hidden layer the probe was trained on (-1 = last).
    layer: int = -1

    @property
    def available(self) -> bool:
        """True if the probe checkpoint exists on disk."""
        if self.model_path is None:
            return False
        return Path(self.model_path).exists()


@dataclass
class ConceptVectorsConfig:
    """Configuration for the cosine-map concept vectors."""

    # Directory containing one .pt or .npy file per concept.
    # File stem is used as the concept name: "game_theory.pt" → "game_theory".
    # If the directory does not exist, CosineMapAnalyzer returns an empty map.
    vectors_dir: str = str(
        Path(__file__).parent.parent / "configs" / "concept_vectors"
    )

    @property
    def available(self) -> bool:
        """True if the vectors directory exists and contains at least one file."""
        p = Path(self.vectors_dir)
        if not p.is_dir():
            return False
        return any(p.iterdir())

    def list_concepts(self) -> List[str]:
        """Return concept names derived from files present in vectors_dir."""
        p = Path(self.vectors_dir)
        if not p.is_dir():
            return []
        return [
            f.stem
            for f in sorted(p.iterdir())
            if f.suffix in {".pt", ".npy"}
        ]


@dataclass
class ServerConfig:
    """HTTP server settings for eris_server.py."""

    host: str = "0.0.0.0"
    port: int = 8001

    # Maximum size (in bytes) for base64-encoded hidden state payloads.
    # Requests exceeding this are rejected with HTTP 413.
    max_payload_bytes: int = 200 * 1024 * 1024  # 200 MB


@dataclass
class ERISConfig:
    """
    Top-level ERIS v5 configuration.

    All optional components (SAE, Â-hat, concept vectors) are represented
    by sub-configs with an `.available` property. Engine code checks
    `.available` before attempting to load — no component is mandatory.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    # HuggingFace model name or local path for the zombie model.
    model_name: str = "Qwen/Qwen3-14B"

    # Torch device for the zombie model.
    device: str = "cuda:0"

    # ── Optional components ──────────────────────────────────────────────────
    sae: SAEConfig = field(default_factory=SAEConfig)
    a_hat: AHatConfig = field(default_factory=AHatConfig)
    concept_vectors: ConceptVectorsConfig = field(default_factory=ConceptVectorsConfig)

    # ── Server ───────────────────────────────────────────────────────────────
    server: ServerConfig = field(default_factory=ServerConfig)

    # ── Latent rollout defaults ───────────────────────────────────────────────
    # Default number of latent rollout steps if not specified per-request.
    default_n_steps: int = 60

    # ── Path book-keeping (populated by load(), not written to YAML) ─────────
    _config_path: Optional[str] = field(default=None, repr=False, compare=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Factory methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "ERISConfig":
        """Return an in-memory default config without reading any file."""
        return cls()

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ERISConfig":
        """
        Load config from a YAML file.

        Args:
            path: Path to the YAML file.  Defaults to
                  ``configs/eris_config.yaml`` relative to the repo root.

        Returns:
            ERISConfig instance populated from the file.  Falls back to
            defaults for any key not present in the file.

        Raises:
            FileNotFoundError: If an explicit path was given but does not exist.
        """
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

        if not config_path.exists():
            if path is not None:
                raise FileNotFoundError(
                    f"[ERISConfig] Config file not found: {config_path}"
                )
            # Default path missing → use in-memory defaults silently.
            cfg = cls()
            cfg._config_path = None
            return cfg

        with open(config_path, "r", encoding="utf-8") as fh:
            raw: Dict = yaml.safe_load(fh) or {}

        cfg = cls._from_dict(raw)
        cfg._config_path = str(config_path.resolve())
        return cfg

    @classmethod
    def _from_dict(cls, d: Dict) -> "ERISConfig":
        """Populate an ERISConfig from a parsed YAML dictionary."""

        def _get(section: str) -> Dict:
            return d.get(section) or {}

        sae_d = _get("sae")
        a_hat_d = _get("a_hat")
        cv_d = _get("concept_vectors")
        srv_d = _get("server")

        return cls(
            model_name=d.get("model_name", "Qwen/Qwen3-14B"),
            device=d.get("device", "cuda:0"),
            sae=SAEConfig(
                model_path=sae_d.get("model_path"),
                layer=sae_d.get("layer", -1),
                labels_path=sae_d.get("labels_path"),
                top_k=sae_d.get("top_k", 20),
            ),
            a_hat=AHatConfig(
                model_path=a_hat_d.get("model_path"),
                layer=a_hat_d.get("layer", -1),
            ),
            concept_vectors=ConceptVectorsConfig(
                vectors_dir=cv_d.get(
                    "vectors_dir",
                    str(Path(__file__).parent.parent / "configs" / "concept_vectors"),
                )
            ),
            server=ServerConfig(
                host=srv_d.get("host", "0.0.0.0"),
                port=srv_d.get("port", 8001),
                max_payload_bytes=srv_d.get("max_payload_bytes", 50 * 1024 * 1024),
            ),
            default_n_steps=d.get("default_n_steps", 60),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of which components are available."""
        lines = [
            "── ERIS v5 Config ──────────────────────────────",
            f"  model        : {self.model_name}",
            f"  device       : {self.device}",
            f"  config file  : {self._config_path or '(defaults, no file)'}",
            f"  SAE          : {'✓ ' + self.sae.model_path if self.sae.available else '✗ not configured'}",
            f"  Â-hat probe  : {'✓ ' + self.a_hat.model_path if self.a_hat.available else '✗ not configured'}",
            f"  concept vecs : {'✓ ' + str(len(self.concept_vectors.list_concepts())) + ' concepts' if self.concept_vectors.available else '✗ none found'}",
            f"  server       : {self.server.host}:{self.server.port}",
            "────────────────────────────────────────────────",
        ]
        return "\n".join(lines)
