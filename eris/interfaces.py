"""
eris/interfaces.py
==================

Contrats abstraits d'ERIS.

L'architecture est indépendante de tout fournisseur spécifique.
N'importe quel LLM peut être orchestrateur (Claude, Gemini, OpenAI, local).
N'importe quel modèle open-weights peut être sonde (Qwen3, DeepSeek-R1, etc).

Toute implémentation concrète doit hériter des classes abstraites définies ici
et implémenter toutes leurs méthodes.  La factory (eris/factory.py) instancie
les backends appropriés selon la configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Shared data types ─────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    """One step of reasoning produced by an OrchestratorLLM."""
    content:     str    # text of the reasoning step
    step_idx:    int
    uncertainty: float  # [0, 1] — self-assessed by the model
    metadata:    dict = field(default_factory=dict)


@dataclass
class RecalibrationNote:
    """A recalibration signal produced by OrchestratorLLM.interpret_activations()."""
    content:                      str
    suggested_steering_direction: Optional[np.ndarray]  # None if no steering suggested
    suggested_alpha:              float                  # steering intensity
    confidence:                   float                  # [0, 1]


@dataclass
class DriftReport:
    """
    Output of a single DriftDetector.compute_drift() call.

    This V1 dataclass remains valid for older backends while exposing a small
    compatibility surface for V2 SAE-based callers.
    """
    step:                 int
    drift_score:          float   # smoothed, [0, 1]
    raw_drift_score:      float   # unsmoothed, this step only
    cosine_distances:     dict[int, float]
    l2_distances:         dict[int, float]
    llc_scores:           dict[int, float]
    layers_affected:      list[int]   # ranked highest drift first
    should_consult_probe: bool
    threshold:            float

    @property
    def layers_ranked(self) -> list[int]:
        return list(self.layers_affected)

    @property
    def comparison_mode(self) -> str:
        return "reference"

    @property
    def severity(self) -> str:
        score = self.drift_score
        if score < 0.10:
            return "stable"
        if score < 0.35:
            return "low"
        if score < 0.80:
            return "medium"
        return "high"

    @property
    def layer_scores(self) -> dict[int, float]:
        return {}

    @property
    def features_lost(self) -> dict[int, list[int]]:
        return {}

    @property
    def features_gained(self) -> dict[int, list[int]]:
        return {}

    @property
    def n_active_per_layer(self) -> dict[int, int]:
        return {}

    @property
    def n_layers_evaluated(self) -> int:
        return len(self.layers_affected)

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "drift_score": self.drift_score,
            "raw_drift_score": self.raw_drift_score,
            "cosine_distances": self.cosine_distances,
            "l2_distances": self.l2_distances,
            "llc_scores": self.llc_scores,
            "layers_affected": self.layers_affected,
            "layers_ranked": self.layers_ranked,
            "should_consult_probe": self.should_consult_probe,
            "threshold": self.threshold,
            "comparison_mode": self.comparison_mode,
            "severity": self.severity,
            "layer_scores": self.layer_scores,
            "features_lost": self.features_lost,
            "features_gained": self.features_gained,
            "n_active_per_layer": self.n_active_per_layer,
            "n_layers_evaluated": self.n_layers_evaluated,
        }


# ── OrchestratorLLM ───────────────────────────────────────────────────────────

class OrchestratorLLM(ABC):
    """
    Interface for the orchestrator model.

    Responsible for primary reasoning and interpretation of activation
    descriptions.  Does NOT receive raw numpy arrays — only structured
    text descriptions of the activation geometry.
    """

    @abstractmethod
    def reason_step(
        self,
        problem: str,
        history: list[ReasoningStep],
        recalibration_context: Optional[str] = None,
    ) -> ReasoningStep:
        """
        Produce one reasoning step.

        Args:
            problem:                 The original problem statement.
            history:                 All prior reasoning steps in this run.
            recalibration_context:   Optional note from the previous probe
                                     consultation, if one was triggered.

        Returns:
            ReasoningStep with content, step_idx, uncertainty, metadata.
        """
        ...

    @abstractmethod
    def interpret_activations(
        self,
        activations_description: str,
        drift_report: DriftReport,
        problem_context: str,
    ) -> RecalibrationNote:
        """
        Receive a structured text description of probe activations and
        produce a recalibration note.

        Never receives raw numpy arrays — only a human-readable description
        of the activation geometry (norms, directions, LLC scores, etc).

        Args:
            activations_description: Text output of
                                     ERISOrchestrator._format_activations_for_claude().
            drift_report:            The DriftReport that triggered this consultation.
            problem_context:         The current accumulated reasoning context.

        Returns:
            RecalibrationNote with content and optional steering suggestion.
        """
        ...

    @abstractmethod
    def estimate_uncertainty(self, reasoning_step: ReasoningStep) -> float:
        """
        Estimate the uncertainty level of the given reasoning step.

        Implementations may use self-ask prompting, logprobs, or text
        heuristics.  Must return a float in [0, 1].
        """
        ...


# ── ProbeModel ────────────────────────────────────────────────────────────────

class ProbeModel(ABC):
    """
    Interface for the probe model (zombie).

    Pure representation transformation — never generates text.
    Input → {layer: np.ndarray}.  That is all.

    Implementations must enforce max_new_tokens=0 (or equivalent) at all
    times.  Steering is applied via hidden-state manipulation, not prompting.
    """

    @abstractmethod
    def probe(
        self,
        input_text: str,
        layers: list[int],
        pooling: str = "last_token",
        centered: bool = True,
    ) -> dict[int, np.ndarray]:
        """
        Extract hidden-state activations for one input.

        Args:
            input_text: Text to encode.
            layers:     Layer indices to extract.
            pooling:    "last_token" or "mean".
            centered:   Subtract per-layer sequence mean before pooling.

        Returns:
            {layer_idx: np.ndarray[hidden_dim]}

        Raises:
            ProbeError: On encoding failure or unexpected output.
        """
        ...

    @abstractmethod
    def probe_batch(
        self,
        inputs: list[str],
        layers: list[int],
        pooling: str = "last_token",
        centered: bool = True,
    ) -> list[dict[int, np.ndarray]]:
        """Extract activations for a batch of inputs."""
        ...

    @abstractmethod
    def steer(
        self,
        input_text: str,
        direction: np.ndarray,
        alpha: float,
        layers: list[int],
        mode: str = "add",
    ) -> dict[int, np.ndarray]:
        """
        Apply a steering vector to the hidden activations and return
        the steered activations.  No text is generated.

        Args:
            input_text:  Text to encode.
            direction:   Steering vector (same dim as hidden state).
            alpha:       Steering intensity.
            layers:      Layers to steer.
            mode:        "add"         — activation += alpha * direction
                         "project_out" — remove component along direction
                         "replace"     — replace component with alpha * direction

        Returns:
            {layer_idx: steered_np.ndarray[hidden_dim]}
        """
        ...

    @abstractmethod
    def steer_batch(
        self,
        inputs: list[str],
        direction: np.ndarray,
        alpha: float,
        layers: list[int],
        mode: str = "add",
    ) -> list[dict[int, np.ndarray]]:
        """Apply steering to a batch of inputs."""
        ...

    @abstractmethod
    def save_direction(self, name: str, vector: np.ndarray) -> None:
        """Save a named steering direction to the in-memory library."""
        ...

    @abstractmethod
    def load_direction(self, name: str) -> np.ndarray:
        """
        Load a named steering direction from the library.

        Raises:
            KeyError: If the direction is not in the library.
        """
        ...

    @abstractmethod
    def list_directions(self) -> list[str]:
        """Return the names of all saved steering directions."""
        ...
