"""
eris/drift_detector.py
=======================

Détecteur de drift latent V2 — basé sur features SAE.

Rôle dans ERIS V2 : mesurer la divergence entre la représentation SAE
initiale (référence) et les représentations aux checkpoints suivants.

Pourquoi les features SAE plutôt que les activations brutes ?
    - Les features sont sparse et sémantiques : un index = un concept.
    - La divergence devient une diff de sets interprétable.
    - Claude peut lire "feature 412 a disparu" et y attacher du sens
      (via Neuronpedia) plutôt que "dim 412 a varié de 0.3".

Métriques combinées :
    1. Distance Jaccard sur les features actives (drift thématique)
    2. Distance cosine sur les activations brutes (drift géométrique)
    → Score final = moyenne pondérée, lissée sur une fenêtre mobile.

Nouveautés :
    - pondération optionnelle par couche via ``layer_weights``
    - mode de comparaison ``reference`` ou ``previous``
    - sévérité et classement des couches les plus instables
    - export sérialisable via ``DriftReport.to_dict()``

Usage::

    detector = DriftDetector(threshold=0.35, window=3)

    ref = sae_probe.probe(problem_statement)
    detector.register_reference(ref, step=0)

    for step, reasoning_chunk in enumerate(steps):
        cur = sae_probe.probe(reasoning_chunk)
        report = detector.compute_drift(cur, step=step + 1)
        if report.should_consult_probe:
            # passer report.summary à Claude
            ...
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Note : DriftReport V2 est défini localement.
# Le DriftReport V1 (eris/interfaces.py) reste intact pour les backends V1.
from eris.sae_probe import ProbeOutput

log = logging.getLogger("eris.drift_detector")

_VALID_COMPARISON_MODES = {"reference", "previous"}


def _safe_cosine_distance(ref_vec: np.ndarray, cur_vec: np.ndarray) -> float:
    ref32 = ref_vec.astype(np.float32)
    cur32 = cur_vec.astype(np.float32)
    norm = float(np.linalg.norm(ref32) * np.linalg.norm(cur32))
    if norm <= 0:
        return 0.0
    score = float(1.0 - np.dot(ref32, cur32) / norm)
    return max(0.0, min(1.0, score))


def _severity_from_score(score: float) -> str:
    if score < 0.10:
        return "stable"
    if score < 0.35:
        return "low"
    if score < 0.80:
        return "medium"
    return "high"


@dataclass
class DriftReport:
    """
    Résultat d'un appel DriftDetector.compute_drift() — version V2 SAE.

    Attributes:
        step:                  Étape de raisonnement.
        drift_score:           Score [0, 1], lissé sur la fenêtre.
        raw_drift_score:       Score brut non lissé pour ce step.
        should_consult_probe:  True si drift_score > threshold.
        threshold:             Seuil actif.
        comparison_mode:       "reference" ou "previous".
        severity:              stable | low | medium | high.
        features_lost:         {layer: [indices]} features actives dans la baseline
                               absentes maintenant.
        features_gained:       {layer: [indices]} features nouvelles
                               absentes dans la baseline.
        cosine_distances:      {layer: float} distance cosine activations brutes.
        jaccard_distances:     {layer: float} distance Jaccard features actives.
        layer_scores:          {layer: float} score combiné par couche.
        layers_ranked:         Couches triées du drift le plus fort au plus faible.
        n_active_per_layer:    {layer: int} features actives au step courant.
        n_layers_evaluated:    Nombre de couches comparées.
        summary:               Description lisible pour Claude.
    """

    step: int
    drift_score: float
    raw_drift_score: float
    should_consult_probe: bool
    threshold: float
    comparison_mode: str = "reference"
    severity: str = "stable"
    features_lost: dict[int, list[int]] = field(default_factory=dict)
    features_gained: dict[int, list[int]] = field(default_factory=dict)
    cosine_distances: dict[int, float] = field(default_factory=dict)
    jaccard_distances: dict[int, float] = field(default_factory=dict)
    layer_scores: dict[int, float] = field(default_factory=dict)
    layers_ranked: list[int] = field(default_factory=list)
    n_active_per_layer: dict[int, int] = field(default_factory=dict)
    n_layers_evaluated: int = 0
    summary: str = ""

    @property
    def layers_affected(self) -> list[int]:
        return list(self.layers_ranked)

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "drift_score": self.drift_score,
            "raw_drift_score": self.raw_drift_score,
            "should_consult_probe": self.should_consult_probe,
            "threshold": self.threshold,
            "comparison_mode": self.comparison_mode,
            "severity": self.severity,
            "features_lost": self.features_lost,
            "features_gained": self.features_gained,
            "cosine_distances": self.cosine_distances,
            "jaccard_distances": self.jaccard_distances,
            "layer_scores": self.layer_scores,
            "layers_ranked": self.layers_ranked,
            "layers_affected": self.layers_affected,
            "n_active_per_layer": self.n_active_per_layer,
            "n_layers_evaluated": self.n_layers_evaluated,
            "summary": self.summary,
        }


class DriftDetector:
    """
    Détecteur de drift sur features SAE.

    Args:
        threshold:        Seuil de drift_score pour déclencher une consultation.
        window:           Taille de la fenêtre de lissage (moyenne mobile).
        jaccard_weight:   Poids de la distance Jaccard (drift thématique).
        cosine_weight:    Poids de la distance cosine (drift géométrique).
        layer_weights:    Poids optionnels par couche pour l'agrégation globale.
        comparison_mode:  "reference" (baseline fixe) ou "previous" (baseline glissante).
    """

    def __init__(
        self,
        threshold: float = 0.35,
        window: int = 3,
        jaccard_weight: float = 0.6,
        cosine_weight: float = 0.4,
        layer_weights: Optional[dict[int, float]] = None,
        comparison_mode: str = "reference",
    ) -> None:
        if abs(jaccard_weight + cosine_weight - 1.0) > 1e-6:
            raise ValueError("jaccard_weight + cosine_weight doit être égal à 1.0")
        if window <= 0:
            raise ValueError("window doit être >= 1")
        if comparison_mode not in _VALID_COMPARISON_MODES:
            raise ValueError(
                f"comparison_mode doit être dans {_VALID_COMPARISON_MODES}, reçu: {comparison_mode!r}"
            )

        self.threshold = threshold
        self.window = window
        self.jaccard_weight = jaccard_weight
        self.cosine_weight = cosine_weight
        self.layer_weights = dict(layer_weights or {})
        self.comparison_mode = comparison_mode

        self._reference: Optional[dict[int, ProbeOutput]] = None
        self._previous: Optional[dict[int, ProbeOutput]] = None
        self._history: deque[float] = deque(maxlen=window)
        self._max_drift: float = 0.0

    @property
    def max_drift(self) -> float:
        """Score de drift maximum observé depuis le dernier reset()."""
        return self._max_drift

    def register_reference(
        self,
        probe_output: dict[int, ProbeOutput],
        step: int = 0,
    ) -> None:
        """
        Enregistre l'état de référence (step 0 du raisonnement).

        À appeler une fois au début de chaque run() avec le probe du
        problem statement.
        """
        self._reference = probe_output
        self._previous = probe_output
        self._history.clear()
        self._max_drift = 0.0
        log.info(
            "Référence enregistrée (step=%d) : %d layers, features actives : %s",
            step,
            len(probe_output),
            {k: v.n_active for k, v in probe_output.items()},
        )

    def compute_drift(
        self,
        probe_output: dict[int, ProbeOutput],
        step: int,
    ) -> DriftReport:
        """
        Calcule le drift entre la baseline active et l'état courant.

        Raises:
            RuntimeError: Si register_reference() n'a pas été appelé.
        """
        if self._reference is None:
            raise RuntimeError("Appelle register_reference() avant compute_drift().")

        baseline = self._reference if self.comparison_mode == "reference" else self._previous
        if baseline is None:
            raise RuntimeError("Baseline manquante ; appelle register_reference() avant compute_drift().")

        features_lost: dict[int, list[int]] = {}
        features_gained: dict[int, list[int]] = {}
        cosine_distances: dict[int, float] = {}
        jaccard_distances: dict[int, float] = {}
        layer_scores: dict[int, float] = {}
        n_active: dict[int, int] = {}

        for layer_idx, ref in baseline.items():
            cur = probe_output.get(layer_idx)
            if cur is None:
                log.warning("Layer %d absent du probe courant", layer_idx)
                continue

            ref_set = set(ref.all_active_indices)
            cur_set = set(cur.all_active_indices)

            features_lost[layer_idx] = sorted(ref_set - cur_set)
            features_gained[layer_idx] = sorted(cur_set - ref_set)
            n_active[layer_idx] = cur.n_active

            union = len(ref_set | cur_set)
            intersection = len(ref_set & cur_set)
            jaccard_distances[layer_idx] = 1.0 - intersection / union if union > 0 else 0.0
            cosine_distances[layer_idx] = _safe_cosine_distance(ref.raw_activations, cur.raw_activations)
            layer_scores[layer_idx] = (
                self.jaccard_weight * jaccard_distances[layer_idx]
                + self.cosine_weight * cosine_distances[layer_idx]
            )

        if layer_scores:
            layers_ranked = sorted(layer_scores, key=lambda layer: layer_scores[layer], reverse=True)
            weights = np.asarray([float(self.layer_weights.get(layer, 1.0)) for layer in layers_ranked], dtype=np.float32)
            values = np.asarray([float(layer_scores[layer]) for layer in layers_ranked], dtype=np.float32)
            if float(weights.sum()) <= 0:
                weights = np.ones_like(values, dtype=np.float32)
            raw_score = float(np.average(values, weights=weights))
        else:
            layers_ranked = []
            raw_score = 0.0

        raw_score = max(0.0, min(1.0, raw_score))
        self._history.append(raw_score)
        drift_score = float(np.mean(self._history))
        self._max_drift = max(self._max_drift, drift_score)

        severity = _severity_from_score(drift_score)
        should_consult = drift_score > self.threshold

        summary = self._build_summary(
            step=step,
            drift_score=drift_score,
            raw_score=raw_score,
            lost=features_lost,
            gained=features_gained,
            cosine=cosine_distances,
            jaccard=jaccard_distances,
            layer_scores=layer_scores,
            layers_ranked=layers_ranked,
            n_active=n_active,
            should_consult=should_consult,
            severity=severity,
        )

        if self.comparison_mode == "previous":
            self._previous = probe_output

        log.debug(
            "step=%d drift=%.4f (raw=%.4f) severity=%s consult=%s",
            step,
            drift_score,
            raw_score,
            severity,
            should_consult,
        )

        return DriftReport(
            step=step,
            drift_score=drift_score,
            raw_drift_score=raw_score,
            should_consult_probe=should_consult,
            threshold=self.threshold,
            comparison_mode=self.comparison_mode,
            severity=severity,
            features_lost=features_lost,
            features_gained=features_gained,
            cosine_distances=cosine_distances,
            jaccard_distances=jaccard_distances,
            layer_scores=layer_scores,
            layers_ranked=layers_ranked,
            n_active_per_layer=n_active,
            n_layers_evaluated=len(layer_scores),
            summary=summary,
        )

    def reset(self) -> None:
        """Réinitialise le détecteur (référence + historique)."""
        self._reference = None
        self._previous = None
        self._history.clear()
        self._max_drift = 0.0

    def _build_summary(
        self,
        step: int,
        drift_score: float,
        raw_score: float,
        lost: dict[int, list[int]],
        gained: dict[int, list[int]],
        cosine: dict[int, float],
        jaccard: dict[int, float],
        layer_scores: dict[int, float],
        layers_ranked: list[int],
        n_active: dict[int, int],
        should_consult: bool,
        severity: str,
    ) -> str:
        status = "CONSULTER PROBE" if should_consult else "stable"
        lines = [
            f"[Step {step}] Drift score: {drift_score:.3f} (raw: {raw_score:.3f}, seuil: {self.threshold}) "
            f"— {status} | mode={self.comparison_mode} | severity={severity} | top_layers={layers_ranked[:3]}"
        ]
        for layer in layers_ranked:
            n = n_active.get(layer, 0)
            n_lost = len(lost.get(layer, []))
            n_gained = len(gained.get(layer, []))
            cos = cosine.get(layer, 0.0)
            jacc = jaccard.get(layer, 0.0)
            layer_score = layer_scores.get(layer, 0.0)
            lines.append(
                f"  Layer {layer}: score={layer_score:.3f} | {n} actives | -{n_lost} perdues | +{n_gained} nouvelles | "
                f"jaccard={jacc:.3f} | cosine={cos:.3f}"
            )
        return "\n".join(lines)
