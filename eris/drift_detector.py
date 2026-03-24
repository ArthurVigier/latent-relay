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


# ── Result dataclass ──────────────────────────────────────────────────────────

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
        features_lost:         {layer: [indices]} features actives au step 0
                               absentes maintenant.
        features_gained:       {layer: [indices]} features nouvelles
                               absentes au step 0.
        cosine_distances:      {layer: float} distance cosine activations brutes.
        jaccard_distances:     {layer: float} distance Jaccard features actives.
        n_active_per_layer:    {layer: int} features actives au step courant.
        summary:               Description lisible pour Claude.
    """
    step:                 int
    drift_score:          float
    raw_drift_score:      float
    should_consult_probe: bool
    threshold:            float
    features_lost:        dict[int, list[int]]   = field(default_factory=dict)
    features_gained:      dict[int, list[int]]   = field(default_factory=dict)
    cosine_distances:     dict[int, float]        = field(default_factory=dict)
    jaccard_distances:    dict[int, float]        = field(default_factory=dict)
    n_active_per_layer:   dict[int, int]          = field(default_factory=dict)
    summary:              str                     = ""


# ── DriftDetector ─────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Détecteur de drift sur features SAE.

    Args:
        threshold:       Seuil de drift_score pour déclencher une consultation.
        window:          Taille de la fenêtre de lissage (moyenne mobile).
        jaccard_weight:  Poids de la distance Jaccard (drift thématique).
        cosine_weight:   Poids de la distance cosine (drift géométrique).
    """

    def __init__(
        self,
        threshold: float = 0.35,
        window: int = 3,
        jaccard_weight: float = 0.6,
        cosine_weight: float = 0.4,
    ) -> None:
        if abs(jaccard_weight + cosine_weight - 1.0) > 1e-6:
            raise ValueError("jaccard_weight + cosine_weight doit être égal à 1.0")
        self.threshold      = threshold
        self.window         = window
        self.jaccard_weight = jaccard_weight
        self.cosine_weight  = cosine_weight

        self._reference: Optional[dict[int, ProbeOutput]] = None
        self._history:   deque = deque(maxlen=window)
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
        Calcule le drift entre la référence et l'état courant.

        Args:
            probe_output: Résultat de SAEProbe.probe() sur le contexte courant.
            step:         Index du step de raisonnement.

        Returns:
            DriftReport avec métriques par layer et score global.

        Raises:
            RuntimeError: Si register_reference() n'a pas été appelé.
        """
        if self._reference is None:
            raise RuntimeError(
                "Appelle register_reference() avant compute_drift()."
            )

        features_lost:    dict[int, list[int]]  = {}
        features_gained:  dict[int, list[int]]  = {}
        cosine_distances: dict[int, float]       = {}
        jaccard_distances: dict[int, float]      = {}
        n_active:         dict[int, int]         = {}

        for layer_idx, ref in self._reference.items():
            cur = probe_output.get(layer_idx)
            if cur is None:
                log.warning("Layer %d absent du probe courant", layer_idx)
                continue

            ref_set = set(ref.all_active_indices)
            cur_set = set(cur.all_active_indices)

            features_lost[layer_idx]   = sorted(ref_set - cur_set)
            features_gained[layer_idx] = sorted(cur_set - ref_set)
            n_active[layer_idx]        = cur.n_active

            # Distance Jaccard sur les features actives
            union        = len(ref_set | cur_set)
            intersection = len(ref_set & cur_set)
            jaccard_distances[layer_idx] = (
                1.0 - intersection / union if union > 0 else 0.0
            )

            # Distance cosine sur les activations brutes
            ref_vec = ref.raw_activations.astype(np.float32)
            cur_vec = cur.raw_activations.astype(np.float32)
            norm    = float(np.linalg.norm(ref_vec) * np.linalg.norm(cur_vec))
            cosine_distances[layer_idx] = (
                float(1.0 - np.dot(ref_vec, cur_vec) / norm)
                if norm > 0 else 0.0
            )

        # Score global — moyenne pondérée sur les layers
        if jaccard_distances:
            mean_jaccard = float(np.mean(list(jaccard_distances.values())))
            mean_cosine  = float(np.mean(list(cosine_distances.values())))
            raw_score    = (
                self.jaccard_weight * mean_jaccard
                + self.cosine_weight * mean_cosine
            )
        else:
            raw_score = 0.0

        raw_score = max(0.0, min(1.0, raw_score))

        # Moyenne mobile (lissage sur la fenêtre)
        self._history.append(raw_score)
        drift_score = float(np.mean(self._history))
        self._max_drift = max(self._max_drift, drift_score)

        should_consult = drift_score > self.threshold

        summary = self._build_summary(
            step, drift_score, raw_score,
            features_lost, features_gained, cosine_distances, n_active,
            should_consult,
        )

        log.debug(
            "step=%d drift=%.4f (raw=%.4f) consult=%s",
            step, drift_score, raw_score, should_consult,
        )

        return DriftReport(
            step=step,
            drift_score=drift_score,
            raw_drift_score=raw_score,
            should_consult_probe=should_consult,
            threshold=self.threshold,
            features_lost=features_lost,
            features_gained=features_gained,
            cosine_distances=cosine_distances,
            jaccard_distances=jaccard_distances,
            n_active_per_layer=n_active,
            summary=summary,
        )

    def reset(self) -> None:
        """Réinitialise le détecteur (référence + historique)."""
        self._reference = None
        self._history.clear()
        self._max_drift = 0.0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        step: int,
        drift_score: float,
        raw_score: float,
        lost: dict[int, list[int]],
        gained: dict[int, list[int]],
        cosine: dict[int, float],
        n_active: dict[int, int],
        should_consult: bool,
    ) -> str:
        status = "CONSULTER PROBE" if should_consult else "stable"
        lines = [
            f"[Step {step}] Drift score: {drift_score:.3f} "
            f"(raw: {raw_score:.3f}, seuil: {self.threshold}) — {status}",
        ]
        for layer in sorted(lost.keys()):
            n = n_active.get(layer, 0)
            n_lost   = len(lost.get(layer, []))
            n_gained = len(gained.get(layer, []))
            cos      = cosine.get(layer, 0.0)
            lines.append(
                f"  Layer {layer}: {n} actives | "
                f"-{n_lost} perdues | +{n_gained} nouvelles | "
                f"cosine={cos:.3f}"
            )
        return "\n".join(lines)
