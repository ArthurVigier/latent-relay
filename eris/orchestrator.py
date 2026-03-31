"""
eris/orchestrator.py
=====================

Orchestrateur ERIS V2 — Claude comme interprète d'espaces latents SAE.

Paradigme V2 :
    Claude raisonne. SAEProbe observe en parallèle via Gemma Scope 2.
    DriftDetector mesure la divergence de features sémantiques.
    Si drift > seuil → Claude reçoit une description lisible des features
    qui ont disparu/apparu → il recalibre son raisonnement.

Différence clé avec V1 :
    V1 : activations brutes numpy → Claude devine dans le vide
    V2 : features SAE sparse + nommables → Claude lit des concepts

Usage::

    from eris.sae_probe import SAEProbe
    from eris.drift_detector import DriftDetector
    from eris.orchestrator import ERISOrchestrator
    from eris.backends.orchestrators.claude_orchestrator import ClaudeOrchestrator

    probe    = SAEProbe("google/gemma-3-9b-it", layers=[10, 20, 30])
    detector = DriftDetector(threshold=0.35, window=3)
    llm      = ClaudeOrchestrator()

    orch = ERISOrchestrator(probe, detector, llm)
    result = orch.run("Prove that there are infinitely many primes.", max_steps=15)
    print(result.final_answer)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from eris.drift_detector import DriftDetector, DriftReport
from eris.interfaces import OrchestratorLLM, ReasoningStep
from eris.sae_probe import SAEProbe

log = logging.getLogger("eris.orchestrator")

_RECALIBRATION_TEMPLATE = """\
[Observation de la sonde latente — Step {step}]

Le modèle de référence (Gemma 3 + SAEs Gemma Scope 2) a traité le même
contexte que toi. Voici ce que ses features internes révèlent :

{drift_description}

Signification : les features "perdues" représentent des concepts que le
modèle de référence n'active plus sur le contexte courant. Les features
"nouvelles" représentent des concepts qui sont apparus.

Ce n'est pas une correction — c'est un signal de référence externe.
Utilise-le si tu le trouves informatif, ignore-le sinon.

[Fin observation — reprends ton raisonnement]
"""


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class ConsultationRecord:
    """Enregistrement d'une consultation probe."""
    step:                int
    drift_score:         float
    n_features_lost:     dict[int, int]   # {layer: count}
    n_features_gained:   dict[int, int]   # {layer: count}
    lost_concepts:       list[str]
    gained_concepts:     list[str]
    observation_text:    str              # envoyé à Claude
    response_preview:    str              # 200 premiers chars de la réponse Claude
    elapsed_s:           float


@dataclass
class OrchestratorResult:
    """Résultat complet d'un run ERISOrchestrator."""
    problem:             str
    final_answer:        str
    n_steps:             int
    n_consultations:     int
    consultations:       list[ConsultationRecord]
    reasoning_log:       list[dict]       # {step, text[:300], drift_score, consulted}
    max_drift:           Optional[float]
    elapsed_s:           float


# ── Orchestrateur ──────────────────────────────────────────────────────────────

class ERISOrchestrator:
    """
    Connecte un OrchestratorLLM (raisonneur principal) avec un SAEProbe
    (outil de représentation) via un DriftDetector.

    Args:
        probe:           SAEProbe chargé.
        drift_detector:  DriftDetector (réinitialisé à chaque run()).
        llm:             OrchestratorLLM (ClaudeOrchestrator, GeminiOrchestrator, etc.)
        probe_layers:    Layers à utiliser pour le drift (None = tous les layers du probe).
    """

    def __init__(
        self,
        probe: SAEProbe,
        drift_detector: DriftDetector,
        llm: OrchestratorLLM,
        probe_layers: Optional[list[int]] = None,
    ) -> None:
        self.probe    = probe
        self.detector = drift_detector
        self.llm      = llm
        self.layers   = probe_layers or probe.layers

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        problem: str,
        *,
        max_steps: int = 15,
        checkpoint_every: int = 3,
        top_k: int = 20,
    ) -> OrchestratorResult:
        """
        Orchestre un run complet avec recalibration probe-guidée.

        Args:
            problem:          Problème à résoudre.
            max_steps:        Nombre maximum d'étapes de raisonnement.
            checkpoint_every: Extraire les features SAE toutes les N étapes.
            top_k:            Features top-K retournées par SAEProbe.

        Returns:
            OrchestratorResult avec log complet.
        """
        t0 = time.time()
        self.detector.reset()

        # Référence : probe sur l'énoncé nu
        ref_output = self.probe.probe(problem, top_k=top_k)
        self.detector.register_reference(ref_output, step=0)
        log.info("Référence SAE enregistrée. Run (max_steps=%d).", max_steps)

        history:              list[ReasoningStep] = []
        reasoning_log:        list[dict]          = []
        consultations:        list[ConsultationRecord] = []
        final_answer:         str                 = ""
        recalibration_context: Optional[str]      = None

        for step in range(1, max_steps + 1):
            # ── Étape de raisonnement ─────────────────────────────────────
            step_t0 = time.time()
            try:
                rs = self.llm.reason_step(
                    problem=problem,
                    history=history,
                    recalibration_context=recalibration_context,
                )
                recalibration_context = None
            except Exception as e:
                log.error("LLM error à step %d : %s", step, e)
                break

            step_text = rs.content
            history.append(rs)
            log.info("Step %d : %d chars, %.2fs", step, len(step_text), time.time() - step_t0)

            if "[Final Answer]" in step_text or step == max_steps:
                final_answer = step_text
                reasoning_log.append({
                    "step": step, "text": step_text,
                    "drift_score": None, "consulted": False,
                })
                break

            # ── Checkpoint drift ──────────────────────────────────────────
            drift_score: Optional[float] = None
            consulted = False

            if step % checkpoint_every == 0:
                # Probe sur le contexte accumulé
                context_text = problem + "\n" + "\n".join(s.content for s in history)
                cur_output   = self.probe.probe(context_text[-4096:], top_k=top_k)
                report       = self.detector.compute_drift(cur_output, step=step)
                drift_score  = report.drift_score

                if report.should_consult_probe:
                    consulted = True
                    ct0 = time.time()
                    log.info(
                        "Consultation probe déclenchée : drift=%.4f > seuil=%.4f",
                        report.drift_score, report.threshold,
                    )

                    drift_desc = self._format_drift_for_claude(report)
                    recalibration_context = _RECALIBRATION_TEMPLATE.format(
                        step=step,
                        drift_description=drift_desc,
                    )

                    consultations.append(ConsultationRecord(
                        step=step,
                        drift_score=report.drift_score,
                        n_features_lost={
                            k: len(v) for k, v in report.features_lost.items()
                        },
                        n_features_gained={
                            k: len(v) for k, v in report.features_gained.items()
                        },
                        lost_concepts=self._extract_labeled_concepts(
                            report.features_lost,
                            getattr(report, "feature_labels", {}),
                        ),
                        gained_concepts=self._extract_labeled_concepts(
                            report.features_gained,
                            getattr(report, "feature_labels", {}),
                        ),
                        observation_text=drift_desc,
                        response_preview="",    # rempli à la prochaine itération
                        elapsed_s=round(time.time() - ct0, 3),
                    ))

            reasoning_log.append({
                "step":        step,
                "text":        step_text[:300],
                "drift_score": round(drift_score, 5) if drift_score is not None else None,
                "consulted":   consulted,
            })

        return OrchestratorResult(
            problem=problem,
            final_answer=final_answer,
            n_steps=len(reasoning_log),
            n_consultations=len(consultations),
            consultations=consultations,
            reasoning_log=reasoning_log,
            max_drift=self.detector.max_drift,
            elapsed_s=round(time.time() - t0, 2),
        )

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_drift_for_claude(self, report: DriftReport) -> str:
        """
        Convertit un DriftReport V2 en description lisible.

        Claude reçoit des indices de features avec leurs valeurs —
        pas de nombres bruts d'activations. À terme, les indices
        peuvent être liés aux labels Neuronpedia pour plus de sémantique.
        """
        ranked_layers = report.layers_ranked or sorted(report.features_lost.keys())
        lines = [
            f"Drift score global : {report.drift_score:.3f} (seuil : {report.threshold})",
            f"Severity : {getattr(report, 'severity', 'unknown')}",
            f"Comparison mode : {getattr(report, 'comparison_mode', 'reference')}",
            f"Top drifting layers : {ranked_layers[:3]}",
            "",
            "Par couche :",
        ]

        for layer in ranked_layers:
            lost = report.features_lost.get(layer, [])
            gained = report.features_gained.get(layer, [])
            n_act = report.n_active_per_layer.get(layer, 0)
            cos = report.cosine_distances.get(layer, 0.0)
            jacc = report.jaccard_distances.get(layer, 0.0)
            layer_score = getattr(report, "layer_scores", {}).get(layer)
            label_map = getattr(report, "feature_labels", {}).get(layer, {})
            score_text = f"score={layer_score:.3f} | " if layer_score is not None else ""

            lines.append(
                f"  Couche {layer} : {score_text}{n_act} features actives | "
                f"cosine={cos:.3f} | jaccard={jacc:.3f}"
            )
            if lost[:8]:
                lines.append(
                    f"    Features disparues (top 8 indices) : {self._format_feature_refs(lost[:8], label_map)}"
                )
            if gained[:8]:
                lines.append(
                    f"    Features nouvelles (top 8 indices) : {self._format_feature_refs(gained[:8], label_map)}"
                )

        return "\n".join(lines)

    @staticmethod
    def _format_feature_refs(indices: list[int], label_map: dict[int, str | None]) -> list[str]:
        refs: list[str] = []
        for idx in indices:
            label = label_map.get(idx)
            refs.append(f"{idx}:{label}" if label else str(idx))
        return refs

    @staticmethod
    def _extract_labeled_concepts(
        features: dict[int, list[int]],
        feature_labels: dict[int, dict[int, str | None]],
    ) -> list[str]:
        refs: list[str] = []
        for layer, indices in features.items():
            label_map = feature_labels.get(layer, {})
            for idx in indices:
                label = label_map.get(idx)
                if label:
                    refs.append(f"{layer}:{idx}:{label}")
        return refs
