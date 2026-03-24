"""
eris/experiments/drift_detection/kill_criteria.py
==================================================

Kill-gate criteria pour le pipeline de drift detection ERIS V2.

Chaque critère définit :
    metric      : nom de la métrique mesurée
    threshold   : valeur limite
    direction   : "greater" (> threshold) ou "less" (< threshold) pour passer
    description : ce qu'on mesure et pourquoi
    action_if_failed / action_if_passed : suite à donner

Ordre d'exécution strict :
    sae_validation → test_0 → test_1 → test_2 → test_3_scaling
    Ne pas passer au suivant si un critère échoue.

Usage::

    from eris.experiments.drift_detection.kill_criteria import KILL_CRITERIA, check_criterion

    passed, msg = check_criterion("test_0", rho_value)
    if not passed:
        print(msg)
        sys.exit(1)
"""

from __future__ import annotations

KILL_CRITERIA: dict[str, dict] = {

    "sae_validation": {
        "metric":           "mean_active_features",
        "threshold_min":    5,
        "threshold_max":    500,
        "direction":        "range",   # passe si threshold_min < valeur < threshold_max
        "description": (
            "Les SAEs Gemma Scope 2 doivent activer entre 5 et 500 features "
            "sur du raisonnement AIME. En dehors de cette plage, les SAEs "
            "sont soit trop creux (inutiles) soit trop denses (pas sparse)."
        ),
        "action_if_failed": (
            "STOP TOTAL — distribution shift SAE trop sévère. "
            "Les SAEs ne peuvent pas fournir de signal utile sur ce domaine. "
            "Options : (1) essayer une autre layer, (2) essayer 64k au lieu de 16k, "
            "(3) abandonner Gemma Scope sur ce domaine."
        ),
        "action_if_passed": "Continuer vers test_0 (corrélation drift/erreur).",
        "script":           "scripts/validate_sae_on_aime.py",
        "estimated_cost":   "$0 — GPU local seulement",
        "estimated_time":   "~5 min sur A100",
    },

    "test_0": {
        "metric":           "spearman_rho_drift_vs_error",
        "threshold":        0.35,
        "direction":        "greater",
        "description": (
            "Le drift SAE doit prédire les erreurs finales de raisonnement. "
            "Spearman ρ entre drift_score final (Jaccard + cosine sur features SAE) "
            "et erreur binaire (0=correct, 1=faux) sur 20 problèmes AIME. "
            "Probe = Gemma 3 9B + Gemma Scope 2. "
            "Modèle principal = Qwen3-14B."
        ),
        "action_if_failed": (
            "STOP — reframe entier invalide. "
            "Le drift SAE ne corrèle pas avec les erreurs de raisonnement. "
            "Avant de réessayer : (1) vérifier la sélection de layer, "
            "(2) essayer un dataset plus difficile, "
            "(3) vérifier que Qwen3-14B fait bien des erreurs."
        ),
        "action_if_passed": "Continuer vers test_1 (AUC probe divergence).",
        "script":           "eris/experiments/drift_detection/test_0_drift_characterization.py",
        "estimated_cost":   "$0 — GPU local, pas d'API",
        "estimated_time":   "~2h sur A100 (20 problèmes × 8 étapes de raisonnement)",
    },

    "test_1": {
        "metric":           "probe_jaccard_auc",
        "threshold":        0.60,
        "direction":        "greater",
        "description": (
            "La divergence de features SAE (Jaccard) doit classifier les runs "
            "corrects/incorrects mieux qu'un baseline naïf. "
            "AUC du score Jaccard en tant que classifieur binaire."
        ),
        "action_if_failed": (
            "STOP — le probe SAE n'apporte pas de signal au-delà du drift detector seul. "
            "La référence externe (zombie) ne distingue pas les types d'erreur. "
            "Options : autre modèle zombie, autres layers, autre pooling."
        ),
        "action_if_passed": "Continuer vers test_2 (intervention active).",
        "script":           "eris/experiments/drift_detection/test_1_probe_detection.py",
        "estimated_cost":   "~$5 (Claude API, 20 problèmes)",
        "estimated_time":   "~3h",
    },

    "test_2": {
        "metric":           "accuracy_delta_with_intervention",
        "threshold":        0.05,
        "direction":        "greater",
        "description": (
            "La consultation active du probe SAE doit améliorer l'accuracy "
            "de Qwen3-14B d'au moins 5 points de pourcentage vs baseline "
            "sans consultation."
        ),
        "action_if_failed": (
            "PIVOT — la détection de drift est réelle, mais l'intervention "
            "(la façon dont Claude utilise les features SAE) n'est pas efficace. "
            "Ne pas abandonner la direction — retravailler _format_drift_for_claude()."
        ),
        "action_if_passed": (
            "Phase 2 validée. Continuer vers test_3_scaling (Gemma 3 27B)."
        ),
        "script":           "eris/experiments/drift_detection/test_2_intervention.py",
        "estimated_cost":   "~$20 (Claude API avec boucle de consultation)",
        "estimated_time":   "~4h",
    },

    "test_3_scaling": {
        "metric":           "auc_delta_27b_vs_9b",
        "threshold":        0.05,
        "direction":        "greater",
        "description": (
            "Gemma 3 27B + Gemma Scope 2 27B doit apporter au moins 5pp d'AUC "
            "probe supplémentaires par rapport à Gemma 3 9B. "
            "Justifie le coût H100 80GB (~$3-4/h RunPod)."
        ),
        "action_if_failed": (
            "STOP scaling — Gemma 3 9B est suffisant, pas de gain à 27B. "
            "Rester sur la stack 9B pour la suite."
        ),
        "action_if_passed": "Scaling validé. Stack 27B recommandée pour production.",
        "script":           "eris/experiments/drift_detection/test_3_scaling_27b.py",
        "hardware":         "H100 80GB (~$3-4/h RunPod)",
        "estimated_cost":   "~$15-20 GPU + ~$20 Claude API",
        "estimated_time":   "~5h",
    },
}


def check_criterion(test_id: str, metric_value: float) -> tuple[bool, str]:
    """
    Vérifie si une valeur mesurée passe le critère kill-gate.

    Args:
        test_id:       Identifiant du test (voir KILL_CRITERIA).
        metric_value:  Valeur de la métrique mesurée.

    Returns:
        (passed: bool, message: str)

    Raises:
        KeyError: Si test_id est inconnu.
    """
    if test_id not in KILL_CRITERIA:
        raise KeyError(
            f"test_id inconnu: {test_id!r}. Valides: {list(KILL_CRITERIA)}"
        )

    c = KILL_CRITERIA[test_id]

    if c["direction"] == "greater":
        passed = metric_value > c["threshold"]
        op     = ">"
        ref    = c["threshold"]
    elif c["direction"] == "less":
        passed = metric_value < c["threshold"]
        op     = "<"
        ref    = c["threshold"]
    elif c["direction"] == "range":
        passed = c["threshold_min"] < metric_value < c["threshold_max"]
        op     = f"dans [{c['threshold_min']}, {c['threshold_max']}]"
        ref    = None
    else:
        raise ValueError(f"direction inconnue: {c['direction']!r}")

    if passed:
        action = c.get("action_if_passed", "Continuer.")
        status = "PASS"
    else:
        action = c.get("action_if_failed", "STOP.")
        status = "KILL"

    ref_str = f" {op} {ref}" if ref is not None else f" {op}"
    message = (
        f"[{status}] {test_id}: {c['metric']}={metric_value:.4f}{ref_str}\n"
        f"  {c['description']}\n"
        f"  → {action}"
    )

    return passed, message
