"""
scripts/validate_sae_on_aime.py
================================

KILL GATE — Étape 0 du reframe V2.

Vérifie que les SAEs Gemma Scope 2 (Gemma 3 9B IT) produisent des features
cohérentes et non-triviales sur du raisonnement mathématique AIME.

Critère d'arrêt : si mean_active_features < 5 ou > 500, la distribution
shift est trop sévère — les SAEs (entraînés sur du texte général) ne
sont pas utilisables sur du raisonnement dense. Stop total.

Stack :
    Modèle  : google/gemma-3-9b-it
    SAEs    : Gemma Scope 2 — gemma-scope-2-9b-it-res
    SAE ID  : layer_20_width_16k_l0_medium
    Device  : cuda (A100 80GB recommandé)

Usage :
    python scripts/validate_sae_on_aime.py [--layer 20] [--width 16k] [--l0 medium]

Exit code : 0 = PASS (continuer), 1 = KILL (stop total)
"""

from __future__ import annotations

import argparse
import sys
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validate_sae")

# ── Gemma Scope 2 — IDs vérifiés sur huggingface.co/google/gemma-scope ────────
MODEL_ID    = "google/gemma-3-9b-it"
SAE_RELEASE = "gemma-scope-2-9b-it-res"    # Gemma Scope 2 pour Gemma 3 9B IT

AIME_SAMPLES = [
    "Find the number of positive integers less than 1000 that are divisible "
    "by neither 2, 3, nor 5.",

    "Let S be the set of all polynomials of the form z^3 + az^2 + bz + c, "
    "where a, b, and c are integers. Find the number of such polynomials "
    "such that all roots of the polynomial are real and between -1 and 1.",

    "In a sequence of coin tosses, one can keep a record of instances in "
    "which a tail is immediately followed by a head, a head is immediately "
    "followed by a head, and etc. We denote these by TH, HH, and so on. "
    "How many ways can we arrange heads and tails in 10 coin tosses?",
]

# Kill criterion
MIN_ACTIVE = 5
MAX_ACTIVE = 500


def validate(layer: int, width: str, l0: str) -> bool:
    """
    Returns True if SAEs produce sensible features on AIME inputs.
    Returns False (KILL) if feature count is outside [MIN_ACTIVE, MAX_ACTIVE].
    """
    log.info("Loading model: %s", MODEL_ID)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sae_lens import SAE

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load SAE for the target layer.
    # Format Gemma Scope 2 : layer_{n}_width_{w}_l0_{size}
    sae_id = f"layer_{layer}_width_{width}_l0_{l0}"
    log.info("Loading SAE: release=%s sae_id=%s", SAE_RELEASE, sae_id)
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
        device="cuda",
    )
    log.info("SAE config: %s", cfg_dict)
    if sparsity is not None:
        log.info("SAE expected sparsity (L0): %s", sparsity)

    n_active_per_sample: list[int] = []

    for i, text in enumerate(AIME_SAMPLES):
        # Capture residual stream via hook.
        captured: dict[str, torch.Tensor] = {}

        def hook_fn(module, input, output):
            # output is a tuple; [0] is the hidden state [batch, seq, hidden_dim]
            captured["acts"] = output[0].detach()

        handle = model.model.layers[layer].register_forward_hook(hook_fn)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        # Last-token activation — [hidden_dim]
        last_token = captured["acts"][0, -1, :]  # [hidden_dim]

        # Encode via SAE.
        # SAE.encode() accepts [hidden_dim] or [batch, hidden_dim].
        features = sae.encode(last_token.unsqueeze(0))  # → [1, n_features]
        features = features[0]                           # → [n_features]

        active_mask = features > 0
        n_active = int(active_mask.sum().item())
        top_k = min(10, n_active)
        topk = features.topk(top_k) if top_k > 0 else None

        log.info(
            "[Sample %d] active features: %d / %d",
            i + 1, n_active, features.shape[0],
        )
        if topk is not None:
            log.info("  top-10 indices: %s", topk.indices.tolist())
            log.info("  top-10 values:  %s", [round(v, 3) for v in topk.values.tolist()])

        n_active_per_sample.append(n_active)

    mean_active = sum(n_active_per_sample) / len(n_active_per_sample)
    log.info("Mean active features across %d samples: %.1f", len(AIME_SAMPLES), mean_active)

    # Kill criterion
    if mean_active < MIN_ACTIVE:
        log.error(
            "KILL — mean_active_features=%.1f < %d. "
            "SAEs trop creux sur ce domaine. Distribution shift sévère.",
            mean_active, MIN_ACTIVE,
        )
        return False

    if mean_active > MAX_ACTIVE:
        log.error(
            "KILL — mean_active_features=%.1f > %d. "
            "SAEs trop denses — pas de représentation sparse utile.",
            mean_active, MAX_ACTIVE,
        )
        return False

    log.info(
        "PASS — mean_active_features=%.1f dans [%d, %d]. "
        "SAEs utiles sur AIME. Continuer vers eris/sae_probe.py.",
        mean_active, MIN_ACTIVE, MAX_ACTIVE,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Kill gate — valider SAEs sur AIME")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer Gemma 3 9B à sonder (défaut: 20, ~milieu du modèle)")
    parser.add_argument("--width", default="16k",
                        help="Largeur SAE (16k, 64k, 256k) — défaut: 16k")
    parser.add_argument("--l0", default="medium",
                        help="Niveau de sparsité SAE (small, medium, big) — défaut: medium")
    args = parser.parse_args()

    ok = validate(layer=args.layer, width=args.width, l0=args.l0)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
