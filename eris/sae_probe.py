"""
eris/sae_probe.py
==================

Sonde SAE basée sur Gemma Scope 2 (Gemma 3).

Rôle dans ERIS V2 : transformer des inputs textuels en features SAE sparse
et interprétables. Pas de génération. Pas de dialogue. Pas d'accès web.

    input texte → forward pass Gemma 3 → activations → SAE.encode()
                → features sparse [16K ou 64K] → top-K actives avec valeurs

Le zombie ne parle pas. Il encode. C'est tout.

Stack supportée :
    Tests de principe : google/gemma-3-9b-it + gemma-scope-2-9b-it-res (A100 80GB)
    Scaling          : google/gemma-3-27b-it + gemma-scope-2-27b-it-res (H100 80GB)

Release IDs Gemma Scope 2 vérifiés sur huggingface.co/google/gemma-scope.
Format SAE ID : layer_{n}_width_{w}_l0_{size}  (underscores — pas de slashes)

Usage::

    probe = SAEProbe("google/gemma-3-9b-it", layers=[10, 20, 30])
    out = probe.probe("Find all integers n such that n^2 + 1 is divisible by 5.")
    # out[20].active_feature_indices → [412, 891, 3201, ...]
    # out[20].n_active → 47
    # out[20].raw_activations → np.ndarray[hidden_dim] pour DriftDetector
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("eris.sae_probe")

# ── Gemma Scope 2 — release IDs vérifiés ────────────────────────────────────
# Source : huggingface.co/google/gemma-scope + sae-lens pretrained_saes.yaml
_SUPPORTED_RELEASES: dict[str, str] = {
    "gemma-3-9b-it":  "gemma-scope-2-9b-it-res",
    "gemma-3-27b-it": "gemma-scope-2-27b-it-res",
}

# Largeurs SAE disponibles (Gemma Scope 2)
_VALID_WIDTHS = {"16k", "64k", "256k", "1m"}
_VALID_L0     = {"small", "medium", "big"}


@dataclass
class ProbeOutput:
    """
    Résultat d'une sonde SAE pour une couche donnée.

    Attributes:
        layer:                  Index de la couche sondée.
        active_feature_indices: Indices des features actives (top-K).
        active_feature_values:  Valeurs correspondantes (ordre décroissant).
        all_active_indices:     Tous les indices actifs (non limités à top-K).
        n_active:               Nombre total de features actives.
        raw_activations:        Activations brutes [hidden_dim] — pour DriftDetector.
        elapsed_s:              Temps de forward pass.
    """
    layer:                  int
    active_feature_indices: list[int]
    active_feature_values:  list[float]
    all_active_indices:     list[int]
    n_active:               int
    raw_activations:        np.ndarray
    elapsed_s:              float


class SAEProbe:
    """
    Zombie = Gemma 3 9B/27B + SAEs Gemma Scope 2.

    max_new_tokens = 0 est structurel : on appelle model() pas model.generate().
    enable_thinking = False : pas de blocs <think>.

    Args:
        model_id:   HuggingFace model ID. Doit être dans SUPPORTED_RELEASES.
        layers:     Couches à sonder. Défaut : [10, 20, 30] pour Gemma 3 9B.
        sae_width:  Largeur du dictionnaire SAE (16k, 64k, 256k, 1m).
        l0:         Niveau de sparsité SAE (small, medium, big).
        device:     Device torch.
    """

    SUPPORTED_RELEASES = _SUPPORTED_RELEASES

    def __init__(
        self,
        model_id: str = "google/gemma-3-9b-it",
        layers: list[int] = [10, 20, 30],
        sae_width: str = "16k",
        l0: str = "medium",
        device: str = "cuda",
    ) -> None:
        if sae_width not in _VALID_WIDTHS:
            raise ValueError(f"sae_width doit être dans {_VALID_WIDTHS}, reçu: {sae_width!r}")
        if l0 not in _VALID_L0:
            raise ValueError(f"l0 doit être dans {_VALID_L0}, reçu: {l0!r}")

        self.model_id  = model_id
        self.layers    = layers
        self.sae_width = sae_width
        self.l0        = l0
        self.device    = device if torch.cuda.is_available() else "cpu"

        # Résoudre le release ID
        short_id = model_id.split("/")[-1]
        release = _SUPPORTED_RELEASES.get(short_id)
        if release is None:
            raise ValueError(
                f"Modèle {model_id!r} non supporté. "
                f"Disponibles : {list(_SUPPORTED_RELEASES.keys())}\n"
                "Pour ajouter un nouveau modèle, appelle SAEProbe.list_available_releases()."
            )
        self._release = release

        log.info("Chargement du modèle : %s sur %s", model_id, self.device)
        t0 = time.time()
        self._model, self._tokenizer = self._load_model()
        log.info("Modèle prêt en %.1fs", time.time() - t0)

        log.info("Chargement des SAEs pour les layers %s…", layers)
        t1 = time.time()
        self._saes: dict[int, object] = self._load_saes()
        log.info("SAEs prêts en %.1fs", time.time() - t1)

    # ── API publique ──────────────────────────────────────────────────────────

    def probe(
        self,
        input_text: str,
        top_k: int = 20,
    ) -> dict[int, ProbeOutput]:
        """
        Extrait les features SAE actives pour chaque layer configuré.

        Aucune génération. Uniquement forward pass + SAE.encode().

        Args:
            input_text: Texte à encoder.
            top_k:      Nombre de features top-K à retourner dans active_feature_indices.
                        all_active_indices contient TOUTES les features actives.

        Returns:
            {layer_idx: ProbeOutput}
        """
        t0 = time.time()
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Hooks pour capturer les activations au last token.
        raw_acts: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in self.layers:
            def _make_hook(idx: int):
                def _hook(module, inp, output):
                    # output[0] : [batch, seq, hidden_dim]
                    raw_acts[idx] = output[0][:, -1, :].detach()  # [batch, hidden_dim]
                return _hook
            h = self._model.model.layers[layer_idx].register_forward_hook(_make_hook(layer_idx))
            handles.append(h)

        try:
            with torch.no_grad():
                self._model(**inputs)
        finally:
            for h in handles:
                h.remove()

        elapsed = time.time() - t0

        # Encoder via SAE
        results: dict[int, ProbeOutput] = {}
        for layer_idx, sae in self._saes.items():
            acts = raw_acts.get(layer_idx)
            if acts is None:
                log.warning("Pas d'activations pour layer %d", layer_idx)
                continue

            # acts : [1, hidden_dim] → sae.encode → [1, n_features]
            features = sae.encode(acts)[0]  # [n_features]

            active_mask    = features > 0
            all_active_idx = active_mask.nonzero(as_tuple=True)[0].tolist()
            n_active       = len(all_active_idx)

            # Top-K pour le résumé (limité si moins de k features actives)
            k = min(top_k, n_active)
            if k > 0:
                topk = features.topk(k)
                top_indices = topk.indices.tolist()
                top_values  = [round(float(v), 4) for v in topk.values.tolist()]
            else:
                top_indices = []
                top_values  = []

            results[layer_idx] = ProbeOutput(
                layer=layer_idx,
                active_feature_indices=top_indices,
                active_feature_values=top_values,
                all_active_indices=all_active_idx,
                n_active=n_active,
                raw_activations=acts.squeeze(0).float().cpu().numpy(),
                elapsed_s=round(elapsed, 4),
            )
            log.debug(
                "layer=%d n_active=%d top5=%s",
                layer_idx, n_active, top_indices[:5],
            )

        log.info(
            "probe: layers=%s n_active=%s elapsed=%.3fs",
            list(results.keys()),
            {k: v.n_active for k, v in results.items()},
            elapsed,
        )
        return results

    def probe_batch(
        self,
        inputs: list[str],
        top_k: int = 20,
    ) -> list[dict[int, ProbeOutput]]:
        """Extrait les features SAE pour un batch d'inputs (séquentiellement)."""
        return [self.probe(text, top_k=top_k) for text in inputs]

    # ── Discovery ─────────────────────────────────────────────────────────────

    @staticmethod
    def list_available_releases() -> None:
        """
        Affiche les releases Gemma Scope disponibles via sae-lens.

        Utile pour vérifier les IDs avant de les hardcoder.
        Nécessite sae-lens installé.
        """
        try:
            from sae_lens import pretrained_saes
            releases = pretrained_saes.get_pretrained_saes_directory()
            gemma = {k: v for k, v in releases.items() if "gemma" in k.lower()}
            for release_id, info in sorted(gemma.items()):
                print(f"\n{release_id}")
                if hasattr(info, "saes_map"):
                    for sae_id in list(info.saes_map.keys())[:5]:
                        print(f"  {sae_id}")
                    if len(info.saes_map) > 5:
                        print(f"  … ({len(info.saes_map)} total)")
        except Exception as e:
            print(f"Erreur: {e}")
            print("Vérifie sur : https://jbloomaus.github.io/SAELens/sae_table/")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        model.eval()

        # Désactiver thinking si supporté
        if hasattr(model, "generation_config"):
            gc = model.generation_config
            if hasattr(gc, "enable_thinking"):
                gc.enable_thinking = False
            if hasattr(gc, "thinking_budget"):
                gc.thinking_budget = 0

        return model, tokenizer

    def _load_saes(self) -> dict[int, object]:
        """Charge un SAE par layer depuis Gemma Scope 2."""
        try:
            from sae_lens import SAE
        except ImportError:
            raise ImportError(
                "sae-lens requis : pip install sae-lens\n"
                "transformer-lens >= 3.0.0 requis : pip install transformer-lens>=3.0.0b0"
            )

        saes: dict[int, object] = {}
        for layer in self.layers:
            # Format Gemma Scope 2 (underscores)
            sae_id = f"layer_{layer}_width_{self.sae_width}_l0_{self.l0}"
            log.info("  Chargement SAE layer %d : %s / %s", layer, self._release, sae_id)
            sae, cfg_dict, _ = SAE.from_pretrained(
                release=self._release,
                sae_id=sae_id,
                device=self.device,
            )
            saes[layer] = sae
            log.info("  SAE layer %d : %s features, type=%s",
                     layer, cfg_dict.get("d_sae", "?"), cfg_dict.get("architecture", "?"))
        return saes

    def __repr__(self) -> str:
        return (
            f"SAEProbe(model={self.model_id!r}, layers={self.layers}, "
            f"sae={self.sae_width}/l0_{self.l0}, device={self.device!r})"
        )
