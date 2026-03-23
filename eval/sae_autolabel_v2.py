#!/usr/bin/env python3
"""
ERIS v5 — SAE Auto-Labelling v2
=================================

Rigorous feature labelling for trained SAEs. Drop-in replacement for the
auto_label_features() function in train_sae.py.

Three improvements over v1:
  1. Proper token→text boundaries (no modulo approximation)
  2. Per-text aggregated activations (mean + max) for ranking
  3. Contrastive labelling prompt with predictive validation

Can also be run standalone on an existing SAE checkpoint:

  python eval/sae_autolabel_v2.py \\
    --checkpoint checkpoints/sae_qwen35_layer9/sae_weights.pt \\
    --cache-dir sae_cache \\
    --layer 9

Requirements:
  pip install torch numpy anthropic tqdm
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sae_label")


# ═══════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TextBoundary:
    """Maps a text to its token range in the flat hidden-state array."""
    text_idx: int
    start_token: int
    end_token: int


@dataclass
class FeatureLabel:
    feature_idx: int
    label: str
    confidence: int              # 1-5 self-assessed by judge
    prediction_accuracy: float   # fraction of held-out texts correctly predicted
    mean_activation: float       # mean activation across top examples
    n_texts_active: int          # how many texts activate this feature at all
    top_texts: list              # top 5 activating texts (truncated)
    negative_texts: list         # 3 near-miss texts that don't activate
    quality: str                 # "high" / "medium" / "low" based on prediction accuracy


# ═══════════════════════════════════════════════════════════════════
#  Boundary-aware data collection (patch for train_sae.py)
# ═══════════════════════════════════════════════════════════════════

def collect_hidden_states_with_boundaries(
    client,
    texts: list[str],
    layer: int,
    cache_dir: Path,
    hidden_dim: int,
    max_seq_len: int = 128,
    force: bool = False,
) -> tuple[np.ndarray, list[TextBoundary], list[str]]:
    """
    Like collect_hidden_states() but also saves token→text boundaries.

    Returns:
      data:       np.ndarray [N_tokens, hidden_dim]
      boundaries: list[TextBoundary] — one per text
      texts:      the input texts (cleaned, same order as boundaries)

    Saves three files in cache_dir:
      hidden_states_layer{N}.npy
      boundaries_layer{N}.json
      texts_layer{N}.json
    """
    hs_path = cache_dir / f"hidden_states_layer{layer}.npy"
    bd_path = cache_dir / f"boundaries_layer{layer}.json"
    tx_path = cache_dir / f"texts_layer{layer}.json"

    if not force and hs_path.exists() and bd_path.exists() and tx_path.exists():
        log.info(f"Loading cached data from {cache_dir}")
        data = np.load(hs_path)
        with open(bd_path) as f:
            boundaries = [TextBoundary(**b) for b in json.load(f)]
        with open(tx_path) as f:
            loaded_texts = json.load(f)
        return data, boundaries, loaded_texts

    log.info(f"Collecting hidden states: {len(texts)} texts, layer={layer}")
    all_vecs = []
    boundaries = []
    valid_texts = []
    offset = 0
    errors = 0

    for text in tqdm(texts, desc="Encode"):
        try:
            enc = client.encode(text, layer)
            hd = enc.get("hidden_dim", hidden_dim)
            for val in enc.get("hidden_states", {}).values():
                if isinstance(val, str):
                    flat = np.frombuffer(base64.b64decode(val), dtype=np.float32)
                    mat = flat.reshape(-1, hd)
                elif isinstance(val, list):
                    mat = np.array(val, dtype=np.float32)
                    if mat.ndim == 1:
                        mat = mat.reshape(1, -1)
                else:
                    continue

                mat = mat[:max_seq_len]
                n_tokens = len(mat)
                all_vecs.append(mat)
                boundaries.append(TextBoundary(
                    text_idx=len(valid_texts),
                    start_token=offset,
                    end_token=offset + n_tokens,
                ))
                valid_texts.append(text)
                offset += n_tokens
                break  # only take first key from hidden_states
        except Exception as e:
            errors += 1
            if errors <= 5:
                log.warning(f"  encode error: {e}")
            if errors > 50:
                log.error("  Too many errors, stopping collection")
                break

    if not all_vecs:
        raise RuntimeError("No hidden states collected — is the ERIS server running?")

    data = np.concatenate(all_vecs, axis=0)
    log.info(f"  {data.shape[0]:,} tokens, {len(boundaries)} texts, errors={errors}")

    # Save
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(hs_path, data)
    with open(bd_path, "w") as f:
        json.dump([asdict(b) for b in boundaries], f)
    with open(tx_path, "w", encoding="utf-8") as f:
        json.dump(valid_texts, f, ensure_ascii=False)
    log.info(f"  Cached to {cache_dir}")

    return data, boundaries, valid_texts


# ═══════════════════════════════════════════════════════════════════
#  Per-text feature activation aggregation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_per_text_activations(
    sae,
    data: np.ndarray,
    boundaries: list[TextBoundary],
    device: str = "cpu",
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Compute per-text mean activation for every feature.

    Returns: [n_texts, n_features] tensor on CPU.
    """
    sae = sae.to(device).eval()
    tensor_data = torch.from_numpy(data).float().to(device)
    n_texts = len(boundaries)
    n_features = sae.n_features

    # Encode all tokens in chunks
    all_z_parts = []
    for i in range(0, len(tensor_data), chunk_size):
        z = sae.encode(tensor_data[i:i + chunk_size])
        all_z_parts.append(z.cpu())
    all_z = torch.cat(all_z_parts, dim=0)  # [N_tokens, n_features]

    # Aggregate per text: mean activation
    text_acts = torch.zeros(n_texts, n_features)
    for bd in boundaries:
        token_acts = all_z[bd.start_token:bd.end_token]  # [n_tokens, n_features]
        if len(token_acts) > 0:
            text_acts[bd.text_idx] = token_acts.mean(dim=0)

    return text_acts


def find_near_miss_texts(
    text_acts: torch.Tensor,
    feat_idx: int,
    top_text_indices: list[int],
    texts: list[str],
    n_negatives: int = 3,
) -> list[str]:
    """
    Find texts that are semantically similar to the top-activating texts
    (high activation on OTHER features) but don't activate this feature.
    """
    feat_acts = text_acts[:, feat_idx]

    # Texts where this feature is inactive
    inactive_mask = feat_acts < feat_acts.median() * 0.1

    if inactive_mask.sum() < n_negatives:
        lowest = feat_acts.argsort()[:n_negatives * 3]
        inactive_indices = [i.item() for i in lowest if i.item() not in top_text_indices]
    else:
        inactive_indices = inactive_mask.nonzero(as_tuple=True)[0].tolist()

    # Among inactive texts, pick those with highest OVERALL activation
    total_acts = text_acts.sum(dim=1)
    scored = [(idx, total_acts[idx].item()) for idx in inactive_indices
              if idx not in top_text_indices]
    scored.sort(key=lambda x: -x[1])

    return [texts[idx][:200] for idx, _ in scored[:n_negatives]]


# ═══════════════════════════════════════════════════════════════════
#  Labelling prompt with contrastive examples + predictive validation
# ═══════════════════════════════════════════════════════════════════

def build_label_prompt(
    feat_idx: int,
    positive_texts: list[str],
    positive_scores: list[float],
    negative_texts: list[str],
    held_out_texts: list[str],
    held_out_active: list[bool],
) -> tuple[str, list[str]]:
    """
    Build the labelling prompt.

    Returns (prompt_str, correct_letters) where correct_letters is the
    ground-truth answer for scoring (not shown to the judge).
    """
    pos_lines = [f"  {i+1}. [score={s:.3f}] {t[:200]}"
                 for i, (t, s) in enumerate(zip(positive_texts, positive_scores))]
    neg_lines = [f"  {i+1}. {t[:200]}" for i, t in enumerate(negative_texts)]
    held_lines = [f"  {chr(65+i)}. {t[:200]}" for i, t in enumerate(held_out_texts)]

    neg_block = "\n".join(neg_lines) if neg_lines else "  (none available)"
    correct_letters = [chr(65 + i) for i, active in enumerate(held_out_active) if active]

    prompt = (
        f"Analyze neural network feature #{feat_idx}.\n\n"
        f"TEXTS THAT STRONGLY ACTIVATE THIS FEATURE (with activation scores):\n"
        f"{chr(10).join(pos_lines)}\n\n"
        f"TEXTS WHERE THIS FEATURE IS INACTIVE (similar content, feature does not fire):\n"
        f"{neg_block}\n\n"
        f"TASKS:\n"
        f"1. In 3-8 words, what specific concept or pattern does this feature detect?\n"
        f"   Be precise — \"math\" is too vague, \"multi-step arithmetic word problems\" is good.\n"
        f"2. Rate your confidence 1-5 that this label captures the feature's true selectivity.\n"
        f"3. Which of these NEW texts would activate this feature? Reply with letters only.\n\n"
        f"NEW TEXTS:\n"
        f"{chr(10).join(held_lines)}\n\n"
        f"FORMAT (exactly three lines):\n"
        f"LABEL: <your label>\n"
        f"CONFIDENCE: <1-5>\n"
        f"PREDICT: <letters, e.g. A,C,E or NONE>"
    )
    return prompt, correct_letters


def parse_label_response(response: str) -> tuple[str, int, list[str]]:
    """Parse the judge's response into (label, confidence, predictions)."""
    label = ""
    confidence = 3
    predictions = []

    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("LABEL:"):
            label = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                confidence = int(line.split(":", 1)[1].strip()[0])
                confidence = max(1, min(5, confidence))
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("PREDICT:"):
            pred_str = line.split(":", 1)[1].strip().upper()
            if pred_str == "NONE":
                predictions = []
            else:
                predictions = [c.strip() for c in pred_str.split(",")
                               if c.strip().isalpha() and len(c.strip()) == 1]

    return label, confidence, predictions


def score_predictions(predicted: list[str], correct: list[str], n_held_out: int) -> float:
    """accuracy = (true positives + true negatives) / total"""
    all_letters = [chr(65 + i) for i in range(n_held_out)]
    pred_set = set(predicted)
    corr_set = set(correct)
    tp = len(pred_set & corr_set)
    tn = len(set(all_letters) - pred_set - corr_set)
    return (tp + tn) / max(len(all_letters), 1)


# ═══════════════════════════════════════════════════════════════════
#  Main auto-labelling function (v2)
# ═══════════════════════════════════════════════════════════════════

def auto_label_features_v2(
    sae,
    data: np.ndarray,
    boundaries: list[TextBoundary],
    texts: list[str],
    *,
    top_k_features: int = 200,
    top_k_examples: int = 5,
    n_negatives: int = 3,
    n_held_out: int = 5,
    device: str = "cpu",
    model: str = "claude-sonnet-4-6",
) -> list[FeatureLabel]:
    """
    Rigorous auto-labelling for SAE features.

    For each of the top_k most active features:
      1. Compute per-text mean activation (boundary-aware)
      2. Select top examples, near-miss negatives, and held-out texts
      3. Ask Claude to label with contrastive prompt
      4. Validate via predictive accuracy on held-out set
      5. Assign quality tier based on prediction score

    Returns list of FeatureLabel sorted by quality then frequency.
    """
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.warning("ANTHROPIC_API_KEY not set — cannot auto-label")
        return []

    cl = anthropic.Anthropic(api_key=key)

    # Per-text mean activations [n_texts, n_features]
    log.info("Computing per-text feature activations...")
    text_acts = compute_per_text_activations(sae, data, boundaries, device)
    n_features = text_acts.shape[1]

    # Filter: 1%–80% frequency range
    feat_freq = (text_acts > 0).float().mean(dim=0)
    valid_mask = (feat_freq > 0.01) & (feat_freq < 0.80)
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    if len(valid_indices) == 0:
        log.warning("No features in valid frequency range")
        return []

    sorted_order = feat_freq[valid_indices].argsort(descending=True)
    top_features = valid_indices[sorted_order[:top_k_features]].tolist()

    log.info(f"Labelling {len(top_features)} features "
             f"(of {n_features} total, {int(valid_mask.sum())} in valid range)")

    results = []

    for feat_idx in tqdm(top_features, desc="Auto-label v2"):
        feat_acts_col = text_acts[:, feat_idx]
        n_active = int((feat_acts_col > 0).sum().item())

        if n_active < top_k_examples + n_held_out + 2:
            continue

        sorted_indices = feat_acts_col.argsort(descending=True)

        top_indices = sorted_indices[:top_k_examples].tolist()
        held_active_idx = sorted_indices[top_k_examples:top_k_examples + n_held_out // 2].tolist()
        held_inactive_idx = sorted_indices[-(n_held_out - len(held_active_idx)):].tolist()

        held_out_idx = held_active_idx + held_inactive_idx
        held_out_flags = [True] * len(held_active_idx) + [False] * len(held_inactive_idx)

        # Shuffle held-out to avoid position bias
        rng = np.random.default_rng(feat_idx)
        perm = rng.permutation(len(held_out_idx))
        held_out_idx = [held_out_idx[p] for p in perm]
        held_out_flags = [held_out_flags[p] for p in perm]

        pos_texts = [texts[i][:200] for i in top_indices]
        pos_scores = [feat_acts_col[i].item() for i in top_indices]
        neg_texts = find_near_miss_texts(text_acts, feat_idx, top_indices, texts, n_negatives)
        held_texts = [texts[i][:200] for i in held_out_idx]

        prompt, correct_letters = build_label_prompt(
            feat_idx, pos_texts, pos_scores, neg_texts,
            held_texts, held_out_flags,
        )

        try:
            resp = cl.messages.create(
                model=model, max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
                timeout=45,
            )
            raw = resp.content[0].text
            label, confidence, predictions = parse_label_response(raw)

            if not label:
                continue

            pred_acc = score_predictions(predictions, correct_letters, len(held_texts))

            if pred_acc >= 0.7 and confidence >= 4:
                quality = "high"
            elif pred_acc >= 0.5 and confidence >= 3:
                quality = "medium"
            else:
                quality = "low"

            results.append(FeatureLabel(
                feature_idx=feat_idx,
                label=label,
                confidence=confidence,
                prediction_accuracy=round(pred_acc, 3),
                mean_activation=round(float(feat_acts_col[top_indices].mean()), 4),
                n_texts_active=n_active,
                top_texts=pos_texts[:3],
                negative_texts=neg_texts[:2],
                quality=quality,
            ))

        except Exception as e:
            log.debug(f"  Feature {feat_idx}: {e}")

        time.sleep(0.15)  # rate limit

    tier_order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda r: (tier_order[r.quality], -r.n_texts_active))

    n_high = sum(1 for r in results if r.quality == "high")
    n_med  = sum(1 for r in results if r.quality == "medium")
    n_low  = sum(1 for r in results if r.quality == "low")
    mean_pred = float(np.mean([r.prediction_accuracy for r in results])) if results else 0.0

    log.info(f"  Labelled {len(results)} features: "
             f"{n_high} high / {n_med} medium / {n_low} low")
    log.info(f"  Mean prediction accuracy: {mean_pred:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Save labels
# ═══════════════════════════════════════════════════════════════════

def save_labels(labels: list[FeatureLabel], output_path: Path):
    """Save labels in two formats: full (with metadata) and simple (for SAEAnalyzer)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Full format with all metadata
    full_path = output_path.with_suffix(".full.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump([asdict(l) for l in labels], f, indent=2, ensure_ascii=False)

    # Simple format: {feature_idx: label} for SAEAnalyzer
    simple = {str(l.feature_idx): l.label for l in labels}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simple, f, indent=2, ensure_ascii=False)

    # Quality report
    report_path = output_path.with_suffix(".report.txt")
    with open(report_path, "w") as f:
        f.write(f"SAE Feature Labels — {len(labels)} features\n")
        f.write(f"{'='*70}\n\n")
        for tier in ["high", "medium", "low"]:
            tier_labels = [l for l in labels if l.quality == tier]
            f.write(f"\n{'─'*70}\n")
            f.write(f"  {tier.upper()} quality ({len(tier_labels)} features)\n")
            f.write(f"{'─'*70}\n")
            for lb in tier_labels:
                f.write(f"  #{lb.feature_idx:>5d}  {lb.label:<40s}  "
                        f"conf={lb.confidence}  pred={lb.prediction_accuracy:.2f}  "
                        f"active_in={lb.n_texts_active}\n")

    log.info(f"Saved: {output_path} + {full_path.name} + {report_path.name}")


# ═══════════════════════════════════════════════════════════════════
#  Standalone CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    """Run standalone on an existing SAE checkpoint."""
    import argparse
    import torch.nn as nn
    import torch.nn.functional as F

    ap = argparse.ArgumentParser(description="SAE Auto-Labelling v2")
    ap.add_argument("--checkpoint", required=True, help="Path to sae_weights.pt")
    ap.add_argument("--cache-dir",  required=True, help="Directory with cached .npy + boundaries")
    ap.add_argument("--layer",      type=int, required=True)
    ap.add_argument("--top-k",      type=int, default=200, help="Number of features to label")
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--model",      default="claude-sonnet-4-6",
                    help="Claude model for labelling")
    ap.add_argument("--output",     default=None,
                    help="Output path (default: next to checkpoint)")
    args = ap.parse_args()

    # Load SAE checkpoint
    ckpt = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch -- loading our own checkpoint with weights_only=True
        args.checkpoint, map_location="cpu", weights_only=True
    )
    state = ckpt.get("state_dict", ckpt)
    cfg   = ckpt.get("cfg", {})

    hidden_dim = cfg.get("hidden_dim") or state["W_enc"].shape[0]
    n_features = cfg.get("n_features") or state["W_enc"].shape[1]

    # Reconstruct SAE (import from train_sae if available)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from train_sae import SparseAutoencoder
    except ImportError:
        class SparseAutoencoder(nn.Module):  # type: ignore[no-redef]
            def __init__(self, hd: int, nf: int):
                super().__init__()
                self.hidden_dim = hd
                self.n_features = nf
                self.W_enc = nn.Parameter(torch.empty(hd, nf))
                self.b_enc = nn.Parameter(torch.zeros(nf))
                self.W_dec = nn.Parameter(torch.empty(nf, hd))
                self.b_pre = nn.Parameter(torch.zeros(hd))

            def encode(self, h: torch.Tensor) -> torch.Tensor:
                return F.relu((h - self.b_pre) @ self.W_enc + self.b_enc)

    sae = SparseAutoencoder(hidden_dim, n_features)
    sae.W_enc.data = state["W_enc"]
    sae.b_enc.data = state["b_enc"]
    if "W_dec" in state:
        sae.W_dec.data = state["W_dec"]
    if "b_dec" in state:
        sae.b_pre.data = state["b_dec"]
    elif "b_pre" in state:
        sae.b_pre.data = state["b_pre"]
    log.info(f"Loaded SAE: {hidden_dim}→{n_features}")

    # Load cached data
    cache_dir = Path(args.cache_dir)
    hs_path = cache_dir / f"hidden_states_layer{args.layer}.npy"
    bd_path = cache_dir / f"boundaries_layer{args.layer}.json"
    tx_path = cache_dir / f"texts_layer{args.layer}.json"

    missing = [p for p in [hs_path, bd_path, tx_path] if not p.exists()]
    if missing:
        log.error(
            f"Missing cache files: {[p.name for p in missing]}\n"
            f"Re-run train_sae.py — it now saves boundaries automatically."
        )
        return

    data = np.load(hs_path)
    with open(bd_path) as f:
        boundaries = [TextBoundary(**b) for b in json.load(f)]
    with open(tx_path) as f:
        texts = json.load(f)
    log.info(f"Data: {data.shape[0]:,} tokens, {len(boundaries)} texts")

    # Label
    labels = auto_label_features_v2(
        sae, data, boundaries, texts,
        top_k_features=args.top_k,
        device=args.device,
        model=args.model,
    )

    # Save
    out_path = (Path(args.output) if args.output
                else Path(args.checkpoint).parent / "feature_labels.json")
    save_labels(labels, out_path)

    # Summary
    n_high = sum(1 for l in labels if l.quality == "high")
    n_med  = sum(1 for l in labels if l.quality == "medium")
    n_low  = sum(1 for l in labels if l.quality == "low")

    print(f"\n{'═'*60}")
    print(f"  Auto-Labelling Complete")
    print(f"{'═'*60}")
    print(f"  Features labelled: {len(labels)}")
    print(f"  Quality: {n_high} high / {n_med} medium / {n_low} low")
    if labels:
        print(f"  Mean prediction accuracy: {np.mean([l.prediction_accuracy for l in labels]):.3f}")
        print(f"  Mean confidence:          {np.mean([l.confidence for l in labels]):.1f}")
    print(f"  Output: {out_path}")
    high = [l for l in labels if l.quality == "high"]
    if high:
        print(f"\n  Top high-quality labels:")
        for l in high[:10]:
            print(f"    #{l.feature_idx:>5d}  {l.label:<40s}  "
                  f"pred={l.prediction_accuracy:.2f}  conf={l.confidence}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
