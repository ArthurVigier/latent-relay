#!/usr/bin/env python3
"""
ERIS v5 — SAE Trainer
======================
Train a Sparse Autoencoder on hidden states extracted from the zombie model
via the ERIS /v1/encode endpoint.  The resulting checkpoint is saved in the
exact format expected by eris/analyzers.py SAEAnalyzer, so M6 can run for real.

Architecture
------------
Standard SAE with ReLU + L1 sparsity penalty (Bricken et al. 2023):

  pre_act  = (h - b_pre) @ W_enc + b_enc    # [n_features]
  features = ReLU(pre_act)                   # sparse activations
  h_hat    = features @ W_dec + b_pre        # [hidden_dim] reconstruction

  loss = MSE(h, h_hat) + λ * mean(features)

  b_pre ("pre-encoder bias") absorbs the dataset mean so the encoder
  operates on a zero-centred input.  This is the "folded bias" trick
  from the Anthropic SAE paper.

  Decoder columns are constrained to unit norm after every gradient step
  (prevents the decoder from hiding sparsity via large norms).

Checkpoint saved as:
  sae_weights.pt   {"W_enc", "b_enc", "W_dec", "b_dec", "b_pre",
                    "cfg": {layer, hidden_dim, n_features, expansion_factor, ...}}

The SAEAnalyzer in eris/analyzers.py reads W_enc + b_enc directly.
b_pre is applied automatically during encode (step 1 of forward pass).

Data pipeline
-------------
Phase 1: Collect hidden states from /v1/encode into a .npy cache file.
         This is the expensive step (~1 API call per text, GPU-bound).
         Re-run is skipped if cache already exists (use --force-collect to redo).

Phase 2: Train on the cached hidden states entirely on CPU/GPU locally.
         You can re-run training with different hyperparameters without
         re-hitting the ERIS server.

Phase 3: Save checkpoint + optional auto-label features.

Usage
-----
  # Start ERIS server first:
  python eris_server.py --model Qwen/Qwen3.5-4B --port 8001

  # Minimal: collect 2000 texts, train with defaults
  python eval/train_sae.py --eris-url http://localhost:8001 --layer 9

  # Full options:
  python eval/train_sae.py \\
    --eris-url http://localhost:8001 \\
    --layer 9 \\
    --expansion 8 \\
    --n-texts 5000 \\
    --epochs 20 \\
    --l1-coef 4e-4 \\
    --device cuda \\
    --output checkpoints/sae_qwen35_layer9

  # Skip re-collect if cache exists:
  python eval/train_sae.py --layer 9 --force-collect False

  # Auto-label features after training (needs ANTHROPIC_API_KEY):
  ANTHROPIC_API_KEY=sk-... python eval/train_sae.py --layer 9 --auto-label

After training, point eris_config.yaml at the checkpoint:
  sae:
    model_path: "checkpoints/sae_qwen35_layer9"
    layer: 9
    top_k: 20

Requirements:
  pip install torch datasets httpx numpy tqdm scipy
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sae_trainer")


# ═══════════════════════════════════════════════════════════════════
#  ERIS client (minimal, no session)
# ═══════════════════════════════════════════════════════════════════

class _ERISClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def health(self) -> dict:
        r = self.http.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def encode(self, text: str, layer: int) -> dict:
        r = self.http.post(f"{self.base_url}/v1/encode",
                           json={"text": text, "return_layers": [layer], "compact": True})
        r.raise_for_status()
        return r.json()

    def close(self): self.http.close()


def _decode_b64(s: str, hidden_dim: int) -> np.ndarray:
    flat = np.frombuffer(base64.b64decode(s), dtype=np.float32)
    return flat.reshape(-1, hidden_dim)  # [seq_len, hidden_dim]


# ═══════════════════════════════════════════════════════════════════
#  Data collection
# ═══════════════════════════════════════════════════════════════════

def _build_corpus(n_texts: int, seed: int = 42) -> list[str]:
    """Collect n_texts diverse sentences from HuggingFace datasets."""
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    texts = []

    # --- STS-B (natural language, short sentences) ---
    try:
        ds = load_dataset("sentence-transformers/stsb", split="train")
        pool = list({r["sentence1"] for r in ds} | {r["sentence2"] for r in ds})
        rng.shuffle(pool)
        texts.extend(pool[:n_texts // 4])
        log.info(f"  STS-B: {min(len(pool), n_texts // 4)} texts")
    except Exception as e:
        log.warning(f"  STS-B unavailable: {e}")

    # --- GSM8K (math reasoning) ---
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        pool = [r["question"] for r in ds]
        rng.shuffle(pool)
        texts.extend(pool[:n_texts // 4])
        log.info(f"  GSM8K: {min(len(pool), n_texts // 4)} texts")
    except Exception as e:
        log.warning(f"  GSM8K unavailable: {e}")

    # --- MMLU-Pro (factual, multi-domain) ---
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        pool = [r["question"] for r in ds]
        rng.shuffle(pool)
        texts.extend(pool[:n_texts // 4])
        log.info(f"  MMLU-Pro: {min(len(pool), n_texts // 4)} texts")
    except Exception as e:
        log.warning(f"  MMLU-Pro unavailable: {e}")

    # --- WikiText (general prose) ---
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        pool = [r["text"].strip() for r in ds if len(r["text"].strip()) > 80]
        rng.shuffle(pool)
        # Truncate long wiki paragraphs
        wiki = [t[:512] for t in pool[:n_texts // 4]]
        texts.extend(wiki)
        log.info(f"  WikiText: {len(wiki)} texts")
    except Exception as e:
        log.warning(f"  WikiText unavailable: {e}")

    # Fallback if datasets are sparse
    if len(texts) < 100:
        log.warning("  Very few texts loaded — SAE will be undertrained")

    # Deduplicate and shuffle
    texts = list(dict.fromkeys(t for t in texts if len(t.strip()) > 10))
    rng.shuffle(texts)
    return texts[:n_texts]


def collect_hidden_states(
    client: _ERISClient,
    texts: list[str],
    layer: int,
    cache_path: Path,
    hidden_dim: int,
    max_seq_len: int = 128,
) -> np.ndarray:
    """
    Encode all texts, concatenate token-level hidden states into one array.

    Returns np.ndarray of shape [N_tokens, hidden_dim].
    Saves to cache_path so re-runs skip the encode step.
    """
    if cache_path.exists():
        log.info(f"Loading cached hidden states from {cache_path}")
        return np.load(cache_path)

    log.info(f"Collecting hidden states: {len(texts)} texts, layer={layer}")
    all_vecs = []
    errors = 0

    for text in tqdm(texts, desc="Encode"):
        try:
            enc = client.encode(text, layer)
            hd  = enc.get("hidden_dim", hidden_dim)
            for val in enc.get("hidden_states", {}).values():
                if isinstance(val, str):
                    mat = _decode_b64(val, hd)       # [seq_len, hidden_dim]
                elif isinstance(val, list):
                    mat = np.array(val, dtype=np.float32)
                    if mat.ndim == 1: mat = mat.reshape(1, -1)
                else:
                    continue
                # Truncate very long sequences
                mat = mat[:max_seq_len]
                all_vecs.append(mat)
        except Exception as e:
            errors += 1
            if errors <= 5: log.warning(f"  encode error: {e}")
            if errors > 50:
                log.error("  Too many errors, stopping collection")
                break

    if not all_vecs:
        raise RuntimeError("No hidden states collected — is the ERIS server running?")

    data = np.concatenate(all_vecs, axis=0)   # [N_tokens, hidden_dim]
    log.info(f"  Collected {data.shape[0]:,} token vectors, hidden_dim={data.shape[1]}, errors={errors}")
    np.save(cache_path, data)
    log.info(f"  Cached to {cache_path}")
    return data


# ═══════════════════════════════════════════════════════════════════
#  SAE model
# ═══════════════════════════════════════════════════════════════════

class SparseAutoencoder(nn.Module):
    """
    Standard SAE (Bricken et al. 2023).

    Forward:
      z        = ReLU( (h - b_pre) @ W_enc + b_enc )
      h_hat    = z @ W_dec + b_pre
      loss     = MSE(h, h_hat) + l1_coef * z.mean()

    The pre-encoder bias b_pre absorbs the dataset mean, keeping the
    encoder operating on zero-centred inputs.

    Decoder columns are normalised to unit L2 norm after each update
    (constrains the dictionary atoms to the unit sphere).
    """

    def __init__(self, hidden_dim: int, n_features: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(hidden_dim, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder
        self.W_dec = nn.Parameter(torch.empty(n_features, hidden_dim))
        self.b_pre = nn.Parameter(torch.zeros(hidden_dim))   # pre-encoder / decoder bias

        # Kaiming uniform init
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

        # Initialise decoder columns to unit norm
        self._normalise_decoder()

    @torch.no_grad()
    def _normalise_decoder(self):
        """Project decoder column vectors onto the unit sphere."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """h: [batch, hidden_dim] → features: [batch, n_features]"""
        return F.relu((h - self.b_pre) @ self.W_enc + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [batch, n_features] → h_hat: [batch, hidden_dim]"""
        return z @ self.W_dec + self.b_pre

    def forward(self, h: torch.Tensor):
        z     = self.encode(h)
        h_hat = self.decode(z)
        return z, h_hat

    def loss(self, h: torch.Tensor, z: torch.Tensor, h_hat: torch.Tensor,
             l1_coef: float) -> tuple[torch.Tensor, dict]:
        mse = F.mse_loss(h_hat, h)
        l1  = z.mean()
        total = mse + l1_coef * l1
        return total, {"mse": mse.item(), "l1": l1.item(), "total": total.item()}


# ═══════════════════════════════════════════════════════════════════
#  Dead feature resampling
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def resample_dead_features(
    sae: SparseAutoencoder,
    data: torch.Tensor,
    dead_mask: torch.Tensor,
    n_resample: int = 512,
    noise_scale: float = 0.2,
):
    """
    Resample dead encoder features by reinitialising them to random
    high-loss input vectors (with small noise).

    Dead features are features that have not fired in the last
    `dead_window` steps.  Without resampling, dead features stay dead
    forever and waste capacity.
    """
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return 0

    log.info(f"  Resampling {n_dead} dead features")

    # Pick n_resample random inputs weighted by reconstruction loss
    idx   = torch.randperm(len(data))[:n_resample]
    batch = data[idx].to(sae.W_enc.device)

    with torch.no_grad():
        z, h_hat = sae(batch)
        errors = (batch - h_hat).norm(dim=-1)   # [n_resample] per-sample L2 error
        probs  = (errors ** 2 + 1e-8)
        probs /= probs.sum()

    chosen = torch.multinomial(probs, n_dead, replacement=(n_dead > n_resample))
    new_vecs = batch[chosen] - sae.b_pre.unsqueeze(0)   # [n_dead, hidden_dim]

    # Normalise and add small noise
    new_norms = new_vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    new_vecs  = new_vecs / new_norms
    new_vecs += noise_scale * torch.randn_like(new_vecs)

    # Re-initialise encoder rows and decoder columns for dead features
    dead_idx = dead_mask.nonzero(as_tuple=True)[0]
    sae.W_enc.data[:, dead_idx] = new_vecs.T          # [hidden_dim, n_dead]
    sae.b_enc.data[dead_idx]    = 0.0
    sae.W_dec.data[dead_idx]    = new_vecs             # [n_dead, hidden_dim]
    sae._normalise_decoder()

    return n_dead


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════

def train(
    sae: SparseAutoencoder,
    data: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 2e-4,
    l1_coef: float = 4e-4,
    warmup_steps: int = 500,
    dead_window: int = 2000,   # steps without firing → dead
    resample_every: int = 2000,
    device: str = "cpu",
) -> list[dict]:
    """Train the SAE and return per-epoch metrics."""

    sae = sae.to(device)
    tensor_data = torch.from_numpy(data).float().to(device)
    N = len(tensor_data)

    # Centre the data: initialise b_pre to dataset mean
    with torch.no_grad():
        mean = tensor_data.mean(0)
        sae.b_pre.data = mean.clone()

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    steps_per_epoch = (N + batch_size - 1) // batch_size
    total_steps     = epochs * steps_per_epoch
    warmup          = min(warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        # Cosine decay to 10% of peak LR
        t = (step - warmup) / max(total_steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Dead feature tracking: cumulative activation counts
    activation_counts = torch.zeros(sae.n_features, device=device)
    counts_since_reset = 0

    history = []
    global_step = 0

    for epoch in range(epochs):
        # Shuffle data each epoch
        perm = torch.randperm(N, device=device)
        epoch_metrics = {"epoch": epoch + 1, "mse": [], "l1": [], "total": [],
                         "frac_active": [], "dead_features": 0}

        for start in range(0, N, batch_size):
            idx   = perm[start:start + batch_size]
            batch = tensor_data[idx]

            optimizer.zero_grad()
            z, h_hat = sae(batch)
            loss, m  = sae.loss(batch, z, h_hat, l1_coef)
            loss.backward()

            # Gradient clipping — SAE training can be noisy
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Decoder normalisation after every update
            sae._normalise_decoder()

            # Track activation counts for dead feature detection
            with torch.no_grad():
                fired = (z > 0).float().sum(0)  # [n_features]
                activation_counts += fired
                counts_since_reset += len(batch)

            # Dead feature resampling
            if global_step > 0 and global_step % resample_every == 0:
                dead_mask = activation_counts == 0
                n_dead    = resample_dead_features(sae, tensor_data, dead_mask)
                activation_counts.zero_()
                counts_since_reset = 0
                epoch_metrics["dead_features"] += n_dead

            epoch_metrics["mse"].append(m["mse"])
            epoch_metrics["l1"].append(m["l1"])
            epoch_metrics["total"].append(m["total"])

            # Fraction of features active per token (L0)
            with torch.no_grad():
                l0 = (z > 0).float().mean(dim=1).mean().item()
                epoch_metrics["frac_active"].append(l0)

            global_step += 1

        # Epoch summary
        summary = {
            "epoch": epoch + 1,
            "mse":         float(np.mean(epoch_metrics["mse"])),
            "l1":          float(np.mean(epoch_metrics["l1"])),
            "total":       float(np.mean(epoch_metrics["total"])),
            "frac_active": float(np.mean(epoch_metrics["frac_active"])),
            "dead_features_resampled": epoch_metrics["dead_features"],
            "lr":          float(scheduler.get_last_lr()[0]),
        }
        # Dead feature count at end of epoch
        with torch.no_grad():
            n_dead_now = int((activation_counts == 0).sum().item())
        summary["dead_features_now"] = n_dead_now

        history.append(summary)
        log.info(
            f"  Epoch {epoch+1}/{epochs}  "
            f"loss={summary['total']:.5f}  "
            f"mse={summary['mse']:.5f}  "
            f"l1={summary['l1']:.5f}  "
            f"L0={summary['frac_active']:.3f}  "
            f"dead={n_dead_now}"
        )

    return history


# ═══════════════════════════════════════════════════════════════════
#  Auto-labelling  (optional, needs ANTHROPIC_API_KEY)
# ═══════════════════════════════════════════════════════════════════

def auto_label_features(
    sae: SparseAutoencoder,
    data: np.ndarray,
    texts: list[str],   # original texts (parallel to tokens if seq-averaged)
    top_k_features: int = 200,
    top_k_examples: int = 5,
) -> dict[int, str]:
    """
    For the top_k_features most active features, find the texts that
    maximally activate them and ask Claude to summarise the pattern.

    Returns {feature_index: label_string}.
    """
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.warning("ANTHROPIC_API_KEY not set — skipping auto-labelling")
        return {}

    cl = anthropic.Anthropic(api_key=key)
    device = next(sae.parameters()).device

    log.info(f"Auto-labelling top {top_k_features} features...")

    # Compute per-text mean activation for every feature
    # We average over tokens per text to get a [n_texts, n_features] matrix
    tensor_data = torch.from_numpy(data).float().to(device)
    with torch.no_grad():
        # Process in chunks to avoid OOM
        chunk = 1024
        feat_sums  = torch.zeros(sae.n_features, device=device)
        feat_counts = torch.zeros(sae.n_features, device=device)
        for i in range(0, len(tensor_data), chunk):
            z = sae.encode(tensor_data[i:i+chunk])
            feat_sums   += z.sum(0)
            feat_counts += (z > 0).float().sum(0)

    # Pick the top_k most-used features
    feat_freq = feat_counts / max(len(tensor_data), 1)
    top_feat_idx = torch.topk(feat_freq, min(top_k_features, sae.n_features)).indices.tolist()

    # For each feature, find which input texts maximally activate it
    # We need per-text mean activations: recompute with text boundaries
    # Approximation: just use the raw token-level activations sorted by feature
    with torch.no_grad():
        all_z = []
        for i in range(0, len(tensor_data), chunk):
            all_z.append(sae.encode(tensor_data[i:i+chunk]).cpu())
        all_z = torch.cat(all_z, 0)  # [N_tokens, n_features]

    labels = {}
    for feat_i, feat_idx in enumerate(tqdm(top_feat_idx, desc="Auto-label")):
        acts = all_z[:, feat_idx]
        top_tok_idx = acts.topk(min(top_k_examples * 20, len(acts))).indices.tolist()

        # Map token indices back to texts (approximate: use modulo of corpus size)
        example_texts = []
        seen_texts = set()
        for ti in top_tok_idx:
            # Map to a text by index modulo n_texts
            text_i = ti % len(texts)
            if text_i not in seen_texts:
                seen_texts.add(text_i)
                example_texts.append(texts[text_i][:200])
            if len(example_texts) >= top_k_examples:
                break

        if not example_texts:
            continue

        examples_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(example_texts))
        prompt = (
            f"These texts strongly activate a learned neural feature (feature #{feat_idx}):\n\n"
            f"{examples_str}\n\n"
            f"In 3-8 words, what semantic concept or pattern do these share?\n"
            f"Reply with ONLY the label, nothing else."
        )

        try:
            resp = cl.messages.create(
                model="claude-sonnet-4-6", max_tokens=20,
                messages=[{"role": "user", "content": prompt}], timeout=30
            )
            label = resp.content[0].text.strip().strip('"').strip("'")
            labels[feat_idx] = label
        except Exception as e:
            log.debug(f"  feature {feat_idx}: {e}")

        time.sleep(0.1)   # rate limit

    log.info(f"  Labelled {len(labels)} features")
    return labels


# ═══════════════════════════════════════════════════════════════════
#  Save / load checkpoint
# ═══════════════════════════════════════════════════════════════════

def save_checkpoint(
    sae: SparseAutoencoder,
    output_dir: Path,
    cfg: dict,
    labels: dict | None = None,
    history: list | None = None,
):
    """Save in the exact format expected by eris/analyzers.py SAEAnalyzer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {
        "W_enc": sae.W_enc.data.cpu(),   # [hidden_dim, n_features]
        "b_enc": sae.b_enc.data.cpu(),   # [n_features]
        "W_dec": sae.W_dec.data.cpu(),   # [n_features, hidden_dim]
        "b_dec": sae.b_pre.data.cpu(),   # [hidden_dim]  (b_pre acts as decoder bias)
    }

    checkpoint = {"state_dict": state_dict, "cfg": cfg}
    ckpt_path  = output_dir / "sae_weights.pt"
    torch.save(checkpoint, ckpt_path)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch -- save-side only; loader (eris/analyzers.py) uses weights_only=True
    log.info(f"Saved checkpoint → {ckpt_path}")

    if labels:
        labels_path = output_dir / "feature_labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in labels.items()}, f, indent=2)
        log.info(f"Saved feature labels → {labels_path}")

    if history:
        hist_path = output_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    # Print the config block to paste into eris_config.yaml
    print(f"\n{'─'*60}")
    print("Add to eris_config.yaml:")
    print(f"{'─'*60}")
    print("sae:")
    print(f"  model_path: \"{output_dir.resolve()}\"")
    print(f"  layer: {cfg['layer']}")
    print(f"  top_k: 20")
    if labels:
        print(f"  labels_path: \"{(output_dir / 'feature_labels.json').resolve()}\"")
    print(f"{'─'*60}\n")


# ═══════════════════════════════════════════════════════════════════
#  Evaluation — quick sanity check on held-out data
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(sae: SparseAutoencoder, data: np.ndarray, l1_coef: float, device: str) -> dict:
    sae = sae.to(device)
    sae.eval()
    tensor = torch.from_numpy(data).float().to(device)

    chunk, mse_vals, l0_vals, l1_vals = 512, [], [], []
    for i in range(0, len(tensor), chunk):
        b    = tensor[i:i+chunk]
        z, h_hat = sae(b)
        mse_vals.append(F.mse_loss(h_hat, b).item())
        l0_vals.append((z > 0).float().mean(dim=1).mean().item())
        l1_vals.append(z.mean().item())

    # Variance explained (R²)
    h_var   = float(tensor.var().item())
    mse_avg = float(np.mean(mse_vals))
    r2      = max(0.0, 1.0 - mse_avg / (h_var + 1e-10))

    # Dead feature count
    all_z = torch.cat([sae.encode(tensor[i:i+chunk]) for i in range(0, len(tensor), chunk)])
    n_dead = int((all_z.max(0).values == 0).sum().item())

    return {
        "mse":        round(mse_avg, 6),
        "r2":         round(r2, 4),
        "l0_mean":    round(float(np.mean(l0_vals)), 4),
        "l1_mean":    round(float(np.mean(l1_vals)), 6),
        "dead_features": n_dead,
        "pct_dead":   round(100 * n_dead / sae.n_features, 1),
        "n_features": sae.n_features,
        "hidden_dim": sae.hidden_dim,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="ERIS SAE Trainer")
    ap.add_argument("--eris-url",      default="http://localhost:8001")
    ap.add_argument("--layer",         type=int, default=9,
                    help="Hidden layer to train on (9 for Qwen3.5, 35 for Qwen3)")
    ap.add_argument("--expansion",     type=int, default=8,
                    help="n_features = expansion × hidden_dim (default 8, ~8× overcomplete)")
    ap.add_argument("--n-texts",       type=int, default=3000,
                    help="Number of texts to encode for training")
    ap.add_argument("--epochs",        type=int, default=15)
    ap.add_argument("--batch-size",    type=int, default=256)
    ap.add_argument("--lr",            type=float, default=2e-4)
    ap.add_argument("--l1-coef",       type=float, default=4e-4,
                    help="L1 sparsity coefficient (higher → sparser features)")
    ap.add_argument("--warmup-steps",  type=int, default=500)
    ap.add_argument("--resample-every",type=int, default=2000,
                    help="Steps between dead feature resampling")
    ap.add_argument("--max-seq-len",   type=int, default=128,
                    help="Max tokens per text to keep (prevents OOM from very long texts)")
    ap.add_argument("--device",        default="cpu",
                    help="Training device (cpu | cuda | cuda:0)")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--output",        default=None,
                    help="Output directory (default: checkpoints/sae_layer{N})")
    ap.add_argument("--cache-dir",     default="sae_cache",
                    help="Directory for cached hidden state arrays")
    ap.add_argument("--force-collect", action="store_true",
                    help="Re-collect hidden states even if cache exists")
    ap.add_argument("--skip-train",    action="store_true",
                    help="Only collect data, skip training (for large data collection jobs)")
    ap.add_argument("--auto-label",    action="store_true",
                    help="Auto-label top features via Claude (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--val-split",     type=float, default=0.05,
                    help="Fraction of data held out for validation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Connect to ERIS server ──
    client = _ERISClient(args.eris_url)
    health = client.health()
    model_name = health.get("model", "unknown")
    log.info(f"Connected to ERIS: model={model_name}")

    # Detect hidden_dim from a test encode
    test_enc  = client.encode("test", args.layer)
    hidden_dim = test_enc.get("hidden_dim")
    if hidden_dim is None:
        # Try to parse from hidden_states
        for val in test_enc.get("hidden_states", {}).values():
            if isinstance(val, str):
                flat = np.frombuffer(base64.b64decode(val), dtype=np.float32)
                # single token → flat is [hidden_dim]
                hidden_dim = len(flat)
                break
    if hidden_dim is None:
        raise RuntimeError("Could not determine hidden_dim from test encode")
    log.info(f"hidden_dim={hidden_dim}, layer={args.layer}")

    n_features = args.expansion * hidden_dim
    log.info(f"n_features={n_features} (expansion={args.expansion}×{hidden_dim})")

    # ── Output directory ──
    safe_model = model_name.replace("/", "_").replace(" ", "_").lower()[:40]
    output_dir = Path(args.output) if args.output else Path(f"checkpoints/sae_layer{args.layer}_{safe_model}")
    cache_dir  = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"hidden_states_layer{args.layer}_{safe_model}_{args.n_texts}.npy"
    if args.force_collect and cache_path.exists():
        cache_path.unlink()
        log.info("Deleted old cache")

    # ── Phase 1: Collect ──
    log.info("="*60 + "\nPhase 1: Collecting hidden states\n" + "="*60)
    texts = _build_corpus(args.n_texts, args.seed)
    log.info(f"  Corpus: {len(texts)} texts")

    data = collect_hidden_states(
        client, texts, args.layer, cache_path,
        hidden_dim, max_seq_len=args.max_seq_len,
    )
    client.close()
    log.info(f"  Dataset: {data.shape[0]:,} token vectors × {data.shape[1]} dims")

    if args.skip_train:
        log.info("--skip-train set, exiting after collection")
        return

    # ── Train/val split ──
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(data))
    n_val = max(256, int(len(data) * args.val_split))
    val_data   = data[perm[:n_val]]
    train_data = data[perm[n_val:]]
    log.info(f"  Train: {len(train_data):,}  Val: {len(val_data):,}")

    # ── Phase 2: Train ──
    log.info("="*60 + "\nPhase 2: Training SAE\n" + "="*60)
    log.info(f"  hidden_dim={hidden_dim}  n_features={n_features}  "
             f"epochs={args.epochs}  l1={args.l1_coef}  lr={args.lr}")

    sae = SparseAutoencoder(hidden_dim, n_features)
    history = train(
        sae, train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_coef=args.l1_coef,
        warmup_steps=args.warmup_steps,
        resample_every=args.resample_every,
        device=args.device,
    )

    # ── Validate ──
    log.info("="*60 + "\nValidation\n" + "="*60)
    metrics = evaluate(sae, val_data, args.l1_coef, args.device)
    log.info(
        f"  R²={metrics['r2']:.4f}  "
        f"L0={metrics['l0_mean']:.3f}  "
        f"MSE={metrics['mse']:.6f}  "
        f"dead={metrics['dead_features']}/{n_features} ({metrics['pct_dead']}%)"
    )

    # Quality warnings
    if metrics["r2"] < 0.7:
        log.warning(f"  ⚠ R²={metrics['r2']:.4f} is low — try more data or more epochs")
    if metrics["pct_dead"] > 20:
        log.warning(f"  ⚠ {metrics['pct_dead']}% dead features — try lower l1_coef or more resampling")
    if metrics["l0_mean"] > 0.1:
        log.warning(f"  ⚠ L0={metrics['l0_mean']:.3f} — features not sparse enough, try higher l1_coef")

    # ── Phase 3: Auto-label (optional) ──
    labels = {}
    if args.auto_label:
        log.info("="*60 + "\nPhase 3: Auto-labelling features\n" + "="*60)
        labels = auto_label_features(sae, train_data, texts)

    # ── Save ──
    log.info("="*60 + "\nSaving checkpoint\n" + "="*60)
    cfg = {
        "model_name":     model_name,
        "layer":          args.layer,
        "hidden_dim":     hidden_dim,
        "n_features":     n_features,
        "expansion":      args.expansion,
        "l1_coef":        args.l1_coef,
        "epochs":         args.epochs,
        "n_train_tokens": len(train_data),
        "val_r2":         metrics["r2"],
        "val_l0":         metrics["l0_mean"],
        "val_pct_dead":   metrics["pct_dead"],
    }
    save_checkpoint(sae, output_dir, cfg, labels, history)

    # Print final summary
    print(f"\n{'═'*60}")
    print(f"  SAE Training Complete")
    print(f"{'═'*60}")
    print(f"  Model:       {model_name}")
    print(f"  Layer:       {args.layer}")
    print(f"  Features:    {n_features:,}  (expansion {args.expansion}×)")
    print(f"  R²:          {metrics['r2']:.4f}  {'✅' if metrics['r2'] >= 0.7 else '⚠ low'}")
    print(f"  L0 (mean):   {metrics['l0_mean']:.3f}  features active per token")
    print(f"  Dead:        {metrics['dead_features']}/{n_features} ({metrics['pct_dead']}%)")
    print(f"  Checkpoint:  {output_dir.resolve()}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
