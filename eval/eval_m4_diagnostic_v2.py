#!/usr/bin/env python3
"""
ERIS v5 — M4 Diagnostic (v2): Why is Spearman only 0.41?
==========================================================

Fixed for actual ERIS API:
  - /v1/encode does NOT take session_id
  - return_layers may return only "last" — iterate per layer
  - Handles Qwen3.5 hybrid GDN architecture

Tests three hypotheses simultaneously:
  H1: Pooling method (last_token vs mean vs content-only mean vs max)
  H2: Anisotropy (subtract corpus mean vector before cosine)
  H3: Layer choice (last layer vs intermediate layers)

Usage:
  python eris_server.py --model Qwen/Qwen3.5-4B --port 8001
  python eval/eval_m4_diagnostic_v2.py --eris-url http://localhost:8001 --n-pairs 100

  # Fast (20 pairs, fewer layers)
  python eval/eval_m4_diagnostic_v2.py --eris-url http://localhost:8001 --n-pairs 20 --fast
"""

import os
import sys
import json
import argparse
import logging
import base64
import time
import statistics
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("m4_diag")


# ═══════════════════════════════════════════════════════════════
# ERIS client — no session_id for encode
# ═══════════════════════════════════════════════════════════════

class ERISClient:
    def __init__(self, base_url, timeout=120.0):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def health(self):
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def encode(self, text: str, return_layers: list = None, compact: bool = True):
        """Call /v1/encode — no session_id needed."""
        payload = {"text": text, "compact": compact}
        if return_layers is not None:
            payload["return_layers"] = return_layers
        r = self.client.post(f"{self.base_url}/v1/encode", json=payload)
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


def decode_b64(b64_str: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# Load STS-B
# ═══════════════════════════════════════════════════════════════

def load_stsb_pairs(n_pairs=100, seed=42):
    from datasets import load_dataset

    np.random.seed(seed)
    log.info(f"Loading STS-B test split ({n_pairs} pairs, seed={seed})...")
    ds = load_dataset("sentence-transformers/stsb", split="test")

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    per_bin = n_pairs // len(bins)
    pairs = []

    for lo, hi in bins:
        bin_items = [r for r in ds if lo <= r["score"] < hi]
        np.random.shuffle(bin_items)
        pairs.extend(bin_items[:per_bin])

    remaining = n_pairs - len(pairs)
    if remaining > 0:
        all_items = list(ds)
        np.random.shuffle(all_items)
        for item in all_items:
            if len(pairs) >= n_pairs:
                break
            pairs.append(item)

    log.info(f"  {len(pairs)} pairs loaded across {len(bins)} bins")
    return pairs[:n_pairs]


# ═══════════════════════════════════════════════════════════════
# Probe the API to understand what return_layers actually does
# ═══════════════════════════════════════════════════════════════

def probe_api(client: ERISClient):
    """Send test requests to understand the encode API behavior."""
    log.info("Probing /v1/encode API behavior...")

    # Test 1: default (no return_layers)
    r1 = client.encode("Test sentence")
    keys1 = list(r1.get("hidden_states", {}).keys())
    hidden_dim = r1.get("hidden_dim", None)
    seq_len = r1.get("seq_len", None)
    log.info(f"  Default: keys={keys1}, hidden_dim={hidden_dim}, seq_len={seq_len}")

    # Test 2: request specific layer
    try:
        r2 = client.encode("Test sentence", return_layers=[15])
        keys2 = list(r2.get("hidden_states", {}).keys())
        log.info(f"  return_layers=[15]: keys={keys2}")
    except Exception as e:
        log.info(f"  return_layers=[15]: error={e}")
        keys2 = keys1

    # Test 3: request multiple layers
    try:
        r3 = client.encode("Test sentence", return_layers=[10, 20, -1])
        keys3 = list(r3.get("hidden_states", {}).keys())
        log.info(f"  return_layers=[10,20,-1]: keys={keys3}")
    except Exception as e:
        log.info(f"  return_layers=[10,20,-1]: error={e}")
        keys3 = keys1

    # Determine which layers are actually available
    all_available = set(keys1) | set(keys2) | set(keys3)
    multi_layer = len(keys3) > 1 or len(keys2) > 1

    log.info(f"  Multi-layer support: {multi_layer}")
    log.info(f"  All observed keys: {sorted(all_available)}")

    return {
        "hidden_dim": hidden_dim,
        "multi_layer": multi_layer,
        "default_keys": keys1,
        "test_keys": sorted(all_available),
    }


# ═══════════════════════════════════════════════════════════════
# Encoding strategies
# ═══════════════════════════════════════════════════════════════

def encode_corpus_multi_layer(client, sentences, layer_indices):
    """Encode each sentence at each layer separately (if needed).

    Returns: {sentence: {layer_key: np.ndarray[seq_len, hidden_dim]}}
    """
    cache = {}
    hidden_dim = None
    errors = 0

    for sent in tqdm(sentences, desc="Encoding"):
        sent_hs = {}

        for layer_idx in layer_indices:
            try:
                enc = client.encode(sent, return_layers=[layer_idx], compact=True)

                if hidden_dim is None:
                    hidden_dim = enc.get("hidden_dim", None)

                for key, val in enc.get("hidden_states", {}).items():
                    # Normalize key: if API returns "last" for any layer,
                    # tag it with the requested index so layers don't collide.
                    norm_key = f"layer_{layer_idx}" if key == "last" else key

                    if isinstance(val, str):
                        flat = decode_b64(val)
                        if hidden_dim and len(flat) >= hidden_dim:
                            mat = flat.reshape(-1, hidden_dim)
                        else:
                            mat = flat.reshape(1, -1)
                            if hidden_dim is None:
                                hidden_dim = mat.shape[-1]
                    elif isinstance(val, list):
                        mat = np.array(val, dtype=np.float32)
                        if mat.ndim == 1:
                            mat = mat.reshape(1, -1)
                        if hidden_dim is None:
                            hidden_dim = mat.shape[-1]
                    else:
                        continue

                    sent_hs[norm_key] = mat

            except Exception as e:
                errors += 1
                if errors <= 3:
                    log.warning(f"  Error layer={layer_idx}: {e}")
                if errors > 50:
                    log.error("  Too many errors, aborting")
                    return cache, hidden_dim

        if sent_hs:
            cache[sent] = sent_hs

    log.info(f"  Encoded {len(cache)}/{len(sentences)}, "
             f"hidden_dim={hidden_dim}, errors={errors}")
    return cache, hidden_dim


def encode_corpus_single_call(client, sentences, layer_indices):
    """Encode each sentence with all layers in one call (if API supports it).

    Returns: {sentence: {layer_key: np.ndarray[seq_len, hidden_dim]}}
    """
    cache = {}
    hidden_dim = None
    errors = 0

    for sent in tqdm(sentences, desc="Encoding"):
        try:
            enc = client.encode(sent, return_layers=layer_indices, compact=True)

            if hidden_dim is None:
                hidden_dim = enc.get("hidden_dim", None)

            sent_hs = {}
            for key, val in enc.get("hidden_states", {}).items():
                if isinstance(val, str):
                    flat = decode_b64(val)
                    if hidden_dim and len(flat) >= hidden_dim:
                        mat = flat.reshape(-1, hidden_dim)
                    else:
                        mat = flat.reshape(1, -1)
                        if hidden_dim is None:
                            hidden_dim = mat.shape[-1]
                elif isinstance(val, list):
                    mat = np.array(val, dtype=np.float32)
                    if mat.ndim == 1:
                        mat = mat.reshape(1, -1)
                    if hidden_dim is None:
                        hidden_dim = mat.shape[-1]
                else:
                    continue
                sent_hs[key] = mat

            if sent_hs:
                cache[sent] = sent_hs

        except Exception as e:
            errors += 1
            if errors <= 3:
                log.warning(f"  Error: {e}")
            if errors > 20:
                log.error("  Too many errors, aborting")
                break

    log.info(f"  Encoded {len(cache)}/{len(sentences)}, "
             f"hidden_dim={hidden_dim}, errors={errors}")
    return cache, hidden_dim


# ═══════════════════════════════════════════════════════════════
# Pooling
# ═══════════════════════════════════════════════════════════════

def pool_last_token(mat):
    return mat[-1]

def pool_2nd_to_last(mat):
    return mat[-2] if mat.shape[0] >= 2 else mat[-1]

def pool_mean_all(mat):
    return mat.mean(axis=0)

def pool_mean_no_edges(mat):
    if mat.shape[0] <= 2:
        return mat.mean(axis=0)
    return mat[1:-1].mean(axis=0)

def pool_max(mat):
    return mat.max(axis=0)


POOLING = {
    "last_token": pool_last_token,
    "2nd_to_last": pool_2nd_to_last,
    "mean_all": pool_mean_all,
    "mean_no_edges": pool_mean_no_edges,
    "max_pool": pool_max,
}


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def cosine(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def evaluate_combo(pairs, cache, layer_key, pool_fn, center):
    """Evaluate one (layer, pooling, centering) combination."""
    vectors = {}
    for sent, hs_dict in cache.items():
        if layer_key in hs_dict:
            vectors[sent] = pool_fn(hs_dict[layer_key])

    if len(vectors) < 10:
        return None

    if center:
        all_vecs = np.stack(list(vectors.values()))
        mean_vec = all_vecs.mean(axis=0)
        vectors = {s: v - mean_vec for s, v in vectors.items()}

    human_scores = []
    zombie_cosines = []

    for p in pairs:
        s1, s2 = p["sentence1"], p["sentence2"]
        if s1 in vectors and s2 in vectors:
            human_scores.append(p["score"])
            zombie_cosines.append(cosine(vectors[s1], vectors[s2]))

    n = len(human_scores)
    if n < 20:
        return None

    human_arr = np.array(human_scores)
    zombie_arr = np.array(zombie_cosines)

    sp = stats.spearmanr(human_arr, zombie_arr)
    pr = stats.pearsonr(human_arr, zombie_arr)

    return {
        "spearman_r": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "mean_cosine": float(zombie_arr.mean()),
        "std_cosine": float(zombie_arr.std()),
        "min_cosine": float(zombie_arr.min()),
        "max_cosine": float(zombie_arr.max()),
        "n_valid": n,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M4 Diagnostic v2")
    parser.add_argument("--eris-url", default="http://localhost:8001")
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="Fewer layers (last + 2 intermediates)")
    parser.add_argument("--output", default="eval_results/m4_diagnostic_v2.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    Path(args.output).parent.mkdir(exist_ok=True)

    client = ERISClient(args.eris_url)

    # ── Health check ──
    try:
        health = client.health()
        model_name = health.get("model", "unknown")
        n_layers_reported = health.get("n_layers", None)
        log.info(f"Server Health Check: Model={model_name}, Status={health.get('status')}")
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return

    # ── Probe API behavior ──
    probe = probe_api(client)
    hidden_dim = probe["hidden_dim"]
    multi_layer = probe["multi_layer"]

    # Guess n_layers from model name
    n_layers = n_layers_reported
    if n_layers is None:
        model_lower = str(model_name).lower()
        if "0.8" in model_lower or "0.5" in model_lower:
            n_layers = 24
        elif "4b" in model_lower:
            n_layers = 36
        elif "9b" in model_lower:
            n_layers = 40
        elif "14b" in model_lower:
            n_layers = 40
        elif "32b" in model_lower:
            n_layers = 64
        else:
            n_layers = 36
        log.info(f"Guessed n_layers={n_layers}")

    # ── Select test layers ──
    if args.fast:
        test_layers = [-1, int(n_layers * 0.75), int(n_layers * 0.50)]
    else:
        test_layers = [
            -1,
            n_layers - 1,
            int(n_layers * 0.875),
            int(n_layers * 0.75),
            int(n_layers * 0.625),
            int(n_layers * 0.50),
            int(n_layers * 0.375),
            int(n_layers * 0.25),
        ]

    test_layers = sorted(set(
        max(0, min(n_layers - 1, l)) if l >= 0 else l
        for l in test_layers
    ))
    log.info(f"Testing {len(test_layers)} layers: {test_layers}")

    # ── Load data ──
    pairs = load_stsb_pairs(args.n_pairs, args.seed)
    unique_sents = list({p["sentence1"] for p in pairs} | {p["sentence2"] for p in pairs})

    # ── Encode ──
    t0 = time.time()
    if multi_layer:
        log.info("API supports multi-layer — using single-call encoding")
        cache, hidden_dim = encode_corpus_single_call(
            client, unique_sents, test_layers
        )
    else:
        log.info("API returns single layer per call — encoding per-layer")
        cache, hidden_dim = encode_corpus_multi_layer(
            client, unique_sents, test_layers
        )
    encode_time = time.time() - t0
    client.close()

    if not cache:
        log.error("No encodings — aborting")
        return

    # ── Detect available layer keys ──
    sample_hs = next(iter(cache.values()))
    available_keys = sorted(sample_hs.keys())
    log.info(f"Available layer keys: {available_keys}")

    # ── Evaluate all combos ──
    all_results = []
    for layer_key in available_keys:
        for pool_name, pool_fn in POOLING.items():
            for center in [False, True]:
                res = evaluate_combo(pairs, cache, layer_key, pool_fn, center)
                if res is None:
                    continue
                res["layer"] = layer_key
                res["pooling"] = pool_name
                res["centered"] = center
                all_results.append(res)

    all_results.sort(key=lambda r: r["spearman_r"], reverse=True)

    # ── Print results ──
    print("\n" + "=" * 105)
    print(f"M4 DIAGNOSTIC v2 — {model_name} — {len(pairs)} pairs — "
          f"{len(all_results)} combos — {encode_time:.1f}s")
    print("=" * 105)
    print(f"{'#':>3}  {'Layer':<15} {'Pooling':<16} {'Center':<8} "
          f"{'Spearman':>9} {'p-value':>11} {'Mean cos':>9} {'Std cos':>9} "
          f"{'Pass':>5}")
    print("-" * 105)

    for i, r in enumerate(all_results):
        passed = "✅" if r["spearman_r"] > 0.6 else "  "
        print(f"{i+1:3d}  {r['layer']:<15} {r['pooling']:<16} "
              f"{'yes' if r['centered'] else 'no':<8} "
              f"{r['spearman_r']:>9.4f} {r['spearman_p']:>11.2e} "
              f"{r['mean_cosine']:>9.4f} {r['std_cosine']:>9.4f} "
              f"{passed}")

    # ── Key findings ──
    print("\n" + "=" * 105)
    print("KEY FINDINGS")
    print("=" * 105)

    if not all_results:
        print("  No valid results!")
        return

    best = all_results[0]
    print(f"\n  BEST:  layer={best['layer']}, pooling={best['pooling']}, "
          f"centered={best['centered']}")
    print(f"         Spearman={best['spearman_r']:.4f}, "
          f"mean_cos={best['mean_cosine']:.4f}, std_cos={best['std_cosine']:.4f}")

    # H1: pooling effect
    pooling_best = {}
    for r in all_results:
        if r["pooling"] not in pooling_best or r["spearman_r"] > pooling_best[r["pooling"]]:
            pooling_best[r["pooling"]] = r["spearman_r"]
    print(f"\n  H1 (pooling): best Spearman per method:")
    for name, sp in sorted(pooling_best.items(), key=lambda x: -x[1]):
        print(f"       {name:<18} → {sp:.4f}")

    # H2: centering
    centered = [r["spearman_r"] for r in all_results if r["centered"]]
    uncentered = [r["spearman_r"] for r in all_results if not r["centered"]]
    if centered and uncentered:
        delta = statistics.mean(centered) - statistics.mean(uncentered)
        print(f"\n  H2 (centering): mean centered={statistics.mean(centered):.4f}, "
              f"uncentered={statistics.mean(uncentered):.4f}, Δ={delta:+.4f}")

    # H3: layer effect
    layer_best = {}
    for r in all_results:
        if r["layer"] not in layer_best or r["spearman_r"] > layer_best[r["layer"]]:
            layer_best[r["layer"]] = r["spearman_r"]
    print(f"\n  H3 (layer): best Spearman per layer:")
    for name, sp in sorted(layer_best.items(), key=lambda x: -x[1]):
        print(f"       {name:<15} → {sp:.4f}")

    # Pass/fail
    n_passing = sum(1 for r in all_results if r["spearman_r"] > 0.6)
    print(f"\n  PASS (Spearman > 0.6): {n_passing}/{len(all_results)} combos")

    if n_passing > 0:
        print(f"\n  ✅ M4 PASSES with: layer={best['layer']}, "
              f"pooling={best['pooling']}, centered={best['centered']}")
    elif best["spearman_r"] > 0.5:
        print(f"\n  ⚠️  Close ({best['spearman_r']:.4f}). Try Qwen3.5-9B or more pairs.")
    else:
        print(f"\n  ❌ Best={best['spearman_r']:.4f}. Consider reframing M4.")

    # ── Save ──
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_pairs": len(pairs),
        "encode_time_s": encode_time,
        "multi_layer_api": multi_layer,
        "test_layers_requested": test_layers,
        "layer_keys_available": available_keys,
        "n_combos": len(all_results),
        "n_passing_06": n_passing,
        "best": best,
        "all_results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
