#!/usr/bin/env python3
"""
ERIS v5 — M4 Diagnostic v3
============================

Fixed: encodes ONE layer per API call (multi-layer hits max_payload_bytes).

Usage:
  python eris_server.py --model Qwen/Qwen3.5-4B --port 8001
  python eval/eval_m4_diagnostic_v3.py --eris-url http://localhost:8001 --n-pairs 100
  python eval/eval_m4_diagnostic_v3.py --eris-url http://localhost:8001 --n-pairs 20 --fast
"""

import json
import argparse
import logging
import base64
import time
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("m4_diag")


# ═══════════════════════════════════════════════════════════════
# Client
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

    def encode_single_layer(self, text: str, layer: int) -> dict:
        """Encode text and return ONE layer's hidden states."""
        payload = {"text": text, "return_layers": [layer], "compact": True}
        r = self.client.post(f"{self.base_url}/v1/encode", json=payload)
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


def decode_b64(b64_str: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# STS-B
# ═══════════════════════════════════════════════════════════════

def load_stsb_pairs(n_pairs=100, seed=42):
    from datasets import load_dataset
    np.random.seed(seed)
    log.info(f"Loading STS-B test split ({n_pairs} pairs)...")
    ds = load_dataset("sentence-transformers/stsb", split="test")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    per_bin = n_pairs // len(bins)
    pairs = []
    for lo, hi in bins:
        items = [r for r in ds if lo <= r["score"] < hi]
        np.random.shuffle(items)
        pairs.extend(items[:per_bin])
    remaining = n_pairs - len(pairs)
    if remaining > 0:
        all_items = list(ds)
        np.random.shuffle(all_items)
        pairs.extend(all_items[:remaining])
    log.info(f"  {len(pairs[:n_pairs])} pairs loaded")
    return pairs[:n_pairs]


# ═══════════════════════════════════════════════════════════════
# Encode corpus — ONE layer per call
# ═══════════════════════════════════════════════════════════════

def encode_corpus(client: ERISClient, sentences: list, layers: list, hidden_dim_hint=None):
    """Encode all sentences, one layer at a time.

    Returns: {sentence: {layer_key: np.ndarray[seq_len, hidden_dim]}}
    """
    cache = {s: {} for s in sentences}
    hidden_dim = hidden_dim_hint
    errors = 0
    total_calls = len(sentences) * len(layers)

    log.info(f"Encoding {len(sentences)} sentences × {len(layers)} layers "
             f"= {total_calls} API calls...")

    for layer_idx in layers:
        layer_key = f"layer_{layer_idx}" if layer_idx >= 0 else "last"
        desc = f"Layer {layer_idx}"

        for sent in tqdm(sentences, desc=desc, leave=False):
            try:
                enc = client.encode_single_layer(sent, layer_idx)

                if hidden_dim is None:
                    hidden_dim = enc.get("hidden_dim")

                # Extract the single hidden state from response
                hs = enc.get("hidden_states", {})
                # API returns "last" for -1, "layer_N" for N
                val = None
                for k, v in hs.items():
                    val = v
                    break  # only one key expected

                if val is None:
                    errors += 1
                    continue

                if isinstance(val, str):
                    flat = decode_b64(val)
                    mat = flat.reshape(-1, hidden_dim) if hidden_dim else flat.reshape(1, -1)
                    if hidden_dim is None:
                        hidden_dim = mat.shape[-1]
                elif isinstance(val, list):
                    mat = np.array(val, dtype=np.float32)
                    if mat.ndim == 1:
                        mat = mat.reshape(1, -1)
                    if hidden_dim is None:
                        hidden_dim = mat.shape[-1]
                else:
                    errors += 1
                    continue

                cache[sent][layer_key] = mat

            except Exception as e:
                errors += 1
                if errors <= 5:
                    log.warning(f"  Error: {e}")
                if errors > 100:
                    log.error("  Too many errors, aborting")
                    return cache, hidden_dim

        log.info(f"  Layer {layer_idx}: done")

    n_encoded = sum(1 for s in cache.values() if s)
    log.info(f"  {n_encoded}/{len(sentences)} sentences encoded, "
             f"hidden_dim={hidden_dim}, errors={errors}")
    return cache, hidden_dim


# ═══════════════════════════════════════════════════════════════
# Pooling
# ═══════════════════════════════════════════════════════════════

POOLING = {
    "last_token":     lambda m: m[-1],
    "2nd_to_last":    lambda m: m[-2] if m.shape[0] >= 2 else m[-1],
    "mean_all":       lambda m: m.mean(axis=0),
    "mean_no_edges":  lambda m: m[1:-1].mean(axis=0) if m.shape[0] > 2 else m.mean(axis=0),
    "max_pool":       lambda m: m.max(axis=0),
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

    human_scores, zombie_cosines = [], []
    for p in pairs:
        s1, s2 = p["sentence1"], p["sentence2"]
        if s1 in vectors and s2 in vectors:
            human_scores.append(p["score"])
            zombie_cosines.append(cosine(vectors[s1], vectors[s2]))

    if len(human_scores) < 20:
        return None

    h, z = np.array(human_scores), np.array(zombie_cosines)
    sp = stats.spearmanr(h, z)
    pr = stats.pearsonr(h, z)

    return {
        "spearman_r": float(sp.statistic), "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "mean_cosine": float(z.mean()), "std_cosine": float(z.std()),
        "min_cosine": float(z.min()), "max_cosine": float(z.max()),
        "n_valid": len(human_scores),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M4 Diagnostic v3")
    parser.add_argument("--eris-url", default="http://localhost:8001")
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", default="eval_results/m4_diagnostic_v3.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    Path(args.output).parent.mkdir(exist_ok=True)

    client = ERISClient(args.eris_url)

    # Health
    health = client.health()
    model_name = health.get("model", "unknown")
    log.info(f"Model: {model_name}")

    # Verify single-layer encode works
    test = client.encode_single_layer("Test", -1)
    hidden_dim = test.get("hidden_dim")
    log.info(f"Verified: hidden_dim={hidden_dim}")

    # Guess n_layers
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
    log.info(f"n_layers={n_layers} (guessed)")

    # Test layers
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
    log.info(f"Layers: {test_layers}")

    # Load data
    pairs = load_stsb_pairs(args.n_pairs, args.seed)
    unique = list({p["sentence1"] for p in pairs} | {p["sentence2"] for p in pairs})

    # Encode — one layer per call
    t0 = time.time()
    cache, hidden_dim = encode_corpus(client, unique, test_layers, hidden_dim)
    encode_time = time.time() - t0
    client.close()

    if not any(v for v in cache.values()):
        log.error("No encodings — aborting")
        return

    # Available keys
    all_keys = set()
    for hs in cache.values():
        all_keys.update(hs.keys())
    all_keys = sorted(all_keys)
    log.info(f"Layer keys: {all_keys}")

    # Evaluate all combos
    all_results = []
    for lk in all_keys:
        for pn, pf in POOLING.items():
            for c in [False, True]:
                res = evaluate_combo(pairs, cache, lk, pf, c)
                if res:
                    res.update({"layer": lk, "pooling": pn, "centered": c})
                    all_results.append(res)

    all_results.sort(key=lambda r: r["spearman_r"], reverse=True)

    # Print
    print(f"\n{'='*105}")
    print(f"M4 DIAGNOSTIC v3 — {model_name} — {len(pairs)} pairs — "
          f"{len(all_results)} combos — {encode_time:.0f}s")
    print(f"{'='*105}")
    print(f"{'#':>3}  {'Layer':<15} {'Pooling':<16} {'Center':<8} "
          f"{'Spearman':>9} {'p-value':>11} {'Mean cos':>9} {'Std cos':>9} {'':>5}")
    print("-" * 105)

    for i, r in enumerate(all_results):
        flag = "✅" if r["spearman_r"] > 0.6 else "  "
        print(f"{i+1:3d}  {r['layer']:<15} {r['pooling']:<16} "
              f"{'yes' if r['centered'] else 'no':<8} "
              f"{r['spearman_r']:>9.4f} {r['spearman_p']:>11.2e} "
              f"{r['mean_cosine']:>9.4f} {r['std_cosine']:>9.4f} {flag}")

    # Summary
    print(f"\n{'='*105}")
    print("SUMMARY")
    print(f"{'='*105}")

    if all_results:
        best = all_results[0]
        print(f"\n  BEST: {best['layer']} / {best['pooling']} / "
              f"centered={best['centered']} → Spearman={best['spearman_r']:.4f}")

        # H1
        pb = {}
        for r in all_results:
            if r["pooling"] not in pb or r["spearman_r"] > pb[r["pooling"]]:
                pb[r["pooling"]] = r["spearman_r"]
        print(f"\n  H1 pooling:")
        for k, v in sorted(pb.items(), key=lambda x: -x[1]):
            print(f"    {k:<18} → {v:.4f}")

        # H2
        c_scores = [r["spearman_r"] for r in all_results if r["centered"]]
        u_scores = [r["spearman_r"] for r in all_results if not r["centered"]]
        if c_scores and u_scores:
            d = np.mean(c_scores) - np.mean(u_scores)
            print(f"\n  H2 centering: Δ={d:+.4f} "
                  f"(centered={np.mean(c_scores):.4f}, raw={np.mean(u_scores):.4f})")

        # H3
        lb = {}
        for r in all_results:
            if r["layer"] not in lb or r["spearman_r"] > lb[r["layer"]]:
                lb[r["layer"]] = r["spearman_r"]
        print(f"\n  H3 layer:")
        for k, v in sorted(lb.items(), key=lambda x: -x[1]):
            print(f"    {k:<15} → {v:.4f}")

        n_pass = sum(1 for r in all_results if r["spearman_r"] > 0.6)
        print(f"\n  PASS (>0.6): {n_pass}/{len(all_results)}")

        if n_pass > 0:
            print(f"\n  ✅ Use: layer={best['layer']}, pooling={best['pooling']}, "
                  f"centered={best['centered']}")
        elif best["spearman_r"] > 0.5:
            print(f"\n  ⚠️  Close. Try Qwen3.5-9B or more pairs.")
        else:
            print(f"\n  ❌ Reframe M4.")

    # Save
    output = {
        "model": model_name, "n_layers": n_layers, "hidden_dim": hidden_dim,
        "n_pairs": len(pairs), "encode_time_s": encode_time,
        "test_layers": test_layers, "layer_keys": all_keys,
        "n_combos": len(all_results),
        "n_passing": sum(1 for r in all_results if r["spearman_r"] > 0.6),
        "best": all_results[0] if all_results else None,
        "all_results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
