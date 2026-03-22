#!/usr/bin/env python3
"""
ERIS v5 — M4 Diagnostic: Why is Spearman only 0.41?
=====================================================

Symptom: mean cosine = 0.94, all pairs clustered → variance crushed → low correlation.

Tests three hypotheses simultaneously:
  H1: Pooling method (last_token vs mean_all vs mean_no_edges vs max_pool)
  H2: Anisotropy (subtract corpus mean vector before cosine — "centering")
  H3: Layer choice (last layer vs intermediate layers)

For Qwen3.5-4B (36 layers: 18 GDN linear + 6 full attention, interleaved):
  - GDN layers use fixed-size state matrix (no growing KV-cache)
  - Full attention layers use standard KV-cache
  - Semantic content may concentrate differently vs standard transformers

Outputs a ranked table: every (layer × pooling × centering) combo sorted by Spearman.
The winning combo becomes the default for eval_phase1.py.

Usage:
  python eris_server.py --model Qwen/Qwen3.5-4B --port 8001
  python eval/eval_m4_diagnostic.py --eris-url http://localhost:8001 --n-pairs 100

  # Fast test (20 pairs, ~2 min)
  python eval/eval_m4_diagnostic.py --eris-url http://localhost:8001 --n-pairs 20 --fast
"""

import os
import sys
import json
import argparse
import logging
import base64
import time
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
# ERIS client
# ═══════════════════════════════════════════════════════════════

class ERISEvalClient:
    def __init__(self, base_url, timeout=120.0):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self._session_id = None

    def _ensure_session(self):
        if self._session_id is None:
            r = self.client.post(f"{self.base_url}/sessions")
            r.raise_for_status()
            self._session_id = r.json()["session_id"]
        return self._session_id

    def encode(self, text, return_layers=None):
        sid = self._ensure_session()
        payload = {
            "text": text,
            "session_id": sid,
            "return_layers": return_layers or [-1],
            "compact": True,
        }
        r = self.client.post(f"{self.base_url}/v1/encode", json=payload)
        r.raise_for_status()
        return r.json()

    def health(self):
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def preflight(self):
        """Verify the server has ERIS endpoints (/v1/encode).

        Raises SystemExit with a clear message if the server is the base
        server.py instead of eris_server.py.
        """
        try:
            # Probe /v1/encode with an empty body — expect 422 (validation
            # error), not 404 (route missing) or 405 (method not allowed).
            import httpx
            r = self.client.post(f"{self.base_url}/v1/encode", json={})
            if r.status_code == 404:
                log.error(
                    "\n"
                    "  ❌ /v1/encode returned 404 — the server running is server.py\n"
                    "     (base server), not eris_server.py (ERIS endpoints).\n"
                    "\n"
                    "  Fix: start eris_server.py instead:\n"
                    f"     python eris_server.py --model <MODEL> --port 8001\n"
                    "  or:\n"
                    f"     ERIS_CONFIG=configs/eris_config.yaml uvicorn eris_server:app "
                    f"--host 0.0.0.0 --port 8001\n"
                )
                sys.exit(1)
            # 422 = route exists, validation failed (expected for empty body)
            # 200 = route exists and somehow succeeded (unlikely but fine)
            # Anything else: warn but continue
            if r.status_code not in (200, 422):
                log.warning(f"  Preflight /v1/encode returned unexpected {r.status_code} — continuing anyway")
        except Exception as e:
            log.warning(f"  Preflight check failed: {e} — continuing anyway")

    def close(self):
        if self._session_id:
            try:
                self.client.delete(
                    f"{self.base_url}/sessions/{self._session_id}"
                )
            except Exception:
                pass
        self.client.close()


def decode_b64(b64_str: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# Load STS-B (stratified)
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

    # Fill remainder
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
# Encode all unique sentences at multiple layers
# ═══════════════════════════════════════════════════════════════

def encode_corpus(client, pairs, layer_indices):
    """Encode all unique sentences, return cache and hidden_dim.

    Returns:
        cache: {sentence_text: {layer_key: np.ndarray[seq_len, hidden_dim]}}
        hidden_dim: int
    """
    unique = list({p["sentence1"] for p in pairs} | {p["sentence2"] for p in pairs})
    log.info(f"Encoding {len(unique)} unique sentences at layers {layer_indices}...")

    cache = {}
    hidden_dim = None
    errors = 0

    for sent in tqdm(unique, desc="Encoding"):
        try:
            enc = client.encode(sent, return_layers=layer_indices)

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

            cache[sent] = sent_hs

        except Exception as e:
            errors += 1
            if errors <= 3:
                log.warning(f"  Encode error: {e}")
            if errors > 20:
                log.error("  Too many errors, stopping encoding")
                break

    log.info(f"  Encoded {len(cache)}/{len(unique)} sentences, "
             f"hidden_dim={hidden_dim}, errors={errors}")
    return cache, hidden_dim


# ═══════════════════════════════════════════════════════════════
# Pooling strategies
# ═══════════════════════════════════════════════════════════════

def pool_last_token(mat):
    """Hidden state of the last token."""
    return mat[-1]


def pool_mean_all(mat):
    """Mean across all token positions."""
    return mat.mean(axis=0)


def pool_mean_no_edges(mat):
    """Mean excluding first (BOS) and last (EOS/pad) tokens."""
    if mat.shape[0] <= 2:
        return mat.mean(axis=0)
    return mat[1:-1].mean(axis=0)


def pool_max(mat):
    """Element-wise max across token positions."""
    return mat.max(axis=0)


def pool_second_to_last(mat):
    """Second-to-last token — often better than last for causal LMs."""
    if mat.shape[0] < 2:
        return mat[-1]
    return mat[-2]


POOLING = {
    "last_token": pool_last_token,
    "2nd_to_last": pool_second_to_last,
    "mean_all": pool_mean_all,
    "mean_no_edges": pool_mean_no_edges,
    "max_pool": pool_max,
}


# ═══════════════════════════════════════════════════════════════
# Similarity computation
# ═══════════════════════════════════════════════════════════════

def cosine(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def evaluate_combo(pairs, cache, layer_key, pool_fn, center):
    """Evaluate one (layer, pooling, centering) combination.

    Returns dict with spearman, pearson, mean_cosine, std_cosine, n_valid.
    """
    # Pool all vectors
    vectors = {}
    for sent, hs_dict in cache.items():
        if layer_key in hs_dict:
            vectors[sent] = pool_fn(hs_dict[layer_key])

    if len(vectors) < 10:
        return None

    # Centering (anisotropy correction)
    if center:
        all_vecs = np.stack(list(vectors.values()))
        mean_vec = all_vecs.mean(axis=0)
        vectors = {s: v - mean_vec for s, v in vectors.items()}

    # Compute cosine for each pair
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
    parser = argparse.ArgumentParser(description="M4 Diagnostic — find optimal config")
    parser.add_argument("--eris-url", default="http://localhost:8001")
    parser.add_argument("--n-pairs", type=int, default=100,
                        help="Number of STS-B pairs (100 is fast, 200 is thorough)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="Test fewer layers (last + 2 intermediates only)")
    parser.add_argument("--output", default="eval_results/m4_diagnostic.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    Path(args.output).parent.mkdir(exist_ok=True)

    client = ERISEvalClient(args.eris_url)

    # ── Preflight: verify ERIS endpoints are available ──
    client.preflight()

    # ── Detect model config ──
    try:
        health = client.health()
        model_name = health.get("model", "unknown")
        n_layers = health.get("n_layers", None)
        log.info(f"Server Health Check: Model={model_name}, Status={health.get('status', 'unknown')}")
        log.info(f"Model: {model_name}, layers: {n_layers}")
    except Exception as e:
        log.warning(f"Health check failed: {e}, using defaults")
        model_name = "unknown"
        n_layers = None

    # Fallback: guess from model name
    if n_layers is None:
        if "0.8" in str(model_name):
            n_layers = 24
        elif "4B" in str(model_name) or "3.5-4" in str(model_name):
            n_layers = 36
        elif "9B" in str(model_name):
            n_layers = 40
        elif "14B" in str(model_name):
            n_layers = 40
        elif "32B" in str(model_name):
            n_layers = 64
        else:
            n_layers = 36  # safe default for Qwen3.5-4B
        log.info(f"  Guessed n_layers={n_layers}")

    # ── Select test layers ──
    # For Qwen3.5, GDN layers and full-attention layers are interleaved.
    # We want to test both types. The pattern for 3.5-4B is typically:
    #   layers 0..35, with every 4th layer being full attention.
    # We sample across the full depth to find where semantics live.
    if args.fast:
        # Quick: last, 75%, 50%
        test_layers = sorted(set([
            -1,
            int(n_layers * 0.75),
            int(n_layers * 0.50),
        ]))
    else:
        # Thorough: 8 layers spanning full depth
        test_layers = sorted(set([
            -1,
            n_layers - 1,                  # last explicit
            int(n_layers * 0.875),          # 87.5%
            int(n_layers * 0.75),           # 75%
            int(n_layers * 0.625),          # 62.5%
            int(n_layers * 0.50),           # 50%
            int(n_layers * 0.375),          # 37.5%
            int(n_layers * 0.25),           # 25%
        ]))

    # Clamp to valid range
    test_layers = [l for l in test_layers if 0 <= l < n_layers or l == -1]
    test_layers = sorted(set(test_layers))

    log.info(f"Testing {len(test_layers)} layers: {test_layers}")
    log.info(f"Testing {len(POOLING)} pooling methods: {list(POOLING.keys())}")
    log.info(f"Testing centering: [False, True]")
    n_combos = len(test_layers) * len(POOLING) * 2
    log.info(f"Total combinations: {n_combos}")

    # ── Load data ──
    pairs = load_stsb_pairs(args.n_pairs, args.seed)

    # ── Encode everything ──
    t0 = time.time()
    cache, hidden_dim = encode_corpus(client, pairs, test_layers)
    encode_time = time.time() - t0
    client.close()

    if not cache:
        log.error("No encodings — aborting")
        return

    # Detect available layer keys
    sample_hs = next(iter(cache.values()))
    available_keys = sorted(sample_hs.keys())
    log.info(f"Available layer keys in response: {available_keys}")

    # ── Evaluate all combos ──
    log.info("Evaluating all combinations...")
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

    # ── Sort by Spearman ──
    all_results.sort(key=lambda r: r["spearman_r"], reverse=True)

    # ── Print results table ──
    print("\n" + "=" * 105)
    print(f"M4 DIAGNOSTIC — {model_name} — {len(pairs)} pairs — "
          f"{len(all_results)} combos — encoded in {encode_time:.1f}s")
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

    # ── Highlight key findings ──
    print("\n" + "=" * 105)
    print("KEY FINDINGS")
    print("=" * 105)

    best = all_results[0]
    worst = all_results[-1]

    print(f"\n  BEST:  layer={best['layer']}, pooling={best['pooling']}, "
          f"centered={best['centered']}")
    print(f"         Spearman={best['spearman_r']:.4f}, "
          f"mean_cos={best['mean_cosine']:.4f}, std_cos={best['std_cosine']:.4f}")

    print(f"\n  WORST: layer={worst['layer']}, pooling={worst['pooling']}, "
          f"centered={worst['centered']}")
    print(f"         Spearman={worst['spearman_r']:.4f}, "
          f"mean_cos={worst['mean_cosine']:.4f}, std_cos={worst['std_cosine']:.4f}")

    # H1: Best pooling (holding other vars constant)
    pooling_best = {}
    for r in all_results:
        if r["pooling"] not in pooling_best or r["spearman_r"] > pooling_best[r["pooling"]]:
            pooling_best[r["pooling"]] = r["spearman_r"]
    print(f"\n  H1 (pooling): best Spearman per method:")
    for name, sp in sorted(pooling_best.items(), key=lambda x: -x[1]):
        print(f"       {name:<18} → {sp:.4f}")

    # H2: centering effect
    centered_scores = [r["spearman_r"] for r in all_results if r["centered"]]
    uncentered_scores = [r["spearman_r"] for r in all_results if not r["centered"]]
    if centered_scores and uncentered_scores:
        delta = np.mean(centered_scores) - np.mean(uncentered_scores)
        print(f"\n  H2 (centering): mean Spearman centered={np.mean(centered_scores):.4f}, "
              f"uncentered={np.mean(uncentered_scores):.4f}, Δ={delta:+.4f}")
        if delta > 0.05:
            print("       → CONFIRMED: centering helps significantly")
        elif delta > 0.01:
            print("       → Marginal: centering helps slightly")
        else:
            print("       → NOT CONFIRMED: centering doesn't help")

    # H3: layer effect
    layer_best = {}
    for r in all_results:
        if r["layer"] not in layer_best or r["spearman_r"] > layer_best[r["layer"]]:
            layer_best[r["layer"]] = r["spearman_r"]
    print(f"\n  H3 (layer): best Spearman per layer:")
    for name, sp in sorted(layer_best.items(), key=lambda x: -x[1]):
        print(f"       {name:<15} → {sp:.4f}")

    # Qwen3.5 GDN insight
    print(f"\n  Qwen3.5 note: GDN (linear attention) layers compress into fixed-size")
    print(f"  state matrices — may lose per-token detail. Full-attention layers")
    print(f"  retain full sequence info. Check if best layer is full-attention.")

    # ── Pass/fail ──
    n_passing = sum(1 for r in all_results if r["spearman_r"] > 0.6)
    print(f"\n  PASS THRESHOLD (Spearman > 0.6): {n_passing}/{len(all_results)} combos pass")

    if n_passing > 0:
        print(f"\n  ✅ M4 CAN PASS with optimal config:")
        print(f"     Update eval_phase1.py encode() call to use:")
        print(f"       layer  = {best['layer']!r}")
        print(f"       pool   = {best['pooling']!r}")
        print(f"       center = {best['centered']}")
    else:
        # Check if we're close
        if best["spearman_r"] > 0.5:
            print(f"\n  ⚠️  Best Spearman={best['spearman_r']:.4f} — close to 0.6.")
            print(f"     Consider: larger model (Qwen3.5-9B), or sentence-transformers")
            print(f"     embedding as reference baseline.")
        else:
            print(f"\n  ❌ Best Spearman={best['spearman_r']:.4f} — far from 0.6.")
            print(f"     The raw hidden space may not preserve STS-B semantics.")
            print(f"     This doesn't kill ERIS — STS-B measures surface similarity,")
            print(f"     not the deeper reasoning structure the bridge exploits.")
            print(f"     Consider reframing M4 around task-relevant similarity")
            print(f"     (e.g., do hidden states cluster by problem category?).")

    # ── Save full results ──
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_pairs": len(pairs),
        "encode_time_s": encode_time,
        "n_combos": len(all_results),
        "best": best,
        "worst": worst,
        "n_passing_06": n_passing,
        "all_results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
