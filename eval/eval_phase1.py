#!/usr/bin/env python3
"""
ERIS v5 — Phase 1 Evaluation: Does the latent channel add information?
=======================================================================

Uses HuggingFace datasets (no hardcoded data):
  - M4 (projection fidelity):  sentence-transformers/stsb
  - M1-M3 (A/B/C comparison):  TIGER-Lab/MMLU-Pro + openai/gsm8k
  - M5 (LatentMAS gain):       same questions, varying K
  - M6 (implicit features):    SAE analysis on encoded hidden states

Validated config (from M4 diagnostic v3 on Qwen3.5-4B):
  - layer_9 / last_token / centered → Spearman=0.6538
  - Single-layer-per-call encoding (avoids max_payload_bytes limit)

Usage:
  python eris_server.py --model Qwen/Qwen3.5-4B --port 8001

  # M4 only (no Claude API needed, ~7 min)
  python eval/eval_phase1.py --eris-url http://localhost:8001 --metric m4

  # M4 + M5 (no Claude API, ~15 min)
  python eval/eval_phase1.py --eris-url http://localhost:8001 --metric m4 m5

  # Full eval (needs ANTHROPIC_API_KEY, ~$5-8)
  ANTHROPIC_API_KEY=sk-... python eval/eval_phase1.py --eris-url http://localhost:8001 --metric all
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import stats
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_phase1")


# ═══════════════════════════════════════════════════════════════
# Config — baked from M4 diagnostic v3 results
# ═══════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "default": {
        "m4_layer": 9,
        "m4_pooling": "last_token",
        "m4_centered": True,
        "m4_threshold": 0.6,
    },
    "qwen3.5": {
        "m4_layer": 9,         # 25% depth — full-attention layer in GDN hybrid
        "m4_pooling": "last_token",
        "m4_centered": True,
        "m4_threshold": 0.6,
    },
    "qwen3": {
        "m4_layer": 35,        # last layer
        "m4_pooling": "last_token",
        "m4_centered": True,
        "m4_threshold": 0.6,
    },
}

def get_model_config(model_name: str) -> dict:
    model_lower = str(model_name).lower()
    if "3.5" in model_lower or "qwen3.5" in model_lower:
        return MODEL_CONFIGS["qwen3.5"]
    elif "qwen3" in model_lower:
        return MODEL_CONFIGS["qwen3"]
    return MODEL_CONFIGS["default"]


# ═══════════════════════════════════════════════════════════════
# ERIS client — single-layer-per-call, no session_id
# ═══════════════════════════════════════════════════════════════

import base64

class ERISClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self._model = None

    def health(self) -> dict:
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        data = r.json()
        self._model = data.get("model", "unknown")
        return data

    @property
    def model_name(self):
        if self._model is None:
            self.health()
        return self._model

    def encode(self, text: str, layer: int = -1) -> dict:
        r = self.client.post(
            f"{self.base_url}/v1/encode",
            json={"text": text, "return_layers": [layer], "compact": True},
        )
        r.raise_for_status()
        return r.json()

    def think(self, prompt: str, n_steps: int = 60) -> dict:
        r = self.client.post(f"{self.base_url}/sessions")
        r.raise_for_status()
        sid = r.json()["session_id"]
        try:
            r = self.client.post(
                f"{self.base_url}/think",
                json={"session_id": sid, "prompt": prompt, "n_steps": n_steps},
            )
            r.raise_for_status()
            return r.json()
        finally:
            try:
                self.client.delete(f"{self.base_url}/sessions/{sid}")
            except Exception:
                pass

    def latent_think(self, prompt: str, n_steps: int = 60) -> dict:
        r = self.client.post(
            f"{self.base_url}/v1/latent_think",
            json={"prompt": prompt, "n_steps": n_steps, "return_trajectory": False},
        )
        r.raise_for_status()
        return r.json()

    def bridge(self, text: str, mode: str = "ruminate",
               n_steps: int = 60, analyses: list = None) -> dict:
        r = self.client.post(
            f"{self.base_url}/v1/bridge",
            json={
                "claude_text": text, "mode": mode, "n_steps": n_steps,
                "analyses": analyses or [],
                "decode_after": True, "max_new_tokens": 512,
            },
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def decode_b64(b64_str: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float32)

def extract_vector(enc: dict, hidden_dim: int) -> np.ndarray:
    for key, val in enc.get("hidden_states", {}).items():
        if isinstance(val, str):
            return decode_b64(val).reshape(-1, hidden_dim)
        elif isinstance(val, list):
            mat = np.array(val, dtype=np.float32)
            return mat if mat.ndim == 2 else mat.reshape(1, -1)
    return None

def pool(mat: np.ndarray, method: str) -> np.ndarray:
    if method == "last_token":
        return mat[-1]
    elif method == "2nd_to_last":
        return mat[-2] if mat.shape[0] >= 2 else mat[-1]
    elif method == "mean_all":
        return mat.mean(axis=0)
    elif method == "mean_no_edges":
        return mat[1:-1].mean(axis=0) if mat.shape[0] > 2 else mat.mean(axis=0)
    elif method == "max_pool":
        return mat.max(axis=0)
    return mat.mean(axis=0)

def cosine(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_stsb_pairs(n_pairs: int = 200, seed: int = 42):
    from datasets import load_dataset
    np.random.seed(seed)
    log.info(f"Loading STS-B test ({n_pairs} pairs)...")
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
    log.info(f"  {len(pairs[:n_pairs])} pairs")
    return pairs[:n_pairs]


def load_technical_questions(n_questions: int = 50, seed: int = 42):
    from datasets import load_dataset
    np.random.seed(seed)
    questions = []

    log.info("Loading MMLU-Pro (STEM)...")
    try:
        mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        cs_qs = [r for r in mmlu if r.get("category") in
                 ("computer_science", "engineering", "math", "physics")]
        np.random.shuffle(cs_qs)
        for r in cs_qs[:n_questions // 2]:
            questions.append({
                "id": f"mmlu_{len(questions)}", "text": r["question"],
                "source": "MMLU-Pro", "category": r.get("category", "unknown"),
                "answer": r.get("answer"),
            })
    except Exception as e:
        log.warning(f"  MMLU-Pro failed: {e}")

    remaining = n_questions - len(questions)
    if remaining > 0:
        log.info(f"Loading GSM8K ({remaining})...")
        try:
            gsm = load_dataset("openai/gsm8k", "main", split="test")
            indices = list(range(len(gsm)))
            np.random.shuffle(indices)
            for idx in indices[:remaining]:
                r = gsm[idx]
                questions.append({
                    "id": f"gsm_{len(questions)}", "text": r["question"],
                    "source": "GSM8K", "category": "math_reasoning",
                    "answer": r.get("answer"),
                })
        except Exception as e:
            log.warning(f"  GSM8K failed: {e}")

    log.info(f"  {len(questions)} questions")
    return questions


# ═══════════════════════════════════════════════════════════════
# M4 — Projection Fidelity
# ═══════════════════════════════════════════════════════════════

@dataclass
class M4Result:
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    n_pairs: int
    mean_cosine: float
    std_cosine: float
    pass_threshold: bool
    layer: int
    pooling: str
    centered: bool

    def summary(self) -> str:
        s = "✅ PASS" if self.pass_threshold else "❌ FAIL"
        return (
            f"M4 Projection Fidelity ({self.n_pairs} pairs)\n"
            f"  Config: layer={self.layer}, pooling={self.pooling}, "
            f"centered={self.centered}\n"
            f"  Spearman ρ = {self.spearman_r:.4f} (p={self.spearman_p:.2e}) {s}\n"
            f"  Pearson  r = {self.pearson_r:.4f} (p={self.pearson_p:.2e})\n"
            f"  Cosine: mean={self.mean_cosine:.4f}, std={self.std_cosine:.4f}"
        )


def eval_m4(client: ERISClient, cfg: dict, n_pairs: int = 200) -> M4Result:
    pairs = load_stsb_pairs(n_pairs)
    layer, pooling_name, centered = cfg["m4_layer"], cfg["m4_pooling"], cfg["m4_centered"]
    unique = list({p["sentence1"] for p in pairs} | {p["sentence2"] for p in pairs})

    log.info(f"Encoding {len(unique)} sentences at layer {layer}...")
    vectors = {}
    hidden_dim = None
    errors = 0

    for sent in tqdm(unique, desc="M4 encoding"):
        try:
            enc = client.encode(sent, layer=layer)
            if hidden_dim is None:
                hidden_dim = enc.get("hidden_dim")
            mat = extract_vector(enc, hidden_dim)
            if mat is not None:
                vectors[sent] = pool(mat, pooling_name)
        except Exception as e:
            errors += 1
            if errors <= 3:
                log.warning(f"  Error: {e}")
            if errors > 30:
                break

    log.info(f"  {len(vectors)}/{len(unique)} encoded")
    if len(vectors) < 30:
        return M4Result(0, 1, 0, 1, 0, 0, 0, False, layer, pooling_name, centered)

    if centered:
        all_vecs = np.stack(list(vectors.values()))
        mean_vec = all_vecs.mean(axis=0)
        vectors = {s: v - mean_vec for s, v in vectors.items()}

    h, z = [], []
    for p in pairs:
        s1, s2 = p["sentence1"], p["sentence2"]
        if s1 in vectors and s2 in vectors:
            h.append(p["score"])
            z.append(cosine(vectors[s1], vectors[s2]))

    h, z = np.array(h), np.array(z)
    sp = stats.spearmanr(h, z)
    pr = stats.pearsonr(h, z)

    return M4Result(
        spearman_r=float(sp.statistic), spearman_p=float(sp.pvalue),
        pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue),
        n_pairs=len(h), mean_cosine=float(z.mean()), std_cosine=float(z.std()),
        pass_threshold=float(sp.statistic) > cfg["m4_threshold"],
        layer=layer, pooling=pooling_name, centered=centered,
    )


# ═══════════════════════════════════════════════════════════════
# M5 — LatentMAS Gain
# ═══════════════════════════════════════════════════════════════

@dataclass
class M5Result:
    k_values: list
    displacement_means: list
    displacement_stds: list
    n_questions: int
    gain_detected: bool
    endpoint_used: str

    def summary(self) -> str:
        s = "✅ Gain detected" if self.gain_detected else "❌ No gain"
        lines = [f"M5 LatentMAS Gain ({self.n_questions} q, "
                 f"via {self.endpoint_used}) {s}"]
        for k, m, sd in zip(self.k_values, self.displacement_means,
                            self.displacement_stds):
            bar = "█" * max(1, int(m * 20)) if m > 0 else ""
            lines.append(f"  K={k:3d}: {m:.4f} ± {sd:.4f}  {bar}")
        return "\n".join(lines)


def eval_m5(client: ERISClient, n_questions: int = 30,
            k_values: list = None) -> M5Result:
    if k_values is None:
        k_values = [0, 5, 15, 30, 60]

    questions = load_technical_questions(n_questions)

    # Detect working endpoint
    endpoint = "v1/latent_think"
    try:
        client.latent_think("test", n_steps=1)
    except Exception:
        endpoint = "think"
        try:
            client.think("test", n_steps=1)
        except Exception:
            log.error("No think endpoint works")
            return M5Result(k_values, [0]*len(k_values), [0]*len(k_values),
                            0, False, "none")
    log.info(f"Using: {endpoint}")

    results_by_k = {k: [] for k in k_values}
    errors = 0

    for q in tqdm(questions, desc="M5"):
        for k in k_values:
            try:
                if endpoint == "v1/latent_think":
                    res = client.latent_think(q["text"], n_steps=k)
                else:
                    res = client.think(q["text"], n_steps=k)

                disp = (res.get("total_displacement")
                        or res.get("displacement")
                        or res.get("hidden_norm", 0))
                results_by_k[k].append(float(disp))
            except Exception as e:
                errors += 1
                if errors <= 5:
                    log.warning(f"  K={k}: {e}")
                if errors > 80:
                    break

    means = [float(np.mean(results_by_k[k])) if results_by_k[k] else 0 for k in k_values]
    stds = [float(np.std(results_by_k[k])) if results_by_k[k] else 0 for k in k_values]
    gain = len(means) >= 2 and means[-1] > means[0] + 0.01

    return M5Result(k_values, means, stds, len(questions), gain, endpoint)


# ═══════════════════════════════════════════════════════════════
# M1-M3 — A/B/C
# ═══════════════════════════════════════════════════════════════

@dataclass
class ABCResult:
    n_questions: int
    scores_a: list
    scores_b: list
    scores_c: list
    mean_a: float
    mean_b: float
    mean_c: float
    c_vs_a_win: float
    c_vs_b_win: float
    wilcoxon_ca_p: float
    wilcoxon_cb_p: float

    def summary(self) -> str:
        sig_ca = "(*)" if self.wilcoxon_ca_p < 0.05 else "(ns)"
        sig_cb = "(*)" if self.wilcoxon_cb_p < 0.05 else "(ns)"
        return (
            f"M1-M3 A/B/C ({self.n_questions} questions)\n"
            f"  (A) Claude direct:     {self.mean_a:.2f}\n"
            f"  (B) Claude+paraphrase: {self.mean_b:.2f}\n"
            f"  (C) Claude+ruminate:   {self.mean_c:.2f}\n"
            f"  C>A: {self.c_vs_a_win:.0%} p={self.wilcoxon_ca_p:.4f} {sig_ca}\n"
            f"  C>B: {self.c_vs_b_win:.0%} p={self.wilcoxon_cb_p:.4f} {sig_cb}"
        )


def eval_abc(client: ERISClient, n_questions: int = 30,
             n_steps: int = 60) -> ABCResult:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set")
        return ABCResult(0, [], [], [], 0, 0, 0, 0, 0, 1, 1)

    claude = anthropic.Anthropic(api_key=api_key)
    questions = load_technical_questions(n_questions)

    def ask(messages, max_tokens=512):
        resp = claude.messages.create(
            model="claude-sonnet-4-6", max_tokens=max_tokens,
            messages=messages, timeout=60.0)
        return resp.content[0].text

    def judge(question, answer):
        try:
            s = ask([{"role": "user",
                      "content": f"Question: {question}\n\nAnswer: {answer}\n\n"
                      "Rate analytical depth 1-5. Reply with ONLY one integer."}],
                    max_tokens=10)
            return max(1, min(5, int(s.strip()[0])))
        except Exception:
            return 3

    sa, sb, sc = [], [], []
    for q in tqdm(questions, desc="ABC"):
        try:
            text_a = ask([{"role": "user", "content": q["text"]}])

            br_b = client.bridge(text_a, mode="passive", n_steps=0)
            enr_b = br_b.get("enriched_text", br_b.get("decoded_text", text_a))
            text_b = ask([
                {"role": "user", "content": q["text"]},
                {"role": "assistant", "content": text_a},
                {"role": "user",
                 "content": f"Complementary perspective:\n{enr_b}\n\nImproved answer:"},
            ])

            br_c = client.bridge(text_a, mode="ruminate", n_steps=n_steps)
            enr_c = br_c.get("enriched_text", br_c.get("decoded_text", text_a))
            text_c = ask([
                {"role": "user", "content": q["text"]},
                {"role": "assistant", "content": text_a},
                {"role": "user",
                 "content": f"Deep analysis:\n{enr_c}\n\nImproved answer:"},
            ])

            sa.append(judge(q["text"], text_a))
            sb.append(judge(q["text"], text_b))
            sc.append(judge(q["text"], text_c))
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"  ABC error: {e}")

    n = min(len(sa), len(sb), len(sc))
    if n < 5:
        return ABCResult(n, sa, sb, sc, 0, 0, 0, 0, 0, 1, 1)
    sa, sb, sc = sa[:n], sb[:n], sc[:n]

    try:
        w_ca = stats.wilcoxon([c - a for a, c in zip(sa, sc)])
        w_cb = stats.wilcoxon([c - b for b, c in zip(sb, sc)])
    except Exception:
        w_ca = type("W", (), {"pvalue": 1.0})()
        w_cb = type("W", (), {"pvalue": 1.0})()

    return ABCResult(
        n, sa, sb, sc,
        float(np.mean(sa)), float(np.mean(sb)), float(np.mean(sc)),
        float(np.mean([c > a for a, c in zip(sa, sc)])),
        float(np.mean([c > b for b, c in zip(sb, sc)])),
        float(w_ca.pvalue), float(w_cb.pvalue),
    )


# ═══════════════════════════════════════════════════════════════
# M6 — Implicit Features
# ═══════════════════════════════════════════════════════════════

@dataclass
class M6Result:
    n_questions: int
    mean_implicit: float
    mean_total: float
    implicit_ratio: float
    sae_available: bool
    examples: list

    def summary(self) -> str:
        if not self.sae_available:
            return "M6 Implicit Features — SAE not configured (skipped)"
        return (
            f"M6 Implicit Features ({self.n_questions} questions)\n"
            f"  Total SAE features:  {self.mean_total:.1f}\n"
            f"  Implicit features:   {self.mean_implicit:.1f}\n"
            f"  Ratio:               {self.implicit_ratio:.1%}"
        )


def eval_m6(client: ERISClient, n_questions: int = 20) -> M6Result:
    questions = load_technical_questions(n_questions)
    totals, implicits, examples = [], [], []
    sae_ok = True

    for q in tqdm(questions, desc="M6"):
        try:
            res = client.bridge(q["text"], mode="analyze_only", n_steps=0,
                                analyses=["sae_features"])
            an = res.get("analysis", {})
            sae = an.get("sae_features")
            if sae is None:
                sae_ok = False
                break
            imp = an.get("implicit_features", [])
            totals.append(len(sae.get("top_20", sae.get("top_10", []))))
            implicits.append(len(imp))
            if imp and len(examples) < 5:
                examples.append({"q": q["text"][:80], "implicit": imp[:3]})
        except Exception as e:
            log.warning(f"  M6: {e}")

    if not totals:
        return M6Result(0, 0, 0, 0, sae_ok, [])
    mt, mi = float(np.mean(totals)), float(np.mean(implicits))
    return M6Result(len(totals), mi, mt, mi / max(mt, 1), sae_ok, examples)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ERIS v5 Phase 1")
    parser.add_argument("--eris-url", default="http://localhost:8001")
    parser.add_argument("--metric", nargs="+", default=["all"],
                        choices=["all", "m4", "m5", "m6", "abc"])
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--n-questions", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    metrics = set(args.metric)
    if "all" in metrics:
        metrics = {"m4", "m5", "m6", "abc"}

    Path(args.output_dir).mkdir(exist_ok=True)
    client = ERISClient(args.eris_url)
    health = client.health()
    model = health.get("model", "unknown")
    cfg = get_model_config(model)
    log.info(f"Model: {model}, Config: {cfg}")

    results = {"tag": args.tag, "model": model, "config": cfg,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        if "m4" in metrics:
            log.info("=" * 60 + "\nM4 — Projection Fidelity\n" + "=" * 60)
            m4 = eval_m4(client, cfg, n_pairs=args.n_pairs)
            print(f"\n{m4.summary()}\n")
            results["m4"] = asdict(m4)

        if "m5" in metrics:
            log.info("=" * 60 + "\nM5 — LatentMAS Gain\n" + "=" * 60)
            m5 = eval_m5(client, n_questions=args.n_questions)
            print(f"\n{m5.summary()}\n")
            results["m5"] = asdict(m5)

        if "m6" in metrics:
            log.info("=" * 60 + "\nM6 — Implicit Features\n" + "=" * 60)
            m6 = eval_m6(client, n_questions=min(20, args.n_questions))
            print(f"\n{m6.summary()}\n")
            results["m6"] = asdict(m6)

        if "abc" in metrics:
            log.info("=" * 60 + "\nM1-M3 — A/B/C Comparison\n" + "=" * 60)
            abc = eval_abc(client, n_questions=args.n_questions,
                           n_steps=args.n_steps)
            if abc.n_questions > 0:
                print(f"\n{abc.summary()}\n")
            results["abc"] = asdict(abc)
    finally:
        client.close()

    out = Path(args.output_dir) / f"phase1_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved to {out}")

    # Summary
    print("\n" + "=" * 60)
    print(f"PHASE 1 — {model}")
    print("=" * 60)
    if "m4" in results:
        r = results["m4"]
        print(f"  M4 Projection:  ρ={r['spearman_r']:.4f} "
              f"{'✅' if r['pass_threshold'] else '❌'}")
    if "m5" in results:
        r = results["m5"]
        print(f"  M5 LatentMAS:   {'✅' if r['gain_detected'] else '❌'}  "
              f"K=0→{r['displacement_means'][0]:.4f}  "
              f"K=60→{r['displacement_means'][-1]:.4f}")
    if "m6" in results:
        r = results["m6"]
        if r["sae_available"]:
            print(f"  M6 Implicit:    {r['implicit_ratio']:.1%}")
        else:
            print(f"  M6 Implicit:    SAE n/a")
    if "abc" in results and results["abc"]["n_questions"] > 0:
        r = results["abc"]
        print(f"  ABC:  A={r['mean_a']:.2f} B={r['mean_b']:.2f} "
              f"C={r['mean_c']:.2f}  C>A={r['c_vs_a_win']:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
