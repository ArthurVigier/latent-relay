#!/usr/bin/env python3
"""
ERIS v5 — Phase 1 Complete Evaluation
=======================================

All Phase 1 tests in one file. Ten metrics, four categories:

  CHANNEL VALIDATION (no Claude API needed):
    m4        Projection fidelity — STS-B cosine vs human similarity
    m5        LatentMAS gain — displacement grows with K rumination steps
    m6        Implicit features — SAE features absent from surface text

  COGNITIVE COMPARISON (needs ANTHROPIC_API_KEY):
    abc       A/B/C — Claude direct vs Claude+paraphrase vs Claude+ruminate

  STEERING & LOOPS:
    steering      Contrastive Steering — extract concept vectors, test alignment
    loop          Zombie↔Zombie autonomous exploration loop
    dialogue      Claude↔Zombie dialogue — Claude IN the loop at every turn
    steerdialogue Claude↔Zombie dialogue WITH contrastive steering at each turn:
                  zombie ruminates toward a concept direction (rigorous, creative,
                  cautious, concrete) — tracks whether Claude internalizes the direction

  FRONTIER (needs ANTHROPIC_API_KEY, uses web search):
    frontier  Claude vs Claude+Zombie on tasks requiring web + reasoning:
              - Cross-source synthesis (contradictions between sources)
              - Temporal reasoning (events across time, requires sequencing)
              - Adversarial fact-check (plausible-sounding false claims)
              - Open research questions (no single right answer)
    webdialogue  Claude↔Zombie dialogue WITH web search injected at each turn:
              Claude responds → searches web for verification/expansion →
              zombie ruminates (response + web context) → Claude integrates

HuggingFace datasets (no hardcoded data):
  - sentence-transformers/stsb       (M4)
  - TIGER-Lab/MMLU-Pro               (M5, ABC, steering)
  - openai/gsm8k                     (M5, ABC, loop, dialogue)

Web search: DuckDuckGo (no API key needed) via duckduckgo-search package.

Usage:
  python eris_server.py --model Qwen/Qwen3.5-4B --port 8001

  # Channel validation only (~20 min, no API key)
  python eval_phase1.py --metric m4 m5

  # Everything except Claude-dependent tests
  python eval_phase1.py --metric m4 m5 m6 steering loop

  # Full Phase 1 including frontier (~$15-20 API cost)
  ANTHROPIC_API_KEY=sk-... python eval_phase1.py --metric all

  # Just frontier tests
  ANTHROPIC_API_KEY=sk-... python eval_phase1.py --metric frontier webdialogue

Requirements:
  pip install datasets anthropic httpx scipy numpy tqdm duckduckgo-search
"""

import os
import json
import time
import base64
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eris_phase1")

ALL_METRICS = {
    "m4", "m5", "m6", "abc",
    "steering", "loop", "dialogue", "steerdialogue",
    "frontier", "webdialogue",
}


# ═══════════════════════════════════════════════════════════════════
#  CONFIG — baked from M4 diagnostic v3
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "default":  {"m4_layer": 9,  "m4_pooling": "last_token", "m4_centered": True, "m4_threshold": 0.6},
    "qwen3.5":  {"m4_layer": 9,  "m4_pooling": "last_token", "m4_centered": True, "m4_threshold": 0.6},
    "qwen3":    {"m4_layer": 35, "m4_pooling": "last_token", "m4_centered": True, "m4_threshold": 0.6},
}

def get_model_config(model_name: str) -> dict:
    m = str(model_name).lower()
    if "3.5" in m:
        return MODEL_CONFIGS["qwen3.5"]
    if "qwen3" in m:
        return MODEL_CONFIGS["qwen3"]
    return MODEL_CONFIGS["default"]


# ═══════════════════════════════════════════════════════════════════
#  ERIS CLIENT — single-layer-per-call, no session_id
# ═══════════════════════════════════════════════════════════════════

class ERISClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self._model = None

    def health(self) -> dict:
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        d = r.json()
        self._model = d.get("model", "unknown")
        return d

    @property
    def model_name(self):
        if self._model is None:
            self.health()
        return self._model

    def encode(self, text: str, layer: int = -1) -> dict:
        r = self.client.post(f"{self.base_url}/v1/encode",
                             json={"text": text, "return_layers": [layer], "compact": True})
        r.raise_for_status()
        return r.json()

    def think(self, prompt: str, n_steps: int = 60) -> dict:
        r = self.client.post(f"{self.base_url}/sessions")
        r.raise_for_status()
        sid = r.json()["session_id"]
        try:
            r = self.client.post(f"{self.base_url}/think",
                                 json={"session_id": sid, "prompt": prompt, "n_steps": n_steps})
            r.raise_for_status()
            return r.json()
        finally:
            try:
                self.client.delete(f"{self.base_url}/sessions/{sid}")
            except Exception:
                pass

    def latent_think(self, prompt: str, n_steps: int = 60) -> dict:
        r = self.client.post(f"{self.base_url}/v1/latent_think",
                             json={"prompt": prompt, "n_steps": n_steps, "return_trajectory": False})
        r.raise_for_status()
        return r.json()

    def bridge(self, text: str, mode: str = "ruminate", n_steps: int = 60,
               analyses: list = None) -> dict:
        r = self.client.post(f"{self.base_url}/v1/bridge", json={
            "claude_text": text, "mode": mode, "n_steps": n_steps,
            "analyses": analyses or [], "decode_after": True, "max_new_tokens": 512,
        })
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def decode_b64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)

def extract_vector(enc: dict, hidden_dim: int) -> np.ndarray:
    """Extract [seq_len, hidden_dim] matrix from encode response."""
    for val in enc.get("hidden_states", {}).values():
        if isinstance(val, str):
            return decode_b64(val).reshape(-1, hidden_dim)
        elif isinstance(val, list):
            m = np.array(val, dtype=np.float32)
            return m if m.ndim == 2 else m.reshape(1, -1)
    return None

def extract_mean_vector(enc: dict, hidden_dim: int) -> np.ndarray:
    """Extract mean-pooled [hidden_dim] vector."""
    mat = extract_vector(enc, hidden_dim)
    return mat.mean(axis=0) if mat is not None else None

def pool(mat: np.ndarray, method: str) -> np.ndarray:
    if method == "last_token":     return mat[-1]
    if method == "2nd_to_last":    return mat[-2] if len(mat) >= 2 else mat[-1]
    if method == "mean_no_edges":  return mat[1:-1].mean(0) if len(mat) > 2 else mat.mean(0)
    if method == "max_pool":       return mat.max(0)
    return mat.mean(0)  # mean_all default

def cosine(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 1e-10 and n2 > 1e-10 else 0.0


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING — all HuggingFace
# ═══════════════════════════════════════════════════════════════════

def load_stsb_pairs(n_pairs=200, seed=42):
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
    if len(pairs) < n_pairs:
        all_items = list(ds)
        np.random.shuffle(all_items)
        pairs.extend(all_items[:n_pairs - len(pairs)])
    return pairs[:n_pairs]


def load_technical_questions(n=50, seed=42):
    from datasets import load_dataset
    np.random.seed(seed)
    qs = []
    try:
        mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        pool_items = [r for r in mmlu if r.get("category") in
                      ("computer_science", "engineering", "math", "physics",
                       "biology", "chemistry", "philosophy", "economics")]
        np.random.shuffle(pool_items)
        for r in pool_items[:n // 2]:
            qs.append({"id": f"mmlu_{len(qs)}", "text": r["question"],
                       "source": "MMLU-Pro", "category": r.get("category", "?"),
                       "answer": r.get("answer")})
    except Exception as e:
        log.warning(f"MMLU-Pro: {e}")
    remaining = n - len(qs)
    if remaining > 0:
        try:
            gsm = load_dataset("openai/gsm8k", "main", split="test")
            idx = list(range(len(gsm)))
            np.random.shuffle(idx)
            for i in idx[:remaining]:
                r = gsm[i]
                qs.append({"id": f"gsm_{len(qs)}", "text": r["question"],
                           "source": "GSM8K", "category": "math", "answer": r.get("answer")})
        except Exception as e:
            log.warning(f"GSM8K: {e}")
    log.info(f"  {len(qs)} questions loaded")
    return qs


def load_seed_texts(n=5, seed=42):
    """Load diverse seed texts for loop/dialogue suites."""
    from datasets import load_dataset
    np.random.seed(seed)
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        idx = list(range(len(gsm)))
        np.random.shuffle(idx)
        return [gsm[i]["question"] for i in idx[:n]]
    except Exception:
        return ["What are the fundamental tradeoffs in distributed systems?",
                "How does natural selection lead to complex adaptations?",
                "What is the relationship between entropy and information?"][:n]


# ═══════════════════════════════════════════════════════════════════
#  M4 — PROJECTION FIDELITY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class M4Result:
    spearman_r: float; spearman_p: float; pearson_r: float; pearson_p: float
    n_pairs: int; mean_cosine: float; std_cosine: float; pass_threshold: bool
    layer: int; pooling: str; centered: bool
    def summary(self):
        s = "✅ PASS" if self.pass_threshold else "❌ FAIL"
        return (f"M4 Projection Fidelity ({self.n_pairs} pairs)\n"
                f"  layer={self.layer}, pooling={self.pooling}, centered={self.centered}\n"
                f"  Spearman ρ = {self.spearman_r:.4f} (p={self.spearman_p:.2e}) {s}\n"
                f"  Pearson  r = {self.pearson_r:.4f}\n"
                f"  Cosine: mean={self.mean_cosine:.4f} std={self.std_cosine:.4f}")

def eval_m4(client, cfg, n_pairs=200):
    pairs = load_stsb_pairs(n_pairs)
    layer, pm, cent = cfg["m4_layer"], cfg["m4_pooling"], cfg["m4_centered"]
    unique = list({p["sentence1"] for p in pairs} | {p["sentence2"] for p in pairs})
    vecs, hd, errs = {}, None, 0
    for s in tqdm(unique, desc="M4"):
        try:
            enc = client.encode(s, layer=layer)
            if hd is None: hd = enc.get("hidden_dim")
            mat = extract_vector(enc, hd)
            if mat is not None: vecs[s] = pool(mat, pm)
        except Exception:
            errs += 1
            if errs > 30: break
    if len(vecs) < 30:
        return M4Result(0, 1, 0, 1, 0, 0, 0, False, layer, pm, cent)
    if cent:
        mv = np.stack(list(vecs.values())).mean(0)
        vecs = {s: v - mv for s, v in vecs.items()}
    h, z = [], []
    for p in pairs:
        if p["sentence1"] in vecs and p["sentence2"] in vecs:
            h.append(p["score"])
            z.append(cosine(vecs[p["sentence1"]], vecs[p["sentence2"]]))
    h, z = np.array(h), np.array(z)
    sp, pr = stats.spearmanr(h, z), stats.pearsonr(h, z)
    return M4Result(float(sp.statistic), float(sp.pvalue), float(pr.statistic),
                    float(pr.pvalue), len(h), float(z.mean()), float(z.std()),
                    float(sp.statistic) > cfg["m4_threshold"], layer, pm, cent)


# ═══════════════════════════════════════════════════════════════════
#  M5 — LATENTMAS GAIN
# ═══════════════════════════════════════════════════════════════════

@dataclass
class M5Result:
    k_values: list; displacement_means: list; displacement_stds: list
    n_questions: int; gain_detected: bool; endpoint_used: str
    def summary(self):
        s = "✅ Gain" if self.gain_detected else "❌ No gain"
        lines = [f"M5 LatentMAS Gain ({self.n_questions} q, {self.endpoint_used}) {s}"]
        for k, m, sd in zip(self.k_values, self.displacement_means, self.displacement_stds):
            lines.append(f"  K={k:3d}: {m:.4f} ± {sd:.4f}  {'█'*max(1,int(m*20)) if m>0 else ''}")
        return "\n".join(lines)

def eval_m5(client, n_questions=30, k_values=None):
    if k_values is None: k_values = [0, 5, 15, 30, 60]
    qs = load_technical_questions(n_questions)
    ep = "v1/latent_think"
    try: client.latent_think("test", 1)
    except Exception:
        ep = "think"
        try: client.think("test", 1)
        except Exception:
            return M5Result(k_values, [0]*len(k_values), [0]*len(k_values), 0, False, "none")
    by_k = {k: [] for k in k_values}
    for q in tqdm(qs, desc="M5"):
        for k in k_values:
            try:
                res = (client.latent_think(q["text"], k) if ep == "v1/latent_think"
                       else client.think(q["text"], k))
                by_k[k].append(float(res.get("total_displacement") or
                                     res.get("displacement") or
                                     res.get("hidden_norm", 0)))
            except Exception:
                pass
    means = [float(np.mean(by_k[k])) if by_k[k] else 0 for k in k_values]
    stds  = [float(np.std(by_k[k]))  if by_k[k] else 0 for k in k_values]
    return M5Result(k_values, means, stds, len(qs),
                    len(means) >= 2 and means[-1] > means[0] + 0.01, ep)


# ═══════════════════════════════════════════════════════════════════
#  M6 — IMPLICIT FEATURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class M6Result:
    n_questions: int; mean_implicit: float; mean_total: float
    implicit_ratio: float; sae_available: bool; examples: list
    def summary(self):
        if not self.sae_available: return "M6 Implicit Features — SAE not configured"
        return (f"M6 Implicit Features ({self.n_questions} q)\n"
                f"  Total: {self.mean_total:.1f}  Implicit: {self.mean_implicit:.1f}  "
                f"Ratio: {self.implicit_ratio:.1%}")

def eval_m6(client, n_questions=20):
    qs = load_technical_questions(n_questions)
    tot, imp, ex, sae_ok = [], [], [], True
    for q in tqdm(qs, desc="M6"):
        try:
            r = client.bridge(q["text"], mode="analyze_only", n_steps=0, analyses=["sae_features"])
            an = r.get("analysis", {})
            if an.get("sae_features") is None: sae_ok = False; break
            features = an["sae_features"]
            implicit = an.get("implicit_features", [])
            tot.append(len(features.get("top_20", features.get("top_10", []))))
            imp.append(len(implicit))
            if implicit and len(ex) < 5: ex.append({"q": q["text"][:80], "imp": implicit[:3]})
        except Exception: pass
    if not tot: return M6Result(0, 0, 0, 0, sae_ok, [])
    mt, mi = float(np.mean(tot)), float(np.mean(imp))
    return M6Result(len(tot), mi, mt, mi / max(mt, 1), sae_ok, ex)


# ═══════════════════════════════════════════════════════════════════
#  ABC — CLAUDE DIRECT vs PARAPHRASE vs RUMINATE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ABCResult:
    n_questions: int; scores_a: list; scores_b: list; scores_c: list
    mean_a: float; mean_b: float; mean_c: float
    c_vs_a_win: float; c_vs_b_win: float
    wilcoxon_ca_p: float; wilcoxon_cb_p: float
    def summary(self):
        return (f"ABC ({self.n_questions} q)\n"
                f"  A={self.mean_a:.2f} B={self.mean_b:.2f} C={self.mean_c:.2f}\n"
                f"  C>A: {self.c_vs_a_win:.0%} p={self.wilcoxon_ca_p:.4f}\n"
                f"  C>B: {self.c_vs_b_win:.0%} p={self.wilcoxon_cb_p:.4f}")

def eval_abc(client, n_questions=30, n_steps=60):
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.error("ANTHROPIC_API_KEY not set"); return ABCResult(0, [], [], [], 0, 0, 0, 0, 0, 1, 1)
    cl = anthropic.Anthropic(api_key=key)
    qs = load_technical_questions(n_questions)
    def ask(msgs, mt=512):
        return cl.messages.create(model="claude-sonnet-4-6", max_tokens=mt,
                                  messages=msgs, timeout=60).content[0].text
    def judge(q, a):
        try:
            return max(1, min(5, int(ask([{"role": "user",
                "content": f"Question: {q}\n\nAnswer: {a}\n\nRate depth 1-5. ONLY one integer."}],
                10).strip()[0])))
        except Exception:
            return 3
    sa, sb, sc = [], [], []
    for q in tqdm(qs, desc="ABC"):
        try:
            ta = ask([{"role": "user", "content": q["text"]}])
            br_b = client.bridge(ta, mode="passive", n_steps=0)
            tb = ask([{"role": "user", "content": q["text"]},
                      {"role": "assistant", "content": ta},
                      {"role": "user",
                       "content": f"Complementary:\n{br_b.get('enriched_text', br_b.get('decoded_text', ta))}\n\nImproved:"}])
            br_c = client.bridge(ta, mode="ruminate", n_steps=n_steps)
            tc = ask([{"role": "user", "content": q["text"]},
                      {"role": "assistant", "content": ta},
                      {"role": "user",
                       "content": f"Deep analysis:\n{br_c.get('enriched_text', br_c.get('decoded_text', ta))}\n\nImproved:"}])
            sa.append(judge(q["text"], ta))
            sb.append(judge(q["text"], tb))
            sc.append(judge(q["text"], tc))
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"  ABC: {e}")
    n = min(len(sa), len(sb), len(sc))
    if n < 5: return ABCResult(n, sa, sb, sc, 0, 0, 0, 0, 0, 1, 1)
    sa, sb, sc = sa[:n], sb[:n], sc[:n]
    try:
        wca = stats.wilcoxon([c - a for a, c in zip(sa, sc)])
        wcb = stats.wilcoxon([c - b for b, c in zip(sb, sc)])
    except Exception:
        wca = wcb = type("W", (), {"pvalue": 1.0})()
    return ABCResult(n, sa, sb, sc,
                     float(np.mean(sa)), float(np.mean(sb)), float(np.mean(sc)),
                     float(np.mean([c > a for a, c in zip(sa, sc)])),
                     float(np.mean([c > b for b, c in zip(sb, sc)])),
                     float(wca.pvalue), float(wcb.pvalue))


# ═══════════════════════════════════════════════════════════════════
#  CONTRASTIVE STEERING
# ═══════════════════════════════════════════════════════════════════

CONTRAST_TEMPLATES = {
    "rigorous_vs_superficial": {
        "positive": "Give a rigorous, precise, step-by-step analysis: ",
        "negative": "Give a quick, surface-level answer: ",
        "description": "Depth of analysis",
    },
    "creative_vs_conventional": {
        "positive": "Give an original, unexpected, creative answer: ",
        "negative": "Give a standard, conventional textbook answer: ",
        "description": "Creativity and novelty",
    },
    "cautious_vs_confident": {
        "positive": "Be very careful, note uncertainties and caveats: ",
        "negative": "Be maximally confident, give a direct definitive answer: ",
        "description": "Epistemic caution",
    },
    "concrete_vs_abstract": {
        "positive": "Give a concrete answer with specific examples and numbers: ",
        "negative": "Give an abstract, theoretical, high-level answer: ",
        "description": "Concreteness",
    },
}

@dataclass
class SteeringResult:
    concept: str; description: str; n_pairs: int; vector_norm: float
    alignment_mean: float; alignment_std: float; positive_rate: float
    base_texts: list; steered_texts: list
    def summary(self):
        s = "✅" if self.positive_rate > 0.6 else "❌"
        return (f"  {self.concept} ({self.description})\n"
                f"    {self.n_pairs} pairs, align={self.alignment_mean:+.4f}±{self.alignment_std:.4f}, "
                f"pos={self.positive_rate:.0%} {s}")

def eval_steering(client, layer=9, hidden_dim=2560, n_extract=30, n_test=10):
    log.info("Contrastive Steering — extracting 4 concept vectors")
    all_qs = load_technical_questions(n_extract + n_test)
    if len(all_qs) < 20:
        log.error("Not enough questions"); return []
    q_texts = [q["text"] for q in all_qs]
    ext_qs, test_qs = q_texts[:n_extract], q_texts[n_extract:n_extract + n_test]
    results = []
    for name, tmpl in CONTRAST_TEMPLATES.items():
        diffs = []
        for q in tqdm(ext_qs, desc=f"Extract {name}", leave=False):
            try:
                vp = extract_mean_vector(client.encode(tmpl["positive"] + q, layer=layer), hidden_dim)
                vn = extract_mean_vector(client.encode(tmpl["negative"] + q, layer=layer), hidden_dim)
                if vp is not None and vn is not None:
                    diffs.append(vp - vn)
            except Exception:
                pass
        if len(diffs) < 10:
            log.warning(f"  {name}: only {len(diffs)} pairs"); continue
        sv = np.mean(diffs, axis=0); norm = np.linalg.norm(sv)
        if norm > 1e-10: sv /= norm
        aligns, bt, st = [], [], []
        for q in tqdm(test_qs[:n_test], desc=f"Test {name}", leave=False):
            try:
                base = client.bridge(q, mode="passive", n_steps=0)
                strd = client.bridge(tmpl["positive"] + q, mode="passive", n_steps=0)
                b_txt = base.get("enriched_text", base.get("decoded_text", ""))
                s_txt = strd.get("enriched_text", strd.get("decoded_text", ""))
                if not b_txt or not s_txt: continue
                vb = extract_mean_vector(client.encode(b_txt[:500], layer=layer), hidden_dim)
                vs = extract_mean_vector(client.encode(s_txt[:500], layer=layer), hidden_dim)
                if vb is not None and vs is not None:
                    aligns.append(cosine(vs - vb, sv))
                    bt.append(b_txt[:200])
                    st.append(s_txt[:200])
            except Exception:
                pass
        if not aligns: continue
        results.append(SteeringResult(name, tmpl["description"], len(aligns), float(norm),
                                      float(np.mean(aligns)), float(np.std(aligns)),
                                      float(np.mean([a > 0 for a in aligns])),
                                      bt[:3], st[:3]))
    return results


# ═══════════════════════════════════════════════════════════════════
#  ZOMBIE ↔ ZOMBIE LOOP
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LoopResult:
    seed_question: str; n_iterations: int; k_per_step: int; iterations: list
    total_drift: float; convergence_step: int; texts: list
    def summary(self):
        cv = f"conv@{self.convergence_step}" if self.convergence_step >= 0 else "exploring"
        ic = "🌀" if self.total_drift > 0.3 else "📍"
        lines = [f"  {ic} drift={self.total_drift:.4f} {self.n_iterations}steps {cv}",
                 f"    \"{self.seed_question[:70]}...\""]
        for it in self.iterations:
            lines.append(f"    step {it['step']}: novel={it['novelty']:.3f} "
                         f"{'█'*max(1,int(it['novelty']*20))}")
        return "\n".join(lines)

def eval_loop(client, n_seeds=5, n_iter=8, k=30, layer=9, hidden_dim=2560):
    seeds = load_seed_texts(n_seeds)
    results = []
    for i, seed in enumerate(seeds):
        log.info(f"Loop {i+1}/{n_seeds}")
        vecs, texts, iters, cur = [], [], [], seed
        for step in range(n_iter):
            try:
                res = client.bridge(cur, mode="ruminate", n_steps=k)
                out = res.get("enriched_text", res.get("decoded_text", ""))
                if not out or len(out.strip()) < 10: break
                v = extract_mean_vector(client.encode(out[:500], layer=layer), hidden_dim)
                if v is None: break
                disp = 0 if not vecs else 1 - cosine(v, vecs[-1])
                cum  = 0 if not vecs else 1 - cosine(v, vecs[0])
                nov  = 1.0 if not vecs else 1 - max(cosine(v, p) for p in vecs)
                vecs.append(v); texts.append(out)
                iters.append({"step": step, "displacement": disp, "cumulative": cum, "novelty": nov})
                cur = out
            except Exception as e:
                log.warning(f"  step {step}: {e}"); break
        if len(vecs) < 2:
            results.append(LoopResult(seed[:100], 0, k, [], 0, -1, [])); continue
        drift = 1 - cosine(vecs[0], vecs[-1])
        conv = next((it["step"] for it in iters[1:] if it["novelty"] < 0.05), -1)
        results.append(LoopResult(seed[:100], len(iters), k, iters, drift, conv, texts))
    return results


# ═══════════════════════════════════════════════════════════════════
#  CLAUDE ↔ ZOMBIE DIALOGUE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DialogueResult:
    seed_question: str; n_turns: int; k_per_turn: int; turns: list
    claude_drift: float; enrichment_drift: float; mean_enrichment_novelty: float
    convergence_turn: int; texts_claude: list; texts_enrichment: list
    claude_evolves: bool
    def summary(self):
        ev = "🧠 evolves" if self.claude_evolves else "📍 static"
        cv = f"conv@{self.convergence_turn}" if self.convergence_turn >= 0 else "exploring"
        lines = [f"  {ev} c_drift={self.claude_drift:.4f} e_drift={self.enrichment_drift:.4f} {cv}",
                 f"    \"{self.seed_question[:70]}...\""]
        for t in self.turns:
            lines.append(f"    turn {t['step']}: C=\"{t['claude_text'][:50]}...\" "
                         f"novel={t['enrichment_novelty']:.3f}")
        return "\n".join(lines)

def eval_dialogue(client, n_seeds=3, n_turns=6, k=30, layer=9, hidden_dim=2560):
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.error("ANTHROPIC_API_KEY not set"); return []
    cl = anthropic.Anthropic(api_key=key)
    seeds = load_seed_texts(n_seeds)
    results = []

    for i, seed in enumerate(seeds):
        log.info(f"Dialogue {i+1}/{n_seeds}")
        msgs, c_vecs, e_vecs, c_texts, e_texts, turns = [], [], [], [], [], []

        for turn in range(n_turns):
            try:
                if turn == 0:
                    msgs.append({"role": "user", "content": seed})
                else:
                    msgs.append({"role": "user",
                                 "content": (f"Alternative perspective from a different reasoning process:\n\n"
                                             f"{e_texts[-1]}\n\n"
                                             f"Integrate what's useful, challenge what's wrong, deepen your analysis.")})
                resp = cl.messages.create(model="claude-sonnet-4-6", max_tokens=512,
                                          messages=msgs, timeout=60)
                c_text = resp.content[0].text
                msgs.append({"role": "assistant", "content": c_text})
                c_texts.append(c_text)

                vc = extract_mean_vector(client.encode(c_text[:500], layer=layer), hidden_dim)
                if vc is None: break
                c_vecs.append(vc)

                br = client.bridge(c_text, mode="ruminate", n_steps=k)
                e_text = br.get("enriched_text", br.get("decoded_text", c_text))
                if not e_text or len(e_text.strip()) < 10: e_text = c_text
                e_texts.append(e_text)

                ve = extract_mean_vector(client.encode(e_text[:500], layer=layer), hidden_dim)
                if ve is None: break
                e_vecs.append(ve)

                c_disp = 0 if turn == 0 else 1 - cosine(vc, c_vecs[-2])
                e_nov  = 1.0 if turn == 0 else 1 - max(cosine(ve, p) for p in e_vecs[:-1])
                gain   = 0 if turn == 0 else cosine(ve - vc, e_vecs[0] - c_vecs[0])

                turns.append({"step": turn, "claude_text": c_text[:300],
                              "enrichment_text": e_text[:300],
                              "claude_displacement": c_disp,
                              "enrichment_novelty": e_nov, "turn_gain": gain})
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"  turn {turn}: {e}"); break

        if len(c_vecs) < 2:
            results.append(DialogueResult(seed[:100], 0, k, [], 0, 0, 0, -1, [], [], False))
            continue
        cd = 1 - cosine(c_vecs[0], c_vecs[-1])
        ed = 1 - cosine(e_vecs[0], e_vecs[-1])
        novs = [t["enrichment_novelty"] for t in turns[1:]]
        mn = float(np.mean(novs)) if novs else 0
        cv = next((t["step"] for t in turns[1:] if t["enrichment_novelty"] < 0.05), -1)
        results.append(DialogueResult(seed[:100], len(turns), k, turns, cd, ed, mn, cv,
                                      c_texts, e_texts, cd > 0.1))
    return results


# ═══════════════════════════════════════════════════════════════════
#  WEB SEARCH — DuckDuckGo (no API key needed)
# ═══════════════════════════════════════════════════════════════════

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo. Returns concatenated snippets."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"[No results for: {query}]"
        return "\n".join(f"- {r.get('title','')}: {r.get('body','')[:200]}"
                         for r in results)
    except ImportError:
        log.warning("duckduckgo-search not installed: pip install duckduckgo-search")
        return "[Web search unavailable]"
    except Exception as e:
        return f"[Search error: {e}]"


def web_search_multi(queries: list, max_per_query: int = 3) -> dict:
    results = {}
    for q in queries:
        results[q] = web_search(q, max_per_query)
        time.sleep(0.3)
    return results


# ═══════════════════════════════════════════════════════════════════
#  FRONTIER — Claude vs Claude+Zombie on hard web+reasoning tasks
# ═══════════════════════════════════════════════════════════════════

FRONTIER_TASKS = [
    {
        "id": "cross_source",
        "category": "Cross-source synthesis",
        "generator": lambda: {
            "question": "What are the current leading theories about the origin of Fast Radio Bursts (FRBs), "
                        "and where do recent observations contradict each other?",
            "search_queries": ["Fast Radio Bursts origin theories 2025",
                               "FRB contradictions observations 2025",
                               "magnetar FRB evidence against"],
        },
    },
    {
        "id": "temporal_reasoning",
        "category": "Temporal reasoning",
        "generator": lambda: {
            "question": "Trace the chain of events from the first detection of gravitational waves (LIGO 2015) "
                        "to the current state of multi-messenger astronomy. What capabilities exist now that "
                        "were theoretical then, and what predicted capabilities have NOT materialized?",
            "search_queries": ["multi-messenger astronomy progress 2025",
                               "LIGO discoveries timeline",
                               "gravitational wave astronomy unfulfilled predictions"],
        },
    },
    {
        "id": "adversarial_factcheck",
        "category": "Adversarial fact-check",
        "generator": lambda: {
            "question": "Evaluate this claim: 'Recent studies have shown that transformer attention patterns "
                        "are mathematically equivalent to kernel regression, proving that attention is just "
                        "a form of nearest-neighbor lookup.' Is this accurate, misleading, or false? "
                        "Cite specific papers.",
            "search_queries": ["transformer attention kernel regression equivalence",
                               "attention mechanism nearest neighbor",
                               "Tsai 2019 kernel attention criticism"],
        },
    },
    {
        "id": "open_research",
        "category": "Open research question",
        "generator": lambda: {
            "question": "What is the current evidence for and against the hypothesis that large language models "
                        "develop world models rather than merely learning statistical correlations? "
                        "What experiment would definitively resolve this?",
            "search_queries": ["LLM world models evidence 2025",
                               "Othello GPT world model Li 2023",
                               "LLM statistical correlations vs understanding 2025"],
        },
    },
    {
        "id": "policy_tradeoff",
        "category": "Policy tradeoff analysis",
        "generator": lambda: {
            "question": "Compare the EU AI Act, the US Executive Order on AI (Oct 2023), and China's "
                        "AI regulations. Where do they agree, where do they conflict, and what are "
                        "the practical consequences for a company deploying an open-source LLM globally?",
            "search_queries": ["EU AI Act requirements 2025",
                               "US AI executive order implementation 2025",
                               "China AI regulation comparison EU US"],
        },
    },
]


@dataclass
class FrontierTaskResult:
    task_id: str; category: str; question: str
    score_a: int; score_b: int; score_c: int
    text_a: str; text_b: str; text_c: str
    web_context: str


@dataclass
class FrontierResult:
    n_tasks: int; tasks: list
    mean_a: float; mean_b: float; mean_c: float
    c_vs_a_win: float; c_vs_b_win: float

    def summary(self):
        lines = [f"Frontier ({self.n_tasks} tasks)",
                 f"  A (Claude alone):       {self.mean_a:.2f}",
                 f"  B (Claude+web):         {self.mean_b:.2f}",
                 f"  C (Claude+web+zombie):  {self.mean_c:.2f}",
                 f"  C>A: {self.c_vs_a_win:.0%}  C>B: {self.c_vs_b_win:.0%}"]
        for t in self.tasks:
            lines.append(f"    {t['category']:30s} A={t['score_a']} B={t['score_b']} C={t['score_c']}")
        return "\n".join(lines)


def eval_frontier(client, n_steps=60):
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.error("ANTHROPIC_API_KEY required"); return FrontierResult(0, [], 0, 0, 0, 0, 0)
    cl = anthropic.Anthropic(api_key=key)

    def ask(msgs, mt=1024):
        return cl.messages.create(model="claude-sonnet-4-6", max_tokens=mt,
                                  messages=msgs, timeout=90).content[0].text

    judge_prompt = ("Rate this answer 1-7 on: factual accuracy, depth of analysis, "
                    "identification of nuance/contradictions, and actionable insight. "
                    "1=poor, 7=exceptional. Reply with ONLY one integer.")

    def judge(q, a):
        try:
            return max(1, min(7, int(ask([{"role": "user",
                "content": f"Question: {q}\n\nAnswer: {a}\n\n{judge_prompt}"}], 10).strip()[0])))
        except Exception:
            return 4

    task_results = []
    for task_def in tqdm(FRONTIER_TASKS, desc="Frontier"):
        task = task_def["generator"]()
        q = task["question"]
        search_queries = task["search_queries"]

        try:
            log.info(f"  Searching: {search_queries}")
            web_ctx = web_search_multi(search_queries)
            web_text = "\n\n".join(f"[{query}]\n{snip}" for query, snip in web_ctx.items())

            text_a = ask([{"role": "user", "content": q}])
            text_b = ask([{"role": "user",
                           "content": f"{q}\n\nHere is recent web research:\n{web_text}\n\n"
                                      f"Use this to give a thorough, evidence-based answer."}])

            bridge_r = client.bridge(text_b, mode="ruminate", n_steps=n_steps)
            enrichment = bridge_r.get("enriched_text", bridge_r.get("decoded_text", text_b))
            text_c = ask([
                {"role": "user", "content": f"{q}\n\nWeb research:\n{web_text}"},
                {"role": "assistant", "content": text_b},
                {"role": "user",
                 "content": f"A parallel reasoning process produced this analysis:\n\n{enrichment}\n\n"
                            f"Integrate useful insights, challenge errors, give your final answer."},
            ])

            sa = judge(q, text_a)
            sb = judge(q, text_b)
            sc = judge(q, text_c)

            task_results.append(asdict(FrontierTaskResult(
                task_id=task_def["id"], category=task_def["category"], question=q[:200],
                score_a=sa, score_b=sb, score_c=sc,
                text_a=text_a[:500], text_b=text_b[:500], text_c=text_c[:500],
                web_context=web_text[:500],
            )))
            time.sleep(1)

        except Exception as e:
            log.warning(f"  Frontier error: {e}")

    if not task_results:
        return FrontierResult(0, [], 0, 0, 0, 0, 0)

    sa = [t["score_a"] for t in task_results]
    sb = [t["score_b"] for t in task_results]
    sc = [t["score_c"] for t in task_results]
    return FrontierResult(
        n_tasks=len(task_results), tasks=task_results,
        mean_a=float(np.mean(sa)), mean_b=float(np.mean(sb)), mean_c=float(np.mean(sc)),
        c_vs_a_win=float(np.mean([c > a for a, c in zip(sa, sc)])),
        c_vs_b_win=float(np.mean([c > b for b, c in zip(sb, sc)])),
    )


# ═══════════════════════════════════════════════════════════════════
#  WEB DIALOGUE — Claude↔Zombie with web search at each turn
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WebDialogueResult:
    seed_question: str; n_turns: int; k_per_turn: int; turns: list
    claude_drift: float; enrichment_drift: float; mean_enrichment_novelty: float
    convergence_turn: int; claude_evolves: bool
    texts_claude: list; texts_enrichment: list; web_queries_used: list

    def summary(self):
        ev = "🧠 evolves" if self.claude_evolves else "📍 static"
        cv = f"conv@{self.convergence_turn}" if self.convergence_turn >= 0 else "exploring"
        lines = [f"  {ev} c_drift={self.claude_drift:.4f} {cv}",
                 f"    \"{self.seed_question[:70]}...\""]
        for t in self.turns:
            wq = t.get("web_query", "")[:40]
            lines.append(f"    turn {t['step']}: novel={t['enrichment_novelty']:.3f} web=\"{wq}\"")
        return "\n".join(lines)


def eval_webdialogue(client, n_seeds=3, n_turns=6, k=30, layer=9, hidden_dim=2560):
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.error("ANTHROPIC_API_KEY required"); return []
    cl_api = anthropic.Anthropic(api_key=key)

    seeds = [
        "What is the current state of quantum error correction, and how close are we to fault-tolerant quantum computing?",
        "How has the Russo-Ukrainian conflict affected global food security, and what structural changes have occurred in grain supply chains?",
        "What are the most promising approaches to solving the protein folding problem beyond AlphaFold, and what limitations remain?",
    ][:n_seeds]

    results = []
    for i, seed in enumerate(seeds):
        log.info(f"WebDialogue {i+1}/{n_seeds}")
        msgs, c_vecs, e_vecs, c_texts, e_texts, turns_data, web_qs = [], [], [], [], [], [], []

        for turn in range(n_turns):
            try:
                if turn == 0:
                    msgs.append({"role": "user", "content": seed})
                else:
                    msgs.append({"role": "user",
                                 "content": (f"Updated analysis from parallel reasoning + latest web data:\n\n"
                                             f"{e_texts[-1]}\n\n"
                                             f"Integrate, challenge, deepen. What's your updated view?")})

                c_resp = cl_api.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=512, messages=msgs, timeout=60)
                c_text = c_resp.content[0].text
                msgs.append({"role": "assistant", "content": c_text})
                c_texts.append(c_text)

                vc = extract_mean_vector(client.encode(c_text[:500], layer=layer), hidden_dim)
                if vc is None: break
                c_vecs.append(vc)

                sq_resp = cl_api.messages.create(
                    model="claude-sonnet-4-6", max_tokens=60,
                    messages=[{"role": "user",
                               "content": f"Based on this analysis:\n{c_text[:300]}\n\n"
                                          f"Generate ONE specific web search query to verify "
                                          f"or expand the most uncertain claim. "
                                          f"Reply with ONLY the search query, nothing else."}],
                    timeout=30)
                search_query = sq_resp.content[0].text.strip().strip('"').strip("'")
                web_qs.append(search_query)

                web_result = web_search(search_query, max_results=3)

                zombie_input = f"{c_text}\n\n[Web search: {search_query}]\n{web_result}"
                br = client.bridge(zombie_input, mode="ruminate", n_steps=k)
                e_text = br.get("enriched_text", br.get("decoded_text", c_text))
                if not e_text or len(e_text.strip()) < 10: e_text = c_text
                e_texts.append(e_text)

                ve = extract_mean_vector(client.encode(e_text[:500], layer=layer), hidden_dim)
                if ve is None: break
                e_vecs.append(ve)

                c_disp = 0 if turn == 0 else 1 - cosine(vc, c_vecs[-2])
                e_nov  = 1.0 if turn == 0 else 1 - max(cosine(ve, p) for p in e_vecs[:-1])

                turns_data.append({"step": turn, "claude_displacement": c_disp,
                                   "enrichment_novelty": e_nov,
                                   "web_query": search_query[:100],
                                   "claude_text": c_text[:200],
                                   "enrichment_text": e_text[:200]})
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"  turn {turn}: {e}"); break

        if len(c_vecs) < 2:
            results.append(WebDialogueResult(seed[:100], 0, k, [], 0, 0, 0, -1, False, [], [], []))
            continue
        cd = 1 - cosine(c_vecs[0], c_vecs[-1])
        ed = 1 - cosine(e_vecs[0], e_vecs[-1])
        novs = [t["enrichment_novelty"] for t in turns_data[1:]]
        mn = float(np.mean(novs)) if novs else 0
        cv = next((t["step"] for t in turns_data[1:] if t["enrichment_novelty"] < 0.05), -1)
        results.append(WebDialogueResult(seed[:100], len(turns_data), k, turns_data,
                                         cd, ed, mn, cv, cd > 0.1,
                                         c_texts, e_texts, web_qs))
    return results


# ═══════════════════════════════════════════════════════════════════
#  STEERED DIALOGUE — Claude↔Zombie with contrastive steering vectors
# ═══════════════════════════════════════════════════════════════════

def _extract_steering_vectors(client, layer, hidden_dim, n_extract=30):
    """Extract all 4 contrastive steering vectors. Returns {name: normalized_vec}."""
    qs = load_technical_questions(n_extract)
    q_texts = [q["text"] for q in qs]
    vectors = {}
    for name, tmpl in CONTRAST_TEMPLATES.items():
        diffs = []
        for q in tqdm(q_texts, desc=f"Extract {name}", leave=False):
            try:
                vp = extract_mean_vector(client.encode(tmpl["positive"] + q, layer=layer), hidden_dim)
                vn = extract_mean_vector(client.encode(tmpl["negative"] + q, layer=layer), hidden_dim)
                if vp is not None and vn is not None:
                    diffs.append(vp - vn)
            except Exception:
                pass
        if len(diffs) >= 10:
            sv = np.mean(diffs, axis=0)
            norm = np.linalg.norm(sv)
            if norm > 1e-10:
                vectors[name] = sv / norm
                log.info(f"  {name}: {len(diffs)} pairs, |v|={norm:.4f}")
    return vectors


@dataclass
class SteerDialogueResult:
    seed_question: str
    steering_concept: str
    n_turns: int
    k_per_turn: int
    turns: list
    # Claude evolution
    claude_drift: float
    claude_evolves: bool
    # Steering persistence: does alignment with steering vector grow or decay?
    alignment_trajectory: list  # cosine(claude_response_vec, steering_vec) per turn
    alignment_trend: float      # slope of linear fit — positive = steering persists/amplifies
    # Enrichment
    enrichment_drift: float
    mean_enrichment_novelty: float
    convergence_turn: int
    # Text
    texts_claude: list
    texts_enrichment: list

    def summary(self):
        ev = "🧠 evolves" if self.claude_evolves else "📍 static"
        trend = ("📈 amplifies" if self.alignment_trend > 0.01
                 else "📉 decays" if self.alignment_trend < -0.01
                 else "→ stable")
        cv = f"conv@{self.convergence_turn}" if self.convergence_turn >= 0 else "exploring"
        lines = [
            f"  [{self.steering_concept}] {ev} {trend} {cv}",
            f"    c_drift={self.claude_drift:.4f} align_trend={self.alignment_trend:+.4f}",
            f"    \"{self.seed_question[:60]}...\"",
            f"    Alignment: {' → '.join(f'{a:.3f}' for a in self.alignment_trajectory)}",
        ]
        for t in self.turns:
            lines.append(f"    turn {t['step']}: steer_align={t['steering_alignment']:.3f} "
                         f"novel={t['enrichment_novelty']:.3f}")
        return "\n".join(lines)


def run_steered_dialogue(client, seed_question, steering_vec, steering_name,
                         n_turns=6, k_per_turn=30, layer=9, hidden_dim=2560):
    """Claude↔Zombie dialogue where the zombie is steered at each turn.

    Turn 0: Claude answers
            → zombie ruminates with steering concept prefix biasing the latent space
            → enriched output reflects the steering direction
    Turn N: Claude reads steered enrichment → responds
            → zombie ruminates with same steering
            → ...

    Tracks: does the steering direction PERSIST in Claude's responses?
    Does it amplify (Claude internalizes the direction) or decay (Claude resists)?
    """
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None

    cl = anthropic.Anthropic(api_key=key)
    msgs, c_vecs, e_vecs, c_texts, e_texts, turns_data = [], [], [], [], [], []
    align_trajectory = []

    for turn in range(n_turns):
        try:
            # ── Claude's turn ──
            if turn == 0:
                msgs.append({"role": "user", "content": seed_question})
            else:
                msgs.append({"role": "user",
                             "content": (f"Here's an alternative analysis from a parallel reasoning process:\n\n"
                                         f"{e_texts[-1]}\n\n"
                                         f"Integrate what's useful, challenge what's wrong, deepen your analysis.")})

            resp = cl.messages.create(model="claude-sonnet-4-6", max_tokens=512,
                                      messages=msgs, timeout=60)
            c_text = resp.content[0].text
            msgs.append({"role": "assistant", "content": c_text})
            c_texts.append(c_text)

            vc = extract_mean_vector(client.encode(c_text[:500], layer=layer), hidden_dim)
            if vc is None:
                break
            c_vecs.append(vc)

            # Measure Claude's alignment with the steering direction
            steer_align = cosine(vc, steering_vec)
            align_trajectory.append(steer_align)

            # ── Zombie's steered turn ──
            # Prepend the concept's positive prefix to bias the rumination direction
            steered_input = (
                f"{CONTRAST_TEMPLATES[steering_name]['positive']}\n"
                f"{c_text}"
            )
            br = client.bridge(steered_input, mode="ruminate", n_steps=k_per_turn)
            e_text = br.get("enriched_text", br.get("decoded_text", c_text))
            if not e_text or len(e_text.strip()) < 10:
                e_text = c_text
            e_texts.append(e_text)

            ve = extract_mean_vector(client.encode(e_text[:500], layer=layer), hidden_dim)
            if ve is None:
                break
            e_vecs.append(ve)

            c_disp = 0 if turn == 0 else 1 - cosine(vc, c_vecs[-2])
            e_nov  = 1.0 if turn == 0 else 1 - max(cosine(ve, p) for p in e_vecs[:-1])

            turns_data.append({
                "step": turn,
                "claude_text": c_text[:300],
                "enrichment_text": e_text[:300],
                "claude_displacement": c_disp,
                "enrichment_novelty": e_nov,
                "steering_alignment": steer_align,
            })
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"  turn {turn}: {e}"); break

    if len(c_vecs) < 2:
        return None

    cd = 1 - cosine(c_vecs[0], c_vecs[-1])
    ed = 1 - cosine(e_vecs[0], e_vecs[-1])
    novs = [t["enrichment_novelty"] for t in turns_data[1:]]
    mn = float(np.mean(novs)) if novs else 0
    cv = next((t["step"] for t in turns_data[1:] if t["enrichment_novelty"] < 0.05), -1)

    # Linear trend of alignment_trajectory: positive = Claude increasingly aligns
    if len(align_trajectory) >= 2:
        x = np.arange(len(align_trajectory))
        slope, _ = np.polyfit(x, align_trajectory, 1)
        align_trend = float(slope)
    else:
        align_trend = 0.0

    return SteerDialogueResult(
        seed_question=seed_question[:100],
        steering_concept=steering_name,
        n_turns=len(turns_data),
        k_per_turn=k_per_turn,
        turns=turns_data,
        claude_drift=cd,
        claude_evolves=cd > 0.1,
        alignment_trajectory=align_trajectory,
        alignment_trend=align_trend,
        enrichment_drift=ed,
        mean_enrichment_novelty=mn,
        convergence_turn=cv,
        texts_claude=c_texts,
        texts_enrichment=e_texts,
    )


def eval_steerdialogue(client, n_seeds=2, n_turns=6, k=30, layer=9, hidden_dim=2560, n_extract=30):
    """Run steered Claude↔Zombie dialogue for each of the 4 concept directions."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.error("ANTHROPIC_API_KEY not set"); return []

    log.info("Extracting steering vectors...")
    steering_vecs = _extract_steering_vectors(client, layer, hidden_dim, n_extract)
    if not steering_vecs:
        log.error("No steering vectors extracted"); return []

    seeds = load_seed_texts(n_seeds)
    results = []

    for i, seed in enumerate(seeds):
        for concept_name, sv in steering_vecs.items():
            log.info(f"SteerDialogue seed={i+1} concept={concept_name}")
            r = run_steered_dialogue(client, seed, sv, concept_name,
                                     n_turns, k, layer, hidden_dim)
            if r is not None:
                results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="ERIS v5 — Phase 1 Complete Evaluation")
    p.add_argument("--eris-url", default="http://localhost:8001")
    p.add_argument("--metric", nargs="+", default=["all"],
                   choices=["all"] + sorted(ALL_METRICS))
    p.add_argument("--n-pairs",      type=int, default=200, help="STS-B pairs for M4")
    p.add_argument("--n-questions",  type=int, default=30,  help="Questions for M5/ABC")
    p.add_argument("--n-extract",    type=int, default=30,  help="Contrastive pairs for steering")
    p.add_argument("--n-test",       type=int, default=10,  help="Test questions for steering")
    p.add_argument("--n-seeds",      type=int, default=5,   help="Seeds for loop/dialogue")
    p.add_argument("--n-iterations", type=int, default=8,   help="Loop iterations")
    p.add_argument("--n-turns",      type=int, default=6,   help="Dialogue turns")
    p.add_argument("--n-steps",      type=int, default=60,  help="Rumination steps for ABC")
    p.add_argument("--k-per-step",   type=int, default=30,  help="K for loop/dialogue")
    p.add_argument("--tag",          default="default")
    p.add_argument("--output-dir",   default="eval_results")
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    metrics = ALL_METRICS if "all" in args.metric else set(args.metric)
    Path(args.output_dir).mkdir(exist_ok=True)

    client = ERISClient(args.eris_url)
    health = client.health()
    model = health.get("model", "unknown")
    cfg = get_model_config(model)
    layer = cfg["m4_layer"]

    test_enc = client.encode("test", layer=layer)
    hidden_dim = test_enc.get("hidden_dim", 2560)
    log.info(f"Model: {model}, layer: {layer}, hidden_dim: {hidden_dim}")

    R = {"tag": args.tag, "model": model, "config": cfg,
         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        if "m4" in metrics:
            log.info("═"*60 + "\n  M4 — Projection Fidelity\n" + "═"*60)
            r = eval_m4(client, cfg, args.n_pairs)
            print(f"\n{r.summary()}\n"); R["m4"] = asdict(r)

        if "m5" in metrics:
            log.info("═"*60 + "\n  M5 — LatentMAS Gain\n" + "═"*60)
            r = eval_m5(client, args.n_questions)
            print(f"\n{r.summary()}\n"); R["m5"] = asdict(r)

        if "m6" in metrics:
            log.info("═"*60 + "\n  M6 — Implicit Features\n" + "═"*60)
            r = eval_m6(client, min(20, args.n_questions))
            print(f"\n{r.summary()}\n"); R["m6"] = asdict(r)

        if "abc" in metrics:
            log.info("═"*60 + "\n  ABC — Claude A/B/C\n" + "═"*60)
            r = eval_abc(client, args.n_questions, args.n_steps)
            if r.n_questions > 0: print(f"\n{r.summary()}\n")
            R["abc"] = asdict(r)

        if "steering" in metrics:
            log.info("═"*60 + "\n  Contrastive Steering\n" + "═"*60)
            rs = eval_steering(client, layer, hidden_dim, args.n_extract, args.n_test)
            R["steering"] = [asdict(r) for r in rs]
            for r in rs: print(r.summary())

        if "loop" in metrics:
            log.info("═"*60 + "\n  Zombie↔Zombie Loop\n" + "═"*60)
            rs = eval_loop(client, args.n_seeds, args.n_iterations, args.k_per_step, layer, hidden_dim)
            R["loop"] = [asdict(r) for r in rs]
            for r in rs: print(r.summary())

        if "dialogue" in metrics:
            log.info("═"*60 + "\n  Claude↔Zombie Dialogue\n" + "═"*60)
            rs = eval_dialogue(client, args.n_seeds, args.n_turns, args.k_per_step, layer, hidden_dim)
            R["dialogue"] = [asdict(r) for r in rs]
            for r in rs: print(r.summary())

        if "steerdialogue" in metrics:
            log.info("═"*60 + "\n  Steered Claude↔Zombie Dialogue\n" + "═"*60)
            rs = eval_steerdialogue(client, min(2, args.n_seeds), args.n_turns,
                                    args.k_per_step, layer, hidden_dim, args.n_extract)
            R["steerdialogue"] = [asdict(r) for r in rs]
            for r in rs: print(r.summary())

        if "frontier" in metrics:
            log.info("═"*60 + "\n  Frontier — Claude vs Claude+Web vs Claude+Web+Zombie\n" + "═"*60)
            r = eval_frontier(client, args.n_steps)
            print(f"\n{r.summary()}\n"); R["frontier"] = asdict(r)

        if "webdialogue" in metrics:
            log.info("═"*60 + "\n  Web Dialogue — Claude↔Zombie + Live Web\n" + "═"*60)
            rs = eval_webdialogue(client, args.n_seeds, args.n_turns, args.k_per_step, layer, hidden_dim)
            R["webdialogue"] = [asdict(r) for r in rs]
            for r in rs: print(r.summary())

    finally:
        client.close()

    out = Path(args.output_dir) / f"phase1_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    log.info(f"Saved to {out}")

    # ── Grand Summary ──
    print(f"\n{'═'*60}")
    print(f"  PHASE 1 COMPLETE — {model}")
    print(f"{'═'*60}")
    if "m4" in R:
        print(f"  M4 Projection:   ρ={R['m4']['spearman_r']:.4f} {'✅' if R['m4']['pass_threshold'] else '❌'}")
    if "m5" in R:
        print(f"  M5 LatentMAS:    {'✅' if R['m5']['gain_detected'] else '❌'}  "
              f"K=0→{R['m5']['displacement_means'][0]:.4f} K=60→{R['m5']['displacement_means'][-1]:.4f}")
    if "m6" in R:
        m6_val = f"{R['m6']['implicit_ratio']:.1%}" if R["m6"]["sae_available"] else "SAE n/a"
        print(f"  M6 Implicit:     {m6_val}")
    if "abc" in R and R["abc"]["n_questions"] > 0:
        r = R["abc"]
        print(f"  ABC:             A={r['mean_a']:.2f} B={r['mean_b']:.2f} C={r['mean_c']:.2f}  "
              f"C>A={r['c_vs_a_win']:.0%} p={r['wilcoxon_ca_p']:.4f}")
    if "steering" in R:
        rs = R["steering"]
        n_pass = sum(1 for r in rs if r["positive_rate"] > 0.6)
        print(f"  Steering:        {n_pass}/{len(rs)} concepts effective")
    if "loop" in R:
        rs = R["loop"]
        exploring = sum(1 for r in rs if r["convergence_step"] < 0)
        print(f"  Loop:            {exploring}/{len(rs)} seeds still exploring")
    if "dialogue" in R:
        rs = R["dialogue"]
        evolving = sum(1 for r in rs if r["claude_evolves"])
        print(f"  Dialogue:        {evolving}/{len(rs)} Claude instances evolving")
    if "steerdialogue" in R:
        rs = R["steerdialogue"]
        amplifying = sum(1 for r in rs if r["alignment_trend"] > 0.01)
        evolving   = sum(1 for r in rs if r["claude_evolves"])
        print(f"  SteerDialogue:   {amplifying}/{len(rs)} amplifying, "
              f"{evolving}/{len(rs)} evolving")
    if "frontier" in R:
        r = R["frontier"]
        print(f"  Frontier:        A={r['mean_a']:.2f} B={r['mean_b']:.2f} C={r['mean_c']:.2f}  "
              f"C>A={r['c_vs_a_win']:.0%}  C>B={r['c_vs_b_win']:.0%}")
    if "webdialogue" in R:
        rs = R["webdialogue"]
        evolving = sum(1 for r in rs if r["claude_evolves"])
        print(f"  WebDialogue:     {evolving}/{len(rs)} Claude instances evolving")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
