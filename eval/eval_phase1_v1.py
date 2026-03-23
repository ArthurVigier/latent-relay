#!/usr/bin/env python3
"""
ERIS v5 — Phase 1 Evaluation v1
================================

Replaces three weak measurements from eval_phase1.py with stronger ones.
All other metrics (M4, M6, steering, dialogue, steerdialogue, frontier,
webdialogue) are identical and not repeated here.

────────────────────────────────────────────────────────────────────────
WHAT CHANGED AND WHY
────────────────────────────────────────────────────────────────────────

M5_v1  Displacement → GSM8K zombie accuracy curve
  Problem:  total_displacement grows whenever hidden states change, even
            if the model is doing nothing useful.  Displacement measures
            activity, not quality.
  Fix:      Run the zombie on GSM8K math questions at K=0,5,15,30,60.
            Extract the numerical answer from decoded text.  Compare to
            ground truth.  Signal: does more rumination improve accuracy?
  Required: no API key (zombie-only, stateless bridge call)

ABC_v1  Self-judge → ground-truth accuracy on MMLU-Pro + GSM8K
  Problem:  Claude judging Claude answers is biased toward its own style
            and toward longer/more formal outputs.  "Rate depth 1-5" with
            the same model as answerer is circular.
  Fix:      Use questions that have known correct answers.
            • GSM8K:    extract final integer from response, compare to "#### N"
            • MMLU-Pro: extract answer letter (A-E), compare to ground truth
            Score = fraction of correct answers per condition.
            No judge LLM needed.
  Required: ANTHROPIC_API_KEY

Loop_v1  Geometric novelty → relevance-aware exploration
  Problem:  1 - cosine(v_t, v_{t-1}) measures that outputs are different,
            not that they are meaningful.  A loop producing varied nonsense
            scores high on novelty.
  Fix:      Track two things per step:
            • novelty:   1 - max_cosine(v_t, all previous) — same as before
            • relevance: cosine(v_t, seed_vec)             — NEW
            Derive a quality flag:
            • meaningful_exploration: novelty > 0.05 AND relevance > 0.2
            • irrelevance_drift:      relevance drops below 0.15
            A good loop explores (high novelty) without drifting off-topic
            (high relevance).  A bad loop either stagnates OR wanders.
  Required: no API key

────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────

  python eris_server.py --model Qwen/Qwen3.5-4B --port 8001

  # Channel validation only (no API key)
  python eval/eval_phase1_v1.py --metric m4 m5 loop

  # Full v1 suite
  ANTHROPIC_API_KEY=sk-... python eval/eval_phase1_v1.py --metric all

Requirements:
  pip install datasets anthropic httpx scipy numpy tqdm
"""

import os
import re
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
log = logging.getLogger("eris_phase1_v1")

# v1 changes these three; the rest delegate to eval_phase1
ALL_METRICS_V1 = {"m4", "m5", "m6", "abc", "steering", "loop",
                  "dialogue", "steerdialogue", "frontier", "webdialogue"}


# ═══════════════════════════════════════════════════════════════════
#  Shared config / client / helpers  (identical to eval_phase1.py)
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "default":  {"m4_layer": 9,  "m4_pooling": "last_token", "m4_centered": True, "m4_threshold": 0.6},
    "qwen3.5":  {"m4_layer": 9,  "m4_pooling": "last_token", "m4_centered": True, "m4_threshold": 0.6},
    "qwen3":    {"m4_layer": 35, "m4_pooling": "last_token", "m4_centered": True, "m4_threshold": 0.6},
}

def get_model_config(model_name: str) -> dict:
    m = str(model_name).lower()
    if "3.5" in m: return MODEL_CONFIGS["qwen3.5"]
    if "qwen3" in m: return MODEL_CONFIGS["qwen3"]
    return MODEL_CONFIGS["default"]


class ERISClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self._model = None

    def health(self) -> dict:
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        d = r.json(); self._model = d.get("model", "unknown"); return d

    @property
    def model_name(self):
        if self._model is None: self.health()
        return self._model

    def encode(self, text: str, layer: int = -1) -> dict:
        r = self.client.post(f"{self.base_url}/v1/encode",
                             json={"text": text, "return_layers": [layer], "compact": True})
        r.raise_for_status(); return r.json()

    def bridge(self, text: str, mode: str = "ruminate", n_steps: int = 60,
               analyses: list = None) -> dict:
        r = self.client.post(f"{self.base_url}/v1/bridge", json={
            "claude_text": text, "mode": mode, "n_steps": n_steps,
            "analyses": analyses or [], "decode_after": True, "max_new_tokens": 512,
        })
        r.raise_for_status(); return r.json()

    def close(self): self.client.close()


def decode_b64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)

def extract_vector(enc: dict, hidden_dim: int) -> np.ndarray:
    for val in enc.get("hidden_states", {}).values():
        if isinstance(val, str):
            return decode_b64(val).reshape(-1, hidden_dim)
        elif isinstance(val, list):
            m = np.array(val, dtype=np.float32)
            return m if m.ndim == 2 else m.reshape(1, -1)
    return None

def extract_mean_vector(enc: dict, hidden_dim: int) -> np.ndarray:
    mat = extract_vector(enc, hidden_dim)
    return mat.mean(axis=0) if mat is not None else None

def pool(mat: np.ndarray, method: str) -> np.ndarray:
    if method == "last_token":    return mat[-1]
    if method == "2nd_to_last":   return mat[-2] if len(mat) >= 2 else mat[-1]
    if method == "mean_no_edges": return mat[1:-1].mean(0) if len(mat) > 2 else mat.mean(0)
    if method == "max_pool":      return mat.max(0)
    return mat.mean(0)

def cosine(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 1e-10 and n2 > 1e-10 else 0.0


# ═══════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════

def load_stsb_pairs(n_pairs=200, seed=42):
    from datasets import load_dataset
    np.random.seed(seed)
    ds = load_dataset("sentence-transformers/stsb", split="test")
    bins = [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]
    per_bin = n_pairs // len(bins)
    pairs = []
    for lo, hi in bins:
        items = [r for r in ds if lo <= r["score"] < hi]
        np.random.shuffle(items)
        pairs.extend(items[:per_bin])
    if len(pairs) < n_pairs:
        all_items = list(ds); np.random.shuffle(all_items)
        pairs.extend(all_items[:n_pairs - len(pairs)])
    return pairs[:n_pairs]


def load_gsm8k(n=60, seed=42):
    from datasets import load_dataset
    np.random.seed(seed)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds))); np.random.shuffle(idx)
    rows = []
    for i in idx[:n]:
        r = ds[i]
        # Ground truth: "#### 42" at end of answer
        m = re.search(r"####\s*([\d,]+)", r["answer"])
        if m:
            gt = int(m.group(1).replace(",", ""))
            rows.append({"question": r["question"], "answer": r["answer"], "gt": gt})
    log.info(f"  {len(rows)} GSM8K questions with parseable ground truth")
    return rows


def load_mmlu_pro(n=60, seed=42):
    from datasets import load_dataset
    np.random.seed(seed)
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        items = [r for r in ds if r.get("answer") and r.get("question")]
        np.random.shuffle(items)
        return items[:n]
    except Exception as e:
        log.warning(f"MMLU-Pro unavailable: {e}"); return []


def load_seed_texts(n=5, seed=42):
    from datasets import load_dataset
    np.random.seed(seed)
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idx = list(range(len(ds))); np.random.shuffle(idx)
        return [ds[i]["question"] for i in idx[:n]]
    except Exception:
        return ["What are the fundamental tradeoffs in distributed systems?",
                "How does natural selection lead to complex adaptations?",
                "What is the relationship between entropy and information?"][:n]


# ═══════════════════════════════════════════════════════════════════
#  Answer extraction helpers
# ═══════════════════════════════════════════════════════════════════

def extract_integer(text: str) -> int | None:
    """Extract the last integer from a string (for GSM8K answers)."""
    # Strip commas from numbers like "1,234"
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    try:
        return int(float(numbers[-1]))
    except (ValueError, OverflowError):
        return None


def extract_letter(text: str) -> str:
    """Extract answer letter (A-E) from MMLU-Pro response."""
    # Explicit patterns first
    for pat in [
        r"(?:the answer is|answer is|answer:)\s*\(?([A-E])\)?",
        r"^([A-E])[\.:\)]\s",
        r"\(([A-E])\)",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # Fallback: last standalone letter
    letters = re.findall(r"\b([A-E])\b", text.upper())
    return letters[-1] if letters else "?"


# ═══════════════════════════════════════════════════════════════════
#  M4 — unchanged from eval_phase1.py
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
                f"  Pearson  r = {self.pearson_r:.4f}")

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
            h.append(p["score"]); z.append(cosine(vecs[p["sentence1"]], vecs[p["sentence2"]]))
    h, z = np.array(h), np.array(z)
    sp, pr = stats.spearmanr(h, z), stats.pearsonr(h, z)
    return M4Result(float(sp.statistic), float(sp.pvalue), float(pr.statistic),
                    float(pr.pvalue), len(h), float(z.mean()), float(z.std()),
                    float(sp.statistic) > cfg["m4_threshold"], layer, pm, cent)


# ═══════════════════════════════════════════════════════════════════
#  M5_v1 — GSM8K zombie accuracy curve  (no API key needed)
# ═══════════════════════════════════════════════════════════════════
#
#  What it measures:
#    At each K, ask the zombie to solve a GSM8K math problem using
#    bridge(mode="ruminate", n_steps=K, decode_after=True).
#    Parse the integer from decoded_text, compare to ground truth.
#    Report accuracy at each K and the net gain (acc[K_max] - acc[0]).
#
#  Baseline (K=0): bridge(mode="passive") — encode+decode with no rollout.
#  This gives a "cold" zombie answer.
#
#  Signal: if the zombie benefits from latent rumination, accuracy should
#  increase with K.  If displacement was just noise, accuracy stays flat.

@dataclass
class M5v1Result:
    k_values: list
    accuracies: list          # fraction correct at each K
    n_correct: list           # raw count correct at each K
    n_questions: int
    gain: float               # acc[K_max] - acc[K_0]
    gain_detected: bool       # gain > 0.05
    examples_wrong_at_0_right_at_max: list   # cases where rumination helped
    examples_right_at_0_wrong_at_max: list   # cases where rumination hurt

    def summary(self):
        s = "✅ Gain" if self.gain_detected else "❌ No gain"
        bar = lambda a: "█" * max(1, int(a * 20))
        lines = [f"M5_v1 GSM8K Accuracy Curve ({self.n_questions} q) {s}  gain={self.gain:+.3f}"]
        for k, acc, nc in zip(self.k_values, self.accuracies, self.n_correct):
            lines.append(f"  K={k:3d}: {acc:.3f} ({nc}/{self.n_questions})  {bar(acc)}")
        if self.examples_wrong_at_0_right_at_max:
            lines.append(f"  Helped by rumination: {len(self.examples_wrong_at_0_right_at_max)} cases")
        if self.examples_right_at_0_wrong_at_max:
            lines.append(f"  Hurt by rumination:   {len(self.examples_right_at_0_wrong_at_max)} cases")
        return "\n".join(lines)


def eval_m5_v1(client, n_questions=50, k_values=None):
    if k_values is None:
        k_values = [0, 5, 15, 30, 60]

    questions = load_gsm8k(n_questions)
    if not questions:
        log.error("No GSM8K questions loaded")
        return M5v1Result(k_values, [0]*len(k_values), [0]*len(k_values), 0, 0.0, False, [], [])

    # answers[i][k_idx] = (predicted_int | None)
    predictions = [[None] * len(k_values) for _ in range(len(questions))]

    for k_idx, k in enumerate(k_values):
        mode = "passive" if k == 0 else "ruminate"
        desc = f"M5_v1 K={k}"
        for q_idx, q in enumerate(tqdm(questions, desc=desc)):
            try:
                prompt = (
                    f"Solve this math problem step by step. "
                    f"End your answer with '#### <number>'.\n\n{q['question']}"
                )
                res = client.bridge(prompt, mode=mode, n_steps=k)
                decoded = res.get("decoded_text") or res.get("enriched_text") or ""
                predictions[q_idx][k_idx] = extract_integer(decoded)
            except Exception as e:
                log.debug(f"  q={q_idx} k={k}: {e}")

    # Compute accuracy per K
    accuracies, n_correct = [], []
    for k_idx in range(len(k_values)):
        correct = sum(
            1 for q_idx, q in enumerate(questions)
            if predictions[q_idx][k_idx] is not None
            and predictions[q_idx][k_idx] == q["gt"]
        )
        n_correct.append(correct)
        accuracies.append(correct / len(questions))

    gain = accuracies[-1] - accuracies[0]

    # Find interesting examples
    k0_idx = 0
    kmax_idx = len(k_values) - 1
    helped, hurt = [], []
    for q_idx, q in enumerate(questions):
        p0 = predictions[q_idx][k0_idx]
        pm = predictions[q_idx][kmax_idx]
        gt = q["gt"]
        if p0 != gt and pm == gt:
            helped.append({"question": q["question"][:120], "gt": gt,
                           "pred_k0": p0, "pred_kmax": pm})
        elif p0 == gt and pm != gt:
            hurt.append({"question": q["question"][:120], "gt": gt,
                         "pred_k0": p0, "pred_kmax": pm})

    return M5v1Result(
        k_values=k_values,
        accuracies=accuracies,
        n_correct=n_correct,
        n_questions=len(questions),
        gain=gain,
        gain_detected=gain > 0.05,
        examples_wrong_at_0_right_at_max=helped[:5],
        examples_right_at_0_wrong_at_max=hurt[:5],
    )


# ═══════════════════════════════════════════════════════════════════
#  ABC_v1 — Ground-truth accuracy  (needs ANTHROPIC_API_KEY)
# ═══════════════════════════════════════════════════════════════════
#
#  What it measures:
#    Three conditions:
#      A: Claude answers directly (no zombie)
#      B: Claude answers after seeing passive zombie encoding (K=0)
#      C: Claude answers after seeing zombie rumination (K=60)
#
#    Score = fraction of correct answers vs ground truth.
#    • GSM8K: extract last integer, compare to "#### N"
#    • MMLU-Pro: extract answer letter (A-E), compare to ground truth
#
#  No LLM judge.  Ground truth is authoritative.
#
#  Limitation still present: Claude is the answerer in all conditions.
#  What this removes: the self-judging bias.  What it can't remove: the
#  fact that the enrichment text is still Claude-readable prose, not raw
#  latent signal.

@dataclass
class ABCv1Result:
    n_gsm: int; n_mmlu: int
    # GSM8K accuracy per condition
    gsm_acc_a: float; gsm_acc_b: float; gsm_acc_c: float
    # MMLU accuracy per condition
    mmlu_acc_a: float; mmlu_acc_b: float; mmlu_acc_c: float
    # Combined
    combined_acc_a: float; combined_acc_b: float; combined_acc_c: float
    # Per-question detail (for JSON)
    gsm_detail: list; mmlu_detail: list

    def summary(self):
        def delta(c, a): return f"{c-a:+.3f}"
        lines = [
            f"ABC_v1 Ground-Truth Accuracy  (GSM8K n={self.n_gsm}, MMLU n={self.n_mmlu})",
            f"  {'':20s}   A       B       C     C-A",
            f"  {'GSM8K':20s} {self.gsm_acc_a:.3f}   {self.gsm_acc_b:.3f}   {self.gsm_acc_c:.3f}   {delta(self.gsm_acc_c, self.gsm_acc_a)}",
            f"  {'MMLU-Pro':20s} {self.mmlu_acc_a:.3f}   {self.mmlu_acc_b:.3f}   {self.mmlu_acc_c:.3f}   {delta(self.mmlu_acc_c, self.mmlu_acc_a)}",
            f"  {'Combined':20s} {self.combined_acc_a:.3f}   {self.combined_acc_b:.3f}   {self.combined_acc_c:.3f}   {delta(self.combined_acc_c, self.combined_acc_a)}",
        ]
        if self.combined_acc_c > self.combined_acc_a + 0.05:
            lines.append("  ✅ C outperforms A by >5pp — zombie enrichment helps")
        elif self.combined_acc_c < self.combined_acc_a - 0.05:
            lines.append("  ❌ C underperforms A by >5pp — zombie enrichment hurts")
        else:
            lines.append("  → No significant accuracy difference across conditions")
        return "\n".join(lines)


def eval_abc_v1(client, n_gsm=40, n_mmlu=40, n_steps=60):
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        log.error("ANTHROPIC_API_KEY not set")
        return ABCv1Result(0,0,0,0,0,0,0,0,0,0,0,[],[])

    cl = anthropic.Anthropic(api_key=key)

    def ask(msgs, mt=512) -> str:
        return cl.messages.create(model="claude-sonnet-4-6", max_tokens=mt,
                                  messages=msgs, timeout=60).content[0].text

    gsm_qs  = load_gsm8k(n_gsm)
    mmlu_qs = load_mmlu_pro(n_mmlu)

    gsm_detail, mmlu_detail = [], []

    # ── GSM8K ──
    for q in tqdm(gsm_qs, desc="ABC_v1 GSM8K"):
        row = {"question": q["question"][:120], "gt": q["gt"]}
        prompt_base = (
            f"Solve this math problem step by step. "
            f"End with '#### <number>'.\n\n{q['question']}"
        )
        try:
            # A: direct
            ta = ask([{"role": "user", "content": prompt_base}])
            row["pred_a"] = extract_integer(ta)
            row["correct_a"] = (row["pred_a"] == q["gt"])

            # B: passive zombie
            br_b = client.bridge(q["question"], mode="passive", n_steps=0)
            enr_b = br_b.get("decoded_text") or br_b.get("enriched_text") or ""
            tb = ask([{"role": "user", "content": prompt_base},
                      {"role": "assistant", "content": ta},
                      {"role": "user",
                       "content": f"A parallel process suggests:\n{enr_b[:400]}\n\nRevise if needed. End with '#### <number>'."}])
            row["pred_b"] = extract_integer(tb)
            row["correct_b"] = (row["pred_b"] == q["gt"])

            # C: ruminate zombie
            br_c = client.bridge(q["question"], mode="ruminate", n_steps=n_steps)
            enr_c = br_c.get("enriched_text") or br_c.get("decoded_text") or ""
            tc = ask([{"role": "user", "content": prompt_base},
                      {"role": "assistant", "content": ta},
                      {"role": "user",
                       "content": f"A parallel process suggests:\n{enr_c[:400]}\n\nRevise if needed. End with '#### <number>'."}])
            row["pred_c"] = extract_integer(tc)
            row["correct_c"] = (row["pred_c"] == q["gt"])

            gsm_detail.append(row)
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"  GSM8K: {e}")

    # ── MMLU-Pro ──
    for q in tqdm(mmlu_qs, desc="ABC_v1 MMLU"):
        row = {"question": q["question"][:120], "gt": q["answer"]}
        options = q.get("options", [])
        opts_str = "\n".join(f"  {chr(65+i)}) {opt}" for i, opt in enumerate(options))
        prompt_base = f"{q['question']}\n{opts_str}\n\nAnswer with a single letter."
        try:
            ta = ask([{"role": "user", "content": prompt_base}], mt=10)
            row["pred_a"] = extract_letter(ta)
            row["correct_a"] = (row["pred_a"] == str(q["answer"]).upper())

            br_b = client.bridge(q["question"], mode="passive", n_steps=0)
            enr_b = br_b.get("decoded_text") or br_b.get("enriched_text") or ""
            tb = ask([{"role": "user", "content": prompt_base},
                      {"role": "assistant", "content": ta},
                      {"role": "user",
                       "content": f"A parallel analysis:\n{enr_b[:400]}\n\nReconsider. Answer with a single letter."}], mt=10)
            row["pred_b"] = extract_letter(tb)
            row["correct_b"] = (row["pred_b"] == str(q["answer"]).upper())

            br_c = client.bridge(q["question"], mode="ruminate", n_steps=n_steps)
            enr_c = br_c.get("enriched_text") or br_c.get("decoded_text") or ""
            tc = ask([{"role": "user", "content": prompt_base},
                      {"role": "assistant", "content": ta},
                      {"role": "user",
                       "content": f"A parallel analysis:\n{enr_c[:400]}\n\nReconsider. Answer with a single letter."}], mt=10)
            row["pred_c"] = extract_letter(tc)
            row["correct_c"] = (row["pred_c"] == str(q["answer"]).upper())

            mmlu_detail.append(row)
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"  MMLU: {e}")

    def acc(detail, key):
        if not detail: return 0.0
        return float(np.mean([r[key] for r in detail]))

    gsm_a, gsm_b, gsm_c    = acc(gsm_detail, "correct_a"), acc(gsm_detail, "correct_b"), acc(gsm_detail, "correct_c")
    mmlu_a, mmlu_b, mmlu_c = acc(mmlu_detail, "correct_a"), acc(mmlu_detail, "correct_b"), acc(mmlu_detail, "correct_c")
    n_total = len(gsm_detail) + len(mmlu_detail)

    def combined(ga, ma):
        if n_total == 0: return 0.0
        return (ga * len(gsm_detail) + ma * len(mmlu_detail)) / n_total

    return ABCv1Result(
        n_gsm=len(gsm_detail), n_mmlu=len(mmlu_detail),
        gsm_acc_a=gsm_a, gsm_acc_b=gsm_b, gsm_acc_c=gsm_c,
        mmlu_acc_a=mmlu_a, mmlu_acc_b=mmlu_b, mmlu_acc_c=mmlu_c,
        combined_acc_a=combined(gsm_a, mmlu_a),
        combined_acc_b=combined(gsm_b, mmlu_b),
        combined_acc_c=combined(gsm_c, mmlu_c),
        gsm_detail=gsm_detail, mmlu_detail=mmlu_detail,
    )


# ═══════════════════════════════════════════════════════════════════
#  Loop_v1 — relevance-aware exploration  (no API key needed)
# ═══════════════════════════════════════════════════════════════════
#
#  Adds two metrics to the existing loop:
#
#  seed_relevance  cosine(step_vec, seed_vec)
#    Measures whether the loop stays on topic.  Should stay > 0.15.
#    Below that, the loop has semantically left the question behind.
#
#  meaningful_exploration  bool per step
#    True when novelty > 0.05 AND seed_relevance > 0.2.
#    A step that is both new AND related to the original question.
#
#  Loop is flagged as:
#    • "meaningful": ≥50% of steps are meaningful_exploration
#    • "drifting":   final seed_relevance < 0.15
#    • "stagnating": final novelty < 0.05

@dataclass
class Loopv1Result:
    seed_question: str
    n_iterations: int
    k_per_step: int
    iterations: list          # per-step: displacement, novelty, seed_relevance, meaningful
    total_drift: float
    convergence_step: int
    # v1 additions
    final_seed_relevance: float
    mean_seed_relevance: float
    meaningful_steps: int     # steps where novelty>0.05 AND relevance>0.2
    loop_quality: str         # "meaningful" | "drifting" | "stagnating" | "weak"
    texts: list

    def summary(self):
        q_icon = {"meaningful": "✅", "drifting": "🌊", "stagnating": "📍", "weak": "❌"}.get(self.loop_quality, "?")
        cv = f"conv@{self.convergence_step}" if self.convergence_step >= 0 else "exploring"
        lines = [
            f"  {q_icon} [{self.loop_quality}] drift={self.total_drift:.4f} "
            f"rel={self.mean_seed_relevance:.3f} meaningful={self.meaningful_steps}/{self.n_iterations} {cv}",
            f"    \"{self.seed_question[:70]}...\"",
        ]
        for it in self.iterations:
            m = "✓" if it["meaningful"] else "·"
            lines.append(
                f"    step {it['step']}: nov={it['novelty']:.3f} "
                f"rel={it['seed_relevance']:.3f} {m}"
            )
        return "\n".join(lines)


def eval_loop_v1(client, n_seeds=5, n_iter=8, k=30, layer=9, hidden_dim=2560):
    seeds = load_seed_texts(n_seeds)
    results = []

    for i, seed in enumerate(seeds):
        log.info(f"Loop_v1 {i+1}/{n_seeds}")
        vecs, texts, iters, cur = [], [], [], seed

        # Encode the seed once to use as relevance reference
        try:
            seed_enc = client.encode(seed[:500], layer=layer)
            seed_vec = extract_mean_vector(seed_enc, hidden_dim)
        except Exception:
            seed_vec = None

        for step in range(n_iter):
            try:
                res = client.bridge(cur, mode="ruminate", n_steps=k)
                out = res.get("enriched_text", res.get("decoded_text", ""))
                if not out or len(out.strip()) < 10:
                    break
                v = extract_mean_vector(client.encode(out[:500], layer=layer), hidden_dim)
                if v is None:
                    break

                disp = 0 if not vecs else 1 - cosine(v, vecs[-1])
                nov  = 1.0 if not vecs else 1 - max(cosine(v, p) for p in vecs)
                rel  = cosine(v, seed_vec) if seed_vec is not None else 0.0
                meaningful = nov > 0.05 and rel > 0.2

                vecs.append(v); texts.append(out)
                iters.append({
                    "step": step,
                    "displacement": disp,
                    "novelty": nov,
                    "seed_relevance": rel,
                    "meaningful": meaningful,
                })
                cur = out
            except Exception as e:
                log.warning(f"  step {step}: {e}"); break

        if len(vecs) < 2:
            results.append(Loopv1Result(seed[:100], 0, k, [], 0, -1, 0, 0, 0, "weak", []))
            continue

        total_drift   = 1 - cosine(vecs[0], vecs[-1])
        conv          = next((it["step"] for it in iters[1:] if it["novelty"] < 0.05), -1)
        final_rel     = iters[-1]["seed_relevance"]
        mean_rel      = float(np.mean([it["seed_relevance"] for it in iters]))
        meaningful_n  = sum(1 for it in iters if it["meaningful"])

        if final_rel < 0.15:
            quality = "drifting"
        elif conv >= 0:
            quality = "stagnating"
        elif meaningful_n >= len(iters) // 2:
            quality = "meaningful"
        else:
            quality = "weak"

        results.append(Loopv1Result(
            seed_question=seed[:100],
            n_iterations=len(iters),
            k_per_step=k,
            iterations=iters,
            total_drift=total_drift,
            convergence_step=conv,
            final_seed_relevance=final_rel,
            mean_seed_relevance=mean_rel,
            meaningful_steps=meaningful_n,
            loop_quality=quality,
            texts=texts,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="ERIS v5 — Phase 1 Evaluation v1")
    p.add_argument("--eris-url",     default="http://localhost:8001")
    p.add_argument("--metric", nargs="+", default=["all"],
                   choices=["all"] + sorted(ALL_METRICS_V1))
    p.add_argument("--n-pairs",      type=int, default=200, help="STS-B pairs (M4)")
    p.add_argument("--n-gsm",        type=int, default=40,  help="GSM8K questions (M5_v1, ABC_v1)")
    p.add_argument("--n-mmlu",       type=int, default=40,  help="MMLU-Pro questions (ABC_v1)")
    p.add_argument("--n-seeds",      type=int, default=5,   help="Seeds for loop")
    p.add_argument("--n-iterations", type=int, default=8,   help="Loop iterations")
    p.add_argument("--n-steps",      type=int, default=60,  help="Rumination steps (ABC_v1 condition C)")
    p.add_argument("--k-per-step",   type=int, default=30,  help="K for loop")
    p.add_argument("--tag",          default="v1")
    p.add_argument("--output-dir",   default="eval_results")
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    metrics = ALL_METRICS_V1 if "all" in args.metric else set(args.metric)
    Path(args.output_dir).mkdir(exist_ok=True)

    client = ERISClient(args.eris_url)
    health = client.health()
    model  = health.get("model", "unknown")
    cfg    = get_model_config(model)
    layer  = cfg["m4_layer"]

    test_enc   = client.encode("test", layer=layer)
    hidden_dim = test_enc.get("hidden_dim", 2560)
    log.info(f"Model: {model}, layer: {layer}, hidden_dim: {hidden_dim}")

    R = {"tag": args.tag, "model": model, "config": cfg,
         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
         "improvements": {
             "m5": "GSM8K accuracy curve (was: displacement)",
             "abc": "ground-truth accuracy (was: self-judge)",
             "loop": "relevance-aware exploration (was: geometric novelty only)",
         }}

    try:
        if "m4" in metrics:
            log.info("═"*60 + "\n  M4 — Projection Fidelity (unchanged)\n" + "═"*60)
            r = eval_m4(client, cfg, args.n_pairs)
            print(f"\n{r.summary()}\n"); R["m4"] = asdict(r)

        if "m5" in metrics:
            log.info("═"*60 + "\n  M5_v1 — GSM8K Accuracy Curve\n" + "═"*60)
            r = eval_m5_v1(client, args.n_gsm)
            print(f"\n{r.summary()}\n"); R["m5"] = asdict(r)

        if "abc" in metrics:
            log.info("═"*60 + "\n  ABC_v1 — Ground-Truth Accuracy\n" + "═"*60)
            r = eval_abc_v1(client, args.n_gsm, args.n_mmlu, args.n_steps)
            print(f"\n{r.summary()}\n"); R["abc"] = asdict(r)

        if "loop" in metrics:
            log.info("═"*60 + "\n  Loop_v1 — Relevance-Aware Exploration\n" + "═"*60)
            rs = eval_loop_v1(client, args.n_seeds, args.n_iterations,
                              args.k_per_step, layer, hidden_dim)
            R["loop"] = [asdict(r) for r in rs]
            for r in rs: print(r.summary())

        # Metrics not modified in v1 — remind user to run eval_phase1.py for them
        unchanged = metrics - {"m4", "m5", "abc", "loop"}
        if unchanged:
            log.info(f"Metrics {sorted(unchanged)} are unchanged — run eval_phase1.py for those.")

    finally:
        client.close()

    out = Path(args.output_dir) / f"phase1_v1_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    log.info(f"Saved to {out}")

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  PHASE 1 v1 — {model}")
    print(f"{'═'*60}")
    if "m4" in R:
        print(f"  M4 Projection:  ρ={R['m4']['spearman_r']:.4f} {'✅' if R['m4']['pass_threshold'] else '❌'}")
    if "m5" in R:
        r = R["m5"]
        k0, km = r["accuracies"][0], r["accuracies"][-1]
        print(f"  M5_v1 GSM8K:    K=0→{k0:.3f}  K={r['k_values'][-1]}→{km:.3f}  "
              f"gain={r['gain']:+.3f} {'✅' if r['gain_detected'] else '❌'}")
    if "abc" in R:
        r = R["abc"]
        print(f"  ABC_v1:         A={r['combined_acc_a']:.3f}  B={r['combined_acc_b']:.3f}  "
              f"C={r['combined_acc_c']:.3f}  C-A={r['combined_acc_c']-r['combined_acc_a']:+.3f}")
    if "loop" in R:
        rs = R["loop"]
        q_counts = {}
        for r in rs:
            q_counts[r["loop_quality"]] = q_counts.get(r["loop_quality"], 0) + 1
        summary_str = "  ".join(f"{q}:{n}" for q, n in sorted(q_counts.items()))
        print(f"  Loop_v1:        {summary_str}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
