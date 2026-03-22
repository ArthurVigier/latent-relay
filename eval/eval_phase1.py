#!/usr/bin/env python3
"""
ERIS v5 — Phase 1 Evaluation: Does the latent channel add information?
=======================================================================

Uses HuggingFace datasets (no hardcoded data):
  - M4 (projection fidelity):  sentence-transformers/stsb  (1500+ annotated sentence pairs)
  - M1-M3 (A/B/C comparison):  openai/gsm8k + TIGER-Lab/MMLU-Pro (technical questions)
  - M5 (LatentMAS gain):       same questions, varying K
  - M6 (implicit features):    SAE analysis on encoded hidden states

Requirements:
  pip install datasets anthropic httpx sentence-transformers scipy numpy tqdm

Usage:
  # 1. Start the ERIS server
  python eris_server.py --model Qwen/Qwen3-14B --port 8001

  # 2. Run evaluation (M4 only — no Claude API needed)
  python eval/eval_phase1.py --eris-url http://localhost:8001 --metric m4

  # 3. Run full evaluation (needs ANTHROPIC_API_KEY)
  ANTHROPIC_API_KEY=sk-... python eval/eval_phase1.py --eris-url http://localhost:8001 --metric all

  # 4. Run with Qwen3.5
  python eval/eval_phase1.py --eris-url http://localhost:8001 --metric m4 --tag qwen35-4b
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy import stats
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_phase1")


# ═══════════════════════════════════════════════════════════════
# Data loading — all from HuggingFace, no hardcoded samples
# ═══════════════════════════════════════════════════════════════

def load_stsb_pairs(n_pairs: int = 200, split: str = "test"):
    """Load sentence pairs with similarity scores from STS Benchmark.

    Dataset: sentence-transformers/stsb
    Each row: {sentence1, sentence2, score} where score ∈ [0, 1].
    """
    from datasets import load_dataset

    log.info(f"Loading STS-B ({split}, {n_pairs} pairs)...")
    ds = load_dataset("sentence-transformers/stsb", split=split)

    # Stratified sample: equal representation across similarity bins
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    per_bin = n_pairs // len(bins)
    pairs = []

    for lo, hi in bins:
        bin_items = [r for r in ds if lo <= r["score"] < hi]
        np.random.shuffle(bin_items)
        pairs.extend(bin_items[:per_bin])

    # Fill remaining with random samples
    remaining = n_pairs - len(pairs)
    if remaining > 0:
        all_indices = list(range(len(ds)))
        np.random.shuffle(all_indices)
        for idx in all_indices:
            if len(pairs) >= n_pairs:
                break
            pairs.append(ds[idx])

    log.info(f"  Loaded {len(pairs)} pairs across {len(bins)} similarity bins")
    return pairs


def load_technical_questions(n_questions: int = 50):
    """Load technical questions from MMLU-Pro (STEM/CS subset) + GSM8K.

    Datasets:
      - TIGER-Lab/MMLU-Pro: multi-choice STEM questions (we use computer_science, engineering)
      - openai/gsm8k: grade school math (reasoning-heavy, good for latent rollout testing)
    """
    from datasets import load_dataset

    questions = []

    # MMLU-Pro CS/Engineering questions
    log.info("Loading MMLU-Pro (CS + Engineering)...")
    try:
        mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        cs_qs = [r for r in mmlu if r.get("category") in
                 ("computer_science", "engineering", "math", "physics")]
        np.random.shuffle(cs_qs)

        for r in cs_qs[:n_questions // 2]:
            # Format as open question (strip multiple choice)
            q = r["question"]
            questions.append({
                "id": f"mmlu_{len(questions)}",
                "text": q,
                "source": "MMLU-Pro",
                "category": r.get("category", "unknown"),
                "answer": r.get("answer", None),
            })
    except Exception as e:
        log.warning(f"  MMLU-Pro load failed: {e}. Falling back to GSM8K only.")

    # GSM8K questions
    remaining = n_questions - len(questions)
    if remaining > 0:
        log.info(f"Loading GSM8K ({remaining} questions)...")
        try:
            gsm = load_dataset("openai/gsm8k", "main", split="test")
            indices = list(range(len(gsm)))
            np.random.shuffle(indices)

            for idx in indices[:remaining]:
                r = gsm[idx]
                questions.append({
                    "id": f"gsm_{len(questions)}",
                    "text": r["question"],
                    "source": "GSM8K",
                    "category": "math_reasoning",
                    "answer": r.get("answer", None),
                })
        except Exception as e:
            log.warning(f"  GSM8K load failed: {e}")

    log.info(f"  Loaded {len(questions)} technical questions")
    return questions


# ═══════════════════════════════════════════════════════════════
# ERIS client helpers
# ═══════════════════════════════════════════════════════════════

class ERISEvalClient:
    """Lightweight client for evaluation — uses httpx directly."""

    def __init__(self, base_url: str, timeout: float = 120.0):
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

    def encode(self, text: str, return_layers: list = None):
        """Encode text and get hidden states."""
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

    def latent_think(self, text: str, n_steps: int = 60,
                     return_trajectory: bool = False):
        """Run latent rollout."""
        sid = self._ensure_session()
        payload = {
            "session_id": sid,
            "prompt": text,
            "n_steps": n_steps,
            "return_trajectory": return_trajectory,
            "trajectory_analyses": [],
        }
        r = self.client.post(f"{self.base_url}/v1/latent_think", json=payload)
        r.raise_for_status()
        return r.json()

    def bridge(self, text: str, mode: str = "ruminate", n_steps: int = 60,
               analyses: list = None):
        """Full bridge pipeline."""
        payload = {
            "claude_text": text,
            "mode": mode,
            "n_steps": n_steps,
            "analyses": analyses or [],
            "decode_after": True,
            "max_new_tokens": 512,
        }
        r = self.client.post(f"{self.base_url}/v1/bridge", json=payload)
        r.raise_for_status()
        return r.json()

    def analyze(self, handle: str, analyses: list):
        """Run MI analyses on a stored handle."""
        sid = self._ensure_session()
        payload = {"handle": handle, "session_id": sid, "analyses": analyses}
        r = self.client.post(f"{self.base_url}/v1/analyze", json=payload)
        r.raise_for_status()
        return r.json()

    def close(self):
        if self._session_id:
            try:
                self.client.delete(
                    f"{self.base_url}/sessions/{self._session_id}")
            except Exception:
                pass
        self.client.close()


def decode_b64_hidden(b64_str: str) -> np.ndarray:
    """Decode base64 float32 hidden states."""
    import base64
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype=np.float32)


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
    pass_threshold: bool  # Spearman > 0.6

    def summary(self) -> str:
        status = "✅ PASS" if self.pass_threshold else "❌ FAIL"
        return (
            f"M4 Projection Fidelity ({self.n_pairs} pairs)\n"
            f"  Spearman ρ = {self.spearman_r:.4f} (p={self.spearman_p:.2e}) {status}\n"
            f"  Pearson  r = {self.pearson_r:.4f} (p={self.pearson_p:.2e})\n"
            f"  Mean cosine similarity = {self.mean_cosine:.4f}\n"
            f"  Threshold: Spearman > 0.6"
        )


def eval_m4(client: ERISEvalClient, n_pairs: int = 200) -> M4Result:
    """M4: Does the zombie's latent space preserve semantic similarity?

    For each STS-B pair (sentence1, sentence2, human_score):
      1. Encode both sentences via zombie
      2. Compute cosine similarity in zombie's hidden space
      3. Correlate zombie cosine similarity with human similarity score

    If Spearman > 0.6, the projection is faithful enough for the channel.
    """
    pairs = load_stsb_pairs(n_pairs)

    human_scores = []
    zombie_cosines = []
    errors = 0

    for pair in tqdm(pairs, desc="M4: encoding pairs"):
        try:
            enc1 = client.encode(pair["sentence1"], return_layers=[-1])
            enc2 = client.encode(pair["sentence2"], return_layers=[-1])

            # Get the last-layer hidden state (mean-pooled or last token)
            hs1_key = [k for k in enc1.get("hidden_states", {}).keys()
                       if k in ("last", "layer_-1")][0]
            hs2_key = [k for k in enc2.get("hidden_states", {}).keys()
                       if k in ("last", "layer_-1")][0]

            # Decode from base64 if compact mode
            hs1_raw = enc1["hidden_states"][hs1_key]
            hs2_raw = enc2["hidden_states"][hs2_key]

            if isinstance(hs1_raw, str):
                v1 = decode_b64_hidden(hs1_raw)
                v2 = decode_b64_hidden(hs2_raw)
            elif isinstance(hs1_raw, list):
                # Non-compact: list of lists. Take mean across tokens.
                v1 = np.array(hs1_raw, dtype=np.float32).mean(axis=0)
                v2 = np.array(hs2_raw, dtype=np.float32).mean(axis=0)
            else:
                log.warning(f"  Unknown hidden state format: {type(hs1_raw)}")
                errors += 1
                continue

            # Mean-pool if multi-token (v might be [seq_len * hidden_dim])
            hidden_dim = enc1.get("hidden_dim", None)
            if hidden_dim and len(v1) > hidden_dim:
                v1 = v1.reshape(-1, hidden_dim).mean(axis=0)
                v2 = v2.reshape(-1, hidden_dim).mean(axis=0)

            # Cosine similarity
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

            human_scores.append(pair["score"])
            zombie_cosines.append(cos)

        except Exception as e:
            log.warning(f"  Error on pair: {e}")
            errors += 1
            if errors > 20:
                log.error("  Too many errors, aborting M4")
                break

    if len(human_scores) < 30:
        log.error(f"  Only {len(human_scores)} valid pairs — not enough for correlation")
        return M4Result(0, 1, 0, 1, len(human_scores), 0, False)

    human_scores = np.array(human_scores)
    zombie_cosines = np.array(zombie_cosines)

    sp = stats.spearmanr(human_scores, zombie_cosines)
    pr = stats.pearsonr(human_scores, zombie_cosines)

    return M4Result(
        spearman_r=float(sp.statistic),
        spearman_p=float(sp.pvalue),
        pearson_r=float(pr.statistic),
        pearson_p=float(pr.pvalue),
        n_pairs=len(human_scores),
        mean_cosine=float(zombie_cosines.mean()),
        pass_threshold=float(sp.statistic) > 0.6,
    )


# ═══════════════════════════════════════════════════════════════
# M5 — LatentMAS Gain (Quality vs K)
# ═══════════════════════════════════════════════════════════════

@dataclass
class M5Result:
    k_values: list
    displacement_means: list
    displacement_stds: list
    n_questions: int
    gain_detected: bool  # displacement increases with K

    def summary(self) -> str:
        status = "✅ Gain detected" if self.gain_detected else "❌ No gain from rumination"
        lines = [f"M5 LatentMAS Gain ({self.n_questions} questions) {status}"]
        for k, mean, std in zip(self.k_values, self.displacement_means,
                                self.displacement_stds):
            lines.append(f"  K={k:3d}: displacement = {mean:.4f} ± {std:.4f}")
        return "\n".join(lines)


def eval_m5(client: ERISEvalClient, n_questions: int = 30,
            k_values: list = None) -> M5Result:
    """M5: Does latent rollout (K steps) increase displacement?

    For each question, run latent_think with varying K and measure
    total_displacement (cosine distance z_0 → z_K).
    If displacement increases with K, rumination explores the space.
    """
    if k_values is None:
        k_values = [0, 5, 15, 30, 60]

    questions = load_technical_questions(n_questions)

    results_by_k = {k: [] for k in k_values}
    errors = 0

    for q in tqdm(questions, desc="M5: testing K values"):
        for k in k_values:
            try:
                result = client.latent_think(
                    q["text"], n_steps=k, return_trajectory=False
                )
                disp = result.get("total_displacement",
                                  result.get("hidden_norm", 0))
                results_by_k[k].append(disp)
            except Exception as e:
                log.warning(f"  Error K={k}: {e}")
                errors += 1
                if errors > 50:
                    break

    displacement_means = []
    displacement_stds = []
    for k in k_values:
        vals = results_by_k[k]
        if vals:
            displacement_means.append(float(np.mean(vals)))
            displacement_stds.append(float(np.std(vals)))
        else:
            displacement_means.append(0.0)
            displacement_stds.append(0.0)

    # Gain detected if displacement is monotonically increasing (mostly)
    gain = all(displacement_means[i] <= displacement_means[i + 1] + 0.01
               for i in range(len(displacement_means) - 1))

    return M5Result(
        k_values=k_values,
        displacement_means=displacement_means,
        displacement_stds=displacement_stds,
        n_questions=len(questions),
        gain_detected=gain,
    )


# ═══════════════════════════════════════════════════════════════
# M1-M3 — A/B/C Comparison (requires Claude API)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ABCResult:
    n_questions: int
    condition_a_scores: list  # Claude direct
    condition_b_scores: list  # Claude → zombie paraphrase → Claude
    condition_c_scores: list  # Claude → zombie ruminate → Claude
    mean_a: float
    mean_b: float
    mean_c: float
    c_better_than_a: float  # fraction where C > A
    c_better_than_b: float  # fraction where C > B

    def summary(self) -> str:
        return (
            f"M1-M3 A/B/C Comparison ({self.n_questions} questions)\n"
            f"  (A) Claude direct:     mean={self.mean_a:.2f}\n"
            f"  (B) Claude+paraphrase: mean={self.mean_b:.2f}\n"
            f"  (C) Claude+ruminate:   mean={self.mean_c:.2f}\n"
            f"  C > A: {self.c_better_than_a:.0%} of questions\n"
            f"  C > B: {self.c_better_than_b:.0%} of questions"
        )


def eval_abc(client: ERISEvalClient, n_questions: int = 30,
             n_steps: int = 60) -> ABCResult:
    """M1-M3: Compare Claude direct vs Claude+zombie paraphrase vs Claude+ruminate.

    Requires ANTHROPIC_API_KEY.

    For each question:
      (A) Claude answers directly
      (B) Claude answers → zombie paraphrases → Claude answers again with paraphrase
      (C) Claude answers → zombie ruminates (K steps) → Claude answers with enriched text

    Scoring: Claude Sonnet as judge (separate instance), 1-5 scale on:
      - Depth of analysis
      - Presence of reframe/alternative perspective
      - Unexpected but relevant connection
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set — skipping M1-M3")
        return ABCResult(0, [], [], [], 0, 0, 0, 0, 0)

    claude = anthropic.Anthropic(api_key=api_key)
    questions = load_technical_questions(n_questions)

    scores_a, scores_b, scores_c = [], [], []

    judge_prompt = """Rate this answer on a scale of 1-5 for analytical depth and insight.
1 = superficial/generic, 5 = deep/insightful with unexpected connections.
Respond with ONLY a single integer 1-5, nothing else."""

    for q in tqdm(questions, desc="M1-M3: A/B/C comparison"):
        try:
            # (A) Claude direct
            resp_a = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content": q["text"]}],
                timeout=60.0,
            )
            text_a = resp_a.content[0].text

            # (B) Claude → zombie paraphrase → Claude
            bridge_b = client.bridge(text_a, mode="passive", n_steps=0)
            enriched_b = bridge_b.get("enriched_text", text_a)
            resp_b = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[
                    {"role": "user", "content": q["text"]},
                    {"role": "assistant", "content": text_a},
                    {"role": "user",
                     "content": f"Here's a complementary analysis:\n{enriched_b}\n\nNow give your final, improved answer."},
                ],
                timeout=60.0,
            )
            text_b = resp_b.content[0].text

            # (C) Claude → zombie ruminate → Claude
            bridge_c = client.bridge(text_a, mode="ruminate", n_steps=n_steps)
            enriched_c = bridge_c.get("enriched_text", text_a)
            resp_c = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[
                    {"role": "user", "content": q["text"]},
                    {"role": "assistant", "content": text_a},
                    {"role": "user",
                     "content": f"Here's a deep complementary analysis:\n{enriched_c}\n\nNow give your final, improved answer."},
                ],
                timeout=60.0,
            )
            text_c = resp_c.content[0].text

            # Judge all three (using separate Claude call)
            for text, scores_list in [(text_a, scores_a), (text_b, scores_b),
                                       (text_c, scores_c)]:
                judge_resp = claude.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=10,
                    messages=[
                        {"role": "user",
                         "content": f"Question: {q['text']}\n\nAnswer: {text}\n\n{judge_prompt}"},
                    ],
                    timeout=30.0,
                )
                try:
                    score = int(judge_resp.content[0].text.strip()[0])
                    score = max(1, min(5, score))
                except (ValueError, IndexError):
                    score = 3  # neutral fallback
                scores_list.append(score)

            # Rate limit
            time.sleep(1.0)

        except Exception as e:
            log.warning(f"  Error on question: {e}")
            continue

    n = min(len(scores_a), len(scores_b), len(scores_c))
    if n == 0:
        return ABCResult(0, [], [], [], 0, 0, 0, 0, 0)

    sa, sb, sc = scores_a[:n], scores_b[:n], scores_c[:n]

    return ABCResult(
        n_questions=n,
        condition_a_scores=sa,
        condition_b_scores=sb,
        condition_c_scores=sc,
        mean_a=float(np.mean(sa)),
        mean_b=float(np.mean(sb)),
        mean_c=float(np.mean(sc)),
        c_better_than_a=float(np.mean([c > a for a, c in zip(sa, sc)])),
        c_better_than_b=float(np.mean([c > b for b, c in zip(sb, sc)])),
    )


# ═══════════════════════════════════════════════════════════════
# M6 — Implicit Features
# ═══════════════════════════════════════════════════════════════

@dataclass
class M6Result:
    n_questions: int
    mean_implicit_features: float
    mean_total_features: float
    implicit_ratio: float
    examples: list  # top 5 most interesting implicit features

    def summary(self) -> str:
        return (
            f"M6 Implicit Features ({self.n_questions} questions)\n"
            f"  Mean total SAE features: {self.mean_total_features:.1f}\n"
            f"  Mean implicit features:  {self.mean_implicit_features:.1f}\n"
            f"  Implicit ratio:          {self.implicit_ratio:.1%}\n"
            f"  (Features activated in latent space but absent from surface text)"
        )


def eval_m6(client: ERISEvalClient, n_questions: int = 20) -> M6Result:
    """M6: Does the zombie detect features not present in the surface text?

    For each question, encode via zombie, run SAE analysis, and count
    features whose labels don't match any token in the input text.
    Requires SAE analyzer configured in eris_config.yaml.
    """
    questions = load_technical_questions(n_questions)

    total_features_list = []
    implicit_features_list = []
    examples = []

    for q in tqdm(questions, desc="M6: implicit features"):
        try:
            result = client.bridge(
                q["text"], mode="analyze_only", n_steps=0,
                analyses=["sae_features"]
            )

            analysis = result.get("analysis", {})
            sae = analysis.get("sae_features", None)
            implicit = analysis.get("implicit_features", None)

            if sae is None:
                log.info("  SAE analyzer not configured — skipping M6")
                return M6Result(0, 0, 0, 0, [])

            total_count = len(sae.get("top_20", sae.get("top_10", [])))
            implicit_count = len(implicit) if implicit else 0

            total_features_list.append(total_count)
            implicit_features_list.append(implicit_count)

            if implicit and len(examples) < 5:
                examples.append({
                    "question": q["text"][:100],
                    "implicit": implicit[:3],
                })

        except Exception as e:
            log.warning(f"  Error: {e}")
            continue

    n = len(total_features_list)
    if n == 0:
        return M6Result(0, 0, 0, 0, [])

    mean_total = float(np.mean(total_features_list))
    mean_implicit = float(np.mean(implicit_features_list))

    return M6Result(
        n_questions=n,
        mean_implicit_features=mean_implicit,
        mean_total_features=mean_total,
        implicit_ratio=mean_implicit / max(mean_total, 1),
        examples=examples,
    )


# ═══════════════════════════════════════════════════════════════
# Main — orchestrate all evaluations
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ERIS v5 Phase 1 Evaluation")
    parser.add_argument("--eris-url", default="http://localhost:8001",
                        help="ERIS server URL")
    parser.add_argument("--metric", default="all",
                        choices=["all", "m4", "m5", "m6", "abc"],
                        help="Which metric(s) to evaluate")
    parser.add_argument("--n-pairs", type=int, default=200,
                        help="Number of STS-B pairs for M4")
    parser.add_argument("--n-questions", type=int, default=30,
                        help="Number of questions for M5/ABC/M6")
    parser.add_argument("--n-steps", type=int, default=60,
                        help="Latent rollout steps")
    parser.add_argument("--tag", default="default",
                        help="Tag for this run (e.g. 'qwen35-4b')")
    parser.add_argument("--output-dir", default="eval_results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    client = ERISEvalClient(args.eris_url)
    results = {"tag": args.tag, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        # ── M4: Projection Fidelity ──
        if args.metric in ("all", "m4"):
            log.info("=" * 60)
            log.info("M4 — Projection Fidelity (STS-B)")
            log.info("=" * 60)
            m4 = eval_m4(client, n_pairs=args.n_pairs)
            print(f"\n{m4.summary()}\n")
            results["m4"] = asdict(m4)

        # ── M5: LatentMAS Gain ──
        if args.metric in ("all", "m5"):
            log.info("=" * 60)
            log.info("M5 — LatentMAS Gain (displacement vs K)")
            log.info("=" * 60)
            m5 = eval_m5(client, n_questions=args.n_questions)
            print(f"\n{m5.summary()}\n")
            results["m5"] = asdict(m5)

        # ── M6: Implicit Features ──
        if args.metric in ("all", "m6"):
            log.info("=" * 60)
            log.info("M6 — Implicit Features (SAE)")
            log.info("=" * 60)
            m6 = eval_m6(client, n_questions=min(20, args.n_questions))
            print(f"\n{m6.summary()}\n")
            results["m6"] = asdict(m6)

        # ── M1-M3: A/B/C Comparison ──
        if args.metric in ("all", "abc"):
            log.info("=" * 60)
            log.info("M1-M3 — A/B/C Comparison (requires ANTHROPIC_API_KEY)")
            log.info("=" * 60)
            abc = eval_abc(client, n_questions=args.n_questions,
                           n_steps=args.n_steps)
            if abc.n_questions > 0:
                print(f"\n{abc.summary()}\n")
            results["abc"] = asdict(abc)

    finally:
        client.close()

    # ── Save results ──
    out_file = output_dir / f"phase1_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved to {out_file}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    if "m4" in results:
        m4r = results["m4"]
        status = "✅" if m4r["pass_threshold"] else "❌"
        print(f"  M4 Projection Fidelity: Spearman={m4r['spearman_r']:.4f} {status}")
    if "m5" in results:
        m5r = results["m5"]
        status = "✅" if m5r["gain_detected"] else "❌"
        print(f"  M5 LatentMAS Gain:      {status}")
    if "m6" in results:
        m6r = results["m6"]
        print(f"  M6 Implicit Features:   ratio={m6r['implicit_ratio']:.1%}")
    if "abc" in results and results["abc"]["n_questions"] > 0:
        ar = results["abc"]
        print(f"  M1-M3 A/B/C:           C>A={ar['c_better_than_a']:.0%}, "
              f"C>B={ar['c_better_than_b']:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
