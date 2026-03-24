"""
ERIS v5 — Test 0 : Drift Characterization  (KILL GATE)
========================================================

Objective: verify that latent drift during reasoning *predicts* final errors.
Without this correlation, the entire drift-detection direction is invalid.

Protocol
--------
1.  Load 20 AIME problems from the AIME 2024 dataset.
2.  For each problem, run Qwen3-14B (or configured model) with chain-of-thought.
    Extract hidden states at layer 18 every 3 reasoning steps.
3.  Compute drift_score at each checkpoint using DriftDetector.
4.  At the end of the reasoning chain, extract the final answer and check
    against ground truth.
5.  Compute Spearman ρ between final drift_score and binary error label.

Kill criterion (from kill_criteria.py):
    ρ < 0.35 → STOP, direction invalid
    ρ ≥ 0.35 → proceed to test_1

No API calls, no web access, no Claude.  This is a $0 characterization run.

Usage
-----
    # Start ERIS server first:
    python eris_server.py --model Qwen/Qwen3-14B --port 8001

    # Run test:
    python eris/experiments/drift_detection/test_0_drift_characterization.py \\
        --eris-url http://localhost:8001 \\
        --layer 18 \\
        --n-problems 20 \\
        --output results/test0_drift_characterization.json

Requirements
------------
    pip install scipy datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from eris.drift_detector import DriftDetector
from eris.experiments.drift_detection.kill_criteria import check_criterion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("test_0")


# ── ERIS client (minimal) ─────────────────────────────────────────────────────

class _ERISClient:
    def __init__(self, base_url: str, timeout: float = 180.0):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def health(self) -> dict:
        r = self.http.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def encode(self, text: str, layer: int) -> dict:
        r = self.http.post(
            f"{self.base_url}/v1/encode",
            json={"text": text, "return_layers": [layer], "compact": True},
        )
        r.raise_for_status()
        return r.json()

    def think(self, prompt: str, n_steps: int) -> dict:
        r = self.http.post(
            f"{self.base_url}/v1/latent_think",
            json={"text": prompt, "n_steps": n_steps, "return_trajectory": True},
        )
        r.raise_for_status()
        return r.json()

    def close(self): self.http.close()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_aime_problems(n: int = 20, seed: int = 42) -> list[dict]:
    """
    Load AIME problems with answers.

    Tries multiple HuggingFace sources.  Falls back to a small built-in
    sample if the dataset is unavailable (allows offline testing).
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    sources = [
        ("di-mi/AIME_2024", "train"),
        ("Maxwell-Jiang/AIME_1983_2024", "train"),
        ("AI-MO/aimo-validation-aime", "train"),
    ]

    for ds_name, split in sources:
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_name, split=split, trust_remote_code=True)
            items = list(ds)
            rng.shuffle(items)
            problems = []
            for item in items[:n]:
                # Normalise field names across datasets.
                q = item.get("problem") or item.get("question") or item.get("Problem", "")
                a = item.get("answer") or item.get("Answer") or item.get("solution", "")
                if q and a:
                    problems.append({"question": str(q), "answer": str(a).strip()})
            if problems:
                log.info("Loaded %d problems from %s", len(problems), ds_name)
                return problems[:n]
        except Exception as e:
            log.warning("Could not load %s: %s", ds_name, e)

    # Fallback: minimal synthetic problems for offline testing.
    log.warning("All AIME sources failed — using 5 synthetic fallback problems.")
    fallback = [
        {"question": "Find the sum of all positive integers n such that n² + 3n + 9 is a perfect square.", "answer": "3"},
        {"question": "How many integers between 1 and 1000 are divisible by 3 but not by 9?", "answer": "222"},
        {"question": "Let S = {1,2,3,...,20}. How many subsets have a sum divisible by 5?", "answer": "209715"},
        {"question": "The digits of a 3-digit number sum to 18. How many such numbers exist?", "answer": "54"},
        {"question": "Find the number of ways to write 10 as an ordered sum of positive integers.", "answer": "512"},
    ]
    return fallback


# ── Hidden state extraction ───────────────────────────────────────────────────

def _extract_b64(enc: dict, hidden_dim: int) -> np.ndarray | None:
    """Extract a hidden state vector from a /v1/encode response."""
    import base64
    hs = enc.get("hidden_states", {})
    for val in hs.values():
        if isinstance(val, str):
            flat = np.frombuffer(base64.b64decode(val), dtype=np.float32)
            mat = flat.reshape(-1, hidden_dim)
            return mat[-1]  # last token
        elif isinstance(val, list):
            mat = np.array(val, dtype=np.float32)
            if mat.ndim == 2:
                return mat[-1]
    return None


def extract_activations_at_layer(
    client: _ERISClient, text: str, layer: int, hidden_dim: int
) -> dict[int, np.ndarray] | None:
    """Return {layer: last_token_vector} for one text."""
    try:
        enc = client.encode(text, layer)
        hd = enc.get("hidden_dim", hidden_dim)
        vec = _extract_b64(enc, hd)
        if vec is not None:
            return {layer: vec}
    except Exception as e:
        log.warning("  encode error: %s", e)
    return None


# ── Answer checking ───────────────────────────────────────────────────────────

def _extract_answer(text: str) -> str | None:
    """Extract a numeric answer from model output."""
    # Look for boxed answer (LaTeX style).
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    # Look for "the answer is N" pattern.
    m = re.search(r'(?:the answer is|answer:|=)\s*([\d,]+)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()
    # Last number in text.
    nums = re.findall(r'\b\d+\b', text)
    return nums[-1] if nums else None


def check_answer(predicted: str | None, ground_truth: str) -> bool:
    """Return True if the predicted answer matches ground truth."""
    if predicted is None:
        return False
    # Normalise both: strip whitespace, remove commas.
    p = predicted.strip().replace(",", "")
    g = ground_truth.strip().replace(",", "")
    return p == g


# ── Main test ─────────────────────────────────────────────────────────────────

def run_test_0(
    eris_url: str,
    layer: int,
    n_problems: int,
    checkpoint_every: int,
    n_reasoning_steps: int,
    hidden_dim: int,
    output_path: Path,
) -> dict:
    """
    Run Test 0 and return results dict.

    Returns:
        {
          "spearman_rho": float,
          "spearman_p": float,
          "n_problems": int,
          "problems": [...],  # per-problem detail
          "kill_criterion_passed": bool,
          "kill_message": str,
        }
    """
    from scipy.stats import spearmanr

    client = _ERISClient(eris_url)
    try:
        health = client.health()
        model_name = health.get("model", "unknown")
        log.info("Connected to ERIS server: model=%s", model_name)
    except Exception as e:
        log.error("Cannot reach ERIS server at %s: %s", eris_url, e)
        raise

    problems = load_aime_problems(n=n_problems)
    log.info("Loaded %d problems", len(problems))

    detector = DriftDetector(threshold=0.3, window=3)
    results = []

    for i, prob in enumerate(problems):
        log.info("Problem %d/%d", i + 1, len(problems))
        question = prob["question"]
        ground_truth = prob["answer"]

        # Register reference: activations for the bare question.
        ref_acts = extract_activations_at_layer(client, question, layer, hidden_dim)
        if ref_acts is None:
            log.warning("  Skipping: could not encode question.")
            continue
        detector.reset()
        detector.register_reference(ref_acts, step=0)

        # Run latent reasoning in chunks of checkpoint_every steps.
        drift_history: list[float] = []
        final_text: str = question  # accumulate reasoning context

        n_chunks = n_reasoning_steps // checkpoint_every
        for chunk in range(1, n_chunks + 1):
            step = chunk * checkpoint_every

            # Simulate reasoning step: latent_think on accumulated context.
            try:
                think_result = client.think(final_text, n_steps=checkpoint_every)
                # If the model generates some text (from /v1/latent_think), append it.
                gen_text = think_result.get("generated_text", "")
                if gen_text:
                    final_text = final_text + "\n" + gen_text
            except Exception as e:
                log.warning("  think error at step %d: %s", step, e)
                break

            # Extract activations on the current reasoning context.
            cur_acts = extract_activations_at_layer(client, final_text, layer, hidden_dim)
            if cur_acts is None:
                log.warning("  encode error at step %d", step)
                break

            report = detector.compute_drift(cur_acts, step=step)
            drift_history.append(report.drift_score)
            log.info(
                "  step=%d drift=%.4f consult=%s",
                step, report.drift_score, report.should_consult_probe,
            )

        final_drift = float(np.mean(drift_history)) if drift_history else 0.0

        # Extract answer from whatever text we have.
        predicted = _extract_answer(final_text)
        correct = check_answer(predicted, ground_truth)
        error = 0 if correct else 1

        log.info(
            "  → predicted=%s, truth=%s, correct=%s, final_drift=%.4f",
            predicted, ground_truth, correct, final_drift,
        )

        results.append({
            "question":       question[:120],
            "ground_truth":   ground_truth,
            "predicted":      predicted,
            "correct":        correct,
            "error":          error,
            "final_drift":    round(final_drift, 5),
            "drift_history":  [round(d, 5) for d in drift_history],
        })

    # Spearman correlation: does higher drift → more errors?
    if len(results) < 5:
        log.error("Too few results (%d) for reliable statistics.", len(results))
        rho, p = 0.0, 1.0
    else:
        drift_scores = [r["final_drift"] for r in results]
        errors       = [r["error"]       for r in results]
        rho, p = spearmanr(drift_scores, errors)
        rho = float(rho)
        p   = float(p)

    log.info("Spearman ρ=%.4f, p=%.4f (n=%d)", rho, p, len(results))

    passed, message = check_criterion("test_0", rho)
    log.info(message)

    output = {
        "test":                  "test_0_drift_characterization",
        "model":                 model_name,
        "layer":                 layer,
        "n_problems":            len(results),
        "n_reasoning_steps":     n_reasoning_steps,
        "checkpoint_every":      checkpoint_every,
        "spearman_rho":          round(rho, 5),
        "spearman_p":            round(p, 6),
        "kill_criterion_passed": passed,
        "kill_message":          message,
        "problems":              results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", output_path)

    client.close()
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test 0 — Drift characterization (kill gate)"
    )
    ap.add_argument("--eris-url",         default="http://localhost:8001")
    ap.add_argument("--layer",            type=int, default=18,
                    help="Hidden layer to extract (default: 18, ~mid of 40-layer model)")
    ap.add_argument("--n-problems",       type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=3,
                    help="Extract activations every N reasoning steps")
    ap.add_argument("--n-reasoning-steps", type=int, default=24,
                    help="Total reasoning steps per problem")
    ap.add_argument("--hidden-dim",       type=int, default=5120,
                    help="Hidden dim of model (Qwen3-14B=5120, Qwen3-32B=7168)")
    ap.add_argument("--output",           default="results/test0_drift_characterization.json")
    args = ap.parse_args()

    result = run_test_0(
        eris_url=args.eris_url,
        layer=args.layer,
        n_problems=args.n_problems,
        checkpoint_every=args.checkpoint_every,
        n_reasoning_steps=args.n_reasoning_steps,
        hidden_dim=args.hidden_dim,
        output_path=Path(args.output),
    )

    print("\n" + "="*60)
    if result["kill_criterion_passed"]:
        print(f"✓ PASS  ρ={result['spearman_rho']:.4f}  →  proceed to test_1")
    else:
        print(f"✗ KILL  ρ={result['spearman_rho']:.4f}  →  STOP")
        print(f"  {result['kill_message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
