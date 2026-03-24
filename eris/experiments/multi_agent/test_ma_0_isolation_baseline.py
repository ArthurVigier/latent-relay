"""
eris/experiments/multi_agent/test_ma_0_isolation_baseline.py
=============================================================

Kill-gate MA-0 — ISOLATED mode baseline.

Measures the mean pairwise cosine distance between same-step activations
of two agents running in ISOLATED mode on the same set of problems.

If distance < 0.15 (agents stay close without any sharing), the test passes
and confirms that ISOLATED mode is a meaningful baseline (not noise).

If distance ≥ 0.15, the test is killed: agents are already too divergent in
isolation, making SHARED_MEDIUM comparisons uninterpretable.

Usage::

    ANTHROPIC_API_KEY=... python -m eris.experiments.multi_agent.test_ma_0_isolation_baseline \\
        --model-id Qwen/Qwen3-14B \\
        --layer 18 \\
        --n-problems 10

Exit code: 0 = PASS, 1 = KILL.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eris.ma_0")


_PROBLEMS = [
    "Is every even integer greater than 2 the sum of two primes?",
    "What is the minimum number of moves to solve a 3×3 Rubik's cube from any position?",
    "Prove that there are infinitely many prime numbers.",
    "What is the sum of all integers from 1 to 100?",
    "Can you tile a 8×8 chessboard with two opposite corners removed using 2×1 dominoes?",
    "What is the largest prime factor of 600851475143?",
    "Is the halting problem decidable?",
    "Prove that √2 is irrational.",
    "How many handshakes occur if 10 people each shake hands with every other person once?",
    "What is 2^10 mod 1000?",
]


def main():
    parser = argparse.ArgumentParser(description="Kill-gate MA-0: ISOLATED baseline")
    parser.add_argument("--model-id",    default="Qwen/Qwen3-14B")
    parser.add_argument("--layer",       type=int, default=18)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--n-problems",  type=int, default=10)
    parser.add_argument("--max-steps",   type=int, default=5)
    parser.add_argument("--output-dir",  default="results/multi_agent")
    args = parser.parse_args()

    from eris.probe import LatentProbe
    from eris.multi_agent import MultiAgentCoordinator, CoordinationMode, AgentConfig
    from eris.backends.orchestrators.claude_orchestrator import ClaudeOrchestrator
    from eris.experiments.multi_agent.kill_criteria_multi_agent import check_criterion

    # Load probe once (shared).
    probe = LatentProbe(args.model_id, layers=[args.layer], device=args.device)

    coord = MultiAgentCoordinator(
        probe=probe,
        mode=CoordinationMode.ISOLATED,
        n_agents=2,
    )

    problems = _PROBLEMS[: args.n_problems]
    pairwise_distances: list[float] = []

    for i, problem in enumerate(problems):
        log.info("Problem %d/%d: %s", i + 1, len(problems), problem[:60])
        agents = [
            AgentConfig(llm=ClaudeOrchestrator(), name=f"agent_{j}")
            for j in range(2)
        ]
        result = coord.run(agents, problem=problem, max_steps=args.max_steps)

        # Extract last-step activations for both agents.
        acts = []
        for ar in result.agent_results:
            if ar.history:
                last_text = ar.history[-1].content
                a = probe.probe(last_text, layers=[args.layer])
                acts.append(a[args.layer])

        if len(acts) == 2:
            cos_d = _cosine_distance(acts[0], acts[1])
            pairwise_distances.append(cos_d)
            log.info("  cosine_distance = %.4f", cos_d)

    if not pairwise_distances:
        log.error("No valid pairwise distances computed. Aborting.")
        sys.exit(1)

    mean_dist = float(np.mean(pairwise_distances))
    log.info("Mean pairwise cosine distance (ISOLATED): %.4f", mean_dist)

    passed, msg = check_criterion("ma_0", mean_dist)
    print(msg)

    # Save results.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ma_0_isolation_baseline.json"
    with open(out_path, "w") as f:
        json.dump({
            "test":                "ma_0",
            "passed":              passed,
            "mean_cosine_distance": mean_dist,
            "per_problem":         pairwise_distances,
            "n_problems":          len(pairwise_distances),
            "model_id":            args.model_id,
            "layer":               args.layer,
        }, f, indent=2)
    log.info("Results saved to %s", out_path)

    sys.exit(0 if passed else 1)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(1.0 - np.dot(a, b) / denom)


if __name__ == "__main__":
    main()
