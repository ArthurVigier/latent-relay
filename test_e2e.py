"""
Latent Relay — End-to-End Test
===============================
Tests the full pipeline: create session → think (3 agents) → collaborate → verify.

Usage:
    # Direct engine test (no server needed)
    python test_e2e.py --model Qwen/Qwen3-4B --n_steps 60

    # Against running FastAPI server
    python test_e2e.py --server http://localhost:8000
"""

import argparse
import json
import time
import sys


def test_engine_direct(model_name: str, device: str, n_steps: int):
    """Test the engine directly without any server."""
    from engine import LatentRelayEngine

    print("=" * 60)
    print("Latent Relay Engine — End-to-End Test")
    print("=" * 60)

    eng = LatentRelayEngine(model_name=model_name, device=device)

    question = (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
        "every morning and bakes muffins for her friends every day with four. "
        "She sells the remainder at the farmers' market daily for $2 per "
        "fresh duck egg. How much in dollars does she make every day at "
        "the farmers' market?"
    )
    expected = "18"

    # Step 1: Create session
    print("\n[1/5] Creating session...")
    sid = eng.create_session()
    print(f"  session_id: {sid}")

    # Step 2: Planner thinks
    print(f"\n[2/5] Planner thinking ({n_steps} latent steps)...")
    t0 = time.time()
    planner = eng.think(
        session_id=sid,
        prompt=f"You are a Planner Agent. Break down this problem into steps:\n\n{question}\n\nOutline a clear plan.",
        n_steps=n_steps,
        role="planner",
    )
    print(f"  handle: {planner['handle']}")
    print(f"  positions: {planner['n_positions']}, time: {planner['elapsed_s']}s")

    # Step 3: Critic thinks (inherits from planner)
    print(f"\n[3/5] Critic thinking ({n_steps} latent steps, inherits planner)...")
    critic = eng.think(
        session_id=sid,
        prompt=f"You are a Critic Agent. Review the plan and identify any errors or improvements.\n\nQuestion: {question}",
        n_steps=n_steps,
        role="critic",
        inherit_from=[planner["handle"]],
    )
    print(f"  handle: {critic['handle']}")
    print(f"  positions: {critic['n_positions']}, time: {critic['elapsed_s']}s")

    # Step 4: Refiner thinks (inherits from critic)
    print(f"\n[4/5] Refiner thinking ({n_steps} latent steps, inherits critic)...")
    refiner = eng.think(
        session_id=sid,
        prompt=f"You are a Refiner Agent. Produce a refined, correct solution plan.\n\nQuestion: {question}",
        n_steps=n_steps,
        role="refiner",
        inherit_from=[critic["handle"]],
    )
    print(f"  handle: {refiner['handle']}")
    print(f"  positions: {refiner['n_positions']}, time: {refiner['elapsed_s']}s")

    # Step 5: Collaborate — generate final answer
    print("\n[5/5] Collaborating — generating final answer...")
    result = eng.collaborate(
        session_id=sid,
        handles=[refiner["handle"]],
        final_prompt=(
            f"You are given latent reasoning context from a multi-agent team. "
            f"Answer the following question step by step.\n\n"
            f"Question: {question}\n\n"
            f"Reason step by step and output the final answer inside \\boxed{{YOUR_ANSWER}}."
        ),
        max_new_tokens=512,
    )

    total_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"GENERATED ANSWER:")
    print(f"{'='*60}")
    print(result["text"])
    print(f"\n{'='*60}")
    print(f"Tokens generated: {result['tokens_generated']}")
    print(f"Total pipeline time: {total_time:.2f}s")

    # Check if answer contains expected value
    answer_correct = expected in result["text"]
    print(f"\nExpected answer contains '{expected}': {'PASS' if answer_correct else 'FAIL'}")

    # Cleanup
    eng.delete_session(sid)
    print(f"\nSession {sid} deleted.")

    return answer_correct


def test_server(server_url: str, n_steps: int = 60):
    """Test against a running FastAPI server."""
    import requests

    print("=" * 60)
    print(f"Latent Relay Server Test — {server_url}")
    print("=" * 60)

    # Health check
    r = requests.get(f"{server_url}/health")
    print(f"\n[Health] {r.json()}")

    question = (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?"
    )
    expected = "3"

    # Create session
    r = requests.post(f"{server_url}/sessions")
    sid = r.json()["session_id"]
    print(f"\n[Session] {sid}")

    # Think: planner
    r = requests.post(f"{server_url}/think", json={
        "session_id": sid,
        "prompt": f"Plan how to solve: {question}",
        "n_steps": n_steps,
        "role": "planner",
    })
    planner = r.json()
    print(f"[Planner] {planner['handle']} ({planner['elapsed_s']}s)")

    # Think: critic
    r = requests.post(f"{server_url}/think", json={
        "session_id": sid,
        "prompt": f"Critique the plan for: {question}",
        "n_steps": n_steps,
        "role": "critic",
        "inherit_from": [planner["handle"]],
    })
    critic = r.json()
    print(f"[Critic] {critic['handle']} ({critic['elapsed_s']}s)")

    # Collaborate
    r = requests.post(f"{server_url}/collaborate", json={
        "session_id": sid,
        "handles": [critic["handle"]],
        "final_prompt": f"Solve step by step: {question}\n\nAnswer inside \\boxed{{}}.",
        "max_new_tokens": 256,
    })
    result = r.json()
    print(f"\n[Answer] {result['text']}")
    print(f"[Tokens] {result['tokens_generated']} in {result['elapsed_s']}s")

    answer_correct = expected in result["text"]
    print(f"\nExpected '{expected}': {'PASS' if answer_correct else 'FAIL'}")

    # Cleanup
    requests.delete(f"{server_url}/sessions/{sid}")
    return answer_correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_steps", type=int, default=60)
    parser.add_argument("--server", type=str, default=None,
                        help="Test against running server URL instead of direct engine")
    args = parser.parse_args()

    if args.server:
        ok = test_server(args.server, args.n_steps)
    else:
        ok = test_engine_direct(args.model, args.device, args.n_steps)

    sys.exit(0 if ok else 1)
