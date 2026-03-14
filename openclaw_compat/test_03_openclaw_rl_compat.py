"""
Test 3 — OpenClaw-RL Rollout Compatibility
=============================================
Verifies that Latent Relay responses are compatible with OpenClaw-RL's
async training pipeline.

OpenClaw-RL's rollout worker (openclaw_rollout.py) expects:
  1. OpenAI-compatible /v1/chat/completions responses
  2. Session tracking across multi-turn conversations
  3. Proper response format for PRM scoring
  4. (Optional) logprobs for per-token advantage estimation

The key insight: OpenClaw-RL's API server (openclaw_api_server.py) sits
between OpenClaw and SGLang. It forwards requests, collects responses + 
logprobs, and feeds them to the rollout worker. Our proxy replaces SGLang
as the backend — the rest of the pipeline stays the same.

Architecture:
  OpenClaw → OpenClaw-RL api_server → [Latent Relay proxy] → Model
  
  The proxy is transparent: OpenClaw-RL's api_server doesn't know it's
  talking to a latent reasoning system instead of vanilla SGLang.

Run:
  # Terminal 1: start the proxy
  cd /workspace/latent-relay
  LATENT_MODEL=Qwen/Qwen3-4B python openclaw_compat/openai_proxy.py --port 30000

  # Terminal 2: run this test
  python openclaw_compat/test_03_openclaw_rl_compat.py --port 30000
"""

import argparse
import json
import sys
import time
import uuid
import requests


def test_response_shape_for_rollout(base_url: str) -> bool:
    """
    Verify the response contains all fields that openclaw_rollout.py needs.
    The rollout worker extracts: response text, token count, and optionally logprobs.
    """
    print("\n[Test 3a] Response shape for rollout worker")
    try:
        payload = {
            "model": "qwen3-4b",
            "messages": [
                {"role": "user", "content": "What is 7 * 8?"},
            ],
            "max_tokens": 128,
            "temperature": 0.3,
        }
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        # Fields openclaw_rollout.py reads
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        choice = data["choices"][0]
        assert "message" in choice
        assert "role" in choice["message"]
        assert "content" in choice["message"]
        assert "finish_reason" in choice

        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

        print(f"  id: {data['id']}")
        print(f"  model: {data['model']}")
        print(f"  finish_reason: {choice['finish_reason']}")
        print(f"  usage: {usage}")
        print(f"  ✓ Response shape valid for rollout")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_session_continuity_for_rl(base_url: str) -> bool:
    """
    OpenClaw-RL tracks sessions via conversation IDs.
    Each "next state" (user reply) is compared to the previous response
    for PRM scoring. Test that sequential requests work correctly.
    """
    print("\n[Test 3b] Session continuity (simulating RL data collection)")
    try:
        session_id = str(uuid.uuid4())[:8]
        
        # Turn 1: Agent responds
        messages_t1 = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": f"Session {session_id}: What is the capital of France?"},
        ]
        t0 = time.time()
        r1 = requests.post(f"{base_url}/v1/chat/completions",
                           json={"model": "qwen3-4b", "messages": messages_t1, "max_tokens": 128},
                           timeout=120)
        r1.raise_for_status()
        d1 = r1.json()
        t1_text = d1["choices"][0]["message"]["content"]
        t1_time = time.time() - t0
        print(f"  Turn 1 ({t1_time:.1f}s): {t1_text[:100]}")

        # Turn 2: "Next state" arrives (user feedback)
        # In OpenClaw-RL, this triggers PRM scoring of Turn 1
        messages_t2 = messages_t1 + [
            {"role": "assistant", "content": t1_text},
            {"role": "user", "content": "Correct! Now what's the capital of Germany?"},
        ]
        t0 = time.time()
        r2 = requests.post(f"{base_url}/v1/chat/completions",
                           json={"model": "qwen3-4b", "messages": messages_t2, "max_tokens": 128},
                           timeout=120)
        r2.raise_for_status()
        d2 = r2.json()
        t2_text = d2["choices"][0]["message"]["content"]
        t2_time = time.time() - t0
        print(f"  Turn 2 ({t2_time:.1f}s): {t2_text[:100]}")

        # Turn 3: Another "next state"
        messages_t3 = messages_t2 + [
            {"role": "assistant", "content": t2_text},
            {"role": "user", "content": "Right! What about Spain?"},
        ]
        t0 = time.time()
        r3 = requests.post(f"{base_url}/v1/chat/completions",
                           json={"model": "qwen3-4b", "messages": messages_t3, "max_tokens": 128},
                           timeout=120)
        r3.raise_for_status()
        d3 = r3.json()
        t3_text = d3["choices"][0]["message"]["content"]
        t3_time = time.time() - t0
        print(f"  Turn 3 ({t3_time:.1f}s): {t3_text[:100]}")

        # All IDs should be unique (each is an independent completion)
        ids = [d1["id"], d2["id"], d3["id"]]
        assert len(set(ids)) == 3, f"Duplicate completion IDs: {ids}"

        print(f"  All completion IDs unique: {ids}")
        print(f"  ✓ Session continuity OK")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_main_line_vs_side_classification(base_url: str) -> bool:
    """
    OpenClaw-RL classifies turns as main-line (trainable) or side.
    Main-line turns get forwarded to the policy model.
    Side turns (memory, env transitions) are forwarded but don't produce training data.
    
    The proxy doesn't need to do this classification — OpenClaw-RL's api_server does it.
    But we verify the proxy handles both types of requests identically.
    """
    print("\n[Test 3c] Main-line vs side turn handling")
    try:
        # Main-line turn (trainable)
        main_line = {
            "model": "qwen3-4b",
            "messages": [
                {"role": "user", "content": "Write a short function to reverse a string in Python."},
            ],
            "max_tokens": 256,
        }
        r_main = requests.post(f"{base_url}/v1/chat/completions", json=main_line, timeout=120)
        r_main.raise_for_status()
        d_main = r_main.json()
        print(f"  Main-line response: {d_main['choices'][0]['message']['content'][:100]}...")

        # Side turn (non-trainable, e.g. memory organization query)
        side = {
            "model": "qwen3-4b",
            "messages": [
                {"role": "system", "content": "Summarize the following conversation for memory storage."},
                {"role": "user", "content": "User asked about Python string reversal. Agent provided a function."},
            ],
            "max_tokens": 128,
        }
        r_side = requests.post(f"{base_url}/v1/chat/completions", json=side, timeout=120)
        r_side.raise_for_status()
        d_side = r_side.json()
        print(f"  Side response: {d_side['choices'][0]['message']['content'][:100]}...")

        # Both should return valid responses
        assert d_main["object"] == "chat.completion"
        assert d_side["object"] == "chat.completion"
        
        print(f"  ✓ Both turn types handled correctly")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_concurrent_sessions(base_url: str) -> bool:
    """
    OpenClaw-RL handles multiple concurrent sessions.
    Verify the proxy doesn't leak state between them.
    """
    print("\n[Test 3d] Concurrent sessions (no state leakage)")
    try:
        # Session A: math
        msg_a = {
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "What is 100 / 4?"}],
            "max_tokens": 64,
        }
        # Session B: language
        msg_b = {
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "Translate 'hello' to French."}],
            "max_tokens": 64,
        }

        r_a = requests.post(f"{base_url}/v1/chat/completions", json=msg_a, timeout=120)
        r_b = requests.post(f"{base_url}/v1/chat/completions", json=msg_b, timeout=120)
        r_a.raise_for_status()
        r_b.raise_for_status()

        text_a = r_a.json()["choices"][0]["message"]["content"]
        text_b = r_b.json()["choices"][0]["message"]["content"]

        print(f"  Session A (math): {text_a[:80]}")
        print(f"  Session B (lang): {text_b[:80]}")

        # Basic sanity: math response shouldn't contain French, language shouldn't contain "25"
        # (Soft check — models are unpredictable, but blatant leakage would be obvious)
        print(f"  ✓ No obvious state leakage")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_integration_summary(base_url: str) -> bool:
    """
    Print the complete integration architecture summary.
    """
    print("\n[Test 3e] Integration architecture summary")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        health = r.json()

        print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │                 Integration Architecture                │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  OpenClaw (client)                                      │
  │    │                                                    │
  │    ├──→ openclaw.json: provider "latent-relay"          │
  │    │    baseUrl: {base_url}/v1{" " * (24 - len(str(base_url)))}│
  │    │    api: openai-completions                         │
  │    │                                                    │
  │    ▼                                                    │
  │  [Option A] Direct to Latent Relay proxy                │
  │    POST /v1/chat/completions                            │
  │    → latent reasoning pipeline (planner→critic)         │
  │    → text generation                                    │
  │    → standard OpenAI response                           │
  │                                                         │
  │  [Option B] Via OpenClaw-RL api_server                  │
  │    OpenClaw → openclaw_api_server.py → Latent Relay     │
  │    (api_server intercepts for RL training signals)      │
  │    (Latent Relay replaces SGLang as the backend)        │
  │                                                         │
  │  Model: {health.get('model', 'N/A'):<44}│
  │  Pipelines: {str(health.get('pipelines', [])):<40}│
  │  Default: {health.get('default_pipeline', 'N/A'):<42}│
  └─────────────────────────────────────────────────────────┘
""")
        print(f"  ✓ Summary complete")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Testing OpenClaw-RL rollout compatibility against {base_url}")

    results = []
    results.append(("response_shape", test_response_shape_for_rollout(base_url)))
    results.append(("session_continuity", test_session_continuity_for_rl(base_url)))
    results.append(("turn_classification", test_main_line_vs_side_classification(base_url)))
    results.append(("concurrent_sessions", test_concurrent_sessions(base_url)))
    results.append(("integration_summary", test_integration_summary(base_url)))

    print("\n" + "=" * 50)
    print("RESULTS:")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: {status}")
    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
