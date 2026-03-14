"""
Test 1 — OpenAI API Compatibility
====================================
Verifies the proxy responds correctly to:
  - GET /v1/models
  - POST /v1/chat/completions (standard format)
  - POST /v1/chat/completions (with latent_steps override)

Run:
  # Terminal 1: start the proxy
  cd /workspace/latent-relay
  LATENT_MODEL=Qwen/Qwen3-4B python openclaw_compat/openai_proxy.py --port 30000

  # Terminal 2: run this test
  python openclaw_compat/test_01_openai_compat.py --port 30000
"""

import argparse
import json
import sys
import time
import requests


def test_models_endpoint(base_url: str) -> bool:
    """Test GET /v1/models."""
    print("\n[Test 1a] GET /v1/models")
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        data = r.json()
        assert data["object"] == "list", f"Expected 'list', got {data['object']}"
        assert len(data["data"]) > 0, "No models returned"
        model = data["data"][0]
        assert "id" in model, "Model missing 'id'"
        assert model["object"] == "model", f"Expected 'model', got {model['object']}"
        print(f"  ✓ Models endpoint OK — model_id: {model['id']}")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_chat_completions_basic(base_url: str) -> bool:
    """Test POST /v1/chat/completions with a simple math question."""
    print("\n[Test 1b] POST /v1/chat/completions (basic)")
    payload = {
        "model": "qwen3-4b",
        "messages": [
            {"role": "system", "content": "You are a helpful math tutor. Give concise answers."},
            {"role": "user", "content": "What is 15 + 27?"},
        ],
        "max_tokens": 256,
        "temperature": 0.3,
    }
    try:
        t0 = time.time()
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
        elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()

        # Validate OpenAI response shape
        assert data["object"] == "chat.completion", f"Expected 'chat.completion', got {data['object']}"
        assert "id" in data, "Missing 'id'"
        assert "choices" in data, "Missing 'choices'"
        assert len(data["choices"]) > 0, "Empty choices"

        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant", f"Expected role 'assistant'"
        assert len(choice["message"]["content"]) > 0, "Empty content"
        assert choice["finish_reason"] in ("stop", "length"), f"Unexpected finish_reason: {choice['finish_reason']}"

        assert "usage" in data, "Missing 'usage'"
        assert data["usage"]["completion_tokens"] > 0, "Zero completion tokens"

        text = choice["message"]["content"]
        has_42 = "42" in text
        print(f"  Response ({elapsed:.1f}s): {text[:200]}")
        print(f"  Tokens: prompt={data['usage']['prompt_tokens']}, completion={data['usage']['completion_tokens']}")
        print(f"  Contains '42': {has_42}")
        print(f"  ✓ Chat completions basic OK")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_chat_completions_multiturn(base_url: str) -> bool:
    """Test multi-turn conversation (how OpenClaw sends requests)."""
    print("\n[Test 1c] POST /v1/chat/completions (multi-turn)")
    payload = {
        "model": "qwen3-4b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I have 3 apples."},
            {"role": "assistant", "content": "OK, you have 3 apples."},
            {"role": "user", "content": "My friend gives me 5 more. How many do I have now?"},
        ],
        "max_tokens": 256,
        "temperature": 0.3,
    }
    try:
        t0 = time.time()
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
        elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        has_8 = "8" in text
        print(f"  Response ({elapsed:.1f}s): {text[:200]}")
        print(f"  Contains '8': {has_8}")
        print(f"  ✓ Multi-turn OK")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_chat_completions_with_latent_override(base_url: str) -> bool:
    """Test with latent_steps override (extension param, ignored by standard clients)."""
    print("\n[Test 1d] POST /v1/chat/completions (latent_steps override)")
    payload = {
        "model": "qwen3-4b",
        "messages": [
            {"role": "user", "content": "What is 12 * 11?"},
        ],
        "max_tokens": 256,
        "temperature": 0.3,
        "latent_steps": 20,
        "latent_pipeline": "single",
    }
    try:
        t0 = time.time()
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
        elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        print(f"  Response ({elapsed:.1f}s, single pipeline, 20 steps): {text[:200]}")
        print(f"  ✓ Latent override OK")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_health(base_url: str) -> bool:
    """Test /health endpoint."""
    print("\n[Test 1e] GET /health")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        data = r.json()
        assert data["openai_compat"] is True
        assert "pipelines" in data
        print(f"  Model: {data['model']}")
        print(f"  Pipelines: {data['pipelines']}")
        print(f"  ✓ Health OK")
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
    print(f"Testing OpenAI compatibility against {base_url}")

    results = []
    results.append(("health", test_health(base_url)))
    results.append(("models", test_models_endpoint(base_url)))
    results.append(("basic", test_chat_completions_basic(base_url)))
    results.append(("multiturn", test_chat_completions_multiturn(base_url)))
    results.append(("latent_override", test_chat_completions_with_latent_override(base_url)))

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
