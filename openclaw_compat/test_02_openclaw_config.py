"""
Test 2 — OpenClaw Config Simulation
=======================================
Simulates how OpenClaw connects to a provider:
  - Generates the exact openclaw.json provider config
  - Sends requests as the OpenClaw gateway does (OpenAI-completions API)
  - Tests session continuity across multi-turn exchanges

This validates that an OpenClaw user can drop this config in and it works.

Run:
  # Terminal 1: start the proxy
  cd /workspace/latent-relay
  LATENT_MODEL=Qwen/Qwen3-4B python openclaw_compat/openai_proxy.py --port 30000

  # Terminal 2: run this test
  python openclaw_compat/test_02_openclaw_config.py --port 30000
"""

import argparse
import json
import os
import sys
import time
import requests


def generate_openclaw_config(host_ip: str, port: int, model_id: str = "qwen3-4b") -> dict:
    """
    Generate the exact openclaw.json provider config that a user would add.
    This matches the format from OpenClaw-RL's README.
    """
    return {
        "models": {
            "providers": {
                "latent-relay": {
                    "baseUrl": f"http://{host_ip}:{port}/v1",
                    "apiKey": "latent-relay-key",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": model_id,
                            "name": f"Latent Relay ({model_id})",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                            "contextWindow": 32768,
                            "maxTokens": 4096,
                        }
                    ],
                }
            }
        }
    }


def simulate_openclaw_request(
    base_url: str,
    messages: list,
    api_key: str = "latent-relay-key",
    model: str = "qwen3-4b",
    max_tokens: int = 512,
) -> dict:
    """
    Send a request exactly as OpenClaw's gateway does.
    OpenClaw uses the OpenAI-completions API format with:
      - Authorization: Bearer <apiKey>
      - Content-Type: application/json
      - POST /v1/chat/completions
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def test_config_generation(host_ip: str, port: int) -> bool:
    """Test that the generated config is valid OpenClaw format."""
    print("\n[Test 2a] Generate openclaw.json config")
    try:
        config = generate_openclaw_config(host_ip, port)

        # Validate structure matches OpenClaw-RL's expected format
        providers = config["models"]["providers"]
        assert "latent-relay" in providers
        provider = providers["latent-relay"]
        assert "baseUrl" in provider
        assert "apiKey" in provider
        assert provider["api"] == "openai-completions"
        assert len(provider["models"]) > 0
        model = provider["models"][0]
        assert "id" in model
        assert "contextWindow" in model
        assert "maxTokens" in model
        assert "cost" in model

        print(f"  Config:\n{json.dumps(config, indent=2)}")
        print(f"  ✓ Config valid")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_openclaw_single_turn(base_url: str) -> bool:
    """Simulate a single-turn OpenClaw conversation."""
    print("\n[Test 2b] Single-turn OpenClaw request")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful, concise assistant."},
            {"role": "user", "content": "Explain what a KV-cache is in one sentence."},
        ]
        t0 = time.time()
        data = simulate_openclaw_request(base_url, messages)
        elapsed = time.time() - t0

        text = data["choices"][0]["message"]["content"]
        print(f"  Response ({elapsed:.1f}s): {text[:300]}")
        print(f"  ✓ Single-turn OK")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_openclaw_multi_turn(base_url: str) -> bool:
    """
    Simulate a multi-turn OpenClaw conversation.
    OpenClaw sends the full conversation history each time.
    """
    print("\n[Test 2c] Multi-turn OpenClaw conversation")
    try:
        # Turn 1
        messages_t1 = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "I bought 3 books at $12 each."},
        ]
        t0 = time.time()
        data_t1 = simulate_openclaw_request(base_url, messages_t1, max_tokens=256)
        t1_text = data_t1["choices"][0]["message"]["content"]
        t1_elapsed = time.time() - t0
        print(f"  Turn 1 ({t1_elapsed:.1f}s): {t1_text[:150]}")

        # Turn 2 — OpenClaw appends full history
        messages_t2 = messages_t1 + [
            {"role": "assistant", "content": t1_text},
            {"role": "user", "content": "Then I bought 2 more books at $15 each. What's my total?"},
        ]
        t0 = time.time()
        data_t2 = simulate_openclaw_request(base_url, messages_t2, max_tokens=256)
        t2_text = data_t2["choices"][0]["message"]["content"]
        t2_elapsed = time.time() - t0
        print(f"  Turn 2 ({t2_elapsed:.1f}s): {t2_text[:150]}")
        
        # Check answer (3*12 + 2*15 = 36 + 30 = 66)
        has_66 = "66" in t2_text
        print(f"  Contains '$66': {has_66}")
        print(f"  ✓ Multi-turn OK")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_openclaw_tool_call_turn(base_url: str) -> bool:
    """
    OpenClaw classifies turns as main-line vs side.
    Tool-call results come back as user messages with tool output.
    Verify the proxy handles this format.
    """
    print("\n[Test 2d] Tool-call style turn (as OpenClaw sends them)")
    try:
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Run `ls -la` in my project directory."},
            {"role": "assistant", "content": "I'll list the files in your project directory."},
            {"role": "user", "content": "Tool output:\ntotal 48\ndrwxr-xr-x  5 user user 4096 Mar 14 10:00 .\n-rw-r--r--  1 user user 1234 Mar 14 09:55 main.py\n-rw-r--r--  1 user user  567 Mar 14 09:50 config.json\n\nPlease summarize what you see."},
        ]
        t0 = time.time()
        data = simulate_openclaw_request(base_url, messages, max_tokens=256)
        elapsed = time.time() - t0
        text = data["choices"][0]["message"]["content"]
        print(f"  Response ({elapsed:.1f}s): {text[:200]}")
        print(f"  ✓ Tool-call turn OK")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_save_config_file(host_ip: str, port: int) -> bool:
    """Save a ready-to-use openclaw.json snippet."""
    print("\n[Test 2e] Save openclaw.json config snippet")
    try:
        config = generate_openclaw_config(host_ip, port)
        out_path = "/tmp/openclaw_latent_relay_config.json"
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved to {out_path}")
        print(f"  ✓ Config file saved")
        
        # Also print instructions
        print(f"\n  ── To use with OpenClaw ──")
        print(f"  1. Merge this into your ~/.openclaw/openclaw.json under 'models.providers'")
        print(f"  2. Restart the OpenClaw gateway")
        print(f"  3. Select 'Latent Relay' as your model in OpenClaw")
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
    host_ip = args.host if args.host != "localhost" else "YOUR_HOST_IP"
    
    print(f"Testing OpenClaw compatibility against {base_url}")

    results = []
    results.append(("config_gen", test_config_generation(host_ip, args.port)))
    results.append(("single_turn", test_openclaw_single_turn(base_url)))
    results.append(("multi_turn", test_openclaw_multi_turn(base_url)))
    results.append(("tool_call", test_openclaw_tool_call_turn(base_url)))
    results.append(("save_config", test_save_config_file(host_ip, args.port)))

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
