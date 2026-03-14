#!/bin/bash
# ============================================================
# Latent Relay — OpenClaw Compatibility Test Suite
# ============================================================
# Run this on RunPod (1×A100/A40/L40, 40GB+ VRAM)
#
# Prerequisites:
#   - latent-relay repo cloned to /workspace/latent-relay
#   - LatentMAS repo cloned and patched (from Phase 0/1)
#   - Qwen3-4B downloaded to HF cache
#
# Usage:
#   bash run_openclaw_tests.sh
# ============================================================

set -e

echo "============================================"
echo "Latent Relay — OpenClaw Compat Tests"
echo "============================================"
echo ""

# ── 0. Setup ──

WORKDIR="/workspace/latent-relay"
MODEL="Qwen/Qwen3-4B"
PORT=30000
PROXY_PID=""

export HF_HOME="/workspace/hf_cache"
export LATENT_MODEL="$MODEL"
export LATENT_DEVICE="cuda:0"

cd "$WORKDIR"

# Make sure engine.py is accessible
if [ ! -f "engine.py" ]; then
    echo "ERROR: engine.py not found in $WORKDIR"
    echo "Copy your latent-relay files here first."
    exit 1
fi

# Install deps if needed
pip install fastapi uvicorn requests --quiet --break-system-packages 2>/dev/null || true


# ── 1. Start the OpenAI-compatible proxy ──

echo ""
echo "── Step 1: Starting OpenAI-compatible proxy on port $PORT ──"
echo "   Model: $MODEL"
echo "   This will take ~30s for model loading..."

python openclaw_compat/openai_proxy.py \
    --model "$MODEL" \
    --port $PORT \
    --pipeline planner-critic &
PROXY_PID=$!

# Wait for health check
echo "   Waiting for proxy to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "   Proxy ready after ${i}s"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "   ERROR: Proxy failed to start after 60s"
        kill $PROXY_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

echo ""
curl -s "http://localhost:$PORT/health" | python3 -m json.tool


# ── 2. Test 1: OpenAI API compatibility ──

echo ""
echo "============================================"
echo "── Step 2: Test 1 — OpenAI API Compatibility ──"
echo "============================================"

python openclaw_compat/test_01_openai_compat.py --port $PORT || true


# ── 3. Test 2: OpenClaw config simulation ──

echo ""
echo "============================================"
echo "── Step 3: Test 2 — OpenClaw Config Simulation ──"
echo "============================================"

python openclaw_compat/test_02_openclaw_config.py --port $PORT || true


# ── 4. Test 3: OpenClaw-RL rollout compatibility ──

echo ""
echo "============================================"
echo "── Step 4: Test 3 — OpenClaw-RL Rollout Compat ──"
echo "============================================"

python openclaw_compat/test_03_openclaw_rl_compat.py --port $PORT || true


# ── 5. Quick curl sanity check (copy-pasteable for README) ──

echo ""
echo "============================================"
echo "── Step 5: curl sanity check ──"
echo "============================================"

echo ""
echo "GET /v1/models:"
curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool

echo ""
echo "POST /v1/chat/completions:"
curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer latent-relay-key" \
    -d '{
        "model": "qwen3-4b",
        "messages": [{"role": "user", "content": "What is 9 + 10?"}],
        "max_tokens": 128
    }' | python3 -m json.tool


# ── 6. Generate openclaw.json snippet ──

echo ""
echo "============================================"
echo "── Step 6: Ready-to-use openclaw.json ──"
echo "============================================"

HOST_IP=$(hostname -I | awk '{print $1}')
cat << EOF

Add this to your ~/.openclaw/openclaw.json under "models.providers":

{
  "latent-relay": {
    "baseUrl": "http://${HOST_IP}:${PORT}/v1",
    "apiKey": "latent-relay-key",
    "api": "openai-completions",
    "models": [
      {
        "id": "qwen3-4b",
        "name": "Latent Relay (Qwen3-4B)",
        "reasoning": true,
        "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 32768,
        "maxTokens": 4096
      }
    ]
  }
}

For OpenClaw-RL, replace SGLang with this proxy:
  - In run_qwen3_4b_openclaw_rl.sh, set PORT=${PORT}
  - Or point openclaw_api_server.py's backend URL to http://localhost:${PORT}/v1

EOF


# ── Cleanup ──

echo "============================================"
echo "── Cleanup ──"
echo "============================================"
echo "Stopping proxy (PID: $PROXY_PID)..."
kill $PROXY_PID 2>/dev/null
wait $PROXY_PID 2>/dev/null

echo ""
echo "Done. All tests complete."
