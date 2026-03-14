# OpenClaw Live Validation — Test Results

## Setup

- **Hardware**: RunPod 1×GPU (A40/A100), 40GB+ VRAM
- **Model**: Qwen/Qwen3-4B (unit tests), Qwen/Qwen3-8B (live OpenClaw test)
- **OpenClaw**: v2026.3.13 (61d171a)
- **Proxy**: Latent Relay OpenAI-compatible proxy on port 30000
- **Pipeline**: planner-critic (40 + 30 latent steps)

---

## Phase 1 — Unit Tests (15/15 PASS)

### Test 1: OpenAI API Compatibility (Qwen3-4B)

```
Testing OpenAI compatibility against http://localhost:30000

[Test 1e] GET /health
  Model: Qwen/Qwen3-4B
  Pipelines: ['single', 'planner-critic', 'full']
  ✓ Health OK

[Test 1a] GET /v1/models
  ✓ Models endpoint OK — model_id: qwen3-4b

[Test 1b] POST /v1/chat/completions (basic)
  Response (17.2s): The sum of 15 and 27 is 42.
  Tokens: prompt=34, completion=256
  Contains '42': True
  ✓ Chat completions basic OK

[Test 1c] POST /v1/chat/completions (multi-turn)
  Response (12.0s): Now you have 8 apples.
  Contains '8': True
  ✓ Multi-turn OK

[Test 1d] POST /v1/chat/completions (latent_steps override)
  Response (13.8s, single pipeline, 20 steps): [correct]
  ✓ Latent override OK

RESULTS: ALL TESTS PASSED
```

### Test 2: OpenClaw Config Simulation

```
[Test 2a] Generate openclaw.json config
  ✓ Config valid

[Test 2b] Single-turn OpenClaw request
  Response (28.0s): A KV-cache is a data structure that stores key-value
  pairs for fast retrieval...
  ✓ Single-turn OK

[Test 2c] Multi-turn OpenClaw conversation
  Turn 1 (15.2s): [calculates 3 books at $12]
  Turn 2 (15.4s): [adds 2 books at $15, total $66]
  Contains '$66': True
  ✓ Multi-turn OK

[Test 2d] Tool-call style turn
  ✓ Tool-call turn OK

[Test 2e] Save openclaw.json config snippet
  ✓ Config file saved

RESULTS: ALL TESTS PASSED
```

### Test 3: OpenClaw-RL Rollout Compatibility

```
[Test 3a] Response shape for rollout worker
  id: chatcmpl-9216da5a56c5
  model: qwen3-4b
  finish_reason: stop
  usage: {prompt_tokens: 14, completion_tokens: 128, total_tokens: 142}
  ✓ Response shape valid for rollout

[Test 3b] Session continuity (3-turn RL simulation)
  Turn 1 (9.6s): The capital of France is Paris.
  Turn 2 (10.9s): The capital of Germany is Berlin.
  Turn 3 (9.9s): The capital of Spain is Madrid.
  All completion IDs unique: ✓
  ✓ Session continuity OK

[Test 3c] Main-line vs side turn handling
  ✓ Both turn types handled correctly

[Test 3d] Concurrent sessions
  Session A (math): 100 / 4 = 25 ✓
  Session B (lang): "bonjour" ✓
  ✓ No state leakage

[Test 3e] Integration architecture summary
  ✓ Summary complete

RESULTS: ALL TESTS PASSED
```

---

## Phase 2 — Live OpenClaw Integration (Qwen3-8B)

### Configuration

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "latent-relay": {
        "baseUrl": "http://127.0.0.1:30000/v1",
        "apiKey": "latent-relay-key",
        "api": "openai-completions",
        "models": [{
          "id": "qwen3-4b",
          "name": "Latent Relay (Qwen3-4B)",
          "reasoning": true,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 32768,
          "maxTokens": 4096
        }]
      }
    }
  }
}
```

### Gateway startup

```
[gateway] agent model: latent-relay/qwen3-4b
[gateway] listening on ws://127.0.0.1:18789
```

### OpenClaw TUI — live conversation (Qwen3-8B)

```
> how does i do an extended sanity check on a compiler ?

To perform an extended sanity check on a compiler, follow these structured steps:

1. Basic Validation:
- Compile a minimal valid program (e.g., int main() { return 0; }).
  Ensure no syntax errors and that the binary executes correctly.
- Test edge cases like maximum recursion depth, undefined behavior
  (e.g., division by zero), and type mismatches.

2. Test Suite Execution:
- Run predefined test suites (e.g., LLVM's LLVMTests or GCC's
  gfortran-testsuite) to validate core functionality.
- Use CI pipelines to automate testing across platforms.

3. Optimization Checks:
- Compare outputs of optimized vs non-optimized builds for consistency.
- Verify that optimizations produce correct results without introducing bugs.

4. Cross-Platform Testing:
- Validate compilation and execution on target architectures (ARM, x86_64).
- Ensure compatibility with system libraries and dependencies.

5. Integration Tests:
- Link programs with external libraries (POSIX APIs, STL) to confirm
  proper interop.
- Test macro expansion, inline assembly, and attribute handling.

6. Stress Testing:
- Use large input files (millions of lines) to identify performance
  bottlenecks or memory leaks.
- Simulate high-concurrency scenarios if the compiler supports it.

7. Documentation Compliance:
- Confirm adherence to C/C++/Rust standards.
- Review compiler diagnostics for clarity and specificity.

Tools:
- gcc -Wall -Wextra for static analysis.
- clang-format for coding standards.
- valgrind or AddressSanitizer for memory-checking.
```

### Model comparison (same OpenClaw system prompt, ~58KB with 30+ tool definitions)

| | Qwen3-4B | Qwen3-8B |
|---|---|---|
| Handles 58KB system prompt | Repetition loops | Clean structured output |
| Response quality | Basic, repetitive | Detailed, 7-step guide |
| Latent pipeline time | ~15s | ~20s |
| Trailing artifacts | "Assistant" loops | Minor NO_REPLY (fixed in proxy) |

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Integration Architecture                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  OpenClaw (client)                                      │
│    │                                                    │
│    ├──→ openclaw.json: provider "latent-relay"          │
│    │    baseUrl: http://HOST:30000/v1                   │
│    │    api: openai-completions                         │
│    │                                                    │
│    ▼                                                    │
│  [Option A] Direct to Latent Relay proxy                │
│    POST /v1/chat/completions                            │
│    → latent reasoning pipeline (planner→critic)         │
│    → text generation                                    │
│    → standard OpenAI response (+ SSE if stream:true)    │
│                                                         │
│  [Option B] Via OpenClaw-RL api_server                  │
│    OpenClaw → openclaw_api_server.py → Latent Relay     │
│    (api_server intercepts for RL training signals)      │
│    (Latent Relay replaces SGLang as the backend)        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Issues Encountered & Fixed

1. **DynamicCache not subscriptable** — transformers >= 4.36 returns `DynamicCache` objects instead of tuples for `past_key_values`. Fixed by accessing `past_kv.key_cache[0].shape[-2]`.

2. **422 Unprocessable Entity** — OpenClaw sends ~58KB request bodies with 30+ tool definitions, `stream:true`, `tool_choice`, etc. Pydantic strict validation rejected these. Fixed by removing Pydantic models and parsing raw JSON with `request.json()`.

3. **SSE streaming** — OpenClaw's JS client (`OpenAI/JS 6.26.0`) sends `stream:true`. Added SSE response format (single chunk + `[DONE]`).

4. **NO_REPLY / Assistant trailing tokens** — Small models generate padding artifacts. Fixed with regex cleanup in the proxy.

5. **Corrupted model weights** — Incomplete HF download caused `SafetensorError`. Fixed by clearing cache and re-downloading.
