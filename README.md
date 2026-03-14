# Latent Relay

An MCP server that wraps [LatentMAS](https://arxiv.org/abs/2511.20639) (Zou et al., 2025) so that AI agents can use latent-space multi-agent reasoning as a tool.

## What this is

LatentMAS is a research framework where LLM agents collaborate by passing KV-caches instead of text. It works well but exists as a benchmark script tied to specific models and evaluation tasks.

This project repackages that mechanism as:
- A standalone **Python engine** with a simple API
- A **FastAPI server** with REST endpoints
- An **MCP server** with tool definitions that agents can discover and call
- An **OpenAI-compatible proxy** that works as a drop-in provider for [OpenClaw](https://github.com/openclaw/openclaw) and [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)

The idea is straightforward: agents call `latent_think` to reason without generating tokens, pass handles to each other, and call `latent_collaborate` when they need a text answer.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/latent-relay.git
cd latent-relay
pip install -r requirements.txt
```

## Usage

**Direct test** (no server):
```bash
python test_e2e.py --model Qwen/Qwen3-4B --n_steps 60
```

**REST server**:
```bash
python server.py --model Qwen/Qwen3-4B --port 8000
```

**MCP server** (stdio, for Claude Desktop etc.):
```bash
LATENT_MODEL=Qwen/Qwen3-4B python mcp_server.py
```

**OpenAI-compatible proxy** (for OpenClaw / OpenClaw-RL):
```bash
LATENT_MODEL=Qwen/Qwen3-8B python openclaw_compat/openai_proxy.py --port 30000
```

## OpenClaw Integration

The proxy exposes a standard `/v1/chat/completions` endpoint. OpenClaw connects to it like any other provider.

**Add to your `~/.openclaw/openclaw.json`:**

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "latent-relay": {
        "baseUrl": "http://HOST:30000/v1",
        "apiKey": "latent-relay-key",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3-8b",
            "name": "Latent Relay (Qwen3-8B)",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 32768,
            "maxTokens": 4096
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": { "primary": "latent-relay/qwen3-8b" }
    }
  }
}
```

Then restart the gateway. The proxy handles OpenClaw's full request format (30+ tool definitions, streaming, multi-turn sessions).

**For OpenClaw-RL**, the proxy replaces SGLang as the inference backend. Point `openclaw_api_server.py`'s backend URL to `http://localhost:30000/v1`.

### Tested

Validated end-to-end with OpenClaw v2026.3.13 on RunPod (1×GPU). The proxy receives ~58KB requests (system prompt with 30+ tool definitions), runs the latent planner→critic pipeline, and streams responses back via SSE. See [`openclaw_compat/VALIDATION_LOG.md`](openclaw_compat/VALIDATION_LOG.md) for full test results.

| Model | Handles OpenClaw system prompt | Response quality |
|-------|-------------------------------|-----------------|
| Qwen3-4B | Repetition loops | Basic |
| Qwen3-8B | Clean structured output | Good (tested live) |
| Qwen3-14B | Expected to work | Not yet tested |

## MCP Tools

| Tool | Description |
|------|-------------|
| `latent_create_session` | Load model, compute alignment matrix |
| `latent_think` | Run latent reasoning, return a handle (no text output) |
| `latent_collaborate` | Generate text from accumulated latent context |
| `latent_thought_info` | Inspect a stored thought |
| `latent_delete_session` | Free resources |

## How it works

This is a thin wrapper around the LatentMAS mechanism. Each `latent_think` call:
1. Encodes the prompt through the model
2. Runs N steps of latent rollout (hidden state → W_a alignment → forward pass → next hidden state)
3. Stores the resulting KV-cache under a handle

Agents can inherit previous handles, so context accumulates. `latent_collaborate` takes the accumulated KV-cache context and generates text.

The alignment matrix W_a maps output hidden states back to input embedding space. It's computed once at startup via ridge regression on the embedding matrices.

For details, see the [LatentMAS paper](https://arxiv.org/abs/2511.20639).

## Compatible models

Tested with Qwen3-4B and Qwen3-8B. Should work with any HuggingFace model that uses standard KV-cache (`past_key_values`). A patch for DeepSeek-V2 MLA models is included in `patches/` — HF's MLA implementation caches full K,V after up-projection, so the interface is the same.

Models with non-standard attention (Qwen3.5's Gated DeltaNet, GLM-5's sparse attention) are not compatible.

## Project structure

```
engine.py                          Core engine (DynamicCache compatible)
server.py                          FastAPI REST server
mcp_server.py                      MCP tool server
test_e2e.py                        End-to-end test
openclaw_compat/
  openai_proxy.py                  OpenAI-compatible proxy for OpenClaw
  test_01_openai_compat.py         API format tests
  test_02_openclaw_config.py       OpenClaw config simulation
  test_03_openclaw_rl_compat.py    OpenClaw-RL rollout compatibility
  run_openclaw_tests.sh            Test runner
  VALIDATION_LOG.md                Live test results
patches/                           DeepSeek MLA adapter + LatentMAS patches
phase0/                            Feasibility validation suite
```

## Limitations

- Requires all agents to use the same model (same architecture, same weights)
- KV-caches live in GPU memory — limited by VRAM
- No interpretability of intermediate latent reasoning
- The `collaborate` step re-encodes the final prompt (doesn't pass inherited KV-cache to `generate` due to position encoding issues with some models)
- Small models (4B) struggle with OpenClaw's large system prompts (~58KB)
- Only tested on math reasoning tasks (GSM8K) and OpenClaw general conversation

## Acknowledgments

All the core ideas are from the LatentMAS paper by Zou, Yang, Qiu et al. (Princeton, UIUC, Stanford). This project just wraps their work in a different interface.

```bibtex
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

## License

MIT
