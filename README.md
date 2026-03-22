# Latent Relay

An MCP server that wraps [LatentMAS](https://arxiv.org/abs/2511.20639) (Zou et al., 2025) so that AI agents can use latent-space multi-agent reasoning as a tool.

Extended with **ERIS v5** — a Platonic Cognitive Bridge that creates a latent communication channel between Claude (closed-source, text-only) and an open-source "zombie" model whose internal representations are fully inspectable and manipulable.

## What this is

LatentMAS is a research framework where LLM agents collaborate by passing KV-caches instead of text. It works well but exists as a benchmark script tied to specific models and evaluation tasks.

This project repackages that mechanism as:
- A standalone **Python engine** with a simple API
- A **FastAPI server** with REST endpoints
- An **MCP server** with tool definitions that agents can discover and call
- An **OpenAI-compatible proxy** that works as a drop-in provider for [OpenClaw](https://github.com/openclaw/openclaw) and [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)
- An **ERIS v5 layer** — Mechanistic Interpretability bridge, hidden state injection, and full Claude → Zombie → Claude pipeline

The idea is straightforward: agents call `latent_think` to reason without generating tokens, pass handles to each other, and call `latent_collaborate` when they need a text answer.

## Setup

```bash
git clone https://github.com/ArthurVigier/latent-relay.git
cd latent-relay
pip install -r requirements.txt
```

## Usage

**Direct test** (no server):
```bash
python test_e2e.py --model Qwen/Qwen3-4B --n_steps 60
```

**Base REST server**:
```bash
python server.py --model Qwen/Qwen3-4B --port 8000
```

**ERIS v5 server** (superset — all base endpoints + ERIS endpoints):
```bash
python eris_server.py --model Qwen/Qwen3-14B --port 8001
# or with custom config:
ERIS_CONFIG=configs/eris_config.yaml uvicorn eris_server:app --host 0.0.0.0 --port 8001
```

**MCP server** (stdio, for Claude Desktop etc.):
```bash
LATENT_MODEL=Qwen/Qwen3-4B python mcp_server.py
```

**OpenAI-compatible proxy** (for OpenClaw / OpenClaw-RL):
```bash
LATENT_MODEL=Qwen/Qwen3-8B python openclaw_compat/openai_proxy.py --port 30000
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `latent_create_session` | Load model, compute alignment matrix |
| `latent_think` | Run latent reasoning, return a handle (no text output) |
| `latent_collaborate` | Generate text from accumulated latent context |
| `latent_thought_info` | Inspect a stored thought |
| `latent_delete_session` | Free resources |

## ERIS v5 Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/encode` | Encode text, expose per-layer hidden states (base64 float32) |
| `POST /v1/analyze` | Run MI analyses on stored hidden states (SAE, Â-hat, cosine map, PCA, norms) |
| `POST /v1/latent_think` | Extended `/think` — returns full trajectory + perturbation support |
| `POST /v1/inject` | Surgical hidden-state injection (add / steer / replace) |
| `POST /v1/bridge` | Full Claude → Zombie → Claude pipeline |

### Quick example — Claude → Zombie bridge

```python
from eris_client import ClaudeZombieBridge
import os

bridge = ClaudeZombieBridge(
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    eris_base_url="http://localhost:8001",
    bridge_mode="ruminate",
    n_steps=60,
)
turn = bridge.chat("What are the risks of distributed locking?")
print(turn.claude_text)
print(turn.enriched_text)
print(turn.analysis["implicit_features"])
```

### Decode hidden states

```python
from eris_client import ERISClient, decode_hidden_states

with ERISClient("http://localhost:8001") as client:
    enc = client.encode("Hello world", return_layers=[-1, 15], compact=True)
    hs = decode_hidden_states(enc)   # {"last": np.ndarray[seq_len, 3584], "layer_15": ...}
```

### Configuration

Copy `configs/eris_config.yaml` and adjust paths:

```yaml
model_name: "Qwen/Qwen3-14B"
device: "cuda:0"
default_n_steps: 60

sae:
  model_path: "/path/to/sae_weights"  # optional
  top_k: 20

a_hat:
  model_path: "/path/to/a_hat_probe.pt"  # optional

concept_vectors:
  vectors_dir: "configs/concept_vectors"

server:
  host: "0.0.0.0"
  port: 8001
  max_payload_bytes: 52428800  # 50 MB
```

All analyzer sections are optional — the server degrades gracefully when paths are `null`.

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

## How it works

This is a thin wrapper around the LatentMAS mechanism. Each `latent_think` call:
1. Encodes the prompt through the model
2. Runs N steps of latent rollout (hidden state → W_a alignment → forward pass → next hidden state)
3. Stores the resulting KV-cache under a handle

Agents can inherit previous handles, so context accumulates. `latent_collaborate` takes the accumulated KV-cache and generates text from it.

The alignment matrix W_a maps output hidden states back to input embedding space. It's computed once at startup via ridge regression on the embedding matrices.

ERIS extends this with:
- **Encode** — expose raw hidden states per layer for external inspection
- **Analyze** — run Sparse Autoencoder decomposition, Â-hat agentivity probes, cosine similarity maps, and PCA projections on any stored thought
- **Inject** — surgically modify hidden embeddings before the next rollout
- **Bridge** — orchestrate the full Claude ↔ Zombie pipeline, extracting implicit features activated in latent space but absent from the surface text

For details on LatentMAS, see the [paper](https://arxiv.org/abs/2511.20639).

## Compatible models

Tested with Qwen3-4B and Qwen3-8B. ERIS v5 targets Qwen3-14B/32B. Should work with any HuggingFace model that uses standard KV-cache (`past_key_values`) and supports `output_hidden_states=True`. A patch for DeepSeek-V2 MLA models is included in `patches/`.

Models with non-standard attention (Qwen3.5's Gated DeltaNet, GLM-5's sparse attention) are not compatible.

## Project structure

```
engine.py                  Core engine (sessions, W_a, encode, think, collaborate)
server.py                  FastAPI REST server (base endpoints)
eris_server.py             ERIS v5 FastAPI server (superset of server.py)
eris_client.py             Python client — ERISClient + ClaudeZombieBridge
mcp_server.py              MCP tool server
test_e2e.py                End-to-end test (base engine)
test_bridge.py             ERIS v5 unit tests (68 tests)

eris/
  config.py                ERIS config loader (eris_config.yaml)
  trajectory.py            Trajectory tracker for latent rollout
  analyzers.py             SAEAnalyzer, AHatAnalyzer, CosineMapAnalyzer, PCAAnalyzer, NormAnalyzer
  injector.py              Hidden-state injection (add / steer / replace)
  implicit_features.py     Implicit feature detection (SAE activations vs surface tokens)
  bridge.py                encode → analyze → think → decode pipeline

configs/
  eris_config.yaml         Config template (null paths = graceful degradation)
  concept_vectors/         Reference concept vectors (.pt or .npy)

openclaw_compat/
  openai_proxy.py          OpenAI-compatible proxy for OpenClaw
  test_01_openai_compat.py API format tests
  test_02_openclaw_config.py OpenClaw config simulation
  test_03_openclaw_rl_compat.py OpenClaw-RL rollout compatibility
  VALIDATION_LOG.md        Live test results

patches/                   DeepSeek MLA adapter + quantization patches
phase0/                    Feasibility validation suite
.github/workflows/         CI pipeline (tests + ai-rsk security gate)
```

## Tests

```bash
# ERIS v5 unit tests (no GPU needed)
pytest test_bridge.py -v
# 68 passed

# Base engine test (requires GPU + model)
python test_e2e.py --model Qwen/Qwen3-4B

# OpenClaw compatibility tests (needs proxy running on :30000)
LATENT_MODEL=Qwen/Qwen3-4B python openclaw_compat/openai_proxy.py --port 30000 &
sleep 30
python openclaw_compat/test_01_openai_compat.py --port 30000
python openclaw_compat/test_02_openclaw_config.py --port 30000
python openclaw_compat/test_03_openclaw_rl_compat.py --port 30000

# Phase 0 feasibility suite
cd phase0 && python run_phase0.py --model Qwen/Qwen3-4B
```

## Security

All code is validated with **[ai-rsk](https://github.com/Krigsexe/ai-rsk)** — a security gate for AI-generated code with three detection layers (static rules, Semgrep + Gitleaks + osv-scanner, project analysis).

```
ai-rsk scan
# Result: PASS — run ai-rsk scan to see current score

```

Enforced rules: no `pickle.load`, `weights_only=True` on all `torch.load` calls, no wildcard CORS, no hardcoded secrets, all dependencies pinned above known CVEs, 50 MB body limit on ERIS endpoints.

The CI pipeline (`.github/workflows/test.yml`) runs `ai-rsk scan` as a gate before tests — the build fails if any BLOCK-level finding is introduced.

## Limitations

- Requires all agents to use the same model (same architecture, same weights)
- KV-caches live in GPU memory — limited by VRAM
- Trajectory storage is CPU-side (hidden states copied at each step)
- The `collaborate` step re-encodes the final prompt (no inherited KV-cache in `generate` due to position encoding issues with some models)
- Small models (4B) struggle with OpenClaw's large system prompts (~58KB)
- SAE and Â-hat analyzers require pre-trained checkpoints not included in this repo
- Only tested on math reasoning tasks (GSM8K) for the base LatentMAS layer

## Acknowledgments

All the core LatentMAS ideas are from Zou, Yang, Qiu et al. (Princeton, UIUC, Stanford). This project wraps their work and extends it with the ERIS v5 cognitive bridge layer.

Security hardening made possible by **[ai-rsk](https://github.com/Krigsexe/ai-rsk)** ([@Krigsexe](https://github.com/Krigsexe)) — an excellent security gate for AI-generated code that caught real CVEs in our dependency chain and enforced safe serialization patterns throughout the codebase. Highly recommended for any ML research project shipping AI-written code.

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
