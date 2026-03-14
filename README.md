# Latent Relay

An MCP server that wraps [LatentMAS](https://arxiv.org/abs/2511.20639) (Zou et al., 2025) so that AI agents can use latent-space multi-agent reasoning as a tool.

## What this is

LatentMAS is a research framework where LLM agents collaborate by passing KV-caches instead of text. It works well but exists as a benchmark script tied to specific models and evaluation tasks.

This project repackages that mechanism as:
- A standalone **Python engine** with a simple API
- A **FastAPI server** with REST endpoints
- An **MCP server** with tool definitions that agents can discover and call

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

Agents can inherit previous handles, so context accumulates. `latent_collaborate` takes the accumulated KV-cache and generates text from it.

The alignment matrix W_a maps output hidden states back to input embedding space. It's computed once at startup via ridge regression on the embedding matrices.

For details, see the [LatentMAS paper](https://arxiv.org/abs/2511.20639).

## Compatible models

Tested with Qwen3-4B. Should work with any HuggingFace model that uses standard KV-cache (`past_key_values`). A patch for DeepSeek-V2 MLA models is included in `patches/` — HF's MLA implementation caches full K,V after up-projection, so the interface is the same.

Models with non-standard attention (Qwen3.5's Gated DeltaNet, GLM-5's sparse attention) are not compatible.

## Project structure

```
engine.py          Core engine
server.py          FastAPI REST server
mcp_server.py      MCP tool server
test_e2e.py        End-to-end test
patches/           DeepSeek MLA adapter + quantization patches for LatentMAS repo
phase0/            Feasibility validation suite (serialization, quantization, W_a stability)
```

## Limitations

- Requires all agents to use the same model (same architecture, same weights)
- KV-caches live in GPU memory — limited by VRAM
- No interpretability of intermediate latent reasoning
- The `collaborate` step re-encodes the final prompt (doesn't pass inherited KV-cache to `generate` due to position encoding issues with some models)
- Only tested on math reasoning tasks (GSM8K)

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
