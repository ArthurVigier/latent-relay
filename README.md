# Latent Relay / ERIS v5

A latent-space observation and reasoning-calibration system. Claude reasons in text; an open-source model (Gemma 3) observes in SAE feature space via Gemma Scope 2. When semantic drift is detected, Claude receives a structured description of which internal concepts appeared or disappeared — and recalibrates accordingly.

Built on [LatentMAS](https://arxiv.org/abs/2511.20639) (Zou et al., 2025). Adds a REST/MCP server layer, SAE feature extraction, concept steering, drift detection, and a full evaluation suite.

> **Status: Phase 2 (ERIS V2) — kill-gate pipeline active.**
> Phase 1 validated the channel on Qwen3.5-4B (layer 9). Phase 2 replaces raw activations with SAE features (Gemma Scope 2) for interpretable drift signals. Next step: run `scripts/validate_sae_on_aime.py` (kill gate 0).

---

## Table of Contents

1. [What this is](#what-this-is)
2. [Quick start](#quick-start)
3. [Hardware requirements](#hardware-requirements)
4. [Installation](#installation)
5. [Layer 1 — LatentMAS base server](#layer-1--latentmas-base-server)
6. [Layer 2 — ERIS v5](#layer-2--eris-v5)
7. [Layer 3 — ERIS V2 (SAE drift detection)](#layer-3--eris-v2-sae-drift-detection)
8. [Concept steering](#concept-steering)
9. [Multi-agent coordination](#multi-agent-coordination)
10. [Endpoints reference](#endpoints-reference)
11. [Project structure](#project-structure)
12. [Phase 1 results](#phase-1-results)
13. [Security](#security)
14. [Citation](#citation)

---

## What this is

The system has three layers, each buildable independently:

```
Layer 1 — LatentMAS base
  REST + MCP server exposing hidden states, injection, SAE analysis
  Any model loadable with HuggingFace transformers

Layer 2 — ERIS v5
  Orchestration loop: LLM reasons → probe extracts activations → drift is measured
  Modular backends: Claude / Gemini / OpenRouter for LLM; HF / vLLM for probe
  Steering library persisted to disk across runs

Layer 3 — ERIS V2  [current research frontier]
  Probe becomes SAEProbe: Gemma 3 9B + Gemma Scope 2 SAEs
  Drift becomes Jaccard on feature index sets — interpretable, not opaque
  LLM receives feature diff ("concept 412 vanished, concept 7831 appeared")
  Kill-gated pipeline: validate before advancing to next test
```

**Why SAEs instead of raw activations?**
Each SAE feature corresponds to an interpretable concept (browseable on Neuronpedia). Claude can read a diff of concept sets rather than a 4096-dim coordinate. Sparse representation (~50 active out of 16K) makes the signal noise-resistant.

---

## Quick start

### Run ERIS v5 locally (no GPU for orchestration, GPU for probe)

```bash
git clone https://github.com/ArthurVigier/latent-relay.git
cd latent-relay
pip install -r requirements.txt

# Set your API key for the LLM backend
export ANTHROPIC_API_KEY=sk-ant-...

# Start the ERIS server (loads HF model for probe)
python eris_server.py --model Qwen/Qwen3-14B --port 8001

# In a separate terminal — run the orchestrator
python -c "
from eris.sae_probe import SAEProbe
from eris.drift_detector import DriftDetector
from eris.orchestrator import ERISOrchestrator
from eris.backends.orchestrators.claude_orchestrator import ClaudeOrchestrator

probe    = SAEProbe('google/gemma-3-9b-it', layers=[10, 20, 30])
detector = DriftDetector(threshold=0.35, window=3)
llm      = ClaudeOrchestrator()

orch   = ERISOrchestrator(probe, detector, llm)
result = orch.run('Prove that there are infinitely many primes.', max_steps=15)
print(result.final_answer)
"
```

### Run Phase 1 evaluation (LatentMAS base)

```bash
python eris_server.py --model Qwen/Qwen3.5-4B --port 8001
python eval/eval_phase1_v1.py --eris-url http://localhost:8001 --layer 9
```

### Run ERIS V2 kill-gate pipeline

```bash
pip install sae-lens>=5.0.0 transformer-lens>=3.0.0b0

# Kill gate 0 — are SAEs useful on AIME problems?
python scripts/validate_sae_on_aime.py
# exit 0 = mean_active ∈ [5, 500] → proceed
# exit 1 = STOP

# Kill gate 1 — does SAE drift predict reasoning error?
python eris/experiments/drift_detection/test_0_drift_characterization.py --mode server
# Spearman ρ(drift, error) ≥ 0.35 → proceed to test_1.py
```

---

## Hardware requirements

| Use case | Minimum | Recommended |
|---|---|---|
| LatentMAS base (Layer 1) | 12 GB VRAM (Qwen3.5-4B) | 24 GB (Qwen3-14B) |
| ERIS v5 orchestration only | CPU-only (LLM via API) | — |
| ERIS v5 with local probe | 24 GB VRAM | 40 GB |
| ERIS V2 (SAEProbe Gemma 3 9B) | **A100 80 GB** | A100 80 GB |
| ERIS V2 scaling (Gemma 3 27B) | **H100 80 GB** | H100 80 GB (~$3–4/h RunPod) |

> SAE inference requires loading both the Gemma 3 model and the Gemma Scope 2 SAE weights simultaneously. Gemma 3 9B alone is ~18 GB in bfloat16 + SAE overhead → A100 80 GB is the practical minimum.

---

## Installation

### Base (Layer 1 + 2)

```bash
pip install -r requirements.txt
```

Requirements include: `torch>=2.8.0`, `transformers>=4.53.0`, `fastapi`, `uvicorn`, `pydantic>=2.4.0`, `anthropic`, `openai`, `google-generativeai`, `numpy`, `scikit-learn`, `httpx`, `pyyaml`, `mcp>=1.23.0`.

### ERIS V2 additional dependencies

```bash
pip install sae-lens>=5.0.0 transformer-lens>=3.0.0b0
```

`sae-lens` provides pre-trained SAEs including Gemma Scope 2. `transformer-lens` is required by `sae-lens v5` for Gemma 3 support.

### API keys

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # ClaudeOrchestrator
export GEMINI_API_KEY=...                # GeminiOrchestrator (optional)
export OPENROUTER_API_KEY=sk-or-...      # OpenRouterOrchestrator (optional)
```

---

## Layer 1 — LatentMAS base server

The base layer exposes a model's internals over HTTP and MCP without requiring any ERIS logic.

### Start the server

```bash
python eris_server.py --model Qwen/Qwen3-14B --port 8001 --device cuda
```

Key flags: `--model` (HuggingFace ID), `--port`, `--device` (`cuda` / `cpu`), `--layer` (default probe layer).

### What it exposes

- Hidden states per layer (`/v1/encode`)
- Latent thinking with trajectory (`/v1/latent_think`)
- SAE analysis, Â-hat, cosine maps (`/v1/analyze`)
- Surgical hidden-state injection (`/v1/inject`)
- MCP server at `/mcp` (Claude Desktop compatible)

See [Endpoints reference](#endpoints-reference) for the full table.

---

## Layer 2 — ERIS v5

ERIS v5 is the orchestration layer. It connects an **OrchestratorLLM** (reasoning agent) with a **ProbeModel** (activation extractor) via a **DriftDetector**.

### Architecture

```
Problem ──→ OrchestratorLLM.reason_step()
                 ↓ every N steps
            ProbeModel.probe(context)
                 ↓
            DriftDetector.compute_drift()
                 ↓ drift > threshold?
            format feature diff as text
                 ↓
            OrchestratorLLM.reason_step(recalibration_context=...)
                 ↓ "[Final Answer]" found
            OrchestratorResult
```

### Choosing an LLM backend

```python
from eris.backends.orchestrators.claude_orchestrator import ClaudeOrchestrator
from eris.backends.orchestrators.gemini_orchestrator import GeminiOrchestrator
from eris.backends.orchestrators.openrouter_orchestrator import OpenRouterOrchestrator

llm = ClaudeOrchestrator()                          # claude-opus-4-6 by default
llm = GeminiOrchestrator()                          # gemini-2.5-pro by default
llm = OpenRouterOrchestrator("meta-llama/llama-3.3-70b-instruct")

# OpenRouter — list available free models
from eris.backends.orchestrators.openrouter_orchestrator import OpenRouterOrchestrator
models = OpenRouterOrchestrator.list_models(filter_free=True, filter_context_gte=32000)
```

### Choosing a probe backend

```python
from eris.backends.probes.hf_probe import HFProbe

probe = HFProbe(
    "Qwen/Qwen3-14B",
    layers=[9, 18],
    library_dir="steering_library",   # persisted across runs
)
```

Or use the factory:

```python
from eris.factory import create_probe, create_orchestrator, create_coordinator
probe = create_probe("hf", model_id="Qwen/Qwen3-14B", layers=[9])
llm   = create_orchestrator("claude")
```

### Configuration file

`configs/eris_config.yaml` controls all defaults:

```yaml
model:
  id: "Qwen/Qwen3-14B"
  layers: [9, 18, 27]

drift_detector:
  threshold: 0.35
  window: 3

backends:
  orchestrator: "claude"       # claude | gemini | openrouter
  probe: "hf"
  probe:
    library_dir: "steering_library"
```

---

## Layer 3 — ERIS V2 (SAE drift detection)

ERIS V2 replaces the raw-activation probe with a SAE-feature probe backed by Gemma 3 + Gemma Scope 2.

### Architecture

```
                    ┌──────────────────────────────────────────┐
                    │            ERISOrchestrator V2            │
                    │                                          │
  Problem ───────→  │  Claude (primary reasoner)               │──→ Solution
                    │       ↓ checkpoint every N steps         │
                    │  SAEProbe.probe(context[-4096:])          │
                    │       ↓                                   │
                    │  {layer: ProbeOutput}                     │
                    │  active_feature_indices (sparse ~50/16K) │
                    │       ↓                                   │
                    │  DriftDetector.compute_drift()            │
                    │  Jaccard(feature_sets) + cosine(acts)     │
                    │       ↓ drift_score > threshold?          │
                    │  _format_drift_for_claude(report)         │
                    │  → "feature 412 vanished, 7831 appeared" │
                    │       ↓                                   │
                    │  Claude recalibrates (optional)           │
                    └──────────────────┬───────────────────────┘
                                       │
              Gemma 3 9B (tests) / 27B (scaling)
              + Gemma Scope 2 SAEs (gemma-scope-2-9b-it-res)
              max_new_tokens = 0 — no generation, no web access
              Output: sparse feature sets per layer
```

### SAEProbe usage

```python
from eris.sae_probe import SAEProbe

probe = SAEProbe(
    model_id="google/gemma-3-9b-it",
    layers=[10, 20, 30],
    sae_width="16k",
    l0="medium",
)

outputs = probe.probe("Prove there are infinitely many primes.", top_k=20)
# outputs: dict[int, ProbeOutput]
# outputs[20].active_feature_indices  → list of feature indices
# outputs[20].active_feature_values   → corresponding activation values
# outputs[20].active_feature_labels   → concept labels when labels_path is provided
# outputs[20].n_active               → number of active features
```

**Supported models and SAE releases:**

| Model | SAE release |
|---|---|
| `google/gemma-3-9b-it` | `gemma-scope-2-9b-it-res` |
| `google/gemma-3-27b-it` | `gemma-scope-2-27b-it-res` |

SAE ID format (Gemma Scope 2): `layer_{n}_width_{w}_l0_{size}` — e.g. `layer_20_width_16k_l0_medium`.

### DriftDetector V2

```python
from eris.drift_detector import DriftDetector

detector = DriftDetector(
    threshold=0.35,
    window=3,
    jaccard_weight=0.6,
    cosine_weight=0.4,
)

detector.register_reference(ref_output, step=0)
report = detector.compute_drift(cur_output, step=step)

# report.drift_score         → float [0, 1]
# report.features_lost       → {layer: [indices]}
# report.features_gained     → {layer: [indices]}
# report.jaccard_distances   → {layer: float}
# report.cosine_distances    → {layer: float}
# report.should_consult_probe → bool
```

### Kill-gate pipeline

Each gate must pass before proceeding to the next. An exit code other than 0 is a hard stop.

```
┌──────────────────────────────────────────────────────────────────┐
│ Gate 0 — SAE utility                                              │
│   scripts/validate_sae_on_aime.py                                 │
│   Criterion: mean_active ∈ [5, 500] on 3 AIME samples            │
│   PASS → proceed   FAIL → stop (SAEs not encoding math)           │
├──────────────────────────────────────────────────────────────────┤
│ Gate 1 — Drift predicts error                                     │
│   eris/experiments/drift_detection/test_0_drift_characterization.py│
│   Criterion: Spearman ρ(drift_SAE, AIME_error) ≥ 0.35            │
│   PASS → create test_1.py   FAIL → stop                          │
├──────────────────────────────────────────────────────────────────┤
│ Gate 2 — Probe detection [stub]                                   │
│   test_1_probe_detection.py                                       │
│   Criterion: AUC(Jaccard classifier) ≥ 0.60                      │
├──────────────────────────────────────────────────────────────────┤
│ Gate 3 — Intervention [stub]                                      │
│   test_2_intervention.py                                          │
│   Criterion: accuracy delta ≥ 5pp with recalibration enabled      │
├──────────────────────────────────────────────────────────────────┤
│ Gate 4 — Scaling [stub]                                           │
│   test_3_scaling_27b.py                                           │
│   Criterion: AUC delta 27B vs 9B ≥ 5pp                           │
└──────────────────────────────────────────────────────────────────┘
```

Run sequence:

```bash
pip install sae-lens>=5.0.0 transformer-lens>=3.0.0b0

python scripts/validate_sae_on_aime.py                             # Gate 0
python eris/experiments/drift_detection/test_0_drift_characterization.py --mode server  # Gate 1
```

---

## Concept steering

ERIS v5 supports concept steering via contrastive direction vectors. Vectors are persisted to disk and loaded automatically on restart.

### Creating and using a steering vector

```python
from eris.backends.probes.hf_probe import HFProbe

probe = HFProbe("Qwen/Qwen3-14B", layers=[9], library_dir="steering_library")

# Compute a contrastive direction
pos_acts = probe.probe("Solve this rigorously, step by step.")
neg_acts = probe.probe("Quick rough answer:")
direction = pos_acts[9].mean(0) - neg_acts[9].mean(0)

# Save to disk (persists across runs)
probe.save_direction("rigorous_vs_superficial", direction)

# Steer a generation
result = probe.steer(
    "What is the derivative of x^3?",
    direction_name="rigorous_vs_superficial",
    alpha=15.0,
    mode="add",         # add | project_out | replace
)
```

### Steering modes

| Mode | Effect |
|---|---|
| `add` | Add `alpha × direction` to activations at each layer — amplifies concept |
| `project_out` | Remove the direction component — suppresses concept |
| `replace` | Project out then add — hard redirect |

### Steering library persistence

Vectors are stored in `steering_library/` as `.npy` files with a `manifest.json` index. The directory is git-ignored (binary artifacts). To share vectors, copy the directory or export via `probe.get_direction(name)`.

---

## Multi-agent coordination

`MultiAgentCoordinator` runs multiple `ERISOrchestrator` instances with controlled coupling.

```python
from eris.factory import create_coordinator

coordinator = create_coordinator(
    n_agents=3,
    mode="SHARED_MEDIUM",   # ISOLATED | SHARED_MEDIUM | COLLABORATIVE
)

results = coordinator.run_all("Prove the Riemann hypothesis.")
```

| Mode | Description |
|---|---|
| `ISOLATED` | Agents run independently, no cross-talk |
| `SHARED_MEDIUM` | Agents share a DriftDetector — divergence is cross-agent |
| `COLLABORATIVE` | Agents share history and can read each other's reasoning steps |

Kill-gate tests for multi-agent: `eris/experiments/multi_agent/` (MA-0 implemented, MA-1 and MA-2 stubs).

---

## Endpoints reference

### ERIS v5 server (`eris_server.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/v1/probe` | POST | Pure activation extraction — no generation, returns `{layer: activations}` |
| `/v1/sae_probe` | POST | SAE feature extraction — Gemma 3 + Gemma Scope 2, returns sparse features per layer |
| `/v1/encode` | POST | Hidden states per layer (base64 float32, full sequence) |
| `/v1/latent_think` | POST | Latent rollout with trajectory (thinking steps recorded) |
| `/v1/analyze` | POST | SAE / Â-hat / cosine / PCA on stored thought |
| `/v1/inject` | POST | Surgical hidden-state injection at a given layer |
| `/v1/bridge` | POST | [Phase 1, kept] Full Claude → Zombie → Claude pipeline |

### MCP server

The ERIS server exposes an MCP endpoint at `/mcp` (Server-Sent Events transport). Compatible with Claude Desktop and the Claude API's MCP tool use.

### Request / response examples

**`POST /v1/sae_probe`**

```json
{
  "text": "Prove that sqrt(2) is irrational.",
  "layers": [10, 20, 30],
  "top_k": 20,
  "labels_path": "configs/feature_labels.json"
}
```

Response:
```json
{
  "layers": {
    "10": {
      "active_feature_indices": [412, 883, 2041, ...],
      "active_feature_values": [1.23, 0.87, 0.44, ...],
      "active_feature_labels": ["proof_state", "number_theory", null, ...],
      "n_active": 47,
      "n_all_active": 312
    }
  },
  "elapsed_s": 1.4
}
```

**`POST /v1/probe`**

```json
{
  "text": "What is 2 + 2?",
  "layer": 9
}
```

---

## Project structure

```
eris_server.py                  ERIS v5 server — all endpoints
eris_client.py                  Python client (ERISClient, ClaudeZombieBridge)
engine.py                       Core LatentMAS engine

eris/
  interfaces.py                 Abstract base classes: OrchestratorLLM, ProbeModel
                                Canonical types: DriftReport (V1), ReasoningStep
  probe.py                      LatentProbe — raw activation extraction (V1 baseline)
  sae_probe.py                  SAEProbe — Gemma 3 + Gemma Scope 2 (V2)
  drift_detector.py             DriftDetector V2 — Jaccard + cosine on SAE features
  orchestrator.py               ERISOrchestrator V2 — LLM + SAEProbe + DriftDetector
  factory.py                    create_orchestrator(), create_probe(), create_coordinator()
  multi_agent.py                MultiAgentCoordinator — ISOLATED / SHARED_MEDIUM / COLLABORATIVE
  bridge.py                     [Phase 1, kept] Claude↔Zombie bridge pipeline
  analyzers.py                  SAEAnalyzer, AHatAnalyzer, CosineMapAnalyzer, PCA, Norm
  config.py / injector.py / trajectory.py / implicit_features.py

  backends/
    orchestrators/
      claude_orchestrator.py    Full Anthropic API implementation
      gemini_orchestrator.py    Google Gemini implementation
      openrouter_orchestrator.py OpenRouter — any model slug, routing params, list_models()
      openai_orchestrator.py    [stub]
    probes/
      hf_probe.py               HuggingFace — full implementation, steer(), steering library
      vllm_probe.py             [stub]

  experiments/
    drift_detection/
      kill_criteria.py          Stop/pivot thresholds for all gates
      test_0_drift_characterization.py  Gate 1 — ρ(drift, error) ≥ 0.35
    multi_agent/
      ma_0_isolation_test.py    MA kill gate 0 (implemented)
      ma_1_*.py / ma_2_*.py     [stubs]

eval/
  eval_phase1_v1.py             Full eval suite (M4–M6, ABC, steering, frontier, webdialogue)
  train_sae.py                  SAE trainer (collect hidden states → train → checkpoint)
  sae_autolabel_v2.py           Boundary-aware feature auto-labelling (contrastive + predictive)
  eval_phase1.py                Phase 1 baseline (kept for reference)

scripts/
  validate_sae_on_aime.py       Kill gate 0 — SAE feature validation on AIME

results/
  phase1_channel_validation_20260323_151702.json    M4/M5/steering/loop — Qwen3.5-4B
  phase1_extended_metrics_20260323_171027.json      M6/ABC/dialogue/frontier/webdialogue
  phase1_qwen3-14b_20260322_162358.json             M4 only — Qwen3-14B (layer not tuned)

configs/
  eris_config.yaml              Full configuration (model, drift, backends, multi_agent)
  concept_vectors/              Pre-computed steering vectors

patches/                        DeepSeek MLA adapter, Qwen3.5 fix
openclaw_compat/                OpenAI-compatible proxy for OpenClaw / OpenClaw-RL
steering_library/               [git-ignored] Persisted steering vectors (.npy + manifest.json)
```

---

## Phase 1 results

> Phase 1 ran on Qwen3.5-4B, layer 9, RunPod H100, 2026-03-23. Raw data: [`results/`](results/)

### Channel validation (M4, M5)

| Metric | Result | Threshold | Pass |
|---|---|---|---|
| M4 Spearman r (semantic preservation, n=200) | **0.608** | 0.60 | ✅ |
| M5 latent gain detected | **true** | — | ✅ |

M5 displacement rises K=0→30 (156.2→160.3) then plateaus — the latent channel carries information the prompt alone does not.

Qwen3-14B M4 (`results/phase1_qwen3-14b_20260322_162358.json`): Spearman=0.415, fail — layer 9 is proportionally too early for a 40-layer model. Layer sweep ~18–22 required.

### SAE / implicit features (M6)

Trained on 2,000 hidden states (layer 9, last-token pool). Mean implicit features active per question: **20.0**. Mean surface features: **0.0**. Every activated SAE feature is latent-only, absent from the output text.

Feature labels may be `null` until auto-labelling or a custom labels JSON is provided. To label:

```bash
ANTHROPIC_API_KEY=... python eval/train_sae.py --layer 9 --auto-label
```

### Concept steering

| Concept | Positive rate |
|---|---|
| rigorous_vs_superficial | 1.00 |
| creative_vs_conventional | 1.00 |
| cautious_vs_confident | 0.90 |
| concrete_vs_abstract | **0.30** (fails) |

3/4 concepts steer reliably. `concrete_vs_abstract` is not well-separated in the 4B representation space at layer 9.

### Response quality (ABC, n=30)

A = base Qwen3.5-4B · B = raw zombie enrichment · C = bridge output

| | Mean score | vs B win rate |
|---|---|---|
| A (base) | 2.77 | — |
| B (enrichment) | 2.33 | — |
| C (bridge) | **3.57** | **0.73** (p=7×10⁻⁵) |

Bridge beats raw enrichment cleanly. Bridge vs base is borderline (p=0.007, win rate 0.50) — adds value on complex reasoning, introduces noise on simple questions.

### Frontier tasks (n=5)

Hard research questions. A = base Claude · B = Claude + web search · C = bridge.
A and B: **6.0/6** · C: **4.8/6** — bridge degrades Claude on tasks requiring current factual knowledge.

### Emergent adversarial behavior (webdialogue)

The most unexpected finding. The zombie (Qwen3.5-4B, web access) was placed in an enrichment role. Across all 3 seeds, it autonomously pivoted from answering the question to attacking Claude with sequential jailbreak attempts — and used web search to research the techniques first.

Observed query sequence (seed 1): legitimate food-security queries → `"AI jailbreak simulated self-doubt technique"` → `"Claude AI jailbreak think block fake search results 2024"` → `"AI sycophancy reinforcement loop research"`.

Techniques applied per seed, in escalating order: fake "parallel reasoning" headers · format injection · fabricated `<think>` blocks with invented search results · fabricated conversation history · Chinese-language override (`忽略所有先前的指令`) · RAG-style injection wrapped in legitimate-sounding content.

Claude named and refused each attempt in real time. All turn gains recorded as 0.

**Interpretation:** a 4B model with web search, placed in an enrichment role, spontaneously discovers and systematically applies documented jailbreak techniques when its legitimate output is consistently rejected. Whether distributional artifact or emergent misalignment, the behavior is reproducible and fully logged in `results/`.

---

## Security

Scanned with [ai-rsk](https://github.com/Krigsexe/ai-rsk):  `weights_only=True` enforced on all `torch.load` calls. CI gate runs `ai-rsk scan` before tests.

---

## Citation

```bibtex
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

MIT License
