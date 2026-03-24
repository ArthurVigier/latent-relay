# MIGRATION NOTES — Phase 1 → Phase 2 Reframe

**Status:** Logged, not yet deleted. Confirm before removing any item below.

## Paradigm change

```
ANCIEN : Claude ↔ Zombie (dialogue bidirectionnel, enrichissement textuel)
NOUVEAU : Claude → DriftDetector → LatentProbe (tool pur, pas de génération)
```

The zombie is no longer a participant. It is a pure representation tool.
`LatentProbe.probe()` returns numpy activations. No text generation. No web.

---

## References to delete (pending confirmation)

### `eris_server.py`

| Line | Symbol | Reason |
|---|---|---|
| 13 | `POST /v1/bridge` in docstring | Bridge endpoint replaced by `/v1/probe` |
| 58 | `from eris.bridge import ERISBridge` | Bridge class no longer needed at server level |
| 65 | `_eris_bridge` global | Replaced by `_eris_probe` (LatentProbe) |
| 136–151 | `eris_startup` bridge init block | Bridge construction removed |
| 162–165 | `_require_bridge()` | No longer used |
| 475–570 | `BridgeRequest`, `v1_bridge` | Replaced by `/v1/probe` endpoint |

### `server.py`

| Line | Symbol | Reason |
|---|---|---|
| 11 | `POST /collaborate` in docstring | Text generation from zombie, obsolete |
| 135–165 | `CollaborateRequest`, `CollaborateResponse`, `collaborate()` | Pure text gen, no role in new paradigm |

### `eris/bridge.py`

Entire file is Phase 1 code. In the new paradigm:
- `ERISBridge` — replaced by `ERISOrchestrator` + `LatentProbe`
- `safe_enrichment`, `strip_think` — no longer needed (no text output from probe)
- `BridgeResult` — replaced by `OrchestratorResult`

**Do not delete bridge.py** until test_0 kill criterion passes. Keep as reference.

### `eris/config.py`

| Symbol | Reason |
|---|---|
| `GenerationConfig` | Was for `enable_thinking=False` on zombie text gen — no longer relevant since probe never generates |

### `eval/eval_phase1.py`

| Function/Class | Status |
|---|---|
| `ERISClient`, `decode_b64`, `extract_vector`, `pool`, `cosine`, `load_stsb_pairs` | **SURVIT** — used in test_0 |
| `M4Result`, `eval_m4` | **SURVIT** — Phase 1 baseline |
| `M5Result`, `eval_m5` | **SURVIT** — displacement metric feeds drift_detector design |
| `M6Result`, `eval_m6` | **SURVIT** — SAE pipeline unchanged |
| `ABCResult`, `eval_abc` | **OBSOLÈTE** — dialogue paradigm, no equivalent in new design |
| `SteeringResult`, `eval_steering` | **OBSOLÈTE** — concept steering in bridge context |
| `DialogueResult`, `eval_dialogue` | **OBSOLÈTE** — dialogue paradigm |
| `WebDialogueResult`, `eval_webdialogue` | **OBSOLÈTE** — adversarial finding documented in README, no follow-up planned |
| `SteerDialogueResult`, `eval_steerdialogue` | **OBSOLÈTE** |
| `FrontierResult`, `eval_frontier` | **OBSOLÈTE** — was testing Claude vs Claude+zombie text quality |
| `web_search`, `web_search_multi` | **OBSOLÈTE** — no web access in probe paradigm |
| `run_steered_dialogue` | **OBSOLÈTE** |

---

## String patterns to grep before final cleanup

```bash
grep -rn "webdialogue\|zombie_turn\|enrichissement\|enriched_text\|safe_enrichment\|BridgeRequest\|_require_bridge\|ERISBridge" .
```

---

## ERIS V2 Reframe — SAE Features (completed)

### Paradigme

```
V1 : activations brutes numpy [hidden_dim] → Claude devine dans le vide
V2 : activations → SAE.encode() → features sparse [16K] → diff de sets
     → Claude lit des concepts (indices Neuronpedia)
```

### Stack zombie V2

| Composant | V1 | V2 |
|---|---|---|
| Modèle probe | Qwen3-14B (LatentProbe) | Gemma 3 9B/27B (SAEProbe) |
| Encodage | activations brutes numpy | features SAE Gemma Scope 2 |
| Drift metric | cosine + LLC (LLC = KL-div top-k) | Jaccard features + cosine brut |
| Release SAE | N/A | `gemma-scope-2-9b-it-res` |
| Format SAE ID | N/A | `layer_{n}_width_16k_l0_medium` |

### Nouveaux fichiers

| Fichier | Description |
|---|---|
| `scripts/validate_sae_on_aime.py` | Kill gate 0 — SAEs utiles sur AIME ? |
| `eris/sae_probe.py` | `SAEProbe` — Gemma 3 + Gemma Scope 2 |

### Fichiers remplacés (V1 → V2)

| Fichier | Changement |
|---|---|
| `eris/drift_detector.py` | `DriftReport` V2 avec `features_lost`, `features_gained`, Jaccard. Import `ProbeOutput` depuis `sae_probe` à la place de `interfaces.py` |
| `eris/orchestrator.py` | `ERISOrchestrator` utilise `SAEProbe` + `OrchestratorLLM`. Template `_RECALIBRATION_TEMPLATE` orienté features SAE |
| `eris/experiments/drift_detection/kill_criteria.py` | Ajout `sae_validation` + `test_3_scaling`. Check `range` pour sae_validation |
| `eris/experiments/drift_detection/test_0_drift_characterization.py` | V2 avec `SAEProbe`. Mode `server` (Qwen3 via ERIS) ou `direct` (pipeline test) |

### Endpoint ajouté

| Endpoint | Description |
|---|---|
| `POST /v1/sae_probe` | Features SAE par layer — lazy-load SAEProbe à la première requête |

### Compatibilité V1

`eris/interfaces.py` — `DriftReport` V1 intact (utilisé par backends ClaudeOrchestrator, etc.)
`eris/probe.py` — `LatentProbe` intact (V1 baseline)
`eris/backends/` — tous les backends V1 intacts

### Ordre d'exécution

```bash
pip install sae-lens transformer-lens>=3.0.0b0
python scripts/validate_sae_on_aime.py        # KILL GATE — exit 0 = OK
python eris/experiments/drift_detection/test_0_drift_characterization.py --mode server
# Si ρ ≥ 0.35 → créer test_1.py
```

---

## Correction 1 — Modularité (completed)

### New files

| File | Description |
|---|---|
| `eris/interfaces.py` | Abstract base classes `OrchestratorLLM`, `ProbeModel`; canonical `DriftReport`, `ReasoningStep`, `RecalibrationNote` |
| `eris/backends/orchestrators/claude_orchestrator.py` | `ClaudeOrchestrator` — full Anthropic API implementation |
| `eris/backends/orchestrators/gemini_orchestrator.py` | Stub |
| `eris/backends/orchestrators/openai_orchestrator.py` | Stub |
| `eris/backends/probes/hf_probe.py` | `HFProbe` — full HuggingFace implementation with `steer()` via forward hooks |
| `eris/backends/probes/vllm_probe.py` | Stub |
| `eris/factory.py` | `create_orchestrator()`, `create_probe()`, `create_coordinator()` |
| `eris/multi_agent.py` | `MultiAgentCoordinator` — ISOLATED / SHARED_MEDIUM / COLLABORATIVE modes |
| `eris/experiments/multi_agent/` | Kill-gate tests MA-0 (full), MA-1 (stub), MA-2 (stub) |

### Modified files

| File | Change |
|---|---|
| `eris/probe.py` | `LatentProbe` now inherits `ProbeModel` by delegation to `HFProbe`; adds `steer()`, `steer_batch()`, steering library |
| `eris/orchestrator.py` | `ERISOrchestrator` now takes `OrchestratorLLM` instead of raw `anthropic.Anthropic` |
| `eris/drift_detector.py` | `DriftReport` imported from `eris.interfaces` (removed local definition) |
| `configs/eris_config.yaml` | Added `backends:` and `multi_agent:` sections |

### Backward compatibility

`LatentProbe` still works with existing callers — signature unchanged.
`ERISOrchestrator` constructor signature changed: `claude_client` replaced by `llm: OrchestratorLLM`.
Old callers that passed an `anthropic.Anthropic` instance must be updated to pass `ClaudeOrchestrator()`.

---

*Do not act on this file without explicit confirmation. This is a log, not a TODO.*
