# ERIS ML Oracle — Oracle ML Prediction Spinoff

Decentralized ML Oracle for Prediction Markets. Combines ERIS's mechanistic interpretability pipeline with on-chain oracle infrastructure.

## Architecture

```
Polymarket API
     │
     ▼
[PolymarketOracleFeed] ───→ Market Snapshots & Feature Vectors
     │
     ▼
[PredictionWorldModel] ──→ JEPA-style outcome predictions
     │   (lightweight or full JEPA)
     ▼
[MultiAgentOracleVerifier] → 5 specialized agents + kill-gate
     │   (uses existing LatentMAS / SAE probe infrastructure)
     ▼
[SAEDriftGate] ───────────→ Drift detection & gate control
     │   (feature-level anomaly detection)
     ▼
[BittensorOracleMiner] ───→ Subnet-compatible submissions
     │   (targets: SN3, SN4, SN86, SN125)
     ▼
[On-Chain Oracle Output]
```

## Components

- `feeds/polymarket_feed.py` — Live Polymarket data ingestion via public APIs
- `world_model/predictor.py` — JEPA-style world model for event outcome prediction
- `verifier/multi_agent_verifier.py` — Multi-agent consensus with SAE drift-based kill-gates
- `verifier/sae_drift_gate.py` — SAE-based drift detection monitoring the entire pipeline
- `miner.py` — Bittensor subnet-compatible mining/validator formatting
- `oracle_engine.py` — Main orchestrator tying all components together

## Key Features

1. **Live Market Data**: Fetches real-time price, volume, spread, and liquidity from Polymarket
2. **JEPA World Model**: Predicts future event outcomes using context-aware latent prediction (lightweight mode for CPU, full JEPA mode for GPU)
3. **Multi-Agent Consensus**: 5 specialized oracle agents (bullish, bearish, neutral, momentum, fundamental) with SAE-based drift monitoring
4. **SAE Drift Gate**: Feature-level drift detection that kills bad agents and refuses corrupt oracle data
5. **Bittensor Subnet Integration**: Formatted output for SN3 (taumplar), SN4 (Targon), SN86 (Investing), and SN125 (8 Ball)

## Target Subnets (March 2026)

| SN# | Name | Emission | Best Use |
|-----|------|----------|----------|
| 3 | taumplar | 7.58% | LLM oracle / text generation |
| 4 | Targon | 6.29% | Multimodal ML oracle |
| 11 | TrajectoryRL | 1.30% | RL-optimized predictions |
| 86 | Investing | 0.64% | Financial prediction oracle |
| 125 | 8 Ball | emerging | Prediction market oracle |

## Usage

```python
from oracle_ml_prediction import MLOracleEngine

engine = MLOracleEngine(n_agents=5, drift_threshold=0.5)

# Predict a specific event
result = engine.predict("Will the Fed cut rates in Q1 2026?")
print(f"P(Yes): {result.final_probability*100:.1f}%")
print(f"Confidence: {result.final_confidence*100:.0f}%")
print(f"Gate: {result.pipeline_passed}")

# Submit to Bittensor subnet
submission = engine.submit_to_subnet(result, netuid=86)
print(submission.response_text)
```

## Demo

```bash
cd spin_offs/oracle_ml_prediction
python examples/demo_live.py
```

## Design Principles

- **SAE-based transparency**: Every oracle output includes drift analysis with concept-level feature attribution
- **Multi-agent robustness**: Kill-gates terminate agents showing latent drift (integrates with existing ERIS multi-agent infrastructure)
- **Graceful degradation**: Works in lightweight mode without GPU; scales to full JEPA when available
- **Subnet-compatible**: Output format designed for direct submission to Bittensor subnets
