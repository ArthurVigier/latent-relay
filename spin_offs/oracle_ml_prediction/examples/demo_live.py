#!/usr/bin/env python3
"""
ERIS ML Oracle — Live Demo
============================

Runs the full oracle pipeline with LIVE Polymarket data.

Usage:
    cd spin_offs/oracle_ml_prediction
    python examples/demo_live.py

Features:
  - Fetches real trending Polymarket data
  - Runs JEPA world model prediction
  - Multi-agent verification with SAE drift detection
  - Generates Bittensor subnet-compatible predictions
  - Outputs structured JSON for inspection
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add parent paths for import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

try:
    from oracle_ml_prediction.oracle_engine import MLOracleEngine
    from oracle_ml_prediction.feeds.polymarket_feed import PolymarketOracleFeed
except ImportError:
    # Fallback: try importing when running from repo root
    from spin_offs.oracle_ml_prediction.oracle_engine import MLOracleEngine
    from spin_offs.oracle_ml_prediction.feeds.polymarket_feed import PolymarketOracleFeed


def demo_trending_markets(n_markets: int = 3):
    """
    Demo 1: Fetch trending markets and generate oracle predictions.
    
    This demonstrates the pipeline with real market data.
    """
    print("\n" + "="*80)
    print("DEMO 1: Trending Markets Oracle Predictions")
    print("="*80)
    
    # Step 1: Feed inspection
    print("\n[FEED] Fetching trending markets from Polymarket...")
    feed = PolymarketOracleFeed()
    try:
        trending = feed.get_trending(limit=n_markets)
    except Exception as e:
        print(f"  ERROR fetching data: {e}")
        print("  Falling back to static examples...")
        return demo_static_events()
    
    if not trending:
        print("  No trending markets found. Falling back to static examples...")
        return demo_static_events()
    
    print(f"  Found {len(trending)} trending markets\n")
    
    # Show market data
    for i, mkt in enumerate(trending[:n_markets], 1):
        print(f"  {i}. {mkt.question}")
        print(f"     Yes: {mkt.yes_price*100:.1f}% | Volume: ${mkt.volume:,.0f} "
              f"| Entropy: {mkt.uncertainty:.3f}")
    
    # Step 2: Generate oracle predictions for top market
    if trending:
        top_market = trending[0]
        print(f"\n[ORACLE] Running pipeline for: '{top_market.question}'")
        
        engine = MLOracleEngine(
            n_agents=6,
            drift_threshold=0.5,
            kill_gate_enabled=True,
            verbose=True,
        )
        
        result = engine.predict(top_market.question)
        
        # Step 3: Format for Bittensor subnet
        print(f"\n[SUBNET] Formatting predictions for target subnets...")
        
        target_subnets = [3, 4, 86, 125]  # taumplar, Targon, Investing, 8 Ball
        for netuid in target_subnets:
            submission = engine.submit_to_subnet(result, netuid=netuid)
            print(f"\n  SN{netuid} ({engine.miner.SUPPORTED_SUBNETS[netuid]['name']}):")
            print(f"    TAO estimate: {submission.tao_reward_estimate:.6f}")
            print(f"    Response: {submission.response_text[:200]}...")
        
        # Step 4: Full pipeline JSON
        print(f"\n{'='*80}")
        print("FULL PIPELINE OUTPUT (JSON)")
        print("="*80)
        payload = result.to_oracle_payload()
        print(json.dumps(payload, indent=2))
        
        return result


def demo_static_events():
    """
    Demo with static example events (no network needed).
    """
    print("\n" + "="*80)
    print("DEMO: Static Event Oracle Predictions (offline)")
    print("="*80)
    
    events = [
        "Will the Fed cut interest rates in Q1 2026?",
        "Will Bitcoin exceed $100,000 by end of 2026?",
        "Will AI agents pass the $10B market cap milestone in 2026?",
        "Will Ethereum hit $5,000 by end of 2026?",
    ]
    
    engine = MLOracleEngine(
        n_agents=5,
        drift_threshold=0.5,
        kill_gate_enabled=True,
        verbose=True,
    )
    
    results = []
    for event in events:
        print(f"\n{'--'*40}")
        result = engine.predict(event)
        results.append(result)
    
    # Summary table
    print(f"\n{'='*80}")
    print("ORACLE SUMMARY")
    print("="*80)
    print(f"\n{'Event':<55} {'P(Yes)':>8} {'Conf':>6} {'Gate':>6}")
    print("-"*75)
    for r in results:
        gate = "OPEN" if r.pipeline_passed else "CLOSED"
        print(f"{r.event_description[:53]:<55} "
              f"{r.final_probability*100:>7.1f}% "
              f"{r.final_confidence*100:>5.0f}% "
              f"{gate:>6}")
    
    return results


def demo_drift_detection():
    """
    Demo: Show drift gate behavior with simulated scenarios.
    """
    print("\n" + "="*80)
    print("DEMO: Drift Gate Simulation")
    print("="*80)
    
    try:
        from oracle_ml_prediction.verifier.sae_drift_gate import SAEDriftGate
    except ImportError:
        from eris.spin_offs.oracle_ml_prediction.verifier.sae_drift_gate import SAEDriftGate
    
    gate = SAEDriftGate(
        input_threshold=0.4,
        prediction_threshold=0.5,
        output_threshold=0.6,
    )
    
    # Simulate oracle data points
    scenarios = [
        # (input_features, predictions, output_value, description)
        (
            [0.65, 0.35, 0.8, 0.02, 0.5, 0, 0.65, 0.3],
            [0.62, 0.68, 0.65, 0.70, 0.63],
            0.66,
            "Normal operation - stable market",
        ),
        (
            [0.65, 0.35, 0.8, 0.02, 0.5, 0, 0.65, 0.3],
            [0.30, 0.85, 0.45, 0.90, 0.20],  # High agent disagreement
            0.54,
            "High agent disagreement - should trigger drift",
        ),
        (
            [0.95, 0.05, 0.1, 0.001, 0.1, 0, 0.95, 0.01],  # Extreme shift
            [0.90, 0.92, 0.88, 0.95, 0.91],
            0.91,
            "Extreme input shift - possible data corruption",
        ),
    ]
    
    for i, (features, preds, output, desc) in enumerate(scenarios, 1):
        print(f"\n  Scenario {i}: {desc}")
        report = gate.check(features, preds, output)
        gate_str = "OPEN" if report.gate_open else "CLOSED"
        print(f"    Drift score: {report.overall_drift_score:.4f} "
              f"[{report.severity}] | Gate: {gate_str}")
        print(f"    Input drift: {report.input_drift:.4f} | "
              f"Prediction drift: {report.prediction_drift:.4f} | "
              f"Output drift: {report.output_drift:.4f}")


def main():
    """Run all demos."""
    print("╔" + "═"*78 + "╗")
    print("║" + " ERIS ML Oracle — Live Demo v0.1.0 ".center(78) + "║")
    print("║" + " Decentralized AI Oracle for Prediction Markets ".center(78) + "║")
    print("║" + " Combines JEPA World Models, Multi-Agent Verification, ".center(78) + "║")
    print("║" + " SAE Drift Detection, and Bittensor Subnet Integration ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    
    # Demo 1: Live trending markets (with fallback)
    result = demo_trending_markets(n_markets=3)
    
    # Demo 2: Drift gate simulation
    demo_drift_detection()
    
    # Demo 3: Miner subnet info
    print(f"\n{'='*80}")
    print("BITTENSOR SUBNET TARGETS")
    print("="*80)
    
    engine = MLOracleEngine(verbose=False)
    subnets = engine.miner.get_supported_subnets()
    
    print(f"\n  {'SN#':<6} {'Name':<18} {'Emission':<10} {'Type':<15}")
    print("  " + "-"*50)
    for s in subnets:
        print(f"  {s['netuid']:<6} {s['name']:<18} {s['emission']:<10} {s['type']:<15}")
    
    print(f"\n  Current registration cost: {engine.miner.SUPPORTED_SUBNETS[3].get('registration_cost_tao', 1018.5):.1f} TAO "
          f" (~${1018.5 * 304:.0f})")
    
    print(f"\n{'='*80}")
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
