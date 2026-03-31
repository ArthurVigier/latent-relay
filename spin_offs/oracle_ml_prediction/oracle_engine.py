"""
MLOracleEngine
===============

Main orchestrator for the ERIS ML Oracle prediction system.

Ties together:
  1. PolymarketOracleFeed - live market data ingestion
  2. PredictionWorldModel - JEPA-style outcome prediction
  3. MultiAgentOracleVerifier - LatentMAS consensus verification
  4. SAEDriftGate - SAE-based data quality monitoring
  5. BittensorOracleMiner - subnet-compatible submission format

Usage:
    engine = MLOracleEngine()
    
    # Query a prediction topic
    result = engine.predict("Will the Fed cut rates in 2026?")
    print(result.consensus_probability)
    
    # Submit to Bittensor subnet
    submission = engine.submit_to_subnet(result, netuid=86)
    print(submission.response_text)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

# Local imports
try:
    from .feeds.polymarket_feed import (
        PolymarketOracleFeed, MarketSnapshot, TopicFeed,
        build_oracle_feed, fetch_polymarket_trending,
    )
    from .world_model.predictor import PredictionWorldModel, PredictionOutput
    from .verifier.multi_agent_verifier import (
        MultiAgentOracleVerifier, OracleConsensus, AgentRole,
    )
    from .verifier.sae_drift_gate import SAEDriftGate, DriftGateReport
    from .miner import BittensorOracleMiner, OracleSubmission
except ImportError:
    from spins_offs.oracle_ml_prediction.feeds.polymarket_feed import (
        PolymarketOracleFeed, MarketSnapshot, TopicFeed,
        build_oracle_feed, fetch_polymarket_trending,
    )
    from spins_offs.oracle_ml_prediction.world_model.predictor import (
        PredictionWorldModel, PredictionOutput
    )
    from spins_offs.oracle_ml_prediction.verifier.multi_agent_verifier import (
        MultiAgentOracleVerifier, OracleConsensus, AgentRole,
    )
    from spins_offs.oracle_ml_prediction.verifier.sae_drift_gate import (
        SAEDriftGate, DriftGateReport
    )
    from spins_offs.oracle_ml_prediction.miner import (
        BittensorOracleMiner, OracleSubmission
    )

log = logging.getLogger("oracle.engine")


@dataclass
class OracleResult:
    """Complete oracle prediction result."""
    event_description: str
    market_data: Optional[MarketSnapshot] = None
    
    # Pipeline outputs
    world_model_prediction: Optional[PredictionOutput] = None
    agent_consensus: Optional[OracleConsensus] = None
    drift_gate_report: Optional[DriftGateReport] = None
    
    # Synthesized final prediction
    final_probability: float = 0.5
    final_confidence: float = 0.0
    pipeline_passed: bool = True
    
    # Metadata
    latency_ms: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return {
            "event": self.event_description,
            "market_data": self.market_data.to_dict() if self.market_data else None,
            "world_model": (
                self.world_model_prediction.to_dict()
                if self.world_model_prediction else None
            ),
            "consensus": (
                self.agent_consensus.to_dict()
                if self.agent_consensus else None
            ),
            "drift_gate": (
                self.drift_gate_report.to_dict()
                if self.drift_gate_report else None
            ),
            "final_probability": round(self.final_probability, 6),
            "final_confidence": round(self.final_confidence, 4),
            "pipeline_passed": self.pipeline_passed,
            "latency_ms": round(self.latency_ms, 1),
            "timestamp": self.timestamp,
        }
    
    def to_oracle_payload(self) -> dict:
        """Format for on-chain submission."""
        return {
            "type": "eris_oracle_v1",
            "event": self.event_description,
            "prediction": {
                "probability": round(self.final_probability, 6),
                "confidence": round(self.final_confidence, 4),
            },
            "verification": self.agent_consensus.to_oracle_payload() if self.agent_consensus else {},
            "drift_gate": self.drift_gate_report.to_dict() if self.drift_gate_report else {},
            "market_data": self.market_data.to_dict() if self.market_data else None,
            "pipeline_passed": self.pipeline_passed,
        }


class MLOracleEngine:
    """
    ERIS ML Oracle Engine.
    
    Full oracle pipeline:
      Data Feed -> World Model -> Multi-Agent Verify -> Drift Gate -> Output
    
    Usage:
        engine = MLOracleEngine(
            drift_threshold=0.5,
            n_agents=5,
            target_netuid=86,
        )
        result = engine.predict("Will AI replace lawyers by 2030?")
        print(f"Oracle says: {result.final_probability*100:.1f}% yes")
    """
    
    def __init__(
        self,
        n_agents: int = 5,
        drift_threshold: float = 0.5,
        data_feed_timeout: int = 15,
        consensus_mode: str = "weighted",
        miner_uid: int = 0,
        kill_gate_enabled: bool = True,
        verbose: bool = True,
    ):
        # Initialize pipeline components
        self.feed = PolymarketOracleFeed(request_timeout=data_feed_timeout)
        self.world_model = PredictionWorldModel(mode="lightweight")
        self.verifier = MultiAgentOracleVerifier(
            n_agents=n_agents,
            drift_threshold=drift_threshold,
            consensus_mode=consensus_mode,
            kill_gate_enabled=kill_gate_enabled,
        )
        self.drift_gate = SAEDriftGate(
            input_threshold=drift_threshold,
            prediction_threshold=drift_threshold * 1.2,
            output_threshold=drift_threshold * 1.5,
        )
        self.miner = BittensorOracleMiner(uid=miner_uid)
        
        # State
        self._prediction_history: list[OracleResult] = []
        self._verbose = verbose
        
        log.info(
            f"MLOracleEngine initialized: "
            f"agents={n_agents}, drift_threshold={drift_threshold}, "
            f"kill_gate={kill_gate_enabled}, miner_uid={miner_uid}"
        )
    
    def predict(self, event_query: str) -> OracleResult:
        """
        Run the full oracle pipeline for an event.
        
        Args:
            event_query: The prediction event to query
        
        Returns:
            OracleResult with complete pipeline outputs
        """
        t0 = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"ORACLE PREDICTION: '{event_query}'")
        log.info(f"{'='*60}")
        
        # Step 1: Fetch market data
        if self._verbose:
            print(f"\n[1/4] Fetching market data for: '{event_query}'")
        
        markets = self._find_relevant_markets(event_query)
        market = markets[0] if markets else None
        
        if market:
            features = market.to_feature_vector()
            if self._verbose:
                print(f"   Found: '{market.question}' ({market.yes_price*100:.1f}% yes)")
        else:
            features = [0.5, 0.5, 0.5, 0.02, 0.5, 0.0, 0.5, 0.5]
            if self._verbose:
                print("   No direct market found. Using uninformative prior.")
        
        # Step 2: World model prediction
        if self._verbose:
            print(f"[2/4] Running world model prediction...")
        
        history = []
        if market:
            history = self.feed.get_price_history(
                market.condition_id, interval="1day", fidelity=20
            )
        
        wm_pred = self.world_model.predict_event_outcome(
            event_description=event_query,
            current_features=features,
            price_history=history,
            horizon_hours=720,  # ~30 days
        )
        
        if self._verbose:
            print(f"   Model predicts: {wm_pred.predicted_yes_probability*100:.1f}% yes "
                  f"(confidence: {wm_pred.confidence*100:.0f}%)")
        
        # Step 3: Multi-agent verification
        if self._verbose:
            print(f"[3/4] Running multi-agent verification...")
        
        consensus = self.verifier.verify(
            event_description=event_query,
            current_features=features,
            world_model_prediction=wm_pred.to_dict(),
            price_history=history,
        )
        
        if self._verbose:
            print(f"   Consensus: {consensus.consensus_probability*100:.1f}% yes")
            print(f"   Agents: {consensus.surviving_agents}/{consensus.agent_count} survived "
                  f"({consensus.killed_agents} killed)")
        
        # Step 4: Drift gate check
        if self._verbose:
            print(f"[4/4] Checking drift gate...")
        
        agent_preds = [a.predicted_probability for a in consensus.agents if not a.killed]
        drift_report = self.drift_gate.check(
            input_features=features,
            predictions=agent_preds,
            output_value=consensus.consensus_probability,
        )
        
        if self._verbose:
            gate_str = "OPEN" if drift_report.gate_open else "CLOSED"
            print(f"   Drift gate: {gate_str}")
            print(f"   Drift score: {drift_report.overall_drift_score:.4f} "
                  f"(severity: {drift_report.severity})")
        
        # Step 5: Synthesize final prediction
        final_prob, final_conf = self._synthesize(
            market, wm_pred, consensus, drift_report
        )
        
        elapsed_ms = (time.time() - t0) * 1000
        
        result = OracleResult(
            event_description=event_query,
            market_data=market,
            world_model_prediction=wm_pred,
            agent_consensus=consensus,
            drift_gate_report=drift_report,
            final_probability=final_prob,
            final_confidence=final_conf,
            pipeline_passed=drift_report.gate_open,
            latency_ms=elapsed_ms,
            timestamp=drift_report.timestamp,
        )
        
        self._prediction_history.append(result)
        
        if self._verbose:
            print(f"\n{'='*60}")
            print(f"FINAL: {final_prob*100:.1f}% yes (confidence: {final_conf*100:.0f}%)")
            print(f"Pipeline: {drift_report.severity} | Gate: {gate_str}")
            print(f"Latency: {elapsed_ms:.0f}ms")
            print(f"{'='*60}")
        
        return result
    
    def submit_to_subnet(
        self,
        result: OracleResult,
        netuid: int = 86,
    ) -> OracleSubmission:
        """Format oracle result for Bittensor subnet submission."""
        agent_dict = result.agent_consensus.to_dict() if result.agent_consensus else None
        drift_dict = result.drift_gate_report.to_dict() if result.drift_gate_report else None
        
        submission = self.miner.format_prediction_response(
            netuid=netuid,
            event_description=result.event_description,
            consensus_probability=result.final_probability,
            consensus_confidence=result.final_confidence,
            agent_consensus=agent_dict,
            drift_report=drift_dict,
        )
        
        log.info(
            f"Formatted for SN{netuid} ({self.miner.SUPPORTED_SUBNETS.get(netuid, {}).get('name', '?')}): "
            f"{result.final_probability*100:.1f}% yes"
        )
        return submission
    
    def get_trending_predictions(self, limit: int = 5) -> list[OracleResult]:
        """Generate oracle predictions for currently trending markets."""
        trending = self.feed.get_trending(limit=limit)
        results = []
        
        for market in trending[:limit]:
            result = self.predict(market.question)
            results.append(result)
        
        return results
    
    def _find_relevant_markets(
        self, query: str
    ) -> list[MarketSnapshot]:
        """Find Polymarket markets relevant to a query."""
        # Direct search
        markets = self.feed.search_markets(query, limit=5)
        
        if not markets:
            # Try keyword extraction
            keywords = self._extract_keywords(query)
            for kw in keywords:
                markets = self.feed.search_markets(kw, limit=5)
                if markets:
                    break
        
        return markets
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract search keywords from a query."""
        stop_words = {
            "will", "the", "a", "an", "by", "in", "on", "at", "to", "for",
            "of", "with", "is", "are", "was", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "if", "that",
            "this", "these", "those", "what", "when", "where", "who",
            "how", "which", "can", "could", "should", "would",
        }
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Also try bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)
                   if i+1 < len(words)]
        
        return keywords[:3] + bigrams[:3]
    
    def _synthesize(
        self,
        market: Optional[MarketSnapshot],
        wm_pred: PredictionOutput,
        consensus: OracleConsensus,
        drift: DriftGateReport,
    ) -> tuple[float, float]:
        """Synthesize final prediction from all pipeline outputs."""
        # Weight different predictions
        weights = {}
        
        # Market price (most reliable if market exists)
        if market and not market.closed:
            weights["market"] = (0.4, market.yes_price)
        
        # World model
        weights["world_model"] = (0.2, wm_pred.predicted_yes_probability)
        
        # Agent consensus
        if consensus.surviving_agents > 0:
            weights["consensus"] = (0.4, consensus.consensus_probability)
        else:
            weights["consensus"] = (0.0, 0.5)  # All killed, no weight
        
        # Compute weighted average
        total_weight = sum(w for w, _ in weights.values())
        if total_weight > 0:
            final_prob = sum(w * p for w, p in weights.values()) / total_weight
        else:
            final_prob = 0.5
        
        # Confidence from pipeline health
        conf_factors = []
        if market:
            conf_factors.append(min(market.volume / 1_000_000, 1.0) * 0.3 + 0.3)
        conf_factors.append(wm_pred.confidence * 0.3)
        conf_factors.append(consensus.consensus_confidence * 0.4)
        final_conf = sum(conf_factors) / len(conf_factors) if conf_factors else 0.1
        
        # Penalty for drift severity
        if drift.severity == "high":
            final_conf *= 0.5
        elif drift.severity == "critical":
            final_conf *= 0.1
        
        final_prob = max(0.01, min(0.99, final_prob))
        final_conf = max(0.01, min(0.99, final_conf))
        
        return round(final_prob, 6), round(final_conf, 4)
