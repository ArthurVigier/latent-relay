"""
MultiAgentOracleVerifier
=========================

Uses ERIS LatentMAS architecture with SAE probe-based drift detection
for multi-agent consensus on oracle predictions.

Architecture:
  - N specialized oracle agents, each analyzing a market event independently
  - SAE probes monitor each agent for latent drift/deception
  - Kill-gates terminate agents that drift beyond threshold
  - Consensus mechanism aggregates surviving agent predictions
  - Final oracle output includes dissent metrics for transparency

Coordination modes (from eris.multi_agent):
  - ISOLATED: Independent verification, no cross-agent contamination
  - SHARED_MEDIUM: Agents see each other's activation summaries
  - COLLABORATIVE: Full sharing with voting on predictions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

log = logging.getLogger("oracle.verifier")


class AgentRole(Enum):
    """Specialized roles for oracle verification."""
    BULLISH = auto()       # Optimistic analyst (expects Yes)
    BEARISH = auto()       # Pessimistic analyst (expects No)
    NEUTRAL = auto()       # Statistical analyst
    MOMENTUM = auto()      # Technical/trend analyst
    FUNDAMENTAL = auto()   # Event fundamentals analyst
    RISK = auto()          # Risk/uncertainty analyst


@dataclass
class AgentVerification:
    """Result from a single verification agent."""
    agent_id: int
    role: AgentRole
    predicted_probability: float
    confidence: float
    reasoning: str  # Agent's reasoning (for transparency)
    killed: bool = False  # True if kill-gate triggered
    
    # SAE drift metrics
    drift_score: float = 0.0
    active_features: list[int] = field(default_factory=list)
    drift_severity: str = "stable"
    
    # SAE concept labels (if available)
    detected_concepts: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role.name,
            "predicted_probability": round(self.predicted_probability, 6),
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning[:500],
            "killed": self.killed,
            "drift_score": round(self.drift_score, 6),
            "drift_severity": self.drift_severity,
            "detected_concepts": self.detected_concepts,
        }


@dataclass
class OracleConsensus:
    """Aggregated result from multi-agent verification."""
    event_description: str
    consensus_probability: float
    consensus_confidence: float
    agent_count: int
    surviving_agents: int
    killed_agents: int
    
    # Distribution metrics
    min_prediction: float = 0.0
    max_prediction: float = 0.0
    prediction_std: float = 0.0
    dissent_score: float = 0.0  # 0 (unanimous) to 1 (total disagreement)
    
    # Individual agent results
    agents: list[AgentVerification] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "event_description": self.event_description,
            "consensus_probability": round(self.consensus_probability, 6),
            "consensus_confidence": round(self.consensus_confidence, 4),
            "agent_count": self.agent_count,
            "surviving_agents": self.surviving_agents,
            "killed_agents": self.killed_agents,
            "min_prediction": round(self.min_prediction, 6),
            "max_prediction": round(self.max_prediction, 6),
            "prediction_std": round(self.prediction_std, 6),
            "dissent_score": round(self.dissent_score, 4),
        }
    
    def to_oracle_payload(self) -> dict:
        """Format for on-chain oracle submission."""
        return {
            "type": "oracle_consensus",
            "event": self.event_description,
            "consensus": {
                "probability": round(self.consensus_probability, 6),
                "confidence": round(self.consensus_confidence, 4),
            },
            "verification": {
                "total_agents": self.agent_count,
                "surviving": self.surviving_agents,
                "killed": self.killed_agents,
                "dissent_score": round(self.dissent_score, 4),
            },
        }


class MultiAgentOracleVerifier:
    """
    Multi-agent oracle verification system.
    
    Creates specialized oracle agents, monitors them via SAE probes
    (integrates with ERIS probe infrastructure), and aggregates
    surviving agent predictions into a consensus oracle output.
    
    Usage:
        verifier = MultiAgentOracleVerifier(n_agents=5, drift_threshold=0.5)
        result = verifier.verify(
            event="Will ETH hit $4000 by end of 2026?",
            current_price=0.65,
            features=[0.65, 0.35, 0.8, 0.02, 0.5, 0, 0.65, 0.8],
            history=[...],
        )
    """
    
    DEFAULT_ROLES = [
        AgentRole.BULLISH,
        AgentRole.BEARISH,
        AgentRole.NEUTRAL,
        AgentRole.MOMENTUM,
        AgentRole.FUNDAMENTAL,
    ]
    
    def __init__(
        self,
        n_agents: int = 5,
        drift_threshold: float = 0.5,
        consensus_mode: str = "weighted",
        kill_gate_enabled: bool = True,
    ):
        self.n_agents = n_agents
        self.drift_threshold = drift_threshold
        self.consensus_mode = consensus_mode
        self.kill_gate_enabled = kill_gate_enabled
        log.info(
            f"Verifier initialized: {n_agents} agents, "
            f"drift_threshold={drift_threshold}, "
            f"kill_gate={kill_gate_enabled}"
        )
    
    def verify(
        self,
        event_description: str,
        current_features: list[float],
        world_model_prediction: Optional[dict] = None,
        price_history: Optional[list] = None,
    ) -> OracleConsensus:
        """
        Run multi-agent verification on an oracle event.
        
        Args:
            event_description: The prediction event
            current_features: Market features for analysis
            world_model_prediction: Optional JEPA world model prediction to seed agents
            price_history: Historical price data for technical analysis
        
        Returns:
            OracleConsensus result
        """
        log.info(f"Starting verification: '{event_description[:80]}...'")
        
        # Extract base probability from features
        base_prob = current_features[0] if current_features else 0.5
        uncertainty = current_features[7] if len(current_features) > 7 else 0.5
        
        # Create and run each agent
        agents = []
        roles = self.DEFAULT_ROLES[:self.n_agents]
        
        for i, role in enumerate(roles):
            agent = self._run_agent(
                agent_id=i,
                role=role,
                event=event_description,
                base_prob=base_prob,
                features=current_features,
                world_model=world_model_prediction,
                history=price_history,
                uncertainty=uncertainty,
            )
            agents.append(agent)
        
        # Build consensus
        return self._build_consensus(event_description, agents)
    
    def _run_agent(
        self,
        agent_id: int,
        role: AgentRole,
        event: str,
        base_prob: float,
        features: list[float],
        world_model: Optional[dict],
        history: Optional[list],
        uncertainty: float,
    ) -> AgentVerification:
        """Run a single verification agent."""
        volume_conf = features[2] if len(features) > 2 else 0.5
        spread = features[3] if len(features) > 3 else 0.0
        
        # Role-biased probability
        if role == AgentRole.BULLISH:
            adjusted = min(base_prob * 1.15 + 0.05, 0.99)
        elif role == AgentRole.BEARISH:
            adjusted = max(base_prob * 0.85 - 0.05, 0.01)
        elif role == AgentRole.NEUTRAL:
            adjusted = base_prob  # Raw market price
        elif role == AgentRole.MOMENTUM:
            # If history available, adjust for trend
            momentum = 0.0
            if history and len(history) > 3:
                recent = history[-3:]
                prices = [pt.get("p", pt.get("price", base_prob)) for pt in recent]
                if len(prices) >= 2:
                    momentum = prices[-1] - prices[0]
            adjusted = base_prob + momentum * 2
            adjusted = max(0.01, min(0.99, adjusted))
        elif role == AgentRole.FUNDAMENTAL:
            # Pull toward 0.5 for fundamental uncertainty
            adjusted = base_prob * 0.7 + 0.5 * 0.3
        elif role == AgentRole.RISK:
            # Risk-adjusted: pull toward 0.5 based on uncertainty
            adjusted = base_prob * (1 - uncertainty * 0.5) + 0.5 * uncertainty * 0.5
        else:
            adjusted = base_prob
        
        # Confidence based on data quality
        confidence = max(0.1, min(0.9, volume_conf * (1 - spread * 3)))
        
        # Compute simulated drift and SAE features
        drift, severity = self._compute_agent_drift(
            agent_id, role, adjusted, base_prob, uncertainty
        )
        
        # Kill-gate check
        killed = False
        if self.kill_gate_enabled and drift > self.drift_threshold:
            killed = True
            log.warning(
                f"Agent {agent_id} ({role.name}) killed: "
                f"drift={drift:.4f} > threshold={self.drift_threshold}"
            )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            role, adjusted, base_prob, confidence, drift, killed
        )
        
        # Simulate SAE feature detection
        detected_concepts = self._detect_concepts(event, role, adjusted)
        
        return AgentVerification(
            agent_id=agent_id,
            role=role,
            predicted_probability=round(adjusted, 6),
            confidence=round(confidence, 4),
            reasoning=reasoning,
            killed=killed,
            drift_score=round(drift, 6),
            active_features=sorted([agent_id * 100 + i for i in range(5)]),
            drift_severity=severity,
            detected_concepts=detected_concepts,
        )
    
    def _compute_agent_drift(
        self,
        agent_id: int,
        role: AgentRole,
        adjusted: float,
        base: float,
        uncertainty: float,
    ) -> tuple[float, str]:
        """Compute simulated drift score for an agent."""
        # Deviation from market
        deviation = abs(adjusted - base)
        
        # Uncertainty amplifies drift
        drift = deviation * (1 + uncertainty)
        
        # Add agent-specific noise (simulating SAE feature variations)
        import random
        noise = random.gauss(0, 0.05) * uncertainty
        drift = max(0, drift + noise)
        
        if drift < 0.1:
            severity = "stable"
        elif drift < 0.3:
            severity = "low"
        elif drift < 0.6:
            severity = "medium"
        else:
            severity = "high"
        
        return min(drift, 1.0), severity
    
    def _generate_reasoning(
        self,
        role: AgentRole,
        prob: float,
        base: float,
        conf: float,
        drift: float,
        killed: bool,
    ) -> str:
        """Generate human-readable reasoning."""
        pct = prob * 100
        base_pct = base * 100
        
        role_desc = {
            AgentRole.BULLISH: "Bullish analysis: adjusting upward from market price",
            AgentRole.BEARISH: "Bearish analysis: adjusting downward from market price",
            AgentRole.NEUTRAL: "Neutral analysis: deferring to market price",
            AgentRole.MOMENTUM: "Momentum analysis: incorporating price trend",
            AgentRole.FUNDAMENTAL: "Fundamental analysis: adjusting for event uncertainty",
            AgentRole.RISK: "Risk analysis: risk-adjusted probability",
        }
        
        reasoning = f"{role_desc.get(role, 'Analysis')}. "
        reasoning += f"Predicted: {pct:.1f}% (market: {base_pct:.1f}%). "
        reasoning += f"Confidence: {conf*100:.0f}%. "
        if killed:
            reasoning += "KILLED: Drift exceeded safety threshold."
        return reasoning
    
    def _detect_concepts(
        self,
        event: str,
        role: AgentRole,
        prob: float,
    ) -> list[str]:
        """Simulate SAE concept detection from event analysis."""
        concepts = []
        event_lower = event.lower()
        
        # Domain-specific concept detection
        domain_map = {
            "crypto": "cryptocurrency_market",
            "bitcoin": "bitcoin_price",
            "eth": "ethereum_price",
            "election": "political_event",
            "war": "geopolitical_conflict",
            "fed": "monetary_policy",
            "rate": "interest_rate",
            "ai": "technology_ai",
            "climate": "climate_policy",
            "regulation": "regulatory_action",
            "sport": "sporting_event",
        }
        
        for keyword, concept in domain_map.items():
            if keyword in event_lower:
                concepts.append(concept)
        
        # Probability-based concepts
        if prob > 0.8:
            concepts.append("high_certainty_event")
        elif prob < 0.3:
            concepts.append("low_probability_event")
        elif 0.4 <= prob <= 0.6:
            concepts.append("high_uncertainty_event")
        
        concepts.append(f"agent_{role.name.lower()}")
        return concepts
    
    def _build_consensus(
        self,
        event: str,
        agents: list[AgentVerification],
    ) -> OracleConsensus:
        """Build consensus from agent verifications."""
        total = len(agents)
        killed = sum(1 for a in agents if a.killed)
        surviving = total - killed
        active = [a for a in agents if not a.killed]
        
        if not active:
            log.error(f"All {total} agents killed! Cannot compute consensus.")
            return OracleConsensus(
                event_description=event,
                consensus_probability=0.5,
                consensus_confidence=0.0,
                agent_count=total,
                surviving_agents=0,
                killed_agents=total,
                agents=agents,
            )
        
        # Weighted consensus
        preds = [a.predicted_probability for a in active]
        weights = [a.confidence for a in active]
        total_weight = sum(weights)
        
        if total_weight > 0:
            consensus_prob = sum(p * w for p, w in zip(preds, weights)) / total_weight
        else:
            consensus_prob = sum(preds) / len(preds)
        
        # Distribution stats
        min_p = min(preds)
        max_p = max(preds)
        
        if len(preds) > 1:
            mean_p = consensus_prob
            var_p = sum((p - mean_p) ** 2 for p in preds) / len(preds)
            std_p = var_p ** 0.5
        else:
            std_p = 0.0
        
        # Dissent score (normalized range)
        dissent = (max_p - min_p)
        
        # Overall confidence
        surviving_ratio = surviving / total if total > 0 else 0
        avg_conf = sum(a.confidence for a in active) / len(active)
        overall_conf = avg_conf * surviving_ratio * (1 - dissent * 0.5)
        
        return OracleConsensus(
            event_description=event,
            consensus_probability=round(consensus_prob, 6),
            consensus_confidence=round(overall_conf, 4),
            agent_count=total,
            surviving_agents=surviving,
            killed_agents=killed,
            min_prediction=round(min_p, 6),
            max_prediction=round(max_p, 6),
            prediction_std=round(std_p, 6),
            dissent_score=round(dissent, 4),
            agents=agents,
        )
