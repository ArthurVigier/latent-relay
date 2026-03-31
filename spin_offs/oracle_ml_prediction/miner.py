"""
BittensorOracleMiner
=====================

Subnet-compatible mining interface for oracle predictions.

Simulates Bittensor miner/validator behavior for the oracle subnet.
Designed to be compatible with:
  - SN3 (taumplar) - LLM/text generation subnet
  - SN11 (TrajectoryRL) - RL optimization
  - SN86 (Investing) - Financial prediction
  - SN125 (8 Ball) - Prediction markets

Key capabilities:
  - Format oracle predictions as subnet-compatible responses
  - Simulate validator scoring (consine similarity to ground truth)
  - Track TAO reward estimates based on performance
  - Export data in expected subnet formats
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("oracle.miner")


@dataclass
class OracleSubmission:
    """A prediction submission to a Bittensor subnet."""
    netuid: int
    query_text: str
    response_text: str
    metadata: dict
    
    # Performance tracking
    uid: int = 0
    submission_time: float = 0.0
    tao_reward_estimate: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "netuid": self.netuid,
            "query": self.query_text,
            "response": self.response_text,
            "metadata": self.metadata,
            "uid": self.uid,
            "submission_time": self.submission_time,
            "tao_estimate": round(self.tao_reward_estimate, 6),
        }


@dataclass
class SubnetPerformance:
    """Performance tracking for a specific subnet."""
    netuid: int
    subnet_name: str
    total_queries: int = 0
    total_submissions: int = 0
    avg_reward: float = 0.0
    total_tao_earned: float = 0.0
    avg_latency_ms: float = 0.0
    last_submission: str = ""
    
    # Target subnet info
    target_emission: str = ""
    registration_cost_tao: float = 1018.5  # Current registration cost
    
    def to_dict(self) -> dict:
        return {
            "netuid": self.netuid,
            "name": self.subnet_name,
            "queries": self.total_queries,
            "submissions": self.total_submissions,
            "avg_reward_tao": round(self.avg_reward, 6),
            "total_tao": round(self.total_tao_earned, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "registration_cost_tao": self.registration_cost_tao,
        }


class BittensorOracleMiner:
    """
    Bittensor subnet oracle miner.
    
    Wraps oracle predictions in the format expected by Bittensor subnets.
    Tracks performance and estimated TAO rewards.
    
    Supported target subnets:
      SN3:  taumplar   (LLM/text, 7.58% emission)
      SN4:  Targon     (Image/text, 6.29% emission)
      SN11: TrajectoryRL (RL, 1.30% emission)
      SN81: grail      (ML/knowledge, 2.67% emission)
      SN86: Investing  (Prediction/finance, 0.64% emission)
      SN125: 8 Ball    (Prediction oracle, emerging)
    """
    
    SUPPORTED_SUBNETS = {
        3: {"name": "taumplar", "emission": "7.58%", "type": "text"},
        4: {"name": "Targon", "emission": "6.29%", "type": "multimodal"},
        11: {"name": "TrajectoryRL", "emission": "1.30%", "type": "rl"},
        81: {"name": "grail", "emission": "2.67%", "type": "knowledge"},
        86: {"name": "Investing", "emission": "0.64%", "type": "prediction"},
        125: {"name": "8 Ball", "emission": "emerging", "type": "prediction"},
    }
    
    def __init__(self, uid: int = 0):
        self.uid = uid
        self._subnet_performance: dict[int, SubnetPerformance] = {}
        
        for netuid, info in self.SUPPORTED_SUBNETS.items():
            self._subnet_performance[netuid] = SubnetPerformance(
                netuid=netuid,
                subnet_name=info["name"],
                target_emission=info["emission"],
            )
        
        log.info(
            f"Oracle miner initialized: uid={uid}, "
            f"subnets={list(self.SUPPORTED_SUBNETS.keys())}"
        )
    
    def format_prediction_response(
        self,
        netuid: int,
        event_description: str,
        consensus_probability: float,
        consensus_confidence: float,
        agent_consensus: Optional[dict] = None,
        drift_report: Optional[dict] = None,
    ) -> OracleSubmission:
        """Format oracle prediction for Bittensor subnet."""
        info = self.SUPPORTED_SUBNETS.get(netuid, {"name": f"SN{netuid}"})
        
        # Build response text based on subnet type
        if info["type"] == "text":
            response = self._format_text_response(
                event_description, consensus_probability,
                consensus_confidence, agent_consensus
            )
        elif info["type"] == "prediction":
            response = self._format_prediction_response(
                event_description, consensus_probability,
                consensus_confidence, agent_consensus
            )
        else:
            response = self._format_generic_response(
                event_description, consensus_probability,
                consensus_confidence
            )
        
        metadata = {
            "oracle_version": "0.1.0",
            "event": event_description,
            "probability": consensus_probability,
            "confidence": consensus_confidence,
            "subnet_type": info["type"],
        }
        
        if agent_consensus:
            metadata["agent_count"] = agent_consensus.get("agent_count", 0)
            metadata["surviving_agents"] = agent_consensus.get("surviving_agents", 0)
        
        if drift_report:
            metadata["drift_gate"] = drift_report.get("gate_open", True)
            metadata["drift_severity"] = drift_report.get("severity", "unknown")
        
        submission_time = time.time()
        
        # Estimate TAO reward
        tao_reward = self._estimate_tao_reward(
            netuid, consensus_confidence, drift_report
        )
        
        return OracleSubmission(
            netuid=netuid,
            query_text=event_description,
            response_text=response,
            metadata=metadata,
            uid=self.uid,
            submission_time=submission_time,
            tao_reward_estimate=tao_reward,
        )
    
    def _format_text_response(
        self,
        event: str,
        prob: float,
        conf: float,
        agent_info: Optional[dict],
    ) -> str:
        """Format for text-based subnets (SN3, SN4)."""
        response = f"Oracle Prediction for: {event}\n\n"
        response += f"Probability (YES): {prob*100:.2f}%\n"
        response += f"Confidence: {conf*100:.1f}%\n"
        response += f"Probability (NO): {(1-prob)*100:.2f}%\n"
        
        if agent_info:
            n = agent_info.get("surviving_agents", 0)
            k = agent_info.get("killed_agents", 0)
            total = agent_info.get("agent_count", 0)
            response += f"\nVerification: {n}/{total} agents survived "
            response += f"({k} killed by drift gate)\n"
            
            dissent = agent_info.get("dissent_score", 0)
            response += f"Agent dissent score: {dissent:.4f}\n"
        
        response += "\nThis prediction was generated by ERIS ML Oracle v0.1.0"
        response += " using multi-agent verification with SAE drift detection."
        
        return response
    
    def _format_prediction_response(
        self,
        event: str,
        prob: float,
        conf: float,
        agent_info: Optional[dict],
    ) -> str:
        """Format for prediction oracle subnets."""
        response = f"PREDICTION: {event}\n"
        response += f"fYES={prob*100:.6f} fNO={(1-prob)*100:.6f}\n"
        response += f"confidence={conf*100:.1f}%\n"
        
        if agent_info:
            response += (
                f"consensus_agents={agent_info.get('surviving_agents', 0)} "
                f"killed={agent_info.get('killed_agents', 0)} "
                f"dissent={agent_info.get('dissent_score', 0):.4f}"
            )
        
        return response
    
    def _format_generic_response(
        self,
        event: str,
        prob: float,
        conf: float,
    ) -> str:
        """Generic format for other subnet types."""
        return (
            f"Oracle: {event} | P(yes)={prob:.4f} | "
            f"conf={conf:.2f} | ERIS v0.1.0"
        )
    
    def _estimate_tao_reward(
        self,
        netuid: int,
        confidence: float,
        drift_report: Optional[dict],
    ) -> float:
        """Estimate TAO reward based on prediction quality."""
        # Base reward proportional to subnet emission
        emission_pct = {
            3: 0.0758, 4: 0.0629, 11: 0.0130,
            81: 0.0267, 86: 0.0064, 125: 0.0050,
        }
        
        base = emission_pct.get(netuid, 0.01)
        
        # Confidence multiplier
        conf_mult = 0.5 + confidence * 0.5  # 0.5 to 1.0
        
        # Drift penalty
        drift_mult = 1.0
        if drift_report:
            if drift_report.get("severity") == "high":
                drift_mult = 0.3
            elif drift_report.get("severity") == "medium":
                drift_mult = 0.7
            elif drift_report.get("severity") == "critical":
                drift_mult = 0.0
        
        # Per-timestep reward estimate (very rough)
        reward = base * conf_mult * drift_mult
        
        # Update performance tracking
        perf = self._subnet_performance.get(netuid)
        if perf:
            perf.total_queries += 1
            perf.avg_reward = (
                perf.avg_reward * 0.95 + reward * 0.05
            )
            perf.total_tao_earned += reward
        
        return reward
    
    def get_performance_report(self) -> dict:
        """Get performance report for all subnets."""
        return {
            netuid: perf.to_dict()
            for netuid, perf in self._subnet_performance.items()
            if perf.total_queries > 0
        }
    
    def get_supported_subnets(self) -> list[dict]:
        """List all supported target subnets."""
        return [
            {
                "netuid": netuid,
                "name": info["name"],
                "emission": info["emission"],
                "type": info["type"],
            }
            for netuid, info in self.SUPPORTED_SUBNETS.items()
        ]
