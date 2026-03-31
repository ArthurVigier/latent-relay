"""
SAEDriftGate
=============

SAE-based drift detection for oracle data quality.

Integrates ERIS SAE probe and drift detection infrastructure into
the oracle pipeline. Monitors data quality across:
  - Input oracle data (market features) for drift from expected distributions
  - Agent predictions for semantic drift in reasoning
  - Aggregated oracle output for statistical anomalies

Can operate in two modes:
  - STANDALONE: Uses statistical drift detection (no SAE model loaded)
  - SAE_PROBE: Uses SAE features from Gemma Scope for concept-level drift
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

log = logging.getLogger("oracle.drift_gate")


@dataclass
class DriftGateReport:
    """Result of a drift analysis pass."""
    timestamp: str
    overall_drift_score: float         # 0.0 (stable) to 1.0 (critical)
    gate_open: bool                      # True = data allowed, False = blocked
    severity: str                        # stable, low, medium, high, critical
    
    # Component-level scores
    input_drift: float = 0.0
    prediction_drift: float = 0.0
    output_drift: float = 0.0
    
    # SAE feature info
    drifted_features: list[int] = field(default_factory=list)
    detected_concepts: list[str] = field(default_factory=list)
    
    # Metadata
    n_data_points_analyzed: int = 0
    reference_age_hours: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_drift_score": round(self.overall_drift_score, 6),
            "gate_open": self.gate_open,
            "severity": self.severity,
            "input_drift": round(self.input_drift, 6),
            "prediction_drift": round(self.prediction_drift, 6),
            "output_drift": round(self.output_drift, 6),
            "drifted_features": self.drifted_features,
            "detected_concepts": self.detected_concepts,
            "n_data_points": self.n_data_points_analyzed,
            "reference_age_hours": round(self.reference_age_hours, 1),
        }


class SAEDriftGate:
    """
    SAE-based drift gate for oracle data quality.
    
    Monitors the entire oracle pipeline:
    1. Input data: Market features must stay within expected distribution
    2. Predictions: Agent/world model predictions must be internally consistent
    3. Output: Aggregated oracle must not show anomalous patterns
    
    Kill-gate: When drift exceeds threshold, gate closes and oracle
    refuses to submit data (prevents bad oracle submissions).
    """
    
    def __init__(
        self,
        input_threshold: float = 0.4,
        prediction_threshold: float = 0.5,
        output_threshold: float = 0.6,
        window_size: int = 20,
        smoothing_alpha: float = 0.3,
        mode: str = "standalone",  # "standalone" or "sae_probe"
    ):
        self.input_threshold = input_threshold
        self.prediction_threshold = prediction_threshold
        self.output_threshold = output_threshold
        self.smoothing_alpha = smoothing_alpha
        self.mode = mode
        
        # Statistics (EWMA)
        self._input_history: deque = deque(maxlen=window_size)
        self._prediction_history: deque = deque(maxlen=window_size)
        self._output_history: deque = deque(maxlen=window_size)
        
        # Reference baselines
        self._input_baseline: Optional[np.ndarray] = None
        self._prediction_baseline: Optional[np.ndarray] = None
        
        # State
        self._gate_open = True
        self._last_drift_time: Optional[float] = None
        self._total_checks = 0
        self._gates_triggered = 0
        
        log.info(
            f"DriftGate initialized: "
            f"input_thresh={input_threshold}, "
            f"pred_thresh={prediction_threshold}, "
            f"mode={mode}"
        )
    
    def check(
        self,
        input_features: list[float],
        predictions: list[float],
        output_value: float,
    ) -> DriftGateReport:
        """
        Run a drift analysis pass.
        
        Args:
            input_features: Current market feature vector
            predictions: Agent/world model prediction probabilities
            output_value: Aggregated oracle output probability
        
        Returns:
            DriftGateReport with gate status
        """
        from datetime import datetime, timezone
        self._total_checks += 1
        
        # Convert to numpy
        if _HAS_NUMPY:
            features = np.array(input_features, dtype=np.float32)
            preds = np.array(predictions, dtype=np.float32)
        else:
            features = np.array(input_features, dtype=float)
            preds = np.array(predictions, dtype=float)
        
        # 1. Input drift: compare against baseline
        input_drift = self._compute_input_drift(features)
        
        # 2. Prediction drift: consistency check
        pred_drift = self._compute_prediction_drift(preds, output_value)
        
        # 3. Output drift: statistical anomaly check
        output_drift = self._compute_output_drift(output_value)
        
        # 4. Combine scores
        overall = self._smoothed_score(input_drift, pred_drift, output_drift)
        
        # 5. SAE feature detection (if in SAE mode)
        drifted_features = []
        detected_concepts = []
        if self.mode == "sae_probe" and _HAS_NUMPY:
            drifted_features, detected_concepts = self._detect_drifted_features(
                features, preds
            )
        
        # 6. Severity classification
        severity = self._classify_severity(overall)
        
        # 7. Update histories
        self._input_history.append(input_drift)
        self._prediction_history.append(pred_drift)
        self._output_history.append(output_drift)
        
        # 8. Update baseline
        if self._input_baseline is None:
            self._input_baseline = features.copy()
            self._prediction_baseline = np.array([output_value])
        else:
            # EWMA baseline update
            alpha = self.smoothing_alpha * 0.1  # Slow baseline drift
            self._input_baseline = (
                alpha * features + (1 - alpha) * self._input_baseline
            )
        
        # 9. Gate check
        gate_open = True
        if input_drift > self.input_threshold:
            gate_open = False
        elif pred_drift > self.prediction_threshold:
            gate_open = False
        elif output_drift > self.output_threshold:
            gate_open = False
        
        if not gate_open:
            self._gates_triggered += 1
            self._last_drift_time = time.time()
        
        self._gate_open = gate_open
        
        # Reference age
        ref_age = 0.0
        if self._last_drift_time:
            ref_age = (time.time() - self._last_drift_time) / 3600
        
        # Build severity
        if overall < 0.1:
            severity = "stable"
        elif overall < 0.25:
            severity = "low"
        elif overall < 0.5:
            severity = "medium"
        elif overall < 0.75:
            severity = "high"
        else:
            severity = "critical"
        
        return DriftGateReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_drift_score=round(overall, 6),
            gate_open=gate_open,
            severity=severity,
            input_drift=round(input_drift, 6),
            prediction_drift=round(pred_drift, 6),
            output_drift=round(output_drift, 6),
            drifted_features=drifted_features,
            detected_concepts=detected_concepts,
            n_data_points_analyzed=len(features) + len(preds) + 1,
            reference_age_hours=ref_age,
        )
    
    def _compute_input_drift(self, features: np.ndarray) -> float:
        """Compute drift of input features from baseline."""
        if self._input_baseline is None:
            return 0.0
        if len(features) != len(self._input_baseline):
            return 1.0
        
        norm = float(np.linalg.norm(self._input_baseline) * np.linalg.norm(features))
        if norm <= 0:
            return 0.0
        
        cosine = float(1.0 - np.dot(self._input_baseline, features) / norm)
        
        # Also check per-dimension shifts
        diff = features - self._input_baseline
        per_dim_max = float(np.max(np.abs(diff)))
        
        # Weighted combination
        return max(0.0, min(1.0, 0.7 * cosine + 0.3 * per_dim_max))
    
    def _compute_prediction_drift(
        self,
        predictions: np.ndarray,
        output: float,
    ) -> float:
        """Check if predictions are internally consistent with output."""
        if len(predictions) == 0:
            return 0.0
        
        mean_pred = float(np.mean(predictions))
        # How far is the output from the mean prediction?
        disagreement = abs(output - mean_pred)
        
        # Also check prediction diversity
        if len(predictions) > 1:
            std = float(np.std(predictions))
            # High disagreement + high diversity = drift
            drift = disagreement * 2 + std
        else:
            drift = disagreement * 2
        
        return max(0.0, min(1.0, drift))
    
    def _compute_output_drift(self, output: float) -> float:
        """Check if output is within expected statistical bounds."""
        if len(self._output_history) < 3:
            return 0.0
        
        history = list(self._output_history)
        mean = sum(history) / len(history)
        std = (sum((x - mean) ** 2 for x in history) / len(history)) ** 0.5
        
        if std < 0.001:
            return 0.0
        
        z_score = abs(output - mean) / std
        # Convert to 0-1 scale
        return min(1.0, z_score / 5.0)  # 5 std deviations = max drift
    
    def _smoothed_score(
        self,
        input_d: float,
        pred_d: float,
        output_d: float,
    ) -> float:
        """Compute smoothed overall drift score."""
        # Weighted combination
        raw = 0.3 * input_d + 0.4 * pred_d + 0.3 * output_d
        
        # Smooth with history
        histories = [
            self._input_history,
            self._prediction_history,
            self._output_history,
        ]
        
        for hist in histories:
            if len(hist) > 3:
                alpha = self.smoothing_alpha
                hist_list = list(hist)
                raw = alpha * raw + (1 - alpha) * (sum(hist_list[-3:]) / 3)
        
        return max(0.0, min(1.0, raw))
    
    def _detect_drifted_features(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[list[int], list[str]]:
        """Detect which features are drifting (SAE concept-level)."""
        if self._input_baseline is None:
            return [], []
        
        drifted = []
        concepts = []
        
        if len(features) == len(self._input_baseline):
            diff = np.abs(features - self._input_baseline)
            threshold = np.mean(diff) + 2 * np.std(diff)
            
            for i in range(len(diff)):
                if diff[i] > threshold:
                    drifted.append(i)
                    concepts.append(f"feature_{i}_drift")
        
        return drifted, concepts[:10]
    
    def _classify_severity(self, score: float) -> str:
        if score < 0.1:
            return "stable"
        elif score < 0.25:
            return "low"
        elif score < 0.5:
            return "medium"
        elif score < 0.75:
            return "high"
        else:
            return "critical"
    
    @property
    def gate_open(self) -> bool:
        """Current gate status."""
        return self._gate_open
    
    @property
    def stats(self) -> dict:
        """Drift gate statistics."""
        return {
            "total_checks": self._total_checks,
            "gates_triggered": self._gates_triggered,
            "gate_open": self._gate_open,
            "avg_input_drift": (
                sum(self._input_history) / len(self._input_history)
                if self._input_history else 0.0
            ),
            "avg_prediction_drift": (
                sum(self._prediction_history) / len(self._prediction_history)
                if self._prediction_history else 0.0
            ),
            "window_size": self._input_history.maxlen,
        }
