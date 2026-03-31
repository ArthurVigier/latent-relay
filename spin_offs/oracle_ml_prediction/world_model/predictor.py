"""
PredictionWorldModel
====================

JEPA-style world model for event outcome prediction.

Architecture:
  - Context Encoder: encodes market state + historical data into latent representation
  - Predictor: predicts future market state latents
  - Energy-based scoring: evaluates prediction quality against actual outcomes
  
For the oracle use case:
  - Input: MarketSnapshot feature vectors + textual descriptions
  - Output: Predicted probability distribution at future timestamps
  - Drift detection: compares predicted vs actual market evolution
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

log = logging.getLogger("oracle.world_model")


@dataclass
class PredictionOutput:
    """Output from the JEPA world model for oracle use."""
    event_description: str
    predicted_yes_probability: float
    confidence: float  # 0-1, model certainty
    prediction_horizon_hours: float
    # JEPA-specific
    context_latent: Optional[list[float]] = None
    predicted_latent: Optional[list[float]] = None
    energy_score: float = 0.0
    # Validation
    validation_signals: dict = field(default_factory=dict)
    model_version: str = "0.1.0"
    
    def to_dict(self) -> dict:
        return {
            "event_description": self.event_description,
            "predicted_yes_probability": self.predicted_yes_probability,
            "confidence": self.confidence,
            "prediction_horizon_hours": self.prediction_horizon_hours,
            "energy_score": self.energy_score,
            "validation_signals": self.validation_signals,
            "model_version": self.model_version,
        }
    
    def to_oracle_payload(self) -> dict:
        """Format for on-chain oracle submission."""
        return {
            "type": "oracle_prediction",
            "event": self.event_description,
            "prediction": {
                "probability": round(self.predicted_yes_probability, 6),
                "confidence": round(self.confidence, 4),
                "energy": round(self.energy_score, 6),
            },
            "validation": self.validation_signals,
            "model_version": self.model_version,
        }


class PredictionWorldModel:
    """
    JEPA-style world model for prediction market oracle.
    
    This model:
    1. Takes market features as context (yes_price, volume, spread, entropy)
    2. Encodes them into a latent representation
    3. Predicts the latent representation at a future time step
    4. Decodes back to a probability
    5. Computes an energy-based score for prediction quality
    
    Can run in two modes:
    - LIGHTWEIGHT: Uses heuristic/statistical model (no GPU needed)
    - FULL_JEPA: Uses actual JEPA model on GPU (requires le-world-model repo)
    """
    
    def __init__(self, mode: str = "lightweight"):
        self.mode = mode
        self._reference_features: Optional[list[float]] = None
        log.info(f"WorldModel initialized in {mode} mode")
    
    def predict_event_outcome(
        self,
        event_description: str,
        current_features: list[float],
        price_history: Optional[list[dict]] = None,
        horizon_hours: float = 168.0,  # 1 week default
        similar_events: Optional[list[dict]] = None,
    ) -> PredictionOutput:
        """
        Predict the outcome of a prediction market event.
        
        Args:
            event_description: Human-readable event description
            current_features: MarketSnapshot feature vector
            price_history: Historical price data points
            horizon_hours: Prediction horizon in hours
            similar_events: Historical similar events for calibration
        
        Returns:
            PredictionOutput with predicted probability and confidence
        """
        log.info(f"Predicting: '{event_description[:80]}...'")
        
        if self.mode == "lightweight":
            return self._lightweight_predict(
                event_description, current_features, price_history,
                horizon_hours, similar_events
            )
        else:
            return self._full_jepa_predict(
                event_description, current_features, price_history,
                horizon_hours, similar_events
            )
    
    def _lightweight_predict(
        self,
        event_desc: str,
        features: list[float],
        history: Optional[list[dict]],
        horizon: float,
        similar: Optional[list[dict]],
    ) -> PredictionOutput:
        """
        Statistical prediction model combining:
        1. Current market probability (yes_price as base)
        2. Momentum from price history
        3. Volume-weighted confidence
        4. Entropy-based uncertainty adjustment
        5. Time decay to horizon
        """
        # Parse features: [yes_price, no_price, vol_norm, spread, liq_norm, closed, implied, uncertainty]
        yes_price = features[0] if len(features) > 0 else 0.5
        volume = features[2] if len(features) > 2 else 0.5
        spread = features[3] if len(features) > 3 else 0.0
        uncertainty = features[7] if len(features) > 7 else 0.0
        implied = features[6] if len(features) > 6 else yes_price
        
        # Base prediction from market
        base_pred = implied if implied > 0 else yes_price
        
        # History momentum (if available)
        momentum = 0.0
        if history and len(history) > 5:
            recent = history[-5:]
            prices = [pt.get("p", pt.get("price", 0.5)) for pt in recent]
            if len(prices) >= 2:
                # Linear trend
                n = len(prices)
                x_mean = (n - 1) / 2
                y_mean = sum(prices) / n
                numerator = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(prices))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                if denominator > 0:
                    slope = numerator / denominator
                    momentum = min(max(slope * 5, -0.1), 0.1)  # Cap momentum
        
        # Volume confidence (higher volume = more confident in price)
        vol_confidence = min(volume * 2, 1.0)
        
        # Spread penalty (wider spread = less confident)
        spread_penalty = min(spread * 5, 0.2)
        
        # Entropy uncertainty
        uncertainty_penalty = uncertainty * 0.15
        
        # Time decay (shorter horizon = more confident in mean reversion)
        time_factor = min(horizon / 720, 1.0)  # Normalize to 30 days
        
        # Combine all signals
        predicted = base_pred + momentum * (1 - spread_penalty)
        
        # Bayesian shrinkage toward 0.5 based on uncertainty
        alpha = vol_confidence * (1 - uncertainty_penalty - spread_penalty)
        alpha = max(0.1, alpha)
        predicted = alpha * predicted + (1 - alpha) * 0.5
        
        # Clamp
        predicted = max(0.01, min(0.99, predicted))
        
        # Confidence computation
        confidence = vol_confidence * (1 - uncertainty_penalty)
        confidence = max(0.05, min(0.95, confidence))
        
        # Energy score (lower = better prediction quality)
        energy = (
            uncertainty_penalty * 0.3 +
            spread_penalty * 0.2 +
            (1 - vol_confidence) * 0.25 +
            abs(momentum) * 0.15 +
            time_factor * 0.1
        )
        
        # Create mock latents for compatibility with SAE probe interface
        context_latent = features + [momentum, vol_confidence, energy]
        predicted_latent = [predicted, 1-predicted] + context_latent[2:]
        
        return PredictionOutput(
            event_description=event_desc,
            predicted_yes_probability=round(predicted, 6),
            confidence=round(confidence, 4),
            prediction_horizon_hours=horizon,
            context_latent=context_latent,
            predicted_latent=predicted_latent,
            energy_score=round(energy, 6),
            validation_signals={
                "volume_confidence": round(vol_confidence, 4),
                "momentum": round(momentum, 4),
                "uncertainty_penalty": round(uncertainty_penalty, 4),
                "spread_penalty": round(spread_penalty, 4),
                "time_factor": round(time_factor, 4),
                "base_price": round(base_pred, 4),
            },
        )
    
    def _full_jepa_predict(self, event_desc, features, history, horizon, similar):
        """Placeholder for full JEPA model integration."""
        log.warning("Full JEPA mode not yet implemented. Falling back to lightweight.")
        return self._lightweight_predict(event_desc, features, history, horizon, similar)
    
    def compute_prediction_error(
        self,
        prediction: PredictionOutput,
        actual_outcome: bool,
    ) -> dict:
        """Compute prediction accuracy metrics."""
        pred_prob = prediction.predicted_yes_probability
        actual_prob = 1.0 if actual_outcome else 0.0
        
        error = abs(pred_prob - actual_prob)
        log_loss = -(actual_prob * math.log(max(pred_prob, 1e-6)) + 
                     (1 - actual_prob) * math.log(max(1 - pred_prob, 1e-6)))
        
        brier_score = (pred_prob - actual_prob) ** 2
        
        return {
            "absolute_error": round(error, 6),
            "log_loss": round(log_loss, 6),
            "brier_score": round(brier_score, 6),
            "predicted": pred_prob,
            "actual": actual_prob,
            "energy_score": prediction.energy_score,
        }
    
    def register_reference(self, features: list[float]):
        """Register a reference state for drift detection."""
        self._reference_features = features
        log.info(f"Registered reference features: {len(features)} dimensions")
    
    def check_latent_drift(self, current_features: list[float]) -> float:
        """Check drift in latent space compared to reference."""
        if self._reference_features is None:
            return 0.0
        if len(current_features) != len(self._reference_features):
            return 1.0
        
        import numpy as np
        ref = np.array(self._reference_features, dtype=np.float32)
        cur = np.array(current_features, dtype=np.float32)
        
        norm = float(np.linalg.norm(ref) * np.linalg.norm(cur))
        if norm <= 0:
            return 0.0
        
        cosine_dist = 1.0 - np.dot(ref, cur) / norm
        return max(0.0, min(1.0, float(cosine_dist)))
