import time
import numpy as np
import asyncio
from steering.emotion_vectors import detect_emotion_drift

class WebsocketProbe:
    """
    Mocks a low-latency proxy sitting between the LLM inference server and 
    a Voice AI platform (like Vapi/Bland).
    """
    def __init__(self, emotion_vectors, threshold=0.6, intensity=1.5):
        self.emotion_vectors = emotion_vectors
        self.threshold = threshold
        self.intensity = intensity
        self.steering_active = False
        self.current_steering_delta = np.zeros(emotion_vectors.dim)

    def set_steering(self, target_concept, reduce_concept):
        """Activates steering by calculating the necessary vector delta."""
        self.current_steering_delta = self.emotion_vectors.calculate_adjustment(
            target_concept, reduce_concept, self.intensity
        )
        self.steering_active = True
        print(f"[Probe] ACTIVATING STEERING: +{target_concept} -{reduce_concept}")

    def reset_steering(self):
        self.steering_active = False
        self.current_steering_delta = np.zeros(self.emotion_vectors.dim)
        print("[Probe] Steering reset.")

    def process_latent_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Reads a latent frame from the LLM, detects if an unwanted emotion is rising,
        and applies the steering delta if active.
        """
        # 1. Probe for unwanted drift
        detected_emotion = detect_emotion_drift(frame, self.emotion_vectors, self.threshold)
        
        # 2. Apply steering if active
        if self.steering_active:
            # Inject steering 
            steered_frame = frame + self.current_steering_delta
            # Re-normalize to avoid breaking inference stability
            steered_frame = steered_frame * (np.linalg.norm(frame) / (np.linalg.norm(steered_frame) + 1e-9))
            return steered_frame, detected_emotion
        
        return frame, detected_emotion

    async def simulate_stream(self, token_latents_stream, callback):
        """Simulates a streaming connection processing tokens."""
        print("[WebsocketProbe] Connection established.")
        for i, latent_frame in enumerate(token_latents_stream):
            # Process frame
            steered_frame, emotion = self.process_latent_frame(latent_frame)
            
            # Send to callback 
            await callback(i, steered_frame, emotion)
            
            # Simulate latency (e.g. 5ms per token)
            await asyncio.sleep(0.005)
            
        print("[WebsocketProbe] Stream closed.")
