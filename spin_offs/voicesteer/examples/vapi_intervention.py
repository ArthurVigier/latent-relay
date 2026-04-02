import sys
import os
import asyncio
import numpy as np

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steering.emotion_vectors import EmotionVectors
from real_time_stream.websocket_probe import WebsocketProbe

# ================================
# MOCKING AN ANGRY CUSTOMER SCENARIO
# ================================

async def run_vapi_intervention():
    print("Starting VoiceSteer Simulation: Vapi.ai Angry Customer Callback")
    print("-" * 60)

    dim = 128
    vectors = EmotionVectors(dim=dim)
    probe = WebsocketProbe(vectors, threshold=0.55, intensity=2.0)

    # Simulate 50 tokens of conversation
    num_tokens = 50
    base_latents = np.random.randn(num_tokens, dim)
    
    # Introduce "Frustration" into the model's latents around token 15
    # (The AI starts getting defensive)
    frustration_vec = vectors.get_vector('Frustration')
    for i in range(15, 30):
        base_latents[i] += frustration_vec * 15.0 
        base_latents[i] /= np.linalg.norm(base_latents[i])  # keep it normalized

    print("[Vapi Agent]: Start talking...")

    async def streaming_callback(token_idx, steered_frame, emotion):
        # Callback simulated per token generated
        if emotion == "Frustration" and not probe.steering_active:
            print(f"\\n🚨 [Token {token_idx}] CRITICAL EMOTION DETECTED: Frustration!")
            print(f"   [System] Initiating Emotional Control Override within 50ms...")
            probe.set_steering(target_concept="Empathy", reduce_concept="Frustration")

        # After applying steering, simulate that the bot output changes
        if probe.steering_active and token_idx > 25:
            # Let's check how the new frame looks by doing a quick sim test
            test_empathy = np.dot(steered_frame / np.linalg.norm(steered_frame), vectors.get_vector('Empathy'))
            if test_empathy > 0.3: # Lowered threshold
                 print(f"   [Token {token_idx}] Bot output adapting: Calming down due to Empathy vector (+{test_empathy:.2f}).")
                 
            # Turn off after a while
            if token_idx == 40:
                print(f"\\n✅ [Token {token_idx}] Emotional stability restored.")
                probe.reset_steering()

    # Run the stream through our proxy
    await probe.simulate_stream(base_latents, streaming_callback)

    print("-" * 60)
    print("Simulation Complete.")

if __name__ == "__main__":
    asyncio.run(run_vapi_intervention())
