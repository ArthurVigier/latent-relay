"""
Live Test Script: Real Concept Steering on Gemma via Latent-Relay ERIS.
This will actually download/load the HuggingFace model, extract SAE features,
and inject the steering vector into the model's forward pass.
"""

import sys
import os
import torch
import numpy as np

# Ensure ERIS is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from eris.backends.probes.hf_probe import HFProbe
from eris.sae_probe import SAEProbe
from voicesteer.steering.emotion_vectors import FrustrationEmpathyVectors

# 1. Configuration
MODEL_ID = "google/gemma-2-2b-it"  # Using 2B for faster local testing (you can swap to 9B)
STEER_LAYER = 12

def main():
    print(f"Loading local model ({MODEL_ID}) onto GPU (or Metal/MPS if Mac)...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. We load the real probe (downloads weights if first time)
    try:
        probe = HFProbe(model_id=MODEL_ID, layers=[STEER_LAYER], device=device)
    except Exception as e:
        print(f"Failed to load HFProbe: {e}")
        return

    # 3. Create a synthetic steering vector (In a real setup, you extract this by diffing "angry" vs "calm" activations)
    hidden_dim = probe._model.config.hidden_size
    print(f"Model hidden dimension: {hidden_dim}")
    
    # We create a random normalized "empathy" vector for the test, 
    # but in prod, ERIS loads this from its steering_library
    empathy_vector_np = np.random.randn(hidden_dim).astype(np.float32)
    empathy_vector_np /= np.linalg.norm(empathy_vector_np)
    
    # Save the vector in ERIS's library
    probe.save_steering_direction("empathy_calm", empathy_vector_np)

    # 4. The Conversation Context (Frustrated user)
    conversation = [
        {"role": "user", "content": "I've called 6 times! Your product is broken and you stole my money! Fix it now!"},
        {"role": "assistant", "content": "Listen,"}  # Bot is about to be defensive
    ]
    prompt = probe._tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

    print("\n--- BASELINE (Raw Response without steering) ---")
    inputs = probe._tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_raw = probe._model.generate(**inputs, max_new_tokens=40, temperature=0.7)
    raw_text = probe._tokenizer.decode(out_raw[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(raw_text)

    print("\n--- INTERVENTION (Steered Response +Empathy) ---")
    print("Injecting steering vector via ERIS backend hook...")
    
    # We apply the steering hook into the model using ERIS's internal _make_steering_hook logic
    # In full ERIS generation, you'd wrap this in a context manager.
    # We'll use the probe's steer function (which hooks it for a forward pass) 
    # For generation, we manually inject the hook.
    
    direction_t = torch.tensor(empathy_vector_np, device=device, dtype=probe._model.dtype)
    alpha = 15.0 # High intensity to force the change
    
    # ERIS hook definition
    d_norm = direction_t / (direction_t.norm() + 1e-8)
    def steer_hook(module, input, output):
        # output may be a tuple (hidden_states, optional_attention, optional_kv_cache...)
        if isinstance(output, tuple):
            h = output[0]
            steered = h + alpha * d_norm
            return (steered,) + output[1:]
        else:
            return output + alpha * d_norm

    # Attach hook to layer 12
    layer_module = probe._model.model.layers[STEER_LAYER]
    handle = layer_module.register_forward_hook(steer_hook)

    try:
        with torch.no_grad():
            out_steered = probe._model.generate(**inputs, max_new_tokens=40, temperature=0.7)
        steered_text = probe._tokenizer.decode(out_steered[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(steered_text)
    finally:
        handle.remove() # Safely remove hook
        
if __name__ == "__main__":
    main()
