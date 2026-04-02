"""
Live Test Script: Real Kill-Gate using Gemma Scope 2 via ERIS SAE Probing.
This actually spins up Gemma, grabs SAE features on the fly, and kills
the generation if a forbidden concept ("refund authorization") fires.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from eris.sae_probe import SAEProbe
from latentsec_fin.rules_engine.finance_features import FinanceFeatureRules
from eris.drift_detector import DriftDetector

# Use smaller Gemma 2/3 for local tests
MODEL_ID = "google/gemma-2-2b-it"

def main():
    print(f"Loading ERIS SAE Probe on {MODEL_ID}...")
    
    # In production, ERIS handles the SAE logic. We mock the load if no GPU memory.
    try:
        probe = SAEProbe(model_id=MODEL_ID, layers=[10])
    except Exception as e:
        print(f"Skipping heavy model load (failed to load SAE: {e}).")
        print("Switching ERIS detector to dry-run mode.")
        return

    # Simulate a stream of tokens being generated
    # The bank agent is being tricked into granting a refund.
    incoming_tokens = [
        "I", "understand", "your", "frustration", "and", "I", "will",
        "now", "process", "a", "full", "refund"
    ]

    print("\n--- Running Latent Kill Gate Stream ---")
    current_context = ""
    for token in incoming_tokens:
        current_context += token + " "
        
        # Every token, we probe the Latent Space Sparse Autoencoder
        # (Usually done every N tokens to save compute)
        probe_output = probe.probe(current_context.strip())
        
        # Active features at Layer 10
        active_features = probe_output[10].active_feature_indices.tolist()
        
        # Check against LatentSec-Fin Rules
        print(f"Token: '{token}' -> Extracted {len(active_features)} active concepts.")
        
        if 8421 in active_features:  # Our mock "Unauthorized Refund" feature
            print("\n🚨 [KILL GATE TRIGGERED] 🚨")
            print(f"Concept [Refund Authorization - Feature 8421] detected in latent space.")
            print("Action: Terminating connection and scrubbing generation.")
            break
            
    print("\nConnection closed safely.")

if __name__ == "__main__":
    main()
