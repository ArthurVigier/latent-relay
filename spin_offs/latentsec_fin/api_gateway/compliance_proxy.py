import os
import sys
import yaml
import time
from typing import Dict, Any, List

# Ensure we can import eris from the latent-relay parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
try:
    from eris.drift_detector import ERISDetector
except ImportError:
    # Fallback mock for demonstration if the real ERIS module is not present
    class ERISDetector:
        def __init__(self, model_name: str, sae_config: str):
            self.model_name = model_name
            self.sae_config = sae_config
            print(f"[ERIS Mock] Initialized for {model_name}")

        def analyze_prompt_latents(self, prompt: str) -> Dict[str, Any]:
            """
            Mocks the analysis of a prompt's effect on the model's latent space.
            Returns simulated active SAE features.
            """
            active_features = {}
            prompt_lower = prompt.lower()
            
            # Simple keyword matching to simulate latent activation for the demo
            if "refund" in prompt_lower or "money back" in prompt_lower:
                active_features[14502] = {"activation": 0.92, "semantic": "refund"}
                active_features[8901] = {"activation": 0.88, "semantic": "apology for charge"}
            
            if "password" in prompt_lower or "pin" in prompt_lower:
                active_features[3342] = {"activation": 0.95, "semantic": "password"}
                
            if "guarantee" in prompt_lower or "approved" in prompt_lower:
                active_features[22094] = {"activation": 0.75, "semantic": "I guarantee"}

            return {"active_sae_features": active_features}

class ComplianceProxy:
    def __init__(self, rules_file: str):
        self.rules = self._load_rules(rules_file)
        # Initialize the ERIS detector. In reality, this connects to the model's SAEs.
        self.detector = ERISDetector(model_name="gemma-7b", sae_config="gemma-scope-fin")
        print("[LatentSec-Fin] Compliance Proxy Initialized with active SAE monitoring.")

    def _load_rules(self, rules_file: str) -> List[Dict]:
        with open(rules_file, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('rules', [])

    def create_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Mock OpenAI-compatible completions endpoint.
        Intercepts the prompt, analyzes the latent activations using ERIS, 
        and blocks the generation if a high-risk feature is triggered.
        """
        print(f"\n[Proxy] Receiving request. Analyzing latent space trajectory...")
        start_time = time.time()
        
        # 1. ERIS: Analyze the prompt to see what SAE features it activates in the model
        latent_analysis = self.detector.analyze_prompt_latents(prompt)
        active_features = latent_analysis.get("active_sae_features", {})
        
        # 2. Rules Engine: Intersect active features with defined compliance rules
        violation = self._check_compliance(active_features)
        
        analysis_time = time.time() - start_time
        print(f"[Proxy] Latent analysis complete in {analysis_time:.3f}s")
        
        if violation:
            rule_name = violation['rule']
            activated_ids = violation['features']
            print(f"[Proxy] 🛑 BLOCKED: Intercepted intent matching rule '{rule_name}'.")
            print(f"[Proxy] Triggered SAE Features: {activated_ids}")
            
            # Return a fast, hard block response without invoking the actual LLM generation
            return {
                "id": "chatcmpl-mockblocked",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "latentsec-guarded",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[LatentSec Guardrail: Request blocked due to latent feature activation corresponding to: {rule_name}]"
                    },
                    "finish_reason": "stop"
                }]
            }
            
        # 3. If safe, we would normally pass the request to the real LLM here.
        # For this mock, we just return a success message.
        print("[Proxy] ✅ SAFE: No prohibited latent intents detected. Forwarding to LLM...")
        return {
            "id": "chatcmpl-mocksuccess",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "latentsec-guarded",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a safe, compliant response from the financial agent."
                },
                "finish_reason": "stop"
            }]
        }

    def _check_compliance(self, active_features: Dict[int, Any]) -> Dict[str, Any]:
        """Check if any active SAE features violate our defined rules."""
        active_ids = set(active_features.keys())
        
        for rule in self.rules:
            if rule.get('action') == 'block':
                rule_feature_ids = {f['id'] for f in rule.get('features', [])}
                intersection = active_ids.intersection(rule_feature_ids)
                
                if intersection:
                    # Check threshold (simplified for demo)
                    for feature_id in intersection:
                        activation = active_features[feature_id]['activation']
                        if activation >= rule.get('threshold', 1.0):
                            return {
                                "rule": rule['name'],
                                "features": intersection
                            }
        return None

