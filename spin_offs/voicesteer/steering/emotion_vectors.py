import numpy as np

# Mocking Sparse Autoencoder (SAE) feature groupings / steering vectors
# In a real system, these would be trained vectors matching model dimensions (e.g., 4096)

class EmotionVectors:
    def __init__(self, dim=128):
        self.dim = dim
        self._seed_vectors()

    def _seed_vectors(self):
        """Generate mock normalized vectors representing concepts in latent space."""
        np.random.seed(42)  # For reproducible mocks
        
        self.vectors = {
            'Frustration': self._random_norm_vector(),
            'Empathy': self._random_norm_vector(),
            'Aggressive_Sales': self._random_norm_vector(),
            'Confusion': self._random_norm_vector(),
            'Politeness': self._random_norm_vector()
        }

    def _random_norm_vector(self):
        vec = np.random.randn(self.dim)
        return vec / np.linalg.norm(vec)

    def get_vector(self, name):
        if name not in self.vectors:
            raise ValueError(f"Unknown vector: {name}")
        return self.vectors[name]

    def calculate_adjustment(self, target_state: str, current_state: str, intensity: float = 1.0) -> np.ndarray:
        """
        Calculates the steering delta to apply to latents.
        For example: calculate_adjustment('Empathy', 'Frustration', 1.5)
        Returns: (Empathy * intensity) - (Frustration * intensity)
        """
        target = self.get_vector(target_state)
        current_to_reduce = self.get_vector(current_state)
        
        # We add the target concept and subtract the unwanted concept
        steering_delta = (target * intensity) - (current_to_reduce * intensity)
        return steering_delta

def detect_emotion_drift(latent_frame: np.ndarray, vectors: EmotionVectors, threshold: float = 0.6) -> str:
    """
    Mock detection: In reality, we project latents onto SAE features.
    Here we just do a cosine similarity with our mock vectors.
    """
    # Normalize latent fram for cosine sim
    norm_frame = latent_frame / (np.linalg.norm(latent_frame) + 1e-9)
    
    max_sim = -1
    detected = None
    
    for name, vec in vectors.vectors.items():
        sim = np.dot(norm_frame, vec)
        if sim > max_sim:
            max_sim = sim
            detected = name
            
    if max_sim > threshold:
        return detected
    return "Neutral"
