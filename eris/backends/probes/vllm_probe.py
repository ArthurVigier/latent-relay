"""
eris/backends/probes/vllm_probe.py
===================================

Stub — VLLMProbe is not yet implemented.

To implement:
  1. pip install vllm
  2. Inherit ProbeModel
  3. Implement probe(), probe_batch(), steer(), steer_batch() and the
     steering library methods following the same pattern as HFProbe.

Note: vLLM's LLM.encode() returns hidden states without text generation,
which makes it a natural fit for the ProbeModel interface.
"""

from eris.interfaces import ProbeModel


class VLLMProbe(ProbeModel):
    """ProbeModel backed by vLLM. Not yet implemented."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VLLMProbe is a stub. "
            "Implement it by following eris/backends/probes/hf_probe.py."
        )

    def probe(self, input_text, layers, pooling="last_token", centered=True):
        raise NotImplementedError

    def probe_batch(self, inputs, layers, pooling="last_token", centered=True):
        raise NotImplementedError

    def steer(self, input_text, direction, alpha, layers, mode="add"):
        raise NotImplementedError

    def steer_batch(self, inputs, direction, alpha, layers, mode="add"):
        raise NotImplementedError

    def save_direction(self, name, vector):
        raise NotImplementedError

    def load_direction(self, name):
        raise NotImplementedError

    def list_directions(self):
        raise NotImplementedError
