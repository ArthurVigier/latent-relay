from __future__ import annotations

import numpy as np

from eris.drift_detector import DriftDetector
from eris.interfaces import OrchestratorLLM, ReasoningStep
from eris.orchestrator import ERISOrchestrator
from eris.sae_probe import ProbeOutput


class _SequenceProbe:
    def __init__(self, outputs):
        self.layers = [20]
        self._outputs = list(outputs)

    def probe(self, *_args, **_kwargs):
        return self._outputs.pop(0)


class _TwoStepLLM(OrchestratorLLM):
    def reason_step(self, problem: str, history: list[ReasoningStep], recalibration_context=None) -> ReasoningStep:
        if not history:
            return ReasoningStep(content="[Step 1] explore", step_idx=1, uncertainty=0.1)
        return ReasoningStep(content="[Final Answer] done", step_idx=2, uncertainty=0.1)

    def interpret_activations(self, activations_description: str, drift_report, problem_context: str):
        raise NotImplementedError

    def estimate_uncertainty(self, reasoning_step: ReasoningStep) -> float:
        return reasoning_step.uncertainty


def _probe_output(active, labels, raw):
    return ProbeOutput(
        layer=20,
        active_feature_indices=active,
        active_feature_values=[1.0] * len(active),
        all_active_indices=active,
        n_active=len(active),
        raw_activations=np.asarray(raw, dtype=np.float32),
        elapsed_s=0.0,
        active_feature_labels=labels,
    )


def test_consultation_record_captures_labeled_concepts():
    reference = {20: _probe_output([1, 2], ["proof_state", "algebraic_manipulation"], [1.0, 0.0])}
    current = {20: _probe_output([2, 9], ["algebraic_manipulation", "modular_reasoning"], [0.0, 1.0])}

    orch = ERISOrchestrator(
        probe=_SequenceProbe([reference, current]),
        drift_detector=DriftDetector(threshold=0.1, window=1, jaccard_weight=1.0, cosine_weight=0.0),
        llm=_TwoStepLLM(),
    )

    result = orch.run("problem", max_steps=2, checkpoint_every=1, top_k=2)

    assert result.n_consultations == 1
    consultation = result.consultations[0]
    assert consultation.lost_concepts == ["20:1:proof_state"]
    assert consultation.gained_concepts == ["20:9:modular_reasoning"]
