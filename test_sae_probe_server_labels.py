from __future__ import annotations

import numpy as np

from eris.sae_probe import ProbeOutput, serialize_probe_outputs


def test_serialize_probe_outputs_includes_feature_labels():
    payload = serialize_probe_outputs(
        {
            20: ProbeOutput(
                layer=20,
                active_feature_indices=[7, 5],
                active_feature_values=[1.7, 0.9],
                all_active_indices=[5, 7],
                n_active=2,
                raw_activations=np.asarray([0.1, 0.2], dtype=np.float32),
                elapsed_s=0.12,
                active_feature_labels=["layer_seven", "global_five"],
            )
        }
    )

    assert payload["20"]["active_feature_labels"] == ["layer_seven", "global_five"]
    assert payload["20"]["active_feature_indices"] == [7, 5]
