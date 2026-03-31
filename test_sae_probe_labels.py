from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eris.sae_probe import ProbeOutput, SAEProbe


def test_load_labels_supports_flat_and_per_layer_formats(tmp_path):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "12": "global_concept",
                "20": {"99": "layer_twenty_concept"},
            }
        ),
        encoding="utf-8",
    )

    probe = SAEProbe.__new__(SAEProbe)
    loaded = probe._load_labels(str(labels_path))

    assert loaded[None][12] == "global_concept"
    assert loaded[20][99] == "layer_twenty_concept"


def test_labels_for_indices_prefers_layer_specific_labels_and_falls_back_to_global():
    probe = SAEProbe.__new__(SAEProbe)
    probe._labels = {
        None: {7: "global_label", 8: "shared_global"},
        20: {8: "layer_override", 9: "layer_only"},
    }

    labels = probe.labels_for_indices(20, [7, 8, 9, 10])

    assert labels == ["global_label", "layer_override", "layer_only", None]


def test_build_probe_output_includes_active_feature_labels():
    probe = SAEProbe.__new__(SAEProbe)
    probe._labels = {
        None: {5: "global_five"},
        20: {3: "layer_three", 7: "layer_seven"},
    }

    features = np.asarray([0.0, 0.0, 1.2, 0.0, 0.0, 0.9, 0.0, 1.7], dtype=np.float32)
    acts = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    out = probe._build_probe_output(layer_idx=20, raw_activations=acts, features=features, elapsed=0.12, top_k=3)

    assert isinstance(out, ProbeOutput)
    assert out.active_feature_indices == [7, 2, 5]
    assert out.active_feature_labels == ["layer_seven", None, "global_five"]
    assert out.n_active == 3
