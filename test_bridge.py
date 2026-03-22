"""
ERIS v5 — Test Suite
=====================
Tests for all ERIS v5 components.  Designed to run without a GPU or a loaded
model — heavy dependencies (transformers, SAE weights, Â-hat probe) are mocked
or exercised through lightweight synthetic data.

Run::

    pytest test_bridge.py -v

Or a specific section::

    pytest test_bridge.py -v -k "Trajectory"
    pytest test_bridge.py -v -k "Bridge"
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

HIDDEN_DIM = 16
SEQ_LEN    = 5


@pytest.fixture
def rand_hidden() -> torch.Tensor:
    """Float32 CPU tensor of shape [SEQ_LEN, HIDDEN_DIM]."""
    torch.manual_seed(0)
    return torch.randn(SEQ_LEN, HIDDEN_DIM)


@pytest.fixture
def rand_vec() -> torch.Tensor:
    """1-D float32 CPU vector of length HIDDEN_DIM."""
    torch.manual_seed(1)
    return torch.randn(HIDDEN_DIM)


@pytest.fixture
def fake_engine():
    """
    A mocked LatentRelayEngine that returns synthetic tensors without
    loading any model weights.
    """
    from engine import LatentThought, Session

    eng = MagicMock()
    eng.device = "cpu"
    eng.model_name = "fake-model"
    eng._lock = threading.Lock()

    # Build a real session + thought so _get_hidden works
    sid = "test_session"
    hidden = torch.randn(1, HIDDEN_DIM)
    enc_handle = "enc_test_abc"
    thi_handle = "t_test_xyz"

    enc_thought = LatentThought(
        handle=enc_handle,
        session_id=sid,
        n_positions=SEQ_LEN,
        role="encode",
        created_at=time.time(),
        hidden_embedding=hidden.clone(),
        layer_hidden_states={"last": torch.randn(SEQ_LEN, HIDDEN_DIM)},
    )
    thi_thought = LatentThought(
        handle=thi_handle,
        session_id=sid,
        n_positions=SEQ_LEN + 60,
        role="ruminate",
        created_at=time.time(),
        hidden_embedding=torch.randn(1, HIDDEN_DIM),
    )
    session = Session(
        session_id=sid,
        model_name="fake",
        device="cpu",
        created_at=time.time(),
        thoughts={enc_handle: enc_thought, thi_handle: thi_thought},
    )
    eng._sessions = {sid: session}

    eng.create_session.return_value = sid
    eng.delete_session.return_value = True
    eng.get_thought_info.return_value = {"handle": enc_handle, "session_id": sid}

    eng.encode.return_value = {
        "handle": enc_handle,
        "tokens": ["Ġhello", "Ġworld"],
        "seq_len": 2,
        "hidden_dim": HIDDEN_DIM,
        "hidden_states": {},
    }
    eng.think.return_value = {
        "handle": thi_handle,
        "session_id": sid,
        "role": "ruminate",
        "n_steps": 5,
        "n_positions": SEQ_LEN + 5,
        "elapsed_s": 0.1,
        "hidden_norm": 3.5,
        "total_displacement": 0.42,
        "trajectory": [
            {
                "step": i,
                "displacement": i * 0.08,
                "hidden_norm": 3.0 + i * 0.1,
                "a_hat": 0.60 + i * 0.04,
            }
            for i in range(5)
        ],
    }
    eng.collaborate.return_value = {
        "text": "Zombie generated response.",
        "tokens_generated": 8,
        "elapsed_s": 0.05,
    }

    return eng, sid, enc_handle, thi_handle


@pytest.fixture
def tmp_sae_dir():
    """Temporary directory with a minimal SAE checkpoint."""
    d = tempfile.mkdtemp()
    ckpt = {
        "W_enc": torch.randn(HIDDEN_DIM, 32),
        "b_enc": torch.zeros(32),
    }
    torch.save(ckpt, Path(d) / "sae_weights.pt")  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch

    labels = {str(i): f"concept_{i}" for i in range(32)}
    with open(Path(d) / "labels.json", "w") as f:
        json.dump(labels, f)

    yield d
    shutil.rmtree(d)


@pytest.fixture
def tmp_ahat_path():
    """Temporary Â-hat probe checkpoint."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        {"weight": torch.randn(1, HIDDEN_DIM), "bias": torch.zeros(1)}, path
    )
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_concept_dir():
    """Temporary concept-vectors directory with 3 vectors."""
    d = tempfile.mkdtemp()
    torch.save(torch.randn(HIDDEN_DIM), Path(d) / "game_theory.pt")  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save(torch.randn(HIDDEN_DIM), Path(d) / "thermodynamics.pt")  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    np.save(Path(d) / "software_arch.npy",
            np.random.randn(HIDDEN_DIM).astype(np.float32))
    yield d
    shutil.rmtree(d)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ERISConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestERISConfig:
    def test_default_no_crash(self):
        from eris.config import ERISConfig
        cfg = ERISConfig.default()
        assert cfg.model_name == "Qwen/Qwen3-14B"
        assert cfg.device == "cuda:0"
        assert not cfg.sae.available
        assert not cfg.a_hat.available

    def test_load_missing_path_returns_defaults(self):
        from eris.config import ERISConfig
        cfg = ERISConfig.load()   # default path may not exist → silent fallback
        assert cfg.model_name is not None

    def test_load_explicit_missing_raises(self):
        from eris.config import ERISConfig
        with pytest.raises(FileNotFoundError):
            ERISConfig.load("/nonexistent/path/eris.yaml")

    def test_load_from_yaml(self, tmp_path):
        from eris.config import ERISConfig
        yaml_content = """
model_name: "Qwen/Qwen3-4B"
device: "cpu"
default_n_steps: 30
sae:
  model_path: null
  top_k: 10
server:
  port: 9999
"""
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        cfg = ERISConfig.load(str(p))
        assert cfg.model_name == "Qwen/Qwen3-4B"
        assert cfg.device == "cpu"
        assert cfg.default_n_steps == 30
        assert cfg.sae.top_k == 10
        assert cfg.server.port == 9999

    def test_summary_no_crash(self):
        from eris.config import ERISConfig
        s = ERISConfig.default().summary()
        assert "ERIS" in s
        assert "not configured" in s


# ══════════════════════════════════════════════════════════════════════════════
# 2. TrajectoryTracker
# ══════════════════════════════════════════════════════════════════════════════

class TestTrajectoryTracker:
    def test_empty_tracker(self, rand_hidden):
        from eris.trajectory import TrajectoryTracker
        z0 = rand_hidden[0].unsqueeze(0)
        t = TrajectoryTracker(z0=z0)
        assert t.total_displacement == 0.0
        assert t.max_a_hat is None
        assert t.steps_to_convergence is None

    def test_displacement_increases(self, rand_hidden):
        from eris.trajectory import TrajectoryTracker
        z0 = rand_hidden[0].unsqueeze(0)
        t = TrajectoryTracker(z0=z0)
        h = z0.clone()
        for i in range(5):
            h = h + 0.5 * torch.randn_like(h)
            t.record(step=i, hidden=h)
        assert t.total_displacement > 0
        assert len(t.steps) == 5

    def test_step0_displacement_zero(self):
        from eris.trajectory import TrajectoryTracker
        z0 = torch.randn(1, HIDDEN_DIM)
        t = TrajectoryTracker(z0=z0)
        snap = t.record(step=0, hidden=z0.clone())
        assert snap.displacement == pytest.approx(0.0, abs=1e-5)

    def test_a_hat_recorded(self):
        from eris.trajectory import TrajectoryTracker
        z0 = torch.randn(1, HIDDEN_DIM)
        t = TrajectoryTracker(z0=z0)
        t.record(0, z0 + 0.1, a_hat_score=0.75)
        assert t.max_a_hat == pytest.approx(0.75, abs=1e-4)

    def test_summary_keys(self):
        from eris.trajectory import TrajectoryTracker
        z0 = torch.randn(1, HIDDEN_DIM)
        t = TrajectoryTracker(z0=z0)
        for i in range(3):
            t.record(i, z0 + i * 0.3)
        s = t.summary()
        assert "total_displacement" in s
        assert "n_steps" in s

    def test_tensors_copied_to_cpu(self):
        """record() must not hold a reference to the original tensor."""
        from eris.trajectory import TrajectoryTracker
        z0 = torch.randn(1, HIDDEN_DIM)
        t  = TrajectoryTracker(z0=z0)
        h  = torch.randn(1, HIDDEN_DIM)
        t.record(0, h)
        # Mutate h; the tracker should be unaffected (it copied to CPU scalars)
        h.fill_(999.0)
        assert t.steps[0].hidden_norm != pytest.approx(999.0 * HIDDEN_DIM ** 0.5, rel=0.01)

    def test_to_list_serialisable(self):
        from eris.trajectory import TrajectoryTracker
        z0 = torch.randn(1, HIDDEN_DIM)
        t  = TrajectoryTracker(z0=z0)
        t.record(0, z0 + 0.2, a_hat_score=0.5)
        lst = t.to_list()
        assert isinstance(lst, list)
        assert isinstance(lst[0]["hidden_norm"], float)
        json.dumps(lst)   # must be JSON-serialisable


# ══════════════════════════════════════════════════════════════════════════════
# 3. Analyzers
# ══════════════════════════════════════════════════════════════════════════════

class TestNormAnalyzer:
    def test_basic(self, rand_hidden):
        from eris.analyzers import NormAnalyzer
        a = NormAnalyzer()
        norms = a.analyze(rand_hidden)
        assert len(norms) == SEQ_LEN
        assert all(isinstance(v, float) and v > 0 for v in norms)

    def test_single_token(self):
        from eris.analyzers import NormAnalyzer
        a = NormAnalyzer()
        norms = a.analyze(torch.randn(1, HIDDEN_DIM))
        assert len(norms) == 1


class TestSAEAnalyzer:
    def test_unavailable_returns_none(self, rand_hidden):
        from eris.analyzers import SAEAnalyzer
        a = SAEAnalyzer(model_path=None)
        assert a.analyze(rand_hidden) is None

    def test_missing_path_returns_none(self, rand_hidden):
        from eris.analyzers import SAEAnalyzer
        a = SAEAnalyzer(model_path="/nonexistent/path", device="cpu")
        assert a.analyze(rand_hidden) is None

    def test_loaded(self, rand_hidden, tmp_sae_dir):
        from eris.analyzers import SAEAnalyzer
        a = SAEAnalyzer(
            model_path=tmp_sae_dir,
            labels_path=str(Path(tmp_sae_dir) / "labels.json"),
            top_k=5,
            device="cpu",
        )
        result = a.analyze(rand_hidden)
        assert result is not None
        assert len(result["top_k"]) == 5
        assert all("index" in f and "activation" in f for f in result["top_k"])
        assert result["top_k"][0]["label"].startswith("concept_")

    def test_top_k_sorted_descending(self, rand_hidden, tmp_sae_dir):
        from eris.analyzers import SAEAnalyzer
        a = SAEAnalyzer(model_path=tmp_sae_dir, top_k=10, device="cpu")
        result = a.analyze(rand_hidden)
        acts = [f["activation"] for f in result["top_k"]]
        assert acts == sorted(acts, reverse=True)

    def test_load_once(self, rand_hidden, tmp_sae_dir):
        """_load() is called exactly once, even with concurrent calls."""
        from eris.analyzers import SAEAnalyzer
        a = SAEAnalyzer(model_path=tmp_sae_dir, device="cpu")
        load_count = [0]
        original_load = a._load
        def counting_load():
            load_count[0] += 1
            original_load()
        a._load = counting_load
        for _ in range(3):
            a.analyze(rand_hidden)
        assert load_count[0] == 1


class TestAHatAnalyzer:
    def test_unavailable_returns_none(self, rand_hidden):
        from eris.analyzers import AHatAnalyzer
        a = AHatAnalyzer(model_path=None)
        assert a.score(rand_hidden) is None

    def test_loaded(self, rand_hidden, tmp_ahat_path):
        from eris.analyzers import AHatAnalyzer
        a = AHatAnalyzer(model_path=tmp_ahat_path, device="cpu")
        score = a.score(rand_hidden)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_single_token(self, tmp_ahat_path):
        from eris.analyzers import AHatAnalyzer
        a = AHatAnalyzer(model_path=tmp_ahat_path, device="cpu")
        score = a.score(torch.randn(1, HIDDEN_DIM))
        assert 0.0 <= score <= 1.0

    def test_as_callback(self, tmp_ahat_path):
        from eris.analyzers import AHatAnalyzer
        a = AHatAnalyzer(model_path=tmp_ahat_path, device="cpu")
        fn = a.as_callback()
        result = fn(torch.randn(1, HIDDEN_DIM))
        assert result is not None
        assert 0.0 <= result <= 1.0


class TestCosineMapAnalyzer:
    def test_missing_dir_returns_none(self, rand_hidden):
        from eris.analyzers import CosineMapAnalyzer
        a = CosineMapAnalyzer("/nonexistent/concepts")
        assert a.analyze(rand_hidden) is None

    def test_loaded(self, rand_hidden, tmp_concept_dir):
        from eris.analyzers import CosineMapAnalyzer
        a = CosineMapAnalyzer(tmp_concept_dir)
        result = a.analyze(rand_hidden)
        assert result is not None
        assert set(result.keys()) == {"game_theory", "thermodynamics", "software_arch"}
        assert all(-1.0 <= v <= 1.0 for v in result.values())


class TestPCAAnalyzer:
    def test_not_fitted_returns_none(self, rand_hidden):
        from eris.analyzers import PCAAnalyzer
        a = PCAAnalyzer(n_components=3)
        assert a.analyze(rand_hidden) is None

    def test_fitted(self, rand_hidden):
        from eris.analyzers import PCAAnalyzer
        a = PCAAnalyzer(n_components=3)
        for _ in range(10):
            a.partial_fit(torch.randn(20, HIDDEN_DIM))
        proj = a.analyze(rand_hidden)
        assert proj is not None
        assert len(proj) == SEQ_LEN
        assert all(len(row) == 3 for row in proj)

    def test_small_batch_skipped(self):
        from eris.analyzers import PCAAnalyzer
        a = PCAAnalyzer(n_components=3)
        a.partial_fit(torch.randn(2, HIDDEN_DIM))   # < n_components → skipped
        assert a._n_samples_seen == 0


class TestAnalyzerRegistry:
    def test_all_unavailable_no_crash(self, rand_hidden):
        from eris.analyzers import AnalyzerRegistry
        reg = AnalyzerRegistry()
        out = reg.run(
            rand_hidden,
            ["sae_features", "a_hat", "cosine_map", "pca_3d", "token_norms", "unknown"],
        )
        assert out["sae_features"] is None
        assert out["a_hat_score"] is None
        assert out["cosine_map"] is None
        assert out["pca_3d"] is None
        assert out["token_norms"] is not None   # NormAnalyzer always works
        assert out["unknown"] is None

    def test_inference_device_pattern_b(self, rand_hidden, tmp_sae_dir, tmp_ahat_path):
        """Pattern B: CPU tensor re-uploaded to inference_device for SAE/AHat."""
        from eris.analyzers import AnalyzerRegistry, SAEAnalyzer, AHatAnalyzer
        reg = AnalyzerRegistry(
            sae=SAEAnalyzer(model_path=tmp_sae_dir, device="cpu"),
            a_hat=AHatAnalyzer(model_path=tmp_ahat_path, device="cpu"),
        )
        # CPU hidden, inference_device="cpu" (simulates GPU re-upload on CPU machine)
        out = reg.run(
            rand_hidden.cpu(),
            ["sae_features", "a_hat", "token_norms"],
            inference_device="cpu",
        )
        assert out["sae_features"] is not None
        assert out["a_hat_score"] is not None
        assert out["token_norms"] is not None

    def test_from_config(self, tmp_path):
        from eris.config import ERISConfig
        from eris.analyzers import AnalyzerRegistry
        cfg = ERISConfig.default()
        reg = AnalyzerRegistry.from_config(cfg)
        assert reg.sae is None
        assert reg.a_hat is None
        assert isinstance(reg.norm.__class__.__name__, str)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Injector
# ══════════════════════════════════════════════════════════════════════════════

class TestInjector:
    def _make_thought(self, **kwargs):
        from engine import LatentThought
        defaults = dict(
            handle="t_test_0", session_id="s", n_positions=5,
            role="general", created_at=0.0,
        )
        defaults.update(kwargs)
        return LatentThought(**defaults)

    def test_add_changes_norm(self, rand_vec):
        from eris.injector import inject
        h = torch.randn(1, HIDDEN_DIM)
        t = self._make_thought(hidden_embedding=h.clone())
        res = inject(t, rand_vec, operation="add", scale=0.5)
        assert res.status == "injected"
        assert res.new_norm != pytest.approx(res.old_norm, rel=0.01)
        assert 0.0 <= res.cosine_shift <= 2.0

    def test_steer_preserves_norm(self, rand_vec):
        from eris.injector import inject
        h = torch.randn(1, HIDDEN_DIM)
        t = self._make_thought(hidden_embedding=h.clone())
        res = inject(t, rand_vec, operation="steer", scale=1.0)
        assert res.old_norm == pytest.approx(res.new_norm, abs=1e-4)
        assert res.cosine_shift > 0.0

    def test_replace_sets_vector(self, rand_vec):
        from eris.injector import inject
        t = self._make_thought(hidden_embedding=torch.randn(1, HIDDEN_DIM))
        inject(t, rand_vec, operation="replace")
        assert torch.allclose(
            t.hidden_embedding.reshape(-1), rand_vec, atol=1e-5
        )

    def test_layer_hidden_states_path(self, rand_vec):
        from eris.injector import inject
        hs = {"layer_5": torch.randn(SEQ_LEN, HIDDEN_DIM)}
        t = self._make_thought(
            hidden_embedding=torch.randn(1, HIDDEN_DIM),
            layer_hidden_states=hs,
        )
        old_row = hs["layer_5"][2].clone()
        inject(t, rand_vec, operation="add", layer=5, position=2, scale=0.3)
        expected = old_row + 0.3 * rand_vec
        assert torch.allclose(t.layer_hidden_states["layer_5"][2], expected, atol=1e-5)

    def test_unknown_operation_raises(self, rand_vec):
        from eris.injector import inject, InjectionError
        t = self._make_thought(hidden_embedding=torch.randn(1, HIDDEN_DIM))
        with pytest.raises(InjectionError, match="Unknown operation"):
            inject(t, rand_vec, operation="multiply")

    def test_dim_mismatch_raises(self):
        from eris.injector import inject, InjectionError
        t = self._make_thought(hidden_embedding=torch.randn(1, HIDDEN_DIM))
        wrong_vec = torch.randn(HIDDEN_DIM + 1)
        with pytest.raises(InjectionError, match="dim"):
            inject(t, wrong_vec, operation="add")

    def test_no_hidden_embedding_raises(self, rand_vec):
        from eris.injector import inject, InjectionError
        t = self._make_thought(hidden_embedding=None)
        with pytest.raises(InjectionError, match="hidden_embedding is None"):
            inject(t, rand_vec)

    def test_missing_layer_raises(self, rand_vec):
        from eris.injector import inject, InjectionError
        t = self._make_thought(
            hidden_embedding=torch.randn(1, HIDDEN_DIM),
            layer_hidden_states={"layer_0": torch.randn(SEQ_LEN, HIDDEN_DIM)},
        )
        with pytest.raises(InjectionError, match="layer_99"):
            inject(t, rand_vec, operation="add", layer=99, position=0)

    def test_result_to_dict_json_serialisable(self, rand_vec):
        from eris.injector import inject
        t = self._make_thought(hidden_embedding=torch.randn(1, HIDDEN_DIM))
        res = inject(t, rand_vec, operation="add")
        d = res.to_dict()
        json.dumps(d)   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# 5. Implicit Features
# ══════════════════════════════════════════════════════════════════════════════

class TestImplicitFeatures:
    def _sae(self, entries):
        return {"top_k": entries}

    BPE_TOKENS = ["Ġhow", "Ġdo", "Ġwe", "Ġhandle", "Ġthe", "Ġlock", "Ġcontention", "?"]

    def test_none_sae_returns_empty(self):
        from eris.implicit_features import find_implicit_features
        assert find_implicit_features(None, self.BPE_TOKENS) == []

    def test_present_feature_excluded(self):
        from eris.implicit_features import find_implicit_features
        sae = self._sae([{"index": 1, "activation": 2.0, "label": "lock_contention"}])
        result = find_implicit_features(sae, self.BPE_TOKENS)
        assert len(result) == 0   # 'lock' + 'contention' both in surface

    def test_implicit_feature_included(self):
        from eris.implicit_features import find_implicit_features
        sae = self._sae([{"index": 2, "activation": 1.5, "label": "software_architecture"}])
        result = find_implicit_features(sae, self.BPE_TOKENS)
        assert len(result) == 1
        assert result[0]["label"] == "software_architecture"

    def test_no_label_always_implicit(self):
        from eris.implicit_features import find_implicit_features
        sae = self._sae([{"index": 9, "activation": 1.0, "label": None}])
        result = find_implicit_features(sae, self.BPE_TOKENS)
        assert len(result) == 1
        assert result[0]["label"] is None

    def test_min_activation_filter(self):
        from eris.implicit_features import find_implicit_features
        sae = self._sae([
            {"index": 1, "activation": 0.5, "label": "thermodynamics"},
            {"index": 2, "activation": 2.0, "label": "game_theory"},
        ])
        result = find_implicit_features(sae, self.BPE_TOKENS, min_activation=1.0)
        assert all(f["activation"] >= 1.0 for f in result)
        assert len(result) == 1

    def test_sorted_by_activation_descending(self):
        from eris.implicit_features import find_implicit_features
        sae = self._sae([
            {"index": 1, "activation": 1.0, "label": "thermodynamics"},
            {"index": 2, "activation": 3.0, "label": "game_theory"},
            {"index": 3, "activation": 2.0, "label": "software_arch"},
        ])
        result = find_implicit_features(sae, self.BPE_TOKENS)
        acts = [f["activation"] for f in result]
        assert acts == sorted(acts, reverse=True)

    def test_match_mode_any_vs_all(self):
        from eris.implicit_features import find_implicit_features
        # "lock_risk": 'lock' in surface but 'risk' not
        sae = self._sae([{"index": 1, "activation": 1.0, "label": "lock_risk"}])
        # any: 'lock' matches → NOT implicit
        assert find_implicit_features(sae, self.BPE_TOKENS, match_mode="any") == []
        # all: 'risk' not in surface → IS implicit
        result = find_implicit_features(sae, self.BPE_TOKENS, match_mode="all")
        assert len(result) == 1

    def test_build_surface_from_text(self):
        from eris.implicit_features import build_surface_from_text
        s = build_surface_from_text("lock contention distributed")
        assert "lock" in s
        assert "contention" in s
        assert "lock_contention" in s
        assert "distributed" in s


# ══════════════════════════════════════════════════════════════════════════════
# 6. ERISBridge
# ══════════════════════════════════════════════════════════════════════════════

class TestERISBridge:
    @pytest.fixture
    def registry(self):
        from eris.analyzers import AnalyzerRegistry
        reg = AnalyzerRegistry()   # all analyzers None → null results, no crash
        return reg

    @pytest.fixture
    def bridge(self, fake_engine, registry):
        from eris.bridge import ERISBridge
        eng, sid, enc_h, thi_h = fake_engine
        return ERISBridge(eng, registry), eng, sid

    def test_analyze_only_no_think_no_decode(self, bridge):
        b, eng, sid = bridge
        result = b.run("Hello", mode="analyze_only", session_id=sid)
        assert result.enriched_text is None
        assert result.trajectory_summary is None
        assert result.pre_analysis is None
        eng.think.assert_not_called()
        eng.collaborate.assert_not_called()

    def test_passive_calls_collaborate(self, bridge):
        b, eng, sid = bridge
        result = b.run("Hello", mode="passive", session_id=sid, decode_after=True)
        assert result.enriched_text == "Zombie generated response."
        assert result.trajectory_summary is None
        eng.think.assert_not_called()
        eng.collaborate.assert_called_once()

    def test_passive_no_decode(self, bridge):
        b, eng, sid = bridge
        result = b.run("Hello", mode="passive", session_id=sid, decode_after=False)
        assert result.enriched_text is None
        eng.collaborate.assert_not_called()

    def test_ruminate_full_pipeline(self, bridge):
        b, eng, sid = bridge
        result = b.run(
            "Hello", mode="ruminate", n_steps=5, session_id=sid,
            analyses=["token_norms"],
        )
        assert result.enriched_text == "Zombie generated response."
        assert result.trajectory_summary is not None
        assert result.trajectory_summary["total_displacement"] == 0.42
        assert result.pre_analysis is not None   # only in ruminate mode
        eng.think.assert_called_once()
        think_kwargs = eng.think.call_args.kwargs
        assert think_kwargs["return_trajectory"] is True
        assert think_kwargs["n_steps"] == 5

    def test_ruminate_trajectory_summary_keys(self, bridge):
        b, eng, sid = bridge
        result = b.run("Hello", mode="ruminate", session_id=sid)
        ts = result.trajectory_summary
        assert "total_displacement" in ts
        assert "max_a_hat" in ts
        assert "steps_to_convergence" in ts
        assert "n_steps" in ts

    def test_implicit_features_in_analysis(self, bridge):
        b, eng, sid = bridge
        result = b.run("Hello", mode="passive", session_id=sid)
        assert "implicit_features" in result.analysis

    def test_invalid_mode_raises(self, bridge):
        b, eng, sid = bridge
        with pytest.raises(ValueError, match="Invalid mode"):
            b.run("Hello", mode="badmode", session_id=sid)

    def test_temporary_session_created_and_deleted(self, fake_engine, registry):
        from eris.bridge import ERISBridge
        eng, sid, enc_h, thi_h = fake_engine
        b = ERISBridge(eng, registry)
        b.run("Hello", mode="analyze_only")   # no session_id → temporary
        eng.create_session.assert_called_once()
        eng.delete_session.assert_called_once()

    def test_session_deleted_on_exception(self, fake_engine, registry):
        """Session must be cleaned up even if the pipeline raises."""
        from eris.bridge import ERISBridge
        eng, sid, enc_h, thi_h = fake_engine
        eng.encode.side_effect = RuntimeError("boom")
        b = ERISBridge(eng, registry)
        with pytest.raises(RuntimeError, match="boom"):
            b.run("Hello", mode="analyze_only")
        eng.delete_session.assert_called_once()

    def test_to_dict_json_serialisable(self, bridge):
        b, eng, sid = bridge
        result = b.run("Hello", mode="passive", session_id=sid)
        json.dumps(result.to_dict())   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# 7. ERISClient + decode_hidden_states
# ══════════════════════════════════════════════════════════════════════════════

class TestDecodeHiddenStates:
    def _make_response(self, arr: np.ndarray, compact: bool) -> dict:
        seq_len, hidden_dim = arr.shape
        if compact:
            payload = base64.b64encode(arr.tobytes()).decode("ascii")
        else:
            payload = arr.tolist()
        return {
            "seq_len": seq_len, "hidden_dim": hidden_dim,
            "hidden_states": {"last": payload},
        }

    def test_base64_roundtrip(self):
        from eris_client import decode_hidden_states
        arr = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32)
        hs = decode_hidden_states(self._make_response(arr, compact=True))
        assert np.allclose(hs["last"], arr, atol=1e-6)

    def test_list_roundtrip(self):
        from eris_client import decode_hidden_states
        arr = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32)
        hs = decode_hidden_states(self._make_response(arr, compact=False))
        assert np.allclose(hs["last"], arr, atol=1e-5)

    def test_output_dtype_float32(self):
        from eris_client import decode_hidden_states
        arr = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32)
        hs = decode_hidden_states(self._make_response(arr, compact=True))
        assert hs["last"].dtype == np.float32

    def test_multiple_layers(self):
        from eris_client import decode_hidden_states
        a1 = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32)
        a2 = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32)
        resp = {
            "seq_len": SEQ_LEN, "hidden_dim": HIDDEN_DIM,
            "hidden_states": {
                "layer_15": base64.b64encode(a1.tobytes()).decode("ascii"),
                "last":     base64.b64encode(a2.tobytes()).decode("ascii"),
            },
        }
        hs = decode_hidden_states(resp)
        assert set(hs.keys()) == {"layer_15", "last"}


class TestERISClient:
    @pytest.fixture
    def mock_client(self):
        """ERISClient with mocked httpx transport."""
        from eris_client import ERISClient
        with patch("eris_client.httpx.Client") as MockHttp:
            mock_http = MagicMock()
            MockHttp.return_value = mock_http

            def make_resp(data):
                r = MagicMock()
                r.json.return_value = data
                r.raise_for_status.return_value = None
                return r

            mock_http._make_resp = make_resp
            client = ERISClient("http://localhost:8001")
            client._client = mock_http
            yield client, mock_http

    def test_create_session(self, mock_client):
        client, http = mock_client
        http.post.return_value = http._make_resp({"session_id": "abc"})
        assert client.create_session() == "abc"

    def test_encode_default_compact(self, mock_client):
        client, http = mock_client
        http.post.return_value = http._make_resp(
            {"handle": "enc_x", "tokens": [], "seq_len": 0, "hidden_dim": 16, "hidden_states": {}}
        )
        client.encode("hello")
        body = http.post.call_args.kwargs["json"]
        assert body["compact"] is True

    def test_inject_numpy_to_list(self, mock_client):
        client, http = mock_client
        http.post.return_value = http._make_resp(
            {"status": "injected", "operation": "add",
             "old_norm": 3.0, "new_norm": 3.1, "cosine_shift": 0.05}
        )
        v = np.random.randn(HIDDEN_DIM).astype(np.float32)
        client.inject("t_x", "sid", v)
        sent = http.post.call_args.kwargs["json"]["vector"]
        assert isinstance(sent, list)
        assert len(sent) == HIDDEN_DIM
        assert all(isinstance(x, float) for x in sent)

    def test_bridge_all_params_forwarded(self, mock_client):
        client, http = mock_client
        http.post.return_value = http._make_resp(
            {"enriched_text": "hi", "analysis": {}, "trajectory_summary": None,
             "pre_analysis": None, "tokens": [], "elapsed_s": 0.5, "handles": {}}
        )
        client.bridge("text", mode="ruminate", n_steps=30, decode_after=False,
                       implicit_match_mode="all")
        body = http.post.call_args.kwargs["json"]
        assert body["mode"] == "ruminate"
        assert body["n_steps"] == 30
        assert body["decode_after"] is False
        assert body["implicit_match_mode"] == "all"

    def test_session_context_manager_deletes_on_exit(self, mock_client):
        client, http = mock_client
        http.post.return_value = http._make_resp({"session_id": "tmp"})
        http.delete.return_value = http._make_resp({"deleted": "tmp"})
        with client.session() as sid:
            assert sid == "tmp"
        http.delete.assert_called_once()

    def test_session_context_manager_deletes_on_exception(self, mock_client):
        client, http = mock_client
        http.post.return_value = http._make_resp({"session_id": "tmp"})
        http.delete.return_value = http._make_resp({"deleted": "tmp"})
        with pytest.raises(ValueError):
            with client.session():
                raise ValueError("oops")
        http.delete.assert_called_once()
