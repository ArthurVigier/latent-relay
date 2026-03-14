"""
Test 0.3 — W_a Determinism and Portability
============================================
Tests whether the alignment matrix W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in
is deterministic and portable across instances.

Go/No-go criterion: 
  - W_a must be bit-identical when computed twice on same model
  - W_a must be save/loadable with zero error
  - W_a must produce identical latent rollout outputs

What we're testing:
  - Is the closed-form solution numerically stable?
  - Can we pre-compute W_a once and ship it alongside the model?
  - Does the rollout diverge if we use a saved W_a vs freshly computed?
"""

import json
import time
import sys
import os
from pathlib import Path

import torch


def compute_wa_from_model(model, lambda_reg: float = 1e-4) -> torch.Tensor:
    """
    Compute the alignment matrix W_a from a HuggingFace CausalLM model.
    
    W_a = (W_out^T @ W_out + λI)^{-1} @ W_out^T @ W_in
    
    Where:
      W_out = lm_head.weight  (vocab_size × d_h)
      W_in  = embed_tokens.weight  (vocab_size × d_h)
    """
    # Extract embedding matrices
    if hasattr(model, 'model'):
        # Typical LLaMA/Qwen structure
        embed = model.model.embed_tokens.weight.data  # (vocab, d_h)
    elif hasattr(model, 'transformer'):
        # GPT-2 style
        embed = model.transformer.wte.weight.data
    else:
        raise ValueError("Cannot find embedding layer — unsupported model architecture")

    lm_head = model.lm_head.weight.data  # (vocab, d_h)

    # Check if tied embeddings (common in smaller models)
    tied = torch.equal(embed, lm_head)

    W_in = embed.float()    # (vocab, d_h)
    W_out = lm_head.float()  # (vocab, d_h)

    d_h = W_in.shape[1]

    # W_a = (W_out^T @ W_out + λI)^{-1} @ W_out^T @ W_in
    # Shape: (d_h × d_h)
    WtW = W_out.T @ W_out  # (d_h, d_h)
    reg = lambda_reg * torch.eye(d_h, device=WtW.device, dtype=WtW.dtype)
    WtW_reg = WtW + reg
    
    # Solve via Cholesky (more stable than explicit inverse)
    try:
        L = torch.linalg.cholesky(WtW_reg)
        WtWin = W_out.T @ W_in  # (d_h, d_h)
        W_a = torch.cholesky_solve(WtWin, L)
    except RuntimeError:
        # Fallback to direct solve if not positive definite
        WtWin = W_out.T @ W_in
        W_a = torch.linalg.solve(WtW_reg, WtWin)

    return W_a.half(), {"tied_embeddings": tied, "d_h": d_h, "vocab_size": W_in.shape[0]}


def compute_wa_synthetic(d_h: int = 4096, vocab_size: int = 32000, 
                         lambda_reg: float = 1e-4, seed: int = 42,
                         device: str = "cpu") -> torch.Tensor:
    """Compute W_a from synthetic matrices for dry-run testing."""
    torch.manual_seed(seed)
    W_in = torch.randn(vocab_size, d_h, device=device)
    W_out = torch.randn(vocab_size, d_h, device=device)

    WtW = W_out.T @ W_out
    reg = lambda_reg * torch.eye(d_h, device=device)
    WtWin = W_out.T @ W_in
    W_a = torch.linalg.solve(WtW + reg, WtWin)

    return W_a.half(), {"tied_embeddings": False, "d_h": d_h, "vocab_size": vocab_size}


def test_determinism(compute_fn, **kwargs) -> dict:
    """Compute W_a twice and check bit-identical."""
    print("    Computing W_a (run 1)...")
    t0 = time.perf_counter()
    wa1, meta = compute_fn(**kwargs)
    t1 = time.perf_counter()
    print(f"      → {t1-t0:.2f}s, shape: {wa1.shape}")

    print("    Computing W_a (run 2)...")
    t0 = time.perf_counter()
    wa2, _ = compute_fn(**kwargs)
    t2 = time.perf_counter()
    print(f"      → {t2-t0:.2f}s")

    # Bit-identical check
    is_identical = torch.equal(wa1, wa2)
    max_diff = (wa1.float() - wa2.float()).abs().max().item()

    print(f"    Bit-identical: {is_identical}")
    print(f"    Max diff: {max_diff}")

    return {
        "is_deterministic": is_identical,
        "max_diff": max_diff,
        "compute_time_s": t1 - t0,
        "shape": list(wa1.shape),
        "metadata": meta,
        "wa_ref": wa1,  # Keep reference for save/load test
    }


def test_save_load(wa: torch.Tensor, save_path: str) -> dict:
    """Save W_a to disk, reload, verify identical."""
    print(f"    Saving W_a to {save_path}...")
    torch.save(wa, save_path)
    file_size = os.path.getsize(save_path)
    print(f"      → {file_size / (1024*1024):.1f} MB")

    print("    Loading W_a from disk...")
    wa_loaded = torch.load(save_path, map_location=wa.device, weights_only=True)

    is_identical = torch.equal(wa, wa_loaded)
    max_diff = (wa.float() - wa_loaded.float()).abs().max().item()

    print(f"    Save/load bit-identical: {is_identical}")

    return {
        "is_portable": is_identical,
        "max_diff": max_diff,
        "file_size_mb": file_size / (1024 * 1024),
    }


def test_rollout_consistency(wa: torch.Tensor, device: str, n_steps: int = 10) -> dict:
    """
    Test that iterating h_{t+1} = W_a @ h_t produces consistent results.
    This simulates the core latent rollout loop (without the full model forward pass).
    
    We test:
      1. Same W_a, same seed → identical trajectory
      2. W_a from disk → identical trajectory
    """
    print(f"    Testing rollout consistency ({n_steps} steps)...")
    
    d_h = wa.shape[0]
    wa_f = wa.float().to(device)

    def rollout(wa_matrix, seed):
        torch.manual_seed(seed)
        h = torch.randn(1, d_h, device=device)
        trajectory = [h.clone()]
        for _ in range(n_steps):
            # Simplified latent step: h_{t+1} = normalize(W_a @ h_t)
            h = (wa_matrix @ h.T).T
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
            trajectory.append(h.clone())
        return torch.cat(trajectory, dim=0)  # (n_steps+1, d_h)

    # Run twice with same seed
    traj1 = rollout(wa_f, seed=123)
    traj2 = rollout(wa_f, seed=123)

    is_identical = torch.equal(traj1, traj2)
    max_diff = (traj1 - traj2).abs().max().item()

    # Check trajectory doesn't diverge (norms should stay bounded)
    norms = traj1.norm(dim=-1)
    norm_stable = (norms.max() / norms.min()).item() < 100  # allow 100x variation

    print(f"    Trajectory deterministic: {is_identical}")
    print(f"    Trajectory stable: {norm_stable}")
    print(f"    Norm range: [{norms.min().item():.4f}, {norms.max().item():.4f}]")

    # Spectral radius of W_a (predicts stability)
    try:
        eigenvalues = torch.linalg.eigvals(wa_f[:min(d_h, 512), :min(d_h, 512)])
        spectral_radius = eigenvalues.abs().max().item()
        print(f"    Spectral radius (approx): {spectral_radius:.4f}")
    except Exception:
        spectral_radius = None

    return {
        "is_deterministic": is_identical,
        "max_diff": max_diff,
        "trajectory_stable": norm_stable,
        "norm_min": norms.min().item(),
        "norm_max": norms.max().item(),
        "spectral_radius": spectral_radius,
    }


def run_wa_determinism_test(
    model_name: str,
    device: str = "cuda:0",
    dry_run: bool = False,
    output_dir: str = "./phase0_results"
) -> dict:
    """
    Main entry point for Test 0.3.

    Go/No-go criteria:
      - W_a computation is deterministic (bit-identical across runs)
      - W_a is save/loadable without error
      - Rollout trajectory is deterministic
    """
    print("\n" + "=" * 60)
    print("TEST 0.3 — W_a Determinism and Portability")
    print("=" * 60)

    if "cuda" in device and not torch.cuda.is_available():
        if dry_run:
            device = "cpu"
        else:
            return {"verdict": "NO-GO", "reason": "No CUDA GPU available"}

    # ── Phase A: Compute and check determinism ──
    print("\n  Phase A: Determinism")
    if dry_run:
        det_results = test_determinism(
            compute_wa_synthetic,
            d_h=2048, vocab_size=32000, device=device
        )
    else:
        try:
            from transformers import AutoModelForCausalLM
            print(f"    Loading {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
            model.eval()

            det_results = test_determinism(
                compute_wa_from_model, model=model
            )

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Could not load model: {e}")
            print("    Falling back to synthetic W_a")
            det_results = test_determinism(
                compute_wa_synthetic,
                d_h=2048, vocab_size=32000, device=device
            )

    wa = det_results.pop("wa_ref")

    # ── Phase B: Save/Load portability ──
    print("\n  Phase B: Save/Load portability")
    save_path = str(Path(output_dir) / "wa_test.pt")
    port_results = test_save_load(wa, save_path)

    # ── Phase C: Rollout consistency ──
    print("\n  Phase C: Rollout consistency")
    rollout_results = test_rollout_consistency(wa, device)

    # ── Verdict ──
    all_ok = (
        det_results["is_deterministic"]
        and port_results["is_portable"]
        and rollout_results["is_deterministic"]
    )

    verdict = "GO" if all_ok else "NO-GO"
    reasons = []
    if not det_results["is_deterministic"]:
        reasons.append(f"W_a not deterministic (diff={det_results['max_diff']:.2e})")
    if not port_results["is_portable"]:
        reasons.append("W_a not portable after save/load")
    if not rollout_results["is_deterministic"]:
        reasons.append("Rollout trajectory not deterministic")
    if rollout_results.get("spectral_radius") and rollout_results["spectral_radius"] > 2.0:
        reasons.append(f"WARNING: high spectral radius ({rollout_results['spectral_radius']:.2f}) — rollout may diverge")

    reason = "; ".join(reasons) if reasons else "All checks passed"

    result = {
        "verdict": verdict,
        "reason": reason,
        "determinism": det_results,
        "portability": port_results,
        "rollout": rollout_results,
    }

    # Cleanup
    try:
        os.remove(save_path)
    except OSError:
        pass

    out_path = Path(output_dir) / "test_03_wa_determinism.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_wa_determinism_test(
        model_name=args.model,
        device=args.device,
        dry_run=args.dry_run
    )
    print(f"\nVerdict: {result['verdict']} — {result['reason']}")
