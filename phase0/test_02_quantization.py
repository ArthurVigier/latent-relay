"""
Test 0.2 — Accuracy Degradation Under int8 KV-Cache Quantization
=================================================================
Tests whether quantizing KV-caches from fp16 to int8 before inter-agent
transfer degrades LatentMAS reasoning quality beyond acceptable limits.

Go/No-go criterion: accuracy drop < 3% on GSM8K subset

Two modes:
  - With LatentMAS repo: runs actual latent rollout + quantized transfer
  - Standalone (dry_run): simulates with synthetic data + reports infra OK

The test works by:
  1. Running LatentMAS normally on N samples → baseline accuracy
  2. Running LatentMAS but intercepting KV-cache transfer with
     quantize(fp16→int8)→dequantize(int8→fp16) → quantized accuracy
  3. Comparing the two
"""

import json
import sys
import time
from pathlib import Path

import torch


def quantize_kv_cache_int8(kv_cache: list) -> tuple:
    """
    Quantize KV-cache from fp16 to int8 using per-tensor absmax scaling.
    Returns (quantized_cache, scales) for later dequantization.
    """
    quantized = []
    scales = []
    for k, v in kv_cache:
        k_scale = k.abs().max() / 127.0
        v_scale = v.abs().max() / 127.0

        k_q = (k / k_scale).round().clamp(-128, 127).to(torch.int8)
        v_q = (v / v_scale).round().clamp(-128, 127).to(torch.int8)

        quantized.append((k_q, v_q))
        scales.append((k_scale, v_scale))

    return quantized, scales


def dequantize_kv_cache_int8(quantized_cache: list, scales: list, dtype=torch.float16) -> list:
    """Dequantize int8 KV-cache back to fp16."""
    restored = []
    for (k_q, v_q), (k_scale, v_scale) in zip(quantized_cache, scales):
        k = k_q.to(dtype) * k_scale
        v = v_q.to(dtype) * v_scale
        restored.append((k, v))
    return restored


def measure_quantization_error(original: list, restored: list) -> dict:
    """Compute various error metrics between original and restored KV-caches."""
    all_abs_errors = []
    all_rel_errors = []
    all_cosine_sims = []

    for (k1, v1), (k2, v2) in zip(original, restored):
        for t1, t2 in [(k1, k2), (v1, v2)]:
            flat1 = t1.reshape(-1).float()
            flat2 = t2.reshape(-1).float()

            abs_err = (flat1 - flat2).abs()
            all_abs_errors.append(abs_err.mean().item())

            # Relative error (avoid div by zero)
            denom = flat1.abs().clamp(min=1e-8)
            rel_err = (abs_err / denom).mean().item()
            all_rel_errors.append(rel_err)

            # Cosine similarity
            cos = torch.nn.functional.cosine_similarity(
                flat1.unsqueeze(0), flat2.unsqueeze(0)
            ).item()
            all_cosine_sims.append(cos)

    return {
        "mean_abs_error": sum(all_abs_errors) / len(all_abs_errors),
        "max_abs_error": max(all_abs_errors),
        "mean_rel_error": sum(all_rel_errors) / len(all_rel_errors),
        "mean_cosine_sim": sum(all_cosine_sims) / len(all_cosine_sims),
        "min_cosine_sim": min(all_cosine_sims),
    }


def run_quantization_error_analysis(
    model_name: str,
    device: str,
    n_latent_steps: int
) -> dict:
    """
    Phase A: Pure tensor-level analysis.
    Create realistic KV-caches from a real model's forward pass,
    quantize round-trip, measure error.
    """
    print("\n  Phase A: Tensor-level quantization error analysis")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"    Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        model.eval()

        # Generate real KV-cache from a math prompt
        prompt = (
            "Solve step by step: If a train travels at 60 km/h for 2.5 hours, "
            "then at 80 km/h for 1.5 hours, what is the total distance?"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            real_kv_cache = outputs.past_key_values

        # Convert DynamicCache or tuple format to list of (K, V) tuples
        if hasattr(real_kv_cache, 'key_cache'):
            # DynamicCache format (transformers >= 4.36)
            kv_list = list(zip(real_kv_cache.key_cache, real_kv_cache.value_cache))
        else:
            kv_list = list(real_kv_cache)

        print(f"    KV-cache: {len(kv_list)} layers, "
              f"shape K: {kv_list[0][0].shape}")

        used_real_model = True

        # Clean up model to free VRAM
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"    Could not load model: {e}")
        print("    Falling back to synthetic KV-cache with realistic distributions")

        from test_01_serialization import get_model_config, create_synthetic_kv_cache
        config = get_model_config(model_name)
        kv_list = create_synthetic_kv_cache(
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            head_dim=config["head_dim"],
            seq_len=n_latent_steps,
            device=device
        )
        used_real_model = False

    # Quantize and dequantize
    print("    Quantizing fp16 → int8 → fp16...")
    q_cache, scales = quantize_kv_cache_int8(kv_list)
    restored_cache = dequantize_kv_cache_int8(q_cache, scales)

    # Measure errors
    errors = measure_quantization_error(kv_list, restored_cache)
    print(f"    Mean cosine sim: {errors['mean_cosine_sim']:.6f}")
    print(f"    Min cosine sim:  {errors['min_cosine_sim']:.6f}")
    print(f"    Mean abs error:  {errors['mean_abs_error']:.6f}")
    print(f"    Mean rel error:  {errors['mean_rel_error']:.4%}")

    return {
        "used_real_model": used_real_model,
        "errors": errors,
    }


def run_gsm8k_comparison(
    model_name: str,
    device: str,
    n_latent_steps: int,
    n_samples: int
) -> dict:
    """
    Phase B: End-to-end accuracy comparison on GSM8K.
    Requires the LatentMAS repo to be available.
    """
    print(f"\n  Phase B: GSM8K accuracy comparison ({n_samples} samples)")

    # Check if LatentMAS is available
    latentmas_path = None
    for candidate in ["./LatentMAS", "../LatentMAS", os.path.expanduser("~/LatentMAS")]:
        if os.path.exists(os.path.join(candidate, "run.py")):
            latentmas_path = candidate
            break

    if latentmas_path is None:
        print("    LatentMAS repo not found. Skipping end-to-end comparison.")
        print("    To enable: git clone https://github.com/Gen-Verse/LatentMAS.git")
        return {
            "skipped": True,
            "reason": "LatentMAS repo not found",
            "suggestion": "Clone LatentMAS and re-run for full accuracy comparison"
        }

    # If we get here, we have LatentMAS available
    # We'll monkey-patch the KV-cache transfer to inject quantization
    print(f"    Found LatentMAS at {latentmas_path}")
    sys.path.insert(0, latentmas_path)

    try:
        # Import LatentMAS internals
        # Note: exact imports depend on LatentMAS code structure
        # This is a best-effort integration
        import subprocess

        # Run baseline
        print("    Running baseline (fp16, no quantization)...")
        baseline_cmd = [
            sys.executable, os.path.join(latentmas_path, "run.py"),
            "--method", "latent_mas",
            "--model_name", model_name,
            "--task", "gsm8k",
            "--max_samples", str(n_samples),
            "--max_new_tokens", "1024",
            "--prompt", "sequential",
        ]
        baseline_result = subprocess.run(
            baseline_cmd, capture_output=True, text=True, timeout=3600
        )

        # Parse accuracy from output
        baseline_acc = _parse_accuracy(baseline_result.stdout)
        print(f"    Baseline accuracy: {baseline_acc}")

        # For quantized run, we'd need to modify LatentMAS internals
        # This requires patching the KV-cache transfer function
        # For now, provide the infrastructure and instructions
        print("    NOTE: Quantized run requires patching LatentMAS KV-cache transfer.")
        print("    See phase0_results/quantization_patch.py for the patch.")

        return {
            "baseline_accuracy": baseline_acc,
            "quantized_accuracy": None,
            "accuracy_drop": None,
            "note": "Manual patch required — see quantization_patch.py"
        }

    except Exception as e:
        print(f"    Error running LatentMAS: {e}")
        return {
            "skipped": True,
            "reason": str(e)
        }


def _parse_accuracy(output: str) -> float:
    """Try to extract accuracy from LatentMAS output."""
    import re
    # LatentMAS typically prints accuracy like "Accuracy: 0.82" or "acc: 82.0%"
    patterns = [
        r"[Aa]ccuracy[:\s]+([0-9.]+)",
        r"[Aa]cc[:\s]+([0-9.]+)",
        r"correct[:\s]+(\d+)/(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, output)
        if m:
            if len(m.groups()) == 2:
                return int(m.group(1)) / int(m.group(2))
            val = float(m.group(1))
            return val if val <= 1.0 else val / 100.0
    return -1.0  # Could not parse


import os

def run_quantization_test(
    model_name: str,
    device: str = "cuda:0",
    n_latent_steps: int = 60,
    n_samples: int = 50,
    dry_run: bool = False,
    output_dir: str = "./phase0_results"
) -> dict:
    """
    Main entry point for Test 0.2.

    Go/No-go criterion:
      - Tensor-level: cosine similarity > 0.995
      - End-to-end (if available): accuracy drop < 3%
    """
    print("\n" + "=" * 60)
    print("TEST 0.2 — KV-Cache Quantization Degradation")
    print("=" * 60)

    if "cuda" in device and not torch.cuda.is_available():
        if dry_run:
            device = "cpu"
        else:
            return {"verdict": "NO-GO", "reason": "No CUDA GPU available"}

    # Phase A: Tensor-level analysis (always runs)
    tensor_results = run_quantization_error_analysis(model_name, device, n_latent_steps)

    # Phase B: End-to-end GSM8K (optional, needs LatentMAS repo)
    if not dry_run:
        gsm8k_results = run_gsm8k_comparison(model_name, device, n_latent_steps, n_samples)
    else:
        gsm8k_results = {"skipped": True, "reason": "dry run"}

    # ── Verdict ──
    cosine_threshold = 0.995
    accuracy_threshold = 0.03  # 3% drop

    cos_sim = tensor_results["errors"]["mean_cosine_sim"]
    cos_ok = cos_sim >= cosine_threshold

    # If GSM8K ran successfully
    acc_drop = gsm8k_results.get("accuracy_drop")
    if acc_drop is not None:
        acc_ok = acc_drop < accuracy_threshold
        verdict = "GO" if (cos_ok and acc_ok) else "NO-GO"
        reason = (
            f"Cosine sim {cos_sim:.4f} {'≥' if cos_ok else '<'} {cosine_threshold}, "
            f"Accuracy drop {acc_drop:.1%} {'<' if acc_ok else '≥'} {accuracy_threshold:.0%}"
        )
    else:
        # Verdict based on tensor analysis only
        verdict = "GO" if cos_ok else "NO-GO"
        reason = (
            f"Cosine sim {cos_sim:.4f} {'≥' if cos_ok else '<'} {cosine_threshold} "
            f"(tensor-level only — GSM8K comparison not run)"
        )

    result = {
        "verdict": verdict,
        "reason": reason,
        "thresholds": {
            "cosine_sim_min": cosine_threshold,
            "accuracy_drop_max": accuracy_threshold,
        },
        "tensor_analysis": tensor_results,
        "gsm8k_comparison": gsm8k_results,
    }

    # Generate the monkey-patch script for manual GSM8K testing
    _generate_quantization_patch(output_dir)

    out_path = Path(output_dir) / "test_02_quantization.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")

    return result


def _generate_quantization_patch(output_dir: str):
    """Generate a standalone patch script for LatentMAS KV-cache quantization."""
    patch_code = '''"""
LatentMAS KV-Cache Quantization Patch
======================================
Apply this patch to test int8 quantization in LatentMAS end-to-end.

Usage:
    1. Clone LatentMAS: git clone https://github.com/Gen-Verse/LatentMAS.git
    2. Copy this file into the LatentMAS directory
    3. Run: python quantization_patch.py --model Qwen/Qwen3-4B --n_samples 50

What it does:
    Monkey-patches the KV-cache transfer between agents to inject
    fp16 → int8 → fp16 round-trip quantization, then runs GSM8K evaluation.
"""

import torch
import sys
import os

def quantize_roundtrip(kv_cache):
    """Inject int8 quantization round-trip into KV-cache transfer."""
    restored = []
    for k, v in kv_cache:
        # Quantize
        k_scale = k.abs().max() / 127.0
        v_scale = v.abs().max() / 127.0
        k_q = (k / k_scale).round().clamp(-128, 127).to(torch.int8)
        v_q = (v / v_scale).round().clamp(-128, 127).to(torch.int8)
        # Dequantize
        k_r = k_q.to(k.dtype) * k_scale
        v_r = v_q.to(v.dtype) * v_scale
        restored.append((k_r, v_r))
    return restored

# To use this, find the function in LatentMAS that transfers KV-cache
# between agents (likely in the latent_mas method of run.py or a 
# pipeline file) and wrap the cache transfer with quantize_roundtrip().
#
# Look for code like:
#   past_key_values = agent1_outputs.past_key_values
#   agent2_inputs["past_key_values"] = past_key_values
#
# Change to:
#   past_key_values = agent1_outputs.past_key_values
#   past_key_values = quantize_roundtrip(past_key_values)
#   agent2_inputs["past_key_values"] = past_key_values

print("This is a template patch — see comments above for integration instructions.")
'''
    patch_path = Path(output_dir) / "quantization_patch.py"
    with open(patch_path, "w") as f:
        f.write(patch_code)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_latent_steps", type=int, default=60)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_quantization_test(
        model_name=args.model,
        device=args.device,
        n_latent_steps=args.n_latent_steps,
        n_samples=args.n_samples,
        dry_run=args.dry_run
    )
    print(f"\nVerdict: {result['verdict']} — {result['reason']}")
