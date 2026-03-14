"""
Test 0.1 — KV-cache Serialization Latency
==========================================
Measures round-trip time for: GPU tensor → CPU → compressed bytes → CPU → GPU tensor
Tests multiple compression strategies and KV-cache sizes.

Go/No-go criterion: full round-trip < 2 seconds for a 14B-scale KV-cache (60 steps)

What we're testing:
  - Can we move KV-caches off-GPU fast enough to serve them via network?
  - Which compression method gives the best latency/size tradeoff?
  - How does quantization (fp16 → int8) affect serialization speed?
"""

import time
import io
import sys
import json
from pathlib import Path

import torch

# Optional: lz4 for fast compression
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("WARNING: lz4 not installed. Install with: pip install lz4")


# ── Model configs for realistic KV-cache sizes ──
MODEL_CONFIGS = {
    "Qwen/Qwen3-0.6B": {"n_layers": 28, "n_heads": 16, "head_dim": 64, "d_h": 1024},
    "Qwen/Qwen3-1.7B": {"n_layers": 28, "n_heads": 16, "head_dim": 128, "d_h": 2048},
    "Qwen/Qwen3-4B":   {"n_layers": 36, "n_heads": 32, "head_dim": 128, "d_h": 2560},
    "Qwen/Qwen3-8B":   {"n_layers": 36, "n_heads": 32, "head_dim": 128, "d_h": 4096},
    "Qwen/Qwen3-14B":  {"n_layers": 40, "n_heads": 40, "head_dim": 128, "d_h": 5120},
    # Fallback for unknown models — conservative estimate
    "default":          {"n_layers": 32, "n_heads": 32, "head_dim": 128, "d_h": 4096},
}


def get_model_config(model_name: str) -> dict:
    """Get KV-cache dimensions for a model. Falls back to default if unknown."""
    for key, config in MODEL_CONFIGS.items():
        if key != "default" and key.lower() in model_name.lower():
            return config
    return MODEL_CONFIGS["default"]


def create_synthetic_kv_cache(
    n_layers: int, n_heads: int, head_dim: int, seq_len: int,
    device: str = "cuda:0", dtype=torch.float16
) -> list:
    """Create a synthetic KV-cache mimicking real transformer output."""
    cache = []
    for _ in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        cache.append((k, v))
    return cache


def measure_cache_size(kv_cache: list) -> int:
    """Total bytes of KV-cache on GPU."""
    total = 0
    for k, v in kv_cache:
        total += k.nelement() * k.element_size()
        total += v.nelement() * v.element_size()
    return total


def benchmark_serialize_naive(kv_cache: list, n_runs: int = 5) -> dict:
    """Serialize via torch.save to BytesIO (no compression)."""
    times = []
    sizes = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # GPU → CPU
        cpu_cache = [(k.cpu(), v.cpu()) for k, v in kv_cache]
        # CPU → bytes
        buf = io.BytesIO()
        torch.save(cpu_cache, buf)
        blob = buf.getvalue()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        sizes.append(len(blob))

    return {
        "method": "torch_save_naive",
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
        "blob_size_mb": sizes[0] / (1024 * 1024),
    }


def benchmark_serialize_raw_fp16(kv_cache: list, n_runs: int = 5) -> dict:
    """Serialize by concatenating raw fp16 bytes (fastest possible)."""
    times = []
    sizes = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Stack all layers into one contiguous tensor, then .cpu().numpy().tobytes()
        all_tensors = []
        for k, v in kv_cache:
            all_tensors.append(k)
            all_tensors.append(v)
        stacked = torch.cat([t.reshape(-1) for t in all_tensors])
        blob = stacked.cpu().numpy().tobytes()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        sizes.append(len(blob))

    return {
        "method": "raw_fp16_bytes",
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
        "blob_size_mb": sizes[0] / (1024 * 1024),
    }


def benchmark_serialize_lz4(kv_cache: list, n_runs: int = 5) -> dict:
    """Serialize fp16 bytes + LZ4 compression."""
    if not HAS_LZ4:
        return {"method": "fp16_lz4", "skipped": True, "reason": "lz4 not installed"}

    times = []
    sizes = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        all_tensors = []
        for k, v in kv_cache:
            all_tensors.append(k)
            all_tensors.append(v)
        stacked = torch.cat([t.reshape(-1) for t in all_tensors])
        raw = stacked.cpu().numpy().tobytes()
        blob = lz4.frame.compress(raw)

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        sizes.append(len(blob))

    return {
        "method": "fp16_lz4",
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
        "blob_size_mb": sizes[0] / (1024 * 1024),
    }


def benchmark_serialize_int8_lz4(kv_cache: list, n_runs: int = 5) -> dict:
    """Quantize to int8, then LZ4 compress."""
    if not HAS_LZ4:
        return {"method": "int8_lz4", "skipped": True, "reason": "lz4 not installed"}

    times = []
    sizes = []
    scales_sizes = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Per-tensor absmax quantization to int8
        all_quant = []
        all_scales = []
        for k, v in kv_cache:
            for t in [k, v]:
                scale = t.abs().max() / 127.0
                q = (t / scale).round().clamp(-128, 127).to(torch.int8)
                all_quant.append(q)
                all_scales.append(scale.cpu())

        stacked = torch.cat([t.reshape(-1) for t in all_quant])
        raw = stacked.cpu().numpy().tobytes()
        blob = lz4.frame.compress(raw)

        # Scales are small — just pickle them
        scale_blob = torch.stack(all_scales).numpy().tobytes()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        sizes.append(len(blob))
        scales_sizes.append(len(scale_blob))

    return {
        "method": "int8_lz4",
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
        "blob_size_mb": sizes[0] / (1024 * 1024),
        "scales_size_kb": scales_sizes[0] / 1024,
    }


def benchmark_deserialize_roundtrip(kv_cache: list, device: str, n_runs: int = 5) -> dict:
    """Full round-trip: GPU → serialize → deserialize → GPU. Using raw fp16 + LZ4."""
    if not HAS_LZ4:
        return {"method": "roundtrip_fp16_lz4", "skipped": True, "reason": "lz4 not installed"}

    # Pre-record shapes for reconstruction
    shapes = [(k.shape, v.shape) for k, v in kv_cache]
    dtype = kv_cache[0][0].dtype

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # ── Serialize ──
        all_tensors = []
        for k, v in kv_cache:
            all_tensors.append(k)
            all_tensors.append(v)
        stacked = torch.cat([t.reshape(-1) for t in all_tensors])
        raw = stacked.cpu().numpy().tobytes()
        blob = lz4.frame.compress(raw)

        # ── Deserialize ──
        raw2 = lz4.frame.decompress(blob)
        flat = torch.frombuffer(bytearray(raw2), dtype=dtype)
        # Reconstruct KV-cache
        offset = 0
        reconstructed = []
        for k_shape, v_shape in shapes:
            k_size = 1
            for s in k_shape:
                k_size *= s
            v_size = 1
            for s in v_shape:
                v_size *= s
            k = flat[offset:offset + k_size].reshape(k_shape).to(device)
            offset += k_size
            v = flat[offset:offset + v_size].reshape(v_shape).to(device)
            offset += v_size
            reconstructed.append((k, v))

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    # Verify reconstruction fidelity
    max_diff = 0.0
    for (k1, v1), (k2, v2) in zip(kv_cache, reconstructed):
        max_diff = max(max_diff, (k1 - k2).abs().max().item())
        max_diff = max(max_diff, (v1 - v2).abs().max().item())

    return {
        "method": "roundtrip_fp16_lz4",
        "mean_time_s": sum(times) / len(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
        "max_reconstruction_error": max_diff,
        "is_lossless": max_diff == 0.0,
    }


def run_serialization_test(
    model_name: str,
    device: str = "cuda:0",
    n_latent_steps: int = 60,
    dry_run: bool = False,
    output_dir: str = "./phase0_results"
) -> dict:
    """
    Main entry point for Test 0.1.

    Go/No-go criterion: full round-trip < 2 seconds for target model.
    """
    print("\n" + "=" * 60)
    print("TEST 0.1 — KV-Cache Serialization Latency")
    print("=" * 60)

    config = get_model_config(model_name)
    print(f"Model config: {json.dumps(config, indent=2)}")
    print(f"Latent steps: {n_latent_steps}")

    # Check device
    if "cuda" in device and not torch.cuda.is_available():
        if dry_run:
            print("WARNING: No GPU available, using CPU for dry run")
            device = "cpu"
        else:
            return {
                "verdict": "NO-GO",
                "reason": "No CUDA GPU available",
            }

    # Calculate expected sizes
    cache_bytes = (
        config["n_layers"] * 2  # K + V
        * 1  # batch_size
        * config["n_heads"]
        * n_latent_steps
        * config["head_dim"]
        * 2  # fp16 = 2 bytes
    )
    print(f"Expected KV-cache size: {cache_bytes / (1024*1024):.1f} MB")

    # Create synthetic KV-cache
    print("\nCreating synthetic KV-cache...")
    kv_cache = create_synthetic_kv_cache(
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        head_dim=config["head_dim"],
        seq_len=n_latent_steps,
        device=device
    )
    actual_size = measure_cache_size(kv_cache)
    print(f"Actual KV-cache size: {actual_size / (1024*1024):.1f} MB")

    # Run benchmarks
    print("\nRunning serialization benchmarks (5 runs each)...")
    results = {}

    print("  [1/5] torch.save naive...")
    results["torch_save"] = benchmark_serialize_naive(kv_cache)
    print(f"         → {results['torch_save']['mean_time_s']:.3f}s, "
          f"{results['torch_save']['blob_size_mb']:.1f} MB")

    print("  [2/5] Raw fp16 bytes...")
    results["raw_fp16"] = benchmark_serialize_raw_fp16(kv_cache)
    print(f"         → {results['raw_fp16']['mean_time_s']:.3f}s, "
          f"{results['raw_fp16']['blob_size_mb']:.1f} MB")

    print("  [3/5] fp16 + LZ4...")
    results["fp16_lz4"] = benchmark_serialize_lz4(kv_cache)
    if not results["fp16_lz4"].get("skipped"):
        print(f"         → {results['fp16_lz4']['mean_time_s']:.3f}s, "
              f"{results['fp16_lz4']['blob_size_mb']:.1f} MB")
    else:
        print(f"         → SKIPPED ({results['fp16_lz4']['reason']})")

    print("  [4/5] int8 + LZ4...")
    results["int8_lz4"] = benchmark_serialize_int8_lz4(kv_cache)
    if not results["int8_lz4"].get("skipped"):
        print(f"         → {results['int8_lz4']['mean_time_s']:.3f}s, "
              f"{results['int8_lz4']['blob_size_mb']:.1f} MB")
    else:
        print(f"         → SKIPPED ({results['int8_lz4']['reason']})")

    print("  [5/5] Full round-trip (fp16 + LZ4)...")
    results["roundtrip"] = benchmark_deserialize_roundtrip(kv_cache, device)
    if not results["roundtrip"].get("skipped"):
        print(f"         → {results['roundtrip']['mean_time_s']:.3f}s, "
              f"lossless: {results['roundtrip']['is_lossless']}")
    else:
        print(f"         → SKIPPED ({results['roundtrip']['reason']})")

    # ── Verdict ──
    THRESHOLD_S = 2.0
    roundtrip_time = results.get("roundtrip", {}).get("mean_time_s", float("inf"))

    if results.get("roundtrip", {}).get("skipped"):
        # Fall back to raw_fp16 * 2 as estimate
        roundtrip_time = results["raw_fp16"]["mean_time_s"] * 2.5
        estimated = True
    else:
        estimated = False

    verdict = "GO" if roundtrip_time < THRESHOLD_S else "NO-GO"

    # Also compute 14B projections if testing on a smaller model
    projection_14b = None
    config_14b = MODEL_CONFIGS["Qwen/Qwen3-14B"]
    if "14B" not in model_name:
        # Linear scaling by total cache size
        cache_14b = (
            config_14b["n_layers"] * 2
            * config_14b["n_heads"]
            * n_latent_steps
            * config_14b["head_dim"]
            * 2
        )
        scale_factor = cache_14b / cache_bytes
        projected_time = roundtrip_time * scale_factor
        projection_14b = {
            "projected_roundtrip_s": projected_time,
            "scale_factor": scale_factor,
            "projected_cache_mb": cache_14b / (1024 * 1024),
            "verdict_14b": "GO" if projected_time < THRESHOLD_S else "NO-GO"
        }
        print(f"\n  14B projection: {projected_time:.3f}s "
              f"({'GO' if projected_time < THRESHOLD_S else 'NO-GO'})")

    result = {
        "verdict": verdict,
        "reason": (
            f"Round-trip {'(estimated) ' if estimated else ''}"
            f"{roundtrip_time:.3f}s {'<' if verdict == 'GO' else '>='} "
            f"{THRESHOLD_S}s threshold"
        ),
        "threshold_s": THRESHOLD_S,
        "cache_size_mb": actual_size / (1024 * 1024),
        "model_config": config,
        "benchmarks": results,
        "projection_14b": projection_14b,
    }

    # Save detailed results
    out_path = Path(output_dir) / "test_01_serialization.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_latent_steps", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_serialization_test(
        model_name=args.model,
        device=args.device,
        n_latent_steps=args.n_latent_steps,
        dry_run=args.dry_run
    )
    print(f"\nVerdict: {result['verdict']} — {result['reason']}")
