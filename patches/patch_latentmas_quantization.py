"""
LatentMAS KV-Cache Quantization Patch (Auto-Apply)
====================================================
Monkey-patches methods/latent_mas.py to inject int8 quantization round-trips
at ALL KV-cache transfer points between agents.

The code has two pipelines with three distinct transfer mechanisms:

  Pipeline 1: run_batch() [HF-only mode]
    - Transfer point A (line ~130): past_kv returned by generate_latent_batch()
      → passed back as past_key_values to next agent's generate_latent_batch()
      Format: tuple of (K, V) tensors per layer, or HF Cache object

  Pipeline 2: run_batch_vllm() [Hybrid HF+vLLM mode]
    - Transfer point B (line ~300): past_kv returned by generate_latent_batch_hidden_state()
      → passed back as past_key_values to next agent
      Format: same as A
    - Transfer point C (line ~340): past_embedding (hidden states)
      → concatenated via embedding_record, sent to vLLM as prompt_embeds
      Format: tensor [B, L, H] in float16/bfloat16

Usage:
    # Method 1: Auto-patch (modifies methods/latent_mas.py in-place, creates .bak)
    cd LatentMAS
    python /path/to/patch_latentmas_quantization.py --apply

    # Method 2: Run patched evaluation directly
    python /path/to/patch_latentmas_quantization.py --run \
        --model Qwen/Qwen3-4B --task gsm8k --n_samples 50

    # Method 3: Revert patch
    python /path/to/patch_latentmas_quantization.py --revert

    # Method 4: Monkey-patch at runtime (no file modification)
    python /path/to/patch_latentmas_quantization.py --monkey-patch \
        --model Qwen/Qwen3-4B --task gsm8k --n_samples 50
"""

import argparse
import shutil
import os
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# Core quantization functions (used by all methods)
# ══════════════════════════════════════════════════════════════

QUANTIZE_FUNCTIONS = '''
# ── LatentMAS Quantization Patch (auto-injected) ──
import torch as _qt

try:
    from transformers.cache_utils import Cache as _QCache
except ImportError:
    _QCache = None

_QUANT_PATCH_ACTIVE = True  # Toggle to disable without removing patch
_QUANT_STATS = {"n_roundtrips": 0, "total_abs_error": 0.0, "total_cosine_sim": 0.0}

def _quantize_kv_int8(kv_cache):
    """Quantize KV-cache from fp16/bf16 to int8 (per-tensor absmax)."""
    if kv_cache is None or not _QUANT_PATCH_ACTIVE:
        return kv_cache, None

    # Handle HF Cache objects
    is_cache_obj = _QCache is not None and isinstance(kv_cache, _QCache)
    if is_cache_obj:
        legacy = kv_cache.to_legacy_cache()
    else:
        legacy = kv_cache

    quantized = []
    scales = []
    for layer in legacy:
        layer_q = []
        layer_s = []
        for t in layer:  # K and V tensors
            s = t.abs().max() / 127.0
            q = (t / s).round().clamp(-128, 127).to(_qt.int8)
            layer_q.append(q)
            layer_s.append((s, t.dtype))
        quantized.append(tuple(layer_q))
        scales.append(layer_s)

    return tuple(quantized), (scales, is_cache_obj, type(kv_cache) if is_cache_obj else None)


def _dequantize_kv_int8(quantized_cache, meta):
    """Dequantize int8 KV-cache back to original dtype."""
    if quantized_cache is None or meta is None:
        return quantized_cache

    scales, is_cache_obj, cache_class = meta

    restored = []
    for layer_q, layer_s in zip(quantized_cache, scales):
        layer_r = []
        for q, (s, dtype) in zip(layer_q, layer_s):
            layer_r.append(q.to(dtype) * s)
        restored.append(tuple(layer_r))
    restored = tuple(restored)

    if is_cache_obj and cache_class is not None:
        return cache_class.from_legacy_cache(restored)

    return restored


def _quantize_embedding_int8(embedding):
    """Quantize hidden state embeddings [B, L, H] via int8 round-trip."""
    if embedding is None or not _QUANT_PATCH_ACTIVE:
        return embedding
    dtype = embedding.dtype
    s = embedding.abs().max() / 127.0
    q = (embedding / s).round().clamp(-128, 127).to(_qt.int8)
    restored = q.to(dtype) * s

    # Track stats
    _QUANT_STATS["n_roundtrips"] += 1
    cos = _qt.nn.functional.cosine_similarity(
        embedding.reshape(1, -1).float(),
        restored.reshape(1, -1).float()
    ).item()
    _QUANT_STATS["total_cosine_sim"] += cos
    abs_err = (embedding - restored).abs().mean().item()
    _QUANT_STATS["total_abs_error"] += abs_err

    return restored


def _quant_report():
    """Print quantization statistics."""
    n = _QUANT_STATS["n_roundtrips"]
    if n > 0:
        print(f"\\n[QUANT PATCH] {n} round-trips | "
              f"mean cosine sim: {_QUANT_STATS['total_cosine_sim']/n:.6f} | "
              f"mean abs error: {_QUANT_STATS['total_abs_error']/n:.6f}")
    else:
        print("\\n[QUANT PATCH] No quantization round-trips recorded")

# ── End quantization patch ──
'''

# ══════════════════════════════════════════════════════════════
# Method 1: File-based patch (modifies latent_mas.py)
# ══════════════════════════════════════════════════════════════

def apply_file_patch(latentmas_dir: str = "."):
    """Apply quantization patch to methods/latent_mas.py by text replacement."""
    target = Path(latentmas_dir) / "methods" / "latent_mas.py"
    if not target.exists():
        print(f"ERROR: {target} not found. Run from LatentMAS root directory.")
        sys.exit(1)

    # Backup
    backup = target.with_suffix(".py.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Backup created: {backup}")
    else:
        print(f"Backup already exists: {backup}")

    code = target.read_text()

    if "_QUANT_PATCH_ACTIVE" in code:
        print("Patch already applied. Use --revert to undo.")
        return

    # ── Insert quantization functions after imports ──
    # Find the line "import torch" and insert after it
    anchor = "import torch\n"
    if anchor not in code:
        print("ERROR: Could not find 'import torch' anchor in latent_mas.py")
        sys.exit(1)

    code = code.replace(anchor, anchor + QUANTIZE_FUNCTIONS, 1)

    # ── Patch Transfer Point A (run_batch, line ~130) ──
    # Original:
    #   past_kv = self.model.generate_latent_batch(
    #       wrapped_ids,
    #       attention_mask=wrapped_mask,
    #       latent_steps=self.latent_steps,
    #       past_key_values=past_kv,
    #   )
    # We need to quantize the INPUT past_kv and the OUTPUT past_kv
    # Strategy: wrap the call

    old_a = """                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )"""

    new_a = """                # [QUANT PATCH] Quantize past_kv before passing to next agent
                _qkv, _qmeta = _quantize_kv_int8(past_kv)
                past_kv_input = _dequantize_kv_int8(_qkv, _qmeta) if past_kv is not None else None
                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv_input,
                )"""

    if old_a in code:
        code = code.replace(old_a, new_a, 1)
        print("  ✓ Patched Transfer Point A (run_batch / generate_latent_batch)")
    else:
        print("  ⚠ Could not find Transfer Point A — may need manual patch")

    # ── Patch Transfer Point B (run_batch_vllm, line ~300) ──
    old_b = """                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )"""

    new_b = """                # [QUANT PATCH] Quantize past_kv before passing to next agent
                _qkv, _qmeta = _quantize_kv_int8(past_kv)
                past_kv_input = _dequantize_kv_int8(_qkv, _qmeta) if past_kv is not None else None
                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv_input,
                )
                # [QUANT PATCH] Quantize hidden embeddings
                previous_hidden_embedding = _quantize_embedding_int8(previous_hidden_embedding)"""

    if old_b in code:
        code = code.replace(old_b, new_b, 1)
        print("  ✓ Patched Transfer Point B (run_batch_vllm / generate_latent_batch_hidden_state)")
    else:
        print("  ⚠ Could not find Transfer Point B — may need manual patch")

    # ── Patch Transfer Point C (embedding concatenation, line ~340) ──
    # The past_embedding tensor is already quantized at point B via
    # _quantize_embedding_int8(), so the concatenation in embedding_record
    # naturally carries the quantized embeddings. No separate patch needed.
    print("  ✓ Transfer Point C (embedding_record concat) — covered by Point B patch")

    # ── Add report call at end of both run methods ──
    old_return_1 = "        return results\n    \n    def run_batch_vllm"
    new_return_1 = "        _quant_report()\n        return results\n    \n    def run_batch_vllm"
    if old_return_1 in code:
        code = code.replace(old_return_1, new_return_1, 1)

    # Add report before last return
    # Find the last "return results" in the file
    last_return_idx = code.rfind("        return results")
    if last_return_idx > 0:
        code = code[:last_return_idx] + "        _quant_report()\n" + code[last_return_idx:]

    target.write_text(code)
    print(f"\nPatch applied to {target}")
    print("Run your usual LatentMAS commands to test with int8 quantization.")
    print("Use --revert to undo.")


def revert_file_patch(latentmas_dir: str = "."):
    """Revert methods/latent_mas.py from backup."""
    target = Path(latentmas_dir) / "methods" / "latent_mas.py"
    backup = target.with_suffix(".py.bak")
    if backup.exists():
        shutil.copy2(backup, target)
        print(f"Reverted {target} from {backup}")
    else:
        print(f"No backup found at {backup}")


# ══════════════════════════════════════════════════════════════
# Method 2: Monkey-patch at runtime (no file modification)
# ══════════════════════════════════════════════════════════════

def monkey_patch_latentmas():
    """
    Monkey-patch LatentMAS at runtime by wrapping the key methods.
    Must be called AFTER importing LatentMAS but BEFORE running.
    """
    # We need to import these at function scope
    import torch

    try:
        from transformers.cache_utils import Cache
    except ImportError:
        Cache = None

    # Import the quantization functions into the module namespace
    from methods.latent_mas import LatentMASMethod

    # Store originals
    _orig_run_batch = LatentMASMethod.run_batch
    _orig_run_batch_vllm = LatentMASMethod.run_batch_vllm

    quant_stats = {"n_calls": 0, "total_cosine": 0.0}

    def _q_roundtrip_kv(past_kv):
        """int8 round-trip for KV-cache."""
        if past_kv is None:
            return None

        is_cache = Cache is not None and isinstance(past_kv, Cache)
        if is_cache:
            legacy = past_kv.to_legacy_cache()
        else:
            legacy = past_kv

        restored_layers = []
        for layer in legacy:
            layer_out = []
            for t in layer:
                s = t.abs().max() / 127.0
                q = (t / s).round().clamp(-128, 127).to(torch.int8)
                r = q.to(t.dtype) * s
                layer_out.append(r)
            restored_layers.append(tuple(layer_out))
        result = tuple(restored_layers)

        if is_cache:
            return past_kv.__class__.from_legacy_cache(result)
        return result

    def _q_roundtrip_emb(emb):
        """int8 round-trip for hidden embeddings."""
        if emb is None:
            return emb
        s = emb.abs().max() / 127.0
        q = (emb / s).round().clamp(-128, 127).to(torch.int8)
        r = q.to(emb.dtype) * s
        # Track cosine
        cos = torch.nn.functional.cosine_similarity(
            emb.reshape(1, -1).float(), r.reshape(1, -1).float()
        ).item()
        quant_stats["n_calls"] += 1
        quant_stats["total_cosine"] += cos
        return r

    def patched_run_batch(self, items):
        """Wraps run_batch to inject quantization at KV-cache transfer."""
        # Patch model's generate_latent_batch to quantize its output
        orig_glb = self.model.generate_latent_batch

        def wrapped_glb(input_ids, attention_mask, latent_steps, past_key_values=None):
            past_key_values = _q_roundtrip_kv(past_key_values)
            return orig_glb(input_ids, attention_mask=attention_mask,
                          latent_steps=latent_steps, past_key_values=past_key_values)

        self.model.generate_latent_batch = wrapped_glb
        try:
            result = _orig_run_batch(self, items)
        finally:
            self.model.generate_latent_batch = orig_glb

        if quant_stats["n_calls"] > 0:
            avg_cos = quant_stats["total_cosine"] / quant_stats["n_calls"]
            print(f"[QUANT] {quant_stats['n_calls']} roundtrips, avg cosine: {avg_cos:.6f}")

        return result

    def patched_run_batch_vllm(self, items):
        """Wraps run_batch_vllm to inject quantization at both transfer points."""
        orig_glbhs = self.model.generate_latent_batch_hidden_state

        def wrapped_glbhs(input_ids, attention_mask, latent_steps, past_key_values=None):
            past_key_values = _q_roundtrip_kv(past_key_values)
            past_kv, hidden_emb = orig_glbhs(
                input_ids, attention_mask=attention_mask,
                latent_steps=latent_steps, past_key_values=past_key_values
            )
            hidden_emb = _q_roundtrip_emb(hidden_emb)
            return past_kv, hidden_emb

        self.model.generate_latent_batch_hidden_state = wrapped_glbhs
        try:
            result = _orig_run_batch_vllm(self, items)
        finally:
            self.model.generate_latent_batch_hidden_state = orig_glbhs

        if quant_stats["n_calls"] > 0:
            avg_cos = quant_stats["total_cosine"] / quant_stats["n_calls"]
            print(f"[QUANT] {quant_stats['n_calls']} roundtrips, avg cosine: {avg_cos:.6f}")

        return result

    LatentMASMethod.run_batch = patched_run_batch
    LatentMASMethod.run_batch_vllm = patched_run_batch_vllm
    print("[QUANT PATCH] Monkey-patch applied to LatentMASMethod")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatentMAS Quantization Patch")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true",
                      help="Apply file-based patch to methods/latent_mas.py")
    group.add_argument("--revert", action="store_true",
                      help="Revert patch from backup")
    group.add_argument("--monkey-patch", action="store_true",
                      help="Apply at runtime and run evaluation")
    group.add_argument("--show-points", action="store_true",
                      help="Show transfer points without modifying anything")

    parser.add_argument("--dir", default=".", help="LatentMAS root directory")

    args = parser.parse_args()

    if args.apply:
        apply_file_patch(args.dir)
    elif args.revert:
        revert_file_patch(args.dir)
    elif args.show_points:
        print("""
KV-Cache Transfer Points in methods/latent_mas.py:
═══════════════════════════════════════════════════

Pipeline 1: run_batch() — Pure HF mode
  Point A (line ~130):
    past_kv = self.model.generate_latent_batch(
        wrapped_ids, attention_mask=wrapped_mask,
        latent_steps=self.latent_steps,
        past_key_values=past_kv,  ← INPUT: accumulated KV from previous agents
    )                             → OUTPUT: extended KV with new latent thoughts

Pipeline 2: run_batch_vllm() — Hybrid HF+vLLM mode  
  Point B (line ~300):
    past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
        wrapped_ids, attention_mask=wrapped_mask,
        latent_steps=self.latent_steps,
        past_key_values=past_kv,  ← INPUT: accumulated KV
    )                             → OUTPUT: (extended KV, hidden embeddings [B,L,H])

  Point C (line ~340):
    past_embedding = torch.cat(embedding_record, dim=1)  ← concatenated hidden states
    → injected into judger prompt via whole_prompt_emb
    → sent to vLLM as prompt_embeds for final text generation

The quantization patch injects int8 round-trips at Points A, B (KV-cache input)
and at the hidden_embedding output of Point B (which feeds into Point C).
""")
    elif args.monkey_patch:
        print("Monkey-patch mode requires running from LatentMAS directory.")
        print("Add this to your run script before the evaluation loop:")
        print()
        print("  import sys")
        print("  sys.path.insert(0, '/path/to/latentmas_phase0')")
        print("  from patch_latentmas_quantization import monkey_patch_latentmas")
        print("  monkey_patch_latentmas()")
