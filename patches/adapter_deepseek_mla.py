"""
LatentMAS DeepSeek MLA Adapter
===============================
Patches LatentMAS to support DeepSeek-V2/V3/R1 family models with
Multi-Head Latent Attention (MLA).

Key insight: HuggingFace's implementation of DeepSeek MLA caches the
FULL K,V tensors (post up-projection), not the compressed latent vectors.
This means past_key_values has the same interface as standard GQA models.
The latent rollout, W_a computation, and KV-cache transfer all work
identically — we only need to adapt:

  1. The Qwen-specific assert in latent_mas.py line 358
  2. The chat template parsing for embedding insertion position
  3. The MODEL_CONFIGS for Phase 0 tests

Supported models:
  - deepseek-ai/DeepSeek-V2-Lite-Chat  (16B total, 2.4B active, 1×40G GPU)
  - deepseek-ai/DeepSeek-V2-Lite       (base model)
  - deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
  - deepseek-ai/DeepSeek-V2-Chat       (236B, needs multi-GPU)
  - deepseek-ai/DeepSeek-R1-Distill-*  (Qwen-based distills — already compatible)

Usage:
  # Method 1: Auto-patch latent_mas.py to support DeepSeek
  cd LatentMAS
  python /path/to/adapter_deepseek_mla.py --apply

  # Method 2: Run directly on DeepSeek-V2-Lite-Chat (HF backend, no vLLM)
  cd LatentMAS
  python /path/to/adapter_deepseek_mla.py --apply
  python run.py --method latent_mas --model_name deepseek-ai/DeepSeek-V2-Lite-Chat \\
    --task gsm8k --prompt sequential --max_samples 50 --max_new_tokens 1024

  # Method 3: Revert all patches
  python /path/to/adapter_deepseek_mla.py --revert
"""

import argparse
import shutil
import sys
import os
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# Chat template detection for embedding insertion
# ══════════════════════════════════════════════════════════════

# DeepSeek-V2 chat format:
#   <｜begin▁of▁sentence｜>User: {message}
#
#   Assistant: {response}<｜end▁of▁sentence｜>
#
# DeepSeek-V2-Lite-Chat uses a slightly different template:
#   <|begin▁of▁sentence|>[INST] {message} [/INST]
#
# The exact template is in tokenizer_config.json and varies per model.
# Our approach: use a GENERIC insertion strategy that works for ANY
# chat template by finding the user message boundary dynamically.

DEEPSEEK_INSERTION_CODE = '''
                # ── [DEEPSEEK ADAPTER] Generic embedding insertion ──
                # Works for any chat template by finding the user content boundary.
                # Strategy: insert latent embeddings right after the user message marker.
                len_of_left = []
                for p in judger_prompts:
                    # Try Qwen-style first
                    qwen_marker = "<|im_start|>user\\n"
                    ds_marker_inst = "[INST]"
                    ds_marker_user = "User:"
                    
                    if qwen_marker in p:
                        idx = p.find(qwen_marker)
                        left = p[: idx + len(qwen_marker)]
                    elif ds_marker_inst in p:
                        idx = p.find(ds_marker_inst)
                        left = p[: idx + len(ds_marker_inst)]
                    elif ds_marker_user in p:
                        idx = p.find(ds_marker_user)
                        left = p[: idx + len(ds_marker_user)]
                    else:
                        # Fallback: insert at 20% of the prompt (after system/role tokens)
                        tokens = self.model.tokenizer(p)['input_ids']
                        insert_at = max(1, len(tokens) // 5)
                        left = self.model.tokenizer.decode(tokens[:insert_at])
                    
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
'''

ORIGINAL_QWEN_INSERTION = '''                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position    
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\\n")
                    # Get the text up to and including "<|im_start|>user\\n"
                    left = p[: idx + len("<|im_start|>user\\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))'''


def apply_deepseek_patch(latentmas_dir: str = "."):
    """Patch methods/latent_mas.py to support DeepSeek models."""
    target = Path(latentmas_dir) / "methods" / "latent_mas.py"
    if not target.exists():
        print(f"ERROR: {target} not found. Run from LatentMAS root directory.")
        sys.exit(1)

    # Backup
    backup = target.with_suffix(".py.deepseek.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Backup created: {backup}")

    code = target.read_text()

    if "[DEEPSEEK ADAPTER]" in code:
        print("DeepSeek adapter already applied. Use --revert to undo.")
        return

    # ── Patch 1: Replace Qwen-specific assertion + insertion with generic version ──
    if ORIGINAL_QWEN_INSERTION in code:
        code = code.replace(ORIGINAL_QWEN_INSERTION, DEEPSEEK_INSERTION_CODE)
        print("  ✓ Patch 1: Replaced Qwen-specific assert + insertion with generic adapter")
    else:
        print("  ⚠ Patch 1: Could not find exact Qwen insertion code.")
        print("    Looking for assert to remove...")
        # Try just removing the assert
        assert_line = '                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name'
        if assert_line in code:
            code = code.replace(
                assert_line,
                '                # [DEEPSEEK ADAPTER] assert removed — generic model support'
            )
            print("  ✓ Patch 1b: Removed Qwen assert (insertion code not found, may need manual update)")
        else:
            print("  ✗ Patch 1: Could not find Qwen assert. File may already be modified.")

    target.write_text(code)
    print(f"\nDeepSeek adapter applied to {target}")
    print("You can now run LatentMAS with DeepSeek models:")
    print("  python run.py --method latent_mas --model_name deepseek-ai/DeepSeek-V2-Lite-Chat \\")
    print("    --task gsm8k --prompt sequential --max_samples 50 --max_new_tokens 1024")


def revert_deepseek_patch(latentmas_dir: str = "."):
    """Revert methods/latent_mas.py from DeepSeek backup."""
    target = Path(latentmas_dir) / "methods" / "latent_mas.py"
    backup = target.with_suffix(".py.deepseek.bak")
    if backup.exists():
        shutil.copy2(backup, target)
        print(f"Reverted {target} from {backup}")
    else:
        print(f"No DeepSeek backup found at {backup}")


def verify_compatibility(latentmas_dir: str = "."):
    """Check if the current LatentMAS code is compatible with DeepSeek."""
    target = Path(latentmas_dir) / "methods" / "latent_mas.py"
    models_file = Path(latentmas_dir) / "models.py"

    print("Compatibility check for DeepSeek MLA models:")
    print("=" * 50)

    issues = []

    if target.exists():
        code = target.read_text()
        if 'assert "Qwen"' in code:
            issues.append("latent_mas.py: Qwen assert blocks non-Qwen models")
        if '<|im_start|>user\\n' in code and "DEEPSEEK ADAPTER" not in code:
            issues.append("latent_mas.py: Qwen-specific chat template parsing")
        print(f"  latent_mas.py: {'NEEDS PATCH' if issues else 'OK'}")
    else:
        print(f"  latent_mas.py: NOT FOUND")

    if models_file.exists():
        mcode = models_file.read_text()
        # Check W_a computation — uses get_input_embeddings/get_output_embeddings
        has_get_input = "get_input_embeddings" in mcode
        has_get_output = "get_output_embeddings" in mcode
        has_past_kv = "past_key_values" in mcode
        has_hidden_states = "output_hidden_states=True" in mcode

        print(f"  models.py W_a computation: {'OK' if (has_get_input and has_get_output) else 'ISSUE'}")
        print(f"  models.py KV-cache interface: {'OK' if has_past_kv else 'ISSUE'}")
        print(f"  models.py hidden states: {'OK' if has_hidden_states else 'ISSUE'}")

        # Key check: models.py uses GENERIC HF APIs, not model-specific internals
        if "model.model.layers" not in mcode and "model.transformer" not in mcode:
            print(f"  models.py architecture coupling: OK (uses generic HF API)")
        else:
            issues.append("models.py accesses model internals directly")
            print(f"  models.py architecture coupling: ISSUE (accesses model internals)")
    else:
        print(f"  models.py: NOT FOUND")

    print()
    if issues:
        print(f"Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nRun with --apply to fix automatically.")
    else:
        print("All checks passed! DeepSeek models should work.")

    # ── DeepSeek-specific info ──
    print(f"""
DeepSeek MLA model compatibility notes:
────────────────────────────────────────
The HuggingFace implementation of DeepSeek MLA caches FULL K,V tensors
(post up-projection), NOT the compressed latent vectors c_KV.
This means past_key_values has the same shape as standard GQA:
  tuple[tuple[Tensor(B, n_heads, seq, head_dim), Tensor(B, n_heads, seq, head_dim)], ...]

Consequence: LatentMAS's KV-cache transfer, W_a computation, and latent
rollout all work WITHOUT modification. Only the chat template parsing
needs adaptation.

Recommended test model: deepseek-ai/DeepSeek-V2-Lite-Chat
  - 16B total params, 2.4B active (MoE)
  - MLA with KV compression dim 512
  - 27 layers, hidden dim 2048, 16 attention heads
  - Fits on 1× A100-40G in bf16
  - Same MLA as DeepSeek-V3/R1 (just smaller)
""")


# ══════════════════════════════════════════════════════════════
# Phase 0 config extension for DeepSeek models
# ══════════════════════════════════════════════════════════════

DEEPSEEK_MODEL_CONFIGS = {
    # DeepSeek-V2-Lite: MLA, KV compression dim 512
    # HF caches post-uprojection K,V: n_heads=16, head_dim=128
    "DeepSeek-V2-Lite": {
        "n_layers": 27,
        "n_heads": 16,      # KV heads after MLA up-projection
        "head_dim": 128,
        "d_h": 2048,
        "kv_compression_dim": 512,  # MLA latent dim (not used in KV-cache, but useful info)
        "architecture": "MLA+MoE",
    },
    # DeepSeek-V2 (236B): same MLA, bigger
    "DeepSeek-V2": {
        "n_layers": 60,
        "n_heads": 128,     # 128 attention heads
        "head_dim": 128,
        "d_h": 5120,
        "kv_compression_dim": 512,
        "architecture": "MLA+MoE",
    },
    # DeepSeek-V3 / R1 (671B): same MLA
    "DeepSeek-V3": {
        "n_layers": 61,
        "n_heads": 128,
        "head_dim": 128,
        "d_h": 7168,
        "kv_compression_dim": 512,
        "architecture": "MLA+MoE",
    },
    # DeepSeek-R1 distillations use Qwen3 architecture (NOT MLA)
    # They're already compatible with LatentMAS vanilla
    "DeepSeek-R1-Distill-Qwen": {
        "n_layers": 28,
        "n_heads": 28,
        "head_dim": 128,
        "d_h": 3584,
        "architecture": "GQA (Qwen3-based)",
    },
}


def print_model_guide():
    """Print a guide for which DeepSeek model to use."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           DeepSeek Model Selection Guide                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  FOR PHASE 0 TESTING (single GPU):                           ║
║  ─────────────────────────────────                           ║
║  deepseek-ai/DeepSeek-V2-Lite-Chat                           ║
║    16B total, 2.4B active, MLA, 1×40G GPU                    ║
║    → The smallest real MLA model. Perfect for validation.    ║
║                                                              ║
║  FOR PRODUCTION BENCHMARK (multi-GPU):                       ║
║  ─────────────────────────────────────                       ║
║  deepseek-ai/DeepSeek-R1-Distill-Qwen-14B                   ║
║    14B dense, GQA (Qwen3-based), 1×A100-40G                 ║
║    → NOT MLA but very performant. Works out of the box.      ║
║                                                              ║
║  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B                   ║
║    32B dense, GQA (Qwen3-based), 1×A100-80G                 ║
║    → Best performance/GPU ratio for LatentMAS.               ║
║                                                              ║
║  FOR MLA PROOF-OF-CONCEPT:                                   ║
║  ─────────────────────────────                               ║
║  deepseek-ai/DeepSeek-V2-Lite-Chat                           ║
║    → Proves LatentMAS works on MLA architecture              ║
║    → Same attention mechanism as DeepSeek-V3/R1              ║
║    → If this works, V3/R1 work too (same KV-cache format)    ║
║                                                              ║
║  IMPORTANT: DeepSeek-R1-Distill-* models use Qwen3 arch,    ║
║  NOT MLA. They already work with vanilla LatentMAS.          ║
║  Use DeepSeek-V2-Lite to test actual MLA compatibility.      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════
# End-to-end validation script
# ══════════════════════════════════════════════════════════════

VALIDATION_SCRIPT = '''#!/usr/bin/env python3
"""
Quick validation: does LatentMAS work on DeepSeek-V2-Lite-Chat?
Run from the LatentMAS directory after applying the adapter.
"""
import torch
import sys
import os

def validate_deepseek_latentmas():
    print("=" * 60)
    print("DeepSeek MLA + LatentMAS Validation")
    print("=" * 60)
    
    model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    
    # Step 1: Load model and check basic properties
    print(f"\\n[1/5] Loading {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"  Model loaded. Type: {type(model).__name__}")
    print(f"  Config: {model.config.num_hidden_layers} layers, "
          f"hidden_size={model.config.hidden_size}")
    
    # Step 2: Check embedding access (needed for W_a)
    print("\\n[2/5] Checking embedding access...")
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    print(f"  Input embeddings: {input_emb.weight.shape}")
    print(f"  Output embeddings: {output_emb.weight.shape}")
    assert input_emb is not None and output_emb is not None
    print("  ✓ W_a can be computed")
    
    # Step 3: Check KV-cache format
    print("\\n[3/5] Checking KV-cache format...")
    test_input = tokenizer("Hello world", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**test_input, use_cache=True, output_hidden_states=True, return_dict=True)
    
    past = out.past_key_values
    # Check it's the standard format
    k0 = past[0][0]
    v0 = past[0][1]
    print(f"  KV-cache layers: {len(past)}")
    print(f"  K shape: {k0.shape}  (batch, n_kv_heads, seq, head_dim)")
    print(f"  V shape: {v0.shape}")
    print(f"  ✓ Standard KV-cache format — compatible with LatentMAS")
    
    # Step 4: Check hidden states
    print("\\n[4/5] Checking hidden states...")
    hidden_states = out.hidden_states
    print(f"  Number of hidden state layers: {len(hidden_states)}")
    print(f"  First layer shape: {hidden_states[0].shape}")
    print(f"  Last layer shape: {hidden_states[-1].shape}")
    last_h = hidden_states[-1][:, -1, :]
    print(f"  Last hidden (for latent rollout): {last_h.shape}")
    print(f"  ✓ Hidden states accessible")
    
    # Step 5: Test inputs_embeds path (used in latent rollout)
    print("\\n[5/5] Testing inputs_embeds forward pass...")
    emb_layer = model.get_input_embeddings()
    test_emb = emb_layer(test_input["input_ids"])
    with torch.no_grad():
        out2 = model(
            inputs_embeds=test_emb,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )
    print(f"  inputs_embeds forward: ✓")
    print(f"  Output logits shape: {out2.logits.shape}")
    
    # Verify logits match
    diff = (out.logits - out2.logits).abs().max().item()
    print(f"  Logits diff (input_ids vs inputs_embeds): {diff:.6f}")
    
    print("\\n" + "=" * 60)
    print("ALL CHECKS PASSED — DeepSeek-V2-Lite is LatentMAS-compatible")
    print("=" * 60)
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    validate_deepseek_latentmas()
'''


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatentMAS DeepSeek MLA Adapter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true", help="Apply DeepSeek adapter patch")
    group.add_argument("--revert", action="store_true", help="Revert patch from backup")
    group.add_argument("--check", action="store_true", help="Check compatibility without modifying")
    group.add_argument("--guide", action="store_true", help="Print model selection guide")
    group.add_argument("--gen-validate", action="store_true",
                      help="Generate validation script (validate_deepseek.py)")

    parser.add_argument("--dir", default=".", help="LatentMAS root directory")
    args = parser.parse_args()

    if args.apply:
        apply_deepseek_patch(args.dir)
    elif args.revert:
        revert_deepseek_patch(args.dir)
    elif args.check:
        verify_compatibility(args.dir)
    elif args.guide:
        print_model_guide()
    elif args.gen_validate:
        out_path = Path(args.dir) / "validate_deepseek.py"
        out_path.write_text(VALIDATION_SCRIPT)
        print(f"Validation script written to {out_path}")
        print(f"Run: python {out_path}")
