# Qwen3.5 (Gated DeltaNet) — Validation Log

## Exploration results — Qwen3.5-0.8B

Script: `patches/explore_qwen35_cache.py`
Date: 2026-03-22

### Architecture

```
model_type         : qwen3_5_text
num_hidden_layers  : 24
hidden_size        : 1024
pattern            : [GDN, GDN, GDN, FullAttn] × 6  (full_attention_interval=4)
GDN layers         : 18   (linear_attention)
Full attn layers   : 6    (full_attention)
tie_word_embeddings: True
```

### Cache structure — Qwen3_5DynamicCache

```
key_cache[i]       → None       for linear_attention layers
key_cache[i]       → [1, 2, seq_len, 256]  for full_attention layers
recurrent_states[i]→ [1, 16, 128, 128] float32  (GDN state matrix S)
conv_states[i]     → [1, 6144, 4] bfloat16      (conv1d context)
```

### Test results

| Test | Result |
|------|--------|
| Q3 Cache re-injection (`past_key_values`) | ✅ works |
| Q4 `inputs_embeds` + hybrid cache | ✅ works |
| Q5 Serialization (torch.save/load) | ✅ works — 19MB/5tokens |

### Root cause of original incompatibility

`_past_length()` in engine.py accessed `key_cache[0].shape[-2]` directly.
For Qwen3.5, `key_cache[0]` is `None` (layer 0 is linear_attention) → AttributeError.

`Qwen3_5DynamicCache` has `get_seq_length()` which returns the correct value.
Fix: check `get_seq_length()` before accessing `key_cache` entries.

### Changes made to engine.py

1. `_past_length()` — `get_seq_length()` checked first; key_cache iteration
   skips None entries. **Backwards-compatible** with Qwen3, DeepSeek-V2.

2. `__init__()` — hybrid architecture detection log (informational only).

### Tied embeddings note

Qwen3.5 uses `tie_word_embeddings=True` — input and output embedding matrices
are identical. W_a ridge regression gives W_a ≈ I, so `_apply_realignment`
reduces to normalisation only. The latent rollout still works; there is no
cross-space alignment, just norm stabilisation.

This is expected behaviour. A future improvement could learn a proper
projection matrix for hybrid models.

### Validated models

| Model | VRAM | Status |
|-------|------|--------|
| Qwen3.5-0.8B | 2GB | ✅ exploration complete |
| Qwen3.5-4B | ~8GB | pending |
| Qwen3-4B | ~8GB | existing — regression test pending |
