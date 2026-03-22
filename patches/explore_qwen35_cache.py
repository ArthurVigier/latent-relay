"""
explore_qwen35_cache.py — Exploration de la structure cache Qwen3.5
Exécuter sur un GPU avec au moins 8GB VRAM (Qwen3.5-0.8B) ou 16GB (Qwen3.5-4B).
Nécessite : pip install git+https://github.com/huggingface/transformers.git@main
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Utiliser le plus petit modèle pour l'exploration
MODEL = "Qwen/Qwen3.5-0.8B"  # ou Qwen3.5-4B si assez de VRAM

print(f"[1] Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
).cuda().eval()

print(f"\n[2] Model config inspection")
config = model.config
text_config = getattr(config, 'text_config', config)
print(f"  model_type: {getattr(text_config, 'model_type', 'unknown')}")
print(f"  num_hidden_layers: {text_config.num_hidden_layers}")
print(f"  hidden_size: {text_config.hidden_size}")

layer_types = getattr(text_config, 'layer_types', None)
full_attn_interval = getattr(text_config, 'full_attention_interval', None)
print(f"  layer_types: {layer_types}")
print(f"  full_attention_interval: {full_attn_interval}")
if layer_types:
    n_linear = sum(1 for t in layer_types if t != 'full_attention')
    n_full = sum(1 for t in layer_types if t == 'full_attention')
    print(f"  → {n_linear} GDN layers, {n_full} full attention layers")

print(f"\n  GDN config:")
print(f"    linear_attn_head_dim: {getattr(text_config, 'linear_attn_head_dim', 'N/A')}")
print(f"    linear_attn_qk_heads: {getattr(text_config, 'num_linear_attn_qk_heads', 'N/A')}")
print(f"    linear_attn_v_heads: {getattr(text_config, 'num_linear_attn_v_heads', 'N/A')}")
print(f"    linear_conv_kernel: {getattr(text_config, 'linear_conv_kernel_dim', 'N/A')}")

print(f"\n  Full attention config:")
print(f"    num_attention_heads: {getattr(text_config, 'num_attention_heads', 'N/A')}")
print(f"    num_key_value_heads: {getattr(text_config, 'num_key_value_heads', 'N/A')}")
print(f"    head_dim: {getattr(text_config, 'head_dim', 'N/A')}")

print(f"\n[3] Forward pass with cache...")
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs, use_cache=True, output_hidden_states=True, return_dict=True)

past = out.past_key_values
print(f"\n[4] Cache object analysis")
print(f"  type: {type(past)}")
print(f"  type MRO: {[c.__name__ for c in type(past).__mro__]}")
print(f"  dir (non-private): {[a for a in dir(past) if not a.startswith('_')]}")

print(f"\n  Has key_cache: {hasattr(past, 'key_cache')}")
print(f"  Has value_cache: {hasattr(past, 'value_cache')}")
print(f"  Has get_seq_length: {hasattr(past, 'get_seq_length')}")

if hasattr(past, 'key_cache') and past.key_cache:
    print(f"\n  key_cache length: {len(past.key_cache)}")
    for i in range(min(8, len(past.key_cache))):
        k = past.key_cache[i]
        v = past.value_cache[i] if hasattr(past, 'value_cache') else None
        k_info = f"shape={k.shape}, dtype={k.dtype}" if k is not None else "None"
        v_info = f"shape={v.shape}, dtype={v.dtype}" if v is not None else "None"
        layer_type = layer_types[i] if layer_types and i < len(layer_types) else "?"
        print(f"  Layer {i:2d} [{layer_type:15s}]: key={k_info}, val={v_info}")

for attr in ['state_cache', 'linear_state', 'gdn_states', 'recurrent_states',
             'conv_states', 'ssm_states', 'delta_states']:
    if hasattr(past, attr):
        val = getattr(past, attr)
        print(f"\n  Found {attr}: type={type(val)}")
        if isinstance(val, (list, tuple)):
            print(f"    length: {len(val)}")
            for i in range(min(4, len(val))):
                item = val[i]
                if isinstance(item, torch.Tensor):
                    print(f"    [{i}]: shape={item.shape}, dtype={item.dtype}")
                elif isinstance(item, (tuple, list)):
                    print(f"    [{i}]: tuple of {len(item)} items, shapes={[t.shape if hasattr(t,'shape') else type(t) for t in item]}")
                else:
                    print(f"    [{i}]: type={type(item)}")

if hasattr(past, 'get_seq_length'):
    try:
        seq_len = past.get_seq_length()
        print(f"\n  get_seq_length(): {seq_len}")
    except Exception as e:
        print(f"\n  get_seq_length() error: {e}")

hs = out.hidden_states
print(f"\n[5] Hidden states")
print(f"  Count: {len(hs)} (= {text_config.num_hidden_layers} layers + 1 embedding)")
print(f"  Shape: {hs[-1].shape}")
print(f"  Dtype: {hs[-1].dtype}")

print(f"\n[6] Embedding matrices")
input_emb = model.get_input_embeddings()
output_emb = model.get_output_embeddings()
print(f"  Input embeddings: {input_emb.weight.shape}")
print(f"  Output embeddings: {output_emb.weight.shape if output_emb is not None else 'None (tied)'}")
if output_emb is None or (hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings):
    print(f"  ⚠️  TIED EMBEDDINGS — W_a computation needs adjustment")

print(f"\n[7] Cache re-injection test")
next_input = tokenizer(" Paris", return_tensors="pt").to("cuda")
try:
    past_len = past.get_seq_length() if hasattr(past, 'get_seq_length') else 0
    attn_mask = torch.ones((1, past_len + next_input["input_ids"].shape[1]),
                           dtype=torch.long, device="cuda")
    with torch.no_grad():
        out2 = model(
            input_ids=next_input["input_ids"],
            attention_mask=attn_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    print(f"  ✓ Cache re-injection works!")
    print(f"  New cache type: {type(out2.past_key_values)}")
    new_past = out2.past_key_values
    if hasattr(new_past, 'get_seq_length'):
        print(f"  New seq_length: {new_past.get_seq_length()}")
except Exception as e:
    print(f"  ✗ Cache re-injection FAILED: {e}")
    import traceback
    traceback.print_exc()

print(f"\n[8] inputs_embeds test (critical for latent rollout)")
try:
    dummy_embed = torch.randn(1, 1, text_config.hidden_size,
                              dtype=torch.bfloat16, device="cuda")
    past_len = past.get_seq_length() if hasattr(past, 'get_seq_length') else 0
    attn_mask = torch.ones((1, past_len + 1), dtype=torch.long, device="cuda")
    with torch.no_grad():
        out3 = model(
            inputs_embeds=dummy_embed,
            attention_mask=attn_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    print(f"  ✓ inputs_embeds with past_key_values works!")
    print(f"  Output hidden shape: {out3.hidden_states[-1].shape}")
except Exception as e:
    print(f"  ✗ inputs_embeds FAILED: {e}")
    import traceback
    traceback.print_exc()

print(f"\n[9] Cache serialization test")
import io, time
try:
    buf = io.BytesIO()
    t0 = time.time()
    torch.save(past, buf)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    size = buf.tell()
    t_save = time.time() - t0
    buf.seek(0)
    t0 = time.time()
    past_loaded = torch.load(buf, weights_only=False, map_location="cuda")  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    t_load = time.time() - t0
    print(f"  ✓ Serialization works")
    print(f"  Size: {size / 1024:.1f} KB")
    print(f"  Save: {t_save:.3f}s, Load: {t_load:.3f}s")
    print(f"  Loaded type: {type(past_loaded)}")
except Exception as e:
    print(f"  ✗ Serialization FAILED: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")
print(f"EXPLORATION COMPLETE — Colle ces résultats dans Claude Code")
print(f"{'='*60}")
