"""
Smoke test for eris_server.py — validates all 5 ERIS endpoints.
Run with the server already started:

    python eris_server.py --model Qwen/Qwen3-4B --port 8001

Then:
    python patches/smoke_test_eris.py --port 8001

Phase 2 (Claude bridge) runs only if ANTHROPIC_API_KEY is set.
"""
import argparse, base64, os, sys, time
import requests
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--n_steps", type=int, default=5,
                    help="latent steps for think/bridge (keep low for speed)")
args = parser.parse_args()

BASE = f"http://localhost:{args.port}"
PASS = "✓"; FAIL = "✗"
results = {}

def check(name, ok, detail=""):
    results[name] = ok
    mark = PASS if ok else FAIL
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))
    return ok

def post(path, body):
    r = requests.post(f"{BASE}{path}", json=body, timeout=120)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text[:300])
        except Exception:
            detail = r.text[:300]
        raise requests.HTTPError(
            f"HTTP {r.status_code} {path}: {detail}", response=r
        )
    return r.json()

def get(path):
    r = requests.get(f"{BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

# ── Health check ──────────────────────────────────────────────────────────────
print("\n[0] Health check")
try:
    r = requests.get(BASE, timeout=10)
    check("server reachable", r.status_code < 500, f"HTTP {r.status_code}")
except Exception as e:
    print(f"  {FAIL} server unreachable: {e}")
    print("  → Start server first: python eris_server.py --model Qwen/Qwen3-4B --port 8001")
    sys.exit(1)

# ── Session ───────────────────────────────────────────────────────────────────
print("\n[1] Session")
try:
    sid = post("/sessions", {})["session_id"]
    check("create session", bool(sid), f"session_id={sid}")
except Exception as e:
    check("create session", False, str(e)); sys.exit(1)

TEXT = "The key insight in distributed systems is that network partitions are inevitable."

# ── Endpoint 1 : /v1/encode ───────────────────────────────────────────────────
print("\n[2] /v1/encode")
enc_handle = None
enc = {}
try:
    enc = post("/v1/encode", {
        "text": TEXT,
        "return_layers": [0, -1],
        "session_id": sid,
        "compact": True,
    })
    check("has handle",        "handle" in enc and enc["handle"])
    check("has hidden_states", "hidden_states" in enc)
    check("has 'last' layer",  "last" in enc["hidden_states"])
    check("has tokens",        len(enc.get("tokens", [])) > 0)

    # Decode base64 and verify shape
    raw = base64.b64decode(enc["hidden_states"]["last"])
    arr = np.frombuffer(raw, dtype=np.float32).reshape(enc["seq_len"], enc["hidden_dim"])
    check("hidden shape valid",
          arr.shape == (enc["seq_len"], enc["hidden_dim"]),
          f"{arr.shape}")
    enc_handle = enc["handle"]
except Exception as e:
    check("encode", False, str(e))

# ── Endpoint 2 : /v1/analyze ──────────────────────────────────────────────────
print("\n[3] /v1/analyze")
if enc_handle:
    try:
        ana = post("/v1/analyze", {
            "handle":     enc_handle,
            "session_id": sid,           # required field
            "analyses":   ["token_norms", "pca_3d", "cosine_map",
                           "sae_features", "a_hat"],
        })
        check("token_norms present",   "token_norms" in ana)
        check("token_norms non-empty", len(ana.get("token_norms", [])) > 0,
              f"{len(ana.get('token_norms', []))} norms")
        check("pca_3d present",        "pca_3d" in ana)
        # SAE/a_hat may be null — graceful degradation expected without checkpoints
        check("sae_features key present (null ok)", "sae_features" in ana,
              "null (no checkpoint)" if not ana.get("sae_features") else "loaded")
        check("a_hat key present (null ok)", "a_hat_score" in ana,
              "null (no checkpoint)" if ana.get("a_hat_score") is None else
              f"score={ana['a_hat_score']:.3f}")
    except Exception as e:
        check("analyze", False, str(e))
else:
    print("  (skipped — encode failed)")

# ── Endpoint 3 : /v1/latent_think ─────────────────────────────────────────────
print(f"\n[4] /v1/latent_think  (n_steps={args.n_steps})")
think_handle = None
try:
    think = post("/v1/latent_think", {
        "session_id":        sid,
        "prompt":            TEXT,
        "n_steps":           args.n_steps,
        "role":              "planner",
        "return_trajectory": True,
    })
    check("has handle",       "handle" in think and think["handle"])
    check("has trajectory",   "trajectory" in think and
          len(think["trajectory"]) == args.n_steps,
          f"{len(think.get('trajectory', []))} steps")
    check("has displacement", "total_displacement" in think,
          f"{think.get('total_displacement', 'N/A'):.4f}")
    check("n_positions > 0",  think.get("n_positions", 0) > 0,
          f"{think.get('n_positions')} positions")
    think_handle = think["handle"]
except Exception as e:
    check("latent_think", False, str(e))

# ── Endpoint 4 : /v1/inject ───────────────────────────────────────────────────
print("\n[5] /v1/inject")
if think_handle:
    try:
        hidden_dim = enc.get("hidden_dim", 2560)
        # Zero vector + scale=0.0 → no-op injection
        inj = post("/v1/inject", {
            "session_id": sid,
            "handle":     think_handle,
            "operation":  "add",
            "layer":      -1,
            "position":   -1,
            "vector":     [0.0] * hidden_dim,
            "scale":      0.0,
        })
        check("status = injected",  inj.get("status") == "injected",
              f"status={inj.get('status')}")
        check("old_norm present",   "old_norm" in inj,
              f"{inj.get('old_norm', 0):.3f}")
        # cosine_shift is the CHANGE in direction (0 = no change, 1 = orthogonal)
        # A zero-vector no-op injection should produce cosine_shift ≈ 0
        check("cosine_shift ≈ 0 (no-op)", abs(inj.get("cosine_shift", 1)) < 0.01,
              f"shift={inj.get('cosine_shift'):.4f}")
    except Exception as e:
        check("inject", False, str(e))
else:
    print("  (skipped — think failed)")

# ── Endpoint 5 : /v1/bridge ───────────────────────────────────────────────────
print(f"\n[6] /v1/bridge  (passive mode)")
try:
    bridge_passive = post("/v1/bridge", {
        "claude_text": TEXT,
        "mode":        "passive",
        "analyses":    ["token_norms"],
        "decode_after": False,
    })
    check("has analysis",        "analysis" in bridge_passive)
    check("has handles",         "handles" in bridge_passive)
    check("token_norms present", "token_norms" in bridge_passive.get("analysis", {}))
except Exception as e:
    check("bridge passive", False, str(e))

print(f"\n[7] /v1/bridge  (ruminate, n_steps={args.n_steps}, decode_after=True)")
try:
    bridge_rum = post("/v1/bridge", {
        "claude_text":    TEXT,
        "mode":           "ruminate",
        "n_steps":        args.n_steps,
        "analyses":       ["token_norms"],
        "decode_after":   True,
        "max_new_tokens": 64,
    })
    check("has enriched_text",      bool(bridge_rum.get("enriched_text")),
          f"{len(bridge_rum.get('enriched_text', ''))} chars")
    check("has trajectory_summary", "trajectory_summary" in bridge_rum)
    check("has implicit_features",  "implicit_features" in bridge_rum.get("analysis", {}))
    print(f"    enriched: {bridge_rum.get('enriched_text', '')[:120]}")
except Exception as e:
    check("bridge ruminate", False, str(e))

# ── Session cleanup ───────────────────────────────────────────────────────────
try:
    requests.delete(f"{BASE}/sessions/{sid}", timeout=10)
except Exception:
    pass

# ── Phase 2 : Claude bridge (optional) ───────────────────────────────────────
api_key = os.environ.get("ANTHROPIC_API_KEY")
if api_key:
    print(f"\n[8] ClaudeZombieBridge  (ANTHROPIC_API_KEY found)")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eris_client import ClaudeZombieBridge

        bridge = ClaudeZombieBridge(
            anthropic_api_key=api_key,
            eris_base_url=BASE,
            bridge_mode="passive",
            n_steps=args.n_steps,
        )
        turn = bridge.chat("What are the trade-offs of eventual consistency?")
        check("claude_text non-empty",   len(turn.claude_text) > 10)
        check("analysis present",        turn.analysis is not None)
        print(f"    claude: {turn.claude_text[:120]}")
    except Exception as e:
        check("ClaudeZombieBridge", False, str(e))
else:
    print("\n[8] ClaudeZombieBridge — skipped (set ANTHROPIC_API_KEY to enable)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
total  = len(results)
passed = sum(1 for v in results.values() if v)
failed = total - passed
print(f"Result: {passed}/{total} passed" +
      (f"  ({failed} FAILED)" if failed else "  — all OK"))
if failed:
    print("Failed checks:")
    for k, v in results.items():
        if not v:
            print(f"  {FAIL} {k}")
    sys.exit(1)
