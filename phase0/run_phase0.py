#!/usr/bin/env python3
"""
LatentMAS Phase 0 — Feasibility Validation Suite
=================================================
Runs three go/no-go tests before investing in building the Latent Relay Server.

Test 0.1: KV-cache serialization latency (GPU→CPU→bytes→CPU→GPU)
Test 0.2: Accuracy degradation under int8 quantization of KV-caches
Test 0.3: W_a determinism and portability

Usage:
    # Run all tests (requires GPU + LatentMAS repo cloned)
    python run_phase0.py --model Qwen/Qwen3-4B --all

    # Run individual tests
    python run_phase0.py --model Qwen/Qwen3-4B --test serialization
    python run_phase0.py --model Qwen/Qwen3-4B --test quantization --n_samples 50
    python run_phase0.py --model Qwen/Qwen3-4B --test wa_determinism

    # Dry run with synthetic tensors (no model needed, tests infra only)
    python run_phase0.py --dry-run --test serialization

Requirements:
    pip install torch transformers accelerate lz4 rich
    # Optional for Test 0.2: clone LatentMAS repo
    # git clone https://github.com/Gen-Verse/LatentMAS.git

Author: Phase 0 feasibility kit for LatentMAS → MCP tooling project
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="LatentMAS Phase 0 — Feasibility Validation Suite"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B",
        help="HuggingFace model name (default: Qwen/Qwen3-4B for cheap tests)"
    )
    parser.add_argument(
        "--test", type=str, choices=["serialization", "quantization", "wa_determinism", "all"],
        default="all", help="Which test to run"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all tests (same as --test all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use synthetic tensors instead of real model (tests infra only)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Number of GSM8K samples for quantization test (default: 50)"
    )
    parser.add_argument(
        "--n_latent_steps", type=int, default=60,
        help="Number of latent reasoning steps (default: 60)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./phase0_results",
        help="Directory for test results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="GPU device (default: cuda:0)"
    )

    args = parser.parse_args()
    if args.all:
        args.test = "all"

    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "dry_run": args.dry_run,
        "tests": {}
    }

    tests_to_run = (
        ["serialization", "quantization", "wa_determinism"]
        if args.test == "all"
        else [args.test]
    )

    # ── Print header ──
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel(
            "[bold]LatentMAS Phase 0 — Feasibility Validation[/bold]\n"
            f"Model: {args.model}\n"
            f"Device: {args.device}\n"
            f"Tests: {', '.join(tests_to_run)}\n"
            f"Dry run: {args.dry_run}",
            title="Configuration", border_style="blue"
        ))
    except ImportError:
        print("=" * 60)
        print("LatentMAS Phase 0 — Feasibility Validation")
        print(f"Model: {args.model} | Device: {args.device}")
        print(f"Tests: {', '.join(tests_to_run)} | Dry run: {args.dry_run}")
        print("=" * 60)
        console = None

    # ── Run tests ──
    for test_name in tests_to_run:
        if test_name == "serialization":
            from test_01_serialization import run_serialization_test
            result = run_serialization_test(
                model_name=args.model,
                device=args.device,
                n_latent_steps=args.n_latent_steps,
                dry_run=args.dry_run,
                output_dir=args.output_dir
            )
        elif test_name == "quantization":
            from test_02_quantization import run_quantization_test
            result = run_quantization_test(
                model_name=args.model,
                device=args.device,
                n_latent_steps=args.n_latent_steps,
                n_samples=args.n_samples,
                dry_run=args.dry_run,
                output_dir=args.output_dir
            )
        elif test_name == "wa_determinism":
            from test_03_wa_determinism import run_wa_determinism_test
            result = run_wa_determinism_test(
                model_name=args.model,
                device=args.device,
                dry_run=args.dry_run,
                output_dir=args.output_dir
            )

        results["tests"][test_name] = result

        # Print verdict
        verdict = result.get("verdict", "UNKNOWN")
        if console:
            color = "green" if verdict == "GO" else "red" if verdict == "NO-GO" else "yellow"
            console.print(f"\n[bold {color}]Test {test_name}: {verdict}[/bold {color}]")
            if "reason" in result:
                console.print(f"  Reason: {result['reason']}")
        else:
            print(f"\nTest {test_name}: {verdict}")
            if "reason" in result:
                print(f"  Reason: {result['reason']}")

    # ── Save results ──
    results_path = Path(args.output_dir) / "phase0_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Final summary ──
    all_go = all(
        r.get("verdict") == "GO"
        for r in results["tests"].values()
    )

    if console:
        console.print("\n")
        if all_go:
            console.print(Panel(
                "[bold green]ALL TESTS PASSED — Phase 1 is GO[/bold green]\n"
                f"Results saved to {results_path}",
                border_style="green"
            ))
        else:
            failed = [k for k, v in results["tests"].items() if v.get("verdict") != "GO"]
            console.print(Panel(
                f"[bold red]BLOCKED — Failed tests: {', '.join(failed)}[/bold red]\n"
                f"Results saved to {results_path}\n"
                "Review results and consider pivot strategies.",
                border_style="red"
            ))
    else:
        print("\n" + "=" * 60)
        if all_go:
            print("ALL TESTS PASSED — Phase 1 is GO")
        else:
            failed = [k for k, v in results["tests"].items() if v.get("verdict") != "GO"]
            print(f"BLOCKED — Failed tests: {', '.join(failed)}")
        print(f"Results saved to {results_path}")

    sys.exit(0 if all_go else 1)


if __name__ == "__main__":
    main()
