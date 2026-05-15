#!/usr/bin/env python3
"""
AIPerf/GenAI-Perf wrapper for NIM benchmarking.

AIPerf is NVIDIA's recommended tool for benchmarking LLM serving endpoints.
It supports any OpenAI-compatible API and provides comprehensive metrics.

Reference: https://docs.nvidia.com/nim/benchmarking/llm/latest/step-by-step.html

Usage:
    python benchmark_aiperf.py --base-url http://localhost:8000 --model meta/llama-3.1-8b-instruct

Requirements:
    pip install genai-perf
    OR
    pip install aiperf
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"


def run_aiperf(
    base_url: str,
    model: str,
    num_requests: int = 100,
    concurrency: int = 10,
    input_len: int = 128,
    output_len: int = 256,
    streaming: bool = True,
    profile: str = "latency",
    label: str = "",
    timeout: int = 600,
) -> dict:
    """
    Run AIPerf (or GenAI-Perf) against a NIM endpoint.

    AIPerf provides: TTFT, ITL, TPS, RPS with P50/P90/P95/P99 percentiles.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / f"aiperf_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"\n{'='*60}")
    print(f"  AIPerf: {label}")
    print(f"  Model: {model}")
    print(f"  Concurrency: {concurrency}, Requests: {num_requests}")
    print(f"  ISL: {input_len}, OSL: {output_len}, Stream: {streaming}")
    print(f"{'='*60}")

    # Try genai-perf first, then aiperf
    cmd = None
    tool_name = None

    for tool in ["genai-perf", "aiperf"]:
        try:
            subprocess.run([tool, "--help"], capture_output=True, timeout=5)
            tool_name = tool
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if tool_name is None:
        print("  ERROR: Neither genai-perf nor aiperf found in PATH.")
        print("  Install with: pip install genai-perf")
        print("  Falling back to Python load tester...")
        # Fall back to our own benchmark_nim.py
        nim_script = Path(__file__).parent / "benchmark_nim.py"
        if nim_script.exists():
            result = subprocess.run(
                [sys.executable, str(nim_script)],
                capture_output=True, text=True,
            )
            print(result.stdout[-2000:])
        return {}

    # Build genai-perf command
    if tool_name == "genai-perf":
        cmd = [
            "genai-perf",
            "profile",
            "-m", model,
            "-u", base_url,
            "--service-kind", "openai",
            "--num-prompts", str(num_requests),
            "--request-concurrency", str(concurrency),
            "--profile", profile,
        ]

        if streaming:
            cmd.extend(["--streaming"])
        else:
            cmd.extend(["--no-streaming"])

        # Input/output lengths
        cmd.extend([
            "--input-len", str(input_len),
            "--output-len", str(output_len),
        ])

        # Output format
        cmd.extend(["--result-format", "json", "--result-dir", str(RESULTS_DIR)])

    elif tool_name == "aiperf":
        cmd = [
            "aiperf",
            "--url", base_url,
            "--model", model,
            "--num-requests", str(num_requests),
            "--concurrency", str(concurrency),
            "--input-tokens", str(input_len),
            "--output-tokens", str(output_len),
            "--stream" if streaming else "--no-stream",
            "--output", str(output_file),
        ]

    print(f"  CMD: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - start

    print(f"  Completed in {elapsed:.1f}s (rc={result.returncode})")
    if result.stdout:
        print(f"  Output (last 1500 chars):\n{result.stdout[-1500:]}")
    if result.returncode != 0 and result.stderr:
        print(f"  Error:\n{result.stderr[-1000:]}")

    # Try to find and parse the result JSON
    metrics = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "elapsed_seconds": round(elapsed, 2),
        "returncode": result.returncode,
    }

    # genai-perf saves to result-dir
    if tool_name == "genai-perf":
        for f in RESULTS_DIR.glob("genai_perf_*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    metrics["report"] = data
                    break
            except (json.JSONDecodeError, OSError):
                continue

    return metrics


def main():
    parser = argparse.ArgumentParser(description="AIPerf/GenAI-Perf NIM Benchmark Wrapper")
    parser.add_argument("--base-url", default="http://localhost:8000", help="NIM endpoint URL")
    parser.add_argument("--model", default="meta/llama-3.1-8b-instruct", help="Model name")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--streaming", action="store_true", default=True)
    args = parser.parse_args()

    all_results = []

    for conc in args.concurrency_levels:
        result = run_aiperf(
            base_url=args.base_url,
            model=args.model,
            num_requests=args.num_requests,
            concurrency=conc,
            input_len=args.input_len,
            output_len=args.output_len,
            streaming=args.streaming,
            label=f"conc{conc}",
        )
        all_results.append(result)

    summary_file = RESULTS_DIR / f"aiperf_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump({"runs": all_results}, f, indent=2)
    print(f"\nDone! Results saved to {summary_file}")


if __name__ == "__main__":
    main()
