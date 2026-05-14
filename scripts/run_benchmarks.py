#!/usr/bin/env python3
"""
Main benchmark orchestrator for TensorRT-LLM + NIM inference benchmarks.

Usage:
    python run_benchmarks.py --phase baseline
    python run_benchmarks.py --phase sweep
    python run_benchmarks.py --phase nim
    python run_benchmarks.py --phase all

Each phase reads from configs/ and writes results to results/raw/ as JSON.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_config(name: str) -> dict:
    """Load a YAML config from configs/."""
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def run_trtllm_bench(
    engine_dir: str,
    batch_size: int = 1,
    input_len: int = 128,
    output_len: int = 256,
    num_warmup: int = 3,
    num_runs: int = 10,
    cuda_graph: bool = True,
    tp_size: int = 1,
    label: str = "",
) -> dict:
    """
    Run trtllm-bench and capture results.

    Returns parsed metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark: {label}")
    print(f"  Batch={batch_size}, InLen={input_len}, OutLen={output_len}")
    print(f"  CudaGraph={cuda_graph}, TP={tp_size}")
    print(f"{'='*60}")

    output_file = RESULTS_DIR / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    cmd = [
        "trtllm-bench",
        "throughput",
        "--engine_dir", engine_dir,
        "--batch_size", str(batch_size),
        "--input_length", str(input_len),
        "--output_length", str(output_len),
        "--warmup", str(num_warmup),
        "--num_runs", str(num_runs),
    ]

    if cuda_graph:
        cmd.append("--enable_cuda_graph")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    metrics = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "batch_size": batch_size,
            "input_len": input_len,
            "output_len": output_len,
            "cuda_graph": cuda_graph,
            "tp_size": tp_size,
            "num_warmup": num_warmup,
            "num_runs": num_runs,
        },
        "elapsed_seconds": round(elapsed, 2),
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],  # Keep last 2KB
        "stderr": result.stderr[-2000:],
    }

    # Parse key metrics from trtllm-bench output
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "TTFT" in line or "Time to First Token" in line:
                try:
                    metrics["ttft_ms"] = float(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if "TPOT" in line or "Time per Output Token" in line:
                try:
                    metrics["tpot_ms"] = float(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if "tokens/s" in line.lower() or "throughput" in line.lower():
                try:
                    metrics["throughput_tps"] = float(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if "GPU Memory" in line or "memory" in line.lower():
                try:
                    metrics["gpu_memory_mb"] = float(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

    # Save raw output
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Results saved to {output_file}")

    return metrics


def run_baseline():
    """Phase A: Baseline BF16 TP=1 benchmarks."""
    config = load_config("baseline")
    model_dir = os.path.expanduser(config.get("model_dir", "~/models/llama-3.1-8b-instruct"))
    engine_dir = os.path.join(model_dir, "trtllm-bf16", "engine")

    if not os.path.exists(engine_dir):
        print(f"ERROR: Engine not found at {engine_dir}. Run setup.sh first.")
        sys.exit(1)

    results = []
    for batch_size in config.get("batch_sizes", [1, 8, 32, 64]):
        for out_len in config.get("output_lengths", [128, 256, 512]):
            result = run_trtllm_bench(
                engine_dir=engine_dir,
                batch_size=batch_size,
                input_len=config.get("input_len", 128),
                output_len=out_len,
                label=f"baseline_b{batch_size}_o{out_len}",
            )
            results.append(result)

    return results


def run_sweep():
    """Phase B: Optimization sweep across quantization + batching configs."""
    all_results = []

    # FP8 sweep
    fp8_config = load_config("sweep_fp8")
    model_dir = os.path.expanduser(fp8_config.get("model_dir", "~/models/llama-3.1-8b-instruct"))

    fp8_engine = os.path.join(model_dir, "trtllm-fp8", "engine")
    if os.path.exists(fp8_engine):
        for batch_size in fp8_config.get("batch_sizes", [1, 8, 32, 64, 128]):
            result = run_trtllm_bench(
                engine_dir=fp8_engine,
                batch_size=batch_size,
                label=f"fp8_b{batch_size}",
            )
            all_results.append(result)
    else:
        print(f"  FP8 engine not found at {fp8_engine}, skipping.")

    # AWQ sweep
    awq_config = load_config("sweep_awq")
    awq_engine = os.path.join(model_dir, "trtllm-awq4", "engine")
    if os.path.exists(awq_engine):
        for batch_size in awq_config.get("batch_sizes", [1, 8, 32, 64]):
            result = run_trtllm_bench(
                engine_dir=awq_engine,
                batch_size=batch_size,
                label=f"awq4_b{batch_size}",
            )
            all_results.append(result)
    else:
        print(f"  AWQ engine not found at {awq_engine}, skipping.")

    # CUDA Graph comparison
    bf16_engine = os.path.join(model_dir, "trtllm-bf16", "engine")
    if os.path.exists(bf16_engine):
        for cuda_graph in [True, False]:
            result = run_trtllm_bench(
                engine_dir=bf16_engine,
                batch_size=32,
                cuda_graph=cuda_graph,
                label=f"cuda_graph_{'on' if cuda_graph else 'off'}",
            )
            all_results.append(result)

    return all_results


def run_nim():
    """Phase C: NIM container benchmarking."""
    config = load_config("nim_config")
    nim_results = []

    # NIM benchmarks are handled by benchmark_nim.py
    print("Running NIM benchmarks via benchmark_nim.py...")
    nim_script = Path(__file__).parent / "benchmark_nim.py"
    if nim_script.exists():
        result = subprocess.run(
            [sys.executable, str(nim_script)],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"NIM benchmark error: {result.stderr}")

    return nim_results


def main():
    parser = argparse.ArgumentParser(description="Inference Benchmark Orchestrator")
    parser.add_argument("--phase", choices=["baseline", "sweep", "nim", "all"], required=True)
    parser.add_argument("--model_dir", default=None, help="Override model directory")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {"started": datetime.now().isoformat(), "runs": []}

    if args.phase in ("baseline", "all"):
        print("\n" + "="*60)
        print("  PHASE A: BASELINE (BF16, TP=1)")
        print("="*60)
        results = run_baseline()
        all_results["runs"].extend(results)

    if args.phase in ("sweep", "all"):
        print("\n" + "="*60)
        print("  PHASE B: OPTIMIZATION SWEEP")
        print("="*60)
        results = run_sweep()
        all_results["runs"].extend(results)

    if args.phase in ("nim", "all"):
        print("\n" + "="*60)
        print("  PHASE C: NIM DEPLOYMENT")
        print("="*60)
        results = run_nim()
        all_results["runs"].extend(results)

    all_results["completed"] = datetime.now().isoformat()
    all_results["total_runs"] = len(all_results["runs"])

    summary_file = RESULTS_DIR / f"summary_{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done! {len(all_results['runs'])} benchmarks completed.")
    print(f"  Summary: {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
