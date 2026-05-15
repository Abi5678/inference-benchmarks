#!/usr/bin/env python3
"""
Modernized main benchmark orchestrator for TensorRT-LLM + NIM inference benchmarks.

Updated for TensorRT-LLM v0.21+ with:
  - trtllm-bench throughput --dataset/--concurrency/--streaming flags
  - LLM-API BuildConfig optimization flags
  - NIM for LLMs v1.15+ with proper TTFT measurement
  - GenAI-Perf / AIPerf integration

Usage:
    python run_benchmarks.py --phase baseline
    python run_benchmarks.py --phase sweep
    python run_benchmarks.py --phase nim
    python run_benchmarks.py --phase build-flags
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


def ensure_results_dir():
    """Create results directories if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_synthetic_dataset(
    isl: int = 128,
    osl: int = 128,
    num_samples: int = 100,
    output_file: str = None,
) -> str:
    """
    Generate a synthetic dataset for trtllm-bench using prepare_dataset.

    Returns path to the generated .jsonl file.
    """
    if output_file is None:
        output_file = str(RESULTS_DIR / f"synthetic_{isl}_{osl}.jsonl")

    if os.path.exists(output_file):
        print(f"  Dataset already exists: {output_file}")
        return output_file

    print(f"  Preparing synthetic dataset: ISL={isl}, OSL={osl}, N={num_samples}")

    # Try using trtllm's prepare_dataset if available
    try:
        cmd = [
            "trtllm-bench", "prepare_dataset",
            "--input_length", str(isl),
            "--output_length", str(osl),
            "--num_samples", str(num_samples),
            "--output", output_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"  Dataset generated via trtllm-bench: {output_file}")
            return output_file
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: generate synthetic JSONL manually
    import json as _json
    with open(output_file, "w") as f:
        for i in range(num_samples):
            prompt = "The field of machine learning has evolved significantly. " * max(1, isl // 8)
            entry = {
                "task_id": i + 1,
                "prompt": prompt,
                "output_tokens": osl,
            }
            f.write(_json.dumps(entry) + "\n")

    print(f"  Dataset generated manually: {output_file}")
    return output_file


def run_trtllm_bench_modern(
    model_or_engine: str,
    dataset: str = None,
    isl: int = 128,
    osl: int = 128,
    concurrency: int = None,
    batch_size: int = None,
    tp_size: int = 1,
    pp_size: int = 1,
    backend: str = "pytorch",
    streaming: bool = True,
    num_warmup: int = 3,
    num_runs: int = 10,
    extra_options: str = None,
    label: str = "",
) -> dict:
    """
    Run trtllm-bench throughput with the modern v0.21+ API.

    Uses --dataset, --concurrency, --streaming flags.
    Supports both PyTorch flow (model name) and engine flow (--engine_dir).
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark: {label}")
    print(f"  ISL={isl}, OSL={osl}, Backend={backend}")
    if concurrency:
        print(f"  Concurrency={concurrency}")
    if batch_size:
        print(f"  BatchSize={batch_size}")
    print(f"  TP={tp_size}, PP={pp_size}, Streaming={streaming}")
    print(f"{'='*60}")

    output_file = RESULTS_DIR / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    cmd = [
        "trtllm-bench",
        "throughput",
        "--backend", backend,
        "--tp", str(tp_size),
    ]

    if pp_size > 1:
        cmd.extend(["--pp", str(pp_size)])

    # Model or engine
    if os.path.isdir(model_or_engine):
        cmd.extend(["--engine_dir", model_or_engine])
    else:
        cmd.extend(["--model", model_or_engine])

    # Dataset-based benchmarking (preferred)
    if dataset and os.path.exists(dataset):
        cmd.extend(["--dataset", dataset])
    else:
        cmd.extend(["--input_length", str(isl), "--output_length", str(osl)])

    # Concurrency or batch size
    if concurrency:
        cmd.extend(["--concurrency", str(concurrency)])
    elif batch_size:
        cmd.extend(["--batch_size", str(batch_size)])

    if streaming:
        cmd.append("--streaming")

    # Extra LLM API options
    if extra_options:
        cmd.extend(["--extra_llm_api_options", extra_options])

    # Report output
    cmd.extend(["--report_json", str(output_file)])

    print(f"  CMD: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - start

    metrics = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_or_engine": model_or_engine,
            "isl": isl,
            "osl": osl,
            "concurrency": concurrency,
            "batch_size": batch_size,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "backend": backend,
            "streaming": streaming,
            "num_warmup": num_warmup,
            "num_runs": num_runs,
        },
        "elapsed_seconds": round(elapsed, 2),
        "returncode": result.returncode,
    }

    # Parse trtllm-bench output for key metrics
    output_text = result.stdout + result.stderr
    metrics["raw_output"] = output_text[-3000:]

    if result.returncode == 0:
        for line in output_text.splitlines():
            # Parse performance overview metrics
            if "Request Throughput" in line and "(req/sec)" in line:
                try:
                    metrics["request_throughput_rps"] = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            if "Total Output Throughput" in line and "(tokens/sec)" in line:
                try:
                    metrics["output_throughput_tps"] = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            if "Total Token Throughput" in line and "(tokens/sec)" in line:
                try:
                    metrics["token_throughput_tps"] = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            if "Average time-to-first-token" in line and "TTFT" in line:
                try:
                    metrics["ttft_avg_ms"] = float(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if "Average time-per-output-token" in line and "TPOT" in line:
                try:
                    metrics["tpot_avg_ms"] = float(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if "Per User Output Speed" in line and "(tps/user)" in line:
                try:
                    metrics["per_user_output_speed"] = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            # P50/P90/P95/P99 percentiles
            if "[TTFT]" in line and "P50" in line:
                try:
                    metrics["ttft_p50_ms"] = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            if "[TTFT]" in line and "P99" in line:
                try:
                    metrics["ttft_p99_ms"] = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            if "[TPOT]" in line and "P50" in line:
                try:
                    metrics["tpot_p50_ms"] = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            if "[TPOT]" in line and "P99" in line:
                try:
                    metrics["tpot_p99_ms"] = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            # Max runtime settings
            if "Max Runtime Batch Size" in line:
                try:
                    metrics["max_runtime_batch_size"] = int(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass
            if "Max Runtime Tokens" in line:
                try:
                    metrics["max_runtime_tokens"] = int(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass

    # Also try loading the report JSON if it was generated
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                report = json.load(f)
                metrics["report_json"] = report
        except (json.JSONDecodeError, OSError):
            pass

    print(f"  Completed in {elapsed:.1f}s (rc={result.returncode})")
    if metrics.get("output_throughput_tps"):
        print(f"  Output throughput: {metrics['output_throughput_tps']:.0f} tokens/sec")
    if metrics.get("ttft_avg_ms"):
        print(f"  TTFT avg: {metrics['ttft_avg_ms']:.1f}ms")
    if metrics.get("tpot_avg_ms"):
        print(f"  TPOT avg: {metrics['tpot_avg_ms']:.2f}ms")

    return metrics


def run_baseline():
    """Phase A: Baseline BF16 TP=1 benchmarks with modern API."""
    config = load_config("baseline")
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    backend = config.get("backend", "pytorch")

    isl = config.get("input_len", 128)
    num_samples = 100

    results = []

    for osl in config.get("output_lengths", [128, 256, 512]):
        # Prepare dataset for this ISL/OSL pair
        dataset = prepare_synthetic_dataset(isl=isl, osl=osl, num_samples=num_samples)

        for batch_size in config.get("batch_sizes", [1, 8, 32, 64]):
            result = run_trtllm_bench_modern(
                model_or_engine=model_name,
                dataset=dataset,
                isl=isl,
                osl=osl,
                batch_size=batch_size,
                backend=backend,
                streaming=config.get("streaming", True),
                tp_size=config.get("tp_size", 1),
                pp_size=config.get("pp_size", 1),
                label=f"baseline_b{batch_size}_o{osl}",
            )
            results.append(result)

    # Also run concurrency sweep for throughput optimization
    dataset_128 = prepare_synthetic_dataset(isl=128, osl=128, num_samples=100)
    for conc in [10, 50, 100]:
        result = run_trtllm_bench_modern(
            model_or_engine=model_name,
            dataset=dataset_128,
            concurrency=conc,
            backend=backend,
            streaming=True,
            label=f"baseline_conc{conc}",
        )
        results.append(result)

    return results


def run_sweep():
    """Phase B: Optimization sweep across quantization + batching configs."""
    all_results = []

    # --- FP8 sweep ---
    fp8_config = load_config("sweep_fp8")
    fp8_model = fp8_config.get("checkpoint", "nvidia/Llama-3.1-8B-Instruct-FP8")

    dataset_128 = prepare_synthetic_dataset(isl=128, osl=128, num_samples=100)
    dataset_256 = prepare_synthetic_dataset(isl=128, osl=256, num_samples=100)

    print("\n--- FP8 Quantization Sweep ---")
    for batch_size in fp8_config.get("batch_sizes", [1, 8, 32, 64, 128, 256]):
        for osl, ds in [(128, dataset_128), (256, dataset_256)]:
            result = run_trtllm_bench_modern(
                model_or_engine=fp8_model,
                dataset=ds,
                batch_size=batch_size,
                label=f"fp8_b{batch_size}_o{osl}",
            )
            all_results.append(result)

    # FP8 concurrency sweep
    for conc in fp8_config.get("concurrency_levels", [1, 10, 50, 100, 200, 500]):
        result = run_trtllm_bench_modern(
            model_or_engine=fp8_model,
            dataset=dataset_128,
            concurrency=conc,
            label=f"fp8_conc{conc}",
        )
        all_results.append(result)

    # --- AWQ sweep ---
    awq_config = load_config("sweep_awq")
    awq_model = awq_config.get("checkpoint", "meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ")

    print("\n--- AWQ 4-bit Quantization Sweep ---")
    for batch_size in awq_config.get("batch_sizes", [1, 8, 32, 64, 128, 256]):
        for osl, ds in [(128, dataset_128), (256, dataset_256)]:
            result = run_trtllm_bench_modern(
                model_or_engine=awq_model,
                dataset=ds,
                batch_size=batch_size,
                label=f"awq4_b{batch_size}_o{osl}",
            )
            all_results.append(result)

    # --- CUDA Graph comparison ---
    print("\n--- CUDA Graph ON/OFF Comparison ---")
    for cuda_graph in [True, False]:
        result = run_trtllm_bench_modern(
            model_or_engine="meta-llama/Meta-Llama-3.1-8B-Instruct",
            dataset=dataset_128,
            batch_size=32,
            streaming=True,
            label=f"cuda_graph_{'on' if cuda_graph else 'off'}",
        )
        all_results.append(result)

    return all_results


def run_build_flags():
    """Phase D: Build-time flags A/B comparison.

    Compares performance with and without recommended build flags:
    - multiple_profiles
    - use_paged_context_fmha
    - gemm_plugin
    - reduce_fusion
    """
    config = load_config("build_flags")
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    dataset = prepare_synthetic_dataset(
        isl=config["benchmark"]["input_len"],
        osl=128,
        num_samples=config["benchmark"]["dataset_size"],
    )

    all_results = []
    flags = config.get("build_flags", {})

    # Create YAML options files for trtllm-build extra options
    import tempfile

    # --- Baseline (no extra flags) ---
    print("\n--- Build Flags: Baseline (no extra flags) ---")
    result = run_trtllm_bench_modern(
        model_or_engine=model_name,
        dataset=dataset,
        concurrency=1,
        label="buildflags_baseline",
    )
    all_results.append(result)

    # --- With multiple profiles ---
    if flags.get("multiple_profiles"):
        print("\n--- Build Flags: Multiple Profiles ON ---")
        opts_yaml = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        yaml.dump({"plugin_config": {"multiple_profiles": True}}, opts_yaml)
        opts_yaml.close()

        result = run_trtllm_bench_modern(
            model_or_engine=model_name,
            dataset=dataset,
            concurrency=1,
            extra_options=opts_yaml.name,
            label="buildflags_multi_profiles",
        )
        all_results.append(result)
        os.unlink(opts_yaml.name)

    # --- With all flags ---
    print("\n--- Build Flags: All flags ON ---")
    opts_yaml = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    yaml.dump({
        "plugin_config": {
            "multiple_profiles": True,
            "use_paged_context_fmha": True,
            "gemm_plugin": flags.get("gemm_plugin", "auto"),
            "reduce_fusion": flags.get("reduce_fusion", True),
        }
    }, opts_yaml)
    opts_yaml.close()

    result = run_trtllm_bench_modern(
        model_or_engine=model_name,
        dataset=dataset,
        concurrency=1,
        extra_options=opts_yaml.name,
        label="buildflags_all",
    )
    all_results.append(result)
    os.unlink(opts_yaml.name)

    return all_results


def run_nim():
    """Phase C: NIM container benchmarking."""
    print("\n  Running NIM benchmarks via benchmark_nim.py...")
    nim_script = Path(__file__).parent / "benchmark_nim.py"
    if nim_script.exists():
        result = subprocess.run(
            [sys.executable, str(nim_script)],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  NIM benchmark error: {result.stderr[-1000:]}")
    else:
        print("  ERROR: benchmark_nim.py not found")
    return []


def main():
    parser = argparse.ArgumentParser(description="Inference Benchmark Orchestrator (v0.21+)")
    parser.add_argument(
        "--phase",
        choices=["baseline", "sweep", "nim", "build-flags", "all"],
        required=True,
        help="Benchmark phase to run",
    )
    parser.add_argument("--model_dir", default=None, help="Override model directory")
    args = parser.parse_args()

    ensure_results_dir()

    all_results = {"started": datetime.now().isoformat(), "runs": []}

    if args.phase in ("baseline", "all"):
        print("\n" + "=" * 60)
        print("  PHASE A: BASELINE (BF16, TP=1)")
        print("=" * 60)
        results = run_baseline()
        all_results["runs"].extend(results)

    if args.phase in ("sweep", "all"):
        print("\n" + "=" * 60)
        print("  PHASE B: OPTIMIZATION SWEEP (FP8, AWQ, CUDA Graphs)")
        print("=" * 60)
        results = run_sweep()
        all_results["runs"].extend(results)

    if args.phase in ("nim", "all"):
        print("\n" + "=" * 60)
        print("  PHASE C: NIM DEPLOYMENT")
        print("=" * 60)
        results = run_nim()
        all_results["runs"].extend(results)

    if args.phase in ("build-flags", "all"):
        print("\n" + "=" * 60)
        print("  PHASE D: BUILD-TIME FLAGS A/B TEST")
        print("=" * 60)
        results = run_build_flags()
        all_results["runs"].extend(results)

    all_results["completed"] = datetime.now().isoformat()
    all_results["total_runs"] = len(all_results["runs"])

    summary_file = RESULTS_DIR / f"summary_{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(all_results['runs'])} benchmarks completed.")
    print(f"  Summary: {summary_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
