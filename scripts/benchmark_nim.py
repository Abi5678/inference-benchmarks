#!/usr/bin/env python3
"""
NIM container benchmark script.

Deploys NVIDIA NIM, runs load tests, and captures metrics.
Requires: Docker, NVIDIA Container Toolkit, NGC API key.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
import requests

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_config(name: str) -> dict:
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def wait_for_nim(base_url: str, timeout: int = 600) -> bool:
    """Wait for NIM container to be ready."""
    health_url = f"{base_url}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                print(f"  NIM ready after {int(time.time() - start)}s")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    print(f"  NIM failed to start within {timeout}s")
    return False


def run_load_test(
    base_url: str,
    num_requests: int = 100,
    concurrency: int = 1,
    input_len: int = 128,
    output_len: int = 256,
    label: str = "",
) -> dict:
    """
    Run synthetic load test against NIM endpoint.
    Measures TTFT, TPOT, throughput, and P50/P99 latencies.
    """
    print(f"\n  Load test: {label} (reqs={num_requests}, conc={concurrency})")

    # Build prompt of approximate target length
    prompt = "The field of machine learning has evolved significantly. " * (input_len // 8)

    latencies = []
    ttfts = []
    tokens_generated = []
    errors = 0

    import concurrent.futures

    def single_request():
        start = time.time()
        try:
            resp = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": "meta/llama-3.1-8b-instruct",
                    "prompt": prompt,
                    "max_tokens": output_len,
                    "temperature": 0.0,
                },
                timeout=120,
            )
            first_token_time = time.time()  # simplified (streaming would be more accurate)
            data = resp.json()
            total_tokens = len(data.get("choices", [{}])[0].get("text", "").split())
            total_time = time.time() - start

            return {
                "ttft_ms": (first_token_time - start) * 1000,
                "total_ms": total_time * 1000,
                "tokens": total_tokens,
                "tpot_ms": ((total_time * 1000) / max(total_tokens, 1)),
                "error": None,
            }
        except Exception as e:
            return {"error": str(e), "ttft_ms": None, "total_ms": None, "tokens": 0, "tpot_ms": None}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(single_request) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result["error"]:
                errors += 1
            else:
                latencies.append(result["total_ms"])
                ttfts.append(result["ttft_ms"])
                tokens_generated.append(result["tokens"])

    # Compute percentiles
    latencies.sort()
    ttfts.sort()

    def percentile(data, p):
        if not data:
            return None
        k = int(len(data) * p / 100)
        return data[min(k, len(data) - 1)]

    total_tokens = sum(tokens_generated)
    total_time = sum(latencies) / 1000 if latencies else 0

    metrics = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_requests": num_requests,
            "concurrency": concurrency,
            "input_len": input_len,
            "output_len": output_len,
        },
        "summary": {
            "total_requests": num_requests,
            "errors": errors,
            "successful": num_requests - errors,
            "throughput_tps": round(total_tokens / max(total_time, 0.001), 2),
            "ttft_p50_ms": percentile(ttfts, 50),
            "ttft_p95_ms": percentile(ttfts, 95),
            "ttft_p99_ms": percentile(ttfts, 99),
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "latency_p99_ms": percentile(latencies, 99),
            "avg_tokens_per_request": round(total_tokens / max(len(tokens_generated), 1), 1),
        },
    }

    output_file = RESULTS_DIR / f"nim_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to {output_file}")

    return metrics


def get_gpu_metrics() -> dict:
    """Capture GPU metrics via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        parts = result.stdout.strip().split(", ")
        return {
            "power_watts": float(parts[0]),
            "temperature_c": float(parts[1]),
            "gpu_util_pct": float(parts[2]),
            "memory_used_mb": float(parts[3]),
            "memory_total_mb": float(parts[4]),
        }
    except Exception:
        return {"error": "nvidia-smi unavailable"}


def main():
    config = load_config("nim_config")
    base_url = config.get("base_url", "http://localhost:8000")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== NIM Benchmark Suite ===")

    # Check NIM is running
    if not wait_for_nim(base_url, timeout=300):
        print("ERROR: NIM container is not responding. Is it running?")
        print("Start with: docker run -d --gpus all -p 8000:8000 nvcr.io/nim/meta/llama-3.1-8b-instruct:latest")
        sys.exit(1)

    all_results = []

    # Sweep concurrency levels
    for concurrency in config.get("concurrency_levels", [1, 5, 10, 25]):
        result = run_load_test(
            base_url=base_url,
            num_requests=config.get("num_requests", 50),
            concurrency=concurrency,
            input_len=config.get("input_len", 128),
            output_len=config.get("output_len", 256),
            label=f"conc_{concurrency}",
        )
        result["gpu_metrics"] = get_gpu_metrics()
        all_results.append(result)

    # Sweep output lengths
    for out_len in config.get("output_lengths", [64, 128, 256, 512]):
        result = run_load_test(
            base_url=base_url,
            num_requests=config.get("num_requests", 50),
            concurrency=5,
            input_len=128,
            output_len=out_len,
            label=f"outlen_{out_len}",
        )
        result["gpu_metrics"] = get_gpu_metrics()
        all_results.append(result)

    summary = {
        "started": datetime.now().isoformat(),
        "nim_config": config,
        "runs": all_results,
        "total_runs": len(all_results),
    }

    summary_file = RESULTS_DIR / f"nim_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! {len(all_results)} NIM benchmarks completed.")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
