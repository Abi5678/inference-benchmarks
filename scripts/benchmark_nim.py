#!/usr/bin/env python3
"""
NIM container benchmark script with streaming TTFT support.

Updated for NVIDIA NIM for LLMs v1.15+ which has fixed TTFT/E2E metrics.
Uses streaming API for accurate time-to-first-token measurement.

Requires: Docker, NVIDIA Container Toolkit, NGC API key.
"""

import json
import os
import subprocess
import sys
import time
import threading
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
    """Wait for NIM container to be ready via /v1/models endpoint."""
    health_url = f"{base_url}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                elapsed = int(time.time() - start)
                print(f"  NIM ready after {elapsed}s")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    print(f"  NIM failed to start within {timeout}s")
    return False


def measure_gpu_metrics(gpu_id: int = 0, interval: float = 1.0) -> dict:
    """Capture GPU metrics via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "power_watts": float(parts[0]),
            "temperature_c": float(parts[1]),
            "gpu_util_pct": float(parts[2]),
            "memory_used_mb": float(parts[3]),
            "memory_total_mb": float(parts[4]),
            "memory_util_pct": float(parts[5]),
        }
    except Exception:
        return {"error": "nvidia-smi unavailable"}


def run_streaming_request(base_url: str, prompt: str, max_tokens: int, timeout: int = 120) -> dict:
    """
    Send a streaming completion request and measure TTFT accurately.
    
    TTFT is measured from request send to first chunk of the response body.
    TPOT is computed from first token to last token.
    """
    start = time.perf_counter()
    first_chunk_time = None
    token_count = 0
    chunks = []

    try:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "meta/llama-3.1-8b-instruct",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
            stream=True,
            timeout=timeout,
        )

        if resp.status_code != 200:
            return {
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "ttft_ms": None,
                "total_ms": None,
                "tokens": 0,
                "tpot_ms": None,
                "inter_token_latencies_ms": [],
            }

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                text = data.get("choices", [{}])[0].get("text", "")
                if text:
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    chunks.append(text)
                    token_count += len(text.split())
            except json.JSONDecodeError:
                continue

        end = time.perf_counter()
        total_time_ms = (end - start) * 1000

        if first_chunk_time is None:
            return {
                "error": "no tokens generated",
                "ttft_ms": None,
                "total_ms": total_time_ms,
                "tokens": 0,
                "tpot_ms": None,
                "inter_token_latencies_ms": [],
            }

        ttft_ms = (first_chunk_time - start) * 1000
        generation_time_ms = (end - first_chunk_time) * 1000
        tpot_ms = generation_time_ms / max(token_count - 1, 1) if token_count > 1 else 0

        return {
            "ttft_ms": ttft_ms,
            "total_ms": total_time_ms,
            "tokens": token_count,
            "tpot_ms": tpot_ms,
            "generation_time_ms": generation_time_ms,
            "error": None,
        }

    except requests.Timeout:
        return {"error": "timeout", "ttft_ms": None, "total_ms": None, "tokens": 0, "tpot_ms": None}
    except Exception as e:
        return {"error": str(e), "ttft_ms": None, "total_ms": None, "tokens": 0, "tpot_ms": None}


def run_load_test(
    base_url: str,
    num_requests: int = 100,
    concurrency: int = 1,
    input_len: int = 128,
    output_len: int = 256,
    streaming: bool = True,
    label: str = "",
) -> dict:
    """
    Run concurrent load test against NIM endpoint.
    Measures TTFT, TPOT, throughput, and P50/P95/P99 latencies.
    """
    print(f"\n  Load test: {label} (reqs={num_requests}, conc={concurrency}, stream={streaming})")

    prompt = "The field of machine learning has evolved significantly. " * max(1, input_len // 8)

    latencies = []
    ttfts = []
    tpots = []
    tokens_generated = []
    errors = 0

    import concurrent.futures

    def single_request():
        if streaming:
            return run_streaming_request(base_url, prompt, output_len)
        else:
            # Non-streaming fallback
            start = time.perf_counter()
            try:
                resp = requests.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": "meta/llama-3.1-8b-instruct",
                        "prompt": prompt,
                        "max_tokens": output_len,
                        "temperature": 0.0,
                        "stream": False,
                    },
                    timeout=120,
                )
                end = time.perf_counter()
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("choices", [{}])[0].get("text", "")
                    tokens = len(text.split())
                    return {
                        "ttft_ms": (end - start) * 1000,
                        "total_ms": (end - start) * 1000,
                        "tokens": tokens,
                        "tpot_ms": ((end - start) * 1000) / max(tokens, 1),
                        "error": None,
                    }
                return {"error": f"HTTP {resp.status_code}", "ttft_ms": None, "total_ms": None, "tokens": 0, "tpot_ms": None}
            except Exception as e:
                return {"error": str(e), "ttft_ms": None, "total_ms": None, "tokens": 0, "tpot_ms": None}

    start_all = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(single_request) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result["error"]:
                errors += 1
            else:
                latencies.append(result["total_ms"])
                ttfts.append(result["ttft_ms"])
                tpots.append(result["tpot_ms"])
                tokens_generated.append(result["tokens"])

    total_time = time.perf_counter() - start_all
    total_tokens = sum(tokens_generated)

    def percentile(data, p):
        if not data:
            return None
        s = sorted(data)
        k = int(len(s) * p / 100)
        return s[min(k, len(s) - 1)]

    successful = num_requests - errors

    metrics = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_requests": num_requests,
            "concurrency": concurrency,
            "input_len": input_len,
            "output_len": output_len,
            "streaming": streaming,
        },
        "summary": {
            "total_requests": num_requests,
            "errors": errors,
            "successful": successful,
            "wall_time_s": round(total_time, 2),
            "throughput_tps": round(total_tokens / max(total_time, 0.001), 2),
            "requests_per_sec": round(successful / max(total_time, 0.001), 2),
            "ttft_p50_ms": round(percentile(ttfts, 50), 2),
            "ttft_p95_ms": round(percentile(ttfts, 95), 2),
            "ttft_p99_ms": round(percentile(ttfts, 99), 2),
            "ttft_mean_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "tpot_mean_ms": round(sum(tpots) / len(tpots), 2) if tpots else None,
            "tpot_p95_ms": round(percentile(tpots, 95), 2),
            "latency_p50_ms": round(percentile(latencies, 50), 2),
            "latency_p95_ms": round(percentile(latencies, 95), 2),
            "latency_p99_ms": round(percentile(latencies, 99), 2),
            "avg_tokens_per_request": round(total_tokens / max(successful, 1), 1),
            "error_rate_pct": round(errors / max(num_requests, 1) * 100, 2),
        },
    }

    # Save individual result
    output_file = RESULTS_DIR / f"nim_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Throughput: {metrics['summary']['throughput_tps']:.0f} tps")
    print(f"  TTFT P50: {metrics['summary']['ttft_p50_ms']:.1f}ms | P99: {metrics['summary']['ttft_p99_ms']:.1f}ms")
    print(f"  TPOT Mean: {metrics['summary']['tpot_mean_ms']:.2f}ms")
    print(f"  Errors: {errors}/{num_requests}")
    print(f"  Saved: {output_file}")

    return metrics


def main():
    config = load_config("nim_config")
    base_url = config.get("base_url", "http://localhost:8000")
    streaming = config.get("streaming", True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== NIM Benchmark Suite (v1.15+) ===")

    # Check NIM is running
    if not wait_for_nim(base_url, timeout=300):
        print("ERROR: NIM container is not responding.")
        print("Start with:")
        print("  docker run -d --gpus all -p 8000:8000 \\")
        print("    -e NGC_API_KEY=$NGC_API_KEY \\")
        print("    nvcr.io/nim/meta/llama-3.1-8b-instruct:latest")
        sys.exit(1)

    # Capture baseline GPU metrics
    gpu_before = measure_gpu_metrics()
    print(f"\n  GPU before: {gpu_before}")

    all_results = []

    # Sweep 1: Concurrency levels
    print("\n--- Concurrency Sweep ---")
    for concurrency in config.get("concurrency_levels", [1, 5, 10, 25, 50]):
        result = run_load_test(
            base_url=base_url,
            num_requests=config.get("num_requests", 100),
            concurrency=concurrency,
            input_len=config.get("input_len", 128),
            output_len=config.get("max_tokens", 256),
            streaming=streaming,
            label=f"conc_{concurrency}",
        )
        result["gpu_metrics"] = measure_gpu_metrics()
        all_results.append(result)

    # Sweep 2: Output lengths
    print("\n--- Output Length Sweep ---")
    for out_len in config.get("output_lengths", [64, 128, 256, 512]):
        result = run_load_test(
            base_url=base_url,
            num_requests=config.get("num_requests", 50),
            concurrency=10,
            input_len=128,
            output_len=out_len,
            streaming=streaming,
            label=f"outlen_{out_len}",
        )
        result["gpu_metrics"] = measure_gpu_metrics()
        all_results.append(result)

    # Sweep 3: Streaming vs Non-streaming comparison
    print("\n--- Streaming vs Non-Streaming ---")
    for is_stream in [True, False]:
        result = run_load_test(
            base_url=base_url,
            num_requests=50,
            concurrency=10,
            input_len=128,
            output_len=256,
            streaming=is_stream,
            label=f"{'streaming' if is_stream else 'non_streaming'}",
        )
        result["gpu_metrics"] = measure_gpu_metrics()
        all_results.append(result)

    # Save comprehensive summary
    gpu_after = measure_gpu_metrics()
    summary = {
        "started": datetime.now().isoformat(),
        "nim_config": config,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "runs": all_results,
        "total_runs": len(all_results),
        "completed": datetime.now().isoformat(),
    }

    summary_file = RESULTS_DIR / f"nim_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done! {len(all_results)} NIM benchmarks completed.")
    print(f"  Summary: {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
