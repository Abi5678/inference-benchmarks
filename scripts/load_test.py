#!/usr/bin/env python3
"""
Advanced load tester with streaming support and percentile metrics.

Updated for TensorRT-LLM v0.21+ and NIM v1.15+:
  - Streaming TTFT measurement via time-to-first-chunk
  - P50/P90/P95/P99 latency percentile tracking
  - Token counting and throughput calculation
  - GPU memory monitoring during load
  - WARMUP phase before measurement (critical for CUDA Graph activation)

Usage:
    python load_test.py --url http://localhost:8000 --concurrency 50 --requests 500

    # Against NIM container:
    python load_test.py --url http://localhost:8000 --model meta/llama-3.1-8b-instruct \
        --concurrency 50 --requests 500 --streaming

    # Against trtllm-serve:
    python load_test.py --url http://localhost:8000 --concurrency 50 --requests 500
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests
import yaml

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"


@dataclass
class RequestResult:
    success: bool
    ttft_ms: float = 0.0          # Time to first token
    tpot_ms: float = 0.0          # Time per output token
    total_ms: float = 0.0         # End-to-end latency
    tokens: int = 0               # Output token count
    input_tokens: int = 0         # Estimated input tokens
    error: str = ""


@dataclass
class LoadTestResult:
    label: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict = field(default_factory=dict)
    total_requests: int = 0
    errors: int = 0
    wall_time_s: float = 0.0
    total_tokens: int = 0
    ttfts: list = field(default_factory=list)
    tpots: list = field(default_factory=list)
    latencies: list = field(default_factory=list)
    tokens_per_req: list = field(default_factory=list)
    gpu_snapshots: list = field(default_factory=list)

    @property
    def throughput_tps(self):
        return self.total_tokens / max(self.wall_time_s, 0.001)

    @property
    def requests_per_sec(self):
        successful = self.total_requests - self.errors
        return successful / max(self.wall_time_s, 0.001)

    def percentile(self, data, p):
        if not data:
            return None
        s = sorted(data)
        k = int(len(s) * p / 100)
        return s[min(k, len(s) - 1)]


def generate_prompt(target_tokens: int = 128) -> str:
    """Generate a prompt of approximately target_tokens in length."""
    base = "The field of machine learning has evolved significantly over the past decade. "
    repeats = max(1, target_tokens // 8)
    return base * repeats


def make_streaming_request(
    url: str,
    prompt: str,
    model: str,
    max_tokens: int,
    timeout: int = 120,
) -> RequestResult:
    """Single streaming request with precise TTFT measurement."""
    start = time.perf_counter()
    first_chunk_time = None
    token_count = 0
    text_received = ""

    try:
        resp = requests.post(
            f"{url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
            stream=True,
            timeout=timeout,
        )

        if resp.status_code != 200:
            return RequestResult(success=False, error=f"HTTP {resp.status_code}")

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
                    text_received += text
                    token_count += len(text.split())
            except json.JSONDecodeError:
                continue

        end = time.perf_counter()
        total_ms = (end - start) * 1000

        if first_chunk_time is None:
            return RequestResult(success=False, total_ms=total_ms, error="no tokens")

        ttft_ms = (first_chunk_time - start) * 1000
        gen_time_ms = (end - first_chunk_time) * 1000
        tpot_ms = gen_time_ms / max(token_count - 1, 1) if token_count > 1 else 0

        return RequestResult(
            success=True,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_ms=total_ms,
            tokens=token_count,
            input_tokens=len(prompt.split()),
        )

    except requests.Timeout:
        return RequestResult(success=False, error="timeout")
    except Exception as e:
        return RequestResult(success=False, error=str(e))


def make_nonstreaming_request(
    url: str,
    prompt: str,
    model: str,
    max_tokens: int,
    timeout: int = 120,
) -> RequestResult:
    """Single non-streaming request."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=timeout,
        )
        end = time.perf_counter()
        total_ms = (end - start) * 1000

        if resp.status_code != 200:
            return RequestResult(success=False, total_ms=total_ms, error=f"HTTP {resp.status_code}")

        data = resp.json()
        text = data.get("choices", [{}])[0].get("text", "")
        tokens = len(text.split())

        return RequestResult(
            success=True,
            total_ms=total_ms,
            ttft_ms=total_ms,  # For non-streaming, TTFT ≈ total latency
            tokens=tokens,
            input_tokens=len(prompt.split()),
        )
    except Exception as e:
        return RequestResult(success=False, error=str(e))


def monitor_gpu(interval: float, stop_event: threading.Event, results: list):
    """Background thread: periodically capture GPU metrics."""
    while not stop_event.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                results.append({
                    "time_s": time.time(),
                    "power_w": float(parts[0]),
                    "mem_used_mb": float(parts[1]),
                    "mem_total_mb": float(parts[2]),
                    "gpu_util_pct": float(parts[3]),
                })
        except Exception:
            pass
        stop_event.wait(interval)


def run_load_test(
    url: str = "http://localhost:8000",
    model: str = "meta/llama-3.1-8b-instruct",
    num_requests: int = 500,
    concurrency: int = 50,
    max_tokens: int = 256,
    input_tokens: int = 128,
    streaming: bool = True,
    warmup_requests: int = 10,
    warmup_concurrency: int = 1,
    label: str = "",
) -> LoadTestResult:
    """Run a complete load test with warmup, measurement, and percentile analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not label:
        label = f"c{concurrency}_o{max_tokens}_{'stream' if streaming else 'batch'}"

    prompt = generate_prompt(input_tokens)
    print(f"\n{'='*60}")
    print(f"  Load Test: {label}")
    print(f"  URL: {url}")
    print(f"  Requests: {num_requests}, Concurrency: {concurrency}")
    print(f"  Max tokens: {max_tokens}, Input tokens: ~{input_tokens}")
    print(f"  Streaming: {streaming}")
    print(f"{'='*60}")

    result = LoadTestResult(label=label, config={
        "url": url,
        "model": model,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "input_tokens": input_tokens,
        "streaming": streaming,
    })

    # --- Phase 1: Warmup (crucial for CUDA Graph activation) ---
    if warmup_requests > 0:
        print(f"\n  Phase 1: Warmup ({warmup_requests} requests, concurrency={warmup_concurrency})...")
        request_fn = lambda: make_streaming_request(url, prompt, model, max_tokens) if streaming else \
                          lambda: make_nonstreaming_request(url, prompt, model, max_tokens)

        with ThreadPoolExecutor(max_workers=warmup_concurrency) as pool:
            list(pool.map(lambda _: request_fn(), range(warmup_requests)))

        print(f"  Warmup complete.")

    # --- Phase 2: Measurement ---
    print(f"\n  Phase 2: Measurement ({num_requests} requests, concurrency={concurrency})...")

    gpu_stop = threading.Event()
    gpu_snapshots = []
    gpu_thread = threading.Thread(target=monitor_gpu, args=(1.0, gpu_stop, gpu_snapshots), daemon=True)
    gpu_thread.start()

    start_all = time.time()

    request_fn = make_streaming_request if streaming else make_nonstreaming_request

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_fn, url, prompt, model, max_tokens) for _ in range(num_requests)]
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            result.total_requests += 1
            if r.success:
                result.ttfts.append(r.ttft_ms)
                result.tpots.append(r.tpot_ms)
                result.latencies.append(r.total_ms)
                result.tokens_per_req.append(r.tokens)
                result.total_tokens += r.tokens
            else:
                result.errors += 1

            # Progress
            if (i + 1) % max(1, num_requests // 10) == 0:
                print(f"    Progress: {i+1}/{num_requests} ({(i+1)/num_requests*100:.0f}%)")

    wall_time = time.time() - start_all
    result.wall_time_s = wall_time

    gpu_stop.set()
    gpu_thread.join(timeout=3)
    result.gpu_snapshots = gpu_snapshots

    # --- Phase 3: Report ---
    p = result.percentile
    print(f"\n  {'='*40}")
    print(f"  RESULTS: {label}")
    print(f"  {'='*40}")
    print(f"  Completed: {result.total_requests - result.errors}/{result.total_requests} requests")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Throughput: {result.throughput_tps:.0f} tokens/s")
    print(f"  Requests/s: {result.requests_per_sec:.1f}")
    print(f"  Avg tokens/req: {statistics.mean(result.tokens_per_req):.0f}" if result.tokens_per_req else "  Avg tokens/req: N/A")
    print(f"\n  TTFT (ms):")
    print(f"    Mean: {statistics.mean(result.ttfts):.1f}" if result.ttfts else "    N/A")
    print(f"    P50:  {p(result.ttfts, 50):.1f}" if result.ttfts else "    N/A")
    print(f"    P90:  {p(result.ttfts, 90):.1f}" if result.ttfts else "    N/A")
    print(f"    P95:  {p(result.ttfts, 95):.1f}" if result.ttfts else "    N/A")
    print(f"    P99:  {p(result.ttfts, 99):.1f}" if result.ttfts else "    N/A")
    print(f"\n  TPOT (ms):")
    print(f"    Mean: {statistics.mean(result.tpots):.2f}" if result.tpots else "    N/A")
    print(f"    P95:  {p(result.tpots, 95):.2f}" if result.tpots else "    N/A")
    print(f"    P99:  {p(result.tpots, 99):.2f}" if result.tpots else "    N/A")
    print(f"\n  E2E Latency (ms):")
    print(f"    P50:  {p(result.latencies, 50):.0f}" if result.latencies else "    N/A")
    print(f"    P99:  {p(result.latencies, 99):.0f}" if result.latencies else "    N/A")
    print(f"    Error rate: {result.errors/max(result.total_requests,1)*100:.1f}%")

    if gpu_snapshots:
        avg_power = statistics.mean([s["power_w"] for s in gpu_snapshots])
        peak_mem = max(s["mem_used_mb"] for s in gpu_snapshots)
        avg_util = statistics.mean([s["gpu_util_pct"] for s in gpu_snapshots])
        print(f"\n  GPU:")
        print(f"    Avg power: {avg_power:.0f}W")
        print(f"    Peak VRAM: {peak_mem:.0f}MB")
        print(f"    Avg util: {avg_util:.0f}%")

    # Save results
    output = {
        "label": label,
        "timestamp": result.timestamp,
        "config": result.config,
        "summary": {
            "total_requests": result.total_requests,
            "errors": result.errors,
            "successful": result.total_requests - result.errors,
            "wall_time_s": round(wall_time, 2),
            "throughput_tps": round(result.throughput_tps, 2),
            "requests_per_sec": round(result.requests_per_sec, 2),
            "avg_tokens_per_request": round(statistics.mean(result.tokens_per_req), 1) if result.tokens_per_req else None,
            "error_rate_pct": round(result.errors / max(result.total_requests, 1) * 100, 2),
            "ttft_mean_ms": round(statistics.mean(result.ttfts), 2) if result.ttfts else None,
            "ttft_p50_ms": round(p(result.ttfts, 50), 2) if result.ttfts else None,
            "ttft_p90_ms": round(p(result.ttfts, 90), 2) if result.ttfts else None,
            "ttft_p95_ms": round(p(result.ttfts, 95), 2) if result.ttfts else None,
            "ttft_p99_ms": round(p(result.ttfts, 99), 2) if result.ttfts else None,
            "tpot_mean_ms": round(statistics.mean(result.tpots), 3) if result.tpots else None,
            "tpot_p50_ms": round(p(result.tpots, 50), 3) if result.tpots else None,
            "tpot_p95_ms": round(p(result.tpots, 95), 3) if result.tpots else None,
            "tpot_p99_ms": round(p(result.tpots, 99), 3) if result.tpots else None,
            "latency_p50_ms": round(p(result.latencies, 50), 2) if result.latencies else None,
            "latency_p99_ms": round(p(result.latencies, 99), 2) if result.latencies else None,
        },
        "gpu": {
            "avg_power_w": round(statistics.mean([s["power_w"] for s in gpu_snapshots]), 1) if gpu_snapshots else None,
            "peak_memory_mb": max((s["mem_used_mb"] for s in gpu_snapshots), default=None),
            "avg_gpu_util_pct": round(statistics.mean([s["gpu_util_pct"] for s in gpu_snapshots]), 1) if gpu_snapshots else None,
            "snapshots": gpu_snapshots[-10:],  # Keep last 10 for debugging
        },
    }

    output_file = RESULTS_DIR / f"loadtest_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Advanced LLM Load Tester (v0.21+)")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--model", default="meta/llama-3.1-8b-instruct", help="Model name for API")
    parser.add_argument("--requests", type=int, default=500, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    parser.add_argument("--input-tokens", type=int, default=128, help="Approx input token count")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming API")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup requests")
    parser.add_argument("--label", default="", help="Custom label for results")
    args = parser.parse_args()

    run_load_test(
        url=args.url,
        model=args.model,
        num_requests=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        input_tokens=args.input_tokens,
        streaming=args.streaming,
        warmup_requests=args.warmup,
        label=args.label,
    )


if __name__ == "__main__":
    main()
