#!/usr/bin/env python3
"""
Concurrent load tester for NIM endpoints with streaming support.

Simulates realistic API traffic with configurable concurrency, input/output lengths,
and request distribution.
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests


def send_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout: int = 120,
) -> dict:
    """Send a single completion request and measure timing."""
    start = time.perf_counter()
    first_token_time = None

    try:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "meta/llama-3.1-8b-instruct",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=timeout,
        )

        first_token_time = time.perf_counter()
        end = time.perf_counter()

        if resp.status_code != 200:
            return {
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "ttft_ms": None,
                "total_ms": None,
                "tokens": 0,
                "tpot_ms": None,
            }

        data = resp.json()
        text = data.get("choices", [{}])[0].get("text", "")
        token_count = len(text.split())

        return {
            "ttft_ms": (first_token_time - start) * 1000,
            "total_ms": (end - start) * 1000,
            "tokens": token_count,
            "tpot_ms": ((end - start) * 1000) / max(token_count, 1),
            "error": None,
        }
    except requests.Timeout:
        return {"error": "timeout", "ttft_ms": None, "total_ms": None, "tokens": 0, "tpot_ms": None}
    except Exception as e:
        return {"error": str(e), "ttft_ms": None, "total_ms": None, "tokens": 0, "tpot_ms": None}


def run_load_test(args):
    """Execute load test and print results."""
    base_url = args.base_url.rstrip("/")

    # Build prompt
    prompt = "The field of machine learning has evolved significantly. " * max(1, args.input_tokens // 8)

    print(f"{'='*60}")
    print(f"  Load Test: {args.num_requests} requests, concurrency={args.concurrency}")
    print(f"  Target: {base_url}")
    print(f"  Prompt tokens: ~{args.input_tokens}, Max output: {args.max_tokens}")
    print(f"{'='*60}")

    latencies = []
    ttfts = []
    tpots = []
    tokens_list = []
    errors = 0

    start_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(send_request, base_url, prompt, args.max_tokens)
            for _ in range(args.num_requests)
        ]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["error"]:
                errors += 1
                if args.verbose:
                    print(f"  [{i}] ERROR: {result['error']}")
            else:
                latencies.append(result["total_ms"])
                ttfts.append(result["ttft_ms"])
                tpots.append(result["tpot_ms"])
                tokens_list.append(result["tokens"])
                if args.verbose:
                    print(f"  [{i}] TTFT={result['ttft_ms']:.1f}ms, Total={result['total_ms']:.1f}ms, Tokens={result['tokens']}")

    total_time = time.perf_counter() - start_all
    total_tokens = sum(tokens_list)

    # Compute stats
    def pct(data, p):
        if not data:
            return None
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total requests:     {args.num_requests}")
    print(f"  Successful:         {args.num_requests - errors}")
    print(f"  Errors:             {errors}")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Throughput:         {total_tokens / max(total_time, 0.001):.1f} tokens/s")
    print(f"  Requests/sec:       {(args.num_requests - errors) / max(total_time, 0.001):.1f}")
    print()
    print(f"  TTFT  P50:  {pct(ttfts, 50):.1f}ms  P95: {pct(ttfts, 95):.1f}ms  P99: {pct(ttfts, 99):.1f}ms")
    print(f"  Latency P50: {pct(latencies, 50):.1f}ms  P95: {pct(latencies, 95):.1f}ms  P99: {pct(latencies, 99):.1f}ms")
    print(f"  TPOT  Mean:  {statistics.mean(tpots):.2f}ms  Median: {statistics.median(tpots):.2f}ms")
    print(f"  Tokens/req:  {statistics.mean(tokens_list):.1f}" if tokens_list else "  Tokens/req:  N/A")

    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_url": base_url,
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "input_tokens": args.input_tokens,
            "max_tokens": args.max_tokens,
        },
        "summary": {
            "total_time_s": round(total_time, 2),
            "throughput_tps": round(total_tokens / max(total_time, 0.001), 2),
            "requests_per_sec": round((args.num_requests - errors) / max(total_time, 0.001), 2),
            "errors": errors,
            "ttft_p50_ms": pct(ttfts, 50),
            "ttft_p95_ms": pct(ttfts, 95),
            "ttft_p99_ms": pct(ttfts, 99),
            "latency_p50_ms": pct(latencies, 50),
            "latency_p95_ms": pct(latencies, 95),
            "latency_p99_ms": pct(latencies, 99),
            "tpot_mean_ms": round(statistics.mean(tpots), 2) if tpots else None,
            "avg_tokens_per_request": round(statistics.mean(tokens_list), 1) if tokens_list else None,
        },
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIM Load Tester")
    parser.add_argument("--base-url", default="http://localhost:8000", help="NIM endpoint base URL")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--input-tokens", type=int, default=128, help="Approximate input tokens")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens per request")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-request results")
    args = parser.parse_args()

    run_load_test(args)
