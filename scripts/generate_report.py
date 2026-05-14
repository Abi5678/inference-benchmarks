#!/usr/bin/env python3
"""
Generate analysis report and charts from benchmark results.

Reads results/raw/*.json, produces charts/ visualizations and a summary report.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
CHARTS_DIR = RESULTS_DIR / "charts"
REPORT_FILE = RESULTS_DIR / "benchmark_report.md"


def load_all_results() -> list[dict]:
    """Load all JSON result files from raw/."""
    results = []
    for f in RAW_DIR.glob("*.json"):
        with open(f) as fh:
            try:
                data = json.load(fh)
                results.append(data)
            except json.JSONDecodeError:
                pass
    return results


def extract_metrics(results: list[dict]) -> dict:
    """Extract structured metrics from benchmark results."""
    baseline = []
    fp8 = []
    awq = []
    nim = []

    for r in results:
        label = r.get("label", "")
        summary = r.get("summary", {})

        entry = {
            "label": label,
            "batch_size": r.get("config", {}).get("batch_size", 1),
            "throughput_tps": r.get("throughput_tps") or summary.get("throughput_tps"),
            "ttft_ms": r.get("ttft_ms") or summary.get("ttft_p50_ms"),
            "tpot_ms": r.get("tpot_ms"),
            "gpu_memory_mb": r.get("gpu_memory_mb"),
            "cuda_graph": r.get("config", {}).get("cuda_graph", True),
        }

        if "baseline" in label:
            baseline.append(entry)
        elif "fp8" in label:
            fp8.append(entry)
        elif "awq" in label:
            awq.append(entry)
        elif "nim" in label:
            nim.append(entry)

    return {"baseline": baseline, "fp8": fp8, "awq": awq, "nim": nim}


def plot_throughput_comparison(metrics: dict):
    """Bar chart: throughput across configs at different batch sizes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    groups = {}
    for key, data in metrics.items():
        if key == "nim":
            continue
        for entry in data:
            bs = entry["batch_size"]
            groups.setdefault(bs, {})[key] = entry.get("throughput_tps")

    batch_sizes = sorted(groups.keys())
    x = np.arange(len(batch_sizes))
    width = 0.25
    colors = {"baseline": "#4285F4", "fp8": "#EA4335", "awq": "#34A853"}

    for i, (key, color) in enumerate(colors.items()):
        values = [groups[bs].get(key, 0) or 0 for bs in batch_sizes]
        bars = ax.bar(x + i * width, values, width, label=key.upper(), color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f"{val:.0f}", ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Inference Throughput: BF16 vs FP8 vs AWQ (Llama 3.1 8B)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = CHARTS_DIR / "throughput_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_latency_comparison(metrics: dict):
    """Line chart: TTFT and TPOT across batch sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for key, data in metrics.items():
        if key == "nim":
            continue
        data = sorted(data, key=lambda x: x["batch_size"])
        bs = [d["batch_size"] for d in data]
        ttft = [d.get("ttft_ms", 0) or 0 for d in data]
        tpot = [d.get("tpot_ms", 0) or 0 for d in data]

        color = {"baseline": "#4285F4", "fp8": "#EA4335", "awq": "#34A853"}.get(key, "gray")
        ax1.plot(bs, ttft, "o-", label=key.upper(), color=color, linewidth=2)
        ax2.plot(bs, tpot, "s-", label=key.upper(), color=color, linewidth=2)

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Time to First Token")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("TPOT (ms)")
    ax2.set_title("Time per Output Token")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "latency_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_nim_concurrency(nim_results: list[dict]):
    """Chart NIM performance across concurrency levels."""
    if not nim_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    concurrencies = []
    throughputs = []
    p99_latencies = []

    for r in sorted(nim_results, key=lambda x: x.get("label", "")):
        conc = r.get("config", {}).get("concurrency", 0)
        if not conc:
            continue
        concurrencies.append(conc)
        throughputs.append(r.get("summary", {}).get("throughput_tps", 0))
        p99_latencies.append(r.get("summary", {}).get("latency_p99_ms", 0))

    ax1.bar(range(len(concurrencies)), throughputs, color="#76B900", alpha=0.85)
    ax1.set_xticks(range(len(concurrencies)))
    ax1.set_xticklabels([str(c) for c in concurrencies])
    ax1.set_xlabel("Concurrent Requests")
    ax1.set_ylabel("Throughput (tokens/s)")
    ax1.set_title("NIM Throughput vs Concurrency")
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(concurrencies, p99_latencies, "o-", color="#76B900", linewidth=2)
    ax2.set_xlabel("Concurrent Requests")
    ax2.set_ylabel("P99 Latency (ms)")
    ax2.set_title("NIM P99 Latency vs Concurrency")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "nim_concurrency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_report(metrics: dict, results: list[dict]):
    """Generate a markdown summary report."""
    report = f"""# Inference Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Model:** Meta Llama 3.1 8B Instruct
**Total benchmarks:** {len(results)}

## Summary

| Config | Batch=1 | Batch=8 | Batch=32 | Batch=64 |
|--------|---------|---------|----------|----------|
"""

    # Add throughput table
    for key in ["baseline", "fp8", "awq"]:
        data = metrics.get(key, [])
        row = f"| {key.upper()} "
        for bs in [1, 8, 32, 64]:
            entry = next((d for d in data if d["batch_size"] == bs), None)
            tps = entry.get("throughput_tps") if entry else None
            row += f"| {tps:.0f} tps " if tps else "| — "
        row += "|"
        report += row + "\n"

    report += """
## Charts

- `charts/throughput_comparison.png` — Throughput across batch sizes and quantization
- `charts/latency_comparison.png` — TTFT and TPOT trends
- `charts/nim_concurrency.png` — NIM serving under load

## Methodology

- **TensorRT-LLM benchmarks** run via `trtllm-bench throughput` with synthetic data
- **NIM benchmarks** run via HTTP load tester against NIM container API
- Each configuration: 3 warmup runs + 10 measurement runs (TRT-LLM), or 50 requests (NIM)
- GPU: NVIDIA A10G 24GB (AWS g5.xlarge)

## Key Observations

> Add observations after running benchmarks.

"""
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"  Report saved: {REPORT_FILE}")


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = load_all_results()
    if not results:
        print("No results found in results/raw/. Run benchmarks first.")
        sys.exit(1)

    print(f"Loaded {len(results)} result files")

    metrics = extract_metrics(results)

    print("Generating charts...")
    plot_throughput_comparison(metrics)
    plot_latency_comparison(metrics)

    nim_data = [r for r in results if "nim" in r.get("label", "")]
    plot_nim_concurrency(nim_data)

    print("Generating report...")
    generate_report(metrics, results)

    print("\nDone! Check results/ for charts and report.")


if __name__ == "__main__":
    main()
