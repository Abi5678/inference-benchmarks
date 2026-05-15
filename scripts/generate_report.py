#!/usr/bin/env python3
"""
Generate analysis report and charts from benchmark results.

Updated for TensorRT-LLM v0.21+ metrics:
  - Output throughput, token throughput, request throughput
  - TTFT/TPOT percentiles (P50, P90, P95, P99)
  - Per-user output speed vs GPU throughput tradeoff curves
  - Build-time flags impact analysis
  - NIM vs raw TRT-LLM comparison
  - GPU utilization and power metrics

Reads results/raw/*.json, produces charts/ visualizations and a summary report.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
CHARTS_DIR = RESULTS_DIR / "charts"
REPORT_FILE = RESULTS_DIR / "benchmark_report.md"

# Color palette (NVIDIA-inspired)
COLORS = {
    "baseline": "#76B900",    # NVIDIA green
    "bf16": "#76B900",
    "fp8": "#E31937",         # Red
    "awq": "#00AEEF",         # Blue
    "gptq": "#F7941D",        # Orange
    "nim": "#7B2D8E",         # Purple
    "buildflags_all": "#76B900",
    "buildflags_baseline": "#999999",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def load_all_results() -> list[dict]:
    """Load all JSON result files from raw/."""
    results = []
    for f in sorted(RAW_DIR.glob("*.json")):
        if "summary" in f.name:
            continue
        with open(f) as fh:
            try:
                data = json.load(fh)
                data["_source_file"] = f.name
                results.append(data)
            except json.JSONDecodeError:
                pass
    return results


def extract_metrics(results: list[dict]) -> dict:
    """Extract structured metrics from benchmark results."""
    categories = {
        "baseline": [],
        "fp8": [],
        "awq": [],
        "gptq": [],
        "nim": [],
        "buildflags": [],
        "cuda_graph": [],
    }

    for r in results:
        label = r.get("label", "")
        summary = r.get("summary", {})

        entry = {
            "label": label,
            "source": r.get("_source_file", ""),
            "batch_size": r.get("config", {}).get("batch_size") or 1,
            "concurrency": r.get("config", {}).get("concurrency"),
            "isl": r.get("config", {}).get("isl") or r.get("config", {}).get("input_len", 128),
            "osl": r.get("config", {}).get("osl") or r.get("config", {}).get("output_len", 256),
            # Throughput metrics (TRT-LLM style)
            "output_throughput_tps": r.get("output_throughput_tps") or summary.get("throughput_tps"),
            "token_throughput_tps": r.get("token_throughput_tps"),
            "request_throughput_rps": r.get("request_throughput_rps"),
            "per_user_output_speed": r.get("per_user_output_speed"),
            # Latency metrics
            "ttft_avg_ms": r.get("ttft_avg_ms") or summary.get("ttft_mean_ms"),
            "ttft_p50_ms": r.get("ttft_p50_ms") or summary.get("ttft_p50_ms"),
            "ttft_p95_ms": r.get("ttft_p95_ms") or summary.get("ttft_p95_ms"),
            "ttft_p99_ms": r.get("ttft_p99_ms") or summary.get("ttft_p99_ms"),
            "tpot_avg_ms": r.get("tpot_avg_ms") or r.get("tpot_ms") or summary.get("tpot_mean_ms"),
            "tpot_p50_ms": r.get("tpot_p50_ms"),
            "tpot_p99_ms": r.get("tpot_p99_ms") or summary.get("tpot_p95_ms"),
            # GPU metrics
            "gpu_metrics": r.get("gpu_metrics"),
            # Config
            "cuda_graph": r.get("config", {}).get("cuda_graph", True),
            "streaming": r.get("config", {}).get("streaming", True),
        }

        # Categorize
        if "buildflags" in label:
            categories["buildflags"].append(entry)
        elif "cuda_graph" in label:
            categories["cuda_graph"].append(entry)
        elif "nim" in label:
            categories["nim"].append(entry)
        elif "fp8" in label:
            categories["fp8"].append(entry)
        elif "awq" in label:
            categories["awq"].append(entry)
        elif "gptq" in label:
            categories["gptq"].append(entry)
        elif "baseline" in label:
            categories["baseline"].append(entry)
        else:
            categories["baseline"].append(entry)

    return categories


def plot_throughput_by_quantization(metrics: dict):
    """Bar chart: output throughput across quantization configs at different batch sizes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    groups = {}
    for key in ["baseline", "fp8", "awq"]:
        for entry in metrics[key]:
            if entry.get("concurrency") is not None:
                continue  # Skip concurrency-based results
            bs = entry["batch_size"]
            groups.setdefault(bs, {})[key] = entry.get("output_throughput_tps")

    batch_sizes = sorted(groups.keys())
    x = np.arange(len(batch_sizes))
    width = 0.25
    labels = {"baseline": "BF16", "fp8": "FP8", "awq": "AWQ 4-bit"}

    for i, (key, color_key) in enumerate([("baseline", "baseline"), ("fp8", "fp8"), ("awq", "awq")]):
        values = [groups[bs].get(key, 0) or 0 for bs in batch_sizes]
        bars = ax.bar(x + i * width, values, width, label=labels[key],
                      color=COLORS[color_key], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                       f"{val:.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel("Batch Size", fontweight='bold')
    ax.set_ylabel("Output Throughput (tokens/s)", fontweight='bold')
    ax.set_title("Inference Throughput by Quantization — Llama 3.1 8B Instruct", fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend(title="Precision")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = CHARTS_DIR / "throughput_by_quantization.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_latency_breakdown(metrics: dict):
    """Dual chart: TTFT and TPOT percentiles across batch sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for key in ["baseline", "fp8", "awq"]:
        data = sorted(
            [d for d in metrics[key] if d.get("concurrency") is None],
            key=lambda x: x["batch_size"]
        )
        if not data:
            continue
        bs = [d["batch_size"] for d in data]
        ttft = [d.get("ttft_p50_ms") or d.get("ttft_avg_ms") or 0 for d in data]
        tpot = [d.get("tpot_p50_ms") or d.get("tpot_avg_ms") or 0 for d in data]

        color = COLORS[key]
        label = {"baseline": "BF16", "fp8": "FP8", "awq": "AWQ"}[key]
        ax1.plot(bs, ttft, "o-", label=label, color=color, linewidth=2, markersize=6)
        ax2.plot(bs, tpot, "s-", label=label, color=color, linewidth=2, markersize=6)

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("TTFT P50 (ms)")
    ax1.set_title("Time to First Token", fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("TPOT P50 (ms)")
    ax2.set_title("Time per Output Token", fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    path = CHARTS_DIR / "latency_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_throughput_vs_latency_tradeoff(metrics: dict):
    """
    Per-user output speed vs per-GPU throughput tradeoff curve.
    This is the KEY visualization for understanding serving capacity.
    Based on NVIDIA blog: sweep concurrency, plot tradeoff.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (key, color, label) in enumerate([
        ("baseline", COLORS["baseline"], "BF16"),
        ("fp8", COLORS["fp8"], "FP8"),
    ]):
        ax = axes[idx]
        # Collect concurrency-sweep results
        conc_data = []
        for entry in metrics[key]:
            if entry.get("concurrency") and entry.get("concurrency", 0) > 1:
                conc_data.append(entry)
        # Also collect batch-size results mapped as concurrency
        for entry in metrics[key]:
            if entry.get("concurrency") is None and entry.get("batch_size", 0) > 1:
                conc_data.append({
                    **entry,
                    "concurrency": entry["batch_size"],
                })

        conc_data.sort(key=lambda x: x.get("concurrency", 0))

        if not conc_data:
            ax.text(0.5, 0.5, f"No data for {label}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f"{label}: No concurrency data")
            continue

        throughputs = [d.get("output_throughput_tps") or 0 for d in conc_data]
        per_user_speeds = [d.get("per_user_output_speed") or 0 for d in conc_data]
        concurrencies = [d.get("concurrency") or d.get("batch_size", 0) for d in conc_data]

        # Filter out zeros
        valid = [(t, u, c) for t, u, c in zip(throughputs, per_user_speeds, concurrencies) if t > 0 and u > 0]
        if not valid:
            ax.text(0.5, 0.5, f"No valid data for {label}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            continue

        tps, pus, concs = zip(*valid)

        ax.plot(pus, tps, "o-", color=color, linewidth=2, markersize=6)
        # Annotate key points
        for u, t, c in zip(pus, tps, concs):
            ax.annotate(f"  c={c}", (u, t), fontsize=7, color='gray')

        # Reference lines
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)

        # SLA reference: 50 tokens/sec/user
        ax.axhline(y=5000, color='red', linewidth=1, linestyle='--', alpha=0.5, label='SLA ref (5000 gpu-tps)')
        ax.axvline(x=50, color='red', linewidth=1, linestyle=':', alpha=0.5, label='50 tps/user target')

        ax.set_xlabel("Per-User Output Speed (tokens/user/s)")
        ax.set_ylabel("Per-GPU Output Throughput (tokens/s)")
        ax.set_title(f"Throughput vs User Experience: {label}", fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = CHARTS_DIR / "throughput_vs_latency_tradeoff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_nim_concurrency(nim_results: list[dict]):
    """Comprehensive NIM performance charts across concurrency levels."""
    if not nim_results:
        return

    # Filter concurrency sweep results
    conc_sweep = sorted(
        [r for r in nim_results if "conc_" in r.get("label", "")],
        key=lambda x: int(r.get("label", "").split("_")[-1])
    )

    if not conc_sweep:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    concurrencies = []
    throughputs = []
    ttft_p50s = []
    ttft_p99s = []
    tpots = []
    rps_list = []

    for r in conc_sweep:
        conc = int(r.get("label", "").split("_")[-1])
        concurrencies.append(conc)
        throughputs.append(r.get("summary", {}).get("throughput_tps", 0))
        ttft_p50s.append(r.get("summary", {}).get("ttft_p50_ms", 0))
        ttft_p99s.append(r.get("summary", {}).get("ttft_p99_ms", 0))
        tpots.append(r.get("summary", {}).get("tpot_mean_ms", 0))
        rps_list.append(r.get("summary", {}).get("requests_per_sec", 0))

    color = COLORS["nim"]

    # 1. Throughput vs Concurrency
    axes[0, 0].bar(range(len(concurrencies)), throughputs, color=color, alpha=0.85, edgecolor="white")
    axes[0, 0].set_xticks(range(len(concurrencies)))
    axes[0, 0].set_xticklabels([str(c) for c in concurrencies])
    axes[0, 0].set_xlabel("Concurrent Requests")
    axes[0, 0].set_ylabel("Throughput (tokens/s)")
    axes[0, 0].set_title("NIM Throughput vs Concurrency", fontweight='bold')
    axes[0, 0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    # 2. TTFT P50 and P99
    axes[0, 1].plot(concurrencies, ttft_p50s, "o-", color=color, linewidth=2, label="TTFT P50")
    axes[0, 1].plot(concurrencies, ttft_p99s, "s--", color="#E31937", linewidth=2, label="TTFT P99")
    axes[0, 1].fill_between(concurrencies, ttft_p50s, ttft_p99s, alpha=0.1, color=color)
    axes[0, 1].set_xlabel("Concurrent Requests")
    axes[0, 1].set_ylabel("TTFT (ms)")
    axes[0, 1].set_title("Time to First Token (Streaming)", fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, linestyle="--")
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    # 3. TPOT vs Concurrency
    axes[1, 0].plot(concurrencies, tpots, "o-", color=color, linewidth=2)
    axes[1, 0].set_xlabel("Concurrent Requests")
    axes[1, 0].set_ylabel("TPOT Mean (ms)")
    axes[1, 0].set_title("Time per Output Token", fontweight='bold')
    axes[1, 0].grid(alpha=0.3, linestyle="--")
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    # 4. Requests/sec vs Concurrency
    axes[1, 1].bar(range(len(concurrencies)), rps_list, color=color, alpha=0.85, edgecolor="white")
    axes[1, 1].set_xticks(range(len(concurrencies)))
    axes[1, 1].set_xticklabels([str(c) for c in concurrencies])
    axes[1, 1].set_xlabel("Concurrent Requests")
    axes[1, 1].set_ylabel("Requests/sec")
    axes[1, 1].set_title("Request Throughput", fontweight='bold')
    axes[1, 1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    plt.tight_layout()
    path = CHARTS_DIR / "nim_comprehensive.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_build_flags_impact(build_results: list[dict]):
    """Compare performance with/without build-time flags."""
    if len(build_results) < 2:
        return

    labels_map = {
        "buildflags_baseline": "Baseline",
        "buildflags_multi_profiles": "+ Multi Profiles",
        "buildflags_all": "+ All Flags",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    data = sorted(build_results, key=lambda x: x.get("label", ""))
    labels = [labels_map.get(d["label"], d["label"]) for d in data]

    # Throughput
    tps = [d.get("output_throughput_tps") or 0 for d in data]
    axes[0].barh(labels, tps, color=[COLORS.get("buildflags_all" if "all" in d["label"] else "buildflags_baseline", "#999") for d in data],
                 alpha=0.85, edgecolor="white")
    for i, v in enumerate(tps):
        if v > 0:
            axes[0].text(v + max(tps)*0.01, i, f"{v:.0f}", va='center', fontsize=9)
    axes[0].set_xlabel("Output Throughput (tokens/s)")
    axes[0].set_title("Build Flags: Throughput", fontweight='bold')
    axes[0].grid(axis="x", alpha=0.3, linestyle="--")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # TTFT
    ttfts = [d.get("ttft_avg_ms") or 0 for d in data]
    axes[1].barh(labels, ttfts, color=[COLORS.get("buildflags_all" if "all" in d["label"] else "buildflags_baseline", "#999") for d in data],
                 alpha=0.85, edgecolor="white")
    for i, v in enumerate(ttfts):
        if v > 0:
            axes[1].text(v + max(ttfts)*0.01, i, f"{v:.1f}ms", va='center', fontsize=9)
    axes[1].set_xlabel("TTFT (ms)")
    axes[1].set_title("Build Flags: TTFT", fontweight='bold')
    axes[1].grid(axis="x", alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # TPOT
    tpots = [d.get("tpot_avg_ms") or 0 for d in data]
    axes[2].barh(labels, tpots, color=[COLORS.get("buildflags_all" if "all" in d["label"] else "buildflags_baseline", "#999") for d in data],
                 alpha=0.85, edgecolor="white")
    for i, v in enumerate(tpots):
        if v > 0:
            axes[2].text(v + max(tpots)*0.01, i, f"{v:.2f}ms", va='center', fontsize=9)
    axes[2].set_xlabel("TPOT (ms)")
    axes[2].set_title("Build Flags: TPOT", fontweight='bold')
    axes[2].grid(axis="x", alpha=0.3, linestyle="--")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    path = CHARTS_DIR / "build_flags_impact.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cuda_graph_comparison(cuda_results: list[dict]):
    """Compare CUDA Graph ON vs OFF."""
    if len(cuda_results) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    tpots = []
    tps_vals = []
    colors_list = []

    for d in cuda_results:
        label = "CUDA Graph ON" if "on" in d["label"] else "CUDA Graph OFF"
        labels.append(label)
        tpots.append(d.get("tpot_avg_ms") or d.get("tpot_p50_ms") or 0)
        tps_vals.append(d.get("output_throughput_tps") or 0)
        colors_list.append(COLORS["baseline"] if "on" in d["label"] else "#999999")

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, tps_vals, width, label="Throughput (tokens/s)", color=colors_list, alpha=0.85)
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("CUDA Graph Impact on Decode Performance", fontweight='bold')
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = CHARTS_DIR / "cuda_graph_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_report(metrics: dict, results: list[dict]):
    """Generate a comprehensive markdown summary report."""
    total = len(results)

    # Build throughput table
    throughput_table = "| Config | Batch=1 | Batch=8 | Batch=32 | Batch=64 | Batch=128 | Batch=256 |\n"
    throughput_table += "|--------|---------|---------|----------|----------|-----------|----------|\n"

    for key, display in [("baseline", "BF16"), ("fp8", "FP8"), ("awq", "AWQ 4-bit")]:
        data = [d for d in metrics[key] if d.get("concurrency") is None]
        row = f"| {display} "
        for bs in [1, 8, 32, 64, 128, 256]:
            entries = [d for d in data if d["batch_size"] == bs and d.get("osl") in (256, None)]
            if entries:
                best = max((d.get("output_throughput_tps") or 0 for d in entries))
                row += f"| {best:.0f} tps " if best else "| — "
            else:
                row += "| — "
        row += "|"
        throughput_table += row + "\n"

    # Build flags impact
    build_data = metrics["buildflags"]
    build_summary = ""
    if len(build_data) >= 2:
        baseline_tps = next((d.get("output_throughput_tps") or 0 for d in build_data if "baseline" in d.get("label", "")), 0)
        all_flags_tps = next((d.get("output_throughput_tps") or 0 for d in build_data if "all" in d.get("label", "")), 0)
        if baseline_tps > 0:
            pct_improvement = (all_flags_tps - baseline_tps) / baseline_tps * 100
            build_summary = f"**Build flags improvement:** {pct_improvement:+.1f}% throughput ({baseline_tps:.0f} → {all_flags_tps:.0f} tokens/s)\n"

    # NIM summary
    nim_data = metrics["nim"]
    nim_summary = ""
    conc_results = [d for d in nim_data if "conc_" in d.get("label", "")]
    if conc_results:
        best_tps = max((d.get("summary", {}).get("throughput_tps") or 0 for d in conc_results))
        best_ttft = min((d.get("summary", {}).get("ttft_p50_ms") or 999 for d in conc_results if d.get("summary", {}).get("ttft_p50_ms")))
        nim_summary = f"**NIM peak throughput:** {best_tps:.0f} tokens/s, **best TTFT:** {best_ttft:.1f}ms\n"

    report = f"""# Inference Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Model:** Meta Llama 3.1 8B Instruct  
**TensorRT-LLM:** v0.21+  
**Total benchmarks:** {total}

## Executive Summary

{build_summary}{nim_summary}

## Throughput Comparison (tokens/s)

{throughput_table}

*ISL=128 tokens. Output length varies by configuration.*

## Latency Comparison

### TTFT (Time to First Token)

| Config | Batch=1 | Batch=8 | Batch=32 | Batch=64 |
|--------|---------|---------|----------|----------|
"""

    for key, display in [("baseline", "BF16"), ("fp8", "FP8"), ("awq", "AWQ")]:
        data = [d for d in metrics[key] if d.get("concurrency") is None]
        row = f"| {display} "
        for bs in [1, 8, 32, 64]:
            entry = next((d for d in data if d["batch_size"] == bs), None)
            ttft = entry.get("ttft_p50_ms") or entry.get("ttft_avg_ms") if entry else None
            row += f"| {ttft:.1f}ms " if ttft else "| — "
        row += "|"
        report += row + "\n"

    report += """
### TPOT (Time per Output Token)

| Config | Batch=1 | Batch=8 | Batch=32 | Batch=64 |
|--------|---------|---------|----------|----------|
"""

    for key, display in [("baseline", "BF16"), ("fp8", "FP8"), ("awq", "AWQ")]:
        data = [d for d in metrics[key] if d.get("concurrency") is None]
        row = f"| {display} "
        for bs in [1, 8, 32, 64]:
            entry = next((d for d in data if d["batch_size"] == bs), None)
            tpot = entry.get("tpot_p50_ms") or entry.get("tpot_avg_ms") if entry else None
            row += f"| {tpot:.2f}ms " if tpot else "| — "
        row += "|"
        report += row + "\n"

    report += f"""
## Charts

| Chart | Description |
|-------|-------------|
| `throughput_by_quantization.png` | Output throughput across batch sizes by quantization |
| `latency_breakdown.png` | TTFT and TPOT trends across batch sizes |
| `throughput_vs_latency_tradeoff.png` | Per-user speed vs GPU throughput (serving capacity) |
| `nim_comprehensive.png` | NIM performance under load (4 metrics) |
| `build_flags_impact.png` | Build-time flags A/B comparison |
| `cuda_graph_comparison.png` | CUDA Graph ON vs OFF decode performance |

## Methodology

- **TensorRT-LLM benchmarks:** `trtllm-bench throughput` v0.21+ with `--dataset`, `--concurrency`, `--streaming`
- **NIM benchmarks:** Streaming HTTP load tester against NIM v1.15+ container API
- **Dataset:** Synthetic prompts (~128 input tokens) with configurable output lengths
- **Warmup:** 3 warmup runs before measurement (TRT-LLM)
- **Metrics:** Output throughput, token throughput, TTFT/TPOT at P50/P95/P99
- **Build flags:** Multiple profiles, paged context FMHA, GEMM plugin, reduce fusion

## Key Observations

> Add observations after running benchmarks on GPU hardware.

## References

- [TensorRT-LLM Performance Tuning Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/index.html)
- [trtllm-bench Documentation](https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-benchmarking.html)
- [NIM LLM Benchmarking Guide](https://docs.nvidia.com/nim/benchmarking/llm/latest/index.html)
- [LLM Inference Benchmarking Blog Series](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/)
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
        print("  python scripts/run_benchmarks.py --phase baseline")
        sys.exit(1)

    print(f"Loaded {len(results)} result files")

    metrics = extract_metrics(results)

    print("\nGenerating charts...")
    plot_throughput_by_quantization(metrics)
    plot_latency_breakdown(metrics)
    plot_throughput_vs_latency_tradeoff(metrics)

    if metrics["nim"]:
        plot_nim_concurrency(metrics["nim"])

    if metrics["buildflags"]:
        plot_build_flags_impact(metrics["buildflags"])

    if metrics["cuda_graph"]:
        plot_cuda_graph_comparison(metrics["cuda_graph"])

    print("\nGenerating report...")
    generate_report(metrics, results)

    print("\nDone! Check results/ for charts and benchmark_report.md")


if __name__ == "__main__":
    main()
