# Inference Benchmarks

**Production-grade LLM inference benchmark suite for TensorRT-LLM v0.21+ and NVIDIA NIM v1.15+.**

Benchmark Llama 3.1 8B Instruct across BF16, FP8, and AWQ quantization with comprehensive throughput, latency (TTFT/TPOT), and GPU utilization metrics.

## Overview

This project provides a complete pipeline for:

- **Measuring LLM inference performance** with `trtllm-bench` v0.21+ (dataset-based, concurrency sweeps, streaming TTFT)
- **Comparing quantization** — BF16 vs FP8 vs AWQ with identical workloads
- **Testing build-time optimizations** — multiple_profiles, paged_context_fmha, GEMM plugin, reduce_fusion
- **Benchmarking NIM deployments** — streaming load tester + GenAI-Perf integration
- **Visualizing tradeoffs** — throughput vs latency curves, percentile latency charts, cost analysis

## Project Structure

```
inference-benchmarks/
├── configs/                  # YAML configs for each benchmark phase
│   ├── baseline.yaml         # BF16 baseline parameters
│   ├── sweep_fp8.yaml        # FP8 sweep configuration
│   ├── sweep_awq.yaml        # AWQ sweep configuration
│   ├── build_flags.yaml      # Build-time flag A/B tests
│   └── nim_config.yaml       # NIM deployment settings
├── scripts/
│   ├── run_benchmarks.py     # Main orchestrator (4 phases)
│   ├── benchmark_nim.py      # NIM streaming load tester with TTFT
│   ├── benchmark_aiperf.py   # GenAI-Perf / AIPerf wrapper
│   ├── convert_model.py      # HF → TRT-LLM engine (with build flags)
│   ├── generate_report.py    # Charts + markdown report generator
│   ├── load_test.py          # Advanced load tester (percentiles + GPU monitoring)
│   └── setup.sh              # GPU environment setup script
├── docker/
│   ├── Dockerfile.benchmark  # Self-contained CUDA 12.4 + TRT-LLM environment
│   └── docker-compose.yml    # NIM server + benchmark runner
├── notebooks/
│   └── analysis.ipynb        # Interactive analysis notebook
├── docs/
│   └── blog-draft.md         # Accompanying blog post draft
├── results/
│   ├── raw/                  # JSON results (auto-generated)
│   └── charts/               # PNG visualizations (auto-generated)
└── README.md
```

## Requirements

- **GPU:** NVIDIA GPU (A10G 24GB, A100 80GB, H100 80GB tested)
- **Driver:** NVIDIA driver 535+
- **CUDA:** 12.1+
- **Python:** 3.10+
- **TensorRT-LLM:** v0.21+
- **Docker + NVIDIA Container Toolkit** (for NIM benchmarks)

## Google Colab (GPU notebook)

Use this if you want a hosted **GPU runtime** without installing TensorRT-LLM locally. The notebook detects VRAM, tries TensorRT-LLM when compatible, and falls back to Hugging Face Transformers.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abi5678/inference-benchmarks/blob/main/inference-benchmarks-colab.ipynb)

**Runtime:** **Runtime → Change runtime type → GPU** (T4 on free tier when available).

**Colab secrets (sidebar → 🔑):**

| Name | Purpose |
|------|---------|
| `HF_TOKEN` | Optional; needed for gated **Llama 3.1** weights on Hugging Face. Without it, the notebook uses **Phi-3-mini**. |
| `NVIDIA_BUILD_API_KEY` | Optional; enables **NVIDIA NIM cloud** benchmarks via [build.nvidia.com](https://build.nvidia.com/). |

After a full run, download generated artifacts (`*.png`, `inference_benchmark_report.md`, `inference_benchmark_results.csv`) if you want them in-repo under `results/colab/` (that folder is documented for those exports).

## Quick Start

### 1. Setup

```bash
git clone https://github.com/Abi5678/inference-benchmarks.git
cd inference-benchmarks
bash scripts/setup.sh
```

Setup installs TensorRT-LLM, downloads Llama 3.1 8B Instruct, and creates result directories.

### 2. Run Benchmarks

```bash
# Phase A: Baseline (BF16, batch + concurrency sweeps)
python scripts/run_benchmarks.py --phase baseline

# Phase B: Optimization sweep (FP8, AWQ, CUDA Graphs)
python scripts/run_benchmarks.py --phase sweep

# Phase C: NIM deployment (requires running NIM container)
python scripts/run_benchmarks.py --phase nim

# Phase D: Build-time flags A/B comparison
python scripts/run_benchmarks.py --phase build-flags

# Run everything:
python scripts/run_benchmarks.py --phase all
```

### 3. Generate Report

```bash
python scripts/generate_report.py
# Outputs: results/benchmark_report.md + results/charts/*.png
```

### 4. Interactive Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

## NIM Benchmarks

### Start NIM Container

```bash
docker run -d --gpus all -p 8000:8000 \
    -e NGC_API_KEY=$NGC_API_KEY \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### Using Docker Compose (NIM + Benchmark Runner)

```bash
cd docker
export NGC_API_KEY=nvapi-...
docker compose up --abort-on-container-exit
```

### Custom Load Test

```bash
# Streaming load test with percentiles
python scripts/load_test.py \
    --url http://localhost:8000 \
    --concurrency 50 \
    --requests 500 \
    --streaming \
    --warmup 20

# Non-streaming comparison
python scripts/load_test.py \
    --url http://localhost:8000 \
    --concurrency 50 \
    --requests 500 \
    --no-streaming
```

## Key Features

### Modern `trtllm-bench` API (v0.21+)
- `--dataset` flag for realistic workload testing
- `--concurrency` for true concurrent request simulation
- `--streaming` for accurate TTFT measurement
- `--report_json` for machine-readable output

### Build-Time Optimizations
- **multiple_profiles** — runtime optimization profile switching (up to 20% throughput gain)
- **use_paged_context_fmha** — reduced memory fragmentation for larger batches
- **gemm_plugin** — auto for BF16/FP16, disabled for FP8
- **reduce_fusion** — TP boundary optimization for multi-GPU

### Comprehensive Metrics
- **Throughput:** Output throughput, token throughput, request throughput
- **Latency:** TTFT (P50/P90/P95/P99), TPOT (P50/P95/P99), end-to-end
- **Tradeoff:** Per-user output speed vs per-GPU throughput curve
- **GPU:** Power draw, VRAM usage, utilization during load

### Three Benchmarking Approaches
1. **trtllm-bench** — Engine-level benchmarking (highest fidelity)
2. **Custom load tester** — HTTP API-level with streaming TTFT
3. **GenAI-Perf** — NVIDIA's recommended NIM benchmarking tool

## Output Charts

| Chart | Description |
|-------|-------------|
| `throughput_by_quantization.png` | Throughput across batch sizes by quantization |
| `latency_breakdown.png` | TTFT and TPOT trends |
| `throughput_vs_latency_tradeoff.png` | Per-user speed vs GPU throughput |
| `nim_comprehensive.png` | NIM performance under load (4-panel) |
| `build_flags_impact.png` | Build-time flags A/B comparison |
| `cuda_graph_comparison.png` | CUDA Graph ON vs OFF |

## Tested Hardware

| GPU | VRAM | Architecture | Notes |
|-----|------|-------------|-------|
| NVIDIA A10G | 24GB | Ampere | Dev, AWQ, smaller batches |
| NVIDIA A100 | 80GB | Ampere | Production, BF16/FP8 |
| NVIDIA H100 | 80GB | Hopper | FP8 optimized, max throughput |

## References

- [TensorRT-LLM Performance Tuning Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/index.html)
- [trtllm-bench Documentation](https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-benchmarking.html)
- [NIM LLM Benchmarking Guide](https://docs.nvidia.com/nim/benchmarking/llm/latest/index.html)
- [LLM Inference Blog Series](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/)
- [FP8 Formats for Deep Learning](https://developer.nvidia.com/blog/fp8-formats-for-deep-learning/)

## Author

**Abishek Bangalore Muralikrishna**
Applied ML Researcher | NVIDIA Ecosystem Power User
