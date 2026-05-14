# Inference Benchmarks: TensorRT-LLM + NIM Performance Analysis

**Author:** Abishek Bangalore Muralikrishna  
**Date:** May 2026  
**Purpose:** Hands-on inference engineering portfolio — demonstrating TensorRT-LLM optimization, NIM deployment, and production-ready benchmarking practices.

## What This Repo Contains

Reproducible benchmarks comparing inference configurations for Llama 3.1 8B Instruct across:

- **Quantization:** BF16 baseline, FP8, AWQ 4-bit, GPTQ 4-bit
- **Parallelism:** TP=1, TP=2 (where hardware allows)
- **Batching:** Batch sizes 1–128 with continuous batching + PagedAttention KV cache
- **CUDA Graphs:** On vs off for decode-phase capture
- **Serving:** Raw TensorRT-LLM vs NIM container deployment

## Quick Start

### Prerequisites

- NVIDIA GPU with ≥24GB VRAM (A10G, A100, RTX 4090, etc.)
- Docker + NVIDIA Container Toolkit
- Python 3.10+
- 100GB free disk space

### One-Command Setup

```bash
# Clone and enter the project
git clone https://github.com/Abi5678/inference-benchmarks.git
cd inference-benchmarks

# Set up environment (installs TRT-LLM, pulls model, creates configs)
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Run Full Benchmark Suite

```bash
# Phase 1: Baseline (BF16, TP=1)
python scripts/run_benchmarks.py --phase baseline

# Phase 2: Optimization sweep (all quantization + batching configs)
python scripts/run_benchmarks.py --phase sweep

# Phase 3: NIM deployment comparison
python scripts/run_benchmarks.py --phase nim

# Phase 4: Generate all charts + results summary
python scripts/generate_report.py
```

### Results

All results are saved to `results/` as JSON and CSV. Charts go to `results/charts/`.

## Project Structure

```
inference-benchmarks/
├── README.md                    # This file
├── scripts/
│   ├── setup.sh                 # Environment setup
│   ├── run_benchmarks.py        # Main benchmark orchestrator
│   ├── benchmark_trtllm.py      # TensorRT-LLM benchmark wrapper
│   ├── benchmark_nim.py         # NIM container benchmark wrapper
│   ├── generate_report.py       # Results analysis + charting
│   └── load_test.py             # Concurrent request load tester
├── configs/
│   ├── baseline.yaml            # BF16 TP=1 configuration
│   ├── sweep_fp8.yaml           # FP8 quantization configs
│   ├── sweep_awq.yaml           # AWQ 4-bit configs
│   └── nim_config.yaml          # NIM deployment settings
├── notebooks/
│   └── analysis.ipynb           # Interactive results exploration
├── results/
│   ├── charts/                  # Generated visualizations
│   └── raw/                     # Raw benchmark outputs
├── docker/
│   ├── Dockerfile.benchmark     # Benchmark environment
│   └── docker-compose.yml       # NIM deployment compose file
└── docs/
    └── blog-draft.md            # Blog post draft
```

## Key Findings (Summary)

> Results pending GPU execution. See `results/` after running benchmarks.

## Technical Stack

| Component | Version | Notes |
|-----------|---------|-------|
| TensorRT-LLM | v0.14+ | Latest stable |
| NVIDIA NIM | Latest | Pulled from NGC |
| Model | Llama 3.1 8B Instruct | HuggingFace `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| GPU | A10G 24GB (AWS g5.xlarge) | Baseline target |
| Python | 3.10 | Required by TRT-LLM |

## License

MIT

## Contact

- **GitHub:** [@Abi5678](https://github.com/Abi5678)
- **LinkedIn:** [Abishek Bangalore Muralikrishna](https://linkedin.com/in/abishek-bm)
- **Blog:** (forthcoming)
