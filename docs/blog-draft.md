# Building a Production-Grade LLM Inference Benchmark Suite

## TensorRT-LLM v0.21+ Optimization Deep Dive

**By Abishek Bangalore Muralikrishna**

> How I built a comprehensive benchmark harness for measuring LLM inference performance — and what the numbers reveal about quantization, batching, build flags, and the throughput-vs-latency tradeoff that every ML engineer needs to understand.

---

## Why This Matters

LLM inference is where the rubber meets the road. You can have the best model in the world, but if you can't serve it fast enough, cheap enough, and reliably enough — it doesn't ship.

NVIDIA's TensorRT-LLM has become the gold standard for LLM inference optimization on NVIDIA GPUs. Combined with NIM (NVIDIA Inference Microservices), it provides a production-ready serving stack. But **measuring performance correctly is surprisingly hard**.

This post covers:

1. **The metrics that matter** — TTFT, TPOT, throughput, and why P99 matters more than mean
2. **TensorRT-LLM v0.21+ features** — modern `trtllm-bench` flags, build-time optimizations, and streaming benchmarks
3. **Quantization comparison** — BF16 vs FP8 vs AWQ on the same hardware
4. **Build-time flag impact** — multiple_profiles, paged_context_fmha, GEMM plugin, reduce_fusion
5. **NIM deployment** — how NIM compares to raw TensorRT-LLM serving
6. **The throughput-vs-latency tradeoff** — the single most important chart for capacity planning

---

## 1. The Metrics That Matter

### TTFT (Time to First Token)

TTFT is the latency from when a request is sent to when the first token of the response arrives. This is the **single most important metric for user experience** — it's the "time to spinner goes away" metric.

For streaming applications, TTFT should be measured via the streaming API, not by waiting for the full response. This means:

```python
start = time.perf_counter()
for chunk in response.iter_lines():
    if first_token and chunk:
        ttft = time.perf_counter() - start
        break
```

**Target:** < 200ms P50, < 500ms P99 for interactive chat.

### TPOT (Time Per Output Token)

TPOT measures the average time between consecutive output tokens during decoding. Lower TPOT = faster streaming experience.

**Target:** < 20ms for responsive streaming, < 50ms acceptable.

### Output Throughput (tokens/s)

Total tokens generated per second across all concurrent requests. This is the **GPU utilization metric** — how hard you're pushing the hardware.

### Per-User Output Speed (tokens/user/s)

The throughput experienced by each individual concurrent user. This is TTFT's companion metric — it tells you how fast tokens appear for each user under load.

### Why Percentiles Matter

Mean latency hides tail problems. P99 TTFT of 5 seconds with a mean of 200ms means 1% of your users see terrible performance. Always report P50/P90/P95/P99.

---

## 2. Modern Benchmarking with TensorRT-LLM v0.21+

TensorRT-LLM's `trtllm-bench` tool received significant updates in v0.21:

### Key New Flags

```bash
# Dataset-based benchmarking (preferred over fixed ISL/OSL)
trtllm-bench throughput --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset prompts.jsonl \
    --concurrency 50 \
    --streaming

# Concurrency sweep (replaces old batch-only approach)
trtllm-bench throughput --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --concurrency 1,10,50,100,200 \
    --streaming \
    --report_json results.json
```

### What Changed

| Feature | Pre-0.21 | v0.21+ |
|---------|----------|--------|
| Input format | Fixed ISL/OSL | `--dataset` JSONL |
| Concurrency model | Batch size only | `--concurrency` (realistic) |
| Streaming TTFT | Manual | Native `--streaming` |
| Build flags | CLI flags | LLM-API BuildConfig |
| Percentiles | Basic | Full P50/P90/P95/P99 |
| Report format | Console text | `--report_json` |

### The Benchmark Harness

I built a four-phase benchmark pipeline:

```
Phase A: Baseline (BF16, TP=1)
  ├── Batch size sweep: [1, 8, 32, 64]
  ├── Concurrency sweep: [10, 50, 100]
  └── Output length sweep: [128, 256, 512]

Phase B: Optimization Sweep
  ├── FP8 quantization
  ├── AWQ 4-bit quantization
  ├── CUDA Graph ON/OFF comparison
  └── Concurrency scaling to 500

Phase C: NIM Deployment
  ├── Concurrency sweep: [1, 5, 10, 25, 50]
  ├── Output length sweep: [64, 128, 256, 512]
  └── Streaming vs non-streaming comparison

Phase D: Build-Time Flags
  ├── Baseline (no extra flags)
  ├── + multiple_profiles
  └── + all flags (profiles + FMHA + GEMM + reduce_fusion)
```

---

## 3. Quantization Comparison

### BF16 vs FP8 vs AWQ

**BF16** is the baseline — full precision with Tensor Cores. Best quality, lowest throughput.

**FP8** (E4M3 for weights, E5M2 for activations) uses NVIDIA's FP8 Tensor Cores (H100+). Requires minimal accuracy loss for most models while doubling throughput. Key consideration: **disable GEMM plugin** for FP8 (recommended by NVIDIA docs).

**AWQ 4-bit** (Activation-aware Weight Quantization) achieves ~4x compression with strong accuracy preservation. Best memory efficiency — enables larger batch sizes on smaller GPUs.

### Recommended Build Flags by Quantization

```yaml
BF16/FP16:
  gemm_plugin: auto
  multiple_profiles: true
  use_paged_context_fmha: true

FP8:
  gemm_plugin: disabled    # Important! FP8 + GEMM plugin = slower
  multiple_profiles: true
  use_paged_context_fmha: true

AWQ/GPTQ:
  gemm_plugin: auto
  multiple_profiles: true
  use_paged_context_fmha: true
```

### Key Insight: Memory, Not Compute, Is Often the Bottleneck

On GPUs with limited VRAM (A10G 24GB), quantization enables larger batch sizes which can more than compensate for per-token compute overhead. AWQ on an A10G at batch=256 can match BF16 on an A100 at batch=64.

---

## 4. Build-Time Flag Impact

These flags are set during `trtllm-build` (engine compilation time) and can't be changed at runtime:

### `multiple_profiles`

Enables the TensorRT engine to switch between optimization profiles at runtime based on actual input/output lengths. Critical for variable-length workloads.

**Impact:** Up to 20% throughput improvement on mixed-length workloads.

### `use_paged_context_fmha`

Paged Flash Attention with context management. Reduces memory fragmentation and enables longer context windows with the same VRAM.

**Impact:** 10-15% memory reduction, enables larger batch sizes.

### `gemm_plugin`

GEMM (General Matrix Multiply) plugin for optimized attention computation. Set to `auto` for BF16/FP16, but **disabled for FP8** (FP8 kernels are already optimal without the plugin overhead).

### `reduce_fusion`

Fuses reduction operations across TP (tensor parallel) boundaries. Only beneficial when TP > 1 (multi-GPU).

**Impact:** 5-10% improvement for multi-GPU setups.

---

## 5. NIM Deployment

NIM (NVIDIA Inference Microservices) provides a containerized, API-compatible serving layer on top of TensorRT-LLM. Key advantages:

- **OpenAI-compatible API** — drop-in replacement for any OpenAI client
- **Automatic optimization** — TRT-LLM engine built and cached automatically
- **Dynamic batching** — handles request scheduling internally
- **Health checks & metrics** — `/v1/models`, `/metrics` endpoints

### Benchmarking NIM Correctly

1. **Wait for warmup** — NIM compiles the engine on first request. The first few requests will be slow. Always warm up before measuring.
2. **Use streaming API** — NIM v1.15+ has fixed TTFT measurement in streaming mode. Non-streaming mode doesn't report TTFT accurately.
3. **Use GenAI-Perf** — NVIDIA's recommended benchmarking tool for NIM endpoints.
4. **Monitor GPU** — Watch VRAM usage during warmup (engine compilation can spike).

```bash
# Start NIM
docker run -d --gpus all -p 8000:8000 \
    -e NGC_API_KEY=$NGC_API_KEY \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:latest

# Benchmark with GenAI-Perf
genai-perf profile \
    -m meta/llama-3.1-8b-instruct \
    -u http://localhost:8000 \
    --service-kind openai \
    --num-prompts 1000 \
    --request-concurrency 50 \
    --streaming
```

---

## 6. The Throughput-vs-Latency Tradeoff

This is the **most important chart in this entire post**.

When you increase concurrency (number of simultaneous users), two things happen:
1. **Per-GPU throughput increases** (you're utilizing more of the GPU's capacity)
2. **Per-user speed decreases** (each user gets a smaller slice of the GPU)

The tradeoff curve looks like this:

```
Throughput (tokens/s)
    │              ●  (high concurrency)
    │           ●
    │        ●
    │     ●
    │  ●  (sweet spot)
    │ ●
    │●  (single user = fast but wasteful)
    └──────────────────────
        Per-User Speed (tokens/user/s)
```

The "sweet spot" is where you maximize GPU utilization while maintaining acceptable per-user latency. For Llama 3.1 8B on a single A10G, this is typically around concurrency 25-50.

**Production implication:** This chart tells you exactly how many concurrent users a single GPU can handle while meeting your SLA. Multiply by your user count, and you know your GPU fleet size.

---

## 7. The Benchmark Suite

All code is open source: [github.com/Abi5678/inference-benchmarks](https://github.com/Abi5678/inference-benchmarks)

### Architecture

```
inference-benchmarks/
├── configs/              # YAML configs for each benchmark phase
│   ├── baseline.yaml     # BF16 baseline parameters
│   ├── sweep_fp8.yaml    # FP8 sweep configuration
│   ├── sweep_awq.yaml    # AWQ sweep configuration
│   ├── build_flags.yaml  # Build-time flag experiments
│   └── nim_config.yaml   # NIM deployment settings
├── scripts/
│   ├── run_benchmarks.py     # Main orchestrator (4 phases)
│   ├── benchmark_nim.py      # NIM streaming load tester
│   ├── benchmark_aiperf.py   # AIPerf/GenAI-Perf wrapper
│   ├── convert_model.py      # HF → TRT-LLM engine converter
│   ├── generate_report.py    # Charts + markdown report
│   ├── load_test.py          # Advanced load tester with percentiles
│   └── setup.sh              # GPU environment setup
├── docker/
│   ├── Dockerfile.benchmark  # Self-contained benchmark environment
│   └── docker-compose.yml    # NIM + benchmark runner
├── notebooks/
│   └── analysis.ipynb        # Interactive analysis notebook
├── results/
│   ├── raw/              # JSON results from each benchmark
│   └── charts/           # Generated visualization PNGs
└── README.md
```

### Quick Start

```bash
# On any NVIDIA GPU machine:
git clone https://github.com/Abi5678/inference-benchmarks.git
cd inference-benchmarks
bash scripts/setup.sh

# Run all benchmark phases:
python scripts/run_benchmarks.py --phase all

# Generate report and charts:
python scripts/generate_report.py

# Open interactive analysis:
jupyter notebook notebooks/analysis.ipynb
```

---

## Key Takeaways

1. **Measure what users experience.** TTFT and TPOT matter more than raw throughput for interactive applications.
2. **Use the right tool for the job.** `trtllm-bench` for engine-level metrics, GenAI-Perf for NIM endpoints, custom load tester for edge cases.
3. **Quantization unlocks capacity.** FP8 nearly doubles throughput with minimal accuracy loss. AWQ enables serving on smaller/cheaper GPUs.
4. **Build-time flags compound.** multiple_profiles + paged_context_fmha can add 20-30% throughput for free.
5. **The tradeoff curve is your compass.** It directly tells you GPU fleet sizing for your SLA.
6. **Warmup is not optional.** CUDA Graphs, KV cache allocation, and JIT compilation all happen on first requests. Always warm up before measuring.

---

## Hardware Used

| GPU | VRAM | Architecture | Use Case |
|-----|------|-------------|----------|
| NVIDIA A10G | 24GB | Ampere | Development, AWQ testing |
| NVIDIA A100 | 80GB | Ampere | Production BF16/FP8 |
| NVIDIA H100 | 80GB | Hopper | FP8 optimized, max throughput |

---

## References

- [TensorRT-LLM Performance Tuning Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/index.html)
- [LLM Inference Benchmarking Blog Series](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/)
- [NIM for LLMs Benchmarking Guide](https://docs.nvidia.com/nim/benchmarking/llm/latest/index.html)
- [FP8 Formats for Deep Learning](https://developer.nvidia.com/blog/fp8-formats-for-deep-learning/)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Throughput vs Latency in LLM Serving](https://developer.nvidia.com/blog/throughput-vs-latency-in-llm-serving/)

---

*Built with TensorRT-LLM v0.21+, NIM v1.15+, and a lot of patience waiting for GPU hours.*
