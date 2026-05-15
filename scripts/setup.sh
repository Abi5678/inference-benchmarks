#!/usr/bin/env bash
# Inference Benchmarks - Environment Setup (Updated for TRT-LLM v0.21+)
# Tested on: Ubuntu 22.04 + NVIDIA A10G (24GB), A100 (80GB), H100 (80GB)
set -euo pipefail

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
FP8_MODEL_NAME="nvidia/Llama-3.1-8B-Instruct-FP8"
MODEL_DIR="${HOME}/models/llama-3.1-8b-instruct"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/results"

echo "╔══════════════════════════════════════════════════════╗"
echo "║     Inference Benchmarks Setup (TRT-LLM v0.21+)     ║"
echo "╚══════════════════════════════════════════════════════╝"

echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || {
    echo "ERROR: No NVIDIA GPU detected. This project requires an NVIDIA GPU."
    exit 1
}

echo ""
echo "=== Step 1: Create directories ==="
mkdir -p "${RESULTS_DIR}/raw" "${RESULTS_DIR}/charts"

echo ""
echo "=== Step 2: Check Python ==="
python3 --version || { echo "ERROR: Python 3.10+ required"; exit 1; }

echo ""
echo "=== Step 3: Install TensorRT-LLM v0.21+ ==="
if ! python3 -c "import tensorrt_llm; v = tensorrt_llm.__version__; assert v >= '0.19'" 2>/dev/null; then
    echo "Installing TensorRT-LLM..."
    pip install "tensorrt-llm>=0.21.0" --extra-index-url https://pypi.nvidia.com
else
    echo "TensorRT-LLM: $(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)')"
fi

echo ""
echo "=== Step 4: Verify trtllm-bench ==="
if command -v trtllm-bench &>/dev/null; then
    echo "trtllm-bench: available"
else
    echo "WARNING: trtllm-bench not found in PATH. It should come with tensorrt-llm."
fi

echo ""
echo "=== Step 5: Install benchmark dependencies ==="
pip install pyyaml matplotlib numpy pandas requests aiohttp

echo ""
echo "=== Step 6: Download model (BF16) ==="
if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Model already exists at ${MODEL_DIR}"
else
    echo "Downloading ${MODEL_NAME}..."
    mkdir -p "${MODEL_DIR}"
    huggingface-cli download "${MODEL_NAME}" --local-dir "${MODEL_DIR}"
fi

echo ""
echo "=== Step 7: (Optional) Download FP8 checkpoint ==="
FP8_DIR="${MODEL_DIR}-fp8"
if [ -d "${FP8_DIR}" ] && [ -f "${FP8_DIR}/config.json" ]; then
    echo "FP8 model already exists at ${FP8_DIR}"
else
    echo "Downloading ${FP8_MODEL_NAME}..."
    mkdir -p "${FP8_DIR}"
    huggingface-cli download "${FP8_MODEL_NAME}" --local-dir "${FP8_DIR}" || {
        echo "WARNING: FP8 download failed. FP8 benchmarks will use on-the-fly quantization."
    }
fi

echo ""
echo "=== Step 8: (Optional) Install GenAI-Perf for NIM benchmarking ==="
if command -v genai-perf &>/dev/null; then
    echo "GenAI-Perf: already installed"
else
    echo "Installing GenAI-Perf (optional)..."
    pip install genai-perf 2>/dev/null || echo "WARNING: GenAI-Perf install failed. NIM benchmarks will use the Python load tester."
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                Setup Complete!                      ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
echo "║  Run benchmarks:                                     ║"
echo "║    python scripts/run_benchmarks.py --phase baseline ║"
echo "║    python scripts/run_benchmarks.py --phase sweep    ║"
echo "║    python scripts/run_benchmarks.py --phase nim      ║"
echo "║    python scripts/run_benchmarks.py --phase all      ║"
echo "║                                                      ║"
echo "║  Generate report:                                    ║"
echo "║    python scripts/generate_report.py                 ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
