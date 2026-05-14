#!/usr/bin/env bash
# Inference Benchmarks - Environment Setup
# Tested on: Ubuntu 22.04 + NVIDIA A10G (24GB)
set -euo pipefail

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_DIR="${HOME}/models/llama-3.1-8b-instruct"
RESULTS_DIR="$(cd "$(dirname "$0")/../results" && pwd)"

echo "=== Inference Benchmarks Setup ==="
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
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
echo "=== Step 3: Install TensorRT-LLM ==="
if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
    echo "Installing TensorRT-LLM..."
    pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
else
    echo "TensorRT-LLM already installed: $(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)')"
fi

echo ""
echo "=== Step 4: Download model ==="
if [ -d "${MODEL_DIR}" ]; then
    echo "Model already exists at ${MODEL_DIR}"
else
    echo "Downloading ${MODEL_NAME}..."
    mkdir -p "${MODEL_DIR}"
    huggingface-cli download "${MODEL_NAME}" --local-dir "${MODEL_DIR}"
fi

echo ""
echo "=== Step 5: Convert model to TRT-LLM format (BF16 baseline) ==="
python3 "$(dirname "$0")/convert_model.py" \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${MODEL_DIR}/trtllm-bf16" \
    --dtype bf16

echo ""
echo "=== Setup complete ==="
echo "Run benchmarks with: python3 scripts/run_benchmarks.py --phase baseline"
