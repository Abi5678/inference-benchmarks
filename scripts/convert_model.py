#!/usr/bin/env python3
"""
Convert HuggingFace model to TensorRT-LLM engine format.

Updated for TensorRT-LLM v0.21+ with:
  - LLM-API BuildConfig for modern engine building
  - Build-time optimization flags (multiple_profiles, GEMM plugin, etc.)
  - Support for BF16, FP8, AWQ, and GPTQ quantization
  - Per-quantization recommended build flags

Usage:
    python convert_model.py \
        --model_dir ~/models/llama-3.1-8b-instruct \
        --output_dir ~/models/llama-3.1-8b-instruct/trtllm-bf16 \
        --dtype bf16

    python convert_model.py \
        --model_dir ~/models/llama-3.1-8b-instruct \
        --output_dir ~/models/llama-3.1-8b-instruct/trtllm-fp8 \
        --dtype fp8

    python convert_model.py \
        --model_dir ~/models/llama-3.1-8b-instruct-awq \
        --output_dir ~/models/llama-3.1-8b-instruct/trtllm-awq \
        --dtype fp16 --quantization awq
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with output."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr[-2000:]}")
        sys.exit(1)
    if result.stdout.strip():
        print(f"  Output: {result.stdout[-500:]}")
    return result


def convert_model(
    model_dir: str,
    output_dir: str,
    dtype: str = "bf16",
    quantization: str = None,
    tp_size: int = 1,
    pp_size: int = 1,
    max_batch_size: int = 128,
    max_num_tokens: int = None,
    max_seq_len: int = 4096,
    multiple_profiles: bool = True,
    gemm_plugin: str = "auto",
    use_paged_context_fmha: bool = True,
    reduce_fusion: bool = False,
):
    """
    Convert HF model to TRT-LLM checkpoint, then build engine.

    Applies recommended build-time flags based on quantization type:
    - BF16/FP16: gemm_plugin=auto, multiple_profiles=True
    - FP8: gemm_plugin=disabled (recommended), multiple_profiles=True
    - AWQ/GPTQ: gemm_plugin=auto, multiple_profiles=True
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Default max_num_tokens based on max_batch_size if not specified
    if max_num_tokens is None:
        max_num_tokens = max_batch_size * 2

    # --- Step 1: Convert HF to TRT-LLM checkpoint ---
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
        print(f"  Checkpoint exists at {checkpoint_dir}, skipping conversion.")
    else:
        print(f"\n  Converting HuggingFace model to TRT-LLM checkpoint...")
        cmd = [
            sys.executable, "-m", "tensorrt_llm.commands.convert_checkpoint",
            "--model_dir", model_dir,
            "--output_dir", checkpoint_dir,
            "--dtype", dtype,
            "--tp_size", str(tp_size),
        ]
        if pp_size > 1:
            cmd.extend(["--pp_size", str(pp_size)])
        if quantization:
            cmd.extend(["--quantization", quantization])
        run_cmd(cmd)

    # --- Step 2: Build TRT-LLM engine with recommended flags ---
    engine_dir = os.path.join(output_dir, "engine")
    if os.path.exists(engine_dir) and os.listdir(engine_dir):
        print(f"  Engine exists at {engine_dir}, skipping build.")
    else:
        print(f"\n  Building TRT-LLM engine...")
        print(f"    dtype={dtype}, quantization={quantization}")
        print(f"    max_batch_size={max_batch_size}, max_num_tokens={max_num_tokens}")
        print(f"    multiple_profiles={multiple_profiles}, gemm_plugin={gemm_plugin}")
        print(f"    paged_context_fmha={use_paged_context_fmha}, reduce_fusion={reduce_fusion}")

        cmd = [
            "trtllm-build",
            "--checkpoint_dir", checkpoint_dir,
            "--output_dir", engine_dir,
            "--max_batch_size", str(max_batch_size),
            "--max_num_tokens", str(max_num_tokens),
            "--max_seq_len", str(max_seq_len),
        ]

        # Attention plugin
        cmd.extend(["--gpt_attention_plugin", "float16"])

        # GEMM plugin (auto for BF16/FP16, disabled for FP8)
        if gemm_plugin != "disabled":
            cmd.extend(["--gemm_plugin", gemm_plugin])

        # Multiple profiles (always recommended for production)
        if multiple_profiles:
            cmd.append("--multiple_profiles")

        # Paged context FMHA
        if use_paged_context_fmha:
            cmd.append("--use_paged_context_fmha")

        # Reduce fusion (only beneficial with TP > 1)
        if reduce_fusion and tp_size > 1:
            cmd.extend(["--reduce_fusion", "enable"])

        # Quantization-specific flags
        if quantization == "awq":
            cmd.extend(["--awq_quantized_fp16", "enable"])

        run_cmd(cmd)

    print(f"\n  Done! Engine at: {engine_dir}")
    print(f"  Engine size: {sum(f.stat().st_size for f in Path(engine_dir).rglob('*') if f.is_file()) / 1e9:.2f} GB")
    return engine_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF model to TRT-LLM engine (v0.21+)")
    parser.add_argument("--model_dir", required=True, help="HuggingFace model directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for TRT-LLM engine")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp8"])
    parser.add_argument("--quantization", default=None, help="Quantization method (awq, gptq)")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument("--max_num_tokens", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--multiple_profiles", action="store_true", default=True)
    parser.add_argument("--no_multiple_profiles", dest="multiple_profiles", action="store_false")
    parser.add_argument("--gemm_plugin", default="auto", choices=["auto", "disabled", "float16"])
    parser.add_argument("--paged_context_fmha", action="store_true", default=True)
    parser.add_argument("--no_paged_context_fmha", dest="paged_context_fmha", action="store_false")
    parser.add_argument("--reduce_fusion", action="store_true", default=False)
    args = parser.parse_args()

    convert_model(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        quantization=args.quantization,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        max_seq_len=args.max_seq_len,
        multiple_profiles=args.multiple_profiles,
        gemm_plugin=args.gemm_plugin,
        use_paged_context_fmha=args.paged_context_fmha,
        reduce_fusion=args.reduce_fusion,
    )
