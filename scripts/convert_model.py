#!/usr/bin/env python3
"""
Convert HuggingFace model to TensorRT-LLM engine format.
Supports BF16, FP8, AWQ, and GPTQ quantization.
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
        print(f"  STDERR: {result.stderr}")
        sys.exit(1)
    return result


def convert_model(
    model_dir: str,
    output_dir: str,
    dtype: str = "bf16",
    quantization: str = None,
    tp_size: int = 1,
    max_batch_size: int = 128,
    max_seq_len: int = 4096,
):
    """Convert HF model to TRT-LLM checkpoint, then build engine."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Convert HF to TRT-LLM checkpoint
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    if os.path.exists(checkpoint_dir):
        print(f"  Checkpoint exists at {checkpoint_dir}, skipping conversion.")
    else:
        cmd = [
            sys.executable, "-m", "tensorrt_llm.commands.convert_checkpoint",
            "--model_dir", model_dir,
            "--output_dir", checkpoint_dir,
            "--dtype", dtype,
            "--tp_size", str(tp_size),
        ]
        if quantization:
            cmd.extend(["--quantization", quantization])
        run_cmd(cmd)

    # Step 2: Build TRT-LLM engine
    engine_dir = os.path.join(output_dir, "engine")
    if os.path.exists(engine_dir):
        print(f"  Engine exists at {engine_dir}, skipping build.")
    else:
        cmd = [
            "trtllm-build",
            "--checkpoint_dir", checkpoint_dir,
            "--output_dir", engine_dir,
            "--max_batch_size", str(max_batch_size),
            "--max_seq_len", str(max_seq_len),
            "--gpt_attention_plugin", "float16",
            "--gemm_plugin", "float16",
        ]
        run_cmd(cmd)

    print(f"  Done! Engine at: {engine_dir}")
    return engine_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF model to TRT-LLM engine")
    parser.add_argument("--model_dir", required=True, help="HuggingFace model directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for TRT-LLM engine")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp8"])
    parser.add_argument("--quantization", default=None, help="Quantization method (e.g., awq, gptq)")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    args = parser.parse_args()

    convert_model(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        quantization=args.quantization,
        tp_size=args.tp_size,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
    )
