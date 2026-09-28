#!/usr/bin/env python3
"""Local-only environment check. Does not download weights or start inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_SERVE_FLAGS = (
    "--served-model-name", "--max-model-len", "--max-num-seqs",
    "--tensor-parallel-size", "--gpu-memory-utilization",
    "--max-num-batched-tokens", "--enable-prefix-caching",
    "--reasoning-parser", "--language-model-only",
)


def fail(message: str) -> None:
    raise SystemExit(f"PREFLIGHT_FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("aflow", "maas", "all"), default="all")
    args = parser.parse_args()

    model_path = Path(os.environ.get("RPAS_MODEL_PATH", "")).expanduser()
    if not model_path.is_dir():
        fail("set RPAS_MODEL_PATH to an existing local Qwen3.5-9B model directory (no implicit download)")
    if args.method in ("maas", "all"):
        embedding = Path(os.environ.get("RPAS_MAAS_EMBEDDING_PATH", "")).expanduser()
        if not embedding.is_dir():
            fail("MaAS needs RPAS_MAAS_EMBEDDING_PATH pointing to local all-MiniLM-L6-v2 weights")
        if not any((embedding / name).is_file() for name in (
            "modules.json", "config.json", "sentence_bert_config.json", "tokenizer.json"
        )):
            fail(f"embedding directory has no recognizable local model configuration: {embedding}")

    try:
        import torch
        import transformers
        import vllm
        import openai
        if args.method in ("maas", "all"):
            import sentence_transformers  # noqa: F401
    except Exception as exc:
        fail(f"Python runtime import failed: {type(exc).__name__}: {exc}")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        fail("PyTorch cannot access a CUDA GPU in this environment")
    if torch.cuda.device_count() != 1:
        fail(f"expected exactly one visible GPU for TP=1, found {torch.cuda.device_count()}; set CUDA_VISIBLE_DEVICES")

    device = torch.cuda.get_device_properties(0)
    try:
        from transformers import AutoConfig, AutoTokenizer
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    except Exception as exc:
        fail(f"local model config/tokenizer load failed: {type(exc).__name__}: {exc}")
    text_config = getattr(config, "text_config", config)
    max_positions = getattr(text_config, "max_position_embeddings", None)
    if max_positions is not None and int(max_positions) < 8192:
        fail(f"model config supports only {max_positions} positions, below required context 8192")

    vllm_bin = os.environ.get("RPAS_VLLM", "vllm")
    try:
        help_result = subprocess.run(
            [vllm_bin, "serve", "--help"], check=True, capture_output=True, text=True, timeout=90
        )
    except Exception as exc:
        fail(f"could not inspect vLLM serve options via {vllm_bin!r}: {type(exc).__name__}: {exc}")
    help_text = help_result.stdout + help_result.stderr
    missing = [flag for flag in REQUIRED_SERVE_FLAGS if flag not in help_text]
    if missing:
        fail("installed vLLM lacks required server flags: " + ", ".join(missing))

    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "transformers": transformers.__version__,
        "vllm": getattr(vllm, "__version__", "unknown"),
        "openai": getattr(openai, "__version__", "unknown"),
        "gpu": device.name,
        "compute_capability": f"{device.major}.{device.minor}",
        "gpu_memory_gib": round(device.total_memory / (1024**3), 2),
        "visible_gpu_count": torch.cuda.device_count(),
        "model_path": str(model_path.resolve()),
        "model_context_capacity": max_positions,
        "required_context": 8192,
        "required_output": 6144,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "method": args.method,
        "note": "Environment-only preflight; the actual vLLM load and one-request smoke test remain required.",
    }
    print(json.dumps(report, indent=2))
    print("GPU_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
