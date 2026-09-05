"""Small synthetic CUDA kernel audit; never loads weights or changes a live server."""

import argparse
import ast
import hashlib
import inspect
import json
import os
import statistics
import time
from pathlib import Path


def extract_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def check(server_path: Path, model_config: Path) -> dict:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        raise RuntimeError("Synthetic CUDA audit requires one explicitly allocated Slurm GPU")
    import torch
    import torch.nn.functional as functional
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as implementation

    config = json.loads(model_config.read_text())["text_config"]
    module_path = Path(inspect.getfile(implementation))
    namespace = {"torch": torch, "F": functional, "l2norm": implementation.l2norm}
    recurrent = extract_function(module_path, "torch_recurrent_gated_delta_rule", namespace)
    convolution = extract_function(server_path, "_safe_causal_conv1d_update", namespace)
    torch.manual_seed(2026)
    records = []

    def compare(name, reference, inputs, batches):
        started = time.perf_counter()
        compiled = torch.compile(reference, fullgraph=True, dynamic=False)
        for batch, args, kwargs in zip(batches, inputs[0], inputs[1], strict=True):
            def invoke(fn):
                # The convolution updates its cache in place; each comparison
                # starts from identical inputs and owns its output buffers.
                cloned = [value.clone() if isinstance(value, torch.Tensor) else value for value in args]
                output = fn(*cloned, **kwargs)
                result = list(output) if isinstance(output, tuple) else [output, cloned[1]]
                return [value.clone() if isinstance(value, torch.Tensor) else value for value in result]

            expected = invoke(reference)
            observed = invoke(compiled)
            torch.cuda.synchronize()
            differences = []
            for left, right in zip(expected, observed, strict=True):
                if left is None:
                    if right is not None:
                        raise AssertionError("Optional cache output changed")
                    continue
                differences.append(float((left.float() - right.float()).abs().max()))
                torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-3)
            # Interleaved wall timings include the existing co-resident load.
            timings = {"reference": [], "compiled": []}
            for _ in range(3):
                for label, fn in (("reference", reference), ("compiled", compiled)):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    for _ in range(10):
                        invoke(fn)
                    torch.cuda.synchronize()
                    timings[label].append((time.perf_counter() - tick) * 1000 / 10)
            old, new = (statistics.median(timings[label]) for label in ("reference", "compiled"))
            records.append({"kernel": name, "batch": batch, "max_abs_errors": differences,
                            "reference_ms": old, "compiled_ms": new, "speedup": old / new,
                            "timing_samples_ms": timings, "rtol": 2e-3, "atol": 2e-3,
                            "elapsed_including_compile_s": time.perf_counter() - started})

    with torch.inference_mode():
        batches = (1, 4)
        heads = config["linear_num_value_heads"]
        key_dim, value_dim = config["linear_key_head_dim"], config["linear_value_head_dim"]
        recurrent_args, recurrent_kwargs = [], []
        conv_args, conv_kwargs = [], []
        for batch in batches:
            def random(*shape, dtype=torch.float16):
                return torch.randn(*shape, device="cuda", dtype=dtype)
            recurrent_args.append([
                random(batch, 1, heads, key_dim), random(batch, 1, heads, key_dim),
                random(batch, 1, heads, value_dim), -random(batch, 1, heads, dtype=torch.float32).abs(),
                random(batch, 1, heads).sigmoid(), random(batch, heads, key_dim, value_dim, dtype=torch.float32), True,
            ])
            recurrent_kwargs.append({"use_qk_l2norm_in_kernel": True})
            channels = 2 * config["linear_num_key_heads"] * key_dim + heads * value_dim
            width = config["linear_conv_kernel_dim"]
            conv_args.append([random(batch, channels, 1), random(batch, channels, width), random(channels, width)])
            conv_kwargs.append({"activation": "silu"})
        compare("recurrent_delta_decode", recurrent, (recurrent_args, recurrent_kwargs), batches)
        compare("causal_conv_decode", convolution, (conv_args, conv_kwargs), batches)
    return {"status": "PASS_SYNTHETIC_KERNEL_ONLY", "formal_result": False,
            "production_ready": False, "model_calls": 0, "weights_loaded": False,
            "job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "server_source_sha256": hashlib.sha256(server_path.read_bytes()).hexdigest(),
            "transformers_source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
            "records": records,
            "limitations": ["Not a whole-model throughput benchmark", "Numerical tolerance is not bitwise equality",
                            "Token-level output parity and multi-step cache tests still required",
                            "Timings include co-resident load; no isolated latency claim"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-source", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    report = check(args.server_source, args.model_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report))
