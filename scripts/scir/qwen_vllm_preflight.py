"""Isolated same-token-input vLLM smoke and synthetic backend comparison."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPTS = (
    "Return only Python code for a function add(a, b) that returns their sum.",
    "Return only Python code for a function square(x) that returns x multiplied by itself.",
    "Return only Python code for a function is_even(x) that returns whether the integer x is even.",
    "Return only Python code for a function first(xs) that returns the first item or None for an empty list.",
)


def compare_batches(baseline, outputs, expected_count):
    if expected_count <= 0 or len(baseline) != expected_count or not outputs:
        raise ValueError("Missing synthetic responses")
    if any(len(batch) != expected_count for batch in outputs):
        raise ValueError("Incomplete synthetic batch")
    for row in baseline:
        usage = row["usage"]
        counts = [usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(type(count) is not int or count < 0 for count in counts) or counts[0] + counts[1] != counts[2]:
            raise ValueError("Invalid baseline token accounting")
    parity = [left["content"] == right["content"]
              for left, right in zip(baseline, outputs[-1], strict=True)]
    prompt_parity = [left["usage"]["prompt_tokens"] == right["prompt_tokens"]
                     for left, right in zip(baseline, outputs[-1], strict=True)]
    completion_parity = [left["usage"]["completion_tokens"] == right["completion_tokens"]
                         for left, right in zip(baseline, outputs[-1], strict=True)]
    finish_parity = [left["finish_reason"] == right["finish_reason"] and left["finish_reason"] in {"stop", "length"}
                     for left, right in zip(baseline, outputs[-1], strict=True)]
    stable = all(batch == outputs[0] for batch in outputs)
    return {"status": "PASS_SYNTHETIC_PARITY_ONLY"
            if all(parity + prompt_parity + completion_parity + finish_parity) and stable else "BACKEND_DIFFERENCE",
            "exact_text_parity": parity, "prompt_token_count_parity": prompt_parity,
            "completion_token_count_parity": completion_parity, "finish_reason_parity": finish_parity,
            "vllm_repetition_stable": stable}


def baseline_call(base_url, prompt):
    payload = {"model": "Qwen/Qwen3.5-9B", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": 128,
               "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(base_url + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.load(response)
    return {"content": data["choices"][0]["message"]["content"], "usage": data["usage"],
            "finish_reason": data["choices"][0]["finish_reason"],
            "latency_s": time.perf_counter() - started}


def check(model, baseline_url):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        raise RuntimeError("Preflight requires one explicit Slurm GPU")
    import torch
    import transformers
    import vllm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
    token_ids = []
    for prompt in PROMPTS:
        inputs = processor.apply_chat_template(
            [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt", processor_kwargs={"padding": True})
        token_ids.append(inputs.input_ids[0][inputs.attention_mask[0].bool()].tolist())
    print("SYNTHETIC_TOKEN_INPUTS_READY", flush=True)
    started = time.perf_counter()
    engine = None
    try:
        engine = LLM(model=str(model), dtype="float16", tensor_parallel_size=1,
                     gpu_memory_utilization=0.35, max_model_len=8192, max_num_seqs=4,
                     max_num_batched_tokens=2048, enforce_eager=True, enable_prefix_caching=False,
                     limit_mm_per_prompt={"image": 0, "video": 0}, mm_processor_cache_gb=0,
                     disable_log_stats=True, seed=2026)
        startup_s = time.perf_counter() - started
        print(f"VLLM_LOADED startup_s={startup_s:.3f}", flush=True)
        parameters = SamplingParams(temperature=0.0, max_tokens=128, stop=["<<RPAS_END>>"])
        prompts = [{"prompt_token_ids": ids} for ids in token_ids]
        elapsed, outputs = [], []
        for repetition in range(3):
            tick = time.perf_counter()
            generated = engine.generate(prompts, parameters, use_tqdm=False)
            elapsed.append(time.perf_counter() - tick)
            outputs.append([{"content": row.outputs[0].text, "completion_tokens": len(row.outputs[0].token_ids),
                             "finish_reason": row.outputs[0].finish_reason,
                             "prompt_tokens": len(row.prompt_token_ids)} for row in generated])
            print(f"VLLM_BATCH repetition={repetition} seconds={elapsed[-1]:.3f}", flush=True)
        tick = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            baseline = list(pool.map(lambda prompt: baseline_call(baseline_url, prompt), PROMPTS))
        baseline_s = time.perf_counter() - tick
        comparison = compare_batches(baseline, outputs, len(PROMPTS))
        return {"formal_result": False, "production_ready": False, "synthetic_only": True,
                **comparison,
                "job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__, "transformers": transformers.__version__, "vllm": vllm.__version__,
                "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
                "temperature": 0.0, "max_tokens": 128, "max_model_len": 8192, "enforce_eager": True,
                "limit_mm_per_prompt": {"image": 0, "video": 0}, "mm_processor_cache_gb": 0,
                "gpu_memory_utilization": 0.35, "baseline_url": baseline_url,
                "startup_s": startup_s, "vllm_batch_seconds": elapsed, "baseline_batch_seconds": baseline_s,
                "observed_batch_speedup": baseline_s / statistics.median(elapsed[1:]),
                "prompts": list(PROMPTS),
                "prompt_token_ids": token_ids, "baseline": baseline, "vllm_outputs": outputs,
                "limitations": ["Only four synthetic prompts; not formal backend equivalence",
                                "Timing is co-resident and includes baseline queueing",
                                "Preflight context limit 8192 is not a proposed formal limit",
                                "No live experiment endpoint or client was changed"]}
    finally:
        if engine is not None:
            core = getattr(engine.llm_engine, "engine_core", None)
            shutdown = getattr(core, "shutdown", None)
            if callable(shutdown):
                shutdown()
            del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    report = check(args.model, args.baseline_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in
                      {"prompt_token_ids", "prompts", "baseline", "vllm_outputs"}}))
    raise SystemExit(0 if report["status"] == "PASS_SYNTHETIC_PARITY_ONLY" else 2)
