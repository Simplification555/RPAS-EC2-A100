"""Direct Qwen reference for the frozen EC-1B LiveCodeBench test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from external_comparison.adapters.native_common import call_record, sha256_file, write_native_result
from external_comparison.runners.livecodebench_ec1 import (
    DATASET_REVISION,
    RELEASE_VERSION,
    evaluate_code,
    extract_code,
    generation_instruction,
    load_tasks,
    render_model_prompt,
    validate_frozen_bundle,
)
from external_comparison.runners.native_rpas_ec1 import _require_selected_gpu


MODEL = "Qwen/Qwen3.5-9B"
def run(args: Any) -> None:
    from openai import OpenAI

    gpu = _require_selected_gpu()
    bundle = Path(args.data_dir).resolve()
    frozen = validate_frozen_bundle(bundle)
    tasks = load_tasks(bundle / "test.jsonl")
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE", "")
    if not endpoint.startswith("http://127.0.0.1:") or not endpoint.endswith("/v1"):
        raise ValueError("Single LCB requires the allocation-local model endpoint")
    output = Path(args.output_dir) / "single" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    started = time.time()
    progress_path = output / "progress.jsonl"
    concurrency = max(1, int(os.environ.get("RPAS_EC1_LCB_CONCURRENCY", "4")))
    with OpenAI(base_url=endpoint, api_key="EMPTY", timeout=900, max_retries=0) as client, progress_path.open("x", encoding="utf-8") as progress:
        def evaluate(item: tuple[int, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            index, task = item
            call_started = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": generation_instruction(task)},
                    {"role": "user", "content": render_model_prompt(task)},
                ],
                temperature=0.0,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            usage = response.usage
            tokens = [getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
            if any(type(value) is not int or value < 0 for value in tokens) or tokens[2] != tokens[0] + tokens[1]:
                raise ValueError("missing/inconsistent model token usage")
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            if finish_reason not in {"stop", "length"}:
                raise ValueError(f"unexpected finish_reason {finish_reason!r}")
            code = extract_code(content)
            evaluation = evaluate_code(task, code, public=False)
            record = call_record(
                f"livecodebench-single-seed-{args.seed}", "single", "livecodebench", "test", "single_direct", index,
                {"agent": "single", "model": MODEL, "prompt_tokens": tokens[0], "completion_tokens": tokens[1],
                 "total_tokens": tokens[2], "latency_ms": (time.perf_counter() - call_started) * 1000,
                 "finish_reason": finish_reason},
            )
            record.update(example_id=task.question_id, requested_max_tokens=2048, temperature=0.0)
            row = {
                "task_id": task.question_id, "passed": evaluation["passed"], "status": evaluation["status"],
                "difficulty": task.difficulty, "platform": task.platform, "prediction": code,
                "model_output": content, "private_evaluator_metadata": evaluation["metadata"],
            }
            return row, record

        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="single-lcb") as pool:
            results = pool.map(evaluate, enumerate(tasks))
            for index, (row, record) in enumerate(results):
                rows.append(row)
                calls.append(record)
                progress.write(json.dumps({"row": row, "call": record}, ensure_ascii=False) + "\n")
                progress.flush()
                print(json.dumps({"completed": index + 1, "total": len(tasks), "correct": sum(item["passed"] for item in rows), "total_tokens": sum(item["total_tokens"] for item in calls)}), flush=True)
    manifest = {
        "run_id": f"livecodebench-single-seed-{args.seed}", "method": "single", "dataset": "livecodebench",
        "experiment": "EC-1B", "seed": args.seed, "run_kind": "protocol_v4_seed0_pilot", "formal_result": False,
        "formal_result_reason": "One-seed hard-benchmark pilot; three-seed confirmatory gate remains pending",
        "reference_only": True, "native_search": "none_reference_only", "search_calls": 0, "search_tokens": 0,
        "release_version": RELEASE_VERSION, "dataset_revision": DATASET_REVISION,
        "dataset_manifest_sha256": sha256_file(bundle / "DATASET_MANIFEST.json"), "split_manifest": frozen["splits"],
        "gpu": gpu, "model": MODEL, "temperature": 0.0, "max_tokens": 2048, "thinking_disabled": True,
        "eval_concurrency": concurrency,
        "test_examples": len(tasks), "public_test_calls": 0, "public_test_repairs": 0,
        "prompt_contract": "typed_lcb_functional_or_stdin_v1",
        "prompt_contract_sha256": hashlib.sha256(
            "typed_lcb_functional_or_stdin_v1".encode()
        ).hexdigest(),
        "runner_sha256": sha256_file(__file__), "started_at_epoch": started, "finished_at_epoch": time.time(),
        "rounds": 0, "truncation_rate": sum(row["finish_reason"] == "length" for row in calls) / len(calls),
    }
    write_native_result(output, manifest, rows, calls, selected={"topology": "single_direct", "search": False})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
