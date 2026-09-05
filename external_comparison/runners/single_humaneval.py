"""Frozen direct-call HumanEval reference, without search or test-driven repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from external_comparison.adapters.native_common import call_record, sha256_file, write_native_result
from external_comparison.runners.humaneval import execute_humaneval, extract_code, load_humaneval_tasks, task_manifest
from external_comparison.runners.native_rpas_ec1 import _frozen_aflow_splits, _require_selected_gpu

MODEL = "Qwen/Qwen3.5-9B"
SYSTEM_PROMPT = "Return only a complete Python implementation of the requested function, including necessary imports."


def evaluate_tasks(tasks, client, seed: int, output: Path) -> tuple[list[dict], list[dict]]:
    rows, calls = [], []
    # Exclusive creation prevents an interrupted run being silently overwritten.
    with (output / "progress.jsonl").open("x", encoding="utf-8") as progress:
        for index, task in enumerate(tasks):
            started = time.perf_counter()
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.prompt}]
            try:
                response = client.chat.completions.create(
                    model=MODEL, messages=messages, temperature=0.0, max_tokens=1024,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                usage = response.usage
                tokens = [getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
                if any(type(value) is not int or value < 0 for value in tokens) or tokens[2] != tokens[0] + tokens[1]:
                    raise ValueError("Missing or inconsistent model token usage; estimation is forbidden")
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
                if finish_reason not in {"stop", "length"}:
                    raise ValueError(f"Unexpected completion finish reason: {finish_reason!r}")
            except Exception as exc:
                progress.write(json.dumps({"task_id": task.task_id, "error": str(exc), "formal_result": False}) + "\n")
                progress.flush()
                raise
            latency_ms = (time.perf_counter() - started) * 1000
            record = call_record(f"humaneval-single-seed-{seed}", "single", "humaneval", "test", "single_direct", index, {
                "agent": "single", "model": MODEL, "prompt_tokens": tokens[0], "completion_tokens": tokens[1],
                "total_tokens": tokens[2], "latency_ms": latency_ms, "finish_reason": finish_reason,
            })
            record.update(seed=seed, phase="test", example_id=task.task_id, task_id=task.task_id,
                          round=0, requested_max_tokens=1024, temperature=0.0, token_usage_source="model_response")
            code = extract_code(content, task.entry_point)
            execution = execute_humaneval(code, task, timeout_seconds=10)
            row = {"task_id": task.task_id, "entry_point": task.entry_point, "passed": execution["passed"],
                   "code_valid": bool(code), "status": execution["status"], "model_output": content,
                   "code": code, "execution": execution, "model_error": False}
            rows.append(row)
            calls.append(record)
            progress.write(json.dumps({"row": row, "call": record}, ensure_ascii=False) + "\n")
            progress.flush()
            print(json.dumps({"seed": seed, "completed": index + 1, "total": len(tasks),
                              "total_tokens": sum(call["total_tokens"] for call in calls)}), flush=True)
    return rows, calls


def run(args: Any) -> None:
    from openai import OpenAI

    gpu = _require_selected_gpu()
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE", "")
    if not endpoint.startswith("http://127.0.0.1:") or not endpoint.endswith("/v1"):
        raise ValueError("Single reference requires the selected allocation's loopback model endpoint")
    source = load_humaneval_tasks(args.dataset_path)
    splits = _frozen_aflow_splits(source, validate_path=args.aflow_validate_path, test_path=args.aflow_test_path)
    output = Path(args.output_dir) / "single" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = {
        "run_id": f"humaneval-single-seed-{args.seed}", "method": "single", "dataset": "humaneval",
        "seed": args.seed, "data_seed": 2026, "gpu": gpu, "model": MODEL,
        "reference_only": True, "native_search": "none_reference_only", "search_calls": 0, "search_tokens": 0,
        "search_examples": 0, "test_examples": 131, "public_test_calls": 0, "public_test_repairs": 0,
        "max_tokens": 1024, "temperature": 0.0, "thinking_disabled": True,
        "timeout_seconds": 600, "transport_max_retries": 0,
        "seed_semantics": "independent deterministic reference repetition; no architecture-search randomness",
        "code_extractor_version": "preserve_complete_program_v2",
        "split_manifest": task_manifest(splits), "dataset_sha256": sha256_file(args.dataset_path),
        "search_fixture_source_sha256": sha256_file(args.aflow_validate_path),
        "test_fixture_source_sha256": sha256_file(args.aflow_test_path),
        "public_test_sha256": sha256_file(args.public_test_path),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "runner_sha256": sha256_file(__file__),
        "formal_result": False, "formal_result_reason": "Pending full completion and cross-method protocol audit",
    }
    (output / "started_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with OpenAI(base_url=endpoint, api_key="EMPTY", timeout=600, max_retries=0) as client:
        rows, calls = evaluate_tasks(splits["test"], client, args.seed, output)
    manifest.update(
        run_kind="formal", formal_result=True,
        formal_result_reason="Complete direct reference; paper eligibility still requires the cross-method protocol audit",
        started_at_epoch=started, finished_at_epoch=time.time(), rounds=0,
        valid_answer_rate=sum(row["code_valid"] for row in rows) / len(rows),
        truncation_rate=sum(call["finish_reason"] == "length" for call in calls) / len(calls),
    )
    write_native_result(output, manifest, rows, calls, selected={"topology": "single_direct", "search": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--aflow-validate-path", required=True)
    parser.add_argument("--aflow-test-path", required=True)
    parser.add_argument("--public-test-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
