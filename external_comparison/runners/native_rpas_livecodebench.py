"""EC-1B task adapter that runs LiveCodeBench through native RPAS run_search."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from external_comparison.adapters.native_common import call_record, sha256_file, write_native_result
from external_comparison.runners.livecodebench_ec1 import (
    DATASET_REVISION, RELEASE_VERSION, LiveCodeBenchTask, evaluate_code, extract_code,
    generation_instruction, load_tasks, render_model_prompt, validate_frozen_bundle,
)
from external_comparison.runners.native_rpas_ec1 import _final_agent_name, _require_selected_gpu
from experiments.phase2_wan_agent_search import (
    aggregate_trace_summaries, call_agent, configure_site_penalties, load_models,
    load_network_profiles, load_sites, run_search, run_single_architecture,
)

ORIGINAL_RPAS_CORE_SHA256 = "1b045d2f7e36f2a577be6271893356b7bdf0b6406f5b8b031c1e2d137c300a2f"


def _trace_calls(run_id: str, split: str, candidate_id: str, task_id: str, trace: Any) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(trace.calls):
        row = call_record(run_id, "rpas", "livecodebench", split, candidate_id, index, {
            "agent": f"{task_id}:{item.agent}:{index}", "model": item.model,
            "prompt_tokens": item.prompt_tokens, "completion_tokens": item.completion_tokens,
            "total_tokens": item.total_tokens, "latency_ms": item.observed_latency_ms,
            "finish_reason": item.finish_reason, "error": item.error,
        })
        row.update(example_id=task_id, requested_max_tokens=item.requested_max_tokens)
        rows.append(row)
    return rows


def _evaluate_one(candidate: dict[str, Any], task: LiveCodeBenchTask, models: dict[str, Any],
                  profile: Any, run_id: str, split: str):
    example = {
        "id": task.question_id,
        "dataset": f"livecodebench_{task.execution_mode}",
        "input": render_model_prompt(task),
        "answer": "",
    }
    output, trace = run_single_architecture(candidate=candidate, example=example, models=models, profile=profile)
    code = extract_code(output)
    public = evaluate_code(task, code, public=True)
    events = [{"split": split, "candidate_id": candidate["id"], "task_id": task.question_id,
               "attempt": 0, **public}]
    repaired = False
    if not public["passed"]:
        prompt = (
            f"{generation_instruction(task)}\n\n"
            f"Problem:\n{render_model_prompt(task)}\n\nSubmitted program:\n{code or '<empty>'}\n\n"
            "The shared public-test evaluator rejected the program. Repair likely parsing, boundary, or algorithm "
            "errors. Do not mention the tests and do not return markdown."
        )
        output, _ = call_agent(candidate=candidate, agent_name=_final_agent_name(candidate), models=models,
                               trace=trace, user_content=prompt, temperature=0.0, max_tokens=2048)
        code = extract_code(output)
        public = evaluate_code(task, code, public=True)
        events.append({"split": split, "candidate_id": candidate["id"], "task_id": task.question_id,
                       "attempt": 1, **public})
        repaired = True
    # V4 permits shared public tests as a search-time tool, but private LCB
    # cases must never become RPAS reflection or mutation feedback. D_select
    # and D_test remain evaluator-only and may use their frozen private cases.
    if split == "search":
        scored = public
        evaluator_scope = "public_search_feedback"
    else:
        scored = evaluate_code(task, code, public=False)
        evaluator_scope = "private_selection_or_test"
    calls = _trace_calls(run_id, split, candidate["id"], task.question_id, trace)
    row = {
        "id": task.question_id, "task_id": task.question_id, "dataset": "livecodebench",
        "passed": scored["passed"], "score": float(scored["passed"]),
        "component_score": float(scored["passed"]), "status": scored["status"],
        "difficulty": task.difficulty, "platform": task.platform, "prediction": code,
        "output": output, "model_output": output, "model_error": any(x.get("error") for x in calls),
        "finish_reason_length": any(x.get("finish_reason") == "length" for x in calls),
        "answer_protocol_valid": bool(code), "public_test_passed": public["passed"],
        "public_test_calls": 2 if repaired else 1, "public_test_repairs": int(repaired),
        "evaluator_scope": evaluator_scope,
        "private_evaluator_metadata": scored["metadata"] if split != "search" else {},
        "public_evaluator_metadata": public["metadata"], "trace": trace.summary(profile),
        "call_traces": calls,
    }
    return row, calls, events


def evaluate_candidate(*, candidate: dict[str, Any], tasks: list[LiveCodeBenchTask], models: dict[str, Any],
                       profile: Any, run_id: str, split: str, concurrency: int,
                       capture_outputs: bool, reflection_example_limit: int) -> dict[str, Any]:
    def evaluate(task: LiveCodeBenchTask):
        return _evaluate_one(candidate, task, models, profile, run_id, split)
    if concurrency == 1:
        results = [evaluate(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(tasks)), thread_name_prefix="rpas-lcb") as pool:
            results = list(pool.map(evaluate, tasks))
    rows = [item[0] for item in results]
    calls = [row for item in results for row in item[1]]
    events = [row for item in results for row in item[2]]
    count = max(1, len(rows))
    execution_rate = sum(not row["model_error"] for row in rows) / count
    valid_answer_rate = sum(bool(row["prediction"]) for row in rows) / count
    truncated_unextractable = sum(row["finish_reason_length"] and not row["prediction"] for row in rows) / count
    result = aggregate_trace_summaries([row["trace"] for row in rows])
    result.update({
        "score": sum(row["score"] for row in rows) / count,
        "component_score": sum(row["component_score"] for row in rows) / count,
        "scores": [row["score"] for row in rows], "component_scores": [row["component_score"] for row in rows],
        "correct": sum(bool(row["passed"]) for row in rows), "num_examples": len(rows),
        "valid_answer_rate": valid_answer_rate, "valid_execution_rate": execution_rate,
        "error_example_rate": 1.0 - execution_rate, "unextractable_answer_rate": 1.0 - valid_answer_rate,
        "truncated_unextractable_rate": truncated_unextractable,
        "failure_examples": [{
            "id": row["task_id"],
            "input": next(task.question for task in tasks if task.question_id == row["task_id"])[:2400],
            "gold_answer": "<private LiveCodeBench tests hidden from reflection>",
            "prediction": row["prediction"][-1200:], "final_output_excerpt": row["model_output"][-1200:],
            "trace": row["trace"],
        } for row in rows if not row["passed"]][:max(0, reflection_example_limit)],
        "_adapter_calls": calls, "_adapter_tool_events": events,
    })
    if capture_outputs:
        result["outputs"] = rows
    return result


def _core_example(task: LiveCodeBenchTask, split: str) -> dict[str, Any]:
    return {"id": task.question_id, "dataset": f"livecodebench_{task.execution_mode}",
            "input": render_model_prompt(task),
            "answer": "<private LiveCodeBench tests hidden from search controller>",
            "official_split": split, "lcb_task": task.to_raw_row()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: Any) -> None:
    gpu = _require_selected_gpu()
    bundle = Path(args.data_dir).resolve()
    dataset_manifest = validate_frozen_bundle(bundle)
    tasks = {name: load_tasks(bundle / f"{name}.jsonl") for name in ("search", "select", "test")}
    examples = {name: [_core_example(task, name) for task in rows] for name, rows in tasks.items()}
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(os.environ.get("RPAS_EC1_RPAS_CONFIG",
                                      repo_root / "experiments/phase2_humaneval_qwen35_9b_single_service.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE", "")
    if not endpoint:
        raise RuntimeError("RPAS_EXTERNAL_API_BASE is required")
    for spec in config["models"].values():
        spec["api_base"] = endpoint
        spec["completion_kwargs"] = {"temperature": 0.0, "max_tokens": 2048}
    for role in config["defaults"]["agent_max_tokens"]:
        config["defaults"]["agent_max_tokens"][role] = 2048
    for values in config["search"].get("max_tokens_pool", {}).values():
        values[:] = [2048]
    for topology in config["search"].get("topology_max_tokens_pool", {}).values():
        for values in topology.values():
            values[:] = [2048]
    config.setdefault("reflection", {})["allow_rule_fallback"] = False
    models = load_models(config["models"])
    sites = load_sites(config["sites"])
    configure_site_penalties(sites, "center_a")
    profile = load_network_profiles(config["network_profiles"])["lan_homogeneous"]
    concurrency = max(1, int(os.environ.get("RPAS_EC1_LCB_CONCURRENCY", "4")))
    seed_budget = int(os.environ.get("RPAS_EC1_LCB_SEED_CANDIDATES", "4"))
    new_budget = int(os.environ.get("RPAS_EC1_LCB_NEW_CANDIDATES", "3"))
    run_id = f"livecodebench-rpas-seed-{args.seed}"
    all_calls: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    test_evaluations = 0

    def task_evaluator(**kwargs: Any) -> dict[str, Any]:
        nonlocal test_evaluations
        dataset = kwargs.pop("dataset")
        nominal_split = str(dataset[0]["official_split"])
        split = nominal_split
        if nominal_split == "test":
            split = "test" if test_evaluations == 0 else "test_efficiency"
            test_evaluations += 1
        adapted_tasks = [LiveCodeBenchTask.from_row(example["lcb_task"]) for example in dataset]
        result = evaluate_candidate(tasks=adapted_tasks, run_id=run_id, split=split,
                                    concurrency=kwargs.pop("eval_concurrency"), **kwargs)
        all_calls.extend(result.pop("_adapter_calls"))
        all_events.extend(result.pop("_adapter_tool_events"))
        return result

    output = Path(args.output_dir) / "rpas" / f"seed_{args.seed}"
    native_output = output / "native_core"
    metadata = {
        "dataset": "livecodebench", "test_name": "release_v6", "network_profile": "lan_homogeneous",
        "seed": args.seed, "data_seed": dataset_manifest["data_seed"],
        "search_size": len(examples["search"]), "selection_size": len(examples["select"]),
        "test_size": len(examples["test"]), "split_protocol": "frozen_disjoint_calib_search_select_test",
        "task_adapter": "external_comparison.runners.native_rpas_livecodebench",
    }
    core = run_search(
        config=config, models=models, profile=profile, searchset=examples["search"],
        selectionset=examples["select"], testset=examples["test"], mode="wan_pareto",
        seed_candidate_budget=seed_budget, new_candidate_budget=new_budget, search_examples=None,
        selection_shortlist_size=8, test_top_k=2, eval_concurrency=concurrency,
        output_dir=native_output, seed=args.seed, selection_strategy="quality_band_cost", quality_band=0.05,
        pareto_parent_prob=0.5, parent_score_band=0.05, parent_top_k=6, reflection_mode="llm",
        reflection_model="qwen35_9b", reflection_max_tokens=1024, reflection_children=3,
        reflection_example_limit=3, evaluation_cache_dir=None, resume=False, metadata=metadata,
        candidate_evaluator=task_evaluator,
    )
    for record in _read_jsonl(native_output / "search_overhead_rows.jsonl"):
        for index, trace in enumerate(record.get("call_traces", [])):
            usage = {**trace, "agent": "architecture_reflector",
                     "latency_ms": trace.get("observed_latency_ms", 0.0)}
            all_calls.append(call_record(run_id, "rpas", "livecodebench", "search",
                                         f"reflection:{record.get('parent_candidate_id', 'unknown')}", index, usage))
    selected_tests = core["selected_test_rows"]
    quality = selected_tests[0]
    efficiency = next(row for row in selected_tests if "E" in row.get("operating_points", []))
    selected = {"quality": quality["selection"], "efficiency": efficiency["selection"]}
    search_calls = [row for row in all_calls if row["split"] in {"search", "select"}]
    manifest = {
        "run_id": run_id, "method": "rpas", "dataset": "livecodebench", "seed": args.seed,
        "experiment": "EC-1B", "run_kind": "protocol_v4_seed0_pilot", "formal_result": False,
        "formal_result_reason": "One-seed hard-benchmark pilot; three-seed confirmatory gate remains pending",
        "release_version": RELEASE_VERSION, "dataset_revision": DATASET_REVISION,
        "dataset_manifest_sha256": sha256_file(bundle / "DATASET_MANIFEST.json"),
        "split_manifest": dataset_manifest["splits"], "gpu": gpu,
        "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "temperature": 0.0, "max_tokens": 2048, "thinking_disabled": True, "eval_concurrency": concurrency,
        "native_control_flow": "experiments.phase2_wan_agent_search.run_search",
        "native_core_sha256": sha256_file(repo_root / "experiments/phase2_wan_agent_search.py"),
        "original_rpas_core_sha256": ORIGINAL_RPAS_CORE_SHA256,
        "external_change_scope": "LiveCodeBench task evaluator injection only",
        "search_feedback_contract": "D_search public cases only; private cases restricted to D_select/D_test",
        "mode": "wan_pareto", "reflection_mode": "llm",
        "rule_fallbacks": core["search_overhead"]["rule_fallbacks"],
        "pareto_parent_prob": 0.5, "parent_score_band": 0.05, "parent_top_k": 6,
        "seed_candidates": seed_budget, "new_candidate_budget": new_budget,
        "search_pareto_front_ids": core["search_pareto_front_ids"],
        "selection_pareto_front_ids": core["selection_pareto_front_ids"],
        "selection_shortlist_policy": core["search_shortlist_policy"], "selection_split": "D_select",
        "selection_policy": core["selection_policy"], "quality_candidate_id": quality["candidate_id"],
        "efficiency_candidate_id": efficiency["candidate_id"], "search_calls": len(search_calls),
        "search_tokens": sum(int(row["total_tokens"]) for row in search_calls),
        "public_test_calls": len(all_events),
        "public_test_repairs": sum(row["attempt"] == 1 for row in all_events),
        "test_quality_score": quality["test"]["score"], "test_efficiency_score": efficiency["test"]["score"],
        "native_core_result": "native_core/result.json",
    }
    if manifest["rule_fallbacks"] != 0:
        raise RuntimeError("RPAS EC-1B forbids rule reflection fallback")
    write_native_result(output, manifest, quality["test"].get("outputs", []), all_calls,
                        selected=selected, search_rows=core["search_rows"])
    artifacts = {
        "selection_rows": core["selection_rows"], "tool_events": all_events,
        "test_outputs_efficiency": efficiency["test"].get("outputs", []),
    }
    for name, values in artifacts.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
