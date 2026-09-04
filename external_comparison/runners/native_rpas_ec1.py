"""Formal-capable RPAS HumanEval runner with an LLM reflector and public tests.

This is intentionally separate from the legacy common-space preparation
runner.  It uses the repository's phase-2 executor, typed LLM-directed
mutation path, and the same frozen AFlow public-test fixture exposed to the
two native external baselines.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

from external_comparison.adapters.native_common import call_record, load_jsonl, sha256_file
from external_comparison.runners.humaneval import (
    HumanEvalTask,
    _call_records,
    execute_humaneval,
    extract_code,
    load_humaneval_tasks,
    task_manifest,
)
from external_comparison.runners.public_test_executor import PublicTestExecutor
from experiments.phase2_wan_agent_search import (
    build_reflection_plan,
    choose_planned_mutations,
    configure_site_penalties,
    load_models,
    load_network_profiles,
    load_sites,
    mutate_candidate,
    run_single_architecture,
    seed_architectures,
    validate_candidate_contract,
)


def _require_selected_gpu() -> str:
    gpu = os.environ.get("RPAS_EC1_GPU", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in {"4", "5"} or visible != gpu:
        raise RuntimeError("native RPAS EC-1 requires exactly CUDA_VISIBLE_DEVICES=RPAS_EC1_GPU=4 or 5")
    return gpu


def _final_agent_name(candidate: dict[str, Any]) -> str:
    topology = candidate.get("topology")
    if topology in {"solver_verifier", "planner_solver_verifier", "debate"}:
        return "verifier"
    if topology == "dag_decompose":
        return "aggregator"
    return "solver"


def _frozen_aflow_splits(
    source: list[HumanEvalTask], *, validate_path: str | Path, test_path: str | Path
) -> dict[str, list[HumanEvalTask]]:
    """Load the exact EC-1 AFlow fixtures shared by every method.

    EC-1 defines its 33/131 split through the AFlow fixture artifacts, not a
    method-local shuffle.  Verify each fixture row against the official 164
    task source before the search executor ever receives it.
    """
    paths = {"search": Path(validate_path), "test": Path(test_path)}
    expected_counts = {"search": 33, "test": 131}
    source_by_id = {task.task_id: task for task in source}
    splits: dict[str, list[HumanEvalTask]] = {}
    seen: set[str] = set()
    for split, path in paths.items():
        fixture = load_jsonl(path)
        if len(fixture) != expected_counts[split]:
            raise ValueError(f"EC-1 frozen {split} fixture must contain {expected_counts[split]} tasks")
        tasks: list[HumanEvalTask] = []
        for index, row in enumerate(fixture):
            task_id = str(row.get("task_id", ""))
            canonical = source_by_id.get(task_id)
            if canonical is None:
                raise ValueError(f"EC-1 frozen {split} fixture has unknown task at row {index}: {task_id!r}")
            if any(row.get(field) != getattr(canonical, field) for field in ("prompt", "test", "entry_point")):
                raise ValueError(f"EC-1 frozen {split} fixture differs from official HumanEval at {task_id}")
            if task_id in seen:
                raise ValueError(f"EC-1 frozen fixtures overlap at {task_id}")
            seen.add(task_id)
            tasks.append(canonical)
        splits[split] = tasks
    if len(seen) != 164:
        raise ValueError("EC-1 frozen fixtures must partition all 164 official HumanEval tasks")
    splits["select"] = []
    return splits


def _summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [task["trace"] for task in tasks]
    public_available = [task for task in tasks if task["public_test_available"]]
    return {
        "score": sum(bool(task["passed"]) for task in tasks) / len(tasks) if tasks else 0.0,
        "correct": sum(bool(task["passed"]) for task in tasks),
        "num_examples": len(tasks),
        "total_calls": sum(int(trace.get("calls", 0)) for trace in calls),
        "total_tokens": sum(int(trace.get("total_tokens", 0)) for trace in calls),
        "total_prompt_tokens": sum(int(trace.get("prompt_tokens", 0)) for trace in calls),
        "total_completion_tokens": sum(int(trace.get("completion_tokens", 0)) for trace in calls),
        "model_errors": sum(bool(task["model_error"]) for task in tasks),
        "public_test_calls": sum(int(task["public_test_calls"]) for task in tasks),
        "public_test_repairs": sum(int(task["public_test_repairs"]) for task in tasks),
        "public_test_available_tasks": len(public_available),
        "public_test_coverage": len(public_available) / len(tasks) if tasks else 0.0,
        "public_test_pass_rate": sum(bool(task["public_test_passed"]) for task in public_available) / len(public_available) if public_available else 0.0,
    }


def _evaluate_candidate(
    *,
    candidate: dict[str, Any],
    tasks: list[HumanEvalTask],
    models: dict[str, Any],
    profile: Any,
    executor: PublicTestExecutor,
    run_id: str,
    split: str,
    capture_outputs: bool,
) -> dict[str, Any]:
    from experiments.phase2_wan_agent_search import call_agent

    rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    for task in tasks:
        example = {"id": task.task_id, "dataset": "humaneval", "input": task.prompt, "answer": ""}
        output, trace = run_single_architecture(candidate=candidate, example=example, models=models, profile=profile)
        code = extract_code(output, task.entry_point)
        public_available = executor.has_task(task.task_id)
        public = executor.run(task.task_id, task.entry_point, code) if public_available else None
        if public is not None:
            tool_events.append({"split": split, "candidate_id": candidate["id"], "attempt": 0, **public.to_dict()})
        repaired = False
        if public is not None and not public.passed:
            repair_prompt = (
                "Return a complete corrected Python implementation, with no explanation.\n\n"
                f"Task:\n{task.prompt}\n\nSubmitted implementation:\n{code or '<no parseable function>'}\n\n"
                "The shared HumanEval public-test tool rejected it with:\n"
                f"{public.feedback}\n\nRepair the implementation while preserving the required entry point "
                f"`{task.entry_point}`."
            )
            output, _ = call_agent(
                candidate=candidate,
                agent_name=_final_agent_name(candidate),
                models=models,
                trace=trace,
                user_content=repair_prompt,
                temperature=0.0,
                max_tokens=1024,
            )
            code = extract_code(output, task.entry_point)
            public = executor.run(task.task_id, task.entry_point, code)
            tool_events.append({"split": split, "candidate_id": candidate["id"], "attempt": 1, **public.to_dict()})
            repaired = True
        execution = execute_humaneval(code, task, timeout_seconds=10.0)
        calls.extend(
            _call_records(
                run_id=run_id,
                method="rpas",
                split=split,
                candidate_id=candidate["id"],
                task_id=task.task_id,
                trace=trace,
            )
        )
        row = {
            "task_id": task.task_id,
            "entry_point": task.entry_point,
            "passed": bool(execution["passed"]),
            "status": execution["status"],
            "model_error": any(call.error for call in trace.calls),
            "trace": trace.summary(profile),
            "execution": execution,
            "public_test_calls": (2 if repaired else 1) if public_available else 0,
            "public_test_available": public_available,
            "public_test_repairs": int(repaired),
            "public_test_passed": public.passed if public is not None else None,
        }
        if capture_outputs:
            row.update({"model_output": output, "code": code})
        rows.append(row)
    summary = _summary(rows)
    summary.update(
        {
            "tasks": rows,
            "calls": calls,
            "tool_events": tool_events,
            "valid": summary["model_errors"] == 0,
            "failure_examples": [
                {"id": row["task_id"], "prediction": row.get("code", ""), "answer": "", "reason": row["status"]}
                for row in rows if not row["passed"]
            ][:3],
        }
    )
    return summary


def _search_row(candidate: dict[str, Any], result: dict[str, Any], origin: str) -> dict[str, Any]:
    count = max(1, int(result["num_examples"]))
    return {
        "candidate": candidate,
        "candidate_id": candidate["id"],
        "candidate_name": candidate["name"],
        "topology": candidate["topology"],
        "candidate_origin": origin,
        "score": result["score"],
        "valid": result["valid"],
        "is_valid_candidate": result["valid"],
        "avg_calls": result["total_calls"] / count,
        "avg_total_tokens": result["total_tokens"] / count,
        "avg_cross_center_tokens": 0.0,
        "avg_network_latency_ms": 0.0,
        "avg_emulated_latency_ms": 0.0,
        "failure_examples": result["failure_examples"],
        "public_test_calls": result["public_test_calls"],
        "public_test_repairs": result["public_test_repairs"],
    }


def _reflection_calls(run_id: str, parent_id: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for index, trace in enumerate(plan.get("call_traces", [])):
        records.append(call_record(run_id, "rpas", "humaneval", "search", f"reflection:{parent_id}", index, trace))
    return records


def run(args: Any) -> dict[str, Any]:
    gpu = _require_selected_gpu()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(
        os.environ.get(
            "RPAS_EC1_RPAS_CONFIG",
            str(repo_root / "experiments" / "phase2_humaneval_qwen35_9b_single_service.json"),
        )
    ).resolve()
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE", "")
    if not endpoint:
        raise RuntimeError("RPAS_EXTERNAL_API_BASE must point to the selected GPU's resident service")
    for model in raw_config["models"].values():
        model["api_base"] = endpoint
        model["completion_kwargs"] = {"temperature": 0.0, "max_tokens": 1024}
    models = load_models(raw_config["models"])
    sites = load_sites(raw_config["sites"])
    configure_site_penalties(sites, "center_a")
    profile = load_network_profiles(raw_config["network_profiles"])["lan_homogeneous"]
    source = load_humaneval_tasks(args.dataset_path)
    validate_fixture = getattr(args, "aflow_validate_path", None)
    test_fixture = getattr(args, "aflow_test_path", None)
    if not validate_fixture or not test_fixture:
        raise RuntimeError("RPAS EC-1 requires the frozen AFlow validate and test fixtures")
    splits = _frozen_aflow_splits(
        source, validate_path=validate_fixture, test_path=test_fixture
    )
    executor = PublicTestExecutor(args.public_test_path)
    seed_budget = int(os.environ.get("RPAS_EC1_RPAS_SEED_CANDIDATES", "4"))
    new_budget = int(os.environ.get("RPAS_EC1_RPAS_NEW_CANDIDATES", "3"))
    if args.run_kind == "formal" and (
        "RPAS_EC1_RPAS_SEED_CANDIDATES" not in os.environ or "RPAS_EC1_RPAS_NEW_CANDIDATES" not in os.environ
    ):
        raise RuntimeError("formal RPAS EC-1 requires explicitly frozen seed and new-candidate budgets")
    candidates = seed_architectures(raw_config)[:seed_budget]
    if len(candidates) != seed_budget or new_budget < 1:
        raise ValueError("RPAS EC-1 requires at least one generated candidate and available seed architectures")
    for candidate in candidates:
        errors = validate_candidate_contract(candidate, raw_config, models)
        if errors:
            raise ValueError(f"invalid RPAS seed candidate {candidate['name']}: {errors}")
    rng = random.Random(args.seed)
    run_id = f"humaneval-rpas-seed-{args.seed}"
    search_rows: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    for candidate in candidates:
        result = _evaluate_candidate(
            candidate=candidate, tasks=splits["search"], models=models, profile=profile,
            executor=executor, run_id=run_id, split="search", capture_outputs=False,
        )
        search_rows.append(_search_row(candidate, result, "seed"))
        all_calls.extend(result["calls"])
        tool_events.extend(result["tool_events"])
    reflections: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    seen = {row["candidate_id"] for row in search_rows}
    while len([row for row in search_rows if row["candidate_origin"] == "generated"]) < new_budget:
        parent = max(
            (row for row in search_rows if row["valid"]),
            key=lambda row: (float(row["score"]), -float(row["avg_total_tokens"]), str(row["candidate_id"])),
            default=None,
        )
        if parent is None:
            raise RuntimeError("RPAS EC-1 has no valid parent for LLM reflection")
        plan = build_reflection_plan(
            row=parent, config=raw_config, models=models, profile=profile, reflection_mode="llm",
            reflection_model="qwen35_9b", reflection_max_tokens=1024, max_proposals=3,
        )
        if plan.get("mode") != "llm":
            raise RuntimeError("formal RPAS EC-1 forbids rule-reflection fallback")
        reflections.append({"parent_id": parent["candidate_id"], "plan": plan})
        all_calls.extend(_reflection_calls(run_id, parent["candidate_id"], plan))
        proposals = choose_planned_mutations(plan, parent["candidate"], raw_config, limit=3)
        if not proposals:
            raise RuntimeError("LLM reflection returned no applicable typed mutation")
        accepted = False
        for proposal in proposals:
            child = mutate_candidate(
                parent["candidate"], raw_config, rng, parent_row=parent, mode="wan_pareto",
                reflection_plan=plan, planned_mutation_override=proposal,
            )
            if child["id"] in seen:
                proposal_rows.append({"status": "duplicate", "candidate_id": child["id"], "proposal": proposal})
                continue
            errors = validate_candidate_contract(child, raw_config, models)
            if errors:
                proposal_rows.append({"status": "invalid", "candidate_id": child["id"], "proposal": proposal, "reasons": errors})
                continue
            result = _evaluate_candidate(
                candidate=child, tasks=splits["search"], models=models, profile=profile,
                executor=executor, run_id=run_id, split="search", capture_outputs=False,
            )
            row = _search_row(child, result, "generated")
            search_rows.append(row)
            all_calls.extend(result["calls"])
            tool_events.extend(result["tool_events"])
            proposal_rows.append({"status": "accepted", "candidate_id": child["id"], "proposal": proposal})
            seen.add(child["id"])
            accepted = True
            break
        if not accepted:
            raise RuntimeError("LLM reflection produced no unique valid typed mutation")
    selected = max(
        (row for row in search_rows if row["valid"]),
        key=lambda row: (float(row["score"]), -float(row["avg_total_tokens"]), str(row["candidate_id"])),
    )
    test_result = _evaluate_candidate(
        candidate=selected["candidate"], tasks=splits["test"], models=models, profile=profile,
        executor=executor, run_id=run_id, split="test", capture_outputs=True,
    )
    all_calls.extend(test_result["calls"])
    tool_events.extend(test_result["tool_events"])
    search_calls = [row for row in all_calls if row["split"] == "search"]
    return {
        "manifest": {
            "implementation_status": "repository_phase2_llm_reflection_typed_mutation_public_test_executor",
            "native_search": "llm_reflection_only__no_rule_fallback",
            "gpu": gpu,
            "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "public_test_fixture": str(Path(args.public_test_path).resolve()),
            "public_test_sha256": sha256_file(args.public_test_path),
            "public_test_fixture_tasks": executor.task_count,
            "public_test_fixture_coverage_over_official_humaneval": executor.task_count / len(source),
            "split_manifest": task_manifest(splits),
            "fixed_split": "aflow_validate_test_fixtures",
            "search_fixture_source": str(Path(validate_fixture).resolve()),
            "search_fixture_source_sha256": sha256_file(validate_fixture),
            "test_fixture_source": str(Path(test_fixture).resolve()),
            "test_fixture_source_sha256": sha256_file(test_fixture),
            "search_examples": len(splits["search"]),
            "test_examples": len(splits["test"]),
            "seed_candidates": seed_budget,
            "new_candidate_budget": new_budget,
            "new_candidates": len([row for row in search_rows if row["candidate_origin"] == "generated"]),
            "reflection_calls": sum(len(item["plan"].get("call_traces", [])) for item in reflections),
            "mutation_logs": sum(row["status"] == "accepted" for row in proposal_rows),
            "rule_fallbacks": 0,
            "search_calls": len(search_calls),
            "search_tokens": sum(int(row["total_tokens"]) for row in search_calls),
            "public_test_calls": len(tool_events),
            "public_test_repairs": sum(int(event["attempt"]) == 1 for event in tool_events),
            "formal_result": args.run_kind == "formal",
            "run_kind": args.run_kind,
        },
        "search_rows": search_rows,
        "proposal_rows": proposal_rows,
        "reflections": reflections,
        "test_rows": test_result["tasks"],
        "calls": all_calls,
        "tool_events": tool_events,
        "selected": selected,
    }
