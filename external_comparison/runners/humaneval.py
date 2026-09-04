"""Common-space HumanEval runner.

The runner owns the benchmark boundary.  Search adapters only propose typed
architectures; they never see D_select or D_test and never run Python code.
This module is preparation infrastructure and deliberately writes
``formal_result: false`` until the repository-level gates are passed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from external_comparison.common.manifest import relative_hashes, sha256_file, sha256_json
from external_comparison.common.pareto import efficiency_operating_point, pareto_frontier, quality_operating_point
from external_comparison.common.protocol import (
    RPAS_MAX_ARCHIVE_SIZE,
    RPAS_SHARED_SEED_COUNT,
    RPAS_UNIQUE_CANDIDATE_BUDGET,
    SELECTION_SCORE_DELTA,
)
from external_comparison.common.schema import CallRecord, CandidateRecord
from external_comparison.common.telemetry import append_jsonl, read_jsonl
from experiments.phase2_wan_agent_search import (
    NetworkProfile,
    configure_site_penalties,
    load_models,
    load_network_profiles,
    load_sites,
    run_single_architecture,
    seed_architectures,
    validate_candidate_contract,
)
from experiments.search_adapters.base import CandidateObservation, Proposal
from experiments.search_adapters.common_space import CommonSpaceAdapter
from experiments.search_adapters.registry import build_adapter


@dataclass(frozen=True)
class HumanEvalTask:
    task_id: str
    prompt: str
    test: str
    entry_point: str


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(payload, dict):
            rows = payload.get("data", payload.get("tasks", []))
        else:
            rows = payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"HumanEval source must contain a list of objects: {path}")
    return rows


def load_humaneval_tasks(path: str | Path) -> list[HumanEvalTask]:
    """Load the canonical HumanEval fields without consulting answer labels."""

    source = Path(path)
    tasks: list[HumanEvalTask] = []
    seen: set[str] = set()
    for index, row in enumerate(_read_json_or_jsonl(source)):
        task_id = str(row.get("task_id", row.get("id", index)))
        prompt_value = row.get("prompt")
        test_value = row.get("test")
        entry_point_value = row.get("entry_point")
        if not (
            isinstance(prompt_value, str)
            and prompt_value.strip()
            and isinstance(test_value, str)
            and test_value.strip()
            and isinstance(entry_point_value, str)
            and entry_point_value.strip()
        ):
            raise ValueError(f"HumanEval row {task_id} lacks prompt/test/entry_point")
        if task_id in seen:
            raise ValueError(f"duplicate HumanEval task id: {task_id}")
        seen.add(task_id)
        tasks.append(HumanEvalTask(task_id, prompt_value, test_value, entry_point_value))
    if not tasks:
        raise ValueError(f"HumanEval source has no tasks: {source}")
    return tasks


def split_humaneval(
    tasks: list[HumanEvalTask],
    *,
    data_seed: int,
    search_size: int,
    select_size: int,
    test_size: int,
) -> dict[str, list[HumanEvalTask]]:
    """Create one deterministic, disjoint split manifest."""

    requested = search_size + select_size + test_size
    if requested > len(tasks):
        raise ValueError(f"requested {requested} tasks but source contains {len(tasks)}")
    ordered = list(tasks)
    import random

    random.Random(data_seed).shuffle(ordered)
    return {
        "search": ordered[:search_size],
        "select": ordered[search_size : search_size + select_size],
        "test": ordered[search_size + select_size : requested],
    }


def task_manifest(tasks: dict[str, list[HumanEvalTask]]) -> dict[str, Any]:
    ids = {split: [task.task_id for task in rows] for split, rows in tasks.items()}
    all_ids = [task_id for values in ids.values() for task_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("HumanEval split manifest contains overlapping task IDs")
    return {"task_ids": ids, "counts": {split: len(values) for split, values in ids.items()}}


def _code_candidates(output: str, entry_point: str) -> list[str]:
    fenced = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", output, flags=re.IGNORECASE | re.DOTALL)
    candidates = fenced + [output]
    function_marker = f"def {entry_point}"
    prioritized = [candidate for candidate in candidates if function_marker in candidate]
    return prioritized + [candidate for candidate in candidates if candidate not in prioritized]


def extract_code(output: str, entry_point: str) -> str:
    """Extract a Python completion while rejecting prose and missing functions."""

    for candidate in _code_candidates(output, entry_point):
        candidate = candidate.strip()
        marker = f"def {entry_point}"
        if marker in candidate:
            candidate = candidate[candidate.find(marker) :]
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point for node in tree.body):
            return candidate
    return ""


def execute_humaneval(
    code: str,
    task: HumanEvalTask,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run one completion in a fresh isolated interpreter process."""

    if not code:
        return {"passed": False, "status": "empty_or_unparseable", "stderr": ""}
    program = f"{code}\n\n{task.test}\n\ncheck({task.entry_point})\n"
    with tempfile.TemporaryDirectory(prefix="rpas_humaneval_") as directory:
        script = Path(directory) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "status": "timeout",
                "stderr": str(exc)[:1000],
                "runtime_ms": (time.perf_counter() - started) * 1000,
            }
    status = "passed" if completed.returncode == 0 else "failed"
    return {
        "passed": completed.returncode == 0,
        "status": status,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-1000:],
        "stdout": completed.stdout[-1000:],
        "runtime_ms": (time.perf_counter() - started) * 1000,
    }


def _architecture_for_cache(candidate: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(candidate)
    for key in ("id", "name", "parent_id", "mutation", "applied_mutation", "parent_reflection", "mutation_observation"):
        copied.pop(key, None)
    return copied


def _cache_key(
    candidate: dict[str, Any],
    tasks: list[HumanEvalTask],
    *,
    split: str,
    model_manifest: dict[str, Any],
    evaluator_version: str,
) -> str:
    return sha256_json(
        {
            "candidate": _architecture_for_cache(candidate),
            "task_ids": [task.task_id for task in tasks],
            "split": split,
            "models": model_manifest,
            "evaluator": evaluator_version,
        }
    )


def _call_records(
    *,
    run_id: str,
    method: str,
    split: str,
    candidate_id: str,
    task_id: str,
    trace: Any,
) -> list[dict[str, Any]]:
    records = []
    for index, call in enumerate(trace.calls):
        records.append(
            CallRecord(
                run_id=run_id,
                method=method,
                dataset="humaneval",
                split=split,
                candidate_id=candidate_id,
                agent=f"{task_id}:{call.agent}:{index}",
                model=call.model,
                site=call.site,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                total_tokens=call.total_tokens,
                input_cost=call.input_cost_usd,
                output_cost=call.output_cost_usd,
                inference_cost=call.inference_cost_usd,
                model_latency_ms=call.observed_latency_ms,
                wall_latency_ms=call.observed_latency_ms,
                error=call.error,
                finish_reason=call.finish_reason,
            ).to_dict()
        )
    return records


def _message_records(*, task_id: str, trace: Any) -> list[dict[str, Any]]:
    return [{"task_id": task_id, **asdict(message)} for message in trace.messages]


def evaluate_candidate(
    *,
    candidate: dict[str, Any],
    tasks: list[HumanEvalTask],
    models: dict[str, Any],
    profile: NetworkProfile,
    method: str,
    split: str,
    run_id: str,
    timeout_seconds: float,
    capture_outputs: bool,
    cache_dir: Path | None,
    model_manifest: dict[str, Any],
    evaluator_version: str,
) -> dict[str, Any]:
    cache_path: Path | None = None
    if cache_dir is not None and split in {"search", "select"}:
        cache_path = cache_dir / f"{_cache_key(candidate, tasks, split=split, model_manifest=model_manifest, evaluator_version=evaluator_version)}.json"
        if cache_path.exists():
            return {**json.loads(cache_path.read_text(encoding="utf-8")), "cache_status": "hit"}

    task_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for task in tasks:
        example = {"id": task.task_id, "dataset": "humaneval", "input": task.prompt, "answer": ""}
        output, trace = run_single_architecture(candidate=candidate, example=example, models=models, profile=profile)
        code = extract_code(output, task.entry_point)
        execution = execute_humaneval(code, task, timeout_seconds=timeout_seconds)
        calls.extend(_call_records(run_id=run_id, method=method, split=split, candidate_id=candidate["id"], task_id=task.task_id, trace=trace))
        messages.extend(_message_records(task_id=task.task_id, trace=trace))
        task_row = {
            "task_id": task.task_id,
            "entry_point": task.entry_point,
            "passed": execution["passed"],
            "status": execution["status"],
            "code_valid": bool(code),
            "model_error": any(call.error for call in trace.calls),
            "trace": trace.summary(profile),
            "execution": execution,
        }
        if capture_outputs:
            task_row["model_output"] = output
            task_row["code"] = code
        task_rows.append(task_row)

    summary = {
        "score": sum(bool(row["passed"]) for row in task_rows) / len(task_rows) if task_rows else 0.0,
        "pass_at_1": sum(bool(row["passed"]) for row in task_rows) / len(task_rows) if task_rows else 0.0,
        "correct": sum(bool(row["passed"]) for row in task_rows),
        "num_examples": len(task_rows),
        "total_calls": sum(int(row["trace"].get("calls", 0)) for row in task_rows),
        "total_prompt_tokens": sum(int(row["trace"].get("prompt_tokens", 0)) for row in task_rows),
        "total_completion_tokens": sum(int(row["trace"].get("completion_tokens", 0)) for row in task_rows),
        "total_tokens": sum(int(row["trace"].get("total_tokens", 0)) for row in task_rows),
        "total_cost": sum(float(row["trace"].get("inference_cost_usd", 0.0)) for row in task_rows),
        "observed_latency_ms": sum(float(row["trace"].get("observed_model_wall_latency_ms", 0.0) or 0.0) for row in task_rows),
        "cross_center_tokens": sum(int(row["trace"].get("cross_center_tokens", 0)) for row in task_rows),
        "model_errors": sum(bool(row["model_error"]) for row in task_rows),
        "tasks": task_rows,
        "calls": calls,
        "messages": messages,
        "cache_status": "miss",
    }
    summary["valid"] = summary["model_errors"] == 0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _candidate_record(
    *,
    run_id: str,
    method: str,
    split: str,
    seed: int,
    candidate: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return CandidateRecord(
        run_id=run_id,
        method=method,
        dataset="humaneval",
        split=split,
        candidate_id=candidate["id"],
        seed=seed,
        score=float(result["score"]) if result.get("valid", False) else None,
        valid=bool(result.get("valid", False)),
        total_calls=int(result.get("total_calls", 0)),
        total_tokens=int(result.get("total_tokens", 0)),
        total_cost=float(result.get("total_cost", 0.0)),
        observed_latency_ms=float(result.get("observed_latency_ms", 0.0)),
        cross_center_tokens=int(result.get("cross_center_tokens", 0)),
        invalid_reason=None if result.get("valid", False) else "model_or_runtime_error",
        architecture=_architecture_for_cache(candidate),
    ).to_dict()


def _shortlist(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("valid") and row.get("score") is not None]
    return sorted(valid, key=lambda row: (-float(row["score"]), int(row["total_tokens"]), row["candidate_id"]))[: max(1, size)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, search_rows: list[dict[str, Any]], selection_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> None:
    import csv

    output_rows = []
    for split, values in (("search", search_rows), ("select", selection_rows)):
        for row in values:
            output_rows.append({"split": split, "candidate_id": row["candidate_id"], "score": row.get("score"), "total_calls": row.get("total_calls"), "total_tokens": row.get("total_tokens"), "total_cost": row.get("total_cost"), "selected_rank": "", "operating_point": ""})
    for row in test_rows:
        result = row["test"]
        for operating_point in row["operating_points"]:
            output_rows.append({"split": "test", "candidate_id": row["candidate_id"], "score": result.get("score"), "total_calls": result.get("total_calls"), "total_tokens": result.get("total_tokens"), "total_cost": result.get("total_cost"), "selected_rank": row["selected_rank"], "operating_point": operating_point})
    if not output_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def run_experiment(
    *,
    repo_root: Path,
    dataset_path: Path,
    model_config_path: Path,
    output_dir: Path,
    method: str,
    seed: int,
    data_seed: int = 2026,
    search_size: int = 80,
    select_size: int = 40,
    test_size: int = 44,
    shortlist_size: int = 8,
    timeout_seconds: float = 10.0,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    import random

    if method not in {"random_as", "aflow_style", "adas_style", "rpas_quality", "rpas"}:
        raise ValueError(f"unsupported common-space method: {method}")
    source_tasks = load_humaneval_tasks(dataset_path)
    splits = split_humaneval(
        source_tasks,
        data_seed=data_seed,
        search_size=search_size,
        select_size=select_size,
        test_size=test_size,
    )
    split_info = task_manifest(splits)
    raw_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    models = load_models(raw_config["models"])
    profiles = load_network_profiles(raw_config["network_profiles"])
    profile_name = "lan_homogeneous"
    profile = profiles[profile_name]
    sites = load_sites(raw_config["sites"])
    configure_site_penalties(sites, raw_config.get("defaults", {}).get("orchestrator_site", "center_a"))
    run_id = f"humaneval-{method}-seed-{seed}"
    run_dir = output_dir / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_files: list[str | Path] = [
        "EXPERIMENT_PROTOCOL.md",
        "experiments/phase2_wan_agent_search.py",
        "experiments/search_adapters/base.py",
        "experiments/search_adapters/common_space.py",
        "external_comparison/common/protocol.py",
        "external_comparison/common/schema.py",
        "external_comparison/runners/humaneval.py",
    ]
    manifest = {
        "run_id": run_id,
        "method": method,
        "dataset": "humaneval",
        "data_seed": data_seed,
        "search_seed": seed,
        "protocol_version": "external-comparison-v1-common-space",
        "dataset_source": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(model_config_path),
        "protocol_sha256": sha256_file(repo_root / "EXPERIMENT_PROTOCOL.md"),
        "source_sha256": relative_hashes(repo_root, source_files),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False).stdout.strip(),
        "model_manifest": {name: vars(spec) for name, spec in models.items()},
        "split_manifest": split_info,
        "evaluator": {"name": "humaneval_pass_at_1_subprocess", "version": "v1", "timeout_seconds": timeout_seconds},
        "formal_result": False,
        "status": "dry_run" if dry_run else "prepared_execution",
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    if dry_run:
        return {"status": "dry_run", "formal_result": False, **split_info, "run_dir": str(run_dir)}

    adapter = cast(CommonSpaceAdapter, build_adapter(method))
    seeds = seed_architectures(raw_config)
    if len(seeds) != RPAS_SHARED_SEED_COUNT:
        raise ValueError(f"expected {RPAS_SHARED_SEED_COUNT} seed architectures, got {len(seeds)}")
    adapter.initialize(seeds, raw_config, random.Random(seed))
    cache_dir = output_dir / "_shared_eval_cache" / f"seed_{seed}"
    model_manifest = manifest["model_manifest"]
    calls_path = run_dir / "calls.jsonl"
    messages_path = run_dir / "messages.jsonl"
    search_path = run_dir / "search_rows.jsonl"
    selection_path = run_dir / "selection_rows.jsonl"
    proposal_path = run_dir / "proposal_rows.jsonl"
    existing_rows = read_jsonl(search_path) if resume and search_path.exists() else []
    search_rows: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = read_jsonl(proposal_path) if resume and proposal_path.exists() else []
    for row in existing_rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("resume requires full candidate records in search_rows.jsonl")
        adapter.register_candidate(candidate)
        adapter.observe(CandidateObservation(row["candidate_id"], bool(row.get("valid")), row.get("score"), int(row.get("total_calls", 0)), int(row.get("total_tokens", 0)), float(row.get("total_cost", 0.0)), diagnostics={"architecture": candidate}))
        search_rows.append(row)
    existing_ids = {row["candidate_id"] for row in search_rows}
    for candidate in seeds:
        if candidate["id"] in existing_ids:
            continue
        result = evaluate_candidate(
            candidate=candidate,
            tasks=splits["search"],
            models=models,
            profile=profile,
            method=method,
            split="search",
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            capture_outputs=False,
            cache_dir=cache_dir,
            model_manifest=model_manifest,
            evaluator_version="v1",
        )
        row = _candidate_record(run_id=run_id, method=method, split="search", seed=seed, candidate=candidate, result=result)
        row["candidate_origin"] = "seed"
        row["candidate"] = candidate
        row["evaluation"] = {key: value for key, value in result.items() if key not in {"tasks", "calls", "messages"}}
        search_rows.append(row)
        append_jsonl(search_path, row)
        for call in result.get("calls", []):
            append_jsonl(calls_path, call)
        for message in result.get("messages", []):
            append_jsonl(messages_path, message)
        adapter.observe(CandidateObservation(candidate["id"], row["valid"], row["score"], row["total_calls"], row["total_tokens"], row["total_cost"], diagnostics={"architecture": candidate}))

    accepted = sum(row.get("candidate_origin") == "generated" for row in search_rows)
    attempts = len(proposals)
    seen = {str(row["candidate_id"]) for row in search_rows} | {str(candidate["id"]) for candidate in seeds}
    while accepted < RPAS_UNIQUE_CANDIDATE_BUDGET:
        attempts += 1
        if attempts > RPAS_UNIQUE_CANDIDATE_BUDGET * 20:
            raise RuntimeError("proposal policy could not produce enough unique valid candidates")
        proposal: Proposal = adapter.propose()
        candidate = proposal.architecture
        candidate_id = str(proposal.candidate_id)
        contract_errors = validate_candidate_contract(candidate, raw_config, models)
        if candidate_id in seen:
            proposals.append({"attempt": attempts, "status": "duplicate", "candidate_id": candidate_id, "metadata": proposal.metadata})
            continue
        if contract_errors:
            proposals.append({"attempt": attempts, "status": "invalid", "candidate_id": candidate_id, "reasons": contract_errors, "metadata": proposal.metadata})
            continue
        seen.add(candidate_id)
        result = evaluate_candidate(
            candidate=candidate,
            tasks=splits["search"],
            models=models,
            profile=profile,
            method=method,
            split="search",
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            capture_outputs=False,
            cache_dir=cache_dir,
            model_manifest=model_manifest,
            evaluator_version="v1",
        )
        row = _candidate_record(run_id=run_id, method=method, split="search", seed=seed, candidate=candidate, result=result)
        row["candidate_origin"] = "generated"
        row["candidate"] = candidate
        row["evaluation"] = {key: value for key, value in result.items() if key not in {"tasks", "calls", "messages"}}
        search_rows.append(row)
        append_jsonl(search_path, row)
        for call in result.get("calls", []):
            append_jsonl(calls_path, call)
        for message in result.get("messages", []):
            append_jsonl(messages_path, message)
        adapter.observe(CandidateObservation(candidate_id, row["valid"], row["score"], row["total_calls"], row["total_tokens"], row["total_cost"], diagnostics={"architecture": candidate}))
        proposals.append({"attempt": attempts, "status": "accepted", "candidate_id": candidate_id, "metadata": proposal.metadata})
        accepted += 1

    # Native EC-1 uses one 33-task development partition.  The official
    # baselines perform their search/training and selection on this same
    # partition, so RPAS must not invent a second selection split here.
    shortlist = _shortlist(search_rows, shortlist_size)
    selection_rows: list[dict[str, Any]] = []
    if selection_path.exists():
        selection_path.unlink()
    for search_row in shortlist:
        if not splits["select"]:
            selection_rows.append({**search_row, "split": "select", "selection_source": "development_search_split"})
            append_jsonl(selection_path, selection_rows[-1])
            continue
        candidate = adapter.candidate(search_row["candidate_id"])
        result = evaluate_candidate(
            candidate=candidate,
            tasks=splits["select"],
            models=models,
            profile=profile,
            method=method,
            split="select",
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            capture_outputs=False,
            cache_dir=cache_dir,
            model_manifest=model_manifest,
            evaluator_version="v1",
        )
        row = _candidate_record(run_id=run_id, method=method, split="select", seed=seed, candidate=candidate, result=result)
        row["search_candidate_id"] = search_row["candidate_id"]
        row["candidate"] = candidate
        row["evaluation"] = {key: value for key, value in result.items() if key not in {"tasks", "calls", "messages"}}
        selection_rows.append(row)
        append_jsonl(selection_path, row)
        for call in result.get("calls", []):
            append_jsonl(calls_path, call)
        for message in result.get("messages", []):
            append_jsonl(messages_path, message)

    quality = quality_operating_point(selection_rows)
    efficiency = efficiency_operating_point(pareto_frontier(selection_rows), delta=SELECTION_SCORE_DELTA)
    if quality is None or efficiency is None:
        raise RuntimeError("no valid candidate remains on D_select")
    selected = [quality]
    if efficiency["candidate_id"] != quality["candidate_id"]:
        selected.append(efficiency)
    test_rows = []
    for rank, selected_row in enumerate(selected):
        candidate = selected_row["candidate"]
        result = evaluate_candidate(
            candidate=candidate,
            tasks=splits["test"],
            models=models,
            profile=profile,
            method=method,
            split="test",
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            capture_outputs=True,
            cache_dir=None,
            model_manifest=model_manifest,
            evaluator_version="v1",
        )
        test_row = {"selected_rank": rank, "operating_points": ["Q" if selected_row is quality else "E"], "candidate_id": candidate["id"], "candidate": candidate, "selection": selected_row, "test": result}
        test_rows.append(test_row)
        _write_json(run_dir / f"test_outputs_{candidate['id']}.json", result.get("tasks", []))
        for call in result.get("calls", []):
            append_jsonl(calls_path, call)
        for message in result.get("messages", []):
            append_jsonl(messages_path, message)

    artifact = {
        **manifest,
        "formal_result": False,
        "archive_size": len(search_rows),
        "expected_archive_size": RPAS_MAX_ARCHIVE_SIZE,
        "proposal_summary": {
            "attempts": len(proposals),
            "accepted": sum(item["status"] == "accepted" for item in proposals),
            "duplicates": sum(item["status"] == "duplicate" for item in proposals),
            "invalid": sum(item["status"] == "invalid" for item in proposals),
        },
        "operating_points": {"Q": quality, "E": efficiency},
        "search_rows": search_rows,
        "selection_rows": selection_rows,
        "selected_test_rows": test_rows,
        "test_split_accessed": True,
        "gates": {"G1_G9": "pending", "test_leakage": "runner_enforced", "candidate_budget": len(search_rows) == RPAS_MAX_ARCHIVE_SIZE},
    }
    with proposal_path.open("w", encoding="utf-8") as handle:
        for proposal_row in proposals:
            handle.write(json.dumps(proposal_row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(run_dir / "selected_quality_candidate.json", quality["candidate"])
    _write_json(run_dir / "selected_efficiency_candidate.json", efficiency["candidate"])
    _write_json(run_dir / "result.json", artifact)
    _write_json(run_dir / "search_overhead.json", {"controller_calls": 0, "controller_total_tokens": 0, "note": "adapter-only controller accounting; native LLM overhead is not claimed"})
    _write_json(run_dir / "search_checkpoint.json", {"completed_candidates": len(search_rows), "target_candidates": RPAS_MAX_ARCHIVE_SIZE, "resumable": True, "resume_supported": "--resume", "adapter_state": adapter.state_dict()})
    _write_summary_csv(run_dir / "summary.csv", search_rows, selection_rows, test_rows)
    _write_json(run_dir / "best_candidate.json", quality["candidate"])
    return {"status": "completed_preparation", "formal_result": False, "run_dir": str(run_dir), "archive_size": len(search_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or run common-space HumanEval experiments.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-config", default="experiments/phase2_wan_agent_config_qwen35_9b_homogeneous.json")
    parser.add_argument("--output-dir", default="outputs/external_comparison/humaneval")
    parser.add_argument("--method", choices=["random_as", "aflow_style", "adas_style", "rpas_quality", "rpas"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--search-size", type=int, default=80)
    parser.add_argument("--select-size", type=int, default=40)
    parser.add_argument("--test-size", type=int, default=44)
    parser.add_argument("--shortlist-size", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_experiment(
        repo_root=Path(args.repo_root).resolve(),
        dataset_path=Path(args.dataset_path).resolve(),
        model_config_path=Path(args.model_config).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        method=args.method,
        seed=args.seed,
        data_seed=args.data_seed,
        search_size=args.search_size,
        select_size=args.select_size,
        test_size=args.test_size,
        shortlist_size=args.shortlist_size,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
