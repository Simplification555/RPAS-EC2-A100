"""Locked EC-3 HotpotQA runner for RPAS and the single-agent reference.

Search and selection are deliberately separate from test execution.  This
module never opens the held-out fixture during ``calibration`` or ``pretest``;
``test`` first requires the six immutable states accepted by ec3_formal_gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from external_comparison.adapters.native_common import call_record, sha256_file
from external_comparison.runners.ec3_formal_gate import freeze_state
from external_comparison.runners.ec3_preflight import preflight
from external_comparison.runners.hotpotqa_ec3_data import HotpotExample, answer_scores, render_prompt
from experiments.phase2_wan_agent_search import (
    build_reflection_plan,
    candidate_validity,
    choose_planned_mutations,
    configure_site_penalties,
    evaluate_candidate,
    load_models,
    load_network_profiles,
    load_sites,
    mutate_candidate,
    pareto_front,
    scientific_config_payload,
    seed_architectures,
    sha256_json,
    validate_candidate_contract,
)


PROTOCOL_VERSION = "EC3_HOTPOTQA_V3"
MIN_VALID_ANSWER_RATE = 0.99
MAX_TRUNCATION_RATE = 0.01


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_one_allocated_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError("EC-3 requires exactly one Slurm-allocated CUDA_VISIBLE_DEVICES token")
    return visible


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _load_split(manifest: dict[str, Any], name: str) -> list[HotpotExample]:
    details = manifest.get("splits", {}).get(name, {})
    path = Path(str(details.get("path", "")))
    if not path.is_file() or sha256_file(path) != details.get("sha256"):
        raise ValueError(f"EC-3 {name} fixture differs from frozen manifest")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(details.get("count", -1)):
        raise ValueError(f"EC-3 {name} fixture count differs from frozen manifest")
    examples = [HotpotExample(**row) for row in rows]
    if len({row.task_id for row in examples}) != len(examples):
        raise ValueError(f"EC-3 {name} fixture has duplicate IDs")
    return examples


def _generic_examples(rows: list[HotpotExample]) -> list[dict[str, Any]]:
    return [
        {
            "id": row.task_id,
            "dataset": "hotpotqa",
            "input": render_prompt(row),
            "answer": row.answer,
        }
        for row in rows
    ]


def _runtime(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    raw_config = _read_json(config_path)
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE", "").strip()
    if not endpoint:
        raise RuntimeError("RPAS_EXTERNAL_API_BASE must point to this worker's local model service")
    executor_max_tokens = int(os.environ.get("RPAS_EC3_EXECUTOR_MAX_TOKENS", "256"))
    if executor_max_tokens not in {256, 512}:
        raise ValueError("EC-3 executor cap must be a calibrated 256 or 512")
    for model in raw_config["models"].values():
        model["api_base"] = endpoint
        model["completion_kwargs"] = {"temperature": 0.0, "max_tokens": executor_max_tokens}
    models = load_models(raw_config["models"])
    sites = load_sites(raw_config["sites"])
    orchestrator = str(raw_config["defaults"].get("orchestrator_site", "center_a"))
    configure_site_penalties(sites, orchestrator)
    profile = load_network_profiles(raw_config["network_profiles"])["lan_homogeneous"]
    return raw_config, models, profile


def _call_rows(
    *, run_id: str, split: str, candidate_id: str, outputs: list[dict[str, Any]], method: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example_index, output in enumerate(outputs):
        for call_index, trace in enumerate(output.get("call_traces", [])):
            usage = dict(trace)
            usage["latency_ms"] = float(usage.get("observed_latency_ms", 0.0))
            record = call_record(run_id, method, "hotpotqa", split, candidate_id, call_index, usage)
            record["example_id"] = output["id"]
            record["example_index"] = example_index
            rows.append(record)
    return rows


def _reflection_call_rows(run_id: str, candidate_id: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, trace in enumerate(plan.get("call_traces", [])):
        usage = dict(trace)
        usage["latency_ms"] = float(usage.get("observed_latency_ms", 0.0))
        record = call_record(run_id, "rpas", "hotpotqa", "search_reflection", candidate_id, index, usage)
        record["example_id"] = f"reflection:{candidate_id}"
        rows.append(record)
    return rows


def _evaluate(
    *, candidate: dict[str, Any], rows: list[HotpotExample], models: dict[str, Any], profile: Any,
    config: dict[str, Any], split: str, run_id: str, method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = evaluate_candidate(
        candidate=candidate,
        dataset=_generic_examples(rows),
        models=models,
        profile=profile,
        capture_outputs=True,
        eval_concurrency=1,
        reflection_example_limit=3,
    )
    outputs = result.pop("outputs", [])
    answer_metrics = [answer_scores(str(row.get("prediction", "")), str(row.get("answer", ""))) for row in outputs]
    result["answer_f1"] = float(result["score"])
    result["answer_em"] = sum(metric["em"] for metric in answer_metrics) / len(answer_metrics) if answer_metrics else 0.0
    total_calls = int(result.get("sum_calls", 0))
    maxed_calls = int(result.get("sum_maxed_calls", 0))
    result["generation_truncation_rate"] = maxed_calls / total_calls if total_calls else 1.0
    validity = candidate_validity(
        result, contract_errors=validate_candidate_contract(candidate, config, models)
    )
    reasons = list(validity["invalid_reasons"])
    if float(result["valid_answer_rate"]) < MIN_VALID_ANSWER_RATE:
        reasons.append(f"ec3_valid_answer_rate<{MIN_VALID_ANSWER_RATE:g}")
    if float(result["generation_truncation_rate"]) >= MAX_TRUNCATION_RATE:
        reasons.append(f"ec3_generation_truncation_rate>={MAX_TRUNCATION_RATE:g}")
    result.update(validity)
    result["invalid_reasons"] = sorted(set(reasons))
    result["is_valid_candidate"] = not result["invalid_reasons"]
    result["split"] = split
    result["outputs"] = outputs
    return result, _call_rows(
        run_id=run_id, split=split, candidate_id=str(candidate["id"]), outputs=outputs, method=method
    )


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"outputs", "scores", "component_scores"}}


def _candidate_row(candidate: dict[str, Any], result: dict[str, Any], origin: str) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "candidate_id": candidate["id"],
        "candidate_name": candidate["name"],
        "topology": candidate["topology"],
        "candidate_origin": origin,
        **_compact_result(result),
    }


def _shortlist(rows: list[dict[str, Any]], maximum: int = 5) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get("is_valid_candidate")]
    if not eligible:
        raise RuntimeError("EC-3 RPAS has no valid D_search candidate")
    selected: list[dict[str, Any]] = []
    for row in sorted(eligible, key=lambda item: (-float(item["answer_f1"]), str(item["candidate_id"])))[:3]:
        selected.append(row)
    for row in sorted(pareto_front(eligible), key=lambda item: (float(item["avg_total_tokens"]), str(item["candidate_id"]))):
        if row["candidate_id"] not in {item["candidate_id"] for item in selected}:
            selected.append(row)
        if len(selected) >= maximum:
            break
    return selected[:maximum]


def _select(selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in selection_rows if row.get("is_valid_candidate")]
    if not valid:
        raise RuntimeError("EC-3 RPAS has no valid D_select finalist")
    return min(
        valid,
        key=lambda row: (-float(row["answer_f1"]), float(row["avg_total_tokens"]), float(row["avg_calls"]), str(row["candidate_id"])),
    )


def _manifest_base(manifest: dict[str, Any], *, method: str, seed: int, config: dict[str, Any], gpu: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "hotpotqa",
        "dataset_setting": "HotpotQA distractor / provided-context",
        "method": method,
        "seed": seed,
        "data_seed": manifest["data_seed"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "split_protocol": "calib__search__select__test_locked",
        "d_test_accessed": False,
        "executor_model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "executor_max_tokens": int(os.environ.get("RPAS_EC3_EXECUTOR_MAX_TOKENS", "256")),
        "meta_max_tokens": int(config["reflection"]["max_tokens"]),
        "temperature": 0.0,
        "runtime_cuda_visible_devices": gpu,
        "config_sha256": sha256_json(scientific_config_payload(config)),
    }


def run_calibration(args: argparse.Namespace) -> Path:
    gpu = _require_one_allocated_gpu()
    manifest = _read_json(args.manifest)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("EC-3 requires a V3 frozen manifest")
    raw_config, models, profile = _runtime(Path(args.config))
    calibration = _load_split(manifest, "calib")
    seeds = seed_architectures(raw_config)[:2]
    if len(seeds) != 2:
        raise RuntimeError("EC-3 calibration requires two RPAS seed workflows")
    root = Path(args.output_root) / "calibration" / "rpas"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite EC-3 calibration artifact: {root}")
    run_id = "ec3-hotpotqa-rpas-calibration"
    rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for candidate in seeds:
        errors = validate_candidate_contract(candidate, raw_config, models)
        if errors:
            raise ValueError(f"invalid RPAS calibration candidate: {errors}")
        result, call_rows = _evaluate(candidate=candidate, rows=calibration, models=models, profile=profile, config=raw_config, split="calib", run_id=run_id, method="rpas")
        rows.append(_candidate_row(candidate, result, "seed"))
        calls.extend(call_rows)
        _append_jsonl(root / "calibration_outputs.jsonl", result["outputs"])
    truncation = sum(int(row.get("maxed_calls", 0)) for row in calls) / len(calls) if calls else 1.0
    valid_rate = sum(float(row["valid_answer_rate"]) for row in rows) / len(rows)
    payload = {
        **_manifest_base(manifest, method="rpas", seed=-1, config=raw_config, gpu=gpu),
        "run_kind": "calibration", "d_test_accessed": False, "calibration_candidates": len(rows),
        "executor_generation_truncation_rate": truncation, "valid_answer_rate": valid_rate,
        "status": "passed" if truncation < MAX_TRUNCATION_RATE and valid_rate >= MIN_VALID_ANSWER_RATE else "failed",
    }
    _write_json(root / "calibration_manifest.json", payload)
    _append_jsonl(root / "calibration_rows.jsonl", rows)
    _append_jsonl(root / "calls.jsonl", calls)
    if payload["status"] != "passed":
        raise RuntimeError("EC-3 RPAS calibration failed; do not start formal search")
    return root


def run_pretest(args: argparse.Namespace) -> Path:
    gpu = _require_one_allocated_gpu()
    manifest = _read_json(args.manifest)
    raw_config, models, profile = _runtime(Path(args.config))
    search = _load_split(manifest, "search")
    select = _load_split(manifest, "select")
    root = Path(args.output_root) / "rpas" / f"seed_{args.seed}"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to reuse a frozen EC-3 search directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"ec3-hotpotqa-rpas-seed-{args.seed}"
    rng = random.Random(args.seed)
    candidates = seed_architectures(raw_config)[:4]
    if len(candidates) != 4:
        raise RuntimeError("EC-3 RPAS requires four seed candidates")
    evaluated: list[dict[str, Any]] = []
    search_calls: list[dict[str, Any]] = []
    all_outputs: list[dict[str, Any]] = []
    for candidate in candidates:
        errors = validate_candidate_contract(candidate, raw_config, models)
        if errors:
            raise ValueError(f"invalid RPAS seed candidate: {errors}")
        result, call_rows = _evaluate(candidate=candidate, rows=search, models=models, profile=profile, config=raw_config, split="search", run_id=run_id, method="rpas")
        evaluated.append(_candidate_row(candidate, result, "seed"))
        search_calls.extend(call_rows)
        all_outputs.extend(result["outputs"])
    seed_archive_size = len(pareto_front(evaluated))
    mutation_logs: list[dict[str, Any]] = []
    generated = 0
    while generated < 3:
        parent = min(evaluated, key=lambda row: (-float(row["answer_f1"]), str(row["candidate_id"])))
        started = time.perf_counter()
        plan = build_reflection_plan(
            row=parent, config=raw_config, models=models, profile=profile, reflection_mode="llm",
            reflection_model="qwen35_9b", reflection_max_tokens=int(raw_config["reflection"]["max_tokens"]), max_proposals=3,
        )
        if plan.get("mode") != "llm":
            raise RuntimeError("EC-3 RPAS forbids rule-based reflection fallback")
        reflection_calls = _reflection_call_rows(run_id, parent["candidate_id"], plan)
        search_calls.extend(reflection_calls)
        proposals = choose_planned_mutations(plan, parent["candidate"], raw_config, limit=3)
        if not proposals:
            raise RuntimeError("EC-3 RPAS reflection returned no applicable typed mutation")
        enqueued = False
        for proposal in proposals:
            if generated >= 3:
                break
            child = mutate_candidate(
                parent["candidate"], raw_config, rng, parent_row=parent, mode="wan_pareto",
                reflection_plan=plan, planned_mutation_override=proposal,
            )
            errors = validate_candidate_contract(child, raw_config, models)
            mutation_log = {
                "parent_candidate_id": parent["candidate_id"], "child_candidate_id": child["id"],
                "applied_mutation": child.get("applied_mutation"), "reflection_mode": plan["mode"],
                "reflection_wall_time_ms": (time.perf_counter() - started) * 1000, "contract_errors": errors,
            }
            if errors or any(child["id"] == row["candidate_id"] for row in evaluated):
                mutation_log["status"] = "rejected"
                mutation_logs.append(mutation_log)
                continue
            result, call_rows = _evaluate(candidate=child, rows=search, models=models, profile=profile, config=raw_config, split="search", run_id=run_id, method="rpas")
            evaluated.append(_candidate_row(child, result, "generated"))
            search_calls.extend(call_rows)
            all_outputs.extend(result["outputs"])
            mutation_log["status"] = "evaluated"
            mutation_logs.append(mutation_log)
            generated += 1
            enqueued = True
        if not enqueued:
            raise RuntimeError("EC-3 RPAS could not materialize an LLM-planned valid mutation")
    finalists = _shortlist(evaluated)
    selection_rows: list[dict[str, Any]] = []
    selection_calls: list[dict[str, Any]] = []
    for row in finalists:
        result, call_rows = _evaluate(candidate=row["candidate"], rows=select, models=models, profile=profile, config=raw_config, split="select", run_id=run_id, method="rpas")
        selection_rows.append(_candidate_row(row["candidate"], result, row["candidate_origin"]))
        selection_calls.extend(call_rows)
        all_outputs.extend(result["outputs"])
    selected = _select(selection_rows)
    selected_payload = {
        "candidate": selected["candidate"], "candidate_id": selected["candidate_id"],
        "selection_answer_f1": selected["answer_f1"], "selection_answer_em": selected["answer_em"],
        "selection_policy": "max_d_select_f1__min_tokens__min_calls__candidate_id",
        "finalists": [row["candidate_id"] for row in finalists],
    }
    _write_json(root / "selected_candidate.json", selected_payload)
    _append_jsonl(root / "search_rows.jsonl", evaluated)
    _append_jsonl(root / "selection_rows.jsonl", selection_rows)
    _append_jsonl(root / "search_outputs.jsonl", all_outputs)
    _append_jsonl(root / "mutation_logs.jsonl", mutation_logs)
    _append_jsonl(root / "calls.jsonl", [*search_calls, *selection_calls])
    archive_size = len(pareto_front(evaluated))
    payload = {
        **_manifest_base(manifest, method="rpas", seed=args.seed, config=raw_config, gpu=gpu),
        "run_id": run_id, "run_kind": "pretest_search_select", "code_commit": _git_commit(Path(args.repo_root)),
        "search_calls": len(search_calls) + len(selection_calls),
        "search_tokens": sum(int(row["total_tokens"]) for row in [*search_calls, *selection_calls]),
        "search_wall_clock_seconds": float(time.time() - args.started_at),
        "rpas_search": {
            "reflection_calls": sum(row["split"] == "search_reflection" for row in search_calls),
            "new_candidates": generated, "mutation_logs": len([row for row in mutation_logs if row["status"] == "evaluated"]),
            "seed_archive_size": seed_archive_size, "pareto_archive_size": archive_size,
            "rule_fallbacks": 0, "finalists": len(finalists),
        },
    }
    _write_json(root / "run_manifest.json", payload)
    freeze_state(root)
    return root


def _require_unlock(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    unlock_path = root / "d_test_unlock.json"
    if not unlock_path.is_file():
        raise RuntimeError("EC-3 D_test is locked until all six immutable final states have been unlocked")
    unlock = _read_json(unlock_path)
    if not unlock.get("d_test_unlocked") or unlock.get("split_manifest_sha256") != manifest.get("split_manifest_sha256"):
        raise RuntimeError("EC-3 D_test unlock does not match this frozen data manifest")
    return unlock


def run_test(args: argparse.Namespace, *, single: bool = False) -> Path:
    gpu = _require_one_allocated_gpu()
    manifest = _read_json(args.manifest)
    root = Path(args.output_root)
    _require_unlock(root, manifest)
    raw_config, models, profile = _runtime(Path(args.config))
    test = _load_split(manifest, "test")
    method = "single_agent" if single else "rpas"
    output = root / method / f"seed_{args.seed}"
    if (output / "test_outputs.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite EC-3 held-out output: {output}")
    if single:
        candidate = seed_architectures(raw_config)[0]
    else:
        selected_path = root / "rpas" / f"seed_{args.seed}" / "selected_candidate.json"
        if not selected_path.is_file():
            raise FileNotFoundError(f"missing frozen RPAS selection: {selected_path}")
        candidate = _read_json(selected_path)["candidate"]
    run_id = f"ec3-hotpotqa-{method}-seed-{args.seed}"
    result, calls = _evaluate(candidate=candidate, rows=test, models=models, profile=profile, config=raw_config, split="test", run_id=run_id, method=method)
    output.mkdir(parents=True, exist_ok=True)
    _append_jsonl(output / "test_outputs.jsonl", result["outputs"])
    _append_jsonl(output / "test_calls.jsonl", calls)
    summary = {
        "protocol_version": PROTOCOL_VERSION, "method": method, "seed": args.seed,
        "split_manifest_sha256": manifest["split_manifest_sha256"], "d_test_accessed": True,
        "runtime_cuda_visible_devices": gpu, "candidate_id": candidate["id"], **_compact_result(result),
    }
    _write_json(output / "test_summary.json", summary)
    if not result["is_valid_candidate"]:
        raise RuntimeError(f"EC-3 {method} held-out run failed validity gate: {result['invalid_reasons']}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="EC-3 HotpotQA V3 RPAS runner")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="experiments/ec3_hotpotqa_qwen35_9b.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--aflow-root", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibration")
    pretest = sub.add_parser("pretest")
    pretest.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    test = sub.add_parser("test")
    test.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    single_test = sub.add_parser("single-test")
    single_test.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    args = parser.parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())
    args.manifest = str(Path(args.manifest).resolve())
    args.config = str(Path(args.config).resolve())
    args.output_root = str(Path(args.output_root).resolve())
    args.started_at = time.time()
    if args.command in {"calibration", "pretest"}:
        preflight(manifest_path=Path(args.manifest), aflow_root=Path(args.aflow_root), expected_endpoint=os.environ.get("RPAS_EXTERNAL_API_BASE"))
    if args.command == "calibration":
        target = run_calibration(args)
    elif args.command == "pretest":
        target = run_pretest(args)
    elif args.command == "test":
        target = run_test(args)
    else:
        target = run_test(args, single=True)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
