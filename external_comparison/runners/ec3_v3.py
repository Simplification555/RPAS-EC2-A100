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
    run_search,
    scientific_config_payload,
    select_operating_points,
    select_parent,
    seed_architectures,
    sha256_json,
    shortlist_rows_for_selection,
    validate_candidate_contract,
)


PROTOCOL_VERSION = "EC3_HOTPOTQA_V3"
MIN_VALID_ANSWER_RATE = 0.99
MAX_TRUNCATION_RATE = 0.01
EXECUTOR_MAX_TOKENS = 512
META_MAX_TOKENS = 4096
RPAS_MODE = "wan_pareto"
SELECTION_STRATEGY = "quality_band_cost"
SELECTION_QUALITY_BAND = 0.05
PARETO_PARENT_PROB = 0.5
PARENT_SCORE_BAND = 0.05
PARENT_TOP_K = 6


def _read_json(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return os.environ.get("RPAS_CODE_COMMIT", "bundle_without_git_metadata")


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


def _core_examples(rows: list[HotpotExample], split: str) -> list[dict[str, Any]]:
    return [
        {
            "id": row.task_id,
            "dataset": "hotpotqa",
            "input": render_prompt(row),
            "answer": row.answer,
            "official_split": split,
            "hotpot_task": asdict(row),
        }
        for row in rows
    ]


def _runtime(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    raw_config = _read_json(config_path)
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE", "").strip()
    if not endpoint:
        raise RuntimeError("RPAS_EXTERNAL_API_BASE must point to this worker's local model service")
    executor_max_tokens = int(os.environ.get("RPAS_EC3_EXECUTOR_MAX_TOKENS", str(EXECUTOR_MAX_TOKENS)))
    meta_max_tokens = int(os.environ.get("RPAS_EC3_META_MAX_TOKENS", str(META_MAX_TOKENS)))
    if executor_max_tokens != EXECUTOR_MAX_TOKENS or meta_max_tokens != META_MAX_TOKENS:
        raise ValueError(
            "EC-3 V3 requires the frozen executor/meta decoding contract: 512/4096"
        )
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
        # Native Phase-2 selection expects score. For HotpotQA it is answer F1.
        "score": float(result["answer_f1"]),
        **_compact_result(result),
    }


def _shortlist(rows: list[dict[str, Any]], maximum: int = 5) -> list[dict[str, Any]]:
    shortlisted, _ = shortlist_rows_for_selection(
        rows,
        mode=RPAS_MODE,
        shortlist_size=maximum,
        selection_strategy=SELECTION_STRATEGY,
        quality_band=SELECTION_QUALITY_BAND,
    )
    return shortlisted


def _select(selection_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return select_operating_points(selection_rows)


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
        "executor_max_tokens": EXECUTOR_MAX_TOKENS,
        "meta_max_tokens": META_MAX_TOKENS,
        "temperature": 0.0,
        "runtime_cuda_visible_devices": gpu,
        "config_sha256": sha256_json(scientific_config_payload(config)),
    }


def _native_seeds(config: dict[str, Any], count: int) -> list[dict[str, Any]]:
    # Singleton model/site pools can collapse differently named native seeds.
    unique: dict[str, dict[str, Any]] = {}
    for candidate in seed_architectures(config):
        unique.setdefault(candidate["id"], candidate)
    seeds = list(unique.values())[:count]
    if len(seeds) != count:
        raise RuntimeError(f"EC-3 requires {count} distinct RPAS seed workflows")
    return seeds


def _calibration_seeds(config: dict[str, Any]) -> list[dict[str, Any]]:
    return _native_seeds(config, 2)


def _select_search_parent(rows: list[dict[str, Any]], rng: random.Random) -> tuple[dict[str, Any], str]:
    return select_parent(
        rows, rng, RPAS_MODE,
        pareto_parent_prob=PARETO_PARENT_PROB,
        parent_score_band=PARENT_SCORE_BAND,
        parent_top_k=PARENT_TOP_K,
    )


def run_calibration(args: argparse.Namespace) -> Path:
    gpu = _require_one_allocated_gpu()
    manifest = _read_json(args.manifest)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("EC-3 requires a V3 frozen manifest")
    raw_config, models, profile = _runtime(Path(args.config))
    calibration = _load_split(manifest, "calib")
    seeds = _calibration_seeds(raw_config)
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
        "calibration_candidate_ids": [candidate["id"] for candidate in seeds],
        "calibration_seed_policy": "first_two_distinct_native_ids",
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
    calibration = _load_split(manifest, "calib")
    pilot = args.command == "pilot"
    if pilot:
        # Frozen fixture order, chosen without examining answer scores.
        search, select = search[:8], select[:8]
        if len(search) != 8 or len(select) != 8:
            raise ValueError("EC-3 minimum pilot requires eight search and selection examples")
    root = Path(args.output_root) / "rpas" / f"seed_{args.seed}"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to reuse a frozen EC-3 search directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"ec3-hotpotqa-rpas-seed-{args.seed}"
    run_contract = {
        "run_id": run_id, "formal_result": False, "d_test_accessed": False,
        "run_kind": "minimum_search_select_calib_pilot" if pilot else "pretest_search_select",
        "status": "running", "seed": args.seed,
        "code_commit": _git_commit(Path(args.repo_root)),
        "runner_sha256": sha256_file(__file__),
        "search_examples": len(search), "select_examples": len(select),
        "seed_candidates": 4, "new_candidate_budget": 3,
        "subset_policy": "first_eight_frozen_fixture_rows" if pilot else "full_frozen_splits",
        "search_ids": [row.task_id for row in search], "select_ids": [row.task_id for row in select],
        "parent_selection": "repository_native_run_search.select_parent",
        "fidelity_scope": "repository-native phase2 run_search; HotpotQA evaluator injection only",
        "native_control_flow": "experiments.phase2_wan_agent_search.run_search",
        "d_test_substitute": "D_calib validity check after Q/E freeze; held-out D_test remains unopened",
    }
    _write_json(root / "minimum_run_contract.json", run_contract)
    if raw_config.get("reflection", {}).get("allow_rule_fallback") is not False:
        raise ValueError("EC-3 RPAS requires LLM reflection with rule fallback disabled")
    all_calls: list[dict[str, Any]] = []
    calibration_evaluations = 0

    def task_evaluator(**kwargs: Any) -> dict[str, Any]:
        nonlocal calibration_evaluations
        dataset = kwargs.pop("dataset")
        nominal_split = str(dataset[0]["official_split"])
        split = nominal_split
        if nominal_split == "calib":
            split = "calib_quality" if calibration_evaluations == 0 else "calib_efficiency"
            calibration_evaluations += 1
        tasks = [HotpotExample(**row["hotpot_task"]) for row in dataset]
        result, calls = _evaluate(
            candidate=kwargs["candidate"], rows=tasks, models=kwargs["models"],
            profile=kwargs["profile"], config=raw_config, split=split,
            run_id=run_id, method="rpas",
        )
        all_calls.extend(calls)
        if not kwargs.get("capture_outputs", False):
            result.pop("outputs", None)
        return result

    native_output = root / "native_core"
    core = run_search(
        config=raw_config,
        models=models,
        profile=profile,
        searchset=_core_examples(search, "search"),
        selectionset=_core_examples(select, "select"),
        # The native executor requires a terminal evaluation split. D_calib is
        # disjoint and is used only after Q/E selection; D_test remains locked.
        testset=_core_examples(calibration, "calib"),
        mode=RPAS_MODE,
        seed_candidate_budget=4,
        new_candidate_budget=3,
        search_examples=None,
        selection_shortlist_size=5,
        test_top_k=2,
        eval_concurrency=1,
        output_dir=native_output,
        seed=args.seed,
        selection_strategy=SELECTION_STRATEGY,
        quality_band=SELECTION_QUALITY_BAND,
        pareto_parent_prob=PARETO_PARENT_PROB,
        parent_score_band=PARENT_SCORE_BAND,
        parent_top_k=PARENT_TOP_K,
        reflection_mode="llm",
        reflection_model="qwen35_9b",
        reflection_max_tokens=META_MAX_TOKENS,
        reflection_children=3,
        reflection_example_limit=3,
        evaluation_cache_dir=None,
        resume=False,
        metadata={
            "dataset": "hotpotqa",
            "test_name": "distractor_provided_context",
            "data_seed": manifest["data_seed"],
            "split_manifest_sha256": manifest["split_manifest_sha256"],
            "d_test_accessed": False,
            "task_adapter": "external_comparison.runners.ec3_v3.task_evaluator",
        },
        candidate_evaluator=task_evaluator,
    )
    for record in _read_jsonl(native_output / "search_overhead_rows.jsonl"):
        plan = {"call_traces": record.get("call_traces", [])}
        all_calls.extend(_reflection_call_rows(run_id, str(record.get("parent_candidate_id", "unknown")), plan))
    selected_tests = core["selected_test_rows"]
    selected_test = selected_tests[0]
    efficient_test = next(row for row in selected_tests if "E" in row.get("operating_points", []))
    selected = selected_test["selection"]
    efficient = efficient_test["selection"]
    selected_payload = {
        "candidate": selected["candidate"], "candidate_id": selected["candidate_id"],
        "selection_answer_f1": selected["answer_f1"], "selection_answer_em": selected["answer_em"],
        "operating_point": "quality",
        "selection_policy": "protocol_q_e.delta=0.05",
        "quality_operating_point": {
            "candidate": selected["candidate"], "candidate_id": selected["candidate_id"],
            "answer_f1": selected["answer_f1"], "answer_em": selected["answer_em"],
        },
        "efficiency_operating_point": {
            "candidate": efficient["candidate"], "candidate_id": efficient["candidate_id"],
            "answer_f1": efficient["answer_f1"], "answer_em": efficient["answer_em"],
        },
        "finalists": [row["candidate_id"] for row in core["selection_rows"]],
        "native_core_result": "native_core/result.json",
    }
    _write_json(root / "selected_candidate.json", selected_payload)
    selected_sha = sha256_file(root / "selected_candidate.json")
    _write_json(root / "pilot_selection_lock.json", {
        "selected_candidate_sha256": selected_sha,
        "d_test_accessed": False,
        "selected_candidate_id": selected["candidate_id"],
        "frozen_at_epoch": time.time(),
    })
    seed_rows = [row for row in core["search_rows"] if row.get("candidate_origin") == "seed"]
    seed_archive_size = len(pareto_front(seed_rows))
    archive_size = len(core["search_pareto_front_ids"])
    search_calls = [row for row in all_calls if row["split"] in {"search", "select", "search_reflection"}]
    calibration_calls = [row for row in all_calls if row["split"].startswith("calib_")]
    calibration_result = selected_test["test"]
    _append_jsonl(root / "calls.jsonl", all_calls)
    _append_jsonl(root / "pilot_calibration_rows.jsonl", calibration_result.get("outputs", []))
    payload = {
        **_manifest_base(manifest, method="rpas", seed=args.seed, config=raw_config, gpu=gpu),
        **run_contract,
        "status": "passed" if calibration_result["is_valid_candidate"] else "failed",
        "search_calls": len(search_calls),
        "search_tokens": sum(int(row["total_tokens"]) for row in search_calls),
        "search_wall_clock_seconds": float(time.time() - args.started_at),
        "calibration_calls": len(calibration_calls),
        "calibration_tokens": sum(int(row["total_tokens"]) for row in calibration_calls),
        "calibration_answer_f1": calibration_result["answer_f1"],
        "total_model_calls": len(all_calls),
        "total_tokens": sum(int(row["total_tokens"]) for row in all_calls),
        "native_core_sha256": sha256_file(Path(args.repo_root) / "experiments/phase2_wan_agent_search.py"),
        "native_core_result": "native_core/result.json",
        "rpas_search": {
            "reflection_calls": sum(row["split"] == "search_reflection" for row in search_calls),
            "new_candidates": core["num_new_candidates"],
            "mutation_logs": core["num_new_candidates"],
            "seed_archive_size": seed_archive_size, "pareto_archive_size": archive_size,
            "pareto_front_constructed": archive_size > 0,
            "rule_fallbacks": core["search_overhead"]["rule_fallbacks"],
            "finalists": len(core["selection_rows"]),
            "mode": RPAS_MODE, "selection_strategy": SELECTION_STRATEGY,
            "seed_policy": "first_four_distinct_native_ids",
            "parent_selection": "repository_native_run_search.select_parent",
            "pareto_parent_prob": PARETO_PARENT_PROB,
            "parent_score_band": PARENT_SCORE_BAND, "parent_top_k": PARENT_TOP_K,
            "shortlist_policy": core["search_shortlist_policy"],
            "selection_policy": core["selection_policy"],
            "quality_candidate_id": selected["candidate_id"],
            "efficiency_candidate_id": efficient["candidate_id"],
        },
    }
    _write_json(root / "run_manifest.json", payload)
    if sha256_file(root / "selected_candidate.json") != selected_sha:
        raise RuntimeError("EC-3 selection changed after native Q/E freeze")
    if payload["status"] != "passed":
        raise RuntimeError("EC-3 selected candidate failed frozen D_calib validity")
    if not pilot:
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
        selected = _read_json(selected_path)
        candidate = selected["quality_operating_point"]["candidate"]
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
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--seed", type=int, required=True, choices=(0,))
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
    if args.command in {"calibration", "pretest", "pilot"}:
        preflight(manifest_path=Path(args.manifest), aflow_root=Path(args.aflow_root), expected_endpoint=os.environ.get("RPAS_EXTERNAL_API_BASE"))
    if args.command == "calibration":
        target = run_calibration(args)
    elif args.command in {"pretest", "pilot"}:
        target = run_pretest(args)
    elif args.command == "test":
        target = run_test(args)
    else:
        target = run_test(args, single=True)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
