#!/usr/bin/env python3
"""Fail-closed quality gate for one native MASBench search/select/test run."""

from __future__ import annotations

import argparse
import json
import math
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONTEXT_LIMIT = 8192
OUTPUT_LIMIT = 6144
MAX_SAMPLE_FAILURE_RATE = 0.05
PROTOCOL_VERSION = "masbench_external_protocol_v1"
AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
BASELINE_REQUIRED = {
    "aflow": (
        "scripts/optimizer.py", "scripts/evaluator.py",
        "scripts/optimizer_utils/evaluation_utils.py", "scripts/optimizer_utils/graph_utils.py",
        "benchmarks/math.py",
    ),
    "maas": (
        "maas/ext/maas/scripts/optimizer.py", "maas/ext/maas/scripts/evaluator.py",
        "maas/ext/maas/scripts/optimizer_utils/evaluation_utils.py",
        "maas/ext/maas/scripts/optimizer_utils/graph_utils.py",
        "maas/ext/maas/benchmark/benchmark.py", "maas/ext/maas/benchmark/math.py",
    ),
}
BASELINE_COMMITS = {
    "aflow": "3f457218fc716093fe53f6df8a5d5e6379d66346",
    "maas": "987f3c1bc9a96e844fe090db3791446e3ef0f5c7",
}


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail_if(condition: bool, reason: str, errors: list[str]) -> None:
    if condition:
        errors.append(reason)


def _points(result: dict[str, Any], axis: str) -> dict[str, dict[str, Any]]:
    if result.get("method") == "aflow":
        return result.get("test", {})
    if result.get("method") == "maas":
        axis_result = result.get("test", {}).get(axis, {})
        return {key: axis_result.get(key, {}) for key in ("Q", "E")}
    return {}


def validate_run(run_dir: Path, rpas_root: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = run_dir / "run_manifest.json"
    result_path = run_dir / "native_result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        return {"passed": False, "errors": ["run manifest or native result is missing"]}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    method, axis, seed = manifest.get("method"), manifest.get("axis"), manifest.get("seed")
    _fail_if(manifest.get("protocol_version") != PROTOCOL_VERSION, "unexpected protocol version", errors)
    _fail_if(manifest.get("status") != "complete", f"run status is {manifest.get('status')!r}, not complete", errors)
    _fail_if(method not in ("aflow", "maas"), "unknown method", errors)
    _fail_if(axis not in AXES, "unknown MASBench axis", errors)
    _fail_if(seed != 0, "seed is not the requested single frozen seed 0", errors)
    _fail_if(result.get("run_id") != manifest.get("run_id"), "run ID mismatch", errors)
    _fail_if(result.get("method") != method, "method mismatch between result and manifest", errors)
    _fail_if(manifest.get("pairwise_disjoint") is not True, "frozen splits are not verified pairwise-disjoint", errors)
    _fail_if(manifest.get("d_test_opened") is not True, "D_test was not evaluated after the selection lock", errors)
    d_test_access = manifest.get("d_test_access", {})
    selection_path = run_dir / "selection_frozen.json"
    selection_hash = manifest.get("selection_lock_sha256")
    _fail_if(not selection_path.is_file(), "frozen D_select selection lock is missing", errors)
    if selection_path.is_file():
        lock = json.loads(selection_path.read_text(encoding="utf-8"))
        _fail_if(lock.get("status") != "frozen_before_test_access", "selection lock status is invalid", errors)
        _fail_if(lock.get("d_test_opened") is not False, "selection lock was not frozen before test access", errors)
        _fail_if(lock.get("method") != method or lock.get("axis") != axis or lock.get("seed") != seed,
                 "selection lock does not match run method/axis/seed", errors)
        _fail_if(not selection_hash or sha256_file(selection_path) != selection_hash,
                 "selection lock hash mismatch", errors)
        _fail_if(d_test_access.get("selection_lock_sha256") != selection_hash,
                 "D_test access audit does not reference the frozen selection lock", errors)
        _fail_if(d_test_access.get("examples") != 60, "D_test access audit does not record all 60 examples", errors)
        _fail_if(not d_test_access.get("test_file_sha256"), "D_test file hash missing from access audit", errors)
        result_lock = result.get("selection_lock", {})
        _fail_if(result_lock.get("path") != selection_path.name or result_lock.get("sha256") != selection_hash,
                 "native result does not reference the frozen selection lock", errors)
        try:
            locked_at = datetime.fromisoformat(str(lock.get("frozen_at_utc")))
            opened_at = datetime.fromisoformat(str(d_test_access.get("opened_at_utc")))
            _fail_if(opened_at <= locked_at, "D_test access timestamp precedes candidate freeze", errors)
        except (TypeError, ValueError):
            errors.append("selection-lock or D_test-access timestamp is missing/invalid")
    if axis == "robustness":
        _fail_if(not manifest.get("task_protocol_exception"), "robustness raw-fact/magic-number rule is not recorded", errors)
    _fail_if(manifest.get("native_result_sha256") != sha256_file(result_path), "native result hash mismatch", errors)

    controls = result.get("controls", {})
    expected_controls = {
        "model": "Qwen/Qwen3.5-9B",
        "temperature": 0.0,
        "top_p": 1.0,
        "thinking": False,
        "max_tokens": OUTPUT_LIMIT,
        "max_model_len": CONTEXT_LIMIT,
        "max_num_seqs": 24,
        "tensor_parallel_size": 1,
        "concurrency": 8,
        "data_seed": 2026,
        "search_seed": seed,
        "selection_split": "D_select",
        "test_split": "D_test",
        "search_size": 24,
        "selection_size": 24,
        "test_size": 60,
    }
    # MaAS stores nested Q/E points and omits explicit split-size keys in its
    # result controls only if a legacy runner produced the artifact.
    for key, expected in expected_controls.items():
        actual = controls.get(key)
        if key == "test_size" and method == "maas":
            actual = controls.get("test_size", manifest.get("split_counts", {}).get("test"))
        _fail_if(actual != expected, f"control {key}={actual!r}, expected {expected!r}", errors)

    method_config = result.get("native_method_config", {})
    if method == "aflow":
        expected_method_config = {
            "dataset": "MATH", "question_type": "math",
            "operators": ["Custom", "ScEnsemble", "Programmer"],
            "sample": 4, "check_convergence": False, "initial_round": 1,
            "max_rounds": 8, "validation_rounds": 1,
            "mutation_generation_retry_cap": 8,
        }
        for key, expected in expected_method_config.items():
            _fail_if(method_config.get(key) != expected,
                     f"AFlow native configuration {key}={method_config.get(key)!r}, expected {expected!r}", errors)
    elif method == "maas":
        _fail_if(method_config.get("dataset") != "MATH" or method_config.get("question_type") != "math",
                 "MaAS native task configuration is not MATH/math", errors)
        _fail_if(not isinstance(method_config.get("operators"), list) or not method_config.get("operators"),
                 "MaAS native operator/controller configuration is missing", errors)
        for key, expected in {"controller_sample": 1, "round": 1, "batch_size": 8,
                              "learning_rate": 0.01, "is_textgrad": False,
                              "training_repetitions": 4}.items():
            _fail_if(method_config.get(key) != expected,
                     f"MaAS native configuration {key}={method_config.get(key)!r}, expected {expected!r}", errors)

    guard = result.get("telemetry", {}).get("context_guard")
    _fail_if(not isinstance(guard, dict), "context guard summary missing", errors)
    if isinstance(guard, dict):
        _fail_if(guard.get("context_limit") != CONTEXT_LIMIT, "context guard did not enforce 8192 tokens", errors)
        _fail_if(int(guard.get("calls_seen", 0)) <= 0, "context guard saw no model requests", errors)

    effective_rpas_root = rpas_root or (Path(manifest["rpas_root"]) if manifest.get("rpas_root") else None)
    if effective_rpas_root is not None:
        data_root = effective_rpas_root / "data" / "masbench"
        for relative, expected_hash in manifest.get("data_sha256", {}).items():
            path = data_root / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                errors.append(f"frozen dataset hash mismatch: {relative}")
        parser_path = effective_rpas_root / "experiments" / "masbench_answer_protocol.py"
        if not parser_path.is_file() or sha256_file(parser_path) != manifest.get("answer_parser_sha256"):
            errors.append("answer parser hash mismatch")
        runner_path = effective_rpas_root / "experiments" / "native_masbench_external.py"
        if not runner_path.is_file() or sha256_file(runner_path) != manifest.get("runner_sha256"):
            errors.append("runner hash mismatch")
        selection_manifest_path = data_root / "manifest_search_select.json"
        expected_safe_manifest = manifest.get("data_sha256", {}).get("manifest_search_select.json")
        if not selection_manifest_path.is_file() or sha256_file(selection_manifest_path) != expected_safe_manifest:
            errors.append("pre-selection search/select manifest hash mismatch")

    baseline_root = Path(manifest.get("official_baseline_source", ""))
    baseline_hashes = manifest.get("baseline_source_sha256", {})
    if not baseline_root.is_dir():
        errors.append("native baseline source directory is missing")
    for relative in BASELINE_REQUIRED.get(str(method), ()):
        expected_hash = baseline_hashes.get(relative)
        source_file = baseline_root / relative
        if not expected_hash or not source_file.is_file() or sha256_file(source_file) != expected_hash:
            errors.append(f"native baseline source hash mismatch: {relative}")
    pinned_commit = manifest.get("baseline_git_commit")
    if not pinned_commit or pinned_commit != BASELINE_COMMITS.get(str(method)):
        errors.append("native baseline Git commit is not the frozen upstream revision")
    elif baseline_root.is_dir():
        completed = subprocess.run(
            ["git", "-C", str(baseline_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0 or completed.stdout.strip() != pinned_commit:
            errors.append("native baseline checkout does not match its frozen Git commit")

    points = _points(result, str(axis))
    test_n = int(manifest.get("split_counts", {}).get("test", 0))
    _fail_if(set(points) != {"Q", "E"}, "both frozen Q and E test operating points are required", errors)
    point_summary: dict[str, Any] = {}
    for point_name in ("Q", "E"):
        point = points.get(point_name, {})
        _fail_if(not isinstance(point, dict) or not point, f"{point_name} test result missing", errors)
        if not isinstance(point, dict) or not point:
            continue
        _fail_if(point.get("num_examples") != test_n or test_n != 60,
                 f"{point_name} test count {point.get('num_examples')} does not match frozen D_test size {test_n}", errors)
        correct, accuracy = point.get("correct"), point.get("accuracy")
        _fail_if(not isinstance(correct, int) or not isinstance(accuracy, (int, float)),
                 f"{point_name} exact-match metrics are missing", errors)
        if isinstance(correct, int) and isinstance(accuracy, (int, float)) and test_n:
            _fail_if(not math.isclose(float(accuracy), correct / test_n, rel_tol=0.0, abs_tol=1e-8),
                     f"{point_name} accuracy does not equal exact correct / N", errors)
        audit = point.get("output_audit", {})
        _fail_if(audit.get("csv_rows") != test_n, f"{point_name} failure audit row count mismatch", errors)
        failure_fraction = audit.get("failure_fraction")
        _fail_if(not isinstance(failure_fraction, (int, float)) or failure_fraction > MAX_SAMPLE_FAILURE_RATE,
                 f"{point_name} sample failure fraction exceeds 5% or is missing", errors)
        usage = point.get("resource_usage", {})
        calls, tokens = point.get("calls"), point.get("total_tokens")
        _fail_if(not isinstance(calls, int) or calls <= 0, f"{point_name} has no recorded model calls", errors)
        _fail_if(not isinstance(tokens, int) or tokens <= 0, f"{point_name} has no recorded token usage", errors)
        if isinstance(usage, dict):
            _fail_if(usage.get("calls") != calls or usage.get("total_tokens") != tokens,
                     f"{point_name} resource telemetry disagrees with point totals", errors)
        avg_cost = point.get("avg_cost")
        if isinstance(tokens, int) and test_n and isinstance(avg_cost, (int, float)):
            _fail_if(not math.isclose(float(avg_cost), tokens / test_n, rel_tol=0.0, abs_tol=1e-8),
                     f"{point_name} token cost per example disagrees with total tokens", errors)
        point_summary[point_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "tokens_per_example": point.get("avg_cost"),
            "calls": calls,
            "failure_fraction": failure_fraction,
            "parser_valid_rate": point.get("valid_answer_rate"),
            "component_accuracy": point.get("component_accuracy"),
            "mean_score_0_1_2": point.get("mean_score_0_1_2"),
        }

    selection_audit = result.get("selection", {}).get("output_audit") if method == "maas" else None
    if method == "maas":
        selection = result.get("selection", {})
        _fail_if(selection.get("num_examples") != 24, "MaAS D_select evaluation must contain 24 rows", errors)
        selection_fraction = (selection_audit or {}).get("failure_fraction")
        _fail_if(not isinstance(selection_fraction, (int, float)) or selection_fraction > MAX_SAMPLE_FAILURE_RATE,
                 "MaAS D_select failure fraction exceeds 5% or is missing", errors)
        if selection_audit:
            _fail_if(selection_audit.get("csv_rows") != 24, "MaAS D_select failure audit row count mismatch", errors)
        checkpoint = Path(str(result.get("controller_checkpoint", "")))
        _fail_if(not checkpoint.is_file(), "MaAS selected controller checkpoint is missing", errors)
        if checkpoint.is_file():
            _fail_if(sha256_file(checkpoint) != result.get("controller_checkpoint_sha256"),
                     "MaAS controller checkpoint hash mismatch", errors)
    else:
        valid = [row for row in result.get("selection_rows", []) if row.get("status") == "evaluated"]
        _fail_if(not valid, "AFlow has no valid D_select candidate", errors)
        for row in valid:
            if row.get("output_audit", {}).get("failure_fraction", 1.0) > MAX_SAMPLE_FAILURE_RATE:
                errors.append(f"AFlow selected from an invalid D_select round: {row.get('round')}")
        selected_rounds = {
            "Q": result.get("selected_round", {}).get("round"),
            "E": result.get("selected_efficiency_round", {}).get("round"),
        }
        valid_rounds = {row.get("round") for row in valid}
        for point_name, selected_round in selected_rounds.items():
            _fail_if(selected_round not in valid_rounds,
                     f"AFlow {point_name} round was not frozen from a valid D_select candidate", errors)
            point = points.get(point_name, {})
            _fail_if(point.get("selected_round") != selected_round,
                     f"AFlow {point_name} D_test result does not match selected D_select round", errors)

    return {
        "passed": not errors,
        "errors": errors,
        "run_id": manifest.get("run_id"),
        "method": method,
        "axis": axis,
        "seed": seed,
        "point_summary": point_summary,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rpas-root", type=Path, default=None)
    args = parser.parse_args()
    report = validate_run(args.run_dir.resolve(), args.rpas_root.resolve() if args.rpas_root else None)
    (args.run_dir / "quality_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
