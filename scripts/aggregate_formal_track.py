#!/usr/bin/env python3
"""Aggregate frozen RPAS search runs into an auditable main-table snapshot.

The script never promotes a run to ``formal_result: true``. It verifies the
artifacts that are available locally and writes both a machine-readable summary
and a concise Markdown status report. Formal promotion remains a separate,
manual protocol decision because non-local gates (for example benchmark access
and repository visibility) cannot be inferred from result files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MODES = ("baselines", "random", "aflow_style", "adas_style", "quality_only", "wan_pareto")
T_975_BY_N = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0, 0.0
    std = statistics.stdev(values)
    critical = T_975_BY_N.get(len(values), 1.96)
    return mean, std, critical * std / math.sqrt(len(values))


def numeric(payload: dict[str, Any], key: str) -> float:
    return float(payload.get(key, 0.0) or 0.0)


def selected_quality_test(result: dict[str, Any]) -> dict[str, Any]:
    for row in result.get("selected_test_rows", []):
        if row.get("selected_rank") == 0:
            return dict(row.get("test", {}))
    raise ValueError("result has no selected_rank=0 quality test row")


def compact_run_record(result_path: Path) -> dict[str, Any]:
    run_dir = result_path.parent
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest next to result: {manifest_path}")
    result = read_json(result_path)
    manifest = read_json(manifest_path)
    test = selected_quality_test(result)
    candidate_overhead = result.get("candidate_evaluation_overhead", {})
    controller_overhead = result.get("search_overhead", {})
    metadata = result.get("metadata", {})
    return {
        "path": str(run_dir),
        "dataset": metadata.get("dataset", ""),
        "test_name": metadata.get("test_name", ""),
        "mode": result.get("mode", manifest.get("mode", "")),
        "seed": int(result.get("seed", manifest.get("search_seed", -1))),
        "accuracy": numeric(test, "score"),
        "correct": int(test.get("correct", 0) or 0),
        "test_examples": int(test.get("num_examples", 0) or 0),
        "valid_answer_rate": numeric(test, "valid_answer_rate"),
        "valid_execution_rate": numeric(test, "valid_execution_rate"),
        "truncated_unextractable_rate": numeric(test, "truncated_unextractable_rate"),
        "test_calls": numeric(test, "sum_calls"),
        "test_tokens": numeric(test, "sum_total_tokens"),
        "test_latency_ms": numeric(test, "sum_emulated_wall_latency_ms"),
        "search_candidate_calls": numeric(candidate_overhead, "calls"),
        "search_candidate_tokens": numeric(candidate_overhead, "total_tokens"),
        "search_controller_calls": numeric(controller_overhead, "controller_calls"),
        "search_controller_tokens": numeric(controller_overhead, "controller_total_tokens"),
        "search_size": int(metadata.get("search_size", 0) or 0),
        "selection_size": int(metadata.get("selection_size", 0) or 0),
        "configured_test_size": int(metadata.get("test_size", 0) or 0),
        "seed_candidates": int(result.get("seed_candidate_budget", 0) or 0),
        "new_candidates": int(result.get("new_candidate_budget", 0) or 0),
        "num_candidates": int(result.get("num_candidates", 0) or 0),
        "selection_policy": result.get("selection_policy", ""),
        "formal_result": bool(manifest.get("formal_result", False)),
        "code_commit": manifest.get("code_commit", ""),
        "config_sha256": manifest.get("config_sha256", ""),
        "model_manifest_sha256": manifest.get("model_manifest_sha256", ""),
        "dataset_manifest_sha256": metadata.get("dataset_manifest_sha256", ""),
        "split_hashes": {
            name: payload.get("normalized_content_sha256", "")
            for name, payload in manifest.get("dataset_splits", {}).items()
        },
        "runtime_cuda_visible_devices": manifest.get("runtime_cuda_visible_devices", ""),
    }


def all_equal(values: list[Any]) -> bool:
    return len(set(values)) <= 1


def make_row(mode: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "accuracy",
        "valid_answer_rate",
        "valid_execution_rate",
        "truncated_unextractable_rate",
        "test_calls",
        "test_tokens",
        "search_candidate_calls",
        "search_candidate_tokens",
        "search_controller_calls",
        "search_controller_tokens",
        "test_latency_ms",
    )
    row: dict[str, Any] = {"method": mode, "n_seeds": len(runs), "seeds": sorted(run["seed"] for run in runs)}
    for metric in metric_names:
        mean, std, ci95 = mean_std_ci([numeric(run, metric) for run in runs])
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[f"{metric}_ci95"] = ci95
    row["search_calls_mean"] = row["search_candidate_calls_mean"] + row["search_controller_calls_mean"]
    row["search_tokens_mean"] = row["search_candidate_tokens_mean"] + row["search_controller_tokens_mean"]
    row["total_calls_mean"] = row["test_calls_mean"] + row["search_calls_mean"]
    row["total_tokens_mean"] = row["test_tokens_mean"] + row["search_tokens_mean"]
    row["test_examples"] = sorted({run["test_examples"] for run in runs})
    row["correct_total"] = sum(run["correct"] for run in runs)
    row["example_total"] = sum(run["test_examples"] for run in runs)
    return row


def format_pct(value: float, spread: float) -> str:
    return f"{100.0 * value:.2f} +/- {100.0 * spread:.2f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method", "n_seeds", "seeds", "accuracy_mean", "accuracy_std", "accuracy_ci95",
        "valid_answer_rate_mean", "valid_execution_rate_mean", "truncated_unextractable_rate_mean",
        "test_calls_mean", "search_calls_mean", "total_calls_mean", "test_tokens_mean",
        "search_tokens_mean", "total_tokens_mean", "test_latency_ms_mean", "correct_total", "example_total",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], gates: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    lines = [
        "# Formal Track Status",
        "",
        "This report is generated from completed run artifacts. It does not promote any result to formal status.",
        "",
        "| Method | Seeds | Accuracy (mean +/- 95% CI) | Valid answer | Test calls | Search calls | Total tokens |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {seeds} | {accuracy} | {valid} | {test_calls:.0f} | {search_calls:.0f} | {tokens:.0f} |".format(
                method=row["method"],
                seeds=",".join(str(seed) for seed in row["seeds"]),
                accuracy=format_pct(row["accuracy_mean"], row["accuracy_ci95"]),
                valid=f"{100.0 * row['valid_answer_rate_mean']:.2f}%",
                test_calls=row["test_calls_mean"],
                search_calls=row["search_calls_mean"],
                tokens=row["total_tokens_mean"],
            )
        )
    lines.extend(["", "## Artifact Gates", ""])
    for name, value in gates.items():
        rendered = "PASS" if value is True else "FAIL" if value is False else str(value)
        lines.append(f"- {name}: {rendered}")
    lines.extend(["", "## Run Inventory", ""])
    for run in sorted(runs, key=lambda item: (item["mode"], item["seed"])):
        lines.append(
            f"- {run['mode']} seed {run['seed']}: `{run['path']}` "
            f"(commit `{run['code_commit']}`, device `{run['runtime_cuda_visible_devices']}`)"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and audit frozen formal-track runs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/formal_v1"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/formal_v1"))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_paths = sorted(args.output_root.glob("**/seed_*/result.json"))
    runs = [compact_run_record(path) for path in result_paths]
    expected = {(mode, seed) for mode in args.modes for seed in args.seeds}
    found = {(run["mode"], run["seed"]) for run in runs}
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    selected_runs = [run for run in runs if (run["mode"], run["seed"]) in expected]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in selected_runs:
        grouped[run["mode"]].append(run)
    table_rows = [make_row(mode, grouped[mode]) for mode in args.modes if grouped[mode]]

    split_hash_sets = {
        split: sorted({run["split_hashes"].get(split, "") for run in selected_runs})
        for split in ("search", "selection", "test")
    }
    expected_new_candidate_budget = {
        mode: (0 if mode == "baselines" else 24) for mode in args.modes
    }
    gates: dict[str, Any] = {
        "all_expected_method_seed_artifacts_present": not missing,
        "missing_method_seed_artifacts": missing or "none",
        "unexpected_method_seed_artifacts": unexpected or "none",
        "all_runs_remain_unpromoted": all(not run["formal_result"] for run in selected_runs),
        "single_code_commit": all_equal([run["code_commit"] for run in selected_runs]),
        "single_config_hash": all_equal([run["config_sha256"] for run in selected_runs]),
        "single_model_manifest_hash": all_equal([run["model_manifest_sha256"] for run in selected_runs]),
        "single_dataset_manifest_hash": all_equal([run["dataset_manifest_sha256"] for run in selected_runs]),
        "identical_frozen_search_split": len(split_hash_sets["search"]) <= 1 and bool(selected_runs),
        "identical_frozen_selection_split": len(split_hash_sets["selection"]) <= 1 and bool(selected_runs),
        "identical_frozen_test_split": len(split_hash_sets["test"]) <= 1 and bool(selected_runs),
        "all_runs_use_60_30_30": all(
            run["search_size"] == 60 and run["selection_size"] == 30 and run["configured_test_size"] == 30
            for run in selected_runs
        ),
        "candidate_budgets_match_protocol": all(
            run["seed_candidates"] == 9 and run["new_candidates"] == expected_new_candidate_budget[run["mode"]]
            for run in selected_runs
        ),
        "formal_promotion": "MANUAL: requires all external G1-G9 checks; this aggregator never promotes results",
    }
    summary = {"runs": selected_runs, "table_rows": table_rows, "gates": gates, "split_hash_sets": split_hash_sets}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "main_table_snapshot.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(args.report_dir / "main_table_snapshot.csv", table_rows)
    write_markdown(args.report_dir / "main_table_status.md", table_rows, gates, selected_runs)
    print(json.dumps({"completed_runs": len(selected_runs), "missing": missing, "report_dir": str(args.report_dir)}))
    if missing and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
