#!/usr/bin/env python3
"""Aggregate the ten strictly quality-gated AFlow/MaAS seed-0 runs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path
from typing import Any

from masbench_quality_gate import AXES, validate_run

METHODS = ("aflow", "maas")
METRICS = ("accuracy", "tokens_per_example", "component_accuracy", "mean_score_0_1_2")
POINTS = ("Q", "E")
CSV_FIELDS = (
    "method", "axis", "seed", "point", "accuracy", "tokens_per_example",
    "component_accuracy", "mean_score_0_1_2", "metric", "value", "n_axes",
)


def extract_points(result: dict[str, Any], method: str, axis: str) -> dict[str, dict[str, Any]]:
    if method == "aflow":
        return result.get("test", {})
    axis_result = result.get("test", {}).get(axis, {})
    return {point: axis_result.get(point, {}) for point in POINTS}


def aggregate_seed0(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runs: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    run_ids: list[str] = []
    details: list[dict[str, Any]] = []
    for method in METHODS:
        runs[method] = {}
        for axis in AXES:
            run_dir = root / method / axis / "seed_0"
            gate_path = run_dir / "quality_gate.json"
            if not gate_path.is_file():
                raise FileNotFoundError(f"quality gate missing: {gate_path}")
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            if gate.get("passed") is not True or (gate.get("method"), gate.get("axis"), gate.get("seed")) != (method, axis, 0):
                raise ValueError(f"stored quality gate failed or has wrong identity: {run_dir}")
            fresh = validate_run(run_dir)
            if fresh.get("passed") is not True:
                raise ValueError(f"fresh quality gate failed for {run_dir}: {fresh.get('errors')}")
            result = json.loads((run_dir / "native_result.json").read_text(encoding="utf-8"))
            points = extract_points(result, method, axis)
            if set(points) != set(POINTS):
                raise ValueError(f"missing Q/E test results for {method}/{axis}")
            runs[method][axis] = {}
            for point in POINTS:
                metrics = fresh["point_summary"][point]
                if any(metrics.get(metric) is None for metric in METRICS):
                    raise ValueError(f"missing aggregate metric for {method}/{axis}/seed_0/{point}")
                runs[method][axis][point] = {metric: float(metrics[metric]) for metric in METRICS}
                details.append({
                    "method": method, "axis": axis, "seed": 0, "point": point,
                    **runs[method][axis][point],
                })
            run_ids.append(str(gate.get("run_id")))

    summary: dict[str, Any] = {
        "schema_version": "masbench_single_seed_summary_v1",
        "seed": 0,
        "axes": list(AXES),
        "methods": {},
        "run_ids": run_ids,
        "uncertainty_note": "Single search seed only: no across-seed standard deviation or significance claim is estimable.",
    }
    macro_rows: list[dict[str, Any]] = []
    for method in METHODS:
        summary["methods"][method] = {"axes": {}, "macro_avg": {}}
        for axis in AXES:
            summary["methods"][method]["axes"][axis] = runs[method][axis]
        for point in POINTS:
            summary["methods"][method]["macro_avg"][point] = {}
            for metric in METRICS:
                mean = statistics.mean(runs[method][axis][point][metric] for axis in AXES)
                summary["methods"][method]["macro_avg"][point][metric] = {
                    "mean_across_five_axes": mean,
                    "n_axes": len(AXES),
                    "seed": 0,
                }
                macro_rows.append({
                    "method": method, "axis": "macro_avg", "seed": 0, "point": point,
                    "metric": metric, "value": mean, "n_axes": len(AXES),
                })
    return summary, details + macro_rows


def render_csv(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="directory containing aflow/ and maas/")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--verify-existing", action="store_true",
        help="recompute from all gated runs and verify an existing summary byte-for-byte",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or root / "seed0_summary").resolve()
    summary, rows = aggregate_seed0(root)
    json_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    csv_text = render_csv(rows)
    if args.verify_existing:
        if not out_dir.is_dir():
            raise FileNotFoundError(f"summary directory missing: {out_dir}")
        expected_names = {"aggregate.json", "aggregate.csv"}
        actual_names = {path.name for path in out_dir.iterdir() if path.is_file()}
        if actual_names != expected_names:
            raise ValueError(f"summary file set differs: {sorted(actual_names)}")
        if (out_dir / "aggregate.json").read_text(encoding="utf-8") != json_text:
            raise ValueError("existing aggregate.json does not match a fresh aggregation")
        with (out_dir / "aggregate.csv").open("r", encoding="utf-8", newline="") as stream:
            existing_csv = stream.read()
        if existing_csv != csv_text:
            raise ValueError("existing aggregate.csv does not match a fresh aggregation")
        print(f"Verified existing single-seed summary: {out_dir}")
        return
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite prior summary: {out_dir}")
    out_dir.mkdir(parents=True)
    (out_dir / "aggregate.json").write_text(json_text, encoding="utf-8")
    with (out_dir / "aggregate.csv").open("w", encoding="utf-8", newline="") as stream:
        stream.write(csv_text)
    print(f"Wrote single-seed summary: {out_dir}")


if __name__ == "__main__":
    main()
