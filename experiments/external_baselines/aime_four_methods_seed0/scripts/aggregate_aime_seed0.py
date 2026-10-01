#!/usr/bin/env python3
"""Verify four completed AIME seed-0 runs and publish one immutable summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "aflow": "aflow_maas/aflow/seed_0",
    "maas": "aflow_maas/maas/seed_0",
    "adas": "adas_gdesigner/adas/seed_0",
    "gdesigner": "adas_gdesigner/gdesigner/seed_0",
}
FIELDS = (
    "method", "seed", "test_split", "point", "num_examples", "exact_accuracy",
    "correct_of_30", "mean_score_0_1_2", "tokens_per_example", "total_tokens",
    "calls", "prompt_tokens", "completion_tokens", "same_candidate_as",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_check(method: str, run_dir: Path) -> Path:
    python = os.environ.get(f"AIME_{method.upper()}_PYTHON") or sys.executable
    if method in ("aflow", "maas"):
        checker = BUNDLE_ROOT / "experiments/aime_quality_gate.py"
        command = [python, str(checker), "--run-dir", str(run_dir), "--method", method, "--seed", "0"]
    else:
        checker = BUNDLE_ROOT / "experiments/aggregate_external_methods.py"
        command = [python, str(checker), "--check-run", str(run_dir), "--method", method,
                   "--dataset", "aime", "--seed", "0"]
    checked = subprocess.run(command, cwd=BUNDLE_ROOT, capture_output=True, text=True)
    if checked.returncode:
        raise RuntimeError(f"Fresh quality check failed for {method}:\n{checked.stdout}{checked.stderr}")
    if read_json(run_dir / "run_manifest.json").get("status") != "complete":
        raise ValueError(f"{method}: fresh check did not establish a complete run")
    print(f"PASS fresh check: {method} seed=0", flush=True)
    return checker


def summary_row(method: str, split: str, point_name: str, point: dict, telemetry: dict) -> dict:
    count = int(point["num_examples"])
    correct = int(point["correct"])
    accuracy = float(point.get("accuracy", point.get("score")))
    mean_score = float(point["mean_score_0_1_2"])
    total_tokens = int(point["total_tokens"])
    tokens_per_example = float(point.get("tokens_per_example", point.get("avg_cost", total_tokens / count)))
    if (count != 30 or not 0 <= correct <= count or not math.isfinite(accuracy)
            or not math.isclose(accuracy, correct / count, abs_tol=1e-12)
            or not math.isfinite(mean_score) or not 0 <= mean_score <= 2
            or total_tokens < 0 or not math.isfinite(tokens_per_example)
            or not math.isclose(tokens_per_example, total_tokens / count, abs_tol=1e-9)):
        raise ValueError(f"Invalid frozen test metrics: {method}/{split}/{point_name}")
    phase_point = "Q" if point.get("same_candidate_as") == "Q" else point_name
    phases = telemetry.get("phases", telemetry.get("phase_totals", {}))
    usage = point.get("resource_usage") or next(
        (phases[phase] for phase in (point.get("phase"), f"test:{split}:{phase_point}", f"test:{split}")
         if phase in phases), {}
    )
    row = {
        "method": method, "seed": 0, "test_split": split, "point": point_name,
        "num_examples": count, "exact_accuracy": accuracy, "correct_of_30": correct,
        "mean_score_0_1_2": mean_score, "tokens_per_example": tokens_per_example,
        "total_tokens": total_tokens, "same_candidate_as": point.get("same_candidate_as"),
    }
    for field in ("calls", "prompt_tokens", "completion_tokens"):
        value = point.get(field, usage.get(field))
        row[field] = int(value) if value is not None else None
    return row


def recompute(root: Path) -> dict:
    # Check the complete set before any checker reads the post-selection data.
    for method, relative_dir in RUNS.items():
        run_dir = root / relative_dir
        manifest = read_json(run_dir / "run_manifest.json")
        if (not (run_dir / "native_result.json").is_file()
                or manifest.get("method") != method
                or manifest.get("status") not in ("complete", "results_ready")):
            raise ValueError(f"Incomplete AIME run: {run_dir}")
    rows, sources = [], {}
    for method, relative_dir in RUNS.items():
        run_dir = root / relative_dir
        checker = fresh_check(method, run_dir)
        manifest = read_json(run_dir / "run_manifest.json")
        result_path = run_dir / "native_result.json"
        result = read_json(result_path)
        for split in ("aime_2025", "aime_2026"):
            for point in ("Q", "E"):
                rows.append(summary_row(method, split, point, result["test"][split][point],
                                        result.get("telemetry", {})))
        sources[method] = {
            "run_manifest": f"{relative_dir}/run_manifest.json",
            "native_result": f"{relative_dir}/native_result.json",
            "native_result_sha256": sha256(result_path),
            "run_id": manifest.get("run_id", result.get("run_id")),
            "protocol_version": manifest.get("protocol_version"),
            "upstream_provenance": manifest.get("upstream_provenance"),
            "runner_sha256": manifest.get("runner_sha256"),
            "answer_parser_sha256": manifest.get("answer_parser_sha256", manifest.get("answer_protocol_sha256")),
            "data_sha256": manifest.get("data_sha256"),
            "selection_lock_sha256": sha256(run_dir / "selection_frozen.json"),
            "dtest_access_manifest_sha256": sha256(run_dir / "dtest_access_manifest.json"),
            "fresh_check": "PASS",
            "checker_source": str(checker.relative_to(BUNDLE_ROOT)),
            "checker_sha256": sha256(checker),
            "controls": result.get("controls"),
        }
    return {
        "schema": "aime_four_external_methods_seed0_summary_v1", "dataset": "aime",
        "seed": 0, "n_seeds": 1, "num_runs": 4, "num_rows": len(rows),
        "uncertainty_note": "Single seed pilot; no across-seed standard deviation or statistical significance is estimated.",
        "method_label": "External methods under the shared AIME adapter protocol; not unmodified upstream reproductions.",
        "Q_E_note": "A point marked same_candidate_as=Q reuses Q evaluation; it is not a separate replicate.",
        "sources": sources, "rows": rows,
    }


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "seed0_summary"
    if output.exists() and not args.verify_existing:
        raise FileExistsError(f"Refusing to overwrite {output}; use --verify-existing to check it")
    if args.verify_existing and not output.is_dir():
        raise FileNotFoundError(f"No existing summary: {output}")
    report = recompute(root)
    encoded_csv = csv_text(report["rows"])
    if args.verify_existing:
        if (read_json(output / "aggregate.json") != report
                or (output / "aggregate.csv").read_text(encoding="utf-8") != encoded_csv):
            raise ValueError("Existing summary differs from freshly verified seed-0 results")
        print("PASS: existing seed-0 JSON/CSV exactly match recomputation", flush=True)
        return 0
    output.mkdir()
    with (output / "aggregate.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (output / "aggregate.csv").open("x", encoding="utf-8", newline="") as stream:
        stream.write(encoded_csv)
    print(f"PASS: four verified seed-0 runs, 16 rows; wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
