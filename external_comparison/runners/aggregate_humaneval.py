"""Aggregate completed common-space HumanEval artifacts without test tuning."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def paired_bootstrap_ci(
    left: dict[str, float],
    right: dict[str, float],
    *,
    repetitions: int = 5000,
    seed: int = 2026,
) -> dict[str, float]:
    """Return a two-sided paired bootstrap CI for per-task score differences."""

    if set(left) != set(right) or not left:
        raise ValueError("paired bootstrap inputs must contain the same non-empty task IDs")
    differences = [float(left[key]) - float(right[key]) for key in sorted(left)]
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        samples.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    samples.sort()
    low_index = max(0, int(0.025 * repetitions) - 1)
    high_index = min(repetitions - 1, int(0.975 * repetitions))
    return {
        "estimate": sum(differences) / len(differences),
        "lower": samples[low_index],
        "upper": samples[high_index],
        "n_tasks": len(differences),
        "repetitions": repetitions,
    }


def _q_test_scores(result: dict[str, Any]) -> dict[str, float]:
    selected = [row for row in result.get("selected_test_rows", []) if "Q" in row.get("operating_points", [])]
    if not selected:
        raise ValueError("result has no Q operating-point test row")
    return {str(row["task_id"]): float(bool(row.get("passed"))) for row in selected[0]["test"].get("tasks", [])}


def aggregate(root: str | Path, *, methods: list[str], seeds: list[int], output_dir: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    output_path = Path(output_dir)
    loaded: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for method in methods:
        for seed in seeds:
            path = root_path / method / f"seed_{seed}" / "result.json"
            if not path.exists():
                raise FileNotFoundError(path)
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("formal_result") is not False:
                raise ValueError(f"formal result flag must remain false before G1-G9: {path}")
            loaded[method].append(result)
    summary = []
    score_maps: dict[str, dict[str, float]] = {}
    for method, results in loaded.items():
        scores = [_q_test_scores(result) for result in results]
        score_maps[method] = scores[0]
        values = [sum(score.values()) / len(score) for score in scores]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        summary.append({"method": method, "seeds": len(values), "mean_pass_at_1": mean, "std_pass_at_1": variance**0.5})
    reference = methods[0]
    confidence = {
        method: paired_bootstrap_ci(score_maps[method], score_maps[reference])
        for method in methods[1:]
    }
    artifact = {
        "status": "aggregated_preparation",
        "formal_result": False,
        "reference_method": reference,
        "summary": summary,
        "paired_bootstrap_vs_reference": confidence,
        "methods": methods,
        "seeds": seeds,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_path / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate common-space HumanEval results.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["random_as", "aflow_style", "adas_style", "rpas_quality", "rpas"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root, methods=args.methods, seeds=args.seeds, output_dir=args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
