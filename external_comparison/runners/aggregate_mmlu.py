"""Aggregate EC-2 MMLU-57x10 controlled-subset runs.

The aggregator reads only per-seed artifacts and emits a reproducible main-table
view. It keeps test inference cost separate from search cost; a zero search
counter means that the method did not separately instrument a search phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict


def _ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else 0.0
        return value, value
    margin = 1.96 * stdev(values) / math.sqrt(len(values))
    center = mean(values)
    return center - margin, center + margin


def _load_seed(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    summary = result.get("summary", result)
    if manifest.get("formal_result") is not False:
        raise ValueError(f"formal_result must be false for controlled pilot: {run_dir}")
    if summary.get("num_examples") != 570:
        raise ValueError(f"expected 570 test examples for MMLU-57x10: {run_dir}")
    test_calls = int(summary.get("inference_calls", 0))
    search_calls = int(summary.get("search_calls", manifest.get("search_calls", 0)))
    test_tokens = int(summary.get("inference_tokens", 0))
    search_tokens = int(summary.get("search_tokens", manifest.get("search_tokens", 0)))
    rows = result.get("rows", [])
    by_subject: dict[str, list[bool]] = defaultdict(list)
    item_scores: dict[str, float] = {}
    for row in rows:
        subject = str(row.get("subject", ""))
        # The official G-Designer loader appends ``_test`` to CSV stems;
        # normalize that adapter-specific suffix for cross-method pairing.
        if subject.endswith("_test"):
            subject = subject[:-5]
        if subject:
            by_subject[subject].append(bool(row.get("correct", False)))
        example_id = str(row.get("example_id", ""))
        if ":_test:" in example_id:
            example_id = example_id.replace(":_test:", ":")
        elif "_test:" in example_id:
            example_id = example_id.replace("_test:", ":")
        if example_id:
            item_scores[example_id] = float(bool(row.get("correct", False)))
    subject_scores = [sum(values) / len(values) for values in by_subject.values() if values]
    return {
        "method": str(manifest["method"]),
        "seed": int(manifest["seed"]),
        "accuracy": float(summary["score"]),
        "valid_answer_rate": float(summary["valid_answer_rate"]),
        "test_calls": test_calls,
        "search_calls": search_calls,
        "total_calls": test_calls + search_calls,
        "test_tokens": test_tokens,
        "search_tokens": search_tokens,
        "total_tokens": test_tokens + search_tokens,
        "search_instrumented": bool(search_calls or search_tokens),
        "subject_macro": sum(subject_scores) / len(subject_scores) if subject_scores else None,
        "item_scores": item_scores,
    }


def _paired_bootstrap(left: dict[str, float], right: dict[str, float], *, repetitions: int = 5000, seed: int = 2026) -> dict:
    common = sorted(set(left) & set(right))
    if not common:
        return {"available": False, "reason": "no matching example IDs"}
    differences = [left[key] - right[key] for key in common]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(repetitions))
    return {
        "available": True,
        "n_examples": len(common),
        "estimate": sum(differences) / len(differences),
        "lower": samples[max(0, int(0.025 * repetitions) - 1)],
        "upper": samples[min(repetitions - 1, int(0.975 * repetitions))],
        "repetitions": repetitions,
    }


def _mcnemar(left: dict[str, float], right: dict[str, float]) -> dict:
    common = sorted(set(left) & set(right))
    if not common:
        return {"available": False, "reason": "no matching example IDs"}
    left_only = sum(left[key] == 1 and right[key] == 0 for key in common)
    right_only = sum(left[key] == 0 and right[key] == 1 for key in common)
    discordant = left_only + right_only
    statistic = ((abs(left_only - right_only) - 1) ** 2 / discordant) if discordant else 0.0
    return {"available": True, "n_examples": len(common), "left_only": left_only, "right_only": right_only, "discordant": discordant, "chi_square_cc": statistic}


def aggregate(
    root: str | Path,
    output_dir: str | Path,
    methods: tuple[str, ...] = ("rpas", "gdesigner"),
) -> dict:
    root = Path(root)
    rows = []
    for method in methods:
        for seed in (0, 1, 2):
            rows.append(_load_seed(root / method / f"seed_{seed}"))
    grouped: dict[str, list[dict]] = {method: [r for r in rows if r["method"] == method] for method in methods}
    table = []
    for method, method_rows in grouped.items():
        if len(method_rows) != 3:
            raise ValueError(f"expected three seeds for {method}")
        accuracy = [r["accuracy"] for r in method_rows]
        valid = [r["valid_answer_rate"] for r in method_rows]
        fields = {key: [r[key] for r in method_rows] for key in ("test_calls", "search_calls", "total_calls", "test_tokens", "search_tokens", "total_tokens")}
        row = {
            "method": method,
            "seeds": 3,
            "test_examples": 570,
            "accuracy_mean": mean(accuracy),
            "accuracy_ci95_low": _ci95(accuracy)[0],
            "accuracy_ci95_high": _ci95(accuracy)[1],
            "valid_answer_rate_mean": mean(valid),
            "subject_macro_accuracy_mean": mean([r["subject_macro"] for r in method_rows if r["subject_macro"] is not None]) if any(r["subject_macro"] is not None for r in method_rows) else None,
            **{f"{key}_mean": mean(values) for key, values in fields.items()},
            "search_cost_note": "instrumented" if any(r["search_instrumented"] for r in method_rows) else "not separately instrumented",
        }
        table.append(row)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference = methods[0]
    reference_rows = {r["seed"]: r for r in grouped[reference]}
    paired = {}
    for method in methods[1:]:
        comparisons = {}
        for seed in (0, 1, 2):
            left = next(r for r in grouped[method] if r["seed"] == seed)
            right = reference_rows[seed]
            comparisons[str(seed)] = {
                "bootstrap": _paired_bootstrap(left["item_scores"], right["item_scores"], seed=2026 + seed),
                "mcnemar": _mcnemar(left["item_scores"], right["item_scores"]),
            }
        paired[method] = comparisons
    for row in rows:
        row.pop("item_scores", None)
    payload = {"dataset": "MMLU-57x10 controlled subset", "formal_result": False, "reference_method": reference, "rows": table, "seed_rows": rows, "paired_statistics": paired}
    (out / "main_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = list(table[0])
    with (out / "main_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help=".../ec2_gpu6/{rpas,gdesigner}")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["vanilla", "rpas_no_selection", "gdesigner", "rpas"])
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root, args.output_dir, tuple(args.methods)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
