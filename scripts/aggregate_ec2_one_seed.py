"""Audit and summarize one complete EC-2 v2 seed without promoting it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from external_comparison.runners.aggregate_mmlu_v2 import (
    METHODS,
    SEEDS,
    _load_seed,
    _mcnemar,
    _paired_bootstrap,
)
from external_comparison.runners.ec2_v2 import EC2_V2_PROTOCOL


def aggregate_one_seed(root: str | Path, output_dir: str | Path, seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    root = Path(root)
    runs = [_load_seed(root / method / f"seed_{seed}") for method in METHODS]
    for method, run in zip(METHODS, runs, strict=True):
        if run["method"] != method or run["seed"] != seed:
            raise ValueError(f"EC-2 method/seed identity mismatch: expected {method}/seed_{seed}")
    split_hashes = {run["split_manifest_sha256"] for run in runs}
    if len(split_hashes) != 1:
        raise ValueError("EC-2 one-seed methods do not share one frozen split manifest")

    reference = next(run for run in runs if run["method"] == "full_connected")
    table = []
    for run in runs:
        row = {
            key: value
            for key, value in run.items()
            if key not in {"item_scores", "heldout_tokens"}
        }
        if run["method"] != "full_connected":
            row["paired_vs_full_connected"] = _paired_bootstrap(
                run["item_scores"], reference["item_scores"], repetitions=5000, seed=2026 + seed
            )
            row["mcnemar_vs_full_connected"] = _mcnemar(
                run["item_scores"], reference["item_scores"]
            )
        table.append(row)

    payload = {
        "protocol_version": EC2_V2_PROTOCOL,
        "dataset": "MMLU-57x10 controlled subset",
        "seed": seed,
        "one_seed_complete": True,
        "formal_result": False,
        "formal_result_reason": "One protocol-valid seed cannot replace the registered three-seed aggregate.",
        "split_manifest_sha256": next(iter(split_hashes)),
        "methods": table,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "one_seed_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_fields = [
        "method", "seed", "accuracy", "subject_macro", "valid_answer_rate",
        "test_calls", "search_calls", "total_calls", "test_tokens", "search_tokens",
        "total_tokens", "active_edges_per_query", "messages_per_query",
        "inter_agent_tokens_per_query", "judge_input_tokens_per_query",
        "total_test_tokens_per_query", "all_call_truncation_rate", "truncation_gate_passed",
    ]
    with (output / "one_seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate_one_seed(args.root, args.output_dir, args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
