#!/usr/bin/env python3
"""Offline preflight: verify D_search/D_select and native baseline pins, without reading D_test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from native_masbench_external import AXES, baseline_commit, baseline_source_hashes, validate_search_select


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpas-root", type=Path, default=Path.home() / "RPAS")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--aflow-source", type=Path, default=Path.home() / "challenge_cup_ec1/external_baselines/AFlow")
    parser.add_argument("--maas-source", type=Path, default=Path.home() / "challenge_cup_ec1/external_baselines/MaAS")
    parser.add_argument("--inspect-only", action="store_true", help="summarize answer domains without running the formal validator")
    args = parser.parse_args()
    data_dir = (args.data_dir or args.rpas_root / "data/masbench").resolve()
    manifest = json.loads((data_dir / "manifest_search_select.json").read_text(encoding="utf-8"))
    result = {"data_seed": manifest.get("data_seed"), "axes": {}}
    for axis in AXES:
        rows = {split: read_jsonl(data_dir / axis / f"{split}_24.jsonl") for split in ("search", "select")}
        if args.inspect_only:
            axis_report = {}
            for split, split_rows in rows.items():
                parts = [part.strip() for row in split_rows for part in str(row.get("answer", "")).split("<<horizon>>")]
                numeric = [int(part) for part in parts if part.lstrip("+-").isdigit()]
                out_of_z23 = [row for row in split_rows if any(
                    part.strip().lstrip("+-").isdigit() and not 0 <= int(part.strip()) <= 22
                    for part in str(row.get("answer", "")).split("<<horizon>>")
                )]
                axis_report[split] = {
                    "rows": len(split_rows),
                    "numeric_answer_min": min(numeric) if numeric else None,
                    "numeric_answer_max": max(numeric) if numeric else None,
                    "components_outside_0_22": sum(
                        1 for part in parts if part.lstrip("+-").isdigit() and not 0 <= int(part) <= 22
                    ),
                    "sample_out_of_range_rows": [
                        {"id": row.get("id"), "answer": row.get("answer"), "axis_value": row.get("axis_value"), "input_prefix": str(row.get("input", ""))[:280]}
                        for row in out_of_z23[:2]
                    ],
                }
            result["axes"][axis] = axis_report
            continue
        from native_aime_formal import sha256_file
        audit = validate_search_select(
            rows, manifest, axis, 2026,
            {split: sha256_file(data_dir / axis / f"{split}_24.jsonl") for split in ("search", "select")},
        )
        result["axes"][axis] = {
            "split_counts": {split: len(ids) for split, ids in audit["split_ids"].items()},
            "axis_value_counts": audit["axis_value_counts"],
            "search_select_disjoint": audit["search_select_disjoint"],
            "d_test_not_accessed": True,
        }
    contract = args.rpas_root / "experiments/MASBENCH_TASK_CONTRACT_v1.txt"
    if not contract.is_file():
        raise FileNotFoundError(contract)
    result["baseline_hashes"] = {
        "aflow": baseline_source_hashes(args.aflow_source.resolve(), "aflow"),
        "maas": baseline_source_hashes(args.maas_source.resolve(), "maas"),
    }
    result["baseline_commits"] = {
        "aflow": baseline_commit(args.aflow_source.resolve(), "aflow"),
        "maas": baseline_commit(args.maas_source.resolve(), "maas"),
    }
    print(json.dumps(result, indent=2))
    print("MASBENCH_INSPECTION_COMPLETE" if args.inspect_only else "MASBENCH_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
