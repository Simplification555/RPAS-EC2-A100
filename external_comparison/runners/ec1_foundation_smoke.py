"""Offline EC-1 contract smoke test.

This checks candidate-budget accounting and adapter isolation without model/API
calls. It is a prerequisite for the real HumanEval run, not a benchmark result.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from external_comparison.common.schema import CandidateRecord
from external_comparison.common.telemetry import append_jsonl
from experiments.search_adapters.base import CandidateObservation
from experiments.search_adapters.registry import build_adapter


def run_smoke(output_dir: str | Path, *, new_candidate_budget: int = 2) -> dict[str, Any]:
    if new_candidate_budget < 1:
        raise ValueError("new_candidate_budget must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = build_adapter("fake")
    adapter.initialize(
        seed_archive=[{"candidate_id": f"seed-{index}"} for index in range(9)],
        search_space={"topologies": ["single", "solver_verifier"]},
        rng=random.Random(0),
    )
    rows: list[dict[str, Any]] = []
    for index in range(new_candidate_budget):
        proposal = adapter.propose()
        row = CandidateRecord(
            run_id="ec1-foundation-smoke",
            method="fake",
            dataset="humaneval",
            split="search",
            candidate_id=proposal.candidate_id,
            seed=0,
            score=0.5 + index * 0.1,
            valid=True,
            total_calls=1,
            total_tokens=10,
            total_cost=0.0,
            architecture=proposal.architecture,
        ).to_dict()
        rows.append(row)
        append_jsonl(output / "search_rows.jsonl", row)
        adapter.observe(
            CandidateObservation(
                candidate_id=row["candidate_id"],
                valid=row["valid"],
                score=row["score"],
                total_calls=row["total_calls"],
                total_tokens=row["total_tokens"],
                total_cost=row["total_cost"],
                split=row["split"],
            )
        )
    result = {
        "status": "smoke_passed",
        "formal_result": False,
        "dataset": "humaneval",
        "new_candidate_budget": new_candidate_budget,
        "unique_executable_candidates": len(rows),
        "test_split_accessed": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/external_comparison/ec1_foundation_smoke")
    parser.add_argument("--new-candidate-budget", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir, new_candidate_budget=args.new_candidate_budget), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
