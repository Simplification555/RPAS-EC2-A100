"""Read-only publication-gap inventory; never promotes a run to a formal result."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from external_comparison.common.paired_statistics import heldout_token_map
from external_comparison.runners.aggregate_mmlu_v2 import METHODS as EC2_METHODS, _load_seed
from scripts.aggregate_ec1_native_formal import METHODS as EC1_METHODS, load


def common_budget_interval(tokens: list[int]) -> dict:
    """Necessary realized-cost condition; not proof of a pre-registered budget."""
    if not tokens or any(type(value) is not int or value <= 0 for value in tokens):
        return {"status": "PENDING", "reason": "positive_search_costs_required"}
    lower = max(tokens) / 1.10
    upper = min(tokens) / 0.90
    return {"status": "FAIL" if lower > upper else "NEEDS_FROZEN_BUDGET",
            "minimum_feasible_budget": lower, "maximum_feasible_budget": upper,
            "costs": tokens, "observations": len(tokens),
            "reason": "Necessary +/-10% condition only; native-unit overshoot needs explicit protocol adjudication."}


def telemetry_gaps(directory: Path) -> dict:
    calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = [json.loads(line) for line in (directory / "test_outputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {str(row.get("example_id", row.get("task_id", row.get("id", "")))) for row in outputs}
    attribution = heldout_token_map(calls, ids)
    finish = Counter(str(call.get("finish_reason")) for call in calls)
    known = sum(call.get("finish_reason") in {"stop", "length"} for call in calls)
    truncated = sum(call.get("finish_reason") == "length" for call in calls)
    # An unknown stop reason is not evidence of an untruncated response.
    truncation_status = "UNKNOWN" if known != len(calls) or not calls else (
        "PASS" if truncated / len(calls) < 0.01 else "FAIL"
    )
    return {"calls": len(calls), "total_tokens": sum(call["total_tokens"] for call in calls),
            "finish_reasons": dict(finish), "truncation_gate": truncation_status,
            "observed_length_fraction": truncated / len(calls) if calls else None,
            "missing_example_id_calls": sum(not call.get("example_id") for call in calls),
            "per_item_test_tokens_available": attribution["available"],
            "per_item_test_tokens_reason": attribution["reason"]}


def audit(ec1_root: Path, rpas_root: Path, single_root: Path, ec2_root: Path) -> dict:
    rows = []
    search_costs = []
    for experiment, methods in (("EC1", EC1_METHODS), ("EC2", EC2_METHODS)):
        for method in methods:
            root = ({"rpas": rpas_root, "single": single_root}.get(method, ec1_root)
                    if experiment == "EC1" else ec2_root)
            for seed in range(3):
                directory = root / method / f"seed_{seed}"
                row = {"experiment": experiment, "method": method, "seed": seed}
                try:
                    result = load(root, method, seed) if experiment == "EC1" else _load_seed(directory)
                    if result["method"] != method or result["seed"] != seed:
                        raise ValueError("method/seed identity differs from requested directory")
                    row.update(artifact_integrity="PASS", telemetry=telemetry_gaps(directory))
                    if experiment == "EC1" and method != "single":
                        search_costs.append(result["search_tokens"])
                    if experiment == "EC2":
                        row["checkpoint_files_verified"] = result["checkpoint_files_verified"]
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row.update(artifact_integrity="INCOMPLETE_OR_INVALID", reason=str(error))
                rows.append(row)
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "formal_result": False,
            "scope": "artifact, telemetry and necessary budget checks; not a complete publication certification",
            "expected_seed_results": 27,
            "integrity_passed_seed_results": sum(row["artifact_integrity"] == "PASS" for row in rows),
            "ec1_observed_search_budget_interval": common_budget_interval(search_costs),
            "ec1_search_seeds_observed": len(search_costs), "rows": rows,
            "unresolved_manual_gates": ["pre-registered EC1 matched budget and native-unit overshoot adjudication",
                                        "common timeout/retry policy", "upstream compatibility-patch fidelity",
                                        "held-out access only after final-state freeze",
                                        "operational co-resident timings are not isolated latency comparisons"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ec1-root", type=Path, required=True)
    parser.add_argument("--rpas-root", type=Path, required=True)
    parser.add_argument("--single-root", type=Path, required=True)
    parser.add_argument("--ec2-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.ec1_root, args.rpas_root, args.single_root, args.ec2_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
