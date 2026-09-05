"""Aggregate only protocol-valid EC-2 v2 communication-topology artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from external_comparison.runners.ec2_v2 import EC2_V2_PROTOCOL, validate_v2_manifest
from external_comparison.common.paired_statistics import two_level_paired_bootstrap
from external_comparison.common.manifest import sha256_json
from external_comparison.runners.mmlu import MMLU_SUBJECTS

METHODS = ("single_agent", "full_connected", "chain", "gdesigner", "rpas_comm")
SEEDS = (0, 1, 2)
TEST_EXAMPLES = 570


def _validate_frozen_split(run_dir: Path, manifest: dict[str, Any], item_ids: set[str]) -> None:
    journal = run_dir / "frozen_split.jsonl"
    if journal.is_file():
        records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != 1:
            raise ValueError(f"EC2 requires one frozen split record: {run_dir}")
        frozen = records[0]
    else:
        # Earlier runs saved only the shared split; validate its content, not just its claim.
        frozen = json.loads((run_dir.parent.parent / "split_manifest.json").read_text(encoding="utf-8"))
    ids = {phase: frozen[phase]["ids"] for phase in ("search", "select", "test")}
    for phase, count in (("search", 57), ("select", 57), ("test", TEST_EXAMPLES)):
        if len(ids[phase]) != count or len(set(ids[phase])) != count:
            raise ValueError(f"EC2 frozen {phase} IDs have invalid coverage: {run_dir}")
    if any(set(ids[a]) & set(ids[b]) for a, b in (("search", "select"), ("search", "test"), ("select", "test"))):
        raise ValueError(f"EC2 frozen splits overlap: {run_dir}")
    digest = sha256_json(ids)
    if digest != frozen.get("split_manifest_sha256") or digest != manifest.get("split_manifest_sha256"):
        raise ValueError(f"EC2 frozen split hash disagrees with actual IDs: {run_dir}")
    if item_ids != set(ids["test"]):
        raise ValueError(f"EC2 outputs do not match frozen held-out IDs: {run_dir}")


def _validate_checkpoints(run_dir: Path, manifest: dict[str, Any]) -> bool | None:
    if manifest["method"] != "gdesigner":
        return None
    files = manifest.get("checkpoint_files", {})
    if not files:
        if manifest.get("runtime_audit_version") == "journal_checkpoints_v1":
            raise ValueError(f"EC2 audited runtime is missing its checkpoint declarations: {run_dir}")
        return False  # Older running workers cannot retroactively deliver checkpoints.
    if set(files) != {"gdesigner_initial.pt", "gdesigner_pretest.pt"}:
        raise ValueError(f"EC2 checkpoint set is incomplete: {run_dir}")
    for name, digest in files.items():
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0 or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"EC2 checkpoint is missing or corrupt: {path}")
    return True


def _ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else 0.0
        return value, value
    margin = 1.96 * stdev(values) / math.sqrt(len(values))
    center = mean(values)
    return center - margin, center + margin


def _paired_bootstrap(left: dict[str, float], right: dict[str, float], *, repetitions: int, seed: int) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    if not common:
        return {"available": False, "reason": "no matching held-out test IDs"}
    differences = [left[item_id] - right[item_id] for item_id in common]
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(repetitions)
    )
    return {
        "available": True,
        "n_examples": len(common),
        "estimate": sum(differences) / len(differences),
        "lower": samples[max(0, math.floor(0.025 * repetitions))],
        "upper": samples[min(repetitions - 1, math.ceil(0.975 * repetitions) - 1)],
        "repetitions": repetitions,
    }


def _mcnemar(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    if not common:
        return {"available": False, "reason": "no matching held-out test IDs"}
    left_only = sum(left[item_id] == 1 and right[item_id] == 0 for item_id in common)
    right_only = sum(left[item_id] == 0 and right[item_id] == 1 for item_id in common)
    discordant = left_only + right_only
    statistic = (max(0, abs(left_only - right_only) - 1) ** 2 / discordant) if discordant else 0.0
    return {
        "available": True,
        "n_examples": len(common),
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "chi_square_cc": statistic,
    }


def _load_seed(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != EC2_V2_PROTOCOL:
        raise ValueError(f"legacy or unknown EC-2 artifact is not eligible for v2 aggregation: {run_dir}")
    validate_v2_manifest(manifest)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if any(result.get(key) != value for key, value in manifest.items()):
        raise ValueError(f"EC2 result and manifest disagree: {run_dir}")
    summary = result.get("summary", {})
    rows = result.get("rows", [])
    if int(summary.get("num_examples", 0)) != TEST_EXAMPLES or len(rows) != TEST_EXAMPLES:
        raise ValueError(f"EC-2 v2 main-table artifact must contain {TEST_EXAMPLES} held-out rows: {run_dir}")
    if int(manifest.get("test_examples", 0)) != TEST_EXAMPLES:
        raise ValueError(f"EC-2 v2 main-table artifact has a nonstandard test budget: {run_dir}")
    item_scores: dict[str, float] = {}
    by_subject: dict[str, list[float]] = defaultdict(list)
    communication = {"active_edges": 0, "messages": 0, "inter_agent_tokens": 0, "judge_input_tokens": 0}
    for row in rows:
        item_id = str(row.get("example_id", ""))
        subject = str(row.get("subject", ""))
        if not item_id or not subject or item_id in item_scores:
            raise ValueError(f"EC-2 v2 output has missing or duplicate held-out IDs: {run_dir}")
        if type(row.get("correct")) is not bool or not isinstance(row.get("answer"), str) or row["answer"] not in {"A", "B", "C", "D"}:
            raise ValueError(f"EC2 row lacks a boolean outcome or valid gold answer: {run_dir}")
        if row["correct"] != (row.get("prediction") == row["answer"]):
            raise ValueError(f"EC2 correctness disagrees with the saved answer/prediction: {run_dir}")
        correct = float(row["correct"])
        item_scores[item_id] = correct
        by_subject[subject].append(correct)
        for key in communication:
            communication[key] += int(row.get(key, 0))
    if set(by_subject) != set(MMLU_SUBJECTS) or any(len(scores) != 10 for scores in by_subject.values()):
        raise ValueError(f"EC-2 v2 must preserve all 57 MMLU subjects: {run_dir}")
    _validate_frozen_split(run_dir, manifest, set(item_scores))
    checkpoints_verified = _validate_checkpoints(run_dir, manifest)
    subject_macro = mean(mean(scores) for scores in by_subject.values())
    test_calls = int(summary.get("inference_calls", 0))
    test_tokens = int(summary.get("inference_tokens", 0))
    search_calls = int(summary.get("search_calls", manifest.get("search_calls", 0)))
    search_tokens = int(summary.get("search_tokens", manifest.get("search_tokens", 0)))
    if not math.isclose(summary["score"], mean(item_scores.values()), abs_tol=1e-12):
        raise ValueError(f"EC2 summary score disagrees with the saved outcomes: {run_dir}")
    valid_rate = sum(row.get("prediction") in {"A", "B", "C", "D"} for row in rows) / len(rows)
    if not math.isclose(summary["valid_answer_rate"], valid_rate, abs_tol=1e-12):
        raise ValueError(f"EC2 parser-valid rate disagrees with predictions: {run_dir}")
    saved_rows = [json.loads(line) for line in (run_dir / "test_outputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if saved_rows != rows:
        raise ValueError(f"EC2 JSON and JSONL outputs disagree: {run_dir}")
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not calls or summary.get("model_errors") != 0 or any(call.get("error") for call in calls):
        raise ValueError(f"EC2 has missing calls or model errors: {run_dir}")
    for call in calls:
        values = [call.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(type(value) is not int or value < 0 for value in values) or values[2] != values[0] + values[1]:
            raise ValueError(f"EC2 token accounting is invalid: {run_dir}")
        if call.get("split") not in {"search", "select", "test"}:
            raise ValueError(f"EC2 call has an unknown phase: {run_dir}")
    for phases, expected_calls, expected_tokens in (({"test"}, test_calls, test_tokens), ({"search", "select"}, search_calls, search_tokens)):
        subset = [call for call in calls if call["split"] in phases]
        if len(subset) != expected_calls or sum(call["total_tokens"] for call in subset) != expected_tokens:
            raise ValueError(f"EC2 summary disagrees with call telemetry: {run_dir}")
    truncation_rate = sum(call.get("finish_reason") == "length" for call in calls) / len(calls)
    return {
        "method": str(manifest["method"]),
        "seed": int(manifest["seed"]),
        "split_manifest_sha256": str(manifest["split_manifest_sha256"]),
        "accuracy": float(summary["score"]),
        "valid_answer_rate": float(summary["valid_answer_rate"]),
        "subject_macro": subject_macro,
        "test_calls": test_calls,
        "search_calls": search_calls,
        "test_tokens": test_tokens,
        "search_tokens": search_tokens,
        "total_calls": test_calls + search_calls,
        "total_tokens": test_tokens + search_tokens,
        "active_edges_per_query": communication["active_edges"] / TEST_EXAMPLES,
        "messages_per_query": communication["messages"] / TEST_EXAMPLES,
        "inter_agent_tokens_per_query": communication["inter_agent_tokens"] / TEST_EXAMPLES,
        "judge_input_tokens_per_query": communication["judge_input_tokens"] / TEST_EXAMPLES,
        "total_test_tokens_per_query": test_tokens / TEST_EXAMPLES,
        "item_scores": item_scores,
        "checkpoint_files_verified": checkpoints_verified,
        "all_call_truncation_rate": truncation_rate,
        "truncation_gate_passed": truncation_rate < 0.01 and all(call.get("finish_reason") in {"stop", "length"} for call in calls),
    }


def aggregate(root: str | Path, output_dir: str | Path, methods: tuple[str, ...] = METHODS) -> dict[str, Any]:
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a nonempty subset of {METHODS}")
    root = Path(root)
    seeds = []
    for method in methods:
        for seed in SEEDS:
            row = _load_seed(root / method / f"seed_{seed}")
            if row["method"] != method or row["seed"] != seed:
                raise ValueError(f"EC2 method/seed does not match its artifact directory: {method}/seed_{seed}")
            seeds.append(row)
    grouped = {method: [row for row in seeds if row["method"] == method] for method in methods}
    split_hashes = {row["split_manifest_sha256"] for row in seeds}
    if len(split_hashes) != 1:
        raise ValueError("EC-2 v2 methods/seeds did not use exactly the same frozen split manifest")
    table: list[dict[str, Any]] = []
    for method in methods:
        method_rows = sorted(grouped[method], key=lambda row: row["seed"])
        if [row["seed"] for row in method_rows] != list(SEEDS):
            raise ValueError(f"EC-2 v2 requires seeds {SEEDS} for {method}")
        accuracy = [row["accuracy"] for row in method_rows]
        row = {
            "method": method,
            "seeds": len(method_rows),
            "accuracy_mean": mean(accuracy),
            "accuracy_std": stdev(accuracy),
            "accuracy_ci95_low": _ci95(accuracy)[0],
            "accuracy_ci95_high": _ci95(accuracy)[1],
            "subject_macro_accuracy_mean": mean(row["subject_macro"] for row in method_rows),
            "valid_answer_rate_mean": mean(row["valid_answer_rate"] for row in method_rows),
            "all_call_truncation_rate_mean": mean(row["all_call_truncation_rate"] for row in method_rows),
            "truncation_gate_passed_all_seeds": all(row["truncation_gate_passed"] for row in method_rows),
            "checkpoint_delivery_gate": "not_applicable" if method != "gdesigner" else (
                "passed" if all(row["checkpoint_files_verified"] for row in method_rows) else "missing_legacy_checkpoints"
            ),
        }
        for field in (
            "active_edges_per_query",
            "messages_per_query",
            "inter_agent_tokens_per_query",
            "judge_input_tokens_per_query",
            "total_test_tokens_per_query",
            "test_calls",
            "search_calls",
            "total_calls",
            "test_tokens",
            "search_tokens",
            "total_tokens",
        ):
            row[f"{field}_mean"] = mean(seed_row[field] for seed_row in method_rows)
        table.append(row)
    paired: dict[str, Any] = {}
    reference_rows = {row["seed"]: row for row in grouped["full_connected"]} if "full_connected" in grouped else {}
    for method in methods:
        if method == "full_connected" or not reference_rows:
            continue
        paired[method] = {
            str(seed): {
                "bootstrap_vs_full_connected": _paired_bootstrap(
                    next(row for row in grouped[method] if row["seed"] == seed)["item_scores"],
                    reference_rows[seed]["item_scores"],
                    repetitions=5000,
                    seed=2026 + seed,
                ),
                "mcnemar_vs_full_connected": _mcnemar(
                    next(row for row in grouped[method] if row["seed"] == seed)["item_scores"],
                    reference_rows[seed]["item_scores"],
                ),
            }
            for seed in SEEDS
        }
    seed_rows = [{key: value for key, value in row.items() if key != "item_scores"} for row in seeds]
    hierarchical = {}
    if reference_rows:
        reference_scores = {seed: row["item_scores"] for seed, row in reference_rows.items()}
        for method in methods:
            if method != "full_connected":
                scores = {row["seed"]: row["item_scores"] for row in grouped[method]}
                hierarchical[method] = two_level_paired_bootstrap(scores, reference_scores)
    payload = {
        "protocol_version": EC2_V2_PROTOCOL,
        "dataset": "MMLU-57x10 controlled subset",
        "formal_result": False,
        "formal_result_reason": "Aggregation is protocol-valid, but publication status remains gated by the repository formal-result checklist.",
        "split_manifest_sha256": next(iter(split_hashes)),
        "communication_metric_scope": "Worker-to-worker edges/messages/tokens exclude FinalRefer inputs; judge_input_tokens_per_query is reported separately.",
        "rows": table,
        "seed_rows": seed_rows,
        "paired_statistics_vs_full_connected": paired,
        "two_level_paired_statistics_vs_full_connected": hierarchical,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "main_table.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "main_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate only strict EC-2 v2 artifacts.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root, args.output_dir, tuple(args.methods)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
