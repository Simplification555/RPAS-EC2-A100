"""Audit and aggregate the complete EC-3 held-out matrix.

Aggregation is deliberately not a publication-release action. Even a valid
three-method, three-seed matrix remains ``formal_result=false`` until the
repository-level protocol and release checklist are reviewed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


METHODS = ("single_agent", "aflow", "rpas")
SEEDS = (0, 1, 2)
TEST_EXAMPLES = 800
PROTOCOL_VERSION = "EC3_HOTPOTQA_V3"
MIN_VALID_ANSWER_RATE = 0.99
MAX_TRUNCATION_RATE = 0.01


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def _output_id(row: dict[str, Any]) -> str:
    return str(row.get("id", row.get("task_id", row.get("example_id", ""))))


def _load_run(root: Path, method: str, seed: int) -> dict[str, Any]:
    run_dir = root / method / f"seed_{seed}"
    summary_path = run_dir / "test_summary.json"
    outputs_path = run_dir / "test_outputs.jsonl"
    calls_path = run_dir / "test_calls.jsonl"
    for path in (summary_path, outputs_path, calls_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    summary = _read_json(summary_path)
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "seed": seed,
        "d_test_accessed": True,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"invalid EC-3 {key} in {summary_path}: {summary.get(key)!r}")
    split_sha = summary.get("split_manifest_sha256")
    if not isinstance(split_sha, str) or len(split_sha) != 64:
        raise ValueError(f"missing EC-3 split manifest SHA-256: {summary_path}")

    outputs = _read_jsonl(outputs_path)
    output_ids = [_output_id(row) for row in outputs]
    if len(outputs) != TEST_EXAMPLES or any(not item_id for item_id in output_ids) or len(set(output_ids)) != TEST_EXAMPLES:
        raise ValueError(f"EC-3 requires {TEST_EXAMPLES} unique held-out rows: {outputs_path}")

    calls = _read_jsonl(calls_path)
    if not calls or any(call.get("error") for call in calls):
        raise ValueError(f"EC-3 has missing calls or model errors: {calls_path}")
    for call in calls:
        usage = [call.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(type(value) is not int or value < 0 for value in usage) or usage[2] != usage[0] + usage[1]:
            raise ValueError(f"EC-3 token accounting is invalid: {calls_path}")
    total_tokens = sum(int(call["total_tokens"]) for call in calls)
    saved_calls = summary.get("test_calls")
    saved_tokens = summary.get("test_tokens")
    if saved_calls is not None and int(saved_calls) != len(calls):
        raise ValueError(f"EC-3 test call count disagrees with telemetry: {summary_path}")
    if saved_tokens is not None and int(saved_tokens) != total_tokens:
        raise ValueError(f"EC-3 test token count disagrees with telemetry: {summary_path}")

    valid_rate = float(summary.get("valid_answer_rate", 0.0))
    truncation_rate = float(summary.get("generation_truncation_rate", 1.0))
    if not math.isfinite(valid_rate) or valid_rate < MIN_VALID_ANSWER_RATE:
        raise ValueError(f"EC-3 valid-answer gate failed: {summary_path}")
    if not math.isfinite(truncation_rate) or truncation_rate >= MAX_TRUNCATION_RATE:
        raise ValueError(f"EC-3 truncation gate failed: {summary_path}")

    answer_f1 = float(summary.get("answer_f1", summary.get("score", float("nan"))))
    answer_em = float(summary.get("answer_em", float("nan")))
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (answer_f1, answer_em)):
        raise ValueError(f"EC-3 answer metrics are invalid: {summary_path}")
    return {
        "method": method,
        "seed": seed,
        "answer_f1": answer_f1,
        "answer_em": answer_em,
        "valid_answer_rate": valid_rate,
        "generation_truncation_rate": truncation_rate,
        "test_calls": len(calls),
        "test_tokens": total_tokens,
        "split_manifest_sha256": split_sha,
    }


def aggregate(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root)
    unlock = _read_json(root / "d_test_unlock.json")
    if unlock.get("protocol_version") != PROTOCOL_VERSION or unlock.get("d_test_unlocked") is not True:
        raise ValueError("EC-3 aggregation requires the protocol-valid D_test unlock record")
    final_states = unlock.get("final_states", [])
    identities = {(row.get("method"), row.get("seed")) for row in final_states if isinstance(row, dict)}
    if identities != {(method, seed) for method in ("aflow", "rpas") for seed in SEEDS}:
        raise ValueError("EC-3 unlock record must contain all six AFlow/RPAS frozen final states")

    runs = [_load_run(root, method, seed) for method in METHODS for seed in SEEDS]
    split_hashes = {run["split_manifest_sha256"] for run in runs}
    if len(split_hashes) != 1 or next(iter(split_hashes)) != unlock.get("split_manifest_sha256"):
        raise ValueError("EC-3 held-out runs and unlock record do not share one split manifest")

    table = []
    for method in METHODS:
        method_runs = [run for run in runs if run["method"] == method]
        table.append({
            "method": method,
            "seeds": [run["seed"] for run in method_runs],
            "answer_f1_mean": statistics.fmean(run["answer_f1"] for run in method_runs),
            "answer_f1_std": statistics.stdev(run["answer_f1"] for run in method_runs),
            "answer_em_mean": statistics.fmean(run["answer_em"] for run in method_runs),
            "valid_answer_rate_mean": statistics.fmean(run["valid_answer_rate"] for run in method_runs),
            "generation_truncation_rate_mean": statistics.fmean(run["generation_truncation_rate"] for run in method_runs),
            "test_calls_mean": statistics.fmean(run["test_calls"] for run in method_runs),
            "test_tokens_mean": statistics.fmean(run["test_tokens"] for run in method_runs),
        })
    payload = {
        "protocol": PROTOCOL_VERSION,
        "formal_result": False,
        "formal_result_reason": "Complete aggregation does not itself certify the repository-level publication gate.",
        "matrix_complete": True,
        "split_manifest_sha256": next(iter(split_hashes)),
        "runs": runs,
        "summary": table,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
