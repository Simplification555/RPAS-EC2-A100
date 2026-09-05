"""Audit and aggregate the nine native EC-1 formal seed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

METHODS = ("aflow", "maas", "rpas")
SEEDS = (0, 1, 2)


def load(root: Path, method: str, seed: int) -> dict:
    directory = root / method / f"seed_{seed}"
    result_path = directory / "result.json"
    manifest_path = directory / "run_manifest.json"
    outputs_path = directory / "test_outputs.jsonl"
    for path in (result_path, manifest_path, outputs_path, directory / "calls.jsonl", directory / "SHA256SUMS"):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("formal_result") is not True or result.get("formal_result") is not True:
        raise ValueError(f"{method}/seed_{seed} is not a formal result")
    if int(result.get("seed", -1)) != seed or int(result.get("summary", {}).get("num_examples", 0)) != 131:
        raise ValueError(f"unexpected seed or row count: {result_path}")
    rows = [json.loads(line) for line in outputs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 131 or len({str(row.get("task_id", row.get("id", ""))) for row in rows}) != 131:
        raise ValueError(f"duplicate or incomplete test rows: {outputs_path}")
    split = result.get("split_manifest", {})
    return {
        "method": method,
        "seed": seed,
        "score": float(result["summary"]["score"]),
        "num_examples": len(rows),
        "valid_answer_rate": result["summary"].get("valid_answer_rate"),
        "inference_calls": int(result["summary"].get("inference_calls", 0)),
        "inference_tokens": int(result["summary"].get("inference_tokens", 0)),
        "search_calls": int(result["summary"].get("search_calls", 0)),
        "search_tokens": int(result["summary"].get("search_tokens", 0)),
        "config_sha256": result.get("config_sha256", ""),
        "public_test_sha256": result.get("public_test_sha256", ""),
        "split_manifest": split,
    }


def aggregate(root: Path, output: Path) -> dict:
    runs = [load(root, method, seed) for method in METHODS for seed in SEEDS]
    split_keys = {json.dumps(run["split_manifest"], sort_keys=True) for run in runs}
    if len(split_keys) != 1:
        raise ValueError("EC-1 formal runs do not share one frozen split manifest")
    rows = []
    for method in METHODS:
        subset = [run for run in runs if run["method"] == method]
        scores = [run["score"] for run in subset]
        rows.append({
            "method": method,
            "seeds": [run["seed"] for run in subset],
            "accuracy_mean": statistics.fmean(scores),
            "accuracy_std": statistics.stdev(scores),
            "accuracy_min": min(scores),
            "accuracy_max": max(scores),
            "inference_calls_mean": statistics.fmean(run["inference_calls"] for run in subset),
            "search_calls_mean": statistics.fmean(run["search_calls"] for run in subset),
            "inference_tokens_mean": statistics.fmean(run["inference_tokens"] for run in subset),
            "search_tokens_mean": statistics.fmean(run["search_tokens"] for run in subset),
        })
    payload = {"protocol": "EC-1 HumanEval native formal", "formal_result": True, "runs": runs, "summary": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root, args.output), ensure_ascii=False, indent=2))
