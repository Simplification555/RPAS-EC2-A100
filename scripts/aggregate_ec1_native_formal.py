"""Audit and aggregate the nine native EC-1 formal seed artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

METHODS = ("aflow", "maas", "rpas")
SEEDS = (0, 1, 2)


def canonical_split(manifest: dict) -> dict:
    data = manifest.get("data", {})
    ids = manifest.get("split_manifest", {}).get("task_ids", {})
    search = ids.get("search", data.get("search_tasks", []))
    test = ids.get("test", data.get("test_tasks", []))
    if len(search) != 33 or len(set(search)) != 33 or len(test) != 131 or len(set(test)) != 131:
        raise ValueError("missing or invalid frozen 33/131 split IDs")
    if set(search) & set(test) or ids.get("select", []):
        raise ValueError("overlapping or unexpected selection split")
    split = {"search": sorted(search), "test": sorted(test)}
    for name in ("search_fixture_source_sha256", "test_fixture_source_sha256", "public_test_sha256"):
        value = manifest.get(name, data.get(name, ""))
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"missing or invalid {name}")
        split[name] = value
    return split


def verify_checksums(directory: Path, required: tuple[str, ...]) -> None:
    # Archived manifests can reference remote absolute paths. Only read files
    # inside this seed directory, never the paths supplied by the checksum file.
    checksums = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        basename = Path(name.lstrip("*")).name
        if basename in checksums:
            raise ValueError(f"duplicate checksum entry: {basename}")
        checksums[basename] = digest
    for name in required:
        observed = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if checksums.get(name) != observed:
            raise ValueError(f"missing or mismatched checksum: {directory / name}")


def load(root: Path, method: str, seed: int) -> dict:
    directory = root / method / f"seed_{seed}"
    result_path = directory / "result.json"
    manifest_path = directory / "run_manifest.json"
    outputs_path = directory / "test_outputs.jsonl"
    required = ("result.json", "run_manifest.json", "test_outputs.jsonl", "calls.jsonl", "run_metrics.json")
    for path in [*(directory / name for name in required), directory / "SHA256SUMS"]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    verify_checksums(directory, required)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("formal_result") is not True or result.get("formal_result") is not True:
        raise ValueError(f"{method}/seed_{seed} is not a formal result")
    if int(result.get("seed", -1)) != seed or int(result.get("summary", {}).get("num_examples", 0)) != 131:
        raise ValueError(f"unexpected seed or row count: {result_path}")
    if manifest.get("seed") != seed or manifest.get("method") != method or result.get("method") != method:
        raise ValueError(f"method/seed identity mismatch: {directory}")
    if any(result.get(key) != value for key, value in manifest.items()):
        raise ValueError(f"manifest/result divergence: {directory}")
    if method == "maas" and "hashed" in manifest.get("staged_compatibility_patch", "").lower():
        raise ValueError(f"substituted MaAS embeddings are not a native baseline: {directory}")
    rows = [json.loads(line) for line in outputs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 131 or len({str(row.get("task_id", row.get("id", ""))) for row in rows}) != 131:
        raise ValueError(f"duplicate or incomplete test rows: {outputs_path}")
    split = canonical_split(manifest)
    if sorted(str(row.get("task_id", row.get("id", ""))) for row in rows) != split["test"]:
        raise ValueError(f"test outputs do not match frozen held-out IDs: {directory}")
    if rows != result.get("rows"):
        raise ValueError(f"result rows differ from test_outputs: {directory}")
    outcomes = [row.get("correct", row.get("passed")) for row in rows]
    if any(type(value) is not bool for value in outcomes):
        raise ValueError(f"missing boolean outcome: {directory}")
    summary = result["summary"]
    if not math.isclose(summary["score"], sum(outcomes) / len(rows), abs_tol=1e-12):
        raise ValueError(f"score disagrees with held-out rows: {directory}")
    calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not calls or any(call.get("error") for call in calls) or summary.get("model_errors") != 0:
        raise ValueError(f"empty calls or model errors: {directory}")
    for call in calls:
        if call.get("split") not in {"search", "select", "test"}:
            raise ValueError(f"unknown call phase: {directory}")
        tokens = [call.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(type(value) is not int or value < 0 for value in tokens) or tokens[2] != tokens[0] + tokens[1]:
            raise ValueError(f"invalid token accounting: {directory}")
    for prefix, phase_calls in (
        ("inference", [call for call in calls if call["split"] == "test"]),
        ("search", [call for call in calls if call["split"] != "test"]),
    ):
        if summary.get(prefix + "_calls") != len(phase_calls) or summary.get(prefix + "_tokens") != sum(call["total_tokens"] for call in phase_calls):
            raise ValueError(f"{prefix} call accounting disagrees with telemetry: {directory}")
    metrics = json.loads((directory / "run_metrics.json").read_text(encoding="utf-8"))
    if metrics.get("total_model_calls") != len(calls) or metrics.get("total_tokens") != sum(call["total_tokens"] for call in calls):
        raise ValueError(f"metrics disagree with calls: {directory}")
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
        "total_model_calls": len(calls),
        "total_tokens": metrics["total_tokens"],
        "wall_clock_seconds": metrics.get("wall_clock_seconds"),
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
