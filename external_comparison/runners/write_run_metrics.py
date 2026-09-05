"""Materialize a compact, auditable metrics ledger for one completed seed.

The source artifacts remain authoritative.  This utility only summarizes their
manifest and call telemetry after the runner has completed successfully.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Write read-only per-seed experiment metrics")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--job-root")
    parser.add_argument("--stage", default="seed")
    args = parser.parse_args()

    root = Path(args.result_dir).resolve()
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    calls = _read_jsonl(root / "calls.jsonl")
    by_phase: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0})
    model_counts: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    for call in calls:
        phase = str(call.get("split", call.get("phase", "unknown")))
        bucket = by_phase[phase]
        bucket["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"):
            bucket[key] += _int(call.get(key))
        if call.get("model"):
            model_counts[str(call["model"])] += 1
        if call.get("finish_reason"):
            finish_reasons[str(call["finish_reason"])] += 1

    job_root = Path(args.job_root).resolve() if args.job_root else None
    timestamp_sources = [root / "run_manifest.json"]
    if job_root:
        timestamp_sources.extend((job_root / "environment.txt", job_root / "smoke_completion.json"))
    start_epoch = min((path.stat().st_mtime for path in timestamp_sources if path.is_file()), default=None)
    resolved_model = manifest.get("model") or os.environ.get("RPAS_EXTERNAL_MODEL")
    if not resolved_model and len(model_counts) == 1:
        resolved_model = next(iter(model_counts))
    metrics = {
        "schema_version": "rpas-run-metrics-v1",
        "generated_at_epoch": time.time(),
        "stage": args.stage,
        "job_id": args.job_id,
        "job_root": str(job_root) if job_root else None,
        "method": manifest.get("method"),
        "dataset": manifest.get("dataset"),
        "seed": manifest.get("seed"),
        "model": resolved_model,
        "models_seen": dict(sorted(model_counts.items())),
        "parameters": {
            key: manifest.get(key)
            for key in (
                "communication_rounds", "aflow_max_rounds", "aflow_sample",
                "aflow_validation_rounds", "selected_round", "maas_sample",
                "maas_batch_size", "maas_lr", "seed_candidates",
                "new_candidate_budget", "search_examples", "select_examples",
                "test_examples", "max_tokens", "data_seed",
            )
            if key in manifest
        },
        "calls": {
            "total": len(calls),
            "by_phase": dict(sorted(by_phase.items())),
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "model_errors": sum(bool(call.get("error")) for call in calls),
        },
        "wall_clock_seconds_observed": max(0.0, time.time() - start_epoch) if start_epoch else None,
        "source_files": {
            "run_manifest": str(manifest_path),
            "calls": str(root / "calls.jsonl"),
        },
    }
    (root / "run_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
