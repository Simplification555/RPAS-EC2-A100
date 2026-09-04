"""Method-agnostic JSONL telemetry and aggregate accounting."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from external_comparison.common.protocol import validate_shared_budget


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object without rewriting earlier records."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read telemetry records and fail loudly on malformed lines."""

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def summarize_calls(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Aggregate the fields that are comparable across implementations."""

    calls = list(records)
    summary: dict[str, float | int] = {
        "calls": len(calls),
        "prompt_tokens": sum(int(record.get("prompt_tokens", 0)) for record in calls),
        "completion_tokens": sum(int(record.get("completion_tokens", 0)) for record in calls),
        "total_tokens": sum(int(record.get("total_tokens", 0)) for record in calls),
        "input_cost": sum(float(record.get("input_cost", 0.0)) for record in calls),
        "output_cost": sum(float(record.get("output_cost", 0.0)) for record in calls),
        "inference_cost": sum(float(record.get("inference_cost", 0.0)) for record in calls),
        "network_latency_ms": sum(float(record.get("network_latency_ms", 0.0)) for record in calls),
        "retries": sum(int(record.get("retry_count", 0)) for record in calls),
    }
    validate_shared_budget(realized_calls=int(summary["calls"]), realized_tokens=int(summary["total_tokens"]))
    return summary

