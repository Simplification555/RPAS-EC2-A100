#!/usr/bin/env python3
"""Normalize the frozen AIME sources and emit a source-hashed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def normalize(rows: list[dict[str, Any]], *, prefix: str) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        raw_id = row.get("id", row.get("problem_idx", index))
        if not problem or not answer:
            raise ValueError(f"{prefix} row {index} has no problem or answer")
        normalized.append({"id": str(raw_id), "problem": problem, "answer": answer})
    if len({row["id"] for row in normalized}) != len(normalized):
        raise ValueError(f"{prefix} contains duplicate problem IDs")
    return normalized


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare source-hashed AIME protocol-v1 data.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    sources = {
        "development": raw_root / "aimo_validation/data/train-00000-of-00001.parquet",
        "aime_2025": raw_root / "aime_2025/data/train-00000-of-00001.parquet",
        "aime_2026": raw_root / "aime_2026/data/train-00000-of-00001.parquet",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing AIME source files: {missing}")

    splits = {name: normalize(read_rows(path), prefix=name) for name, path in sources.items()}
    expected_counts = {"development": 90, "aime_2025": 30, "aime_2026": 30}
    actual_counts = {name: len(rows) for name, rows in splits.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected AIME source counts: {actual_counts}; expected {expected_counts}")

    output_dir = args.output_dir.resolve()
    write_jsonl(output_dir / "aimo-validation-aime.jsonl", splits["development"])
    write_jsonl(output_dir / "aime_2025.jsonl", splits["aime_2025"])
    write_jsonl(output_dir / "aime_2026.jsonl", splits["aime_2026"])
    manifest = {
        "protocol_version": "FROZEN-CORE-v1.0",
        "dataset": "AIME",
        "data_seed": 2026,
        "source_repositories": {
            "development": "AI-MO/aimo-validation-aime",
            "aime_2025": "MathArena/aime_2025",
            "aime_2026": "MathArena/aime_2026",
        },
        "sources": {
            name: {"sha256": sha256_file(path), "rows": actual_counts[name]}
            for name, path in sources.items()
        },
        "normalized_files": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.glob("*.jsonl"))
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
