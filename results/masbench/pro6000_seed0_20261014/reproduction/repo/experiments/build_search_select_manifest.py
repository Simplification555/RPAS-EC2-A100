#!/usr/bin/env python3
"""Build a pre-selection manifest using only D_search and D_select files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
SPLITS = {"search": 24, "select": 24}
SCHEMA = "rpas_masbench_search_select_blind_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty split: {path}")
    return rows


def build_manifest(data_dir: Path) -> dict[str, Any]:
    """Never opens D_test or the per-axis full manifest (which contains test IDs)."""
    axes: dict[str, Any] = {}
    for axis in AXES:
        axis_dir = data_dir / axis
        axes[axis] = {"row_ids": {}, "file_sha256": {}}
        for split, expected_count in SPLITS.items():
            path = axis_dir / f"{split}_24.jsonl"
            rows = load_rows(path)
            if len(rows) != expected_count:
                raise ValueError(f"{axis}/{split} must have {expected_count} rows, got {len(rows)}")
            ids = [str(row.get("id", "")) for row in rows]
            if any(not item for item in ids) or len(set(ids)) != expected_count:
                raise ValueError(f"{axis}/{split} IDs must be nonempty and unique")
            if any(row.get("axis") != axis or row.get("dataset") != "masbench"
                   or row.get("official_split") != "train" for row in rows):
                raise ValueError(f"{axis}/{split} contains a row outside the frozen official train split")
            axes[axis]["row_ids"][split] = ids
            axes[axis]["file_sha256"][split] = sha256_file(path)
    return {
        "schema_version": SCHEMA,
        "dataset": "Salesforce/MASBench",
        "source_repository": "JiangyueAnn/RPAS",
        "source_revision": "e12f58823be5f91a32f05f9af4d36e54838ffe59",
        "data_seed": 2026,
        "split_sizes": SPLITS,
        "axes": axes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    projected = build_manifest(args.data_dir)
    output_path = args.data_dir / "manifest_search_select.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.data_dir, prefix=".manifest_search_select.",
        suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(projected, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    print(f"WROTE_D_TEST_BLIND_MANIFEST {output_path}")


if __name__ == "__main__":
    main()
