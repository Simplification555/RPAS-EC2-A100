#!/usr/bin/env python3
"""Build a reproducible EC-3 calibration pool from official HotpotQA parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("_id", row.get("id", row.get("task_id", "")))).strip()
            if not task_id:
                raise ValueError(f"missing task id in {path}:{line_no}")
            if task_id in result:
                raise ValueError(f"duplicate fixture task id: {task_id}")
            result.add(task_id)
    return result


def canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("context") or {}
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    if len(titles) != len(sentences):
        raise ValueError(f"malformed context for {row.get('id')}")
    supporting = row.get("supporting_facts") or {}
    return {
        "_id": str(row["id"]),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
        "type": str(row.get("type", "unknown")),
        "level": str(row.get("level", "unknown")),
        "supporting_facts": [
            [title, sent_id]
            for title, sent_id in zip(
                supporting.get("title") or [], supporting.get("sent_id") or [], strict=True
            )
        ],
        "context": [[title, list(text)] for title, text in zip(titles, sentences, strict=True)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--exclude-fixture", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--source-repository", default="hotpotqa/hotpot_qa")
    parser.add_argument("--source-revision", default="1908d6afbbead072334abe2965f91bd2709910ab")
    parser.add_argument("--source-path", default="distractor/train-00000-of-00002.parquet")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    excluded = fixture_ids(args.exclude_fixture)
    table = pq.read_table(args.source_parquet)
    candidates = [canonical_row(row) for row in table.to_pylist() if str(row.get("id", "")) not in excluded]
    if args.pool_size < 40 or len(candidates) < args.pool_size:
        raise ValueError(f"invalid pool size {args.pool_size}; available={len(candidates)}")
    random.Random(args.data_seed).shuffle(candidates)
    selected = candidates[: args.pool_size]
    selected_ids = [row["_id"] for row in selected]
    if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) & excluded:
        raise AssertionError("calibration pool ID isolation failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    provenance = {
        "dataset": "HotpotQA distractor train",
        "source_repository": args.source_repository,
        "source_revision": args.source_revision,
        "source_path": args.source_path,
        "source_parquet_sha256": sha256_file(args.source_parquet),
        "source_rows": table.num_rows,
        "excluded_fixture_ids": len(excluded),
        "selection": "seeded_shuffle_after_fixture_id_exclusion",
        "data_seed": args.data_seed,
        "pool_size": len(selected),
        "question_type_counts": dict(sorted(Counter(row["type"] for row in selected).items())),
        "selected_ids_sha256": hashlib.sha256(
            json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "output_sha256": sha256_file(args.output),
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
