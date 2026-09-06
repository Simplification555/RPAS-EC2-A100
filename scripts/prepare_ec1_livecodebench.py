"""Freeze the protocol-v4 LiveCodeBench release-v6 EC-1B subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from external_comparison.adapters.native_common import sha256_file
from external_comparison.runners.livecodebench_ec1 import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DATA_SEED,
    RELEASE_VERSION,
    SOURCE_FILENAME,
    frozen_split,
    load_tasks,
    validate_frozen_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty frozen bundle: {args.output_dir}")
    tasks = [task for path in args.source for task in load_tasks(path)]
    ids = [task.question_id for task in tasks]
    if len(ids) != len(set(ids)):
        parser.error("release_v6 source shards contain duplicate question IDs")
    splits = frozen_split(tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "rpas_ec1_livecodebench_release_v6_v1",
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "release_version": RELEASE_VERSION,
        "scenario": "codegeneration",
        "dataset_config": "code_generation_lite",
        "source_filename": SOURCE_FILENAME,
        "source_shards": [
            {"filename": path.name, "sha256": sha256_file(path), "rows": len(load_tasks(path))}
            for path in args.source
        ],
        "data_seed": DATA_SEED,
        "private_tests_visible_to": "evaluator_only",
        "splits": {},
    }
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_raw_row(), ensure_ascii=False, sort_keys=True) + "\n")
        manifest["splits"][name] = {
            "count": len(rows),
            "sha256": sha256_file(path),
            "question_ids": [row.question_id for row in rows],
            "strata": sorted({f"{row.difficulty}/{row.platform}" for row in rows}),
        }
    (args.output_dir / "DATASET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validate_frozen_bundle(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "source_rows": len(tasks), "splits": {k: len(v) for k, v in splits.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
