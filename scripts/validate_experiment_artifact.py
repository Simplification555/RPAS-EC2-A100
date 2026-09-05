#!/usr/bin/env python3
"""Strict pre-publication checks for an EC artifact directory.

The checker intentionally fails closed: incomplete pilots may be retained for
debugging but cannot be labelled formal or included in a paper aggregate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    root = args.artifact
    required = ["environment.txt", "health.json", "models.json", "job_status.txt"]
    missing = [name for name in required if not (root / name).is_file()]
    errors: list[str] = [f"missing:{name}" for name in missing]
    for name in ("health.json", "models.json"):
        path = root / name
        if path.exists():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"invalid_json:{name}:{exc.msg}")
    status = (root / "job_status.txt").read_text(encoding="utf-8").strip() if (root / "job_status.txt").exists() else ""
    if status not in {"formal_result=true", "formal_result=false"}:
        errors.append("invalid_job_status")
    manifests = list(root.rglob("run_manifest.json"))
    if status == "formal_result=true" and not manifests:
        errors.append("formal_without_run_manifest")
    for path in root.rglob("*.jsonl"):
        try:
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_jsonl:{path.relative_to(root)}:{line_no}")
            break
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print(f"PASS artifact={root} status={status} files={sum(1 for _ in root.rglob('*') if _.is_file())}")
    print(f"root_sha256={sha256(root / 'environment.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
