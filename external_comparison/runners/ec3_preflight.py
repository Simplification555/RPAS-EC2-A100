"""Fail-fast preflight for EC-3 HotpotQA V3 calibration and formal runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from external_comparison.runners.hotpotqa_ec3_data import FORMAL_COUNTS
from external_comparison.runners.ec3_formal_gate import sha256_file


AFLOW_COMMIT = "3f457218fc716093fe53f6df8a5d5e6379d66346"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preflight(*, manifest_path: Path, aflow_root: Path, expected_endpoint: str | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != "EC3_HOTPOTQA_V3":
        raise ValueError("EC-3 requires an EC3_HOTPOTQA_V3 data manifest")
    if manifest.get("aflow_commit") != AFLOW_COMMIT:
        raise ValueError("EC-3 manifest does not pin the required AFlow commit")
    actual_commit = subprocess.run(["git", "-C", str(aflow_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if actual_commit != AFLOW_COMMIT:
        raise ValueError(f"EC-3 requires AFlow {AFLOW_COMMIT}, found {actual_commit}")
    dirty = subprocess.run(["git", "-C", str(aflow_root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise ValueError("EC-3 refuses a dirty official AFlow source")
    all_ids: set[str] = set()
    checks: dict[str, dict[str, Any]] = {}
    for name, expected_count in FORMAL_COUNTS.items():
        details = manifest.get("splits", {}).get(name, {})
        path = Path(details.get("path", ""))
        if not path.is_file() or int(details.get("count", -1)) != expected_count:
            raise ValueError(f"EC-3 {name} fixture is missing or has an invalid count")
        if sha256_file(path) != details.get("sha256") or path.stat().st_size != int(details.get("bytes", -1)):
            raise ValueError(f"EC-3 {name} fixture hash or byte size differs from its manifest")
        rows = _load_jsonl(path)
        ids = [str(row.get("task_id", "")) for row in rows]
        if len(rows) != expected_count or len(set(ids)) != expected_count or not all(ids):
            raise ValueError(f"EC-3 {name} fixture has duplicate or missing IDs")
        if all_ids & set(ids):
            raise ValueError(f"EC-3 {name} fixture overlaps an earlier split")
        all_ids.update(ids)
        checks[name] = {"path": str(path), "count": len(rows), "sha256": sha256_file(path)}
    if len(all_ids) != sum(FORMAL_COUNTS.values()):
        raise ValueError("EC-3 split cardinality is inconsistent")
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE", "")
    if expected_endpoint and endpoint != expected_endpoint:
        raise ValueError(f"EC-3 endpoint mismatch: {endpoint!r} != {expected_endpoint!r}")
    return {
        "protocol_version": "EC3_HOTPOTQA_V3", "status": "ready_for_calibration",
        "manifest": str(manifest_path.resolve()), "split_manifest_sha256": manifest["split_manifest_sha256"],
        "aflow_commit": actual_commit, "endpoint": endpoint, "splits": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="EC-3 V3 preflight")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--aflow-root", required=True)
    parser.add_argument("--expected-endpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = preflight(manifest_path=Path(args.manifest), aflow_root=Path(args.aflow_root), expected_endpoint=args.expected_endpoint)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
