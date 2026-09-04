"""Validate EC-1 inputs and formal-run prerequisites without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from external_comparison.adapters.registry import native_adapter_status


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    values: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"row at {path}:{line_number} is not an object")
            values.append(value)
    return values


def run_preflight(dataset_path: Path, public_test_path: Path, config_path: Path, *, validate_path: Path | None = None, test_path: Path | None = None) -> dict:
    errors: list[str] = []
    if not dataset_path.is_file():
        errors.append(f"missing HumanEval dataset: {dataset_path}")
        dataset_rows: list[dict] = []
    else:
        dataset_rows = _rows(dataset_path)
        if len(dataset_rows) != 164:
            errors.append(f"HumanEval dataset must contain 164 tasks, found {len(dataset_rows)}")
        required = {"prompt", "test", "entry_point"}
        for index, row in enumerate(dataset_rows):
            missing = sorted(key for key in required if not isinstance(row.get(key), str) or not row[key].strip())
            if missing:
                errors.append(f"dataset row {index} missing non-empty fields: {','.join(missing)}")
                break
    if not public_test_path.is_file():
        errors.append(f"missing canonical HumanEval public-test file: {public_test_path}")
    for label, path in (("AFlow validate", validate_path), ("AFlow test", test_path)):
        if path is not None and not path.is_file():
            errors.append(f"missing {label} fixture: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if config.get("split_sizes") != {"search": 33, "select": 0, "test": 131}:
        errors.append("EC-1 config split_sizes must be search=33, select=0, test=131")
    for method in ("aflow", "maas", "rpas"):
        if not native_adapter_status(method).get("available"):
            errors.append(f"native adapter unavailable: {method}")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_gpu:
        allowed = {item.strip() for item in visible_gpu.split(",") if item.strip()}
        if not allowed or not allowed.issubset({"4", "5"}):
            errors.append("CUDA_VISIBLE_DEVICES must contain only GPU 4 and/or GPU 5")
    result = {
        "status": "ready" if not errors else "blocked",
        "formal_result": False,
        "dataset": {"path": str(dataset_path), "rows": len(dataset_rows), "sha256": _sha256(dataset_path) if dataset_path.is_file() else None},
        "public_test": {"path": str(public_test_path), "sha256": _sha256(public_test_path) if public_test_path.is_file() else None},
        "aflow_fixtures": {label: {"path": str(path), "sha256": _sha256(path) if path and path.is_file() else None} for label, path in (("validate", validate_path), ("test", test_path))},
        "split_sizes": config.get("split_sizes"),
        "methods": {method: native_adapter_status(method) for method in ("aflow", "maas", "rpas")},
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="EC-1 formal-run preflight; never contacts an inference server")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--public-test-path", required=True)
    parser.add_argument("--aflow-validate-path")
    parser.add_argument("--aflow-test-path")
    parser.add_argument("--config", default="external_comparison/configs/ec1_humaneval.json")
    args = parser.parse_args()
    print(json.dumps(run_preflight(Path(args.dataset_path), Path(args.public_test_path), Path(args.config), validate_path=Path(args.aflow_validate_path) if args.aflow_validate_path else None, test_path=Path(args.aflow_test_path) if args.aflow_test_path else None), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
