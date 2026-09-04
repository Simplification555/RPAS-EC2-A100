"""Reproducibility helpers for external-comparison artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative_hashes(root: str | Path, paths: list[str | Path]) -> dict[str, str]:
    """Hash only declared repository-relative files, never secrets or env files."""

    root_path = Path(root).resolve()
    result: dict[str, str] = {}
    for item in paths:
        candidate = Path(item)
        if candidate.is_absolute():
            raise ValueError(f"manifest inputs must be repository-relative: {item}")
        resolved = (root_path / candidate).resolve()
        if root_path not in resolved.parents and resolved != root_path:
            raise ValueError(f"manifest input escapes repository root: {item}")
        result[candidate.as_posix()] = sha256_file(resolved)
    return result


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

