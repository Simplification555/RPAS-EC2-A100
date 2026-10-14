"""Validation for RPAS's canonical, frozen MASBench splits."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from masbench_answer_protocol import SEPARATOR, extract_masbench_answer


EXPECTED_COUNTS = {"search": 24, "select": 24, "test": 60}


def canonical_gold(answer: Any, allow_raw: bool = False) -> str:
    parsed = extract_masbench_answer(f"FINAL ANSWER: {answer}")
    if not parsed:
        raise ValueError(f"MASBench gold answer is not a numeric scalar/vector: {answer!r}")
    parts = parsed.split(SEPARATOR)
    if any(not re.fullmatch(r"\d+", part) for part in parts):
        raise ValueError(f"MASBench gold answer must contain nonnegative integers: {answer!r}")
    if not allow_raw and any(int(part) > 22 for part in parts):
        raise ValueError(f"MASBench algebraic gold answer is outside Z_23: {answer!r}")
    return parsed


def expected_horizon_width(problem: Any) -> int:
    """Count the ordered answer slots in the prompt's explicit final-answer template."""
    text = str(problem or "")
    heading = re.search(r"(?im)^\s*#{1,6}\s*Final\s+Answers?\s*#*\s*$", text)
    if not heading:
        raise ValueError("horizon prompt has no explicit Final Answers section")
    tail = text[heading.end():]
    labels = [int(value) for value in re.findall(
        r"(?im)^\s*Problem\s+(\d+)\s*:\s*[^\n]*\\boxed\s*\{", tail
    )]
    if not labels or labels != list(range(1, len(labels) + 1)):
        raise ValueError("horizon prompt's final-answer slots are missing or out of order")
    return len(labels)


def validate_frozen_splits(
    rows: dict[str, list[dict[str, Any]]], dataset_manifest: dict[str, Any], axis: str,
    data_seed: int, file_hashes: dict[str, str],
) -> dict[str, Any]:
    """Validate the official axis manifest and bytes. Caller invokes after D_select is locked."""
    if dataset_manifest.get("schema") != "rpas_frozen_splits_v1":
        raise ValueError("unexpected official MASBench split-manifest schema")
    if dataset_manifest.get("group") != axis:
        raise ValueError(f"official MASBench manifest group does not match {axis!r}")
    if int(dataset_manifest.get("seed", -1)) != data_seed:
        raise ValueError("official MASBench split seed does not match requested frozen seed")
    source = dataset_manifest.get("source", {})
    if source.get("dataset") != "Salesforce/MASBench" or source.get("axis") != axis:
        raise ValueError("official MASBench source identity does not match dataset/axis")
    if dataset_manifest.get("counts") != EXPECTED_COUNTS:
        raise ValueError(f"official split counts differ from required {EXPECTED_COUNTS}")

    split_ids: dict[str, list[str]] = {}
    split_value_counts: dict[str, dict[str, int]] = {}
    expected_hashes = dataset_manifest.get("content_sha256", {})
    for split, expected_n in EXPECTED_COUNTS.items():
        current = rows[split]
        ids = [str(row.get("id", "")) for row in current]
        if len(current) != expected_n or any(not value for value in ids) or len(set(ids)) != expected_n:
            raise ValueError(f"{axis}/{split} must contain {expected_n} rows with unique, nonempty IDs")
        id_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        if id_hash != dataset_manifest.get("id_sha256", {}).get(split):
            raise ValueError(f"{axis}/{split} IDs do not match official frozen id_sha256")
        row_content_hashes = [str(row.get("content_sha256", "")) for row in current]
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in row_content_hashes):
            raise ValueError(f"{axis}/{split} has a missing or malformed row content hash")
        content_hash = hashlib.sha256("\n".join(row_content_hashes).encode("utf-8")).hexdigest()
        if content_hash != expected_hashes.get(split):
            raise ValueError(f"{axis}/{split} prompts do not match official frozen content_sha256")
        if file_hashes.get(split) is None:
            raise ValueError(f"{axis}/{split} file hash is missing from the run audit")
        expected_file = dataset_manifest.get("files", {}).get(split)
        if expected_file != f"{axis}/{('search_24' if split == 'search' else 'select_24' if split == 'select' else 'test_60')}.jsonl":
            raise ValueError(f"official file mapping for {axis}/{split} is invalid")
        expected_official_split = "test" if split == "test" else "train"
        values: Counter[str] = Counter()
        for row in current:
            if row.get("axis") != axis or row.get("dataset") != "masbench":
                raise ValueError(f"row {row.get('id')} has mismatched dataset/axis metadata")
            if row.get("official_split") != expected_official_split:
                raise ValueError(f"row {row.get('id')} is from the wrong official split")
            canonical = canonical_gold(row.get("answer"), allow_raw=(axis == "robustness"))
            if axis == "horizon":
                expected_width = expected_horizon_width(row.get("input"))
                if len(canonical.split(SEPARATOR)) != expected_width:
                    raise ValueError(f"horizon gold width mismatch for row {row.get('id')}")
            values[str(row.get("axis_value"))] += 1
        split_ids[split] = ids
        split_value_counts[split] = dict(sorted(values.items()))

    id_sets = {name: set(values) for name, values in split_ids.items()}
    if any(id_sets[left] & id_sets[right] for left, right in (
        ("search", "select"), ("search", "test"), ("select", "test")
    )):
        raise ValueError("D_search, D_select, and D_test must be pairwise disjoint")
    return {"split_ids": split_ids, "axis_value_counts": split_value_counts, "pairwise_disjoint": True}
