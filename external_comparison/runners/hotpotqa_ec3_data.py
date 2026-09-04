"""Frozen-data and answer-metric primitives for EC-3 HotpotQA V3.

This module intentionally never downloads a dataset.  EC-3 accepts an
explicitly materialized AFlow HotpotQA fixture so every formal run can record
the exact provenance and byte hashes rather than silently drifting with a hub
revision or a third-party preprocessing script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


DATA_SEED = 2026
FORMAL_COUNTS = {"search": 120, "select": 80, "test": 800, "calib": 40}


@dataclass(frozen=True)
class HotpotExample:
    task_id: str
    question: str
    answer: str
    context: list[list[Any]]
    question_type: str
    answer_kind: str
    source_split: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    values = json.loads(raw) if stripped.startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path} must contain a JSON array or JSONL objects")
    return values


def _answer_kind(answer: str) -> str:
    return "yes_no" if answer.strip().lower() in {"yes", "no"} else "span"


def canonicalize(rows: Iterable[dict[str, Any]], *, source_split: str) -> list[HotpotExample]:
    """Accept the released HotpotQA/AFlow schema and reject incomplete rows."""
    examples: list[HotpotExample] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        task_id = str(row.get("_id", row.get("id", ""))).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        context = row.get("context")
        question_type = str(row.get("type", "unknown")).strip().lower() or "unknown"
        if not task_id or not question or not answer or not isinstance(context, list):
            raise ValueError(f"HotpotQA row {index} is missing _id/id, question, answer, or context")
        normalized_context: list[list[Any]] = []
        for paragraph in context:
            if not isinstance(paragraph, list) or len(paragraph) != 2 or not isinstance(paragraph[0], str) or not isinstance(paragraph[1], list):
                raise ValueError(f"HotpotQA row {task_id} has malformed context")
            if not all(isinstance(sentence, str) for sentence in paragraph[1]):
                raise ValueError(f"HotpotQA row {task_id} has non-text context sentence")
            normalized_context.append([paragraph[0], list(paragraph[1])])
        if task_id in seen:
            raise ValueError(f"duplicate HotpotQA task ID: {task_id}")
        seen.add(task_id)
        examples.append(HotpotExample(task_id, question, answer, normalized_context, question_type, _answer_kind(answer), source_split))
    if not examples:
        raise ValueError("HotpotQA fixture is empty")
    return examples


def render_prompt(example: HotpotExample) -> str:
    """Render only provided distractor paragraphs and the question.

    Gold answers, supporting-fact labels, and the HotpotQA question type never
    enter this prompt.  Titles are retained because they are benchmark-provided
    context, not retrieved metadata.
    """
    paragraphs = "\n\n".join(
        f"{title}: {' '.join(sentences)}" for title, sentences in example.context
    )
    return f"Context:\n{paragraphs}\n\nQuestion:\n{example.question}\n\nReturn only the concise final answer."


def normalize_answer(value: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(character for character in text if character not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(value.lower())))


def answer_scores(prediction: str, reference: str) -> dict[str, float]:
    """Official HotpotQA answer EM/F1 normalization and empty-answer handling."""
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    exact_match = float(predicted_tokens == reference_tokens)
    if not predicted_tokens or not reference_tokens:
        return {"em": exact_match, "f1": exact_match}
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if not overlap:
        return {"em": exact_match, "f1": 0.0}
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return {"em": exact_match, "f1": 2 * precision * recall / (precision + recall)}


def _stratified_take(rows: list[HotpotExample], count: int, rng: random.Random) -> list[HotpotExample]:
    groups: dict[tuple[str, str], list[HotpotExample]] = defaultdict(list)
    for row in rows:
        groups[(row.question_type, row.answer_kind)].append(row)
    for group in groups.values():
        rng.shuffle(group)
    if count > len(rows):
        raise ValueError(f"requested {count} rows from a pool of {len(rows)}")
    # Proportional allocation with deterministic largest-remainder tie breaks.
    allocation = {key: int(count * len(group) / len(rows)) for key, group in groups.items()}
    remainder = count - sum(allocation.values())
    fractions = sorted(
        ((count * len(group) / len(rows) - allocation[key], key) for key, group in groups.items()),
        key=lambda item: (-item[0], item[1]),
    )
    for _, key in fractions[:remainder]:
        allocation[key] += 1
    selected: list[HotpotExample] = []
    for key in sorted(groups):
        selected.extend(groups[key][: allocation[key]])
    rng.shuffle(selected)
    return selected


def make_splits(
    validation_fixture: list[HotpotExample], test_fixture: list[HotpotExample], calibration_pool: list[HotpotExample], *, data_seed: int
) -> dict[str, list[HotpotExample]]:
    """Preserve AFlow's 200 validation / 800 test fixture boundary.

    Only its validation fixture is stratified into D_search and D_select.  The
    supplied AFlow test fixture becomes D_test byte-for-byte in row content,
    preventing a method-local re-sampling of the benchmark's held-out pool.
    """
    if len(validation_fixture) != 200 or len(test_fixture) != 800:
        raise ValueError(
            "EC-3 requires AFlow fixtures with exactly 200 validation and 800 test examples"
        )
    fixture_ids = {row.task_id for row in validation_fixture} | {row.task_id for row in test_fixture}
    if len(fixture_ids) != 1000:
        raise ValueError("EC-3 AFlow validation and test fixtures overlap")
    calibration_pool = [row for row in calibration_pool if row.task_id not in fixture_ids]
    if len(calibration_pool) < FORMAL_COUNTS["calib"]:
        raise ValueError("EC-3 calibration source has fewer than 40 examples disjoint from the formal fixture")
    rng = random.Random(data_seed)
    remaining = list(validation_fixture)
    search = _stratified_take(remaining, FORMAL_COUNTS["search"], rng)
    chosen = {row.task_id for row in search}
    remaining = [row for row in remaining if row.task_id not in chosen]
    select = _stratified_take(remaining, FORMAL_COUNTS["select"], rng)
    chosen.update(row.task_id for row in select)
    if len(select) != FORMAL_COUNTS["select"]:
        raise AssertionError("EC-3 validation split did not leave exactly 80 selection rows")
    test = list(test_fixture)
    calib = _stratified_take(calibration_pool, FORMAL_COUNTS["calib"], random.Random(data_seed + 1))
    return {"calib": calib, "search": search, "select": select, "test": test}


def _write_jsonl(path: Path, rows: Iterable[HotpotExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")


def prepare(
    *, validate_fixture_path: Path, test_fixture_path: Path, calibration_path: Path, output_dir: Path, data_seed: int, aflow_commit: str
) -> dict[str, Any]:
    validation_fixture = canonicalize(_read_records(validate_fixture_path), source_split="aflow_validate_fixture")
    test_fixture = canonicalize(_read_records(test_fixture_path), source_split="aflow_test_fixture")
    calibration = canonicalize(_read_records(calibration_path), source_split="hotpotqa_train")
    splits = make_splits(validation_fixture, test_fixture, calibration, data_seed=data_seed)
    split_manifest_sha256 = hashlib.sha256(
        json.dumps({name: [row.task_id for row in rows] for name, rows in splits.items()}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        _write_jsonl(output_dir / f"hotpotqa_{name}.jsonl", rows)
    manifest = {
        "protocol_version": "EC3_HOTPOTQA_V3",
        "benchmark": "HotpotQA distractor / provided-context",
        "aflow_commit": aflow_commit,
        "data_seed": data_seed,
        "aflow_fixtures": {
            "validate": {"path": str(validate_fixture_path.resolve()), "sha256": _sha256(validate_fixture_path), "bytes": validate_fixture_path.stat().st_size, "count": len(validation_fixture)},
            "test": {"path": str(test_fixture_path.resolve()), "sha256": _sha256(test_fixture_path), "bytes": test_fixture_path.stat().st_size, "count": len(test_fixture)},
        },
        "calibration_source": {"path": str(calibration_path.resolve()), "sha256": _sha256(calibration_path), "bytes": calibration_path.stat().st_size, "count": len(calibration)},
        "splits": {
            name: {"path": str((output_dir / f"hotpotqa_{name}.jsonl").resolve()), "sha256": _sha256(output_dir / f"hotpotqa_{name}.jsonl"), "bytes": (output_dir / f"hotpotqa_{name}.jsonl").stat().st_size, "count": len(rows), "ids": [row.task_id for row in rows]}
            for name, rows in splits.items()
        },
        "split_manifest_sha256": split_manifest_sha256,
        "prompt_policy": "provided_context_and_question_only; no_answer_supporting_facts_type_or_external_retrieval",
        "answer_evaluator": "hotpotqa_normalize_answer_em_f1",
    }
    (output_dir / "hotpotqa_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze EC-3 HotpotQA V3 data without downloading it")
    parser.add_argument("--aflow-validate-fixture", required=True, help="Materialized official AFlow 200-example HotpotQA validation fixture")
    parser.add_argument("--aflow-test-fixture", required=True, help="Materialized official AFlow 800-example HotpotQA test fixture")
    parser.add_argument("--calibration-source", required=True, help="Disjoint HotpotQA train source for D_calib")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-seed", type=int, default=DATA_SEED)
    parser.add_argument("--aflow-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(validate_fixture_path=Path(args.aflow_validate_fixture), test_fixture_path=Path(args.aflow_test_fixture), calibration_path=Path(args.calibration_source), output_dir=Path(args.output_dir), data_seed=args.data_seed, aflow_commit=args.aflow_commit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
