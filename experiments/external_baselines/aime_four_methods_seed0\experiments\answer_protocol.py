"""Frozen answer extraction and scoring for the cross-method experiments.

This module is intentionally dependency-light so the native baseline adapters
and RPAS use exactly the same parser.  It never guesses from an unfinished
reasoning trace: a prediction is valid only when it is a scalar, boxed scalar,
or an explicit final-answer marker.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


VERSION = "answer_protocol_v4"
MASBENCH_SEPARATOR = "<<horizon>>"
_NUMBER = r"[-+]?\d+(?:/\d+)?"


def _strip_thinking(text: Any) -> str:
    value = str(text or "").replace("\r\n", "\n")
    return re.sub(r"<think>.*?(?:</think>|$)", "", value, flags=re.I | re.S).strip()


def _normalise_scalar(value: str) -> str:
    value = value.strip().strip("`").strip()
    if re.fullmatch(r"[-+]?\d+", value):
        return str(int(value))
    return value


def extract_aime_answer(text: Any) -> str:
    """Extract an explicit final AIME scalar without mistaking section headers for answers."""
    cleaned = _strip_thinking(text)
    if re.fullmatch(_NUMBER, cleaned):
        return _normalise_scalar(cleaned)

    # Explicit answer labels are authoritative and should win over boxed
    # intermediate values that may appear earlier in a chain of reasoning.
    answer_labels = list(
        re.finditer(
            r"(?im)^\s*\*{0,2}\s*(?:FINAL\s+ANSWER|ANSWER)\s*:?\s*\*{0,2}\s*(?::\s*)?\*{0,2}\s*\$?(" + _NUMBER + r")\$?\s*[.!]?\s*$",
            cleaned,
        )
    )
    if answer_labels:
        return _normalise_scalar(answer_labels[-1].group(1))

    final_sentence_answers = list(
        re.finditer(
            r"(?im)^\s*(?:THE\s+)?FINAL\s+ANSWER(?:\s+SEEMS?\s+TO\s+BE|\s+IS)?\s*[:：]?\s*\$?(" + _NUMBER + r")\$?\s*[.!]?\s*$",
            cleaned,
        )
    )
    if final_sentence_answers:
        return _normalise_scalar(final_sentence_answers[-1].group(1))

    hash_answers = list(re.finditer(r"(?im)^\s*####\s*\$?(" + _NUMBER + r")\$?\s*$", cleaned))
    if hash_answers:
        return _normalise_scalar(hash_answers[-1].group(1))

    boxed = re.findall(r"\\boxed\{\s*(" + _NUMBER + r")\s*\}", cleaned)
    if boxed:
        return _normalise_scalar(boxed[-1])

    # Accept only answer-shaped headings, not ordinary Markdown headings such
    # as ``### Step 1``. The old broad heading regex swallowed those and
    # returned an empty answer before reaching a valid boxed answer below it.
    answer_headings = list(
        re.finditer(r"(?im)^\s*###\s*(?:ANSWER\s*:?\s*)?(" + _NUMBER + r")\s*$", cleaned)
    )
    if answer_headings:
        return _normalise_scalar(answer_headings[-1].group(1))
    answer_markers = list(re.finditer(r"(?im)^\s*###\s*ANSWER\s*:?\s*$", cleaned))
    for marker in reversed(answer_markers):
        for line in cleaned[marker.end() :].splitlines():
            payload = line.strip().strip("$").strip()
            if not payload:
                continue
            if re.fullmatch(_NUMBER, payload):
                return _normalise_scalar(payload)
            break

    # Some native math workflows end with a standalone scalar instead of a
    # box/marker. This remains unambiguous because only the final non-empty
    # line is accepted; numbers elsewhere in the reasoning are never guessed.
    lines = [line.strip().strip("$").strip() for line in cleaned.splitlines() if line.strip()]
    if lines and re.fullmatch(_NUMBER, lines[-1]):
        return _normalise_scalar(lines[-1])
    return ""


def score_aime(output: Any, gold: Any) -> dict[str, Any]:
    prediction = extract_aime_answer(output)
    expected = _normalise_scalar(str(gold))
    parser_valid = bool(prediction)
    correct = bool(prediction and prediction == expected)
    return {
        "prediction": prediction,
        "gold": expected,
        "parser_valid": parser_valid,
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "score_0_1_2": 2 if correct else (1 if parser_valid else 0),
    }


def extract_masbench_answer(text: Any) -> str:
    cleaned = _strip_thinking(text)
    matches = list(re.finditer(r"(?im)^FINAL\s+ANSWER\s*:\s*([^\n]+)", cleaned))
    if not matches:
        return ""
    payload = matches[-1].group(1).strip()
    parts = re.split(r"\s*<<\s*horizon\s*>>\s*", payload, flags=re.I)
    if not all(re.fullmatch(_NUMBER, part.strip()) for part in parts):
        return ""
    return MASBENCH_SEPARATOR.join(_normalise_scalar(part) for part in parts)


def score_masbench(output: Any, gold: Any) -> dict[str, Any]:
    prediction = extract_masbench_answer(output)
    expected = MASBENCH_SEPARATOR.join(_normalise_scalar(p) for p in str(gold).split(MASBENCH_SEPARATOR))
    return {
        "prediction": prediction,
        "gold": expected,
        "parser_valid": bool(prediction),
        "correct": bool(prediction and prediction == expected),
        "score": 1.0 if prediction and prediction == expected else 0.0,
    }


def canonical_manifest(data_dir: Path, *, data_seed: int, search: list[dict[str, Any]], select: list[dict[str, Any]], tests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def split(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [str(row.get("id", row.get("problem_idx", ""))) for row in rows]
        content = [str(row.get("problem", row.get("input", ""))) for row in rows]
        return {
            "count": len(rows),
            "ids": ids,
            "content_sha256": hashlib.sha256(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        }
    return {
        "version": VERSION,
        "data_seed": data_seed,
        "data_dir": str(data_dir.resolve()),
        "search": split(search),
        "select": split(select),
        "tests": {name: split(rows) for name, rows in tests.items()},
    }


def mean_std(values: Iterable[float]) -> dict[str, float | int]:
    vals = [float(v) for v in values]
    return {
        "n": len(vals),
        "mean": statistics.mean(vals) if vals else float("nan"),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }
