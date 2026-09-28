"""Frozen, shared MASBench answer parser and 0/1/2 scorer."""

from __future__ import annotations

import re
from typing import Any


VERSION = "masbench_answer_protocol_v1"
SEPARATOR = "<<horizon>>"
_NUMBER = r"[-+]?\d+(?:/\d+)?"


def _clean(text: Any) -> str:
    value = str(text or "").replace("\r\n", "\n")
    return re.sub(r"<think>.*?(?:</think>|$)", "", value, flags=re.I | re.S).strip()


def _normalise(value: str) -> str:
    value = value.strip().strip("`$").strip()
    if re.fullmatch(r"[-+]?\d+", value):
        return str(int(value))
    return value


def _numeric_parts(value: str) -> list[str] | None:
    parts = re.split(r"\s*<<\s*horizon\s*>>\s*", value, flags=re.I)
    normalized = [_normalise(part) for part in parts]
    return normalized if normalized and all(re.fullmatch(_NUMBER, part) for part in normalized) else None


def extract_masbench_answer(text: Any, expected_count: int | None = None) -> str:
    """Extract a final scalar or ordered multi-horizon answer, never reasoning numbers."""
    cleaned = _clean(text)

    explicit = list(re.finditer(r"(?im)^\s*FINAL\s+ANSWER\s*:\s*(.*?)\s*$", cleaned))
    for match in reversed(explicit):
        payload = match.group(1).strip()
        parts = _numeric_parts(payload)
        if parts and (expected_count is None or len(parts) == expected_count):
            return SEPARATOR.join(parts)
        # Some native templates put labelled boxed answers on the final-answer line.
        boxed = re.findall(r"\\boxed\{\s*(" + _NUMBER + r")\s*\}", payload)
        if boxed and (expected_count is None or len(boxed) == expected_count):
            return SEPARATOR.join(_normalise(part) for part in boxed)

    headings = list(re.finditer(r"(?im)^\s*#{1,6}\s*FINAL\s+ANSWERS?\s*#*\s*$", cleaned))
    region = cleaned[headings[-1].end():] if headings else cleaned
    numbered = list(
        re.finditer(
            r"(?im)^\s*Problem\s+(\d+)\s*:\s*[^\n]*?\\boxed\{\s*(" + _NUMBER + r")\s*\}[^\n]*$",
            region,
        )
    )
    if numbered:
        # Use only a complete, ordered sequence; malformed or missing answers
        # must not shift later answers into earlier horizon positions.
        labels = [int(match.group(1)) for match in numbered]
        values = [_normalise(match.group(2)) for match in numbered]
        if labels == list(range(1, len(labels) + 1)) and (expected_count is None or len(values) == expected_count):
            return SEPARATOR.join(values)
        return ""

    boxed = re.findall(r"\\boxed\{\s*(" + _NUMBER + r")\s*\}", region)
    if expected_count == 1 and boxed:
        return _normalise(boxed[-1])
    if expected_count is not None and expected_count > 1 and len(boxed) >= expected_count:
        return SEPARATOR.join(_normalise(part) for part in boxed[-expected_count:])
    if expected_count is None and boxed:
        return SEPARATOR.join(_normalise(part) for part in boxed)

    lines = [line.strip() for line in region.splitlines() if line.strip()]
    if expected_count in (None, 1) and lines and re.fullmatch(_NUMBER, lines[-1]):
        return _normalise(lines[-1])
    return ""


def score_masbench(output: Any, gold: Any) -> dict[str, Any]:
    expected = _numeric_parts(str(gold))
    if not expected:
        raise ValueError(f"MASBench gold answer is not a numeric scalar/vector: {gold!r}")
    prediction = extract_masbench_answer(output, expected_count=len(expected))
    predicted = prediction.split(SEPARATOR) if prediction else []
    correct_parts = [actual == wanted for actual, wanted in zip(predicted, expected)]
    correct_count = sum(correct_parts)
    component_accuracy = correct_count / len(expected)
    exact = bool(predicted) and len(predicted) == len(expected) and all(correct_parts)
    ordinal_score = 2 if exact else (1 if correct_count else 0)
    return {
        "prediction": prediction,
        "gold": SEPARATOR.join(expected),
        "parser_valid": bool(prediction),
        "correct": bool(exact),
        "score": 1.0 if exact else 0.0,
        "score_0_1_2": ordinal_score,
        "component_correct": correct_count,
        "component_total": len(expected),
        "component_accuracy": component_accuracy,
        "answer_parser": VERSION,
    }


def native_objective_score(output: Any, gold: Any) -> tuple[int, str]:
    """Return the shared ordinal score consumed by native search optimizers."""
    result = score_masbench(output, gold)
    return int(result["score_0_1_2"]), str(result["prediction"])
