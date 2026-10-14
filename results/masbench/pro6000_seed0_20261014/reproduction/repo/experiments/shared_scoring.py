"""Frozen RPAS parser/scorer and common answer-format instructions; no model dependencies."""
from __future__ import annotations
import os
import re

SOURCE_SHA256 = '249cb23f6a71c62f4a15d62abbe1ea467ee28f862d175dfe5a0bf2fd2cc8240b'

MASBENCH_ANSWER_SEPARATOR = "<<horizon>>"

ANSWER_FIRST_INSTRUCTION = (
    "Solve the problem with concise reasoning. End with a separate final line exactly `### ANSWER`, "
    "where ANSWER is only the final numeric answer or simplified fraction.\n\n"
)

DELIBERATE_INSTRUCTION = (
    "Solve the problem carefully. You may write concise reasoning, but avoid hidden chain-of-thought tags. "
    "End your response with a separate final line exactly in the format `### ANSWER`, where ANSWER is only "
    "the final numeric answer or simplified fraction.\n\n"
)

THINKING_DELIBERATE_INSTRUCTION = (
    "Solve the problem carefully. You may use the model's normal mathematical reasoning mode. "
    "End your response with a separate final line exactly in the format `### ANSWER`, where ANSWER is only "
    "the final numeric answer or simplified fraction.\n\n"
)

THINKING_ANSWER_FIRST_INSTRUCTION = (
    "Solve the problem. Put the final answer on its own final line exactly in the format `### ANSWER`, "
    "where ANSWER is only the final numeric answer or simplified fraction.\n\n"
)

GAIA_INSTRUCTION = (
    "Answer the GAIA task using concise, tool-aware reasoning. If an attachment path is provided, use it only "
    "when the current runner exposes file/tool access; otherwise state the best answer from the visible prompt. "
    "End your response with a separate final line exactly in the format `FINAL ANSWER: ANSWER`. The final answer "
    "should be a number, a few words, or a comma-separated list as requested by the question.\n\n"
)

MASBENCH_INSTRUCTION = (
    "Solve every subproblem in the MASBench task carefully and preserve the requested answer order. Follow only "
    "the dependency paths needed for the requested quantities; do not restate every fact or repeatedly speculate "
    "about missing definitions. For iGSM arithmetic, every quantity is in Z_23: reduce additions, subtractions, "
    "multiplications, and intermediate results modulo 23 to an integer from 0 through 22. A plural category or "
    "abstract total denotes the modulo-23 sum of its direct listed members. "
    f"If there are multiple answers, join them using exactly `{MASBENCH_ANSWER_SEPARATOR}` with no omitted "
    "items or reordering. End with one separate line `FINAL ANSWER: ANSWER`; ANSWER must contain only the "
    "single answer or the ordered separator-delimited answer sequence.\n\n"
)

def normalize_numeric_token(token: str) -> str:
    token = token.strip()
    if re.fullmatch(r"[-+]?\d+", token):
        return str(int(token))
    return token

def extract_final_answer(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).replace("\r\n", "\n")
    gaia_matches = list(re.finditer(r"(?im)^FINAL\s+ANSWER\s*:\s*(.+?)\s*$", cleaned))
    if gaia_matches:
        return gaia_matches[-1].group(1).strip()
    matches = list(re.finditer(r"(?m)^###\s*(.+?)\s*$", cleaned))
    if matches:
        payload = matches[-1].group(1).strip()
        if payload.upper() == "ANSWER":
            following = cleaned[matches[-1].end() :].strip()
            numeric_tokens = re.findall(r"[-+]?\d+(?:/\d+)?", following)
            if numeric_tokens:
                return normalize_numeric_token(numeric_tokens[0])
        if "=" in payload:
            rhs = payload.rsplit("=", 1)[-1].strip()
            if re.fullmatch(r"[-+]?\d+(?:/\d+)?", rhs):
                payload = rhs
        numeric_tokens = re.findall(r"[-+]?\d+(?:/\d+)?", payload)
        if numeric_tokens:
            return normalize_numeric_token(numeric_tokens[-1])
        return payload.strip()
    numeric_tokens = re.findall(r"[-+]?\d+(?:/\d+)?", cleaned)
    return normalize_numeric_token(numeric_tokens[-1]) if numeric_tokens else ""

def task_instruction(dataset: str | None = None) -> str:
    if dataset == "gaia":
        return GAIA_INSTRUCTION
    if dataset == "masbench":
        return MASBENCH_INSTRUCTION
    mode = os.environ.get("GEPA_RPAS_PROMPT_MODE", "deliberate").lower()
    if mode in {"think", "thinking", "think_deliberate", "cot"}:
        return THINKING_DELIBERATE_INSTRUCTION
    if mode in {"think_answer_first", "thinking_answer_first"}:
        return THINKING_ANSWER_FIRST_INSTRUCTION
    if mode in {"answer_first", "fast", "short"}:
        return ANSWER_FIRST_INSTRUCTION
    return DELIBERATE_INSTRUCTION

def score_exact_answer(output: str, answer: str) -> float:
    expected = extract_final_answer(str(answer))
    if not expected:
        expected = normalize_numeric_token(str(answer).replace("###", "").strip())
    actual = extract_explicit_math_answer(output)
    return 1.0 if actual and expected and actual == expected else 0.0

def normalize_gaia_answer(text: str) -> str:
    normalized = extract_final_answer(text) or str(text)
    normalized = re.sub(r"<think>.*?</think>", "", normalized, flags=re.DOTALL)
    normalized = normalized.strip()
    normalized = re.sub(r"^(final\s+answer\s*:|###)\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip().strip("`").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    return normalized.strip().rstrip(".").lower()

def normalize_masbench_answer(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
    # MASBench's embedded instructions also use literal \\n\\n separators.
    # Accept only an entirely numeric sequence; never mine reasoning for answers.
    sequence = re.sub(r"^FINAL\s+ANSWER\s*:\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    alternate = re.split(r"\\n\\n", sequence)
    if len(alternate) > 1 and all(re.fullmatch(r"\s*[+-]?\d+(?:\.\d+)?\s*", part) for part in alternate):
        cleaned = MASBENCH_ANSWER_SEPARATOR.join(part.strip() for part in alternate)
    has_sequence = bool(re.search(r"<<\s*horizon\s*>>", cleaned, flags=re.IGNORECASE))
    normalized = extract_final_answer(cleaned) if not has_sequence else cleaned
    final_matches = list(re.finditer(r"(?im)^FINAL\s+ANSWER\s*:\s*(.+?)\s*$", cleaned))
    if final_matches:
        normalized = final_matches[-1].group(1).strip()
    normalized = re.sub(r"^(final\s+answer\s*:|###)\s*", "", normalized.strip(), flags=re.IGNORECASE)
    normalized = normalized.strip().strip("`").strip()
    parts = re.split(r"\s*<<\s*horizon\s*>>\s*", normalized, flags=re.IGNORECASE)
    normalized_parts = []
    for part in parts:
        compact = re.sub(r"\s+", " ", part).strip().rstrip(".")
        normalized_parts.append(normalize_numeric_token(compact))
    return MASBENCH_ANSWER_SEPARATOR.join(normalized_parts)

def extract_masbench_final_answer(text: str) -> str:
    cleaned = strip_reasoning_blocks(text)
    matches = list(re.finditer(r"(?im)^FINAL\s+ANSWER\s*:\s*(.+?)\s*$", cleaned))
    return matches[-1].group(1).strip() if matches else ""

def score_example_answer(output: str, answer: str, dataset: str | None = None) -> float:
    if dataset == "gaia":
        expected = normalize_gaia_answer(answer)
        actual = normalize_gaia_answer(output)
        return 1.0 if actual and expected and actual == expected else 0.0
    if dataset == "masbench":
        explicit_answer = extract_masbench_final_answer(output)
        if not explicit_answer:
            return 0.0
        expected = normalize_masbench_answer(answer)
        actual = normalize_masbench_answer(explicit_answer)
        return 1.0 if actual and expected and actual == expected else 0.0
    return score_exact_answer(output, answer)

def score_answer_components(output: str, answer: str, dataset: str | None = None) -> float:
    if dataset != "masbench" or MASBENCH_ANSWER_SEPARATOR not in str(answer):
        return score_example_answer(output, answer, dataset)
    explicit_answer = extract_masbench_final_answer(output)
    if not explicit_answer:
        return 0.0
    expected_parts = normalize_masbench_answer(answer).split(MASBENCH_ANSWER_SEPARATOR)
    actual_parts = normalize_masbench_answer(explicit_answer).split(MASBENCH_ANSWER_SEPARATOR)
    correct = sum(
        expected == actual
        for expected, actual in zip(expected_parts, actual_parts, strict=False)
    )
    return correct / len(expected_parts) if expected_parts else 0.0

def extract_prediction_for_dataset(output: str, dataset: str | None = None) -> str:
    if dataset == "gaia":
        return normalize_gaia_answer(output)
    if dataset == "masbench":
        explicit_answer = extract_masbench_final_answer(output)
        return normalize_masbench_answer(explicit_answer) if explicit_answer else ""
    return extract_explicit_math_answer(output)

def strip_reasoning_blocks(text: str) -> str:
    return re.sub(r"<think>.*?(?:</think>|$)", "", str(text), flags=re.DOTALL | re.IGNORECASE).strip()

def extract_explicit_math_answer(text: str) -> str:
    """Never infer an answer from the last number in unfinished reasoning."""
    cleaned = strip_reasoning_blocks(text)
    number = r"[-+]?\d+(?:/\d+)?"
    if re.fullmatch(number, cleaned):
        return normalize_numeric_token(cleaned)
    markers = list(re.finditer(r"(?im)(?:###\s*|^FINAL\s+ANSWER\s*:\s*)([^\n]*)", cleaned))
    if markers:
        marker = markers[-1]
        payload = marker.group(1).strip()
        payload = re.sub(r"(?i)^ANSWER[ \t]*:?[ \t]+(?=[-+]?\d)", "", payload)
        if payload.upper() == "ANSWER":
            following = cleaned[marker.end():].strip().splitlines()
            payload = following[0].strip() if following else ""
        if re.fullmatch(number, payload):
            return normalize_numeric_token(payload)
        return ""
    boxed = re.findall(r"\\boxed\{\s*(" + number + r")\s*\}", cleaned)
    return normalize_numeric_token(boxed[-1]) if boxed else ""
