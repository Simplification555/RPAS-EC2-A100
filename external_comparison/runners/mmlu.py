"""MMLU data boundary and shared evaluator for EC-2.

The runner intentionally exposes only the common execution boundary. Native
G-Designer integration is a separate adapter and must be present before a
formal EC-2 job can be submitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.phase2_wan_agent_search import (
    NetworkProfile,
    RolloutTrace,
    run_single_architecture,
)
from external_comparison.common.manifest import sha256_json
from external_comparison.common.schema import CallRecord

MMLU_SUBJECTS: tuple[str, ...] = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
)


@dataclass(frozen=True)
class MMLUExample:
    example_id: str
    subject: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str

    def prompt(self) -> str:
        labels = "ABCD"
        options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, self.choices, strict=True))
        return f"Subject: {self.subject}\nQuestion: {self.question}\n{options}"


def _source_path(data_dir: Path, subject: str, split: str) -> Path:
    candidates = (
        data_dir / split / f"{subject}_{split}.csv",
        data_dir / f"{subject}_{split}.csv",
        data_dir / split / f"{subject}.csv",
        data_dir / split / f"{subject}_{split}.parquet",
        data_dir / f"{subject}_{split}.parquet",
        data_dir / split / f"{subject}.parquet",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"MMLU {split} file for {subject} not found under {data_dir}")


def load_mmlu_subject(data_dir: str | Path, subject: str, split: str) -> list[MMLUExample]:
    path = _source_path(Path(data_dir), subject, split)
    rows: list[MMLUExample] = []
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency.
            raise RuntimeError("Reading MMLU parquet files requires pyarrow") from exc
        parquet_rows = pq.read_table(path).to_pylist()
        for index, value in enumerate(parquet_rows):
            choices = value.get("choices")
            answer = value.get("answer")
            if not isinstance(choices, list) or len(choices) != 4 or not isinstance(answer, int) or answer not in range(4):
                raise ValueError(f"malformed MMLU parquet row {path}:{index + 1}")
            rows.append(MMLUExample(f"mmlu:{split}:{subject}:{index}", subject, str(value["question"]), tuple(str(choice) for choice in choices), "ABCD"[answer]))
        if not rows:
            raise ValueError(f"MMLU file is empty: {path}")
        return rows
    with path.open(encoding="utf-8", newline="") as handle:
        for index, values in enumerate(csv.reader(handle)):
            if len(values) != 6:
                raise ValueError(f"MMLU row {path}:{index + 1} has {len(values)} fields, expected 6")
            question, *choices, answer = values
            if answer not in {"A", "B", "C", "D"}:
                raise ValueError(f"MMLU row {path}:{index + 1} has invalid answer {answer!r}")
            rows.append(MMLUExample(f"mmlu:{split}:{subject}:{index}", subject, question, tuple(choices), answer))
    if not rows:
        raise ValueError(f"MMLU file is empty: {path}")
    return rows


def load_mmlu_split(
    data_dir: str | Path,
    split: str,
    *,
    per_subject: int,
    seed: int,
) -> list[MMLUExample]:
    if per_subject < 1:
        raise ValueError("per_subject must be positive")
    rng = random.Random(seed)
    selected: list[MMLUExample] = []
    for subject in MMLU_SUBJECTS:
        rows = load_mmlu_subject(data_dir, subject, split)
        if per_subject > len(rows):
            raise ValueError(f"MMLU {split}/{subject} has {len(rows)} rows, need {per_subject}")
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        selected.extend(rows[index] for index in sorted(indices[:per_subject]))
    return selected


def build_mmlu_manifest(
    data_dir: str | Path,
    *,
    data_seed: int = 2026,
    search_per_subject: int = 5,
    select_per_subject: int = 5,
    test_per_subject: int = 10,
) -> dict[str, Any]:
    search = load_mmlu_split(data_dir, "dev", per_subject=search_per_subject, seed=data_seed)
    select = load_mmlu_split(data_dir, "val", per_subject=select_per_subject, seed=data_seed)
    test = load_mmlu_split(data_dir, "test", per_subject=test_per_subject, seed=data_seed)
    search_ids = [item.example_id for item in search]
    select_ids = [item.example_id for item in select]
    test_ids = [item.example_id for item in test]
    overlaps = {
        "search_select": set(search_ids) & set(select_ids),
        "search_test": set(search_ids) & set(test_ids),
        "select_test": set(select_ids) & set(test_ids),
    }
    if any(overlaps.values()):
        raise ValueError(f"MMLU split IDs overlap: {overlaps}")
    source_files = {}
    for split in ("dev", "val", "test"):
        for subject in MMLU_SUBJECTS:
            path = _source_path(Path(data_dir), subject, split)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source_files[f"{split}/{subject}"] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    return {
        "dataset": "mmlu",
        "data_seed": data_seed,
        "subjects": list(MMLU_SUBJECTS),
        "search": {"source_split": "dev", "per_subject": search_per_subject, "count": len(search), "ids": search_ids},
        "select": {"source_split": "val", "per_subject": select_per_subject, "count": len(select), "ids": select_ids},
        "test": {"source_split": "test", "per_subject": test_per_subject, "count": len(test), "ids": test_ids},
        "source_files": source_files,
        "split_manifest_sha256": sha256_json(
            {"search": search_ids, "select": select_ids, "test": test_ids}
        ),
    }


def parse_mmlu_choice(output: str) -> str:
    """Parse the final declared option, rejecting ambiguous free-form output."""

    cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.IGNORECASE | re.DOTALL)
    marked = re.findall(r"(?im)final\s+answer\s*:\s*\**\s*([ABCD])\b", cleaned)
    if marked:
        return marked[-1].upper()
    boxed = re.findall(r"(?i)\\boxed\s*\{\s*([ABCD])\s*\}", cleaned)
    if boxed:
        return boxed[-1].upper()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) == 1 and re.fullmatch(r"\**[ABCD]\**[.!]?", lines[0], flags=re.IGNORECASE):
        return lines[0].strip("*.").upper()
    if lines:
        marked_line = re.fullmatch(r"###\s*(?:answer\s*)?([ABCD])\b", lines[0], flags=re.IGNORECASE)
        if marked_line:
            return marked_line.group(1).upper()
        first_line = re.fullmatch(r"[*`]*([ABCD])[*`]*(?:[.):].*)?", lines[0], flags=re.IGNORECASE)
        if first_line:
            return first_line.group(1).upper()
    return ""


def summarize_communication(trace: RolloutTrace) -> dict[str, Any]:
    edges = {(message.src_agent, message.dst_agent) for message in trace.messages}
    return {
        "active_edges": len(edges),
        "messages": len(trace.messages),
        "inter_agent_tokens": sum(message.message_tokens for message in trace.messages),
        "average_message_tokens": (
            sum(message.message_tokens for message in trace.messages) / len(trace.messages)
            if trace.messages else 0.0
        ),
        "communication_rounds": max((index + 1 for index, _ in enumerate(trace.messages)), default=0),
    }


def evaluate_candidate(
    *,
    candidate: dict[str, Any],
    examples: Iterable[MMLUExample],
    models: dict[str, Any],
    profile: NetworkProfile,
    run_id: str,
    method: str,
    split: str,
    eval_concurrency: int = 1,
) -> dict[str, Any]:
    examples = list(examples)
    rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    communication = {"active_edges": 0, "messages": 0, "inter_agent_tokens": 0, "average_message_tokens": 0.0, "communication_rounds": 0}

    def evaluate_example(example: MMLUExample) -> tuple[MMLUExample, str, RolloutTrace]:
        output, trace = run_single_architecture(
            candidate=candidate,
            example={"id": example.example_id, "dataset": "mmlu", "input": example.prompt(), "answer": example.answer},
            models=models,
            profile=profile,
        )
        return example, output, trace

    worker_count = max(1, min(int(eval_concurrency), len(examples))) if examples else 1
    if worker_count == 1:
        evaluated = [evaluate_example(example) for example in examples]
    else:
        # vLLM batches concurrent requests while executor.map preserves the frozen sample order.
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="rpas-mmlu-eval") as executor:
            evaluated = list(executor.map(evaluate_example, examples))

    for example, output, trace in evaluated:
        prediction = parse_mmlu_choice(output)
        summary = trace.summary(profile)
        comm = summarize_communication(trace)
        communication["active_edges"] += comm["active_edges"]
        communication["messages"] += comm["messages"]
        communication["inter_agent_tokens"] += comm["inter_agent_tokens"]
        communication["communication_rounds"] += comm["communication_rounds"]
        rows.append({"example_id": example.example_id, "subject": example.subject, "prediction": prediction, "answer": example.answer, "correct": prediction == example.answer, "trace": summary})
        for index, call in enumerate(trace.calls):
            calls.append(CallRecord(run_id, method, "mmlu", split, str(candidate["id"]), f"{example.example_id}:{call.agent}:{index}", call.model, call.site, call.prompt_tokens, call.completion_tokens, call.total_tokens, call.input_cost_usd, call.output_cost_usd, call.inference_cost_usd, call.observed_latency_ms, call.observed_latency_ms, error=call.error, finish_reason=call.finish_reason).to_dict())
    count = len(rows)
    communication["average_message_tokens"] = communication["inter_agent_tokens"] / communication["messages"] if communication["messages"] else 0.0
    valid_answer_rate = sum(bool(row["prediction"]) for row in rows) / count if count else 0.0
    return {
        "accuracy": sum(row["correct"] for row in rows) / count if count else 0.0,
        "valid_answer_rate": valid_answer_rate,
        "valid": valid_answer_rate >= 0.99 and not any(call.get("error") for call in calls),
        "calls": sum(row["trace"]["calls"] for row in rows),
        "total_tokens": sum(row["trace"]["total_tokens"] for row in rows),
        "latency_ms": sum(float(row["trace"].get("emulated_wall_latency_ms", 0.0)) for row in rows),
        "communication": communication,
        "rows": rows,
        "calls_detail": calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and freeze the EC-2 MMLU split.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="outputs/external_comparison/mmlu/split_manifest.json")
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--search-per-subject", type=int, default=5)
    parser.add_argument("--select-per-subject", type=int, default=5)
    parser.add_argument("--test-per-subject", type=int, default=10)
    args = parser.parse_args()
    manifest = build_mmlu_manifest(
        args.data_dir,
        data_seed=args.data_seed,
        search_per_subject=args.search_per_subject,
        select_per_subject=args.select_per_subject,
        test_per_subject=args.test_per_subject,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
