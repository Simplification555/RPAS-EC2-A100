"""Frozen LiveCodeBench release-v6 data and evaluator boundary for EC-1B.

The model and architecture search only receive ``question_content`` and the
public tests.  Private tests remain inside this evaluator module and are never
placed in reflection payloads or model prompts.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import pickle
import random
import re
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from external_comparison.adapters.native_common import load_jsonl, sha256_file


DATASET_REPOSITORY = "livecodebench/code_generation_lite"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RELEASE_VERSION = "release_v6"
SOURCE_FILENAME = "test6.jsonl"
SPLIT_SIZES = {"calib": 20, "search": 64, "select": 64, "test": 256}
DATA_SEED = 2026


@dataclass(frozen=True)
class LiveCodeBenchTask:
    question_id: str
    question: str
    difficulty: str
    platform: str
    starter_code: str
    public_test_cases: str
    private_test_cases: str
    metadata: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "LiveCodeBenchTask":
        if "question_content" not in row and "question" in row:
            row = {**row, "question_content": row["question"]}
        required = ("question_id", "question_content", "public_test_cases", "private_test_cases")
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"malformed LiveCodeBench row; missing {missing}")
        return cls(
            question_id=str(row["question_id"]),
            question=str(row["question_content"]),
            difficulty=str(row.get("difficulty", "unknown")),
            platform=str(row.get("platform", "unknown")),
            starter_code=str(row.get("starter_code", "")),
            public_test_cases=str(row["public_test_cases"]),
            private_test_cases=str(row["private_test_cases"]),
            metadata=str(row.get("metadata", "{}")),
        )

    def to_raw_row(self) -> dict[str, str]:
        return {
            "question_id": self.question_id,
            "question_content": self.question,
            "difficulty": self.difficulty,
            "platform": self.platform,
            "starter_code": self.starter_code,
            "public_test_cases": self.public_test_cases,
            "private_test_cases": self.private_test_cases,
            "metadata": self.metadata,
        }

    @property
    def function_name(self) -> str | None:
        return _metadata_func_name(self)

    @property
    def execution_mode(self) -> str:
        return "functional" if self.function_name else "stdin"


def render_model_prompt(task: LiveCodeBenchTask) -> str:
    """Render the public task contract without exposing evaluator-only tests."""
    if task.function_name:
        starter = task.starter_code.strip()
        suffix = f"\n\nStarter code:\n{starter}" if starter else ""
        return task.question.rstrip() + suffix
    return task.question


def generation_instruction(task: LiveCodeBenchTask) -> str:
    if task.function_name:
        return (
            "Solve the programming problem in Python 3. Return only complete Python code that preserves the "
            f"provided class/function contract and implements `{task.function_name}`. Do not read standard input, "
            "print an answer, use markdown, or add explanations."
        )
    return (
        "Solve the programming problem in Python 3. Return only the complete executable program, including "
        "imports. Read standard input and write the required output to standard output. Do not use markdown or "
        "explanations."
    )


def load_tasks(path: str | Path) -> list[LiveCodeBenchTask]:
    tasks = [LiveCodeBenchTask.from_row(row) for row in load_jsonl(path)]
    ids = [task.question_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate LiveCodeBench question_id in {path}")
    return tasks


def _decode_cases(value: str) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        # This is the encoding used by the pinned AFlow/LiveCodeBench loader.
        decoded = pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8"))))
        if isinstance(decoded, str):
            decoded = json.loads(decoded)
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("LiveCodeBench test payload must be a non-empty list")
    for case in decoded:
        if not isinstance(case, dict) or "input" not in case or "output" not in case:
            raise ValueError("malformed LiveCodeBench test case")
    return decoded


def extract_code(output: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(output), flags=re.DOTALL).strip()
    fenced = re.findall(r"```(?:python)?\s*\n?(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    code = fenced[-1].strip() if fenced else cleaned
    return code if code and not code.startswith("Error:") else ""


def _metadata_func_name(task: LiveCodeBenchTask) -> str | None:
    try:
        value = json.loads(task.metadata or "{}")
    except json.JSONDecodeError:
        value = {}
    name = value.get("func_name") if isinstance(value, dict) else None
    return str(name) if name else None


def _runner() -> Callable[..., tuple[list[Any], dict[str, Any]]]:
    root = Path(os.environ.get("RPAS_AFLOW_ROOT", "/home/jianbaizhao/external_baselines/AFlow_formal_clean"))
    if not root.is_dir():
        raise FileNotFoundError(f"pinned AFlow checkout is required for the shared LCB evaluator: {root}")
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("scripts.utils.lcb_runner")
    return module.run_test


def evaluate_code(task: LiveCodeBenchTask, code: str, *, public: bool, timeout_seconds: int = 6) -> dict[str, Any]:
    if not code.strip():
        return {"passed": False, "status": "empty_or_unparseable", "test_results": [], "metadata": {}}
    cases = _decode_cases(task.public_test_cases if public else task.private_test_cases)
    sample = {
        "question": task.question,
        "question_id": task.question_id,
        "input_output": json.dumps(
            {
                "inputs": [case["input"] for case in cases],
                "outputs": [case["output"] for case in cases],
                "fn_name": _metadata_func_name(task),
            }
        ),
    }
    payload = {"sample": sample, "code": code, "timeout_seconds": timeout_seconds}
    hard_timeout = max(30.0, float(timeout_seconds) * 5.0)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "external_comparison.runners.livecodebench_eval_worker"],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=hard_timeout,
            check=False,
            start_new_session=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"isolated evaluator exited {completed.returncode}: {completed.stderr[-1000:]}"
            )
        worker = json.loads(completed.stdout)
        if worker.get("protocol") != "rpas_lcb_isolated_eval_v1":
            raise ValueError("isolated evaluator returned an unknown protocol")
        results = worker["results"]
        metadata = worker.get("metadata", {})
        normalized = [bool(value) for value in results]
        passed = bool(normalized) and all(normalized)
        return {
            "passed": passed,
            "status": "passed" if passed else "failed",
            "test_results": normalized,
            "metadata": metadata,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "status": "timeout",
            "test_results": [],
            "metadata": {"hard_timeout_seconds": hard_timeout},
        }
    except Exception as exc:
        return {
            "passed": False,
            "status": "evaluator_error",
            "test_results": [],
            "metadata": {"error": type(exc).__name__, "message": str(exc)[:1000]},
        }


def _stratum(task: LiveCodeBenchTask) -> tuple[str, str]:
    return task.difficulty.lower(), task.platform.lower()


def frozen_split(tasks: Iterable[LiveCodeBenchTask], *, seed: int = DATA_SEED) -> dict[str, list[LiveCodeBenchTask]]:
    groups: dict[tuple[str, str], list[LiveCodeBenchTask]] = {}
    for task in tasks:
        groups.setdefault(_stratum(task), []).append(task)
    rng = random.Random(seed)
    ordered: list[LiveCodeBenchTask] = []
    shuffled = {key: sorted(value, key=lambda task: task.question_id) for key, value in groups.items()}
    for value in shuffled.values():
        rng.shuffle(value)
    while any(shuffled.values()):
        for key in sorted(shuffled):
            if shuffled[key]:
                ordered.append(shuffled[key].pop())
    required = sum(SPLIT_SIZES.values())
    if len(ordered) < required:
        raise ValueError(f"release_v6 requires at least {required} rows, found {len(ordered)}")
    result: dict[str, list[LiveCodeBenchTask]] = {}
    offset = 0
    for name, size in SPLIT_SIZES.items():
        result[name] = ordered[offset : offset + size]
        offset += size
    return result


def validate_frozen_bundle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "DATASET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_revision") != DATASET_REVISION or manifest.get("release_version") != RELEASE_VERSION:
        raise ValueError("LiveCodeBench manifest revision/release mismatch")
    seen: set[str] = set()
    for name, expected in SPLIT_SIZES.items():
        path = root / f"{name}.jsonl"
        tasks = load_tasks(path)
        if len(tasks) != expected:
            raise ValueError(f"LCB {name} expected {expected} rows, found {len(tasks)}")
        ids = {task.question_id for task in tasks}
        if seen & ids:
            raise ValueError(f"LCB split overlap in {name}")
        seen |= ids
        recorded = manifest["splits"][name]
        if recorded["sha256"] != sha256_file(path) or recorded["question_ids"] != [task.question_id for task in tasks]:
            raise ValueError(f"LCB {name} differs from frozen manifest")
    return manifest
