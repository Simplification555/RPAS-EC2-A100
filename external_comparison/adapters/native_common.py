"""Small runtime helpers shared by the native external adapters."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(value)
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def split_rows(rows: list[dict[str, Any]], seed: int, search: int, select: int, test: int) -> dict[str, list[dict[str, Any]]]:
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    requested = search + select + test
    if requested > len(ordered):
        raise ValueError(f"requested {requested} rows, found {len(ordered)}")
    return {
        "search": ordered[:search],
        "select": ordered[search : search + select],
        "test": ordered[search + select : requested],
    }


def humaneval_external_split(rows: list[dict[str, Any]], data_seed: int) -> dict[str, list[dict[str, Any]]]:
    """Apply the frozen EC-1 split: 33 development and 131 held-out tasks."""
    return split_rows(rows, data_seed, search=33, select=0, test=131)


def load_mmlu_csv(data_dir: str | Path, split: str) -> list[dict[str, Any]]:
    root = Path(data_dir)
    rows: list[dict[str, Any]] = []
    paths = sorted((root / split).glob("*.csv")) if (root / split).exists() else sorted(root.glob(f"*_{split}.csv"))
    if not paths:
        parquet_paths = sorted((root / split).glob("*.parquet")) if (root / split).exists() else sorted(root.glob(f"*_{split}.parquet"))
        if parquet_paths:
            try:
                import pyarrow.parquet as pq
            except ModuleNotFoundError as exc:
                raise RuntimeError("MMLU parquet input requires pyarrow") from exc
            for path in parquet_paths:
                for index, value in enumerate(pq.read_table(path).to_pylist()):
                    choices = value.get("choices")
                    answer = value.get("answer")
                    if not isinstance(choices, list) or len(choices) != 4 or not isinstance(answer, int) or answer not in range(4):
                        raise ValueError(f"malformed MMLU parquet row: {path}:{index + 1}")
                    rows.append({"id": f"mmlu:{split}:{path.stem}:{index}", "subject": path.stem, "question": str(value["question"]), "choices": [str(choice) for choice in choices], "answer": "ABCD"[answer]})
            return rows
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for index, values in enumerate(csv.reader(handle)):
                if len(values) != 6 or values[-1] not in {"A", "B", "C", "D"}:
                    raise ValueError(f"malformed MMLU row: {path}:{index + 1}")
                question, a, b, c, d, answer = values
                rows.append({"id": f"mmlu:{split}:{path.stem}:{index}", "subject": path.stem, "question": question, "choices": [a, b, c, d], "answer": answer})
    if not rows:
        raise ValueError(f"no MMLU CSV files under {root}")
    return rows


def stratified_mmlu(data_dir: str | Path, split: str, per_subject: int, seed: int) -> list[dict[str, Any]]:
    rows = load_mmlu_csv(data_dir, split)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["subject"]), []).append(row)
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    for subject in sorted(groups):
        group = list(groups[subject])
        rng.shuffle(group)
        result.extend(group[:per_subject])
    return result


def extract_code(output: str, entry_point: str) -> str:
    candidates = [output]
    marker = f"def {entry_point}"
    if "```" in output:
        parts = output.split("```")
        candidates.extend(parts[1::2])
    for candidate in candidates:
        if marker not in candidate:
            continue
        candidate = candidate[candidate.find(marker) :].strip()
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point for node in tree.body):
            return candidate
    return ""


def execute_humaneval(code: str, task: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    if not code:
        return {"passed": False, "status": "empty_or_unparseable", "runtime_ms": 0.0}
    program = f"{code}\n\n{task['test']}\n\ncheck({task['entry_point']})\n"
    with tempfile.TemporaryDirectory(prefix="rpas_native_he_") as directory:
        script = Path(directory) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        started = time.perf_counter()
        try:
            completed = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"passed": False, "status": "timeout", "runtime_ms": (time.perf_counter() - started) * 1000}
    return {"passed": completed.returncode == 0, "status": "passed" if completed.returncode == 0 else "failed", "stderr": completed.stderr[-1000:], "runtime_ms": (time.perf_counter() - started) * 1000}


def call_record(run_id: str, method: str, dataset: str, split: str, candidate_id: str, index: int, usage: dict[str, Any]) -> dict[str, Any]:
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
    total = int(usage.get("total_tokens", prompt + completion))
    return {
        "run_id": run_id, "method": method, "dataset": dataset, "split": split,
        "candidate_id": candidate_id, "agent": str(usage.get("agent", "native")),
        "model": str(usage.get("model", os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"))),
        "site": "center_a", "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": total, "input_cost": 0.0, "output_cost": 0.0,
        "inference_cost": 0.0, "model_latency_ms": float(usage.get("latency_ms", 0.0)),
        "wall_latency_ms": float(usage.get("latency_ms", 0.0)), "network_latency_ms": 0.0,
        "retry_count": int(usage.get("retry_count", 0)), "finish_reason": usage.get("finish_reason"),
        "error": usage.get("error"),
    }


def require_valid_answer_rate(
    rows: list[dict[str, Any]],
    *,
    context: str,
    minimum: float = 0.99,
) -> float:
    """Fail fast when a completed run contains mostly unparseable answers."""

    prediction_rows = [row for row in rows if "prediction" in row]
    if not prediction_rows:
        return 1.0
    valid_rate = sum(bool(row.get("prediction")) for row in prediction_rows) / len(prediction_rows)
    if valid_rate < minimum:
        raise RuntimeError(
            f"{context} produced an invalid answer rate of {valid_rate:.3f}; "
            f"minimum required is {minimum:.3f}"
        )
    return valid_rate


def write_native_result(
    output_dir: str | Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    selected: dict[str, Any] | None = None,
    search_rows: list[dict[str, Any]] | None = None,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (root / "search_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in search_rows or []:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (root / "calls.jsonl").open("w", encoding="utf-8") as handle:
        for record in calls:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with (root / "test_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    prediction_rows = [row for row in rows if "prediction" in row]
    valid_answer_rate = (
        sum(bool(row.get("prediction")) for row in prediction_rows) / len(prediction_rows)
        if prediction_rows
        else None
    )
    test_calls = [row for row in calls if row.get("split") == "test"]
    summary = {
        "method": manifest["method"], "dataset": manifest["dataset"], "score": sum(bool(row.get("correct", row.get("passed", False))) for row in rows) / len(rows) if rows else 0.0,
        "num_examples": len(rows), "valid_answer_rate": valid_answer_rate,
        # Search/selection records are retained in calls.jsonl, but never
        # silently folded into held-out inference cost.
        "inference_calls": len(test_calls), "inference_tokens": sum(int(row.get("total_tokens", 0)) for row in test_calls),
        "model_errors": sum(bool(row.get("error")) for row in calls),
        "maxed_calls": sum(row.get("finish_reason") == "length" for row in calls),
        "search_calls": int(manifest.get("search_calls", 0)), "search_tokens": int(manifest.get("search_tokens", 0)),
    }
    (root / "result.json").write_text(json.dumps({**manifest, "summary": summary, "selected": selected, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "summary.csv").write_text(",".join(summary) + "\n" + ",".join(str(summary[key]) for key in summary) + "\n", encoding="utf-8")


def git_commit(repo: Path) -> str:
    completed = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return completed.stdout.strip()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
