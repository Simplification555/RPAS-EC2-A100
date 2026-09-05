"""Rescore saved completions for diagnosis, never replace authoritative results."""

import argparse
import json
from pathlib import Path

from external_comparison.adapters.native_common import sha256_file
from external_comparison.runners.humaneval import execute_humaneval, extract_code, load_humaneval_tasks


def audit(result_path: Path, dataset_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    tasks = {task.task_id: task for task in load_humaneval_tasks(dataset_path)}
    comparisons = []
    for row in result["rows"]:
        task = tasks[row["task_id"]]
        raw = row["model_output"]
        code = extract_code(raw, task.entry_point)
        evaluation = execute_humaneval(code, task, timeout_seconds=10)
        comparisons.append({
            "task_id": task.task_id,
            "old_passed": row["passed"],
            "rescored_passed": evaluation["passed"],
            "code_changed": row.get("code") != code,
            "status": evaluation["status"],
        })
    return {
        "formal_result": False,
        "purpose": "Evaluator diagnosis only; original search used legacy extraction and is not replayed here.",
        "source_result_sha256": sha256_file(result_path),
        "dataset_sha256": sha256_file(dataset_path),
        "old_passed": sum(row["old_passed"] for row in comparisons),
        "rescored_passed": sum(row["rescored_passed"] for row in comparisons),
        "num_examples": len(comparisons),
        "rows": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in {args.result.resolve(), args.dataset.resolve()}:
        parser.error("diagnostic output must be a new, separate file")
    payload = audit(args.result, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}))
