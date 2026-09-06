"""Fail-closed preflight for the frozen EC-1B LiveCodeBench bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from external_comparison.adapters.native_common import git_commit, sha256_file
from external_comparison.runners.livecodebench_ec1 import (
    _decode_cases,
    evaluate_code,
    load_tasks,
    validate_frozen_bundle,
)


def _public_oracle(task) -> str:
    """Build a public-fixture-only probe to verify both evaluator ABIs."""
    cases = _decode_cases(task.public_test_cases)
    if task.function_name:
        table = {}
        for case in cases:
            arguments = [json.loads(line) for line in case["input"].splitlines()]
            table[repr(arguments)] = json.loads(case["output"])
        return (
            "class Solution:\n"
            f"    def {task.function_name}(self, *args):\n"
            f"        table = {table!r}\n"
            "        return table[repr(list(args))]\n"
        )
    table = {str(case["input"]): str(case["output"]) for case in cases}
    return (
        "import sys\n"
        f"table = {table!r}\n"
        "sys.stdout.write(table[sys.stdin.read()])\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--aflow-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_frozen_bundle(args.data_dir)
    commit = git_commit(args.aflow_root)
    if commit != "3f457218fc716093fe53f6df8a5d5e6379d66346":
        parser.error(f"AFlow commit mismatch: {commit}")
    os.environ["RPAS_AFLOW_ROOT"] = str(args.aflow_root.resolve())
    calibration = load_tasks(args.data_dir / "calib.jsonl")
    typed_tasks = {
        mode: next((task for task in calibration if task.execution_mode == mode), None)
        for mode in ("stdin", "functional")
    }
    if any(task is None for task in typed_tasks.values()):
        parser.error("calibration split must contain both stdin and functional LiveCodeBench tasks")
    probes = {
        mode: evaluate_code(task, _public_oracle(task), public=True)
        for mode, task in typed_tasks.items()
    }
    failed = {mode: probe for mode, probe in probes.items() if not probe["passed"]}
    if failed:
        parser.error(f"shared LCB evaluator failed typed public oracle probe: {failed}")
    if not callable(os.chdir) or not callable(subprocess.Popen):
        parser.error("shared LCB evaluator poisoned its parent process")
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_sha256": sha256_file(args.data_dir / "DATASET_MANIFEST.json"),
                "split_counts": {name: value["count"] for name, value in manifest["splits"].items()},
                "aflow_commit": commit,
                "evaluator_probe_statuses": {
                    mode: probe["status"] for mode, probe in probes.items()
                },
                "evaluator_process_isolation": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
