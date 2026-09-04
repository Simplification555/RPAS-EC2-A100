"""Frozen HumanEval public-test tool shared by formal EC-1 methods.

The fixture is an AFlow-derived artifact, not a substitute for the official
HumanEval evaluator.  It may be used during search and before a held-out
completion is submitted, but never supplies the held-out answer labels.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublicTestResult:
    task_id: str
    entry_point: str
    passed: bool
    status: str
    runtime_ms: float
    feedback: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "entry_point": self.entry_point,
            "passed": self.passed,
            "status": self.status,
            "runtime_ms": self.runtime_ms,
            "feedback": self.feedback,
        }


class PublicTestExecutor:
    """Execute only the commit-pinned AFlow public test cases in a child process."""

    def __init__(self, fixture_path: str | Path, *, timeout_seconds: float = 10.0) -> None:
        self.path = Path(fixture_path).resolve()
        self.timeout_seconds = timeout_seconds
        self._tests: dict[str, tuple[str, tuple[str, ...]]] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = str(row.get("problem_id", "")).strip()
                entry_point = str(row.get("entry_point", "")).strip()
                tests = row.get("test")
                if not task_id or not entry_point or not isinstance(tests, list) or not all(isinstance(item, str) for item in tests):
                    raise ValueError(f"malformed public-test fixture: {self.path}:{line_number}")
                if task_id in self._tests:
                    raise ValueError(f"duplicate public-test task: {task_id}")
                self._tests[task_id] = (entry_point, tuple(tests))
        if not self._tests:
            raise ValueError(f"empty public-test fixture: {self.path}")

    def run(self, task_id: str, entry_point: str, code: str) -> PublicTestResult:
        fixture = self._tests.get(task_id)
        if fixture is None:
            raise KeyError(f"public-test fixture has no task {task_id}")
        fixture_entry_point, assertions = fixture
        if fixture_entry_point != entry_point:
            raise ValueError(
                f"public-test entry-point mismatch for {task_id}: {fixture_entry_point} != {entry_point}"
            )
        if not code.strip():
            return PublicTestResult(task_id, entry_point, False, "empty_or_unparseable", 0.0, "No parseable function was returned.")
        program = "\n".join((code, "", f"candidate = {entry_point}", *assertions, ""))
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="rpas_ec1_public_test_") as directory:
            script = Path(directory) / "candidate.py"
            script.write_text(program, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(script)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return PublicTestResult(
                    task_id, entry_point, False, "timeout", (time.perf_counter() - started) * 1000,
                    "The public-test execution timed out.",
                )
        feedback = (completed.stderr or completed.stdout or "public-test assertion failed").strip()[-1200:]
        return PublicTestResult(
            task_id,
            entry_point,
            completed.returncode == 0,
            "passed" if completed.returncode == 0 else "failed",
            (time.perf_counter() - started) * 1000,
            "" if completed.returncode == 0 else feedback,
        )

    def has_task(self, task_id: str) -> bool:
        """Whether the frozen AFlow artifact exposes a test for this task."""
        return task_id in self._tests

    @property
    def task_count(self) -> int:
        return len(self._tests)
