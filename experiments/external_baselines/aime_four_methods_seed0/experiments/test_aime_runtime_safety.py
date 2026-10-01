"""Offline regression tests for bounded generated-code execution and AFlow guards."""

from __future__ import annotations

import asyncio
import ast
import csv
import os
from pathlib import Path
import time
from unittest.mock import patch

import pytest

import native_aime_formal as runner


def fake_run_code(code: str) -> tuple[str, str]:
    if code.startswith("hang:"):
        Path(code.split(":", 1)[1]).write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(30)
    if code == "hang":
        time.sleep(30)
    if code == "raise":
        raise ValueError("fixture failure")
    return "Success", code


def test_aime_contract_is_self_contained_and_not_masbench_metadata():
    contract = runner.task_contract_manifest()
    assert contract["version"] == "AIME_TASK_CONTRACT_v1"
    assert "MASBENCH" not in contract["version"]
    assert contract["sha256"] == runner.AIME_TASK_CONTRACT_SHA256


def test_output_audit_counts_native_attribute_and_worker_failures_only(tmp_path: Path):
    failures = [
        "'Workflow' object has no attribute 'custom'",
        "AttributeError: 'Workflow' object has no attribute 'custom'",
        "Code execution failed: ValueError: fixture failure",
    ]
    solutions = [
        "An error in the first calculation is corrected below.\nFINAL ANSWER: 7",
        "Error bounds give the same integer.\nFINAL ANSWER: 7",
        "A discarded approach reported 'Workflow' object has no attribute 'custom'.\nFINAL ANSWER: 7",
        "The previous code execution failed: its assumption was wrong.\nFINAL ANSWER: 7",
    ]
    output = tmp_path / "predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["prediction"])
        writer.writeheader()
        writer.writerows({"prediction": value} for value in failures + solutions)

    audit = runner.audit_aflow_outputs(tmp_path)
    assert audit["csv_rows"] == len(failures) + len(solutions)
    assert audit["failure_predictions"] == len(failures)
    assert audit["failure_markers"] == failures
    assert audit["failure_fraction"] == len(failures) / (len(failures) + len(solutions))


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"AIME_RUN_ID": "aime:seed_0", "MASBENCH_RUN_ID": "legacy"}, "aime:seed_0"),
        ({"AIME_RUN_ID": "aime:seed_0"}, "aime:seed_0"),
        ({"MASBENCH_RUN_ID": "legacy"}, "legacy"),
        ({"AIME_RUN_ID": "", "MASBENCH_RUN_ID": "legacy"}, "legacy"),
        ({}, ""),
    ],
)
def test_aime_run_id_precedence_and_legacy_fallback(environment, expected):
    with patch.dict(os.environ, environment, clear=True):
        assert runner.current_run_id() == expected


def test_process_executor_success_and_failure_are_cleanly_closed():
    limiter = runner.LoopBoundSemaphore(2)

    async def exercise():
        ok = await runner.execute_generated_code(fake_run_code, "ok", 5, limiter)
        bad = await runner.execute_generated_code(fake_run_code, "raise", 5, limiter)
        return ok, bad

    ok, bad = asyncio.run(exercise())
    assert ok == ("Success", "ok")
    assert bad[0] == "Error"
    assert "ValueError" in bad[1]


def track_terminated_workers():
    observed = []
    original = runner._shutdown_process_executor

    def wrapped(executor, *, terminate):
        processes = list((getattr(executor, "_processes", None) or {}).values())
        original(executor, terminate=terminate)
        if terminate:
            observed.extend(process.is_alive() for process in processes)

    return observed, wrapped


def test_process_executor_timeout_kills_runaway_worker_promptly(tmp_path: Path):
    limiter = runner.LoopBoundSemaphore(1)
    pid_file = tmp_path / "timeout-worker.pid"
    started = time.monotonic()
    async def exercise():
        task = asyncio.create_task(
            runner.execute_generated_code(fake_run_code, f"hang:{pid_file}", 1.2, limiter)
        )
        for _ in range(80):
            if pid_file.exists():
                break
            await asyncio.sleep(0.05)
        assert pid_file.exists(), "worker did not start before the timeout window"
        return await task

    observed, wrapped = track_terminated_workers()
    with patch.object(runner, "_shutdown_process_executor", wrapped):
        result = asyncio.run(exercise())
    elapsed = time.monotonic() - started
    assert result == ("Error", "Code execution timed out")
    assert elapsed < 6.0
    assert observed and not any(observed)


def test_process_executor_cancellation_kills_runaway_worker(tmp_path: Path):
    limiter = runner.LoopBoundSemaphore(1)
    pid_file = tmp_path / "cancelled-worker.pid"

    async def exercise():
        task = asyncio.create_task(
            runner.execute_generated_code(fake_run_code, f"hang:{pid_file}", 30, limiter)
        )
        for _ in range(80):
            if pid_file.exists():
                break
            await asyncio.sleep(0.05)
        assert pid_file.exists(), "worker did not start before cancellation"
        started = time.monotonic()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return time.monotonic() - started, int(pid_file.read_text(encoding="utf-8"))
        raise AssertionError("cancellation was swallowed")

    observed, wrapped = track_terminated_workers()
    with patch.object(runner, "_shutdown_process_executor", wrapped):
        elapsed, pid = asyncio.run(exercise())
    assert elapsed < 6.0
    assert observed and not any(observed)


def test_loop_limiter_can_be_reused_across_native_event_loops():
    limiter = runner.LoopBoundSemaphore(2)

    async def get_one():
        async with limiter.current():
            return True

    assert asyncio.run(get_one())
    assert asyncio.run(get_one())


def test_aflow_retry_patch_is_bounded_and_fails_closed(tmp_path: Path):
    optimizer_dir = tmp_path / "scripts"
    optimizer_dir.mkdir()
    optimizer_path = optimizer_dir / "optimizer.py"
    optimizer_path.write_text(
        """import asyncio

class Optimizer:
    async def _optimize_graph(self):
        # Create a loop until the generated graph meets the check conditions
        while True:
            if check:
                break
        # Save the graph and evaluate
        return 1
""",
        encoding="utf-8",
    )

    runner.patch_aflow_optimizer_retry_guard(tmp_path)
    patched = optimizer_path.read_text(encoding="utf-8")
    compile(patched, str(optimizer_path), "exec")
    assert "import os\n" in patched
    assert "while True:" not in patched
    assert "range(max_mutation_attempts)" in patched
    assert "failed to produce an accepted graph mutation" in patched
    runner.patch_aflow_optimizer_retry_guard(tmp_path)  # idempotent


def test_aflow_round_limit_includes_initial_round_and_last_mutation(tmp_path: Path):
    for round_number in range(1, 11):
        round_dir = tmp_path / "workflows" / f"round_{round_number}"
        round_dir.mkdir(parents=True)
        (round_dir / "graph.py").write_text("# offline fixture\n", encoding="utf-8")

    assert runner.aflow_search_rounds(tmp_path / "workflows", max_mutations=8) == list(range(1, 10))


def test_shared_aime_scorer_exports_uniform_ordinal_scores(tmp_path: Path):
    output = tmp_path / "predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question", "expected_output", "prediction"])
        writer.writeheader()
        writer.writerows([
            {"question": "p1", "expected_output": r"\boxed{7}", "prediction": "FINAL ANSWER: 7"},
            {"question": "p2", "expected_output": r"\boxed{8}", "prediction": "FINAL ANSWER: 9"},
            {"question": "p3", "expected_output": r"\boxed{3}", "prediction": "reasoning only: 3"},
        ])
    metrics = runner.shared_aime_metrics(output.parent, [
        {"problem": "p1", "answer": "7", "aime_id": "id1"},
        {"problem": "p2", "answer": "8", "aime_id": "id2"},
        {"problem": "p3", "answer": "3", "aime_id": "id3"},
    ])
    assert metrics["score_0_1_2"] == [2, 1, 0]
    assert metrics["mean_score_0_1_2"] == 1.0
    assert [row["id"] for row in metrics["rows"]] == ["id1", "id2", "id3"]

def test_aflow_native_search_score_uses_shared_aime_parser():
    class DummyBenchmark:
        calculate_score = runner.aflow_aime_calculate_score

    benchmark = DummyBenchmark()
    assert benchmark.calculate_score(r"\boxed{7}", "FINAL ANSWER: 7") == (1, "7")
    assert benchmark.calculate_score(r"\boxed{7}", "FINAL ANSWER: 8") == (0, "8")
    assert benchmark.calculate_score(r"\boxed{7}", "reasoning without an answer") == (0, "")
    try:
        benchmark.calculate_score("unparseable reference", "FINAL ANSWER: 7")
    except ValueError as exc:
        assert "reference answer is not parseable" in str(exc)
    else:
        raise AssertionError("unparseable AIME gold must fail closed")
