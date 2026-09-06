from __future__ import annotations

import json
import os
import py_compile
import subprocess
from pathlib import Path

import pytest

from external_comparison.adapters.native_telemetry import NativeCallRecorder
from external_comparison.runners.native_aflow_livecodebench import _stage_lcb
from external_comparison.runners import native_rpas_livecodebench
from external_comparison.runners.livecodebench_ec1 import (
    SPLIT_SIZES,
    LiveCodeBenchTask,
    _decode_cases,
    evaluate_code,
    extract_code,
    frozen_split,
    generation_instruction,
    render_model_prompt,
)
from experiments import phase2_wan_agent_search as native_search


def task(index: int, difficulty: str = "hard", platform: str = "codeforces") -> LiveCodeBenchTask:
    cases = json.dumps([{"input": str(index), "output": str(index)}])
    return LiveCodeBenchTask(
        question_id=f"problem_{index:04d}",
        question=f"Echo {index}",
        difficulty=difficulty,
        platform=platform,
        starter_code="",
        public_test_cases=cases,
        private_test_cases=cases,
        metadata="{}",
    )


def test_frozen_split_has_protocol_sizes_and_no_overlap() -> None:
    tasks = [task(index, "hard" if index % 2 else "medium", "atcoder" if index % 3 else "codeforces") for index in range(500)]
    first = frozen_split(tasks)
    second = frozen_split(tasks)
    assert {name: len(rows) for name, rows in first.items()} == SPLIT_SIZES
    assert {name: [row.question_id for row in rows] for name, rows in first.items()} == {
        name: [row.question_id for row in rows] for name, rows in second.items()
    }
    ids = [row.question_id for rows in first.values() for row in rows]
    assert len(ids) == len(set(ids)) == sum(SPLIT_SIZES.values())


def test_task_serialization_keeps_aflow_schema() -> None:
    row = task(1).to_raw_row()
    assert "question_content" in row
    assert "question" not in row
    assert LiveCodeBenchTask.from_row(row) == task(1)


def test_code_extraction_preserves_complete_stdin_program() -> None:
    output = "explanation\n```python\nimport sys\nprint(sys.stdin.read())\n```"
    assert extract_code(output) == "import sys\nprint(sys.stdin.read())"


def test_prompt_contract_distinguishes_stdin_and_functional_tasks() -> None:
    stdin = task(1)
    functional = LiveCodeBenchTask(
        **{
            **stdin.__dict__,
            "starter_code": "class Solution:\n    def echo(self, value):\n        ",
            "metadata": json.dumps({"func_name": "echo"}),
        }
    )
    assert stdin.execution_mode == "stdin"
    assert "standard input" in generation_instruction(stdin)
    assert functional.execution_mode == "functional"
    assert "Do not read standard input" in generation_instruction(functional)
    assert functional.starter_code.strip() in render_model_prompt(functional)


def test_json_test_cases_decode_without_private_deserialization() -> None:
    value = json.dumps([{"input": "1", "output": "2", "testtype": "stdin"}])
    assert _decode_cases(value)[0]["output"] == "2"


def test_evaluator_isolated_process_does_not_poison_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream"
    package = upstream / "scripts" / "utils"
    package.mkdir(parents=True)
    (upstream / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "lcb_runner.py").write_text(
        "def clean_if_name(code): return code\n"
        "def make_function(code): return code\n"
        "def run_test(sample, test, debug, timeout):\n"
        "    import os, subprocess, sys\n"
        "    assert sys.stderr.fileno() >= 0\n"
        "    os.chdir = None\n"
        "    subprocess.Popen = None\n"
        "    return [1], {'isolated': True}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RPAS_AFLOW_ROOT", str(upstream))
    original_chdir = os.chdir
    original_popen = subprocess.Popen
    for _ in range(2):
        result = evaluate_code(task(1), "print(input())", public=True)
        assert result["passed"] is True
        assert result["metadata"] == {"isolated": True}
    assert os.chdir is original_chdir
    assert subprocess.Popen is original_popen


def test_native_recorder_can_label_livecodebench(tmp_path: Path) -> None:
    recorder = NativeCallRecorder(tmp_path / "calls.jsonl", method="aflow", seed=0, dataset="livecodebench")
    assert recorder.dataset == "livecodebench"


def test_aflow_task_adapter_materializes_importable_initial_graph(tmp_path: Path) -> None:
    human = tmp_path / "workspace" / "HumanEval" / "workflows"
    (human / "round_1").mkdir(parents=True)
    (human / "template").mkdir()
    (human / "round_1" / "graph.py").write_text("import workspace.HumanEval\n", encoding="utf-8")
    (human / "template" / "operator.py").write_text("# workspace.HumanEval\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("search.jsonl", "select.jsonl", "test.jsonl"):
        (bundle / name).write_text(json.dumps(task(1).to_raw_row()) + "\n", encoding="utf-8")
    _stage_lcb(tmp_path, bundle)
    graph = tmp_path / "workspace" / "LiveCodeBench" / "workflows" / "round_1" / "graph.py"
    py_compile.compile(str(graph), doraise=True)
    assert "async def __call__(self, problem, entry_point, question_id)" in graph.read_text(encoding="utf-8")
    staged_search = json.loads(
        (tmp_path / "data" / "datasets" / "livecodebench_validate.jsonl").read_text()
    )
    assert staged_search["private_test_cases"] == staged_search["public_test_cases"]


def _native_config() -> dict:
    path = Path(__file__).parents[1] / "experiments" / "phase2_humaneval_qwen35_9b_single_service.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _adapter_result(candidate: dict, dataset: list[dict], **_: object) -> dict:
    count = len(dataset)
    return {
        "score": 1.0,
        "component_score": 1.0,
        "scores": [1.0] * count,
        "component_scores": [1.0] * count,
        "correct": count,
        "num_examples": count,
        "avg_calls": 1.0,
        "sum_calls": float(count),
        "avg_total_tokens": float(len(candidate.get("agents", [])) + 1),
        "sum_total_tokens": float(count * (len(candidate.get("agents", [])) + 1)),
        "avg_inference_cost_usd": 0.0,
        "sum_inference_cost_usd": 0.0,
        "avg_cross_center_tokens": 0.0,
        "avg_network_latency_ms": 0.0,
        "avg_emulated_latency_ms": 0.0,
        "avg_errors": 0.0,
        "valid_answer_rate": 1.0,
        "valid_execution_rate": 1.0,
        "error_example_rate": 0.0,
        "truncated_unextractable_rate": 0.0,
        "failure_examples": [],
        "outputs": [{"id": row["id"], "prediction": "print(1)", "passed": True} for row in dataset],
    }


def _native_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, new_candidates: int) -> dict:
    config = _native_config()
    config["reflection"]["empty_plan_retry_limit"] = 2
    models = native_search.load_models(config["models"])
    profile = native_search.load_network_profiles(config["network_profiles"])["lan_homogeneous"]
    rows = lambda split: [{"id": f"{split}-0", "dataset": "livecodebench", "input": "x", "answer": "hidden"}]
    if new_candidates:
        monkeypatch.setattr(native_search, "build_reflection_plan", lambda **_: {"mode": "llm", "mutations": [], "call_traces": []})
    return native_search.run_search(
        config=config, models=models, profile=profile, searchset=rows("search"),
        selectionset=rows("select"), testset=rows("test"), mode="wan_pareto",
        seed_candidate_budget=1, new_candidate_budget=new_candidates, search_examples=None,
        selection_shortlist_size=1, test_top_k=1, eval_concurrency=1, output_dir=tmp_path,
        seed=0, selection_strategy="quality_band_cost", quality_band=0.05,
        pareto_parent_prob=0.5, parent_score_band=0.05, parent_top_k=6,
        reflection_mode="llm", reflection_model="qwen35_9b", reflection_max_tokens=128,
        reflection_children=1, reflection_example_limit=1, candidate_evaluator=_adapter_result,
    )


def test_native_run_search_accepts_only_task_evaluator_injection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _native_run(tmp_path, monkeypatch, new_candidates=0)
    assert result["selection_policy"] == "protocol_q_e.delta=0.05"
    assert result["selected_test_rows"][0]["evaluation_cache_status"] == "external_task_adapter"
    assert result["selected_test_rows"][0]["test"]["outputs"][0]["id"] == "test-0"


def test_native_run_search_retries_llm_and_forbids_rule_mutation_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _native_config()
    assert config["reflection"]["allow_rule_fallback"] is False
    with pytest.raises(RuntimeError, match="bounded no-progress retry limit"):
        _native_run(tmp_path, monkeypatch, new_candidates=1)
    proposal_rows = [json.loads(line) for line in (tmp_path / "proposal_rows.jsonl").read_text().splitlines()]
    assert [row["status"] for row in proposal_rows] == ["empty_llm_plan", "empty_llm_plan"]
    assert all(row["reasons"] == ["no_new_applicable_typed_mutation"] for row in proposal_rows)


class _FakeTrace:
    calls: list[object] = []

    @staticmethod
    def summary(_profile: object) -> dict[str, float]:
        return {"calls": 1.0, "total_tokens": 1.0}


def test_rpas_search_feedback_never_executes_private_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluated_scopes: list[bool] = []

    def fake_evaluate(_task: LiveCodeBenchTask, _code: str, *, public: bool) -> dict[str, object]:
        evaluated_scopes.append(public)
        return {
            "passed": public,
            "status": "passed" if public else "failed",
            "metadata": {"scope": "public" if public else "private"},
        }

    monkeypatch.setattr(native_rpas_livecodebench, "evaluate_code", fake_evaluate)
    monkeypatch.setattr(
        native_rpas_livecodebench,
        "run_single_architecture",
        lambda **_: ("print(1)", _FakeTrace()),
    )
    monkeypatch.setattr(native_rpas_livecodebench, "_trace_calls", lambda *args: [])
    candidate = {"id": "candidate", "agents": [{"name": "solver"}]}

    search_row, _, _ = native_rpas_livecodebench._evaluate_one(
        candidate, task(1), {}, object(), "run", "search"
    )
    assert evaluated_scopes == [True]
    assert search_row["score"] == 1.0
    assert search_row["evaluator_scope"] == "public_search_feedback"
    assert search_row["private_evaluator_metadata"] == {}

    evaluated_scopes.clear()
    select_row, _, _ = native_rpas_livecodebench._evaluate_one(
        candidate, task(1), {}, object(), "run", "select"
    )
    assert evaluated_scopes == [True, False]
    assert select_row["score"] == 0.0
    assert select_row["evaluator_scope"] == "private_selection_or_test"
