import json
import sys
from pathlib import Path

import pytest

from external_comparison.adapters.native_runtime import stage_checkout, stage_humaneval_data
from external_comparison.runners.humaneval import load_humaneval_tasks
from external_comparison.runners.native_rpas_ec1 import _frozen_aflow_splits
from external_comparison.runners.public_test_executor import PublicTestExecutor
from external_comparison.runners import native_humaneval


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_aflow_seed_staging_uses_frozen_aflow_33_131_fixtures(tmp_path: Path):
    source = tmp_path / "AFlow"
    (source / "workspace" / "HumanEval" / "workflows" / "round_1").mkdir(parents=True)
    (source / "workspace" / "HumanEval" / "workflows" / "round_1" / "graph.py").write_text("class Workflow: pass\n", encoding="utf-8")
    rows = [{"task_id": f"HumanEval/{index}", "prompt": f"p{index}", "test": "def check(x): pass", "entry_point": "f"} for index in range(164)]
    dataset = tmp_path / "humaneval.jsonl"
    public = tmp_path / "public.jsonl"
    validate = tmp_path / "humaneval_validate.jsonl"
    test = tmp_path / "humaneval_test.jsonl"
    _write_jsonl(dataset, rows)
    _write_jsonl(public, [{"entry_point": "f", "test": "assert True"}])
    _write_jsonl(validate, rows[:33])
    _write_jsonl(test, rows[33:])

    workspace = stage_checkout(source, tmp_path / "outputs", "aflow", 0)
    manifest = stage_humaneval_data(
        workspace, "aflow", dataset, public, 2026,
        search_fixture=validate, test_fixture=test,
    )

    assert len(manifest["search_tasks"]) == 33
    assert len(manifest["test_tasks"]) == 131
    assert set(manifest["search_tasks"]).isdisjoint(manifest["test_tasks"])
    assert sum(1 for _ in Path(manifest["search_path"]).open(encoding="utf-8")) == 33
    assert sum(1 for _ in Path(manifest["test_path"]).open(encoding="utf-8")) == 131
    assert Path(manifest["search_path"]).read_bytes() == validate.read_bytes()
    assert Path(manifest["test_path"]).read_bytes() == test.read_bytes()
    assert Path(manifest["public_test_path"]).read_bytes() == public.read_bytes()


def test_aflow_fixture_content_must_match_official_humaneval(tmp_path: Path):
    source = tmp_path / "AFlow"
    (source / "workspace" / "HumanEval" / "workflows" / "round_1").mkdir(parents=True)
    (source / "workspace" / "HumanEval" / "workflows" / "round_1" / "graph.py").write_text("class Workflow: pass\n", encoding="utf-8")
    rows = [{"task_id": f"HumanEval/{index}", "prompt": f"p{index}", "test": "def check(x): pass", "entry_point": "f"} for index in range(164)]
    dataset = tmp_path / "humaneval.jsonl"
    public = tmp_path / "public.jsonl"
    validate = tmp_path / "humaneval_validate.jsonl"
    test = tmp_path / "humaneval_test.jsonl"
    _write_jsonl(dataset, rows)
    _write_jsonl(public, [{"entry_point": "f", "test": "assert True"}])
    corrupt_validate = [dict(row) for row in rows[:33]]
    corrupt_validate[0]["prompt"] = "not the official prompt"
    _write_jsonl(validate, corrupt_validate)
    _write_jsonl(test, rows[33:])

    workspace = stage_checkout(source, tmp_path / "outputs", "aflow", 1)
    with pytest.raises(ValueError, match="differs from official HumanEval"):
        stage_humaneval_data(
            workspace, "aflow", dataset, public, 2026,
            search_fixture=validate, test_fixture=test,
        )


def test_rpas_uses_the_same_frozen_aflow_fixtures(tmp_path: Path):
    rows = [
        {"task_id": f"HumanEval/{index}", "prompt": f"p{index}", "test": "def check(x): pass", "entry_point": "f"}
        for index in range(164)
    ]
    source = tmp_path / "humaneval.jsonl"
    validate = tmp_path / "humaneval_validate.jsonl"
    test = tmp_path / "humaneval_test.jsonl"
    _write_jsonl(source, rows)
    _write_jsonl(validate, rows[:33])
    _write_jsonl(test, rows[33:])

    splits = _frozen_aflow_splits(
        load_humaneval_tasks(source), validate_path=validate, test_path=test
    )

    assert [task.task_id for task in splits["search"]] == [row["task_id"] for row in rows[:33]]
    assert [task.task_id for task in splits["test"]] == [row["task_id"] for row in rows[33:]]


def test_public_test_executor_exposes_partial_fixture_coverage(tmp_path: Path):
    fixture = tmp_path / "public.jsonl"
    _write_jsonl(fixture, [{"problem_id": "HumanEval/0", "entry_point": "f", "test": ["assert candidate() == 1"]}])
    executor = PublicTestExecutor(fixture)
    assert executor.task_count == 1
    assert executor.has_task("HumanEval/0")
    assert not executor.has_task("HumanEval/1")


@pytest.mark.parametrize(
    ("selected_gpu", "visible_gpu"),
    [("", "4"), ("4", ""), ("4", "5"), ("6", "6")],
)
def test_ec1_dispatch_rejects_missing_or_mismatched_gpu_binding(
    monkeypatch: pytest.MonkeyPatch, selected_gpu: str, visible_gpu: str
):
    monkeypatch.setenv("RPAS_EC1_GPU", selected_gpu)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_gpu)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_humaneval",
            "--repo-root", ".",
            "--dataset-path", "ignored.jsonl",
            "--method", "aflow",
            "--seed", "0",
            "--output-dir", "ignored",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        native_humaneval.main()
