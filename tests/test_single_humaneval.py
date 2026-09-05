import json
from types import SimpleNamespace

import pytest

from external_comparison.runners.humaneval import HumanEvalTask
from external_comparison.runners.single_humaneval import evaluate_tasks


def test_single_is_one_call_with_no_test_feedback(tmp_path):
    requests = []
    task = HumanEvalTask("fixture/1", "def solve(xs):\n    # sum the list", "def check(f):\n    assert f([2,3]) == 5", "solve")
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="from typing import List\ndef solve(xs: List[int]):\n    return sum(xs)"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    rows, calls = evaluate_tasks([task], client, 2, tmp_path)
    assert len(requests) == len(calls) == len(rows) == 1
    assert requests[0]["messages"][1]["content"] == task.prompt
    assert task.test not in json.dumps(requests)
    assert requests[0]["temperature"] == 0.0
    assert requests[0]["max_tokens"] == 1024
    assert rows[0]["passed"] is True
    assert calls[0]["example_id"] == task.task_id
    assert calls[0]["total_tokens"] == 50
    assert calls[0]["seed"] == 2
    assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        evaluate_tasks([task], client, 2, tmp_path)
    assert len(requests) == 1


def test_missing_usage_fails_closed_and_keeps_diagnostic(tmp_path):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(usage=None))))
    task = HumanEvalTask("fixture/1", "def solve(): pass", "def check(f): pass", "solve")
    with pytest.raises(ValueError, match="token usage"):
        evaluate_tasks([task], client, 0, tmp_path)
    diagnostic = json.loads((tmp_path / "progress.jsonl").read_text())
    assert diagnostic["formal_result"] is False
