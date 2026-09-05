import asyncio
import contextvars
import json

import pytest

from external_comparison.runners.ec2_v2 import (
    OfficialGDesignerRuntime, _TrainingRowsDataset, _llm_topology_mutation,
    _reflection_call_records, communication_candidate,
)
from external_comparison.runners.mmlu import MMLUExample


def bare_runtime():
    runtime = object.__new__(OfficialGDesignerRuntime)
    runtime.seed = 2
    runtime.output_dir = None
    runtime.example_id = contextvars.ContextVar("test_example", default="")
    runtime.candidate_id = contextvars.ContextVar("test_candidate", default="")
    runtime.training_iteration = contextvars.ContextVar("test_iteration", default=None)
    return runtime


def test_journal_persists_and_duplicate_execution_is_rejected(tmp_path):
    runtime = bare_runtime()
    runtime.configure_artifacts(tmp_path)
    runtime.journal("live_rows.jsonl", {"phase": "test", "row": {"example_id": "test:1"}})
    assert json.loads((tmp_path / "live_rows.jsonl").read_text())["row"]["example_id"] == "test:1"
    with pytest.raises(FileExistsError):
        bare_runtime().configure_artifacts(tmp_path)


def test_training_task_context_is_not_overwritten_by_next_example():
    runtime = bare_runtime()
    rows = [MMLUExample(str(i), "s", "q", ("a", "b", "c", "d"), "A") for i in range(8)]
    dataset = _TrainingRowsDataset(rows, runtime)
    async def capture():
        await asyncio.sleep(0)
        return runtime.example_id.get(), runtime.training_iteration.get()
    async def launch():
        tasks = []
        for row in rows:
            assert dataset.record_to_input(row)["task"].startswith("Question: q")
            tasks.append(asyncio.create_task(capture()))
        return await asyncio.gather(*tasks)
    assert asyncio.run(launch()) == [(str(i), i // 4) for i in range(8)]


def test_call_materialization_retains_item_and_iteration():
    runtime = bare_runtime()
    runtime.usage = [{"phase": "search", "example_id": "dev:42", "candidate_id": "gdesigner_training",
                      "seed": 2, "round": 0, "training_iteration": 3,
                      "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}]
    calls = runtime.calls(run_id="fixture", method="gdesigner")
    assert calls[0]["example_id"] == "dev:42"
    assert calls[0]["training_iteration"] == 3
    assert calls[0]["seed"] == 2
    assert calls[0]["total_tokens"] == 5


def test_reflection_latency_uses_observed_call_time():
    calls = _reflection_call_records({"call_traces": [{"prompt_tokens": 2, "completion_tokens": 3,
        "total_tokens": 5, "observed_latency_ms": 1275, "requested_max_tokens": 768}]},
        run_id="fixture", method="rpas_comm", candidate_id="parent", start_index=0)
    assert calls[0]["wall_latency_ms"] == 1275
    assert calls[0]["example_scope"] == "candidate_failure_summary"


def test_reflection_excludes_already_evaluated_topologies(monkeypatch):
    import experiments.phase2_wan_agent_search as search
    seen = [communication_candidate(name)["id"] for name in ("full_connected", "chain", "star")]
    parent = {"candidate": communication_candidate("full_connected"), "candidate_id": seen[0], "excluded_candidate_ids": seen}
    def reflector(**kwargs):
        assert kwargs["config"]["allowed_topologies"] == ["layered"]
        assert kwargs["reflection_max_tokens"] == 768
        return {"mode": "llm", "mutations": [{"type": "topology", "target": "candidate", "value": "layered"}]}
    monkeypatch.setattr(search, "build_reflection_plan", reflector)
    child, _ = _llm_topology_mutation(parent, config={"defaults": {"local_model": "model"}}, models={}, profile=None)
    assert child["topology"] == "layered"
    assert child["id"] not in seen
