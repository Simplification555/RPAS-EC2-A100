import asyncio
import contextvars
import json
from types import SimpleNamespace

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


def test_rpas_full_pipeline_keeps_search_select_and_test_separate(tmp_path, monkeypatch):
    import external_comparison.runners.ec2_v2 as ec2
    import experiments.phase2_wan_agent_search as search
    phases = []
    journals = []
    usage = []
    class Runtime:
        def make_graph(self, **kwargs):
            assert kwargs["optimized_spatial"] is False
            return kwargs["topology"]
        async def evaluate(self, graph, examples, *, split, candidate_id, concurrency=1):
            assert concurrency == 4
            phases.append((split, [row.example_id for row in examples]))
            rows = [{"example_id": row.example_id, "prediction": "A", "answer": "A", "correct": True,
                     "inter_agent_tokens": 1} for row in examples]
            usage.extend({"split": split, "total_tokens": 5, "prompt_tokens": 3, "completion_tokens": 2} for _ in rows)
            return rows, {"inter_agent_tokens": len(rows)}
        def calls(self, **kwargs):
            return list(usage)
        def journal(self, name, payload):
            journals.append((name, payload, len(phases)))
    monkeypatch.setenv("RPAS_EC2_V2_NEW_CANDIDATES", "3")
    monkeypatch.setattr(ec2, "_reflection_context", lambda _: ({"defaults": {"local_model": "model"}}, {}, None))
    def reflector(**kwargs):
        return {"mode": "llm", "mutations": [{"type": "topology", "value": kwargs["config"]["allowed_topologies"][0]}],
                "call_traces": [{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}]}
    monkeypatch.setattr(search, "build_reflection_plan", reflector)
    rows = {phase: [MMLUExample(f"{phase}:{i}", "s", "q", ("a", "b", "c", "d"), "A") for i in range(count)]
            for phase, count in (("search", 57), ("select", 57), ("test", 570))}
    rows.update(output_dir=str(tmp_path), manifest={"split_manifest_sha256": "fixture",
                "search": {"count": 57}, "select": {"count": 57}, "test": {"count": 570}})
    asyncio.run(ec2._run_rpas_comm(Runtime(), rows, 0, tmp_path))
    assert [phase for phase, _ in phases] == ["search"] * 4 + ["select"] * 4 + ["test"]
    assert sum(len(ids) for phase, ids in phases if phase == "search") == 40
    assert sum(len(ids) for phase, ids in phases if phase == "select") == 228
    assert len(phases[-1][1]) == 570
    assert all(all(item.startswith(phase + ":") for item in ids) for phase, ids in phases)
    assert next(count for name, _, count in journals if name == "selection_frozen.jsonl") == 8
    result = json.loads((tmp_path / "rpas_comm" / "seed_0" / "result.json").read_text())
    assert result["rpas_reflection"]["new_candidates"] == 3
    assert result["rpas_reflection"]["rule_fallbacks"] == 0


def test_fixed_batch_bounds_concurrency_and_preserves_order():
    runtime = bare_runtime()
    live = 0
    peak = 0
    async def serial(graph, rows, **kwargs):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep((8 - rows[0]) / 1000)
        live -= 1
        return [{"id": rows[0]}], {key: 1 for key in ("active_edges", "messages", "inter_agent_tokens", "judge_input_tokens")}
    runtime._evaluate_serial = serial
    graph = SimpleNamespace(optimized_spatial=False, optimized_temporal=False)
    outputs, totals = asyncio.run(runtime.evaluate(graph, list(range(9)), split="test", candidate_id="fixed", concurrency=4))
    assert peak == 4 and live == 0
    assert [row["id"] for row in outputs] == list(range(9))
    assert totals["messages"] == 9
    graph.optimized_spatial = True
    with pytest.raises(ValueError, match="fixed graphs"):
        asyncio.run(runtime.evaluate(graph, [0], split="test", candidate_id="learned", concurrency=4))


def test_failed_batch_cancels_remaining_items():
    runtime = bare_runtime()
    cancelled = []
    async def serial(graph, rows, **kwargs):
        if rows[0] == 0:
            await asyncio.sleep(0)
            raise RuntimeError("synthetic failure")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(rows[0])
            raise
    runtime._evaluate_serial = serial
    graph = SimpleNamespace(optimized_spatial=False, optimized_temporal=False)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        asyncio.run(runtime.evaluate(graph, list(range(4)), split="test", candidate_id="fixed", concurrency=4))
    assert sorted(cancelled) == [1, 2, 3]


def test_shared_split_publish_is_atomic_and_never_overwrites(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from external_comparison.runners.ec2_v2 import publish_frozen_split

    path = tmp_path / "split_manifest.json"
    manifest = {"test": {"ids": ["test:1", "test:2"]}}
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: publish_frozen_split(path, manifest), range(20)))
    assert json.loads(path.read_text()) == manifest
    before = path.read_bytes()
    with pytest.raises(ValueError, match="refusing overwrite"):
        publish_frozen_split(path, {"test": {"ids": ["different"]}})
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".split-*.tmp")) == []


def test_runner_hash_is_frozen_at_import(tmp_path, monkeypatch):
    import external_comparison.runners.ec2_v2 as ec2

    imported = ec2.RUNNER_SOURCE_SHA256
    monkeypatch.setattr(ec2, "sha256_file", lambda _: "changed-on-disk")
    manifest = ec2._base_manifest("chain", 0, {"split_manifest_sha256": "fixture",
        "search": {"count": 57}, "select": {"count": 57}, "test": {"count": 570}})
    bare_runtime().configure_artifacts(tmp_path)
    identity = json.loads((tmp_path / "execution_identity.json").read_text())
    assert identity["runner_sha256"] == manifest["runner_sha256"] == imported
