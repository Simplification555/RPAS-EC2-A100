import asyncio
import json
from types import FunctionType, SimpleNamespace

import pytest

from external_comparison.adapters.native_common import call_record as legacy_call_record
from external_comparison.adapters.native_telemetry import NativeCallRecorder, SCHEMA, call_record


def response(tokens=5):
    return SimpleNamespace(usage={"prompt_tokens": 2, "completion_tokens": tokens - 2, "total_tokens": tokens},
                           choices=[SimpleNamespace(finish_reason="length")], id="observed-response", model="fixture-model")


def read_calls(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_records_response_usage_elapsed_time_and_preserves_requests(tmp_path, monkeypatch):
    requests = []
    observed = response()
    class Resource:
        _client = SimpleNamespace(timeout=600, max_retries=2)
        async def create(self, *args, **kwargs):
            requests.append((args, kwargs))
            await asyncio.sleep(0.025)
            return observed
    path = tmp_path / "calls.jsonl"
    recorder = NativeCallRecorder(path, method="maas", seed=2)
    original = Resource.create
    recorder.patch_completions_class(Resource)
    monkeypatch.setenv("RPAS_EC1_PHASE", "search")
    payload = {"model": "fixture-model", "max_tokens": 1024, "temperature": 0,
               "messages": [{"role": "user", "content": "synthetic"}]}
    try:
        assert asyncio.run(Resource().create(**payload)) is observed
    finally:
        recorder.restore()
    assert Resource.create is original
    assert requests == [((), payload)]
    record = read_calls(path)[0]
    assert record["total_tokens"] == 5 and record["usage_observed"] is True
    assert record["latency_ms"] >= 20 and record["finish_reason"] == "length"
    assert record["seed"] == 2 and record["phase"] == "search"
    assert record["sdk_max_retries"] == 2 and record["retry_count"] is None
    assert record["transport_attempts_observed"] is False
    assert record["example_scope"] == "native_optimizer"


def test_concurrent_benchmark_ids_do_not_leak_between_coroutines(tmp_path, monkeypatch):
    class Resource:
        async def create(self, *, item):
            await asyncio.sleep(0.01 if item == "a" else 0)
            return response()
    class Benchmark:
        async def evaluate_problem(self, data, graph):
            return await Resource().create(item=data["task_id"])
    path = tmp_path / "calls.jsonl"
    recorder = NativeCallRecorder(path, method="maas", seed=0)
    recorder.patch_completions_class(Resource)
    recorder.bind_benchmark(Benchmark)
    monkeypatch.setenv("RPAS_EC1_PHASE", "test")
    async def run():
        result = await asyncio.gather(*(Benchmark().evaluate_problem({"task_id": item}, object()) for item in ("a", "b")))
        assert recorder.context.get() is None
        return result
    try:
        assert len(asyncio.run(run())) == 2
    finally:
        recorder.restore()
    records = read_calls(path)
    assert [record["example_id"] for record in records] == ["b", "a"]
    assert {record["call_sequence"] for record in records} == {0, 1}


@pytest.mark.parametrize("error", [RuntimeError("sensitive diagnostic"), asyncio.CancelledError()])
def test_errors_and_cancellation_are_recorded_then_propagated(tmp_path, error):
    class Resource:
        async def create(self):
            raise error
    path = tmp_path / "calls.jsonl"
    recorder = NativeCallRecorder(path, method="aflow", seed=1)
    recorder.patch_completions_class(Resource)
    try:
        with pytest.raises(type(error)):
            asyncio.run(Resource().create())
    finally:
        recorder.restore()
    record = read_calls(path)[0]
    assert record["error"] == type(error).__name__
    assert record["usage_observed"] is False
    assert "sensitive diagnostic" not in path.read_text()


def test_legacy_conversion_is_unchanged_and_new_metadata_is_not_dropped():
    usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    args = ("run", "aflow", "humaneval", "test", "candidate", 0)
    assert call_record(*args, usage) == legacy_call_record(*args, usage)
    record = call_record(*args, {**usage, "telemetry_schema": SCHEMA, "example_id": "HumanEval/0",
                               "seed": 2, "finish_reason": "length", "retry_count": None,
                               "latency_ms": 123.5, "usage_observed": True})
    assert record["example_id"] == "HumanEval/0" and record["seed"] == 2
    assert record["finish_reason"] == "length" and record["wall_latency_ms"] == 123.5
    assert record["retry_count"] is None


def test_missing_usage_is_visible_without_mutating_returned_response(tmp_path):
    observed = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])
    class Resource:
        async def create(self):
            return observed
    recorder = NativeCallRecorder(tmp_path / "calls.jsonl", method="aflow", seed=0)
    recorder.patch_completions_class(Resource)
    try:
        assert asyncio.run(Resource().create()) is observed
    finally:
        recorder.restore()
    assert read_calls(recorder.path)[0]["error"] == "telemetry_missing_or_invalid_response_usage"


def test_operator_is_observed_from_native_python_frame(tmp_path):
    class Resource:
        async def create(self):
            return response()
    async def operator_call(self):
        return await resource.create()
    resource = Resource()
    namespace = {"__name__": "synthetic.workflow.operator", "resource": resource}
    # A distinct globals mapping models the real upstream operator module.
    namespace_call = FunctionType(operator_call.__code__, namespace, closure=operator_call.__closure__)
    Operator = type("CustomCodeGenerate", (), {"__call__": namespace_call})
    recorder = NativeCallRecorder(tmp_path / "calls.jsonl", method="aflow", seed=0)
    recorder.patch_completions_class(Resource)
    try:
        asyncio.run(Operator()())
    finally:
        recorder.restore()
    assert read_calls(recorder.path)[0]["agent"] == "CustomCodeGenerate"
