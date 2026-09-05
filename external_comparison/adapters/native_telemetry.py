"""Observe native EC1 API responses without changing requests or optimizer logic."""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any

from external_comparison.adapters.native_common import call_record as legacy_call_record


SCHEMA = "native_ec1_response_v2"
RECORDER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _operator_frame() -> str | None:
    frame = inspect.currentframe()
    try:
        while frame is not None:
            module = str(frame.f_globals.get("__name__", ""))
            if ".operator" in module:
                owner = frame.f_locals.get("self")
                return type(owner).__qualname__ if owner is not None else frame.f_code.co_name
            frame = frame.f_back
    finally:
        del frame
    return None


def _timeout_fields(value: Any) -> dict[str, Any] | float | None:
    if isinstance(value, (int, float)):
        return float(value)
    fields = {name: getattr(value, name, None) for name in ("connect", "read", "write", "pool")}
    return fields if any(item is not None for item in fields.values()) else None


class NativeCallRecorder:
    def __init__(self, path: Path, *, method: str, seed: int) -> None:
        self.path = path
        self.method = method
        self.seed = seed
        self.context = contextvars.ContextVar("native_ec1_task", default=None)
        self.sequence = 0
        self.patches: list[tuple[type, str, Any, Any]] = []

    def _patch(self, cls: type, name: str, wrapped: Any) -> None:
        original = getattr(cls, name)
        self.patches.append((cls, name, original, wrapped))
        setattr(cls, name, wrapped)

    def restore(self) -> None:
        for cls, name, original, wrapped in reversed(self.patches):
            if getattr(cls, name) is wrapped:
                setattr(cls, name, original)
        self.patches.clear()

    def bind_benchmark(self, cls: type) -> None:
        original = cls.evaluate_problem

        @wraps(original)
        async def evaluate(benchmark, data, graph, *args, **kwargs):
            task = data.get("task_id", data.get("problem_id", data.get("id")))
            if not isinstance(task, str) or not task:
                raise ValueError("Native EC1 benchmark row lacks a task ID for telemetry")
            kind = type(graph)
            token = self.context.set({"example_id": task, "example_scope": "benchmark_item",
                                      "candidate_id": f"{kind.__module__}.{kind.__qualname__}"})
            try:
                return await original(benchmark, data, graph, *args, **kwargs)
            finally:
                self.context.reset(token)

        self._patch(cls, "evaluate_problem", evaluate)

    def patch_completions_class(self, cls: type) -> None:
        original = cls.create

        @wraps(original)
        async def create(resource, *args, **kwargs):
            sequence = self.sequence
            self.sequence += 1
            client = getattr(resource, "_client", None)
            context = self.context.get() or {"example_id": None, "example_scope": "native_optimizer",
                                             "candidate_id": "native_optimizer"}
            started = time.perf_counter()
            record = {
                "telemetry_schema": SCHEMA, "method": self.method, "dataset": "humaneval", "seed": self.seed,
                "run_id": f"humaneval-{self.method}-seed-{self.seed}", "call_sequence": sequence,
                "phase": os.environ.get("RPAS_EC1_PHASE", "unknown"), **context,
                "agent": _operator_frame(), "operator_provenance": "observed_python_call_frame",
                "model": kwargs.get("model"), "requested_max_tokens": kwargs.get("max_tokens"),
                "requested_temperature": kwargs.get("temperature"), "requested_stream": bool(kwargs.get("stream", False)),
                "timeout": _timeout_fields(kwargs.get("timeout", getattr(client, "timeout", None))),
                "sdk_max_retries": getattr(client, "max_retries", None),
                "retry_count": None, "transport_attempts_observed": False,
                "call_scope": "SDK_create_invocation_including_its_internal_retries",
                "started_at_epoch": time.time(), "finish_reason": None, "usage_observed": False,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": None,
            }
            try:
                response = await original(resource, *args, **kwargs)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    values = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
                    counts = [values.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
                    if all(type(value) is int and value >= 0 for value in counts) and counts[2] == counts[0] + counts[1]:
                        record.update(dict(zip(("prompt_tokens", "completion_tokens", "total_tokens"), counts)))
                        record["usage_observed"] = True
                if not record["usage_observed"]:
                    record["error"] = "telemetry_missing_or_invalid_response_usage"
                choices = getattr(response, "choices", None)
                if choices and len(choices) == 1:
                    record["finish_reason"] = getattr(choices[0], "finish_reason", None)
                record["response_id"] = getattr(response, "id", None)
                record["response_model"] = getattr(response, "model", None)
                return response
            except BaseException as error:
                record["error"] = type(error).__name__
                raise
            finally:
                record["latency_ms"] = (time.perf_counter() - started) * 1000
                record["finished_at_epoch"] = time.time()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        self._patch(cls, "create", create)

    def install_openai(self) -> None:
        from openai.resources.chat.completions import AsyncCompletions

        self.patch_completions_class(AsyncCompletions)

    def manifest(self) -> dict:
        return {"new_call_telemetry_schema": SCHEMA, "recorder_sha256": RECORDER_SOURCE_SHA256,
                "call_accounting_scope": "SDK create invocations; internal transport retry attempts not individually observed",
                "new_call_latency_scope": "awaited API invocation, not cost callback",
                "stream_telemetry_supported": False}


def call_record(run_id: str, method: str, dataset: str, split: str, candidate_id: str, index: int, usage: dict) -> dict:
    """Preserve v2 fields only for native adapters; leave legacy conversion unchanged."""
    if usage.get("telemetry_schema") != SCHEMA:
        return legacy_call_record(run_id, method, dataset, split, candidate_id, index, usage)
    base = legacy_call_record(run_id, method, dataset, split, candidate_id, index, {**usage, "retry_count": 0})
    for key in ("telemetry_schema", "seed", "phase", "example_id", "example_scope", "candidate_id", "agent",
                "call_sequence", "started_at_epoch", "finished_at_epoch", "requested_max_tokens", "requested_temperature",
                "requested_stream", "timeout", "sdk_max_retries", "retry_count", "transport_attempts_observed",
                "call_scope", "operator_provenance", "usage_observed", "response_id", "response_model"):
        if key in usage:
            base[key] = usage[key]
    return base
