"""Exercise real upstream provider/benchmark classes with synthetic API responses."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

from external_comparison.adapters.native_runtime import stage_checkout, write_maas_config
from external_comparison.adapters.native_telemetry import NativeCallRecorder


async def check_live_service(base_url: str) -> dict:
    from openai import AsyncOpenAI

    with tempfile.TemporaryDirectory(prefix="native_telemetry_live_") as temporary:
        recorder = NativeCallRecorder(Path(temporary) / "calls.jsonl", method="aflow", seed=0)
        recorder.install_openai()
        token = recorder.context.set({"example_id": "synthetic/smoke", "example_scope": "synthetic_preflight",
                                      "candidate_id": "preflight_only"})
        try:
            async with AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=600, max_retries=0) as client:
                result = await client.chat.completions.create(model="Qwen/Qwen3.5-9B", max_tokens=4, temperature=0,
                    messages=[{"role": "user", "content": "Reply with only the digit 1."}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            record = json.loads(recorder.path.read_text())
            assert result.choices[0].message.content.strip() == "1"
            assert record["total_tokens"] == result.usage.total_tokens
            assert record["finish_reason"] == "stop" and record["usage_observed"] and not record["error"]
            assert record["example_id"] == "synthetic/smoke"
            return {"status": "PASS_REAL_RESPONSE_TELEMETRY", "formal_result": False, "synthetic_only": True,
                    "record": record, "recorder": recorder.manifest()}
        finally:
            recorder.context.reset(token)
            recorder.restore()


def check(method: str, source: Path) -> dict:
    from openai.resources.chat.completions import AsyncCompletions
    from openai.types.chat import ChatCompletion

    prior = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"{method}_telemetry_preflight_") as temporary:
        root = Path(temporary)
        workspace = stage_checkout(source, root, method, 0, require_clean_git=True)
        os.chdir(workspace)
        sys.path.insert(0, str(workspace))
        os.environ["RPAS_EC1_PHASE"] = "test"
        if method == "aflow":
            from scripts.async_llm import AsyncLLM, LLMConfig
            from benchmarks.humaneval import HumanEvalBenchmark

            llm = AsyncLLM(LLMConfig({"model": "Qwen/Qwen3.5-9B", "key": "EMPTY",
                                      "base_url": "http://127.0.0.1:1/v1", "temperature": 0, "top_p": 1}))
            async def invoke(prompt):
                return await llm(prompt)
        else:
            os.environ["METAGPT_PROJECT_ROOT"] = str(workspace)
            from external_comparison.runners.native_ec1_driver import (
                _install_maas_import_compat, _install_maas_optional_encoding_compat,
                _install_maas_provider_compat, _install_maas_actions_compat,
            )
            _install_maas_import_compat(workspace)
            _install_maas_optional_encoding_compat()
            _install_maas_provider_compat(workspace)
            _install_maas_actions_compat(workspace)
            write_maas_config(workspace, "Qwen/Qwen3.5-9B", "http://127.0.0.1:1/v1", "EMPTY", 0)
            from maas.configs.models_config import ModelsConfig
            from maas.provider.openai_api import OpenAILLM
            from maas.ext.maas.benchmark.humaneval import HumanEvalBenchmark

            llm = OpenAILLM(ModelsConfig.default().get("Qwen/Qwen3.5-9B"))
            async def invoke(prompt):
                return await llm.acompletion_text([{"role": "user", "content": prompt}], stream=False)
        requests = []
        async def simulated(resource, *args, **kwargs):
            requests.append((args, kwargs.copy()))
            await asyncio.sleep(0.02)
            return ChatCompletion(id="synthetic", created=0, model="Qwen/Qwen3.5-9B", object="chat.completion",
                                  choices=[{"index": 0, "message": {"role": "assistant", "content": "def f():\n    return 1"},
                                            "finish_reason": "stop"}],
                                  usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11})
        original = AsyncCompletions.create
        AsyncCompletions.create = simulated
        recorder = NativeCallRecorder(root / "observed.jsonl", method=method, seed=2)
        benchmark = object.__new__(HumanEvalBenchmark)
        benchmark.log_path = str(root)
        benchmark.check_solution = lambda *args: (benchmark.PASS, "synthetic evaluator only")
        benchmark.log_mismatch = lambda *args: None
        benchmark.device = "cpu"
        class Graph:
            async def __call__(self, prompt, entry_point, *args):
                answer = await invoke(prompt)
                if method == "maas":
                    import torch
                    return answer, 0.0, torch.tensor(0.0)
                return answer, 0.0
        async def exercise():
            plain = await invoke("synthetic prompt")
            recorder.install_openai()
            observed = await invoke("synthetic prompt")
            assert plain == observed and requests[0] == requests[1]
            recorder.bind_benchmark(HumanEvalBenchmark)
            data = {"prompt": "synthetic benchmark prompt", "entry_point": "f", "canonical_solution": "return 1", "test": ""}
            outputs = await asyncio.gather(*(benchmark.evaluate_problem({**data, "task_id": f"synthetic/{i}"}, Graph()) for i in range(2)))
            assert all(row[3] == 1.0 for row in outputs)
            await llm.aclient.close()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                asyncio.run(exercise())
            rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
            assert len(rows) == 3
            assert all(row["total_tokens"] == 11 and row["finish_reason"] == "stop" and not row["error"] for row in rows)
            assert all(row["latency_ms"] >= 15 for row in rows)
            assert {row["example_id"] for row in rows[1:]} == {"synthetic/0", "synthetic/1"}
            assert recorder.context.get() is None
            return {"method": method, "status": "PASS_NATIVE_PROVIDER_AND_BENCHMARK_TELEMETRY",
                    "formal_result": False, "simulated_api": True, "benchmark_data": "synthetic_only",
                    "actual_upstream_classes": True, "requests_preserved": True, "recorded_calls": len(rows),
                    "complete_usage_finish_latency_and_task_ids": True}
        finally:
            recorder.restore()
            AsyncCompletions.create = original
            os.chdir(prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("aflow", "maas"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.method, args.source.resolve())))
