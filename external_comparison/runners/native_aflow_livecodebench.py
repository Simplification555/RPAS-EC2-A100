"""Pinned upstream AFlow search with a bounded LiveCodeBench task adapter."""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from external_comparison.adapters.native_common import sha256_file, write_native_result
from external_comparison.adapters.native_runtime import seed_everything, source_manifest, stage_checkout, write_aflow_config
from external_comparison.adapters.native_telemetry import NativeCallRecorder, call_record
from external_comparison.runners.livecodebench_ec1 import (
    DATASET_REVISION,
    RELEASE_VERSION,
    LiveCodeBenchTask,
    generation_instruction,
    load_tasks,
    evaluate_code,
    extract_code,
    render_model_prompt,
    validate_frozen_bundle,
)
from external_comparison.runners.native_ec1_driver import _install_aflow_prompt_fallback
from external_comparison.runners.native_rpas_ec1 import _require_selected_gpu
from experiments.phase2_wan_agent_search import select_operating_points


def _stage_lcb(workspace: Path, bundle: Path) -> dict[str, Any]:
    data_root = workspace / "data" / "datasets"
    data_root.mkdir(parents=True, exist_ok=True)
    search_target = data_root / "livecodebench_validate.jsonl"
    with (bundle / "search.jsonl").open(encoding="utf-8") as source, search_target.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            row = json.loads(line)
            # AFlow may optimize against D_search, but private LCB cases must
            # never become search feedback. The model still never sees tests.
            row["private_test_cases"] = row["public_test_cases"]
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    shutil.copyfile(bundle / "test.jsonl", data_root / "livecodebench_test.jsonl")
    workflow_source = workspace / "workspace" / "HumanEval"
    workflow_target = workspace / "workspace" / "LiveCodeBench"
    if not workflow_source.is_dir():
        raise FileNotFoundError("pinned AFlow checkout lacks its official code-task seed workflow")
    shutil.copytree(workflow_source, workflow_target)
    changed = []
    for path in workflow_target.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        adapted = content.replace("workspace.HumanEval", "workspace.LiveCodeBench")
        if adapted != content:
            path.write_text(adapted, encoding="utf-8")
            changed.append(str(path.relative_to(workspace)))
    round_one = workflow_target / "workflows" / "round_1" / "graph.py"
    round_one.write_text(
        "from workspace.LiveCodeBench.workflows.template import operator\n"
        "from scripts.async_llm import create_llm_instance\n\n"
        "class Workflow:\n"
        "    def __init__(self, name, llm_config, dataset):\n"
        "        self.name = name\n"
        "        self.dataset = dataset\n"
        "        self.llm = create_llm_instance(llm_config)\n"
        "        self.custom = operator.Custom(self.llm)\n\n"
        "    async def __call__(self, problem, entry_point, question_id):\n"
        "        if entry_point == 'wrapped_function':\n"
        "            instruction = (\"Return only a complete executable Python 3 program. Read standard input and \"\n"
        "                           \"write standard output. Do not return markdown or explanations.\\n\")\n"
        "        else:\n"
        "            instruction = (f\"Return only complete Python code defining the requested `{entry_point}` \"\n"
        "                           \"class/function contract. Do not read stdin, print an answer, use markdown, \"\n"
        "                           \"or add explanations.\\n\")\n"
        "        result = await self.custom(input=problem, instruction=instruction)\n"
        "        return result['response'], self.llm.get_usage_summary()['total_cost']\n",
        encoding="utf-8",
    )
    compile(round_one.read_text(encoding="utf-8"), str(round_one), "exec")
    return {
        "adapter_scope": (
            "dataset files plus import-path substitution in the official code-task seed/template; "
            "D_search evaluator uses public cases only"
        ),
        "adapted_files": changed,
        "search_sha256": sha256_file(bundle / "search.jsonl"),
        "staged_public_search_sha256": sha256_file(search_target),
        "select_sha256": sha256_file(bundle / "select.jsonl"),
        "test_sha256": sha256_file(bundle / "test.jsonl"),
    }


def _install_lcb_operator_adapter(public_tasks) -> str:
    """Route AFlow's code Test operator to the shared LCB public evaluator."""
    from workspace.LiveCodeBench.workflows.template import operator

    by_id = {task.question_id: task for task in public_tasks}
    by_question = {
        question: task
        for task in public_tasks
        for question in (task.question, render_model_prompt(task))
    }
    original_test_call = operator.Test.__call__

    def exec_code(self, solution, question_id):
        task = by_id.get(str(question_id))
        if task is None:
            return {"exec_fail_case": f"unknown frozen LiveCodeBench question_id: {question_id}"}
        result = evaluate_code(task, extract_code(solution), public=True)
        if result["passed"]:
            return "no error"
        return {"exec_fail_case": result["status"]}

    operator.Test.exec_code = exec_code

    async def test_call(self, problem, solution, entry_point, test_loop=3):
        task = by_question.get(str(problem))
        question_id = task.question_id if task is not None else entry_point
        return await original_test_call(self, problem, solution, question_id, test_loop)

    async def code_generate(self, problem, entry_point, instruction):
        task = by_question.get(str(problem))
        lcb_instruction = generation_instruction(task) + "\n" if task is not None else instruction
        return await self._fill_node(operator.GenerateOp, lcb_instruction + instruction + problem, mode="single_fill")

    operator.Test.__call__ = test_call
    operator.CustomCodeGenerate.__call__ = code_generate
    return (
        "Test routed by frozen question ID to shared LCB public evaluator; "
        "CustomCodeGenerate adapted from function completion to stdin/stdout program generation"
    )


def _install_benchmark_runtime_adapter(benchmark_cls: type) -> str:
    """Preserve AFlow logic while fixing task rendering and its 120s local timeout."""
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

    original_load_data = benchmark_cls.load_data
    benchmark_module = sys.modules[benchmark_cls.__module__]
    original_run_test = benchmark_module.run_test
    runner_module = importlib.import_module("scripts.utils.lcb_runner")

    def typed_run_test(sample, test=None, debug=False, timeout=6):
        in_outs = json.loads(sample["input_output"])
        if test is not None and in_outs.get("fn_name") is None:
            test = runner_module.make_function(runner_module.clean_if_name(test))
        return original_run_test(sample, test=test, debug=debug, timeout=timeout)

    async def typed_load_data(self, specific_indices=None):
        rows = await original_load_data(self, specific_indices)
        for row in rows:
            original = row.get("metadata", {}).get("original_data", {})
            task = LiveCodeBenchTask.from_row(original)
            row["question"] = render_model_prompt(task)
        return rows

    generation_timeout = int(os.environ.get("RPAS_EC1_LCB_GENERATION_TIMEOUT_SECONDS", "300"))
    if generation_timeout < 120:
        raise ValueError("AFlow LiveCodeBench generation timeout must be at least the upstream 120 seconds")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def generate_output(self, agent, prompt, entry_point, question_id=""):
        return await asyncio.wait_for(
            agent(prompt, entry_point, question_id), timeout=generation_timeout
        )

    benchmark_cls.load_data = typed_load_data
    benchmark_cls._generate_output = generate_output
    benchmark_module.run_test = typed_run_test
    return (
        "typed functional/stdin prompt rendering and evaluator ABI; "
        f"allocation-local generation timeout={generation_timeout}s"
    )


def _read_calls(path: Path, run_id: str) -> list[dict[str, Any]]:
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    calls = []
    for index, row in enumerate(raw):
        phase = str(row.get("phase", "unknown"))
        split = "test" if phase == "test_quality" else phase
        calls.append(call_record(run_id, "aflow", "livecodebench", split, str(row.get("candidate_id", phase)), index, row))
    return calls


def _csv_rows(log_dir: Path, tasks) -> list[dict[str, Any]]:
    csvs = sorted(log_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not csvs:
        raise RuntimeError(f"AFlow LCB evaluator wrote no CSV under {log_dir}")
    by_question = {
        question: task
        for task in tasks
        for question in (task.question, render_model_prompt(task))
    }
    rows = []
    with csvs[-1].open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            task = by_question.get(str(raw.get("question", "")))
            if task is None:
                raise ValueError("AFlow LCB CSV contains a question outside the frozen split")
            detail = json.loads(raw.get("evaluation_details", "{}") or "{}")
            rows.append(
                {
                    "task_id": task.question_id,
                    "passed": float(raw.get("score", 0.0) or 0.0) == 1.0,
                    "status": "passed" if float(raw.get("score", 0.0) or 0.0) == 1.0 else "failed",
                    "difficulty": task.difficulty,
                    "platform": task.platform,
                    "prediction": str(raw.get("prediction", "")),
                    "native_evaluator": "pinned_aflow_livecodebench",
                    "private_evaluator_metadata": detail.get("metadata", {}),
                }
            )
    if len(rows) != len(tasks):
        raise RuntimeError(f"AFlow LCB evaluator produced {len(rows)}/{len(tasks)} rows")
    return rows


async def _evaluate_round(optimizer: Any, round_number: int, data_path: Path, log_dir: Path, phase: str):
    from benchmarks.livecodebench import LiveCodeBench

    graph_class = optimizer.graph_utils.load_graph(round_number, "workspace/LiveCodeBench/workflows")
    graph = graph_class(name="LiveCodeBench", llm_config=optimizer.execute_llm_config, dataset="LiveCodeBench")
    benchmark = LiveCodeBench(name="LiveCodeBench", file_path=str(data_path), log_path=str(log_dir))
    os.environ["RPAS_EC1_PHASE"] = phase
    log_dir.mkdir(parents=True, exist_ok=True)
    return await benchmark.run_evaluation(graph, None)


def run(args: Any) -> None:
    gpu = _require_selected_gpu()
    source = Path(os.environ.get("RPAS_AFLOW_ROOT", "/home/jianbaizhao/external_baselines/AFlow_formal_clean")).resolve()
    bundle = Path(args.data_dir).resolve()
    frozen = validate_frozen_bundle(bundle)
    output = Path(args.output_dir) / "aflow" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    workspace = stage_checkout(source, output, "aflow_lcb", args.seed, require_clean_git=True)
    adapter = _stage_lcb(workspace, bundle)
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE", "")
    if not endpoint:
        raise RuntimeError("RPAS_EXTERNAL_API_BASE is required")
    model = os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B")
    write_aflow_config(workspace, model, endpoint, os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"))
    telemetry = output / "native_calls.jsonl"
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    seed_everything(args.seed)

    from benchmarks.benchmark import BaseBenchmark
    from benchmarks.livecodebench import LiveCodeBench
    from scripts.async_llm import AsyncLLM, LLMConfig
    from scripts.optimizer import Optimizer

    original_evaluate_all = BaseBenchmark.evaluate_all_problems

    async def bounded_evaluate_all(self, data, agent, max_concurrent_tasks=50):
        cap = max(1, int(os.environ.get("RPAS_EC1_LCB_CONCURRENCY", "4")))
        return await original_evaluate_all(self, data, agent, min(cap, int(max_concurrent_tasks)))

    BaseBenchmark.evaluate_all_problems = bounded_evaluate_all
    benchmark_runtime_patch = _install_benchmark_runtime_adapter(LiveCodeBench)
    recorder = NativeCallRecorder(telemetry, method="aflow", seed=args.seed, dataset="livecodebench")
    recorder.bind_benchmark(LiveCodeBench)
    recorder.install_openai()
    original_call = AsyncLLM.__call__

    async def instrumented_call(self, prompt):
        original_create = self.aclient.chat.completions.create

        async def capped_create(*call_args, **call_kwargs):
            call_kwargs.setdefault("max_tokens", 2048)
            call_kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
            return await original_create(*call_args, **call_kwargs)

        self.aclient.chat.completions.create = capped_create
        try:
            return await original_call(self, prompt)
        finally:
            self.aclient.chat.completions.create = original_create

    AsyncLLM.__call__ = instrumented_call
    config = LLMConfig({"model": model, "key": "EMPTY", "base_url": endpoint, "temperature": 0.0, "top_p": 1.0})
    optimizer = Optimizer(
        dataset="LiveCodeBench", question_type="code", opt_llm_config=config, exec_llm_config=config,
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"], optimized_path="workspace",
        sample=int(os.environ.get("RPAS_AFLOW_SAMPLE", "4")), initial_round=1,
        max_rounds=int(os.environ.get("RPAS_AFLOW_MAX_ROUNDS", "2")), validation_rounds=1, check_convergence=False,
    )
    operator_patch = _install_lcb_operator_adapter(
        load_tasks(bundle / "search.jsonl") + load_tasks(bundle / "select.jsonl") + load_tasks(bundle / "test.jsonl")
    )
    prompt_patch = _install_aflow_prompt_fallback(
        optimizer,
        "Return ONLY a complete executable Python 3 program that reads standard input and writes standard output. "
        "Do not return markdown, quotes, or explanation.\\n\\nInput program or task:\\n{input}",
    )
    started = time.time()
    os.environ["RPAS_EC1_PHASE"] = "search"
    optimizer.optimize("Graph")
    results_path = workspace / "workspace" / "LiveCodeBench" / "workflows" / "results.json"
    search_rows = json.loads(results_path.read_text(encoding="utf-8"))
    rounds = sorted({int(row["round"]) for row in search_rows if isinstance(row.get("round"), int)})
    if not rounds:
        raise RuntimeError("AFlow LCB search produced no executable workflow")
    selection_rows = []
    select_tasks = load_tasks(bundle / "select.jsonl")
    for round_number in rounds:
        before = recorder.sequence
        log_dir = output / "selection" / f"round_{round_number}"
        score, _, _ = asyncio.run(
            _evaluate_round(optimizer, round_number, bundle / "select.jsonl", log_dir, f"select_round_{round_number}")
        )
        call_count = recorder.sequence - before
        phase_calls = [row for row in _read_calls(telemetry, f"livecodebench-aflow-seed-{args.seed}") if row["split"] == f"select_round_{round_number}"]
        select_outputs = _csv_rows(log_dir, select_tasks)
        valid_output_rate = sum(bool(row["prediction"].strip()) for row in select_outputs) / len(select_outputs)
        execution_rate = 1.0 - sum(bool(row.get("error")) for row in phase_calls) / max(1, len(phase_calls))
        truncated_unextractable = sum(
            row.get("finish_reason") == "length" for row in phase_calls
        ) / max(1, len(phase_calls)) if valid_output_rate < 1.0 else 0.0
        valid = execution_rate >= 0.99 and valid_output_rate >= 0.99 and truncated_unextractable <= 0.05
        selection_rows.append(
            {
                "candidate_id": f"aflow_round_{round_number}", "candidate": {"round": round_number},
                "candidate_name": f"AFlow round {round_number}", "topology": "generated_workflow", "score": score,
                "valid": valid, "is_valid_candidate": valid,
                "valid_output_rate": valid_output_rate, "execution_rate": execution_rate,
                "truncated_unextractable_rate": truncated_unextractable,
                "avg_calls": call_count / len(select_tasks),
                "avg_total_tokens": sum(int(row["total_tokens"]) for row in phase_calls) / len(select_tasks),
                "avg_inference_cost_usd": 0.0, "avg_cross_center_tokens": 0.0,
            }
        )
    operating = select_operating_points(selection_rows)
    test_rows: dict[str, list[dict[str, Any]]] = {}
    for point in ("quality", "efficiency"):
        selected_round = int(operating[point]["candidate"]["round"])
        if point == "efficiency" and selected_round == int(operating["quality"]["candidate"]["round"]):
            test_rows[point] = test_rows["quality"]
            continue
        log_dir = output / f"test_{point}"
        asyncio.run(_evaluate_round(optimizer, selected_round, bundle / "test.jsonl", log_dir, f"test_{point}"))
        test_rows[point] = _csv_rows(log_dir, load_tasks(bundle / "test.jsonl"))
    calls = _read_calls(telemetry, f"livecodebench-aflow-seed-{args.seed}")
    search_calls = [row for row in calls if row["split"] == "search" or row["split"].startswith("select_round_")]
    manifest = {
        **source_manifest(source, workspace, "aflow", args.seed, adapter),
        **recorder.manifest(),
        "run_id": f"livecodebench-aflow-seed-{args.seed}", "method": "aflow", "dataset": "livecodebench",
        "experiment": "EC-1B", "seed": args.seed, "run_kind": "protocol_v4_seed0_pilot", "formal_result": False,
        "formal_result_reason": "One-seed hard-benchmark pilot; three-seed confirmatory gate remains pending",
        "release_version": RELEASE_VERSION, "dataset_revision": DATASET_REVISION,
        "dataset_manifest_sha256": sha256_file(bundle / "DATASET_MANIFEST.json"), "split_manifest": frozen["splits"],
        "gpu": gpu, "model": model, "temperature": 0.0, "max_tokens": 2048, "thinking_disabled": True,
        "native_optimizer": "pinned AFlow Optimizer.optimize('Graph')", "task_adapter": adapter,
        "staged_compatibility_patch": prompt_patch + "; " + operator_patch + "; " + benchmark_runtime_patch,
        "search_rounds": rounds, "selection_split": "D_select", "selection_policy": "shared_select_operating_points",
        "quality_round": operating["quality"]["candidate"]["round"],
        "efficiency_round": operating["efficiency"]["candidate"]["round"],
        "search_calls": len(search_calls), "search_tokens": sum(int(row["total_tokens"]) for row in search_calls),
        "test_quality_score": sum(row["passed"] for row in test_rows["quality"]) / len(test_rows["quality"]),
        "test_efficiency_score": sum(row["passed"] for row in test_rows["efficiency"]) / len(test_rows["efficiency"]),
        "started_at_epoch": started, "finished_at_epoch": time.time(),
    }
    write_native_result(output, manifest, test_rows["quality"], calls, selected=operating, search_rows=search_rows)
    for name, values in (("selection_rows", selection_rows), ("test_outputs_efficiency", test_rows["efficiency"])):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
