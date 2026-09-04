"""Run one isolated native EC-1 baseline search/train and held-out test.

This driver deliberately calls the public upstream ``Optimizer.optimize``
methods.  It is invoked in a separate Python process so imports, globals, and
seed state from AFlow and MaAS cannot contaminate one another.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
import threading
import time
import types
from enum import Enum
from pathlib import Path
from typing import Any

from external_comparison.adapters.native_runtime import (
    seed_everything,
    source_manifest,
    stage_checkout,
    stage_humaneval_data,
    write_aflow_config,
    write_maas_config,
)


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _usage_record(method: str, phase: str, usage: Any, model: str, started: float) -> dict[str, Any]:
    value = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {})
    prompt = int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0)
    completion = int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0)
    return {
        "method": method,
        "phase": phase,
        "model": model,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(value.get("total_tokens", prompt + completion) or prompt + completion),
        "latency_ms": max(0.0, (time.perf_counter() - started) * 1000),
    }


def _csv_rows(root: Path, test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_prompt = {str(row["prompt"]): str(row["task_id"]) for row in test_rows}
    csvs = sorted(root.rglob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not csvs:
        raise RuntimeError(f"native test did not write a CSV under {root}")
    values: list[dict[str, Any]] = []
    with csvs[-1].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt = row.get("inputs", "")
            score = float(row.get("score", "0") or 0.0)
            values.append(
                {
                    "task_id": by_prompt.get(prompt, prompt),
                    "passed": score == 1.0,
                    "native_score": score,
                    "output": row.get("prediction", ""),
                    "native_evaluator": "official_upstream_humaneval",
                }
            )
    if len(values) != 131:
        raise RuntimeError(f"native held-out test must produce 131 rows, found {len(values)}")
    return values


def _install_aflow_runtime_compatibility() -> str:
    """Apply bounded execution-only fixes to the staged AFlow checkout.

    AFlow's public HumanEval entry point explicitly passes its historical
    concurrency of 50 to the benchmark.  EC-1 uses one shared backbone whose
    workflows may make several serial model calls.  Run one workflow at a
    time so each still receives AFlow's unchanged 60-second whole-workflow
    budget; higher task concurrency turns a valid four-call workflow into an
    artificial timeout through backend queueing.  This does not change the
    optimizer, candidate-generation, or evaluator logic.  The daemon worker
    preserves AFlow's 15-second evaluator timeout while allowing a timed-out
    generated program to stop holding the process open.
    """
    from benchmarks.benchmark import BaseBenchmark
    from benchmarks.humaneval import HumanEvalBenchmark

    original_evaluate_all = BaseBenchmark.evaluate_all_problems

    async def bounded_evaluate_all(self, data, agent, max_concurrent_tasks=50):
        return await original_evaluate_all(self, data, agent, min(int(max_concurrent_tasks), 1))

    def daemon_timeout(self, func, call_args, timeout):
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func(*call_args))
            except Exception as exc:  # Preserve upstream exception semantics.
                result.append(exc)
            finally:
                stop_event.set()

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        if not stop_event.wait(timeout):
            raise self.TimeoutError("Function execution timed out")
        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    BaseBenchmark.evaluate_all_problems = bounded_evaluate_all
    HumanEvalBenchmark.run_with_timeout = daemon_timeout
    return "aflow_runtime: max_concurrent_tasks=1; HumanEval timeout worker daemonized"


def _aflow(args, source: Path, run_root: Path) -> dict[str, Any]:
    if args.aflow_test_only:
        return _aflow_test_only(args, source, run_root)
    workspace = stage_checkout(
        source, run_root, "aflow", args.seed, replace=args.replace_workspace, require_clean_git=True
    )
    data = stage_humaneval_data(
        workspace, "aflow", Path(args.dataset_path), Path(args.public_test_path), args.data_seed,
        search_fixture=Path(args.search_fixture), test_fixture=Path(args.test_fixture),
    )
    write_aflow_config(workspace, args.model, args.base_url, args.api_key)
    telemetry = run_root / "_native_aflow_calls.jsonl"
    telemetry.unlink(missing_ok=True)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    seed_everything(args.seed)

    from scripts.async_llm import AsyncLLM, LLMConfig
    from scripts.optimizer import Optimizer

    compatibility_patch = _install_aflow_runtime_compatibility()

    original_call = AsyncLLM.__call__

    async def instrumented_call(self, prompt):
        started = time.perf_counter()
        original_create = self.aclient.chat.completions.create

        async def capped_create(*call_args, **call_kwargs):
            # The upstream client omits this argument; EC-1 must share the
            # same completion cap across methods.
            call_kwargs.setdefault("max_tokens", args.max_tokens)
            call_kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
            return await original_create(*call_args, **call_kwargs)

        self.aclient.chat.completions.create = capped_create
        try:
            result = await original_call(self, prompt)
        finally:
            self.aclient.chat.completions.create = original_create
        history = self.get_usage_summary().get("history", [])
        if history:
            _append(telemetry, _usage_record("aflow", os.environ["RPAS_EC1_PHASE"], history[-1], self.config.model, started))
        return result

    AsyncLLM.__call__ = instrumented_call
    config = LLMConfig({"model": args.model, "key": args.api_key, "base_url": args.base_url, "temperature": 0.0, "top_p": 1.0})
    optimizer = Optimizer(
        dataset="HumanEval",
        question_type="code",
        opt_llm_config=config,
        exec_llm_config=config,
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
        optimized_path="workspace",
        sample=args.aflow_sample,
        initial_round=1,
        max_rounds=args.aflow_max_rounds,
        validation_rounds=args.aflow_validation_rounds,
        check_convergence=False,
    )
    started = time.perf_counter()
    os.environ["RPAS_EC1_PHASE"] = "search"
    optimizer.optimize("Graph")
    search_wall_clock = time.perf_counter() - started
    results_path = workspace / "workspace" / "HumanEval" / "workflows" / "results.json"
    if not results_path.is_file():
        raise RuntimeError("AFlow Optimizer search did not create workflows/results.json")
    search_rows = json.loads(results_path.read_text(encoding="utf-8"))
    candidates = [row for row in search_rows if isinstance(row.get("round"), int) and (workspace / "workspace" / "HumanEval" / "workflows" / f"round_{row['round']}" / "graph.py").is_file()]
    if not candidates:
        raise RuntimeError("AFlow Optimizer search did not materialize an executable workflow")
    selected = max(candidates, key=lambda row: float(row.get("score", 0.0)))
    selected_round = int(selected["round"])
    # Upstream Optimizer.test() has a fixed ``rounds=[1]``.  The copied
    # selected workflow keeps that official evaluator while selecting by the
    # search-only validation artifact.
    test_root = workspace / "workspace" / "HumanEval" / "workflows_test"
    shutil.rmtree(test_root, ignore_errors=True)
    shutil.copytree(workspace / "workspace" / "HumanEval" / "workflows" / f"round_{selected_round}", test_root / "round_1")
    # The source round contains its validation CSV.  Never let that artifact
    # be mistaken for held-out output if the upstream test coroutine fails to
    # write a fresh file.
    for stale_csv in (test_root / "round_1").glob("*.csv"):
        stale_csv.unlink()
    os.environ["RPAS_EC1_PHASE"] = "test"
    # AFlow exposes test() as an async coroutine while optimize("Test") is
    # the synchronous wrapper.  The native driver is synchronous here, so run
    # the official coroutine explicitly and fail loudly on any test error.
    asyncio.run(optimizer.test())
    test_rows = [json.loads(line) for line in Path(data["test_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = _csv_rows(test_root / "round_1", test_rows)
    return {
        "manifest": {**source_manifest(source, workspace, "aflow", args.seed, data), "implementation_status": "official_optimizer_graph_then_selected_workflow_test", "aflow_max_rounds": args.aflow_max_rounds, "aflow_sample": args.aflow_sample, "aflow_validation_rounds": args.aflow_validation_rounds, "selected_round": selected_round, "selected_validation_score": selected.get("score"), "search_wall_clock_seconds": search_wall_clock, "staged_compatibility_patch": compatibility_patch},
        "search_rows": search_rows,
        "test_rows": outputs,
        "telemetry_path": str(telemetry),
    }


def _aflow_test_only(args, source: Path, run_root: Path) -> dict[str, Any]:
    """Evaluate a completed, search-only AFlow workspace on held-out tasks.

    This is intentionally narrow: it may only be used after the native AFlow
    search has completed.  The selected workflow is reconstructed exclusively
    from ``workflows/results.json`` before the held-out evaluator is invoked.
    It never calls ``Optimizer.optimize('Graph')`` or stages a new workspace.
    """
    workspace = Path(args.aflow_existing_workspace).resolve()
    if not workspace.is_dir() or workspace.parent.name != "_workspaces":
        raise ValueError("--aflow-existing-workspace must be an existing isolated AFlow workspace")
    results_path = workspace / "workspace" / "HumanEval" / "workflows" / "results.json"
    if not results_path.is_file():
        raise RuntimeError("AFlow test-only resume requires completed workflows/results.json")
    search_rows = json.loads(results_path.read_text(encoding="utf-8"))
    workflow_root = workspace / "workspace" / "HumanEval" / "workflows"
    candidates = [
        row for row in search_rows
        if isinstance(row.get("round"), int)
        and (workflow_root / f"round_{row['round']}" / "graph.py").is_file()
    ]
    if not candidates:
        raise RuntimeError("AFlow test-only resume found no executable search candidate")
    selected = max(candidates, key=lambda row: float(row.get("score", 0.0)))
    selected_round = int(selected["round"])
    data = stage_humaneval_data(
        workspace, "aflow", Path(args.dataset_path), Path(args.public_test_path), args.data_seed,
        search_fixture=Path(args.search_fixture), test_fixture=Path(args.test_fixture),
    )
    write_aflow_config(workspace, args.model, args.base_url, args.api_key)
    telemetry = run_root / "_native_aflow_calls.jsonl"
    if not telemetry.is_file():
        raise RuntimeError("AFlow test-only resume requires preserved search telemetry")
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    seed_everything(args.seed)
    from scripts.async_llm import AsyncLLM, LLMConfig
    from scripts.optimizer import Optimizer

    compatibility_patch = _install_aflow_runtime_compatibility()
    original_call = AsyncLLM.__call__

    async def instrumented_call(self, prompt):
        started = time.perf_counter()
        original_create = self.aclient.chat.completions.create

        async def capped_create(*call_args, **call_kwargs):
            call_kwargs.setdefault("max_tokens", args.max_tokens)
            call_kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
            return await original_create(*call_args, **call_kwargs)

        self.aclient.chat.completions.create = capped_create
        try:
            result = await original_call(self, prompt)
        finally:
            self.aclient.chat.completions.create = original_create
        history = self.get_usage_summary().get("history", [])
        if history:
            _append(telemetry, _usage_record("aflow", "test", history[-1], self.config.model, started))
        return result

    AsyncLLM.__call__ = instrumented_call
    config = LLMConfig({"model": args.model, "key": args.api_key, "base_url": args.base_url, "temperature": 0.0, "top_p": 1.0})
    optimizer = Optimizer(
        dataset="HumanEval", question_type="code", opt_llm_config=config, exec_llm_config=config,
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"], optimized_path="workspace",
        sample=args.aflow_sample, initial_round=1, max_rounds=args.aflow_max_rounds,
        validation_rounds=args.aflow_validation_rounds, check_convergence=False,
    )
    test_root = workspace / "workspace" / "HumanEval" / "workflows_test"
    shutil.rmtree(test_root, ignore_errors=True)
    shutil.copytree(workflow_root / f"round_{selected_round}", test_root / "round_1")
    for stale_csv in (test_root / "round_1").glob("*.csv"):
        stale_csv.unlink()
    os.environ["RPAS_EC1_PHASE"] = "test"
    asyncio.run(optimizer.test())
    test_rows = [json.loads(line) for line in Path(data["test_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = _csv_rows(test_root / "round_1", test_rows)
    return {
        "manifest": {
            **source_manifest(source, workspace, "aflow", args.seed, data),
            "implementation_status": "official_optimizer_graph_then_selected_workflow_test",
            "aflow_max_rounds": args.aflow_max_rounds,
            "aflow_sample": args.aflow_sample,
            "aflow_validation_rounds": args.aflow_validation_rounds,
            "selected_round": selected_round,
            "selected_validation_score": selected.get("score"),
            "search_reused_completed_workspace": True,
            "workspace_reused": True,
            "search_result_sha256": __import__("hashlib").sha256(results_path.read_bytes()).hexdigest(),
            "staged_compatibility_patch": compatibility_patch,
        },
        "search_rows": search_rows,
        "test_rows": outputs,
        "telemetry_path": str(telemetry),
    }


def _install_maas_import_compat(workspace: Path) -> None:
    import maas

    tools_module = types.ModuleType("maas.tools")
    tools_module.__path__ = [str(workspace / "maas" / "tools")]

    class SearchEngineType(Enum):
        SERPAPI_GOOGLE = "serpapi"
        SERPER_GOOGLE = "serper"
        DIRECT_GOOGLE = "google"
        DUCK_DUCK_GO = "ddg"
        CUSTOM_ENGINE = "custom"
        BING = "bing"

    class WebBrowserEngineType(Enum):
        PLAYWRIGHT = "playwright"
        SELENIUM = "selenium"
        CUSTOM = "custom"

    tools_module.SearchEngineType = SearchEngineType
    tools_module.WebBrowserEngineType = WebBrowserEngineType
    sys.modules["maas.tools"] = tools_module
    maas.tools = tools_module


def _install_maas_optional_encoding_compat() -> None:
    """Supply MaAS's optional encoding detector when its extra is absent.

    The HumanEval code path reads the staged UTF-8 JSONL fixtures explicitly,
    so it never exercises ``chardet.detect``.  MaAS imports the optional
    package eagerly through its general document helpers, however.  Keep the
    upstream importable without changing controller training, evaluation, or
    LLM execution semantics.
    """
    try:
        __import__("chardet")
    except ModuleNotFoundError:
        fallback = types.ModuleType("chardet")
        fallback.detect = lambda _raw: {"encoding": "utf-8", "confidence": 0.0}
        sys.modules["chardet"] = fallback
    try:
        __import__("gitignore_parser")
    except ModuleNotFoundError:
        fallback = types.ModuleType("gitignore_parser")
        fallback.parse_gitignore = lambda *_args, **_kwargs: (lambda _path: False)
        sys.modules["gitignore_parser"] = fallback


def _install_maas_provider_compat(workspace: Path) -> str:
    """Restrict the staged provider registry to EC-1's sole allowed backend.

    The released registry eagerly imports every cloud-provider SDK.  EC-1
    freezes one local OpenAI-compatible Qwen endpoint, so leaving those unused
    imports in place makes reproducibility depend on unrelated credentials and
    packages.  This changes package bootstrap only; ``OpenAILLM`` and all
    HumanEval optimizer code remain upstream.
    """
    init_path = workspace / "maas" / "provider" / "__init__.py"
    init_path.write_text(
        "from maas.provider.openai_api import OpenAILLM\n\n__all__ = ['OpenAILLM']\n",
        encoding="utf-8",
    )
    return "staged provider registry limited to OpenAILLM; unused cloud SDK imports excluded"


def _install_maas_actions_compat(workspace: Path) -> str:
    """Avoid unrelated notebook and project-management action imports.

    HumanEval's official MaAS benchmark reaches ``ActionNode`` plus the base
    action classes.  The upstream package initializer additionally imports
    every notebook, browsing, and document action, each with independent SDK
    extras.  Limit the staged package registry to the primitives actually
    reached by this benchmark.
    """
    init_path = workspace / "maas" / "actions" / "__init__.py"
    init_path.write_text(
        "from maas.actions.action import Action\n"
        "from maas.actions.action_output import ActionOutput\n\n"
        "__all__ = ['Action', 'ActionOutput']\n",
        encoding="utf-8",
    )
    return "staged actions registry limited to HumanEval action primitives"


def _install_maas_public_test_compat(workspace: Path) -> str:
    """Bound MaAS's public-test operator without changing its test policy.

    The released HumanEval ``Test.exec_code`` executes model-provided code in
    the benchmark's main process. A non-terminating candidate then freezes the
    entire controller-training run. Keep the same frozen assertions and the
    same pass/fail feedback contract, but run each candidate in an isolated
    Python interpreter with the common 10-second EC-1 public-test budget.
    """
    replacement = '''    def exec_code(self, solution, entry_point):
        import os
        import subprocess
        import sys
        import tempfile

        test_cases = extract_test_cases_from_jsonl(entry_point, dataset="HumanEval")
        if not test_cases:
            return {"exec_fail_case": f"No public test cases for {entry_point}"}
        assertions = "\\n".join(f"    {case}" for case in test_cases)
        program = "\\n".join((
            str(solution),
            "",
            "def check(candidate):",
            assertions,
            "",
            f"check({entry_point})",
            "",
        ))
        try:
            with tempfile.TemporaryDirectory(prefix="maas_ec1_public_test_") as directory:
                script = os.path.join(directory, "candidate.py")
                with open(script, "w", encoding="utf-8") as handle:
                    handle.write(program)
                completed = subprocess.run(
                    [sys.executable, "-I", script], capture_output=True, text=True,
                    timeout=10, check=False,
                )
        except subprocess.TimeoutExpired:
            return {"exec_fail_case": "Public-test execution timed out after 10 seconds."}
        if completed.returncode == 0:
            return "no error"
        feedback = (completed.stderr or completed.stdout or "public-test assertion failed").strip()[-1200:]
        return {"exec_fail_case": feedback}
'''
    for split in ("train", "test"):
        template = (
            workspace
            / "maas"
            / "ext"
            / "maas"
            / "scripts"
            / "optimized"
            / "HumanEval"
            / split
            / "template"
            / "operator.py"
        )
        source = template.read_text(encoding="utf-8")
        start = source.index("    def exec_code(self, solution, entry_point):")
        end = source.index("\n    async def __call__(", start)
        template.write_text(source[:start] + replacement + source[end:], encoding="utf-8")
    return "MaAS HumanEval Test.exec_code isolated in a 10-second subprocess using the frozen public fixture"


def _maas(args, source: Path, run_root: Path) -> dict[str, Any]:
    workspace = stage_checkout(
        source, run_root, "maas", args.seed, replace=args.replace_workspace, require_clean_git=True
    )
    data = stage_humaneval_data(
        workspace, "maas", Path(args.dataset_path), Path(args.public_test_path), args.data_seed,
        search_fixture=Path(args.search_fixture), test_fixture=Path(args.test_fixture),
    )
    write_maas_config(workspace, args.model, args.base_url, args.api_key, args.seed)
    telemetry = run_root / "_native_maas_calls.jsonl"
    if not args.maas_test_only:
        telemetry.unlink(missing_ok=True)
    os.environ["METAGPT_PROJECT_ROOT"] = str(workspace)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    seed_everything(args.seed)

    _install_maas_import_compat(workspace)
    _install_maas_optional_encoding_compat()
    provider_patch = _install_maas_provider_compat(workspace)
    actions_patch = _install_maas_actions_compat(workspace)
    public_test_patch = _install_maas_public_test_compat(workspace)
    from maas.configs.models_config import ModelsConfig
    from maas.ext.maas.scripts.optimizer import Optimizer
    from maas.ext.maas.scripts.optimizer_utils.data_utils import DataUtils
    from maas.provider.base_llm import BaseLLM
    from maas.provider.openai_api import OpenAILLM

    # The released call sites pass only ``round, score`` although the released
    # DataUtils requires three additional arguments.  This narrow compatibility
    # shim is applied only to the staged checkout and is recorded in the result.
    original_create_result = DataUtils.create_result_data

    def compatible_result(self, round, score, avg_cost=0.0, total_cost=0.0, token=0):
        return original_create_result(self, round, score, avg_cost, total_cost, token)

    DataUtils.create_result_data = compatible_result
    original_update = BaseLLM._update_costs

    def instrumented_update(self, usage, model=None, local_calc_usage=True):
        started = time.perf_counter()
        result = original_update(self, usage, model=model, local_calc_usage=local_calc_usage)
        _append(telemetry, _usage_record("maas", os.environ["RPAS_EC1_PHASE"], usage, str(model or getattr(self, "model", args.model)), started))
        return result

    BaseLLM._update_costs = instrumented_update
    original_kwargs = OpenAILLM._cons_kwargs

    def controlled_kwargs(self, messages, timeout=0, **extra_kwargs):
        values = original_kwargs(self, messages, timeout=timeout, **extra_kwargs)
        values["max_tokens"] = args.max_tokens
        values["top_p"] = 1.0
        values.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
        return values

    OpenAILLM._cons_kwargs = controlled_kwargs
    config = ModelsConfig.default().get(args.model)
    if config is None:
        raise RuntimeError(f"staged MaAS model config not found: {args.model}")
    optimizer = Optimizer(
        dataset="HumanEval",
        question_type="code",
        opt_llm_config=config,
        exec_llm_config=config,
        operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Test", "SelfRefine", "EarlyStop"],
        optimized_path="maas/ext/maas/scripts/optimized",
        sample=args.maas_sample,
        round=1,
        batch_size=args.maas_batch_size,
        lr=args.maas_lr,
        is_textgrad=False,
    )
    checkpoint = workspace / "maas" / "ext" / "maas" / "scripts" / "optimized" / "HumanEval" / "train" / "round_1" / f"HumanEval_controller_sample{args.maas_sample}.pth"
    search_wall_clock: float | None = None
    if args.maas_test_only:
        saved_controller = Path(args.maas_controller_path).resolve()
        if not saved_controller.is_file() or saved_controller.stat().st_size < 1024:
            raise RuntimeError("--maas-controller-path must reference a non-empty trained controller checkpoint")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved_controller, checkpoint)
    else:
        started = time.perf_counter()
        os.environ["RPAS_EC1_PHASE"] = "search"
        optimizer.optimize("Graph")
        search_wall_clock = time.perf_counter() - started
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
        raise RuntimeError(f"MaAS fresh training did not materialize controller checkpoint: {checkpoint}")
    os.environ["RPAS_EC1_PHASE"] = "test"
    optimizer.optimize("Test")
    test_rows = [json.loads(line) for line in Path(data["test_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    test_root = workspace / "maas" / "ext" / "maas" / "scripts" / "optimized" / "HumanEval" / "test" / "round_1"
    outputs = _csv_rows(test_root, test_rows)
    return {
        "manifest": {**source_manifest(source, workspace, "maas", args.seed, data), "implementation_status": "official_optimizer_reused_trained_controller_then_test" if args.maas_test_only else "official_optimizer_fresh_train_checkpoint_then_test", "maas_sample": args.maas_sample, "maas_batch_size": args.maas_batch_size, "maas_lr": args.maas_lr, "search_wall_clock_seconds": search_wall_clock, "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_reused": args.maas_test_only, "checkpoint_source": str(Path(args.maas_controller_path).resolve()) if args.maas_test_only else None, "staged_compatibility_patch": "DataUtils.create_result_data optional avg_cost,total_cost,token; " + provider_patch + "; " + actions_patch + "; " + public_test_patch},
        "search_rows": [{"round": 1, "checkpoint": str(checkpoint), "reused": args.maas_test_only}],
        "test_rows": outputs,
        "telemetry_path": str(telemetry),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Native EC-1 isolated baseline driver")
    parser.add_argument("--method", choices=("aflow", "maas"), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--public-test-path", required=True)
    parser.add_argument("--search-fixture", required=True)
    parser.add_argument("--test-fixture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=1024)
    # Upstream AFlow always evaluates the initial round before entering this
    # loop.  Therefore max_rounds=2 yields rounds 1-3; max_rounds=3 would
    # create an unintended round 4 and is not a three-round budget.
    parser.add_argument("--aflow-max-rounds", type=int, default=2)
    parser.add_argument("--aflow-sample", type=int, default=4)
    parser.add_argument("--aflow-validation-rounds", type=int, default=1)
    parser.add_argument("--maas-sample", type=int, default=4)
    parser.add_argument("--maas-batch-size", type=int, default=4)
    parser.add_argument("--maas-lr", type=float, default=0.01)
    parser.add_argument(
        "--maas-test-only",
        action="store_true",
        help="Run official MaAS held-out test with a preserved trained controller; never invoke training.",
    )
    parser.add_argument(
        "--maas-controller-path",
        help="Trained controller checkpoint required by --maas-test-only.",
    )
    parser.add_argument("--replace-workspace", action="store_true")
    parser.add_argument(
        "--aflow-test-only",
        action="store_true",
        help="Run only the official held-out evaluator against a completed isolated AFlow search workspace.",
    )
    parser.add_argument(
        "--aflow-existing-workspace",
        help="Completed isolated AFlow workspace required by --aflow-test-only.",
    )
    args = parser.parse_args()
    if args.aflow_test_only:
        if args.method != "aflow" or not args.aflow_existing_workspace:
            parser.error("--aflow-test-only requires --method aflow and --aflow-existing-workspace")
        if args.replace_workspace:
            parser.error("--aflow-test-only cannot replace its completed search workspace")
    if args.maas_test_only:
        if args.method != "maas" or not args.maas_controller_path:
            parser.error("--maas-test-only requires --method maas and --maas-controller-path")
        if args.replace_workspace:
            parser.error("--maas-test-only cannot replace its staged workspace")
    source = Path(args.source_root).resolve()
    run_root = Path(args.output_dir).resolve()
    result = _aflow(args, source, run_root) if args.method == "aflow" else _maas(args, source, run_root)
    output = run_root / f"_{args.method}_driver_result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
