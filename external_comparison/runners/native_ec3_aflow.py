"""Native AFlow execution path for the locked EC-3 HotpotQA protocol.

The official ``Optimizer.optimize('Graph')`` owns workflow generation.  This
driver only mounts frozen data, records calls, reevaluates search finalists on
the disjoint selection split, and enforces the shared D_test unlock.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from external_comparison.adapters.native_common import call_record, sha256_file
from external_comparison.adapters.native_runtime import seed_everything, stage_checkout, write_aflow_config
from external_comparison.runners.ec3_formal_gate import freeze_state
from external_comparison.runners.ec3_preflight import preflight
from external_comparison.runners.hotpotqa_ec3_data import HotpotExample, answer_scores


PROTOCOL_VERSION = "EC3_HOTPOTQA_V3"
MIN_VALID_ANSWER_RATE = 0.99
MAX_TRUNCATION_RATE = 0.01


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _require_one_allocated_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError("EC-3 requires exactly one Slurm-allocated CUDA_VISIBLE_DEVICES token")
    return visible


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return os.environ.get("RPAS_CODE_COMMIT", "bundle_without_git_metadata")


def _load_split(manifest: dict[str, Any], name: str) -> list[HotpotExample]:
    details = manifest.get("splits", {}).get(name, {})
    path = Path(str(details.get("path", "")))
    if not path.is_file() or sha256_file(path) != details.get("sha256"):
        raise ValueError(f"EC-3 {name} fixture differs from its frozen manifest")
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != int(details.get("count", -1)):
        raise ValueError(f"EC-3 {name} fixture has an invalid count")
    return [HotpotExample(**value) for value in values]


def _write_split(path: Path, rows: list[HotpotExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # AFlow consumes only question/answer/context. Extra provenance
            # fields are intentionally not passed into its benchmark prompt.
            handle.write(json.dumps({"_id": row.task_id, "question": row.question, "answer": row.answer, "context": row.context}, ensure_ascii=False) + "\n")


def _clean_staged_hotpot_workspace(workspace: Path) -> Path:
    root = workspace / "workspace" / "HotpotQA" / "workflows"
    if not (root / "round_1" / "graph.py").is_file():
        raise RuntimeError("pinned AFlow checkout lacks the official HotpotQA round_1 workflow template")
    for path in (root / "results.json", root / "processed_experience.json"):
        path.unlink(missing_ok=True)
    for directory in root.glob("round_*"):
        if directory.name != "round_1":
            shutil.rmtree(directory)
    for stale in (root / "round_1").glob("*.csv"):
        stale.unlink()
    (root / "round_1" / "log.json").unlink(missing_ok=True)
    return root


def _install_runtime_compatibility() -> str:
    """Bound AFlow task concurrency without altering its search algorithm."""
    from benchmarks.benchmark import BaseBenchmark

    original = BaseBenchmark.evaluate_all_problems

    async def serial_evaluate(self, data, agent, max_concurrent_tasks=50):
        return await original(self, data, agent, min(int(max_concurrent_tasks), 1))

    BaseBenchmark.evaluate_all_problems = serial_evaluate
    return "aflow_runtime:max_concurrent_tasks=1"


def _instrument_async_llm(*, calls_path: Path, run_id: str):
    from scripts.async_llm import AsyncLLM

    original = AsyncLLM.__call__

    async def instrumented(self, prompt):
        started = time.perf_counter()
        original_create = self.aclient.chat.completions.create
        cap = int(getattr(self.config, "rpas_max_tokens", 256))

        async def capped_create(*call_args, **call_kwargs):
            call_kwargs.setdefault("max_tokens", cap)
            call_kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
            return await original_create(*call_args, **call_kwargs)

        self.aclient.chat.completions.create = capped_create
        try:
            response = await original(self, prompt)
        finally:
            self.aclient.chat.completions.create = original_create
        history = self.get_usage_summary().get("history", [])
        usage = dict(history[-1]) if history else {}
        usage["latency_ms"] = max(0.0, (time.perf_counter() - started) * 1000)
        phase = os.environ.get("RPAS_EC3_AFLOW_PHASE", "unknown")
        role = str(getattr(self.config, "rpas_role", "executor"))
        usage["agent"] = f"aflow_{role}"
        record = call_record(run_id, "aflow", "hotpotqa", phase, f"aflow:{phase}", 0, usage)
        record["example_id"] = os.environ.get("RPAS_EC3_AFLOW_EXAMPLE", "")
        _append_jsonl(calls_path, [record])
        return response

    AsyncLLM.__call__ = instrumented
    return original


def _csv_after(directory: Path, before: set[Path]) -> Path:
    candidates = [path for path in directory.glob("*.csv") if path not in before]
    if not candidates:
        raise RuntimeError(f"AFlow evaluation did not produce a fresh CSV under {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _parse_csv(path: Path, rows: list[HotpotExample]) -> dict[str, Any]:
    expected = {row.question: row for row in rows}
    outputs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for value in csv.DictReader(handle):
            question = str(value.get("question", ""))
            example = expected.get(question)
            if example is None:
                raise ValueError(f"AFlow CSV has an unknown HotpotQA question: {question[:120]!r}")
            prediction = str(value.get("prediction", "")).strip()
            metrics = answer_scores(prediction, example.answer)
            failed_execution = prediction.lower().startswith((
                "maximum retries reached", "division by zero", "zerodivisionerror", "error:",
            ))
            outputs.append({
                "id": example.task_id, "prediction": prediction, "answer": example.answer,
                "execution_valid": not failed_execution, **metrics,
            })
    if len(outputs) != len(rows):
        raise RuntimeError(f"AFlow evaluation expected {len(rows)} rows but wrote {len(outputs)}")
    valid = sum(bool(row["prediction"]) and bool(row["execution_valid"]) for row in outputs) / len(outputs) if outputs else 0.0
    return {
        "answer_f1": sum(float(row["f1"]) for row in outputs) / len(outputs) if outputs else 0.0,
        "answer_em": sum(float(row["em"]) for row in outputs) / len(outputs) if outputs else 0.0,
        "valid_answer_rate": valid,
        "outputs": outputs,
    }


def _make_optimizer(workspace: Path, *, model: str, api_key: str, base_url: str, executor_cap: int, meta_cap: int, max_rounds: int):
    from scripts.async_llm import LLMConfig
    from scripts.optimizer import Optimizer

    execute_config = LLMConfig({"model": model, "key": api_key, "base_url": base_url, "temperature": 0.0, "top_p": 1.0})
    execute_config.rpas_max_tokens = executor_cap
    execute_config.rpas_role = "executor"
    optimize_config = LLMConfig({"model": model, "key": api_key, "base_url": base_url, "temperature": 0.0, "top_p": 1.0})
    optimize_config.rpas_max_tokens = meta_cap
    optimize_config.rpas_role = "meta"
    return Optimizer(
        dataset="HotpotQA", question_type="qa", opt_llm_config=optimize_config, exec_llm_config=execute_config,
        operators=["Custom", "AnswerGenerate", "ScEnsemble"], optimized_path="workspace", sample=4,
        initial_round=1, max_rounds=max_rounds, validation_rounds=1, check_convergence=False,
    ), execute_config


def _evaluate_round(optimizer: Any, *, round_number: int, split_rows: list[HotpotExample], split_path: Path, log_dir: Path) -> dict[str, Any]:
    from scripts.evaluator import Evaluator

    _write_split(split_path, split_rows)
    importlib.invalidate_caches()
    graph = optimizer.graph_utils.load_graph(round_number, "workspace/HotpotQA/workflows")
    before = set(log_dir.glob("*.csv"))
    score, _, _ = asyncio.run(
        Evaluator(eval_path=str(log_dir)).graph_evaluate(
            "HotpotQA", graph, {"dataset": "HotpotQA", "llm_config": optimizer.execute_llm_config}, str(log_dir), is_test=False,
        )
    )
    parsed = _parse_csv(_csv_after(log_dir, before), split_rows)
    parsed["upstream_answer_f1"] = float(score)
    parsed["round"] = round_number
    return parsed


def _workflow_rounds(workflows: Path) -> list[int]:
    values = []
    for directory in workflows.glob("round_*"):
        suffix = directory.name.removeprefix("round_")
        if suffix.isdigit() and (directory / "graph.py").is_file():
            values.append(int(suffix))
    return sorted(values)


def _truncation_rate(calls_path: Path) -> float:
    rows = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines() if line.strip()] if calls_path.exists() else []
    generated = [row for row in rows if row.get("split") != "unknown"]
    if not generated:
        return 1.0
    return sum(int(row.get("completion_tokens", 0)) >= (4096 if row.get("agent") == "aflow_meta" else 256) for row in generated) / len(generated)


def _manifest_base(manifest: dict[str, Any], *, seed: int, gpu: str, executor_cap: int, meta_cap: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION, "dataset": "hotpotqa", "method": "aflow", "seed": seed,
        "data_seed": manifest["data_seed"], "split_manifest_sha256": manifest["split_manifest_sha256"],
        "split_protocol": "calib__search__select__test_locked", "d_test_accessed": False,
        "executor_model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "executor_max_tokens": executor_cap, "meta_max_tokens": meta_cap, "temperature": 0.0,
        "runtime_cuda_visible_devices": gpu,
    }


def _setup(args: argparse.Namespace, *, output: Path, seed: int) -> tuple[dict[str, Any], Path, Path, int, int, str]:
    gpu = _require_one_allocated_gpu()
    manifest = _read_json(Path(args.manifest))
    executor_cap = int(os.environ.get("RPAS_EC3_EXECUTOR_MAX_TOKENS", "256"))
    meta_cap = int(os.environ.get("RPAS_EC3_META_MAX_TOKENS", "4096"))
    if executor_cap not in {256, 512} or meta_cap < 2048:
        raise ValueError("EC-3 requires executor cap 256/512 and meta cap at least 2048")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse EC-3 AFlow workspace: {output}")
    workspace = stage_checkout(Path(args.aflow_root), output, "aflow", seed, require_clean_git=True)
    _clean_staged_hotpot_workspace(workspace)
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE", "").strip()
    if not endpoint:
        raise RuntimeError("RPAS_EXTERNAL_API_BASE is required")
    write_aflow_config(workspace, os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"), endpoint, os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"))
    return manifest, workspace, output / "calls.jsonl", executor_cap, meta_cap, gpu


def run_calibration(args: argparse.Namespace) -> Path:
    output = Path(args.output_root) / "calibration" / "aflow"
    manifest, workspace, calls_path, executor_cap, meta_cap, gpu = _setup(args, output=output, seed=2026)
    calibration = _load_split(manifest, "calib")
    split_path = workspace / "data" / "datasets" / "hotpotqa_validate.jsonl"
    _write_split(split_path, calibration)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    seed_everything(2026)
    _install_runtime_compatibility()
    _instrument_async_llm(calls_path=calls_path, run_id="ec3-hotpotqa-aflow-calibration")
    optimizer, _ = _make_optimizer(workspace, model=os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"), api_key=os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"), base_url=os.environ["RPAS_EXTERNAL_API_BASE"], executor_cap=executor_cap, meta_cap=meta_cap, max_rounds=2)
    os.environ["RPAS_EC3_AFLOW_PHASE"] = "calib_search"
    optimizer.optimize("Graph")
    workflows = workspace / "workspace" / "HotpotQA" / "workflows"
    rounds = _workflow_rounds(workflows)
    evaluations = []
    for round_number in rounds[:2]:
        os.environ["RPAS_EC3_AFLOW_PHASE"] = "calib_executor"
        evaluations.append(_evaluate_round(optimizer, round_number=round_number, split_rows=calibration, split_path=split_path, log_dir=workflows / f"round_{round_number}"))
    executable_rate = sum(row["valid_answer_rate"] >= MIN_VALID_ANSWER_RATE for row in evaluations) / len(evaluations) if evaluations else 0.0
    truncation = _truncation_rate(calls_path)
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines() if line.strip()] if calls_path.exists() else []
    payload = {
        **_manifest_base(manifest, seed=-1, gpu=gpu, executor_cap=executor_cap, meta_cap=meta_cap),
        "run_kind": "calibration", "aflow_search": {"new_workflow_rounds": sum(round_number > 1 for round_number in rounds), "optimizer_calls": sum(row.get("agent") == "aflow_meta" for row in calls), "workflow_executable_rate": executable_rate},
        "executor_generation_truncation_rate": truncation, "status": "passed" if executable_rate >= 0.95 and truncation < MAX_TRUNCATION_RATE else "failed",
    }
    _write_json(output / "calibration_manifest.json", payload)
    _append_jsonl(output / "calibration_rows.jsonl", evaluations)
    if payload["status"] != "passed":
        raise RuntimeError("EC-3 AFlow calibration failed; do not begin formal AFlow search")
    return output


def run_pretest(args: argparse.Namespace) -> Path:
    output = Path(args.output_root) / "aflow" / f"seed_{args.seed}"
    manifest, workspace, calls_path, executor_cap, meta_cap, gpu = _setup(args, output=output, seed=args.seed)
    search = _load_split(manifest, "search")
    select = _load_split(manifest, "select")
    split_path = workspace / "data" / "datasets" / "hotpotqa_validate.jsonl"
    _write_split(split_path, search)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    seed_everything(args.seed)
    _install_runtime_compatibility()
    _instrument_async_llm(calls_path=calls_path, run_id=f"ec3-hotpotqa-aflow-seed-{args.seed}")
    optimizer, _ = _make_optimizer(workspace, model=os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"), api_key=os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"), base_url=os.environ["RPAS_EXTERNAL_API_BASE"], executor_cap=executor_cap, meta_cap=meta_cap, max_rounds=2)
    started = time.time()
    os.environ["RPAS_EC3_AFLOW_PHASE"] = "search"
    optimizer.optimize("Graph")
    workflows = workspace / "workspace" / "HotpotQA" / "workflows"
    rounds = _workflow_rounds(workflows)
    if len(rounds) < 2:
        raise RuntimeError("AFlow search did not create a new workflow round")
    # The top five workflow rounds are selected only using upstream D_search scores.
    results = json.loads((workflows / "results.json").read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError("AFlow workflow results.json must be a list")
    scored = {int(row["round"]): float(row.get("score", 0.0)) for row in results if isinstance(row.get("round"), int)}
    finalists = sorted(rounds, key=lambda number: (-scored.get(number, float("-inf")), number))[:5]
    selection = []
    for round_number in finalists:
        os.environ["RPAS_EC3_AFLOW_PHASE"] = "select"
        result = _evaluate_round(optimizer, round_number=round_number, split_rows=select, split_path=split_path, log_dir=workflows / f"round_{round_number}")
        selection.append(result)
    selected = min(selection, key=lambda row: (-float(row["answer_f1"]), row["round"]))
    selected_payload = {"round": selected["round"], "selection_answer_f1": selected["answer_f1"], "selection_answer_em": selected["answer_em"], "selection_policy": "max_d_select_f1__round"}
    _write_json(output / "selected_candidate.json", selected_payload)
    _append_jsonl(output / "selection_rows.jsonl", selection)
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines() if line.strip()] if calls_path.exists() else []
    meta_calls = [row for row in calls if row.get("agent") == "aflow_meta"]
    executor_calls = [row for row in calls if row.get("agent") == "aflow_executor"]
    executable_rate = sum(row["valid_answer_rate"] >= MIN_VALID_ANSWER_RATE for row in selection) / len(selection) if selection else 0.0
    payload = {
        **_manifest_base(manifest, seed=args.seed, gpu=gpu, executor_cap=executor_cap, meta_cap=meta_cap),
        "run_id": f"ec3-hotpotqa-aflow-seed-{args.seed}", "run_kind": "pretest_search_select", "code_commit": _git_commit(Path(args.repo_root)),
        "search_calls": len(calls), "search_tokens": sum(int(row.get("total_tokens", 0)) for row in calls), "search_wall_clock_seconds": time.time() - started,
        "aflow_search": {"new_workflow_rounds": sum(round_number > 1 for round_number in rounds), "optimizer_calls": len(meta_calls), "workflow_executable_rate": executable_rate, "executor_calls": len(executor_calls), "finalists": len(finalists)},
    }
    _write_json(output / "run_manifest.json", payload)
    freeze_state(output)
    return output


def run_test(args: argparse.Namespace) -> Path:
    gpu = _require_one_allocated_gpu()
    manifest = _read_json(Path(args.manifest))
    root = Path(args.output_root)
    unlock = _read_json(root / "d_test_unlock.json") if (root / "d_test_unlock.json").is_file() else {}
    if not unlock.get("d_test_unlocked") or unlock.get("split_manifest_sha256") != manifest.get("split_manifest_sha256"):
        raise RuntimeError("EC-3 D_test is locked until all six immutable final states are present")
    run = root / "aflow" / f"seed_{args.seed}"
    selected = _read_json(run / "selected_candidate.json")
    if (run / "test_summary.json").exists():
        raise FileExistsError(f"refusing to overwrite EC-3 AFlow held-out result: {run}")
    workspace = run / "_workspaces" / f"aflow_seed_{args.seed}"
    if not workspace.is_dir():
        raise FileNotFoundError(f"missing frozen AFlow workspace: {workspace}")
    executor_cap = int(os.environ.get("RPAS_EC3_EXECUTOR_MAX_TOKENS", "256"))
    meta_cap = int(os.environ.get("RPAS_EC3_META_MAX_TOKENS", "4096"))
    test = _load_split(manifest, "test")
    split_path = workspace / "data" / "datasets" / "hotpotqa_validate.jsonl"
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    _install_runtime_compatibility()
    _instrument_async_llm(calls_path=run / "test_calls.jsonl", run_id=f"ec3-hotpotqa-aflow-seed-{args.seed}")
    optimizer, _ = _make_optimizer(workspace, model=os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"), api_key=os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"), base_url=os.environ["RPAS_EXTERNAL_API_BASE"], executor_cap=executor_cap, meta_cap=meta_cap, max_rounds=2)
    os.environ["RPAS_EC3_AFLOW_PHASE"] = "test"
    result = _evaluate_round(optimizer, round_number=int(selected["round"]), split_rows=test, split_path=split_path, log_dir=workspace / "workspace" / "HotpotQA" / "workflows" / f"round_{selected['round']}")
    _append_jsonl(run / "test_outputs.jsonl", result["outputs"])
    calls = [json.loads(line) for line in (run / "test_calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    truncation = _truncation_rate(run / "test_calls.jsonl")
    summary = {"protocol_version": PROTOCOL_VERSION, "method": "aflow", "seed": args.seed, "split_manifest_sha256": manifest["split_manifest_sha256"], "d_test_accessed": True, "runtime_cuda_visible_devices": gpu, "generation_truncation_rate": truncation, "test_calls": len(calls), "test_tokens": sum(int(row["total_tokens"]) for row in calls), **{key: value for key, value in result.items() if key != "outputs"}}
    _write_json(run / "test_summary.json", summary)
    if result["valid_answer_rate"] < MIN_VALID_ANSWER_RATE or truncation >= MAX_TRUNCATION_RATE:
        raise RuntimeError("EC-3 AFlow held-out run failed answer/truncation gate")
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description="Native AFlow EC-3 HotpotQA V3 runner")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--aflow-root", required=True)
    parser.add_argument("--output-root", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibration")
    pretest = sub.add_parser("pretest")
    pretest.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    test = sub.add_parser("test")
    test.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())
    args.manifest = str(Path(args.manifest).resolve())
    args.aflow_root = str(Path(args.aflow_root).resolve())
    args.output_root = str(Path(args.output_root).resolve())
    if args.command in {"calibration", "pretest"}:
        preflight(manifest_path=Path(args.manifest), aflow_root=Path(args.aflow_root), expected_endpoint=os.environ.get("RPAS_EXTERNAL_API_BASE"))
    target = run_calibration(args) if args.command == "calibration" else run_pretest(args) if args.command == "pretest" else run_test(args)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
