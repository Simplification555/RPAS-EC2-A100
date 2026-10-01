#!/usr/bin/env python3
"""Run isolated native-method diagnostics on two embedded synthetic problems.

No frozen AIME file or formal selection/test artifact is opened. The diagnostic
uses the formal transport/template patches, but its concurrency is two, MaAS
trains for one repetition, ADAS executes one seed architecture, and G-Designer
only performs forward inference. These are compatibility checks, not scores.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
NAMES = {"aflow": "AFlow", "maas": "MaAS", "adas": "ADAS", "gdesigner": "GDesigner"}
RELATIVE_ROOT = Path("workspace/rpas_aime_native/MATH")
TASKS = (
    {"id": "synthetic_add", "problem": "Compute 3 + 4. Give the final integer in \\boxed{}.", "answer": "7"},
    {"id": "synthetic_multiply", "problem": "Compute 3 times 4. Give the final integer in \\boxed{}.", "answer": "12"},
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_run_dir(path: Path, source: Path) -> Path:
    result = path.expanduser().resolve()
    forbidden = [ROOT / "outputs", ROOT / "data", ROOT / "upstream", source]
    for key in ("AIME_OUTPUT_DIR", "RPAS_OUTPUT_ROOT"):
        if os.environ.get(key):
            forbidden.append(Path(os.environ[key]).expanduser().resolve())
    if result == ROOT or any(result == item.resolve() or item.resolve() in result.parents for item in forbidden):
        raise ValueError("diagnostic directory must be separate from formal outputs, frozen data, and upstream sources")
    if result.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic: {result}")
    return result


def stage_native(method: str, source: Path, run_dir: Path, endpoint: str, max_tokens: int) -> tuple[Path, dict[str, Any]]:
    import native_aime_formal as native

    repo = run_dir / f"{method}_repo"
    native.copy_tree(source, repo)
    rows = [{"problem": task["problem"], "solution": f"\\boxed{{{task['answer']}}}", "synthetic_id": task["id"]} for task in TASKS]
    if method == "aflow":
        native.patch_aflow_code_formatter(repo)
        native.patch_aflow_optimizer_retry_guard(repo)
        workflow = repo / RELATIVE_ROOT / "workflows"
        for part in ("round_1", "template"):
            shutil.copytree(repo / "workspace/MATH/workflows" / part, workflow / part)
        # The upstream seed also has saved scores/mismatch examples; only
        # its graph and prompt are inputs to this fresh synthetic evaluation.
        for item in (workflow / "round_1").iterdir():
            if item.suffix in {".json", ".csv"}:
                item.unlink()
        native.ensure_package_tree(workflow, repo)
        for filename in ("graph.py", "prompt.py"):
            path = workflow / "round_1" / filename
            path.write_text(path.read_text(encoding="utf-8").replace("workspace.MATH", "workspace.rpas_aime_native.MATH"), encoding="utf-8")
        native.write_jsonl(repo / "data/datasets/math_validate.jsonl", rows)
        native.write_jsonl(repo / "data/datasets/math_test.jsonl", [])
        return repo, {"synthetic_rows": len(rows), "workflow": str(RELATIVE_ROOT / "workflows")}

    for patch in (native.patch_maas_optional_provider_import, native.patch_maas_provider_surface,
                  native.patch_maas_optional_backoff, native.patch_maas_optional_tools,
                  native.patch_maas_embedding_path):
        patch(repo)
    native.make_maas_config(repo, endpoint, max_tokens)
    from maas_context_guard import patch_template_directory
    repairs = []
    for split in ("train", "test"):
        original = repo / "maas/ext/maas/scripts/optimized/MATH" / split
        repairs.append(patch_template_directory(original / "template", concise=False))
        shutil.copytree(original, repo / RELATIVE_ROOT / split)
    native.ensure_package_tree(repo / RELATIVE_ROOT, repo)
    native.write_jsonl(repo / "maas/ext/maas/data/math_train.jsonl", rows)
    native.write_jsonl(repo / "maas/ext/maas/data/math_test.jsonl", [])
    return repo, {"synthetic_rows": len(rows), "training_repetitions": 1, "template_repairs": repairs}


def audit_csv(directory: Path) -> dict[str, Any]:
    from answer_protocol import score_aime

    paths = list(directory.glob("*.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one native prediction CSV, found {len(paths)} in {directory}")
    with paths[0].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(TASKS):
        raise RuntimeError(f"expected two synthetic predictions, found {len(rows)}")
    expected = {task["problem"]: task for task in TASKS}
    if {row["question"] for row in rows} != set(expected):
        raise RuntimeError("native predictions do not cover exactly the synthetic questions")
    scored = [{"id": expected[row["question"]]["id"], "native_score": row["score"],
               **score_aime(row.get("prediction", ""), expected[row["question"]]["answer"])} for row in rows]
    return {"csv": str(paths[0]), "rows": len(rows), "correct": sum(bool(row["correct"]) for row in scored),
            "parser_valid": sum(bool(row["parser_valid"]) for row in scored), "scores": scored}


def export_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    result = dict(telemetry)
    guard = result.pop("context_guard", None) or result.pop("_context_guard", None)
    if guard is not None:
        result["context_guard"] = guard.summary()
    return result


def run_aflow(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    import native_aime_formal as native
    telemetry = native.patch_aflow_runtime(args.max_tokens, concurrency=2)
    from scripts.optimizer import Optimizer
    from scripts.evaluator import Evaluator
    from scripts.optimizer_utils.graph_utils import GraphUtils
    native.patch_aflow_graph_namespace(GraphUtils)
    native.seed_everything(0)
    config = native.make_aflow_config(args.endpoint)
    workflow = RELATIVE_ROOT / "workflows"
    if args.aflow_mutation:
        optimizer = Optimizer(dataset="MATH", question_type="math", opt_llm_config=config,
            exec_llm_config=config, operators=["Custom", "ScEnsemble", "Programmer"], sample=1,
            check_convergence=False, optimized_path=str(RELATIVE_ROOT.parent), initial_round=1,
            max_rounds=1, validation_rounds=1)
        with native.phase_scope("search", telemetry["phase_wall_seconds"]):
            asyncio.run(asyncio.wait_for(optimizer._optimize_graph(), timeout=args.timeout_seconds))
        issues = native.validate_aflow_workflow_round(repo / workflow, 2)
        if issues:
            raise RuntimeError(f"generated native graph failed validation: {issues}")
        rounds = (1, 2)
    else:
        graph = GraphUtils(str(RELATIVE_ROOT)).load_graph(1, str(workflow))
        with native.phase_scope("search", telemetry["phase_wall_seconds"]):
            native.evaluate_aflow_graph(evaluator=Evaluator(str(workflow / "round_1")), graph=graph,
                config=config, dataset_path=repo / "data/datasets/math_validate.jsonl",
                log_dir=repo / workflow / "round_1", is_test=False)
        rounds = (1,)
    audits = {f"round_{number}": audit_csv(repo / workflow / f"round_{number}") for number in rounds}
    write_json(args.run_dir / "telemetry.json", export_telemetry(telemetry))
    if any(audit["parser_valid"] < 1 for audit in audits.values()) or telemetry["calls"] < 2 or telemetry["failed_calls"]:
        raise RuntimeError("native AFlow diagnostic has no valid answer, insufficient calls, or failed model calls")
    return {"entrypoint": "native AFlow Evaluator/Workflow" if not args.aflow_mutation else "native AFlow Optimizer._optimize_graph",
            "mutation": args.aflow_mutation, "rounds": audits, "telemetry": export_telemetry(telemetry)}


def run_maas(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    import native_aime_formal as native
    import torch
    os.environ["METAGPT_PROJECT_ROOT"] = str(repo)
    telemetry = native.patch_maas_runtime(concurrency=2, max_tokens=args.max_tokens)
    from maas.configs.models_config import ModelsConfig
    from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
    from maas.ext.maas.scripts.optimizer import Optimizer
    config = EXPERIMENT_CONFIGS["MATH"]
    llm = ModelsConfig.default().get("qwen35_9b")
    if llm is None:
        raise RuntimeError("native MaAS local model configuration was not loaded")
    native.seed_everything(0)
    optimizer = Optimizer(dataset=config.dataset, question_type=config.question_type,
        opt_llm_config=llm, exec_llm_config=llm, operators=config.operators,
        optimized_path=str(RELATIVE_ROOT.parent), sample=1, round=1, batch_size=2,
        lr=0.01, is_textgrad=False)
    initial = {name: value.detach().clone() for name, value in optimizer.controller.named_parameters()}
    steps = []
    original_step = optimizer.optimizer.step

    def observed_step(*step_args: Any, **step_kwargs: Any) -> Any:
        parameters = [p for group in optimizer.optimizer.param_groups for p in group["params"]]
        assert all(bool(torch.isfinite(p).all()) for p in parameters)
        assert all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)
        before = [p.detach().clone() for p in parameters]
        grad = max((float(p.grad.detach().abs().max().item()) for p in parameters if p.grad is not None), default=0.0)
        result = original_step(*step_args, **step_kwargs)
        deltas = [float((p.detach() - old).abs().max().item()) for p, old in zip(parameters, before)]
        steps.append({"gradient_max_abs": grad, "parameter_max_abs_delta": max(deltas, default=0.0),
                      "changed_parameter_tensors": sum(delta > 0 for delta in deltas)})
        write_json(args.run_dir / "optimizer_steps.json", steps)
        return result

    optimizer.optimizer.step = observed_step
    with native.phase_scope("search", telemetry["phase_wall_seconds"]):
        optimizer.optimize("Graph")
    train = repo / RELATIVE_ROOT / "train/round_1"
    checkpoint = train / "MATH_controller_sample1.pth"
    if not checkpoint.is_file():
        raise RuntimeError("native MaAS did not save its trained controller checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(bool(torch.isfinite(value).all()) and value.shape == initial[name].shape for name, value in saved.items())
    deltas = [float((saved[name] - value.cpu()).abs().max().item()) for name, value in initial.items()]
    audit = audit_csv(train)
    write_json(args.run_dir / "telemetry.json", export_telemetry(telemetry))
    if not steps or max(deltas, default=0.0) <= 0 or telemetry["calls"] < 2 or telemetry["failed_calls"]:
        raise RuntimeError("native MaAS controller did not update, or model requests failed")
    return {"entrypoint": "native MaAS Optimizer.optimize('Graph')", "training_repetitions": 1,
            "optimizer_steps": steps, "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
            "checkpoint_max_abs_delta_from_initial": max(deltas), "predictions": audit,
            "telemetry": export_telemetry(telemetry)}


def run_external(method: str, source: Path, args: argparse.Namespace) -> dict[str, Any]:
    import native_external_methods as adapter
    from answer_protocol import score_aime
    adapter.OUTPUT_LIMIT = args.max_tokens
    adapter.CONCURRENCY = 2
    adapter.seed_everything(0)
    runtime = adapter.ModelRuntime(args.endpoint, method, f"diagnostic:{method}:{args.run_dir.name}")
    if method == "adas":
        smoke = adapter.smoke_native_source(method, source)
        native = importlib.import_module("search")

        def agent_query(self: Any, input_infos: list[Any], instruction: str, iteration_idx: int = -1) -> list[Any]:
            adapter.consume_candidate_agent_call_budget()
            system_prompt, user_prompt = self.generate_prompt(input_infos, instruction)
            content = runtime._messages_call([{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}], temperature=float(self.temperature),
                json_mode=True, phase="search")
            response = adapter._parse_json_object(content)
            # Match the formal adapter's declared field ordering, independent
            # of the ordering of keys generated by the local JSON model.
            return [native.Info(key, self.__repr__(), str(response.get(key, "")), iteration_idx) for key in self.output_fields]

        native.LLMAgentBase.query = agent_query
        architecture = native.get_init_archive()[0]
        forward = adapter.compile_candidate(architecture["code"], native.LLMAgentBase)

        async def one(task: dict[str, str]) -> dict[str, Any]:
            def execute() -> Any:
                adapter._candidate_budget.remaining = 24
                return forward(object(), native.Info("task", "User", task["problem"], -1))
            result = await asyncio.to_thread(execute)
            raw = str(result.content if hasattr(result, "content") else result)
            return {"id": task["id"], "raw_output": raw, **score_aime(raw, task["answer"])}
        architecture_info = {"native_seed_architecture": architecture["name"], "smoke": smoke}
    else:
        sys.path.insert(0, str(source))
        os.chdir(source)
        import GDesigner.prompt.gsm8k_prompt_set
        import GDesigner.agents.math_solver
        import GDesigner.agents.final_decision
        from GDesigner.llm.gpt_chat import GPTChat
        import GDesigner.graph.graph as graph_module
        from GDesigner.graph.graph import Graph
        from sentence_transformers import SentenceTransformer
        import torch
        embedding_dir = Path(os.environ.get("AIME_MINILM_PATH", ""))
        if not embedding_dir.is_dir():
            raise FileNotFoundError("AIME_MINILM_PATH must name the local MiniLM directory")
        encoder = SentenceTransformer(str(embedding_dir), device="cpu")
        graph_module.get_sentence_embedding = lambda text: encoder.encode(text, convert_to_numpy=True)
        adapter._register_gdesigner_prompt()

        async def controlled_agen(self: Any, messages: Any, max_tokens: int | None = None,
                                  temperature: float | None = None, num_comps: int | None = None) -> str:
            material = ([{"role": "user", "content": messages}] if isinstance(messages, str) else
                [{"role": str(item.get("role") if isinstance(item, dict) else item.role),
                  "content": str(item.get("content") if isinstance(item, dict) else item.content)} for item in messages])
            return await runtime.call_async(material, temperature=float(temperature if temperature is not None else 0.2), phase="search")

        GPTChat.agen = controlled_agen
        roles = ["Algebraic Decomposer", "Independent Solver", "Consistency Checker", "Solution Synthesizer"]

        async def one(task: dict[str, str]) -> dict[str, Any]:
            graph = Graph(domain="rpas_external", llm_name="GPTChat", agent_names=["MathSolver"] * 4,
                decision_method="FinalRefer", optimized_spatial=True, optimized_temporal=False,
                fixed_spatial_masks=[[int(i != j) for j in range(4)] for i in range(4)],
                fixed_temporal_masks=[[0] * 4 for _ in range(4)], node_kwargs=[{"role": role} for role in roles])
            assert graph.features.shape == (4, 384) and bool(torch.isfinite(graph.features).all())
            response, log_prob = await graph.arun({"task": task["problem"]}, num_rounds=1,
                max_tries=3, max_time=float(os.environ.get("AIME_EXTERNAL_REQUEST_TIMEOUT_S", "600")))
            if not bool(torch.isfinite(log_prob).all()):
                raise RuntimeError("native G-Designer produced a nonfinite topology log probability")
            raw = str(response[0] if response else "")
            return {"id": task["id"], "raw_output": raw, **score_aime(raw, task["answer"])}
        architecture_info = {"agent_nodes": 4, "decision_method": "FinalRefer", "topology_optimizer": "native GCN/MLP", "training": False}

    async def batch() -> list[dict[str, Any]]:
        return await asyncio.gather(*(one(task) for task in TASKS))
    predictions = asyncio.run(batch())
    write_json(args.run_dir / "telemetry.json", runtime.usage)
    if runtime.usage["totals"]["calls"] < 2 or runtime.usage["totals"]["failed_calls"] or not any(row["parser_valid"] for row in predictions):
        raise RuntimeError(f"native {method} diagnostic failed model calls or produced no parseable answers")
    return {**architecture_info, "predictions": predictions, "correct": sum(bool(row["correct"]) for row in predictions),
            "telemetry": runtime.usage, "context_guard": runtime.guard.summary()}


def worker(spec_path: Path) -> None:
    args = argparse.Namespace(**json.loads(spec_path.read_text(encoding="utf-8")))
    args.run_dir, source = Path(args.run_dir), Path(args.source)
    report = {"status": "running", "method": args.method, "synthetic_only": True,
        "sample_count": 2, "concurrency": 2, "max_tokens": args.max_tokens, "source": str(source),
        "source_info": args.source_info, "script_sha256": digest(Path(__file__)),
        "check_only": args.check_only, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_overrides": {key: value for key, value in os.environ.items()
            if key.startswith(("AIME_", "RPAS_")) and (key.endswith(("_PATH", "_TIMEOUT_S", "_RETRIES", "_WORKERS")))}}
    try:
        if args.method in {"aflow", "maas"}:
            repo, report["staging"] = stage_native(args.method, source, args.run_dir, args.endpoint, args.max_tokens)
            os.chdir(repo)
            sys.path.insert(0, str(repo))
            if args.check_only:
                for path in repo.rglob("*.py"):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                report["result"] = {"patched_source_syntax": "passed"}
            else:
                report["result"] = run_aflow(repo, args) if args.method == "aflow" else run_maas(repo, args)
        else:
            import native_external_methods as adapter
            report["result"] = adapter.smoke_native_source(args.method, source) if args.check_only else run_external(args.method, source, args)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.run_dir / "report.json", report)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-spec":
        worker(Path(sys.argv[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=NAMES, required=True)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "upstream")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--check-only", action="store_true", help="verify native structure/patches without model requests")
    parser.add_argument("--aflow-mutation", action="store_true", help="also generate/evaluate one native mutation")
    args = parser.parse_args()
    if not 64 <= args.max_tokens <= 6144 or not 60 <= args.timeout_seconds <= 7200:
        parser.error("max-tokens must be in 64..6144 and timeout-seconds in 60..7200")
    parsed = urlparse(args.endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.path.rstrip("/") != "/v1":
        parser.error("endpoint must be a local HTTP /v1 endpoint")
    args.endpoint = args.endpoint.rstrip("/")
    baseline = args.baseline_root.expanduser().resolve()
    source = baseline / NAMES[args.method] if (baseline / NAMES[args.method]).is_dir() else baseline
    if args.method in {"aflow", "maas"}:
        from native_aime_formal import verify_pinned_upstream, verify_served_model
        source_info = verify_pinned_upstream(args.method, source)
    else:
        from native_external_methods import verify_upstream_source, verify_model_endpoint
        source_info = verify_upstream_source(args.method, source)
    from answer_protocol import score_aime
    assert len(TASKS) == 2 and len({task["id"] for task in TASKS}) == 2
    assert all(score_aime(f"\\boxed{{{task['answer']}}}", task["answer"])["correct"] for task in TASKS)
    run_dir = safe_run_dir(args.run_dir, source)
    if not args.check_only:
        (verify_served_model if args.method in {"aflow", "maas"} else verify_model_endpoint)(args.endpoint)
    run_dir.mkdir(parents=True)
    spec = {**vars(args), "baseline_root": str(baseline), "run_dir": str(run_dir), "source": str(source), "source_info": source_info}
    write_json(run_dir / "worker_spec.json", spec)
    environment = os.environ.copy()
    environment["AIME_RUN_ID"] = f"diagnostic:{args.method}:{run_dir.name}"
    environment["MASBENCH_PHASE"] = "search"
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(run_dir / "worker_spec.json")],
            env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout_seconds)
        except BaseException as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            write_json(run_dir / "report.json", {"status": "failed", "synthetic_only": True,
                "method": args.method, "error": f"{type(exc).__name__}: diagnostic worker exceeded time limit or was interrupted"})
            raise
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"status": "failed"}
    print(json.dumps({"status": report["status"], "method": args.method, "report": str(report_path)}, ensure_ascii=False))
    return 0 if code == 0 and report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
