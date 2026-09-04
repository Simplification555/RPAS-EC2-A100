"""Native G-Designer adapter for the EC-2 MMLU topology comparison."""

from __future__ import annotations

import asyncio
import copy
import contextvars
import importlib
import os
import sys
import time
import types
from pathlib import Path

from external_comparison.adapters.native_common import (
    call_record,
    env_path,
    git_commit,
    require_valid_answer_rate,
    stratified_mmlu,
    write_native_result,
)
from external_comparison.runners.mmlu import parse_mmlu_choice


def _enforce_authorized_gpu() -> None:
    # The controlled comparison is authorized only on the user's two cards.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "4")
    if visible not in {"4", "5"}:
        raise RuntimeError(f"experiment requires CUDA_VISIBLE_DEVICES=4 or 5, got {visible!r}")


def _root() -> Path:
    return env_path("RPAS_GDESIGNER_ROOT", "/path/to/external_baselines/GDesigner")


def _patch_local_embedding_model() -> str:
    """Resolve G-Designer's profile encoder to the staged local MiniLM copy."""
    model_path = os.environ.get("RPAS_MAAS_EMBEDDING_MODEL", "").strip()
    if not model_path:
        return "sentence-transformers/all-MiniLM-L6-v2"

    local_path = Path(model_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"embedding model directory not found: {local_path}")

    import sentence_transformers

    original_constructor = sentence_transformers.SentenceTransformer
    cached_local_model = None

    def local_constructor(model_name_or_path, *args, **kwargs):
        nonlocal cached_local_model
        if model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2":
            if cached_local_model is not None:
                return cached_local_model
            model_name_or_path = str(local_path)
            kwargs.setdefault("device", "cpu")
            cached_local_model = original_constructor(model_name_or_path, *args, **kwargs)
            return cached_local_model
        return original_constructor(model_name_or_path, *args, **kwargs)

    sentence_transformers.SentenceTransformer = local_constructor
    return str(local_path)


class _Dataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    @staticmethod
    def record_to_input(record):
        choices = record["choices"]
        return {"task": f"Question: {record['question']}\nOption A: {choices[0]}\nOption B: {choices[1]}\nOption C: {choices[2]}\nOption D: {choices[3]}"}

    @staticmethod
    def record_to_target_answer(record):
        return record["answer"]

    @staticmethod
    def postprocess_answer(answer):
        text = answer[0] if isinstance(answer, list) and answer else answer
        return parse_mmlu_choice(text) if isinstance(text, str) else ""


async def _run(rows, output_dir: Path, seed: int) -> None:
    _enforce_authorized_gpu()
    root = _root()
    if not root.exists():
        raise FileNotFoundError(f"G-Designer repository not found: {root}")
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    embedding_model = _patch_local_embedding_model()
    # The official code imports its local ``datasets`` package as a top-level
    # module, which otherwise collides with the installed HF package. Inject
    # it after sentence-transformers/transformers have initialized.
    local_datasets = types.ModuleType("datasets")
    local_datasets.__path__ = [str(root / "datasets")]
    local_datasets.__version__ = "0.0-local-gdesigner"
    local_datasets.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None, is_package=True)
    sys.modules["datasets"] = local_datasets
    from openai import AsyncOpenAI
    import GDesigner.llm.gpt_chat as gpt_chat
    from GDesigner.graph.graph import Graph
    from GDesigner.utils.globals import CompletionTokens, PromptTokens
    torch.manual_seed(seed)
    client = AsyncOpenAI(api_key=os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"), base_url=os.environ.get("RPAS_EXTERNAL_API_BASE", "http://127.0.0.1:29500/v1"))
    usage: list[dict] = []
    current_example_id = contextvars.ContextVar("gdesigner_mmlu_example_id", default="")
    max_tokens = int(os.environ.get("RPAS_MMLU_MAX_TOKENS", "256"))
    if max_tokens != 256:
        raise ValueError(f"EC-2 requires RPAS_MMLU_MAX_TOKENS=256, got {max_tokens}")

    async def achat(model, messages):
        normalized = []
        for item in messages:
            content = item.get("content", "")
            if item.get("role", "user") == "system":
                content += "\nBe concise and finish within 120 words; always complete the requested final answer."
            normalized.append({"role": item.get("role", "user"), "content": content})
        started = time.perf_counter()
        response = await client.chat.completions.create(
            model=os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
            messages=normalized,
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        elapsed = (time.perf_counter() - started) * 1000
        token_usage = response.usage
        record = {
            "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
            "prompt_tokens": getattr(token_usage, "prompt_tokens", 0),
            "completion_tokens": getattr(token_usage, "completion_tokens", 0),
            "total_tokens": getattr(token_usage, "total_tokens", 0),
            "latency_ms": elapsed,
            "example_id": current_example_id.get(),
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
        }
        usage.append(record)
        PromptTokens.instance().value += record["prompt_tokens"]
        CompletionTokens.instance().value += record["completion_tokens"]
        return response.choices[0].message.content or ""

    gpt_chat.achat = achat
    agent_names = ["AnalyzeAgent", "AnalyzeAgent", "AnalyzeAgent"]
    masks = [[1 if i != j else 0 for i in range(3)] for j in range(3)]
    graph = Graph(domain="mmlu", llm_name=os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"), agent_names=agent_names, decision_method="FinalRefer", optimized_spatial=True, fixed_spatial_masks=masks, fixed_temporal_masks=[[1] * 3 for _ in range(3)])
    search_rows = _Dataset(rows["search"])
    if os.environ.get("RPAS_GDESIGNER_TRAIN", "0") == "1":
        trainer = importlib.import_module("experiments.train_mmlu")
        await trainer.train(graph=graph, dataset=search_rows, num_iters=int(os.environ.get("RPAS_GDESIGNER_ITERS", "10")), num_rounds=1, lr=0.1, batch_size=4)
    results = []
    sample_limit = int(os.environ.get("RPAS_NATIVE_SAMPLE_LIMIT", "0"))
    test_rows = rows["test"][:sample_limit] if sample_limit > 0 else rows["test"]
    concurrency = max(1, int(os.environ.get("RPAS_GDESIGNER_EVAL_CONCURRENCY", "8")))
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_row(row):
        async with semaphore:
            input_dict = _Dataset.record_to_input(row)
            token = current_example_id.set(row["id"])
            started = time.perf_counter()
            try:
                # Graph.arun mutates node outputs, connections and decision
                # state. Each example therefore needs an isolated graph.
                realized_graph = copy.deepcopy(graph)
                realized_graph.gcn = graph.gcn
                realized_graph.mlp = graph.mlp
                answers, _ = await realized_graph.arun(input_dict, num_rounds=1)
            finally:
                current_example_id.reset(token)
            elapsed = (time.perf_counter() - started) * 1000
            prediction = _Dataset.postprocess_answer(answers)
            return {"example_id": row["id"], "subject": row["subject"], "prediction": prediction, "answer": row["answer"], "correct": prediction == row["answer"], "latency_ms": elapsed}

    results = []
    # Submit bounded batches so hundreds of examples do not exhaust process FDs.
    for offset in range(0, len(test_rows), concurrency):
        batch = test_rows[offset : offset + concurrency]
        results.extend(await asyncio.gather(*(evaluate_row(row) for row in batch)))
    valid_answer_rate = require_valid_answer_rate(
        results, context=f"G-Designer MMLU seed {seed} test"
    )
    calls = [
        call_record(
            f"mmlu-gdesigner-seed-{seed}",
            "gdesigner",
            "mmlu",
            "test",
            str(record.get("example_id", "")),
            index,
            record,
        )
        for index, record in enumerate(usage)
    ]
    manifest = {
        "run_id": f"mmlu-gdesigner-seed-{seed}",
        "method": "gdesigner",
        "dataset": "mmlu",
        "seed": seed,
        "implementation_status": "controlled_official_graph_executor",
        "native_search": "official_gcn_fixed_topology_execution",
        "official_repo": "external_baselines/GDesigner",
        "official_commit": git_commit(root),
        "search_calls": 0,
        "search_tokens": 0,
        "search_candidates": 0,
        "search_scope": "no separately instrumented topology-search phase",
        "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "embedding_model": "local_sentence_transformers_all-MiniLM-L6-v2",
        "communication_telemetry": "calls_from_official_graph_messages_not_reconstructed",
        "search_examples": len(rows["search"]),
        "test_examples": len(test_rows),
        "agent_count": 3,
        "communication_rounds": 1,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "thinking_disabled": True,
        "answer_parser": "strict_choice_a_b_c_d",
        "valid_answer_rate": valid_answer_rate,
        "formal_result": False,
        "formal_result_reason": "controlled subset and repository formal gates are incomplete",
    }
    write_native_result(output_dir, manifest, results, calls)


def run_mmlu(args) -> None:
    rows = {"search": stratified_mmlu(args.data_dir, "dev", 5, 2026), "test": stratified_mmlu(args.data_dir, "test", 10, 2026)}
    asyncio.run(_run(rows, Path(args.output_dir).expanduser().resolve() / "gdesigner" / f"seed_{args.seed}", args.seed))
