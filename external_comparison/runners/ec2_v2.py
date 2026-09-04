"""EC-2 v2: an auditable, communication-topology-only comparison.

This runner deliberately does not consume legacy ``native_mmlu`` artifacts.
It keeps the official G-Designer graph and its published MMLU training loop,
while executing RPAS-Comm fixed six-agent candidates through that same graph
executor.  That makes the role pool, final judge, model endpoint, decoding
budget, and message semantics common across the two search methods.

The official G-Designer implementation has no compression primitive.  v2
therefore freezes messages to verbatim forwarding for *all* topology methods;
compression is explicitly outside this comparison rather than being granted to
only RPAS.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import copy
import hashlib
import importlib.machinery
import importlib.util
import itertools
import json
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

from external_comparison.adapters.native_common import call_record, git_commit, write_native_result
from external_comparison.runners.mmlu import (
    MMLUExample,
    build_mmlu_manifest,
    load_mmlu_split,
    parse_mmlu_choice,
)

EC2_V2_PROTOCOL = "ec2-mmlu-communication-v2"
BACKBONE = "Qwen/Qwen3.5-9B"
MAX_TOKENS = 256
AGENT_COUNT = 6
ROUNDS = 1
GDESIGNER_COMMIT = "a6efcfa"
ROLES = (
    "Knowlegable Expert",  # Preserve the spelling used by the official prompt set.
    "Critic",
    "Mathematician",
    "Psychologist",
    "Historian",
    "Doctor",
)
TOPOLOGIES = ("full_connected", "chain", "star", "layered")
GDESIGNER_TRAINING_QUERY_BUDGET = 40


def require_authorized_gpu() -> str:
    """Reject an accidental run on any GPU other than the two authorized cards."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"4", "5"}:
        raise RuntimeError(
            "EC-2 v2 requires one explicit physical GPU: CUDA_VISIBLE_DEVICES=4 or CUDA_VISIBLE_DEVICES=5; "
            f"got {visible!r}"
        )
    return visible


def topology_mask(name: str, *, agent_count: int = AGENT_COUNT) -> list[list[int]]:
    """Return a directed acyclic topology over the fixed ordered role pool."""

    if agent_count != AGENT_COUNT:
        raise ValueError(f"EC-2 v2 requires exactly {AGENT_COUNT} agents, got {agent_count}")
    mask = [[0 for _ in range(agent_count)] for _ in range(agent_count)]
    if name == "full_connected":
        # G-Designer prevents cycles at execution time, yielding this ordered DAG.
        for source in range(agent_count):
            for target in range(source + 1, agent_count):
                mask[source][target] = 1
    elif name == "chain":
        for source in range(agent_count - 1):
            mask[source][source + 1] = 1
    elif name == "star":
        for target in range(1, agent_count):
            mask[0][target] = 1
    elif name == "layered":
        for target in (1, 2, 3):
            mask[0][target] = 1
        for source in (1, 2, 3):
            mask[source][4] = 1
        mask[4][5] = 1
    else:
        raise ValueError(f"unsupported EC-2 v2 topology: {name}")
    return mask


def communication_candidate(name: str) -> dict[str, Any]:
    """A topology-only RPAS candidate. No model, role, or decode mutation is legal."""

    candidate = {
        "name": f"rpas_comm_{name}",
        "topology": name,
        "agent_count": AGENT_COUNT,
        "roles": list(ROLES),
        "rounds": ROUNDS,
        "message_policy": "verbatim_official_gdesigner",
        "compression": "not_available_in_official_gdesigner",
        "mask": topology_mask(name),
    }
    canonical = json.dumps(candidate, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    candidate["id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return candidate


def assert_v2_candidate(candidate: dict[str, Any]) -> None:
    """Enforce the common execution space before an architecture is run."""

    forbidden = {"single", "self_consistency", "solver_verifier", "debate", "dag_decompose"}
    if candidate.get("topology") in forbidden:
        raise ValueError("EC-2 v2 RPAS-Comm may not leave the fixed six-agent communication space")
    if candidate.get("agent_count") != AGENT_COUNT or tuple(candidate.get("roles", ())) != ROLES:
        raise ValueError("EC-2 v2 candidate changed the fixed six-agent role pool")
    if candidate.get("rounds") != ROUNDS:
        raise ValueError("EC-2 v2 candidate changed the fixed communication rounds")
    if candidate.get("message_policy") != "verbatim_official_gdesigner":
        raise ValueError("EC-2 v2 candidate changed the shared message policy")
    if candidate.get("compression") != "not_available_in_official_gdesigner":
        raise ValueError("EC-2 v2 cannot give RPAS an unsupported compression capability")
    if candidate.get("mask") != topology_mask(str(candidate.get("topology"))):
        raise ValueError("EC-2 v2 candidate mask does not match its declared topology")


def validate_v2_manifest(manifest: dict[str, Any]) -> None:
    """Machine-check the protocol fields required before aggregation."""

    required = {
        "protocol_version": EC2_V2_PROTOCOL,
        "backbone": BACKBONE,
        "communication_rounds": ROUNDS,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "final_aggregator": "GDesigner.FinalRefer",
        "message_policy": "verbatim_official_gdesigner",
        "split_protocol": "dev_search__val_select__test_heldout",
    }
    # Single Agent is an explicitly labelled reference, not a member of the
    # six-worker topology comparison. All other protocol fields stay shared.
    if manifest.get("method") != "single_agent":
        required["agent_count"] = AGENT_COUNT
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"EC-2 v2 manifest has invalid {key}: {manifest.get(key)!r}, expected {value!r}")
    if manifest.get("method") not in {"single_agent", "full_connected", "chain", "gdesigner", "rpas_comm"}:
        raise ValueError(f"EC-2 v2 does not recognize method {manifest.get('method')!r}")
    if manifest.get("method") == "single_agent":
        if manifest.get("agent_count") != 1 or manifest.get("roles") != [ROLES[0]]:
            raise ValueError("EC-2 v2 single-agent reference must declare exactly one official-role worker")
    elif manifest.get("agent_count") != AGENT_COUNT or manifest.get("roles") != list(ROLES):
        raise ValueError("EC-2 v2 multi-agent run did not use the fixed official six-role pool")
    if manifest.get("method") == "gdesigner":
        if manifest.get("gdesigner_training", {}).get("iterations") != 10:
            raise ValueError("G-Designer v2 requires the official 10-iteration training loop")
        if not manifest.get("gdesigner_training", {}).get("initial_gcn_sha256"):
            raise ValueError("G-Designer v2 is missing its initial GCN checksum")
        if not manifest.get("gdesigner_training", {}).get("trained_gcn_sha256"):
            raise ValueError("G-Designer v2 is missing its trained GCN checksum")
        if (
            manifest["gdesigner_training"]["initial_gcn_sha256"]
            == manifest["gdesigner_training"]["trained_gcn_sha256"]
        ):
            raise ValueError("G-Designer v2 training did not change the GCN checkpoint")
        if int(manifest.get("search_calls", 0)) <= 0 or int(manifest.get("search_tokens", 0)) <= 0:
            raise ValueError("G-Designer v2 must record non-zero training/selection search telemetry")
    if manifest.get("method") == "rpas_comm":
        evidence = manifest.get("rpas_reflection", {})
        if int(evidence.get("reflection_calls", 0)) <= 0:
            raise ValueError("RPAS-Comm v2 requires LLM reflection calls")
        if int(evidence.get("new_candidates", 0)) <= 0:
            raise ValueError("RPAS-Comm v2 requires new communication candidates")
        if int(evidence.get("mutation_logs", 0)) <= 0:
            raise ValueError("RPAS-Comm v2 requires mutation logs")
        if int(evidence.get("rule_fallbacks", 0)) != 0:
            raise ValueError("RPAS-Comm v2 forbids rule-reflection fallback")
        if int(manifest.get("search_calls", 0)) <= 0 or int(manifest.get("search_tokens", 0)) <= 0:
            raise ValueError("RPAS-Comm v2 must account for search and reflection calls")


class _RowsDataset:
    """The minimal official train_mmlu dataset interface over frozen harness rows."""

    def __init__(self, rows: list[MMLUExample]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MMLUExample:
        return self.rows[index]

    @staticmethod
    def record_to_input(row: MMLUExample) -> dict[str, str]:
        choices = "\n".join(
            f"Option {label}: {choice}" for label, choice in zip("ABCD", row.choices, strict=True)
        )
        return {"task": f"Question: {row.question}\n{choices}"}

    @staticmethod
    def record_to_target_answer(row: MMLUExample) -> str:
        return row.answer

    @staticmethod
    def postprocess_answer(answer: Any) -> str:
        text = answer[0] if isinstance(answer, list) and answer else answer
        return parse_mmlu_choice(text) if isinstance(text, str) else ""


class OfficialGDesignerRuntime:
    """Thin instrumentation layer around G-Designer at the pinned upstream commit."""

    def __init__(self, root: Path, *, seed: int) -> None:
        self.root = root
        self.seed = seed
        self.usage: list[dict[str, Any]] = []
        self.phase = contextvars.ContextVar("ec2_v2_phase", default="test")
        self.example_id = contextvars.ContextVar("ec2_v2_example_id", default="")
        self.candidate_id = contextvars.ContextVar("ec2_v2_candidate_id", default="")
        self._tokenizer: Any | None = None
        self._load_official_modules()

    def _load_official_modules(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"G-Designer checkout is missing: {self.root}")
        commit = git_commit(self.root)
        if not commit.startswith(GDESIGNER_COMMIT):
            raise RuntimeError(
                f"G-Designer must be pinned to {GDESIGNER_COMMIT}; found {commit or 'unknown'} at {self.root}"
            )
        tokenizer_path = os.environ.get("RPAS_TOKENIZER_PATH", os.environ.get("RPAS_MODEL_PATH", "")).strip()
        if not tokenizer_path:
            raise RuntimeError("EC-2 v2 requires RPAS_TOKENIZER_PATH or RPAS_MODEL_PATH for tokenizer-consistent telemetry")
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        # The upstream repository imports its local datasets package at top level.
        local_datasets = types.ModuleType("datasets")
        local_datasets.__path__ = [str(self.root / "datasets")]
        local_datasets.__version__ = "0.0-ec2-v2-gdesigner"
        local_datasets.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None, is_package=True)
        sys.modules["datasets"] = local_datasets

        # This fixed encoder is part of G-Designer's native topology feature
        # construction, not an executor LLM.  Prefer a method-specific name;
        # retain the old variable only so existing local setup scripts remain
        # usable until their next invocation.
        embedding_path = os.environ.get(
            "RPAS_GDESIGNER_EMBEDDING_MODEL",
            os.environ.get("RPAS_MAAS_EMBEDDING_MODEL", ""),
        ).strip()
        if embedding_path:
            local_embedding = Path(embedding_path).expanduser().resolve()
            if not local_embedding.is_dir():
                raise FileNotFoundError(f"local MiniLM directory not found: {local_embedding}")
            import sentence_transformers

            constructor = sentence_transformers.SentenceTransformer
            cached_model = None

            def local_constructor(model_name_or_path, *args, **kwargs):
                nonlocal cached_model
                if model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2":
                    if cached_model is None:
                        kwargs.setdefault("device", "cpu")
                        cached_model = constructor(str(local_embedding), *args, **kwargs)
                    return cached_model
                return constructor(model_name_or_path, *args, **kwargs)

            sentence_transformers.SentenceTransformer = local_constructor

        import GDesigner.llm.gpt_chat as gpt_chat
        import GDesigner.prompt.mmlu_prompt_set as mmlu_prompt_set
        from GDesigner.graph.graph import Graph
        from openai import AsyncOpenAI

        self.Graph = Graph
        self.gpt_chat = gpt_chat
        self.mmlu_prompt_set = mmlu_prompt_set
        self.client = AsyncOpenAI(
            api_key=os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"),
            base_url=os.environ.get("RPAS_EXTERNAL_API_BASE", "http://127.0.0.1:29500/v1"),
        )

        async def achat(model: str, messages: list[dict[str, Any]]):
            normalized = []
            for message in messages:
                content = str(message.get("content", ""))
                normalized.append({"role": message.get("role", "user"), "content": content})
            started = time.perf_counter()
            response = await self.client.chat.completions.create(
                model=BACKBONE,
                messages=normalized,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            latency_ms = (time.perf_counter() - started) * 1000
            usage = response.usage
            self.usage.append(
                {
                    "phase": self.phase.get(),
                    "example_id": self.example_id.get(),
                    "candidate_id": self.candidate_id.get(),
                    "model": BACKBONE,
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    "latency_ms": latency_ms,
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                }
            )
            return response.choices[0].message.content or ""

        self.gpt_chat.achat = achat

    def message_tokens(self, content: Any) -> int:
        if not content:
            return 0
        assert self._tokenizer is not None
        return len(self._tokenizer.encode(str(content), add_special_tokens=False))

    def make_graph(self, *, topology: str | None, optimized_spatial: bool) -> Any:
        # The upstream prompt set uses a module-level cycle. Reset it so every
        # graph, candidate, and seed receives exactly the same six roles.
        self.mmlu_prompt_set.roles = itertools.cycle(ROLES)
        options: dict[str, Any] = {}
        if topology is not None:
            options["fixed_spatial_masks"] = topology_mask(topology)
        return self.Graph(
            domain="mmlu",
            llm_name=BACKBONE,
            agent_names=["AnalyzeAgent"] * AGENT_COUNT,
            decision_method="FinalRefer",
            optimized_spatial=optimized_spatial,
            fixed_temporal_masks=[[0] * AGENT_COUNT for _ in range(AGENT_COUNT)],
            **options,
        )

    @staticmethod
    def gcn_checksum(graph: Any) -> str:
        """Hash tensor values rather than pickle serialization metadata."""

        digest = hashlib.sha256()
        for name, value in sorted(graph.gcn.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    async def evaluate(
        self,
        graph: Any,
        rows: list[MMLUExample],
        *,
        split: str,
        candidate_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Evaluate immutable graph parameters and reconstruct official edge traffic."""

        self.phase.set(split)
        outputs: list[dict[str, Any]] = []
        communication = {"active_edges": 0, "messages": 0, "inter_agent_tokens": 0, "judge_input_tokens": 0}
        for row in rows:
            token = self.example_id.set(row.example_id)
            candidate_token = self.candidate_id.set(candidate_id)
            started = time.perf_counter()
            try:
                realized = copy.deepcopy(graph)
                realized.gcn = graph.gcn
                realized.mlp = graph.mlp
                answers, _ = await realized.arun(_RowsDataset.record_to_input(row), num_rounds=ROUNDS)
            finally:
                self.example_id.reset(token)
                self.candidate_id.reset(candidate_token)
            output = answers[0] if isinstance(answers, list) and answers else answers
            prediction = parse_mmlu_choice(output if isinstance(output, str) else "")
            adjacency = realized.spatial_adj_matrix
            node_values = list(realized.nodes.values())
            active_edges = 0
            message_tokens = 0
            for source, source_node in enumerate(node_values):
                source_output = source_node.outputs[-1] if source_node.outputs else ""
                source_tokens = self.message_tokens(source_output)
                for target in range(len(node_values)):
                    if int(adjacency[source, target]):
                        active_edges += 1
                        message_tokens += source_tokens
            judge_tokens = sum(
                self.message_tokens(node.outputs[-1]) if node.outputs else 0
                for node in node_values
            )
            communication["active_edges"] += active_edges
            communication["messages"] += active_edges
            communication["inter_agent_tokens"] += message_tokens
            communication["judge_input_tokens"] += judge_tokens
            outputs.append(
                {
                    "example_id": row.example_id,
                    "subject": row.subject,
                    "prediction": prediction,
                    "answer": row.answer,
                    "correct": prediction == row.answer,
                    "active_edges": active_edges,
                    "messages": active_edges,
                    "inter_agent_tokens": message_tokens,
                    "judge_input_tokens": judge_tokens,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
        return outputs, communication

    def calls(self, *, run_id: str, method: str) -> list[dict[str, Any]]:
        return [
            call_record(
                run_id,
                method,
                "mmlu",
                str(record["phase"]),
                str(record.get("candidate_id", "")),
                index,
                record,
            )
            for index, record in enumerate(self.usage)
        ]


def _candidate_row(candidate: dict[str, Any], rows: list[dict[str, Any]], communication: dict[str, int]) -> dict[str, Any]:
    failures = []
    for row in rows:
        if not row["correct"] and len(failures) < 3:
            failures.append(
                {
                    "id": row["example_id"],
                    "gold_answer": row["answer"],
                    "prediction": row["prediction"],
                    "input": "MMLU item retained in frozen telemetry; raw item is not duplicated in public artifacts.",
                    "final_output_excerpt": "",
                    "trace": {"inter_agent_tokens": row["inter_agent_tokens"], "calls": None},
                }
            )
    count = len(rows)
    accuracy = sum(bool(row["correct"]) for row in rows) / count if count else 0.0
    return {
        "candidate_id": candidate["id"],
        "candidate_name": candidate["name"],
        "candidate": candidate,
        "topology": candidate["topology"],
        "score": accuracy,
        "correct": sum(bool(row["correct"]) for row in rows),
        "num_examples": count,
        "valid": all(bool(row["prediction"]) for row in rows),
        "avg_total_tokens": 0.0,
        "avg_calls": 0.0,
        "avg_cross_center_tokens": 0.0,
        "avg_inter_agent_tokens": communication["inter_agent_tokens"] / count if count else 0.0,
        "failure_examples": failures,
    }


def _reflection_context(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Load the established phase-2 LLM reflector without reimplementing it."""

    from experiments.phase2_wan_agent_search import load_models, load_network_profiles

    config_path = Path(os.environ.get("RPAS_MODEL_CONFIG", repo_root / "experiments" / "phase2_mmlu_qwen35_9b.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE")
    if endpoint:
        for model in config.get("models", {}).values():
            model["api_base"] = endpoint
    # Only topology names are legal proposals.  Other mutation dimensions are
    # frozen to make the method comparison about communication topology.
    config["allowed_topologies"] = list(TOPOLOGIES)
    config["search"] = {
        "model_pool": [config["defaults"]["local_model"]],
        "site_pool": [config["defaults"]["local_site"]],
        "compression_pool": [],
        "max_tokens_pool": {},
    }
    config["reflection"] = {
        **config.get("reflection", {}),
        "model": config["defaults"]["local_model"],
        "allow_rule_fallback": False,
        "children": 1,
        "temperature": 0.0,
    }
    return config, load_models(config["models"]), load_network_profiles(config["network_profiles"])["lan_homogeneous"]


def _llm_topology_mutation(
    parent: dict[str, Any],
    *,
    config: dict[str, Any],
    models: dict[str, Any],
    profile: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from experiments.phase2_wan_agent_search import build_reflection_plan

    remaining_topologies = [
        topology
        for topology in TOPOLOGIES
        if topology != parent["candidate"]["topology"]
        and communication_candidate(topology)["id"] not in set(parent.get("excluded_candidate_ids", []))
    ]
    if not remaining_topologies:
        raise RuntimeError("RPAS-Comm exhausted the finite topology space")
    reflection_config = copy.deepcopy(config)
    reflection_config["allowed_topologies"] = remaining_topologies
    plan = build_reflection_plan(
        row=parent,
        config=reflection_config,
        models=models,
        profile=profile,
        reflection_mode="llm",
        reflection_model=config["defaults"]["local_model"],
        reflection_max_tokens=MAX_TOKENS,
        max_proposals=1,
    )
    if plan.get("mode") != "llm":
        raise RuntimeError(f"EC-2 v2 rejects non-LLM reflection: {plan.get('mode')}")
    for mutation in plan.get("mutations", []):
        if mutation.get("type") == "topology" and mutation.get("value") in remaining_topologies:
            if mutation["value"] == parent["candidate"]["topology"]:
                continue
            child = communication_candidate(str(mutation["value"]))
            child["parent_id"] = parent["candidate_id"]
            child["mutation"] = "topology"
            child["applied_mutation"] = mutation
            assert_v2_candidate(child)
            return child, plan
    raise RuntimeError("LLM reflection produced no legal non-noop EC-2 communication-topology mutation")


def _reflection_call_records(
    plan: dict[str, Any],
    *,
    run_id: str,
    method: str,
    candidate_id: str,
    start_index: int,
) -> list[dict[str, Any]]:
    """Materialize phase-2 reflector telemetry in the shared native call schema."""

    records = []
    for index, trace in enumerate(plan.get("call_traces", []), start_index):
        payload = dict(trace)
        payload["agent"] = "reflector"
        records.append(call_record(run_id, method, "mmlu", "search", candidate_id, index, payload))
    return records


def _rpas_search_assignments(
    rows: list[MMLUExample],
    *,
    candidate_count: int,
    seed: int,
) -> list[list[MMLUExample]]:
    """Allocate exactly the official 40-query train budget across RPAS candidates."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if GDESIGNER_TRAINING_QUERY_BUDGET % candidate_count:
        raise ValueError("RPAS candidate count must divide the 40-query G-Designer training budget")
    per_candidate = GDESIGNER_TRAINING_QUERY_BUDGET // candidate_count
    if len(rows) < per_candidate:
        raise ValueError(f"D_search has {len(rows)} rows, need at least {per_candidate}")
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    # Candidate evaluations may reuse D_search items. This is deliberate: it
    # fixes total query executions to the official G-Designer train budget.
    return [
        [ordered[(candidate_index * per_candidate + offset) % len(ordered)] for offset in range(per_candidate)]
        for candidate_index in range(candidate_count)
    ]


def _base_manifest(method: str, seed: int, split_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": f"ec2-v2-{method}-seed-{seed}",
        "method": method,
        "dataset": "mmlu",
        "seed": seed,
        "protocol_version": EC2_V2_PROTOCOL,
        "implementation_status": "formal_candidate_pending_three_seed_aggregate",
        "formal_result": False,
        "formal_result_reason": "A single seed is never a paper result; aggregate requires all three protocol-valid seeds.",
        "backbone": BACKBONE,
        "agent_count": AGENT_COUNT,
        "roles": list(ROLES),
        "communication_rounds": ROUNDS,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "thinking_disabled": True,
        "answer_parser": "strict_choice_a_b_c_d",
        "final_aggregator": "GDesigner.FinalRefer",
        "message_policy": "verbatim_official_gdesigner",
        "compression_policy": "not_available_in_official_gdesigner__disabled_for_all_methods",
        "split_protocol": "dev_search__val_select__test_heldout",
        "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        "search_examples": split_manifest["search"]["count"],
        "select_examples": split_manifest["select"]["count"],
        "test_examples": split_manifest["test"]["count"],
        "search_budget": {
            "unit": "MMLU graph executions before D_select",
            "gdesigner_official_train_budget": GDESIGNER_TRAINING_QUERY_BUDGET,
            "rpas_comm_candidate_evaluation_budget": GDESIGNER_TRAINING_QUERY_BUDGET,
            "budget_match_scope": "D_search graph executions only; D_select is separately accounted.",
        },
        "communication_definition": {
            "active_edges": "realized directed edges among the six worker agents; excludes final judge inputs",
            "messages": "one verbatim worker-to-worker forwarding event per realized directed edge",
            "inter_agent_tokens": "Qwen tokenizer count of forwarded worker outputs",
            "judge_input_tokens": "separate Qwen tokenizer count of worker output consumed by FinalRefer",
        },
    }


async def _run_fixed(runtime: OfficialGDesignerRuntime, method: str, rows: dict[str, list[MMLUExample]], seed: int) -> None:
    topology = {"full_connected": "full_connected", "chain": "chain"}[method]
    graph = runtime.make_graph(topology=topology, optimized_spatial=False)
    result_rows, communication = await runtime.evaluate(graph, rows["test"], split="test", candidate_id=method)
    manifest = _base_manifest(method, seed, rows["manifest"])
    manifest.update({"native_search": "none_fixed_reference", "fixed_topology": topology, "search_calls": 0, "search_tokens": 0, "test_communication": communication})
    validate_v2_manifest(manifest)
    write_native_result(Path(rows["output_dir"]) / method / f"seed_{seed}", manifest, result_rows, runtime.calls(run_id=manifest["run_id"], method=method), {"topology": topology})


async def _run_single(runtime: OfficialGDesignerRuntime, rows: dict[str, list[MMLUExample]], seed: int) -> None:
    runtime.mmlu_prompt_set.roles = itertools.cycle(ROLES)
    graph = runtime.Graph(
        domain="mmlu", llm_name=BACKBONE, agent_names=["AnalyzeAgent"], decision_method="FinalRefer",
        optimized_spatial=False, fixed_spatial_masks=[[0]], fixed_temporal_masks=[[0]],
    )
    result_rows, communication = await runtime.evaluate(graph, rows["test"], split="test", candidate_id="single_agent")
    manifest = _base_manifest("single_agent", seed, rows["manifest"])
    manifest["agent_count"] = 1
    manifest["roles"] = [ROLES[0]]
    manifest.update({"native_search": "none_reference_only", "reference_only": True, "fixed_topology": "single_agent", "search_calls": 0, "search_tokens": 0, "test_communication": communication})
    # Single Agent is intentionally outside the six-agent competitor set.
    validate_v2_manifest(manifest)
    write_native_result(Path(rows["output_dir"]) / "single_agent" / f"seed_{seed}", manifest, result_rows, runtime.calls(run_id=manifest["run_id"], method="single_agent"), {"topology": "single_agent"})


async def _run_gdesigner(runtime: OfficialGDesignerRuntime, rows: dict[str, list[MMLUExample]], seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Preserve the official graph's full directed candidate-edge mask. Unlike
    # the fixed references, this is query-conditioned topology learning.
    graph = runtime.make_graph(topology=None, optimized_spatial=True)
    initial_sha = runtime.gcn_checksum(graph)
    token = runtime.phase.set("search")
    try:
        train_path = runtime.root / "experiments" / "train_mmlu.py"
        spec = importlib.util.spec_from_file_location("rpas_ec2_v2_gdesigner_train_mmlu", train_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pinned G-Designer train loop: {train_path}")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        await trainer.train(graph=graph, dataset=_RowsDataset(rows["search"]), num_iters=10, num_rounds=ROUNDS, lr=0.1, batch_size=4)
    finally:
        runtime.phase.reset(token)
    trained_sha = runtime.gcn_checksum(graph)
    if initial_sha == trained_sha:
        raise RuntimeError("official G-Designer training did not change the GCN checkpoint")
    # D_select is an audit-only validation pass: the released official loop
    # trains for exactly ten iterations and has no checkpoint-selection step.
    _, select_communication = await runtime.evaluate(graph, rows["select"], split="select", candidate_id="gdesigner_trained")
    result_rows, test_communication = await runtime.evaluate(graph, rows["test"], split="test", candidate_id="gdesigner_trained")
    calls = runtime.calls(run_id=f"ec2-v2-gdesigner-seed-{seed}", method="gdesigner")
    search_calls = sum(1 for call in calls if call["split"] in {"search", "select"})
    search_tokens = sum(int(call["total_tokens"]) for call in calls if call["split"] in {"search", "select"})
    manifest = _base_manifest("gdesigner", seed, rows["manifest"])
    manifest.update(
        {
            "official_repo": "https://github.com/yanweiyue/GDesigner",
            "official_commit": git_commit(runtime.root),
            "native_search": "official_train_mmlu_train__query_conditioned_gcn_topology",
            "search_calls": search_calls,
            "search_tokens": search_tokens,
            "select_policy": "audit_only_no_checkpoint_selection_in_official_ten_iteration_loop",
            "select_communication": select_communication,
            "test_communication": test_communication,
            "gdesigner_training": {"iterations": 10, "batch_size": 4, "lr": 0.1, "initial_gcn_sha256": initial_sha, "trained_gcn_sha256": trained_sha},
        }
    )
    validate_v2_manifest(manifest)
    write_native_result(Path(rows["output_dir"]) / "gdesigner" / f"seed_{seed}", manifest, result_rows, calls, {"gcn_sha256": trained_sha})


async def _run_rpas_comm(runtime: OfficialGDesignerRuntime, rows: dict[str, list[MMLUExample]], seed: int, repo_root: Path) -> None:
    config, models, profile = _reflection_context(repo_root)
    seed_topologies = [TOPOLOGIES[0]]
    candidate_rows: list[dict[str, Any]] = []
    evaluated_ids: set[str] = set()
    reflection_call_records: list[dict[str, Any]] = []
    new_candidate_budget = int(os.environ.get("RPAS_EC2_V2_NEW_CANDIDATES", "3"))
    if new_candidate_budget < 1:
        raise ValueError("RPAS_EC2_V2_NEW_CANDIDATES must be positive for formal EC-2 v2")
    if 1 + new_candidate_budget > len(TOPOLOGIES):
        raise ValueError(
            f"RPAS_EC2_V2_NEW_CANDIDATES={new_candidate_budget} exceeds the {len(TOPOLOGIES)}-topology communication space"
        )
    search_assignments = _rpas_search_assignments(
        rows["search"], candidate_count=1 + new_candidate_budget, seed=seed
    )

    async def evaluate_candidate(candidate: dict[str, Any], split: str, examples: list[MMLUExample]) -> tuple[list[dict[str, Any]], dict[str, int], Any]:
        assert_v2_candidate(candidate)
        graph = runtime.make_graph(topology=candidate["topology"], optimized_spatial=False)
        outputs, communication = await runtime.evaluate(graph, examples, split=split, candidate_id=candidate["id"])
        return outputs, communication, graph

    for candidate_index, topology in enumerate(seed_topologies):
        candidate = communication_candidate(topology)
        outputs, communication, _ = await evaluate_candidate(candidate, "search", search_assignments[candidate_index])
        candidate_rows.append(_candidate_row(candidate, outputs, communication))
        evaluated_ids.add(candidate["id"])

    mutation_logs: list[dict[str, Any]] = []
    reflection_calls = 0
    for index in range(new_candidate_budget):
        parent = dict(max(candidate_rows, key=lambda row: (row["score"], -row["avg_inter_agent_tokens"], row["candidate_id"])))
        parent["excluded_candidate_ids"] = sorted(evaluated_ids)
        child, plan = _llm_topology_mutation(parent, config=config, models=models, profile=profile)
        reflection_calls += len(plan.get("call_traces", []))
        reflection_call_records.extend(
            _reflection_call_records(
                plan,
                run_id=f"ec2-v2-rpas_comm-seed-{seed}",
                method="rpas_comm",
                candidate_id=parent["candidate_id"],
                start_index=len(reflection_call_records),
            )
        )
        if child["id"] in evaluated_ids:
            # A duplicate is not a new candidate and cannot satisfy the formal gate.
            alternatives = [name for name in TOPOLOGIES if name != parent["candidate"]["topology"] and communication_candidate(name)["id"] not in evaluated_ids]
            if not alternatives:
                raise RuntimeError("RPAS-Comm LLM reflection exhausted the finite topology space with duplicates")
            raise RuntimeError("RPAS-Comm LLM reflection proposed a duplicate topology; retry with a distinct typed proposal")
        outputs, communication, _ = await evaluate_candidate(child, "search", search_assignments[index + 1])
        child_row = _candidate_row(child, outputs, communication)
        candidate_rows.append(child_row)
        evaluated_ids.add(child["id"])
        mutation_logs.append({"index": index, "parent_id": parent["candidate_id"], "child_id": child["id"], "plan": plan, "applied_mutation": child["applied_mutation"]})

    if not mutation_logs:
        raise RuntimeError("RPAS-Comm did not create any LLM-reflected candidates")
    selection_rows = []
    for row in candidate_rows:
        outputs, communication, _ = await evaluate_candidate(row["candidate"], "select", rows["select"])
        selection_rows.append(_candidate_row(row["candidate"], outputs, communication))
    valid_selection = [row for row in selection_rows if row["valid"]]
    if not valid_selection:
        raise RuntimeError("RPAS-Comm produced no parse-valid candidate on D_select")
    selected = min(valid_selection, key=lambda row: (-row["score"], row["avg_inter_agent_tokens"], row["candidate_id"]))
    test_rows, test_communication, _ = await evaluate_candidate(selected["candidate"], "test", rows["test"])
    calls = runtime.calls(run_id=f"ec2-v2-rpas_comm-seed-{seed}", method="rpas_comm") + reflection_call_records
    search_calls = sum(1 for call in calls if call["split"] in {"search", "select"})
    search_tokens = sum(int(call["total_tokens"]) for call in calls if call["split"] in {"search", "select"})
    manifest = _base_manifest("rpas_comm", seed, rows["manifest"])
    manifest.update(
        {
            "native_search": "llm_reflection__typed_topology_mutation__pareto_select",
            "search_calls": search_calls,
            "search_tokens": search_tokens,
            "search_candidates": len(candidate_rows),
            "rpas_search_query_executions": GDESIGNER_TRAINING_QUERY_BUDGET,
            "rpas_select_query_executions": len(candidate_rows) * len(rows["select"]),
            "selection_rule": "maximize_D_select_accuracy__then_minimize_inter_agent_tokens__then_candidate_id",
            "test_communication": test_communication,
            "rpas_reflection": {
                "reflection_calls": reflection_calls,
                "new_candidates": len(mutation_logs),
                "mutation_logs": len(mutation_logs),
                "rule_fallbacks": sum(log["plan"].get("mode") == "rule_fallback" for log in mutation_logs),
            },
        }
    )
    validate_v2_manifest(manifest)
    output = Path(rows["output_dir"]) / "rpas_comm" / f"seed_{seed}"
    write_native_result(output, manifest, test_rows, calls, selected["candidate"], candidate_rows)
    (output / "selection_rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selection_rows), encoding="utf-8")
    (output / "mutation_logs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in mutation_logs), encoding="utf-8")


async def _run(args: argparse.Namespace) -> None:
    require_authorized_gpu()
    if int(os.environ.get("RPAS_MMLU_MAX_TOKENS", str(MAX_TOKENS))) != MAX_TOKENS:
        raise ValueError(f"EC-2 v2 fixes RPAS_MMLU_MAX_TOKENS={MAX_TOKENS}")
    repo_root = Path(args.repo_root).resolve()
    split_manifest = build_mmlu_manifest(
        args.data_dir,
        data_seed=args.data_seed,
        search_per_subject=args.search_per_subject,
        select_per_subject=args.select_per_subject,
        test_per_subject=args.test_per_subject,
    )
    rows: dict[str, Any] = {
        "search": load_mmlu_split(args.data_dir, "dev", per_subject=args.search_per_subject, seed=args.data_seed),
        "select": load_mmlu_split(args.data_dir, "val", per_subject=args.select_per_subject, seed=args.data_seed),
        "test": load_mmlu_split(args.data_dir, "test", per_subject=args.test_per_subject, seed=args.data_seed),
        "manifest": split_manifest,
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    runtime = OfficialGDesignerRuntime(Path(args.gdesigner_root).resolve(), seed=args.seed)
    if args.method == "single_agent":
        await _run_single(runtime, rows, args.seed)
    elif args.method in {"full_connected", "chain"}:
        await _run_fixed(runtime, args.method, rows, args.seed)
    elif args.method == "gdesigner":
        await _run_gdesigner(runtime, rows, args.seed)
    elif args.method == "rpas_comm":
        await _run_rpas_comm(runtime, rows, args.seed, repo_root)
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(args.method)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one strict EC-2 v2 MMLU seed on GPU 4 or 5.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--gdesigner-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=["single_agent", "full_connected", "chain", "gdesigner", "rpas_comm"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--search-per-subject", type=int, default=1)
    parser.add_argument("--select-per-subject", type=int, default=1)
    parser.add_argument("--test-per-subject", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
