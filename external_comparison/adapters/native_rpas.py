"""Native RPAS adapter.

The repository's phase-2 runner remains the implementation authority.  This
module only supplies the external-comparison entry point and never aliases an
external baseline as RPAS.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

from external_comparison.adapters.native_common import require_valid_answer_rate, write_native_result


MMLU_MAX_TOKENS = 256
FIXED_NO_SELECTION_CANDIDATE = "solver_verifier_local"


def _protocol_mmlu_candidate(candidate: dict, *, max_tokens: int = MMLU_MAX_TOKENS) -> dict:
    """Freeze the EC-2 decoding budget without changing the search candidate in place."""

    prepared = copy.deepcopy(candidate)
    for agent in prepared.get("agents", []):
        if isinstance(agent, dict):
            agent["max_tokens"] = max_tokens
    if "planner_max_tokens" in prepared:
        prepared["planner_max_tokens"] = max_tokens
    if "temperature" in prepared:
        prepared["temperature"] = 0.0
    return prepared


def _fixed_mmlu_candidate(candidates: list[dict], method: str) -> dict:
    """Return a predeclared ablation architecture without consulting MMLU dev data."""

    target_name = "single_local" if method == "vanilla" else FIXED_NO_SELECTION_CANDIDATE
    for candidate in candidates:
        if candidate.get("name") == target_name:
            return candidate
    raise RuntimeError(f"required fixed MMLU architecture is absent: {target_name}")


def _load_mmlu_runtime(args):
    from experiments.phase2_wan_agent_search import (
        configure_site_penalties,
        load_models,
        load_network_profiles,
        load_sites,
        seed_architectures,
    )

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(
        os.environ.get(
            "RPAS_MODEL_CONFIG",
            str(repo_root / "experiments" / "phase2_mmlu_qwen35_9b.json"),
        )
    ).expanduser().resolve()
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    endpoint = os.environ.get("RPAS_EXTERNAL_API_BASE")
    if endpoint:
        for model in raw_config.get("models", {}).values():
            model["api_base"] = endpoint
    models = load_models(raw_config["models"])
    sites = load_sites(raw_config["sites"])
    configure_site_penalties(sites, raw_config.get("defaults", {}).get("orchestrator_site", "local_a100"))
    profile = load_network_profiles(raw_config["network_profiles"])["lan_homogeneous"]
    max_tokens = int(os.environ.get("RPAS_MMLU_MAX_TOKENS", str(MMLU_MAX_TOKENS)))
    if max_tokens != MMLU_MAX_TOKENS:
        raise ValueError(f"EC-2 requires RPAS_MMLU_MAX_TOKENS={MMLU_MAX_TOKENS}, got {max_tokens}")
    eval_concurrency = max(1, int(os.environ.get("RPAS_MMLU_EVAL_CONCURRENCY", "8")))
    return raw_config, models, profile, max_tokens, eval_concurrency, seed_architectures(raw_config)


def _run_fixed_mmlu(args) -> None:
    from external_comparison.runners.mmlu import evaluate_candidate, load_mmlu_split

    if args.method not in {"vanilla", "rpas_no_selection"}:
        raise ValueError(f"unsupported fixed MMLU method: {args.method}")
    raw_config, models, profile, max_tokens, eval_concurrency, seed_candidates = _load_mmlu_runtime(args)
    candidate = _protocol_mmlu_candidate(
        _fixed_mmlu_candidate(seed_candidates, args.method), max_tokens=max_tokens
    )
    test_per_subject = int(os.environ.get("RPAS_MMLU_TEST_PER_SUBJECT", "10"))
    test = load_mmlu_split(args.data_dir, "test", per_subject=test_per_subject, seed=2026)
    sample_limit = int(os.environ.get("RPAS_NATIVE_SAMPLE_LIMIT", "0"))
    if sample_limit > 0:
        test = test[:sample_limit]
    run_id = f"mmlu-{args.method}-seed-{args.seed}"
    test_result = evaluate_candidate(
        candidate=candidate,
        examples=test,
        models=models,
        profile=profile,
        run_id=run_id,
        method=args.method,
        split="test",
        eval_concurrency=eval_concurrency,
    )
    valid_answer_rate = require_valid_answer_rate(
        test_result["rows"], context=f"{args.method} MMLU seed {args.seed} test"
    )
    implementation = (
        "single_agent_direct_answer" if args.method == "vanilla" else "fixed_predeclared_rpas_executor"
    )
    manifest = {
        "run_id": run_id,
        "method": args.method,
        "dataset": "mmlu",
        "seed": args.seed,
        "implementation_status": "controlled_mmlu_ablation",
        "native_search": "none",
        "fixed_architecture": candidate,
        "fixed_architecture_policy": (
            "single_local predeclared before evaluation"
            if args.method == "vanilla"
            else "solver_verifier_local predeclared before evaluation; no D_search or candidate selection"
        ),
        "official_repo": "repository_root",
        "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "search_calls": 0,
        "search_tokens": 0,
        "search_candidates": 0,
        "search_scope": "no architecture search or selection",
        "search_examples": 0,
        "test_examples": len(test),
        "eval_concurrency": eval_concurrency,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "thinking_disabled": True,
        "answer_parser": "strict_choice_a_b_c_d",
        "valid_answer_rate": valid_answer_rate,
        "formal_result": False,
        "formal_result_reason": "controlled MMLU-57x10 subset, not full MMLU",
    }
    output_dir = Path(args.output_dir) / args.method / f"seed_{args.seed}"
    write_native_result(output_dir, manifest, test_result["rows"], test_result["calls_detail"], candidate)


def run_humaneval(args) -> None:
    from external_comparison.runners.native_rpas_ec1 import run

    output_dir = Path(args.output_dir) / "rpas" / f"seed_{args.seed}"
    result = run(args)
    manifest = {
        "run_id": f"humaneval-rpas-seed-{args.seed}",
        "method": "rpas",
        "dataset": "humaneval",
        "seed": args.seed,
        **result["manifest"],
    }
    write_native_result(
        output_dir,
        manifest,
        result["test_rows"],
        result["calls"],
        selected=result["selected"],
        search_rows=result["search_rows"],
    )
    for name in ("proposal_rows", "reflections", "tool_events"):
        (output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result[name]),
            encoding="utf-8",
        )


def run_mmlu(args) -> None:
    from external_comparison.runners.mmlu import evaluate_candidate, load_mmlu_split
    from external_comparison.common.protocol import RPAS_MAX_ARCHIVE_SIZE
    from experiments.search_adapters.base import CandidateObservation
    from experiments.search_adapters.common_space import build_common_space_adapter

    if args.method in {"vanilla", "rpas_no_selection"}:
        _run_fixed_mmlu(args)
        return
    if args.method != "rpas":
        raise ValueError(f"native_rpas cannot run method {args.method}")
    raw_config, models, profile, max_tokens, eval_concurrency, raw_seed_candidates = _load_mmlu_runtime(args)
    search_per_subject = int(os.environ.get("RPAS_MMLU_SEARCH_PER_SUBJECT", "5"))
    test_per_subject = int(os.environ.get("RPAS_MMLU_TEST_PER_SUBJECT", "10"))
    search = load_mmlu_split(args.data_dir, "dev", per_subject=search_per_subject, seed=2026)
    test = load_mmlu_split(args.data_dir, "test", per_subject=test_per_subject, seed=2026)
    sample_limit = int(os.environ.get("RPAS_NATIVE_SAMPLE_LIMIT", "0"))
    if sample_limit > 0:
        search = search[:sample_limit]
        test = test[:sample_limit]
    adapter = build_common_space_adapter("rpas")
    seed_candidates = [_protocol_mmlu_candidate(candidate, max_tokens=max_tokens) for candidate in raw_seed_candidates]
    adapter.initialize(seed_candidates, raw_config, random.Random(args.seed))
    candidate_list = list(seed_candidates)
    extra = max(0, int(os.environ.get("RPAS_MMLU_NEW_CANDIDATES", "0")))
    for _ in range(min(extra, RPAS_MAX_ARCHIVE_SIZE - len(candidate_list))):
        proposal = adapter.propose()
        candidate = _protocol_mmlu_candidate(proposal.architecture, max_tokens=max_tokens)
        adapter.register_candidate(candidate)
        candidate_list.append(candidate)
    search_rows = []
    for candidate in candidate_list:
        result = evaluate_candidate(
            candidate=candidate,
            examples=search,
            models=models,
            profile=profile,
            run_id=f"mmlu-rpas-seed-{args.seed}",
            method="rpas",
            split="search",
            eval_concurrency=eval_concurrency,
        )
        row = {
            "candidate_id": candidate["id"],
            "candidate": candidate,
            "accuracy": result["accuracy"],
            "valid_answer_rate": result["valid_answer_rate"],
            "valid": result["valid"],
            "total_calls": result["calls"],
            "total_tokens": result["total_tokens"],
            "communication": result["communication"],
        }
        search_rows.append(row)
        adapter.observe(
            CandidateObservation(
                candidate["id"],
                bool(result["valid"]),
                result["accuracy"] if result["valid"] else None,
                result["calls"],
                result["total_tokens"],
                0.0,
                diagnostics={"valid_answer_rate": result["valid_answer_rate"]},
            )
        )
    eligible = [row for row in search_rows if row["valid"]]
    if not eligible:
        raise RuntimeError("RPAS MMLU search produced no candidate with a valid answer rate >= 0.99")
    selected = min(
        eligible,
        key=lambda row: (
            -float(row["accuracy"]),
            int(row["total_tokens"]),
            int(row["total_calls"]),
            str(row["candidate_id"]),
        ),
    )
    test_result = evaluate_candidate(
        candidate=selected["candidate"],
        examples=test,
        models=models,
        profile=profile,
        run_id=f"mmlu-rpas-seed-{args.seed}",
        method="rpas",
        split="test",
        eval_concurrency=eval_concurrency,
    )
    valid_answer_rate = require_valid_answer_rate(
        test_result["rows"], context=f"RPAS MMLU seed {args.seed} test"
    )
    output_dir = Path(args.output_dir) / "rpas" / f"seed_{args.seed}"
    manifest = {
        "run_id": f"mmlu-rpas-seed-{args.seed}",
        "method": "rpas",
        "dataset": "mmlu",
        "seed": args.seed,
        "implementation_status": "controlled_candidate_selection_pilot",
        "native_search": "rpas_pareto_selection_over_predefined_candidates",
        "official_repo": "repository_root",
        "search_calls": sum(int(row["total_calls"]) for row in search_rows),
        "search_tokens": sum(int(row["total_tokens"]) for row in search_rows),
        "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "search_candidates": len(search_rows),
        "new_candidate_budget": extra,
        "reflection_mode": "rule",
        "search_scope": "9 predefined architectures; no new reflective mutation when RPAS_MMLU_NEW_CANDIDATES=0",
        "search_examples": len(search),
        "test_examples": len(test),
        "eval_concurrency": eval_concurrency,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "thinking_disabled": True,
        "answer_parser": "strict_choice_a_b_c_d",
        "valid_answer_rate": valid_answer_rate,
        "formal_result": False,
        "formal_result_reason": "controlled subset and repository formal gates are incomplete",
    }
    write_native_result(output_dir, manifest, test_result["rows"], test_result["calls_detail"], selected)
    (output_dir / "search_rows.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in search_rows) + "\n", encoding="utf-8")
