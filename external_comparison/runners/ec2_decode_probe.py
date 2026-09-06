"""Dev-only paired decode probe. Not a searched-method or held-out result."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from external_comparison.adapters.native_common import git_commit, sha256_file
from external_comparison.runners.ec2_v2 import (
    BACKBONE, ROLES, ROUNDS, OfficialGDesignerRuntime, require_authorized_gpu,
)
from external_comparison.runners.mmlu import load_mmlu_split


def freeze_probe(data_dir: Path, output: Path, count: int = 8) -> list:
    if not 1 <= count <= 57:
        raise ValueError("probe size must be between 1 and 57")
    # Sample subjects without inspecting scores; use the existing dev sampler.
    pool = load_mmlu_split(data_dir, "dev", per_subject=1, seed=2026)
    rows = sorted(random.Random(2026).sample(pool, count), key=lambda row: row.example_id)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "frozen_dev.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    return rows


def summarize(outputs: list[dict], calls: list[dict], elapsed: float) -> dict:
    if not outputs or not calls:
        raise ValueError("cannot summarize an empty probe")
    return {
        "num_examples": len(outputs),
        "accuracy": sum(bool(row["correct"]) for row in outputs) / len(outputs),
        "valid_answer_rate": sum(bool(row["prediction"]) for row in outputs) / len(outputs),
        "calls": len(calls),
        "length_rate": sum(row.get("finish_reason") == "length" for row in calls) / len(calls),
        "missing_finish_reasons": sum(not row.get("finish_reason") for row in calls),
        **{key: sum(int(row.get(key, 0)) for row in calls)
           for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "wall_seconds": elapsed,
    }


async def run(args: argparse.Namespace) -> None:
    gpu = require_authorized_gpu()
    rows = freeze_probe(args.data_dir, args.output, args.count)
    runtime = OfficialGDesignerRuntime(args.gdesigner_root, seed=0)
    # Retries are disabled for the probe so every attempted call is auditable.
    runtime.client = runtime.client.with_options(max_retries=0, timeout=180.0)
    contract = {
        "protocol": "ec2-dev-decode-probe-v1", "formal_result": False,
        "d_test_accessed": False, "dataset_split": "dev", "seed": 0,
        "data_seed": 2026, "num_examples": len(rows), "caps": [256, 1024],
        "methods": ["single_agent", "full_connected"], "model": BACKBONE,
        "temperature": 0.0, "communication_rounds": ROUNDS, "concurrency": 1,
        "max_retries": 0, "request_timeout_seconds": 180,
        "gpu": gpu, "roles": list(ROLES), "official_commit": git_commit(args.gdesigner_root),
        "dev_sha256": sha256_file(args.output / "frozen_dev.jsonl"),
        "runner_sha256": sha256_file(__file__),
        "scope": "fixed-reference decode probe; no GCN training or RPAS search",
    }
    (args.output / "probe_manifest.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    results = []
    try:
        for cap in contract["caps"]:
            runtime.max_tokens = cap
            for method in contract["methods"]:
                directory = args.output / f"{method}_{cap}"
                runtime.configure_artifacts(directory)
                runtime.usage.clear()
                runtime.mmlu_prompt_set.roles = itertools.cycle(ROLES)
                if method == "single_agent":
                    graph = runtime.Graph(
                        domain="mmlu", llm_name=BACKBONE, agent_names=["AnalyzeAgent"],
                        decision_method="FinalRefer", optimized_spatial=False,
                        fixed_spatial_masks=[[0]], fixed_temporal_masks=[[0]],
                    )
                else:
                    graph = runtime.make_graph(topology=method, optimized_spatial=False)
                started = time.perf_counter()
                outputs, communication = await runtime.evaluate(
                    graph, rows, split="calib_dev", candidate_id=f"{method}_{cap}", concurrency=1,
                )
                result = {"method": method, "max_tokens": cap, "formal_result": False,
                          **summarize(outputs, runtime.usage, time.perf_counter() - started),
                          "communication": communication}
                (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                results.append(result)
                print(json.dumps(result), flush=True)
        (args.output / "summary.json").write_text(json.dumps({**contract, "results": results}, indent=2) + "\n", encoding="utf-8")
    finally:
        await runtime.client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gdesigner-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
