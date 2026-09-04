#!/usr/bin/env python3
"""Evaluate validation-selected AIME architectures on a second held-out year."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from experiments.phase2_wan_agent_search import (
    candidate_validity,
    load_aime_dataset,
    load_models,
    load_network_profiles,
    load_sites,
    normalized_split_manifest,
    read_json,
    sha256_file,
    write_json,
    configure_site_penalties,
    evaluate_candidate_cached,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen AIME Q/E candidates on a second test year.")
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-concurrency", type=int, default=4)
    args = parser.parse_args()

    primary_run = args.primary_run.resolve()
    primary_result_path = primary_run / "result.json"
    if not primary_result_path.is_file():
        raise FileNotFoundError(primary_result_path)
    primary = read_json(primary_result_path)
    if primary.get("metadata", {}).get("dataset") != "aime":
        raise ValueError("--primary-run must be an AIME search result")

    config = read_json(args.config)
    models = load_models(config["models"])
    sites = load_sites(config["sites"])
    configure_site_penalties(sites, config.get("defaults", {}).get("orchestrator_site", "local_a100"))
    profiles = load_network_profiles(config["network_profiles"])
    profile = profiles[primary["metadata"]["network_profile"]]
    _, _, testset = load_aime_dataset(
        data_dir=args.data_dir,
        train_file="aimo-validation-aime.jsonl",
        test_file=args.test_file,
        train_size=0,
        val_size=0,
        test_size=30,
        seed=int(primary["metadata"]["data_seed"]),
    )
    selected = {
        "Q": read_json(primary_run / "selected_quality_candidate.json"),
        "E": read_json(primary_run / "selected_efficiency_candidate.json"),
    }
    evaluated: dict[str, dict] = {}
    for point, candidate in selected.items():
        result, cache_status = evaluate_candidate_cached(
            candidate=candidate,
            dataset=testset,
            models=models,
            profile=profile,
            cache_dir=None,
            capture_outputs=True,
            eval_concurrency=max(1, args.eval_concurrency),
        )
        result.update(candidate_validity(result))
        evaluated[point] = {
            "candidate_id": candidate["id"],
            "candidate_name": candidate["name"],
            "topology": candidate["topology"],
            "evaluation_cache_status": cache_status,
            "test": result,
        }
    payload = {
        "protocol_version": primary["metadata"].get("experiment_protocol_version", ""),
        "primary_search_result_sha256": sha256_file(primary_result_path),
        "primary_run_manifest_sha256": sha256_file(primary_run / "run_manifest.json"),
        "selection_source": "D_select only; no secondary-test feedback is used for selection",
        "test_file": args.test_file,
        "test_split": normalized_split_manifest(testset),
        "network_profile": asdict(profile),
        "operating_points": evaluated,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
