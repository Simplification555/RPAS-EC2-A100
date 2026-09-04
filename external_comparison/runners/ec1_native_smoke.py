"""Offline EC-1 native staging smoke test; it never contacts an LLM endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from external_comparison.adapters.native_runtime import (
    source_manifest,
    stage_checkout,
    stage_humaneval_data,
    write_aflow_config,
    write_maas_config,
)


def run_smoke(
    method: str,
    source_root: Path,
    dataset_path: Path,
    public_test_path: Path,
    validate_path: Path,
    test_path: Path,
    output_dir: Path,
    seed: int,
    data_seed: int = 2026,
) -> dict:
    workspace = stage_checkout(source_root, output_dir, method, seed)
    data = stage_humaneval_data(
        workspace, method, dataset_path, public_test_path, data_seed,
        search_fixture=validate_path, test_fixture=test_path,
    )
    if method == "aflow":
        write_aflow_config(workspace, "smoke-model", "http://127.0.0.1:1/v1", "EMPTY")
        required = workspace / "workspace" / "HumanEval" / "workflows" / "round_1" / "graph.py"
    else:
        write_maas_config(workspace, "smoke-model", "http://127.0.0.1:1/v1", "EMPTY", seed)
        required = workspace / "maas" / "ext" / "maas" / "scripts" / "optimized" / "HumanEval" / "train" / "graph.py"
    if not required.is_file():
        raise RuntimeError(f"missing required official native source after staging: {required}")
    return {"status": "smoke_passed", "formal_result": False, "no_model_calls": True, "required_source": str(required), **source_manifest(source_root, workspace, method, seed, data)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("aflow", "maas"), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--public-test-path", required=True)
    parser.add_argument("--aflow-validate-path", default="data/ec1_humaneval/aflow/humaneval_validate.jsonl")
    parser.add_argument("--aflow-test-path", default="data/ec1_humaneval/aflow/humaneval_test.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=2026)
    args = parser.parse_args()
    result = run_smoke(
        args.method, Path(args.source_root), Path(args.dataset_path), Path(args.public_test_path),
        Path(args.aflow_validate_path), Path(args.aflow_test_path), Path(args.output_dir), args.seed, args.data_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
