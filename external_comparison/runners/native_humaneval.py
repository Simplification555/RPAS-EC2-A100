"""Dispatch EC-1 to repository-local native adapters.

This module is intentionally not a proxy. An adapter must export
``run_humaneval(args: argparse.Namespace)`` and own its native search loop;
the shared evaluator/telemetry contract is the integration boundary.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

from external_comparison.adapters.registry import NATIVE_ADAPTER_MODULES, require_native_adapters
from external_comparison.runners.ec1_preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--method", choices=["aflow", "maas", "rpas"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--public-test-path", help="AFlow-derived public-test fixture shared by all EC-1 methods")
    parser.add_argument("--aflow-validate-path", default="data/ec1_humaneval/aflow/humaneval_validate.jsonl", help="Frozen AFlow validation fixture shared by all EC-1 methods")
    parser.add_argument("--aflow-test-path", default="data/ec1_humaneval/aflow/humaneval_test.jsonl", help="Frozen AFlow held-out fixture shared by all EC-1 methods")
    parser.add_argument("--run-kind", choices=("pilot", "formal"), default="pilot")
    args = parser.parse_args()
    selected_gpu = os.environ.get("RPAS_EC1_GPU", "").strip()
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if selected_gpu not in {"4", "5"} or visible_gpu != selected_gpu:
        parser.error(
            "EC-1 requires one matching physical GPU: "
            "RPAS_EC1_GPU=CUDA_VISIBLE_DEVICES=4 or 5"
        )
    if args.run_kind == "formal":
        if not args.public_test_path:
            parser.error("formal EC-1 requires --public-test-path")
        run_preflight(
            Path(args.dataset_path), Path(args.public_test_path),
            Path("external_comparison/configs/ec1_humaneval.json"),
            validate_path=Path(args.aflow_validate_path), test_path=Path(args.aflow_test_path),
        )
    require_native_adapters([args.method])
    module = importlib.import_module(NATIVE_ADAPTER_MODULES[args.method])
    runner = getattr(module, "run_humaneval", None)
    if runner is None:
        raise RuntimeError(f"native adapter {args.method} must export run_humaneval(args)")
    runner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
