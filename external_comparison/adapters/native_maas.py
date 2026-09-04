"""MaAS-from-scratch EC-1 adapter using official fresh train then test."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from external_comparison.adapters.native_common import call_record, env_path, write_native_result
from external_comparison.adapters.native_aflow import _gpu_env, _selected_endpoint


def _root() -> Path:
    default = Path(__file__).resolve().parents[2] / "external_baselines" / "MaAS"
    return env_path("RPAS_MAAS_ROOT", str(default))


def _run_driver(args, output: Path) -> dict:
    public = getattr(args, "public_test_path", None) or os.environ.get("RPAS_EC1_PUBLIC_TEST_PATH", "")
    if not public:
        raise RuntimeError("EC-1 requires --public-test-path or RPAS_EC1_PUBLIC_TEST_PATH")
    search_fixture = getattr(args, "aflow_validate_path", None)
    test_fixture = getattr(args, "aflow_test_path", None)
    if not search_fixture or not test_fixture:
        raise RuntimeError("EC-1 requires frozen AFlow validate and test fixtures")
    if getattr(args, "run_kind", "pilot") == "formal" and "RPAS_MAAS_SAMPLE" not in os.environ:
        raise RuntimeError("formal MaAS requires an explicitly frozen RPAS_MAAS_SAMPLE after the seed-0 pilot")
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable, "-m", "external_comparison.runners.native_ec1_driver",
        "--method", "maas", "--source-root", str(_root()), "--dataset-path", str(args.dataset_path),
        "--public-test-path", str(public), "--output-dir", str(output), "--seed", str(args.seed),
        "--search-fixture", str(search_fixture), "--test-fixture", str(test_fixture),
        "--data-seed", str(args.data_seed), "--model", os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "--base-url", _selected_endpoint(),
        "--api-key", os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"),
        "--max-tokens", os.environ.get("RPAS_HUMANEVAL_MAX_TOKENS", "1024"),
        "--maas-sample", os.environ.get("RPAS_MAAS_SAMPLE", "4"),
        "--maas-batch-size", os.environ.get("RPAS_MAAS_BATCH_SIZE", "4"),
        "--maas-lr", os.environ.get("RPAS_MAAS_LR", "0.01"),
    ]
    if os.environ.get("RPAS_MAAS_TEST_ONLY") == "1":
        checkpoint = os.environ.get("RPAS_MAAS_CONTROLLER_PATH", "").strip()
        if not checkpoint:
            raise RuntimeError("RPAS_MAAS_TEST_ONLY=1 requires RPAS_MAAS_CONTROLLER_PATH")
        command.extend(["--maas-test-only", "--maas-controller-path", checkpoint])
    if os.environ.get("RPAS_EC1_REPLACE_WORKSPACE") == "1":
        command.append("--replace-workspace")
    subprocess.run(command, cwd=root, env=_gpu_env(), check=True)
    return json.loads((output / "_maas_driver_result.json").read_text(encoding="utf-8"))


def run_humaneval(args) -> None:
    output = Path(args.output_dir) / "maas" / f"seed_{args.seed}"
    result = _run_driver(args, output)
    raw_calls = [json.loads(line) for line in Path(result["telemetry_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    calls = [
        call_record(
            f"humaneval-maas-seed-{args.seed}", "maas", "humaneval", row["phase"],
            f"{row['phase']}-{index}", index, row,
        )
        for index, row in enumerate(raw_calls)
    ]
    manifest = {
        "run_id": f"humaneval-maas-seed-{args.seed}", "method": "maas", "dataset": "humaneval", "seed": args.seed,
        "formal_result": getattr(args, "run_kind", "pilot") == "formal",
        "run_kind": getattr(args, "run_kind", "pilot"), "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "api_base": _selected_endpoint(),
        "gpu": os.environ["RPAS_EC1_GPU"], **result["manifest"],
        "search_calls": sum(row["phase"] == "search" for row in raw_calls),
        "search_tokens": sum(int(row["total_tokens"]) for row in raw_calls if row["phase"] == "search"),
    }
    write_native_result(output, manifest, result["test_rows"], calls, selected={"checkpoint": result["manifest"].get("checkpoint")}, search_rows=result["search_rows"])
