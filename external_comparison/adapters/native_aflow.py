"""AFlow-from-scratch EC-1 adapter using the official Optimizer search."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from external_comparison.adapters.native_common import call_record, env_path, write_native_result


def _root() -> Path:
    default = Path(__file__).resolve().parents[2] / "external_baselines" / "AFlow"
    return env_path("RPAS_AFLOW_ROOT", str(default))


def _gpu_env() -> dict[str, str]:
    gpu = os.environ.get("RPAS_EC1_GPU", "").strip()
    if gpu not in {"4", "5"}:
        raise RuntimeError("EC-1 native execution requires RPAS_EC1_GPU=4 or RPAS_EC1_GPU=5")
    configured = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if configured and configured != gpu:
        raise RuntimeError(f"refusing GPU mismatch: RPAS_EC1_GPU={gpu}, CUDA_VISIBLE_DEVICES={configured}")
    child = dict(os.environ)
    child["CUDA_VISIBLE_DEVICES"] = gpu
    return child


def _selected_endpoint() -> str:
    """Return the sole resident service allowed for the selected EC-1 GPU."""
    gpu = os.environ.get("RPAS_EC1_GPU", "").strip()
    endpoints = {
        "4": "http://127.0.0.1:29500/v1",
        "5": "http://127.0.0.1:29501/v1",
    }
    expected = endpoints.get(gpu)
    if expected is None:
        raise RuntimeError("EC-1 endpoint selection requires RPAS_EC1_GPU=4 or 5")
    configured = os.environ.get("RPAS_EXTERNAL_API_BASE", "").strip()
    if configured and configured != expected:
        raise RuntimeError(
            f"refusing endpoint mismatch for GPU {gpu}: {configured} != {expected}"
        )
    return expected


def _run_driver(args, output: Path) -> dict:
    public = getattr(args, "public_test_path", None) or os.environ.get("RPAS_EC1_PUBLIC_TEST_PATH", "")
    if not public:
        raise RuntimeError("EC-1 requires --public-test-path or RPAS_EC1_PUBLIC_TEST_PATH")
    search_fixture = getattr(args, "aflow_validate_path", None)
    test_fixture = getattr(args, "aflow_test_path", None)
    if not search_fixture or not test_fixture:
        raise RuntimeError("EC-1 requires frozen AFlow validate and test fixtures")
    if getattr(args, "run_kind", "pilot") == "formal" and "RPAS_AFLOW_MAX_ROUNDS" not in os.environ:
        raise RuntimeError("formal AFlow requires an explicitly frozen RPAS_AFLOW_MAX_ROUNDS after the seed-0 pilot")
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable, "-m", "external_comparison.runners.native_ec1_driver",
        "--method", "aflow", "--source-root", str(_root()), "--dataset-path", str(args.dataset_path),
        "--public-test-path", str(public), "--output-dir", str(output), "--seed", str(args.seed),
        "--search-fixture", str(search_fixture), "--test-fixture", str(test_fixture),
        "--data-seed", str(args.data_seed), "--model", os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "--base-url", _selected_endpoint(),
        "--api-key", os.environ.get("RPAS_EXTERNAL_API_KEY", "EMPTY"),
        "--max-tokens", os.environ.get("RPAS_HUMANEVAL_MAX_TOKENS", "1024"),
        "--aflow-max-rounds", os.environ.get("RPAS_AFLOW_MAX_ROUNDS", "2"),
        "--aflow-sample", os.environ.get("RPAS_AFLOW_SAMPLE", "4"),
        "--aflow-validation-rounds", os.environ.get("RPAS_AFLOW_VALIDATION_ROUNDS", "1"),
    ]
    if os.environ.get("RPAS_AFLOW_TEST_ONLY") == "1":
        workspace = os.environ.get("RPAS_AFLOW_EXISTING_WORKSPACE", "").strip()
        if not workspace:
            raise RuntimeError(
                "RPAS_AFLOW_TEST_ONLY=1 requires RPAS_AFLOW_EXISTING_WORKSPACE; "
                "a held-out resume may only consume a completed isolated search workspace"
            )
        command.extend(["--aflow-test-only", "--aflow-existing-workspace", workspace])
    if os.environ.get("RPAS_EC1_REPLACE_WORKSPACE") == "1":
        command.append("--replace-workspace")
    subprocess.run(command, cwd=root, env=_gpu_env(), check=True)
    return json.loads((output / "_aflow_driver_result.json").read_text(encoding="utf-8"))


def run_humaneval(args) -> None:
    output = Path(args.output_dir) / "aflow" / f"seed_{args.seed}"
    result = _run_driver(args, output)
    raw_calls = [json.loads(line) for line in Path(result["telemetry_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    calls = [
        call_record(
            f"humaneval-aflow-seed-{args.seed}", "aflow", "humaneval", row["phase"],
            f"{row['phase']}-{index}", index, row,
        )
        for index, row in enumerate(raw_calls)
    ]
    manifest = {
        "run_id": f"humaneval-aflow-seed-{args.seed}", "method": "aflow", "dataset": "humaneval", "seed": args.seed,
        "formal_result": getattr(args, "run_kind", "pilot") == "formal",
        "run_kind": getattr(args, "run_kind", "pilot"), "model": os.environ.get("RPAS_EXTERNAL_MODEL", "Qwen/Qwen3.5-9B"),
        "api_base": _selected_endpoint(),
        "gpu": os.environ["RPAS_EC1_GPU"], **result["manifest"],
        "search_calls": sum(row["phase"] == "search" for row in raw_calls),
        "search_tokens": sum(int(row["total_tokens"]) for row in raw_calls if row["phase"] == "search"),
    }
    write_native_result(output, manifest, result["test_rows"], calls, selected={"round": result["manifest"].get("selected_round")}, search_rows=result["search_rows"])
