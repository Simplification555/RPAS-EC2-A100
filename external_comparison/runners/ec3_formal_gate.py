"""Final-state freeze and D_test unlock gate for EC-3 HotpotQA V3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = ("aflow", "rpas")
SEEDS = (0, 1, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pretest_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("protocol_version") != "EC3_HOTPOTQA_V3":
        raise ValueError("EC-3 formal state has an invalid protocol version")
    if manifest.get("dataset") != "hotpotqa" or manifest.get("split_protocol") != "calib__search__select__test_locked":
        raise ValueError("EC-3 formal state has an invalid data protocol")
    if manifest.get("d_test_accessed") is not False:
        raise ValueError("EC-3 candidate state is invalid because D_test was accessed before freeze")
    if int(manifest.get("search_tokens", 0)) <= 0 or int(manifest.get("search_calls", 0)) <= 0:
        raise ValueError("EC-3 final state requires non-zero pre-test search telemetry")
    if manifest.get("method") == "aflow":
        evidence = manifest.get("aflow_search", {})
        if int(evidence.get("new_workflow_rounds", 0)) <= 0 or int(evidence.get("optimizer_calls", 0)) <= 0:
            raise ValueError("EC-3 AFlow requires newly generated workflows and optimizer calls")
        if float(evidence.get("workflow_executable_rate", 0.0)) < 0.95:
            raise ValueError("EC-3 AFlow executable workflow rate must be at least 95%")
    elif manifest.get("method") == "rpas":
        evidence = manifest.get("rpas_search", {})
        if any(int(evidence.get(key, 0)) <= 0 for key in ("reflection_calls", "new_candidates", "mutation_logs")):
            raise ValueError("EC-3 RPAS requires reflection, typed mutation, and new candidate evidence")
        if int(evidence.get("pareto_archive_size", 0)) <= int(evidence.get("seed_archive_size", 0)):
            raise ValueError("EC-3 RPAS Pareto archive must exceed the seed archive")
    else:
        raise ValueError("EC-3 formal state is not an AFlow or RPAS run")


def freeze_state(run_dir: Path) -> Path:
    manifest_path = run_dir / "run_manifest.json"
    candidate_path = run_dir / "selected_candidate.json"
    if not manifest_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError("EC-3 pre-test run requires run_manifest.json and selected_candidate.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_pretest_manifest(manifest)
    state = {
        "protocol_version": "EC3_HOTPOTQA_V3",
        "method": manifest["method"],
        "seed": manifest["seed"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "run_manifest_sha256": sha256_file(manifest_path),
        "selected_candidate_sha256": sha256_file(candidate_path),
        "selected_candidate_path": str(candidate_path.resolve()),
        "d_test_accessed": False,
    }
    target = run_dir / "final_state.json"
    target.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def unlock(root: Path) -> Path:
    frozen: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            run_dir = root / method / f"seed_{seed}"
            state_path = run_dir / "final_state.json"
            if not state_path.is_file():
                raise FileNotFoundError(f"missing EC-3 final state: {state_path}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("protocol_version") != "EC3_HOTPOTQA_V3" or state.get("method") != method or state.get("seed") != seed:
                raise ValueError(f"invalid EC-3 final state: {state_path}")
            manifest_path = run_dir / "run_manifest.json"
            candidate_path = run_dir / "selected_candidate.json"
            if sha256_file(manifest_path) != state.get("run_manifest_sha256") or sha256_file(candidate_path) != state.get("selected_candidate_sha256"):
                raise ValueError(f"EC-3 final state changed after freeze: {run_dir}")
            frozen.append({"method": method, "seed": seed, "final_state_sha256": sha256_file(state_path), **state})
    hashes = {item["split_manifest_sha256"] for item in frozen}
    if len(hashes) != 1:
        raise ValueError("EC-3 cannot unlock D_test because frozen runs have different split manifests")
    payload = {"protocol_version": "EC3_HOTPOTQA_V3", "d_test_unlocked": True, "split_manifest_sha256": hashes.pop(), "final_states": frozen}
    target = root / "d_test_unlock.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="EC-3 V3 final-state and D_test gate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--run-dir", required=True)
    unlock_parser = subparsers.add_parser("unlock")
    unlock_parser.add_argument("--root", required=True)
    args = parser.parse_args()
    target = freeze_state(Path(args.run_dir)) if args.command == "freeze" else unlock(Path(args.root))
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
