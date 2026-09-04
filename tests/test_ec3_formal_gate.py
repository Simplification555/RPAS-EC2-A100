import json
from pathlib import Path

import pytest

from external_comparison.runners.ec3_formal_gate import freeze_state, unlock


def _manifest(method: str, seed: int) -> dict:
    payload = {
        "protocol_version": "EC3_HOTPOTQA_V3", "dataset": "hotpotqa", "method": method,
        "seed": seed, "split_protocol": "calib__search__select__test_locked", "d_test_accessed": False,
        "search_calls": 10, "search_tokens": 100, "split_manifest_sha256": "fixed-split",
    }
    if method == "aflow":
        payload["aflow_search"] = {"new_workflow_rounds": 1, "optimizer_calls": 1, "workflow_executable_rate": 1.0}
    else:
        payload["rpas_search"] = {"reflection_calls": 1, "new_candidates": 1, "mutation_logs": 1, "seed_archive_size": 2, "pareto_archive_size": 3}
    return payload


def _write_run(root: Path, method: str, seed: int) -> Path:
    run = root / method / f"seed_{seed}"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps(_manifest(method, seed)), encoding="utf-8")
    (run / "selected_candidate.json").write_text(json.dumps({"candidate": f"{method}-{seed}"}), encoding="utf-8")
    return run


def test_ec3_unlock_requires_all_immutable_final_states(tmp_path: Path):
    for method in ("aflow", "rpas"):
        for seed in (0, 1, 2):
            freeze_state(_write_run(tmp_path, method, seed))
    target = unlock(tmp_path)
    payload = json.loads(target.read_text())
    assert payload["d_test_unlocked"] is True
    assert len(payload["final_states"]) == 6


def test_ec3_freeze_rejects_pretest_access(tmp_path: Path):
    run = _write_run(tmp_path, "rpas", 0)
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest["d_test_accessed"] = True
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="D_test"):
        freeze_state(run)
