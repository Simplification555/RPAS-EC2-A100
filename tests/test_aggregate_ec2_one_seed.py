from __future__ import annotations

from pathlib import Path

import pytest

import scripts.aggregate_ec2_one_seed as module


def _run(method: str, seed: int, split_sha: str = "frozen") -> dict:
    return {
        "method": method,
        "seed": seed,
        "split_manifest_sha256": split_sha,
        "accuracy": 0.5,
        "subject_macro": 0.5,
        "valid_answer_rate": 1.0,
        "test_calls": 570,
        "search_calls": 0,
        "test_tokens": 5700,
        "search_tokens": 0,
        "total_calls": 570,
        "total_tokens": 5700,
        "active_edges_per_query": 0.0,
        "messages_per_query": 0.0,
        "inter_agent_tokens_per_query": 0.0,
        "judge_input_tokens_per_query": 0.0,
        "total_test_tokens_per_query": 10.0,
        "checkpoint_files_verified": None,
        "all_call_truncation_rate": 0.0,
        "truncation_gate_passed": True,
        "item_scores": {"item": 1.0},
        "heldout_tokens": {},
    }


def test_one_seed_summary_stays_unpromoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_load_seed", lambda path: _run(path.parent.name, int(path.name.removeprefix("seed_"))))
    payload = module.aggregate_one_seed(tmp_path, tmp_path / "summary", 1)
    assert payload["one_seed_complete"] is True
    assert payload["formal_result"] is False
    assert [row["method"] for row in payload["methods"]] == list(module.METHODS)
    assert (tmp_path / "summary" / "one_seed_summary.csv").is_file()


def test_split_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def load(path: Path) -> dict:
        method = path.parent.name
        return _run(method, 1, "different" if method == "rpas_comm" else "frozen")

    monkeypatch.setattr(module, "_load_seed", load)
    with pytest.raises(ValueError, match="frozen split"):
        module.aggregate_one_seed(tmp_path, tmp_path / "summary", 1)
