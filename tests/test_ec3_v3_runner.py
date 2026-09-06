import json
from pathlib import Path

import pytest

from external_comparison.runners.ec3_v3 import _calibration_seeds, _read_json, _require_unlock, _select, _shortlist
from external_comparison.runners.native_ec3_aflow import _truncation_rate
from experiments.phase2_wan_agent_search import (
    extract_prediction_for_dataset,
    score_example_answer,
    task_instruction,
)


def _row(candidate_id: str, f1: float, tokens: float, calls: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "answer_f1": f1,
        "score": f1,
        "avg_total_tokens": tokens,
        "avg_calls": calls,
        "avg_errors": 0.0,
        "avg_inference_cost_usd": 0.0,
        "avg_cross_center_tokens": 0.0,
        "avg_network_latency_ms": 0.0,
        "is_valid_candidate": True,
    }


def test_hotpotqa_shared_protocol_requires_a_final_answer_line() -> None:
    assert "supplied context" in task_instruction("hotpotqa")
    output = "FINAL ANSWER: New York\n"
    assert extract_prediction_for_dataset(output, "hotpotqa") == "New York"
    assert score_example_answer(output, "the new york", "hotpotqa") == 1.0
    assert extract_prediction_for_dataset("New York", "hotpotqa") == ""


def test_ec3_shortlist_and_selection_apply_deterministic_tie_breaks() -> None:
    rows = [
        _row("a", 0.6, 200, 2), _row("b", 0.6, 100, 3), _row("c", 0.5, 80, 1),
        _row("d", 0.4, 40, 1), _row("e", 0.3, 20, 1), _row("f", 0.2, 10, 1),
    ]
    shortlisted = _shortlist(rows)
    assert len(shortlisted) <= 5
    assert {row["candidate_id"] for row in shortlisted}.issuperset({"a", "b", "c"})
    selected = _select([_row("z", 0.7, 200, 3), _row("y", 0.7, 100, 4)])
    assert selected["quality"]["candidate_id"] == "y"
    assert selected["efficiency"]["candidate_id"] == "y"


def test_ec3_test_unlock_must_match_the_frozen_split(tmp_path: Path) -> None:
    manifest = {"split_manifest_sha256": "frozen"}
    with pytest.raises(RuntimeError, match="locked"):
        _require_unlock(tmp_path, manifest)
    (tmp_path / "d_test_unlock.json").write_text(json.dumps({"d_test_unlocked": True, "split_manifest_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        _require_unlock(tmp_path, manifest)
    (tmp_path / "d_test_unlock.json").write_text(json.dumps({"d_test_unlocked": True, "split_manifest_sha256": "frozen"}), encoding="utf-8")
    assert _require_unlock(tmp_path, manifest)["d_test_unlocked"] is True


def test_ec3_json_reader_accepts_cli_string_paths(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"protocol_version": "EC3_HOTPOTQA_V3"}), encoding="utf-8")
    assert _read_json(str(path))["protocol_version"] == "EC3_HOTPOTQA_V3"


def test_calibration_skips_duplicate_native_ids(monkeypatch) -> None:
    candidates = [
        {"id": "single", "name": "local"},
        {"id": "single", "name": "remote"},
        {"id": "multi", "name": "self_consistency"},
    ]
    monkeypatch.setattr("external_comparison.runners.ec3_v3.seed_architectures", lambda _: candidates)
    assert _calibration_seeds({}) == [candidates[0], candidates[2]]


def test_calibration_rejects_only_one_distinct_candidate(monkeypatch) -> None:
    monkeypatch.setattr("external_comparison.runners.ec3_v3.seed_architectures", lambda _: [{"id": "a"}] * 2)
    with pytest.raises(RuntimeError, match="two distinct"):
        _calibration_seeds({})


def test_calibration_real_singleton_config_has_distinct_seeds() -> None:
    config = _read_json(Path(__file__).resolve().parents[1] / "experiments/ec3_hotpotqa_qwen35_9b.json")
    seeds = _calibration_seeds(config)
    assert len({candidate["id"] for candidate in seeds}) == 2
    assert seeds[0]["topology"] == "single"
    assert seeds[1]["topology"] != "single"


@pytest.mark.parametrize("dataset,expected", [("hotpotqa", "FINAL ANSWER: German"), ("aime", "### German")])
def test_majority_vote_preserves_dataset_output_contract(monkeypatch, dataset, expected):
    import experiments.phase2_wan_agent_search as native
    candidate = {"topology": "self_consistency", "samples": 3,
                 "agents": [{"name": "solver", "site": "center_a"}]}
    answers = iter(["FINAL ANSWER: German", "FINAL ANSWER: French", "FINAL ANSWER: German"])
    monkeypatch.setattr(native, "call_agent", lambda **kwargs: (next(answers), candidate["agents"][0]))
    monkeypatch.setattr(native, "add_message_trace", lambda *args, **kwargs: None)
    output, _ = native.run_single_architecture(
        candidate=candidate, example={"input": "fixture", "dataset": dataset}, models={}, profile=None,
    )
    assert expected in output
    if dataset == "hotpotqa":
        assert native.extract_prediction_for_dataset(output, dataset) == "German"
        assert native.score_example_answer(output, "German", dataset) == 1.0


def test_aflow_truncation_rate_uses_the_frozen_executor_cap(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    rows = [
        {"agent": "aflow_executor", "split": "calib_executor", "completion_tokens": 300},
        {"agent": "aflow_executor", "split": "calib_executor", "completion_tokens": 512},
        {"agent": "aflow_meta", "split": "calib_search", "completion_tokens": 4096},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert _truncation_rate(path, executor_cap=512, meta_cap=4096) == pytest.approx(2 / 3)
