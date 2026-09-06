import json
import inspect
import random
from pathlib import Path

import pytest

from external_comparison.runners.ec3_v3 import (
    _calibration_seeds, _native_seeds, _read_json, _require_unlock,
    _select, _select_search_parent, _shortlist, run_pretest,
)
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
    with pytest.raises(RuntimeError, match="2 distinct"):
        _calibration_seeds({})


def test_calibration_real_singleton_config_has_distinct_seeds() -> None:
    config = _read_json(Path(__file__).resolve().parents[1] / "experiments/ec3_hotpotqa_qwen35_9b.json")
    seeds = _calibration_seeds(config)
    assert len({candidate["id"] for candidate in seeds}) == 2
    assert seeds[0]["topology"] == "single"
    assert seeds[1]["topology"] != "single"


def test_search_seeds_are_four_distinct_native_workflows() -> None:
    config = _read_json(Path(__file__).resolve().parents[1] / "experiments/ec3_hotpotqa_qwen35_9b.json")
    seeds = _native_seeds(config, 4)
    assert len({candidate["id"] for candidate in seeds}) == 4
    assert [candidate["topology"] for candidate in seeds] == [
        "single", "self_consistency", "solver_verifier", "solver_verifier",
    ]
    assert seeds[2]["edges"][0]["compression"] == "full"
    assert seeds[3]["edges"][0]["compression"] == "summary"


def test_search_parent_delegates_to_native_defaults(monkeypatch) -> None:
    rows = [_row("a", 0.8, 100, 1)]
    rng = random.Random(0)

    def native(evaluated, generator, mode, **kwargs):
        assert evaluated is rows
        assert generator is rng
        assert mode == "wan_pareto"
        assert kwargs == {"pareto_parent_prob": 0.5, "parent_score_band": 0.05, "parent_top_k": 6}
        return rows[0], "pareto_front"

    monkeypatch.setattr("external_comparison.runners.ec3_v3.select_parent", native)
    assert _select_search_parent(rows, rng) == (rows[0], "pareto_front")


def test_search_parent_excludes_invalid_high_score_and_is_reproducible() -> None:
    rows = [_row("invalid", 1.0, 10, 1), _row("quality", 0.9, 200, 2), _row("cheap", 0.85, 100, 1)]
    rows[0]["is_valid_candidate"] = False
    first = [_select_search_parent(rows, random.Random(seed)) for seed in range(20)]
    second = [_select_search_parent(rows, random.Random(seed)) for seed in range(20)]
    assert first == second
    assert {row["candidate_id"] for row, _ in first} == {"quality", "cheap"}
    with pytest.raises(ValueError, match="No valid"):
        _select_search_parent([rows[0]], random.Random(0))


def test_ec3_pretest_delegates_the_controller_to_native_run_search() -> None:
    source = inspect.getsource(run_pretest)
    assert "core = run_search(" in source
    assert 'candidate_evaluator=task_evaluator' in source
    assert "build_reflection_plan(" not in source
    assert "mutate_candidate(" not in source


def test_aflow_cli_accepts_seed_zero_pilot(monkeypatch, tmp_path: Path) -> None:
    import external_comparison.runners.native_ec3_aflow as aflow

    observed = {}

    def fake_preflight(**kwargs):
        observed["preflight"] = kwargs

    def fake_run(args):
        observed["command"] = args.command
        observed["seed"] = args.seed
        return tmp_path / "result"

    monkeypatch.setenv("RPAS_EXTERNAL_API_BASE", "http://127.0.0.1:29999/v1")
    monkeypatch.setattr(aflow, "preflight", fake_preflight)
    monkeypatch.setattr(aflow, "run_pretest", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "native_ec3_aflow.py",
            "--repo-root", str(tmp_path),
            "--manifest", str(tmp_path / "manifest.json"),
            "--aflow-root", str(tmp_path / "AFlow"),
            "--output-root", str(tmp_path / "outputs"),
            "pilot", "--seed", "0",
        ],
    )
    assert aflow.main() == 0
    assert observed["command"] == "pilot"
    assert observed["seed"] == 0
    assert observed["preflight"]["expected_endpoint"] == "http://127.0.0.1:29999/v1"


def test_aflow_pilot_is_bounded_and_does_not_freeze_formal_state() -> None:
    from external_comparison.runners.native_ec3_aflow import run_pretest as run_aflow_pretest

    source = inspect.getsource(run_aflow_pretest)
    assert "search, select = search[:8], select[:8]" in source
    assert '"formal_result": False' in source
    assert "if pilot:" in source
    assert "else:\n        freeze_state(output)" in source


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
