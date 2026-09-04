from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from external_comparison.adapters.native_common import write_native_result
from external_comparison.adapters.native_rpas import _fixed_mmlu_candidate, _protocol_mmlu_candidate
from external_comparison.runners.aggregate_mmlu import aggregate
from external_comparison.runners.mmlu import (
    MMLU_SUBJECTS,
    build_mmlu_manifest,
    load_mmlu_subject,
    parse_mmlu_choice,
)


def test_mmlu_choice_parser_requires_unambiguous_final_answer() -> None:
    assert parse_mmlu_choice("Reasoning\nFINAL ANSWER: C") == "C"
    assert parse_mmlu_choice("C") == "C"
    assert parse_mmlu_choice("### B\nSelected by majority") == "B"
    assert parse_mmlu_choice(r"\boxed{D}") == "D"
    assert parse_mmlu_choice("The answer is C, but maybe D") == ""


def test_rpas_mmlu_candidate_freezes_protocol_decoding() -> None:
    candidate = {
        "topology": "planner_solver_verifier",
        "temperature": 0.3,
        "planner_max_tokens": 1024,
        "agents": [{"name": "planner", "max_tokens": 1536}, {"name": "solver"}],
    }
    prepared = _protocol_mmlu_candidate(candidate)
    assert prepared["temperature"] == 0.0
    assert prepared["planner_max_tokens"] == 256
    assert all(agent["max_tokens"] == 256 for agent in prepared["agents"])
    assert candidate["temperature"] == 0.3
    assert "max_tokens" not in candidate["agents"][1]


def test_mmlu_ablations_use_predeclared_distinct_architectures() -> None:
    candidates = [
        {"name": "single_local", "topology": "single"},
        {"name": "solver_verifier_local", "topology": "solver_verifier"},
    ]
    assert _fixed_mmlu_candidate(candidates, "vanilla")["name"] == "single_local"
    assert _fixed_mmlu_candidate(candidates, "rpas_no_selection")["name"] == "solver_verifier_local"


def test_native_result_records_invalid_answer_rate(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    write_native_result(
        output_dir,
        {"method": "gdesigner", "dataset": "mmlu"},
        [{"prediction": "A", "correct": True}, {"prediction": "", "correct": False}],
        [{"total_tokens": 4, "finish_reason": "stop", "error": None}],
    )
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["summary"]["valid_answer_rate"] == 0.5
    assert result["summary"]["num_examples"] == 2


def test_mmlu_loader_rejects_malformed_rows(tmp_path: Path) -> None:
    (tmp_path / "dev").mkdir()
    path = tmp_path / "dev" / "abstract_algebra_dev.csv"
    path.write_text("question,a,b,c,d,E\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid answer"):
        load_mmlu_subject(tmp_path, "abstract_algebra", "dev")


def test_mmlu_manifest_is_deterministic(tmp_path: Path) -> None:
    for split in ("dev", "val", "test"):
        (tmp_path / split).mkdir()
        for subject in MMLU_SUBJECTS:
            path = tmp_path / split / f"{subject}_{split}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                for index in range(3):
                    writer.writerow([f"q{index}", "a", "b", "c", "d", "A"])
    first = build_mmlu_manifest(tmp_path, search_per_subject=2, select_per_subject=2, test_per_subject=2)
    second = build_mmlu_manifest(tmp_path, search_per_subject=2, select_per_subject=2, test_per_subject=2)
    assert first == second
    assert first["search"]["count"] == 2 * len(MMLU_SUBJECTS)
    assert first["select"]["count"] == 2 * len(MMLU_SUBJECTS)
    assert first["test"]["count"] == 2 * len(MMLU_SUBJECTS)


def test_aggregate_mmlu_separates_search_and_test_cost(tmp_path: Path) -> None:
    for method, search_calls in (("rpas", 12), ("gdesigner", 0)):
        for seed in range(3):
            run = tmp_path / method / f"seed_{seed}"
            run.mkdir(parents=True)
            manifest = {"method": method, "seed": seed, "formal_result": False}
            summary = {
                "score": 0.8 if method == "rpas" else 0.7,
                "num_examples": 570,
                "valid_answer_rate": 1.0,
                "inference_calls": 570 if method == "rpas" else 1700,
                "inference_tokens": 1000,
                "search_calls": search_calls,
                "search_tokens": search_calls * 10,
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run / "result.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")
    payload = aggregate(tmp_path, tmp_path / "aggregate")
    rows = {row["method"]: row for row in payload["rows"]}
    assert rows["rpas"]["test_calls_mean"] == 570
    assert rows["rpas"]["search_calls_mean"] == 12
    assert rows["rpas"]["total_calls_mean"] == 582
    assert rows["gdesigner"]["search_cost_note"] == "not separately instrumented"
