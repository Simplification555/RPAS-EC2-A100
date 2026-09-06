from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_ec3_formal import METHODS, SEEDS, aggregate


SPLIT_SHA = "a" * 64


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(root: Path) -> None:
    _write_json(
        root / "d_test_unlock.json",
        {
            "protocol_version": "EC3_HOTPOTQA_V3",
            "d_test_unlocked": True,
            "split_manifest_sha256": SPLIT_SHA,
            "final_states": [
                {"method": method, "seed": seed}
                for method in ("aflow", "rpas")
                for seed in SEEDS
            ],
        },
    )
    for method in METHODS:
        for seed in SEEDS:
            run = root / method / f"seed_{seed}"
            _write_json(
                run / "test_summary.json",
                {
                    "protocol_version": "EC3_HOTPOTQA_V3",
                    "method": method,
                    "seed": seed,
                    "split_manifest_sha256": SPLIT_SHA,
                    "d_test_accessed": True,
                    "answer_f1": 0.5 + seed / 10,
                    "answer_em": 0.4 + seed / 10,
                    "valid_answer_rate": 1.0,
                    "generation_truncation_rate": 0.0,
                    "test_calls": 800,
                    "test_tokens": 2400,
                },
            )
            outputs = "".join(json.dumps({"id": f"item-{index}", "prediction": "x"}) + "\n" for index in range(800))
            calls = "".join(
                json.dumps({"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3, "error": None}) + "\n"
                for _ in range(800)
            )
            (run / "test_outputs.jsonl").write_text(outputs, encoding="utf-8")
            (run / "test_calls.jsonl").write_text(calls, encoding="utf-8")


def test_complete_three_method_matrix_stays_unpromoted(tmp_path: Path) -> None:
    _fixture(tmp_path)
    payload = aggregate(tmp_path, tmp_path / "aggregate")
    assert payload["matrix_complete"] is True
    assert payload["formal_result"] is False
    assert {row["method"] for row in payload["summary"]} == set(METHODS)
    assert len(payload["runs"]) == 9


def test_missing_single_run_is_rejected(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "single_agent" / "seed_2" / "test_summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path, tmp_path / "aggregate")


def test_truncation_gate_is_enforced(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "rpas" / "seed_1" / "test_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generation_truncation_rate"] = 0.01
    _write_json(path, payload)
    with pytest.raises(ValueError, match="truncation gate"):
        aggregate(tmp_path, tmp_path / "aggregate")
