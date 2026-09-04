import json
from pathlib import Path

from external_comparison.runners.hotpotqa_ec3_data import answer_scores, prepare, render_prompt


def _row(index: int) -> dict:
    return {
        "_id": f"id-{index}",
        "question": f"Question {index}?",
        "answer": "yes" if index % 5 == 0 else f"answer {index}",
        "type": "bridge" if index % 2 else "comparison",
        "supporting_facts": [["hidden", 0]],
        "context": [["Title", [f"Context sentence {index}."]]],
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_prepare_freezes_1000_fixture_into_ec3_v3_splits(tmp_path: Path):
    validate = tmp_path / "validate.jsonl"
    test = tmp_path / "test.jsonl"
    calibration = tmp_path / "train.jsonl"
    _write(validate, [_row(index) for index in range(200)])
    _write(test, [_row(index) for index in range(200, 1000)])
    _write(calibration, [_row(index) for index in range(1000, 1040)])

    manifest = prepare(
        validate_fixture_path=validate, test_fixture_path=test, calibration_path=calibration, output_dir=tmp_path / "frozen",
        data_seed=2026, aflow_commit="3f457218fc716093fe53f6df8a5d5e6379d66346",
    )

    assert {name: payload["count"] for name, payload in manifest["splits"].items()} == {
        "calib": 40, "search": 120, "select": 80, "test": 800,
    }
    formal_ids = set(manifest["splits"]["search"]["ids"]) | set(manifest["splits"]["select"]["ids"]) | set(manifest["splits"]["test"]["ids"])
    assert len(formal_ids) == 1000
    assert formal_ids.isdisjoint(manifest["splits"]["calib"]["ids"])
    assert manifest["splits"]["test"]["ids"] == [f"id-{index}" for index in range(200, 1000)]
    row = json.loads((tmp_path / "frozen" / "hotpotqa_test.jsonl").read_text().splitlines()[0])
    prompt = render_prompt(type("Example", (), row)())
    assert "supporting_facts" not in prompt
    assert row["answer"] not in prompt
    assert row["question_type"] not in prompt


def test_hotpotqa_answer_metrics_follow_normalized_em_and_f1():
    assert answer_scores("The New York", "new york") == {"em": 1.0, "f1": 1.0}
    assert answer_scores("new", "new york") == {"em": 0.0, "f1": 2 / 3}
    assert answer_scores("", "") == {"em": 1.0, "f1": 1.0}
