import json
from pathlib import Path

from external_comparison.runners.ec3_preflight import AFLOW_COMMIT, preflight
from external_comparison.runners.hotpotqa_ec3_data import prepare


def _row(index: int) -> dict:
    return {
        "_id": f"id-{index}", "question": f"Q {index}", "answer": "yes" if index % 3 == 0 else f"a {index}",
        "type": "bridge" if index % 2 else "comparison", "context": [["T", [f"C {index}"]]],
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_ec3_preflight_accepts_pinned_clean_fixture(monkeypatch, tmp_path: Path):
    validate = tmp_path / "validate.jsonl"
    test = tmp_path / "test.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    _write(validate, [_row(index) for index in range(200)])
    _write(test, [_row(index) for index in range(200, 1000)])
    _write(calibration, [_row(index) for index in range(1000, 1040)])
    frozen = tmp_path / "frozen"
    prepare(validate_fixture_path=validate, test_fixture_path=test, calibration_path=calibration, output_dir=frozen, data_seed=2026, aflow_commit=AFLOW_COMMIT)

    class Result:
        def __init__(self, text: str): self.stdout = text

    def fake_run(command, **_kwargs):
        return Result(AFLOW_COMMIT + "\n") if command[-2:] == ["rev-parse", "HEAD"] else Result("")

    monkeypatch.setattr("external_comparison.runners.ec3_preflight.subprocess.run", fake_run)
    monkeypatch.setenv("RPAS_EXTERNAL_API_BASE", "http://127.0.0.1:29500/v1")
    payload = preflight(manifest_path=frozen / "hotpotqa_manifest.json", aflow_root=tmp_path / "AFlow", expected_endpoint="http://127.0.0.1:29500/v1")
    assert payload["status"] == "ready_for_calibration"
