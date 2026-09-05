import json
import hashlib

import pytest

from external_comparison.adapters.native_common import write_native_result
from external_comparison.runners.aggregate_mmlu_v2 import _load_seed, _validate_checkpoints
from external_comparison.common.manifest import sha256_json
from external_comparison.runners.ec2_v2 import _base_manifest
from external_comparison.runners.mmlu import MMLU_SUBJECTS


@pytest.fixture
def run_dir(tmp_path):
    manifest = _base_manifest("full_connected", 0, {"split_manifest_sha256": "fixture",
        "search": {"count": 57}, "select": {"count": 57}, "test": {"count": 570}})
    manifest.update(search_calls=0, search_tokens=0)
    rows = [{"example_id": f"test:{subject}:{i}", "subject": subject,
             "prediction": "A", "answer": "A", "correct": True} for subject in MMLU_SUBJECTS for i in range(10)]
    ids = {"search": [f"dev:{subject}:0" for subject in MMLU_SUBJECTS],
           "select": [f"val:{subject}:0" for subject in MMLU_SUBJECTS],
           "test": [row["example_id"] for row in rows]}
    manifest["split_manifest_sha256"] = sha256_json(ids)
    frozen = {phase: {"ids": values} for phase, values in ids.items()}
    frozen["split_manifest_sha256"] = manifest["split_manifest_sha256"]
    (tmp_path / "frozen_split.jsonl").write_text(json.dumps(frozen) + "\n")
    calls = [{"split": "test", "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
              "finish_reason": "length" if i < 50 else "stop", "error": None} for i in range(570)]
    write_native_result(tmp_path, manifest, rows, calls)
    return tmp_path


def test_reports_truncation_failure_without_calling_it_a_pass(run_dir):
    result = _load_seed(run_dir)
    assert result["all_call_truncation_rate"] == 50 / 570
    assert result["truncation_gate_passed"] is False


def test_rejects_summary_score_overriding_rows(run_dir):
    path = run_dir / "result.json"
    data = json.loads(path.read_text())
    data["summary"]["score"] = 0.1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="score disagrees"):
        _load_seed(run_dir)


def test_rejects_inconsistent_call_tokens(run_dir):
    path = run_dir / "calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    calls[0]["total_tokens"] = 100
    path.write_text("\n".join(json.dumps(call) for call in calls))
    with pytest.raises(ValueError, match="accounting is invalid"):
        _load_seed(run_dir)


def test_rejects_rewritten_boolean_outcomes(run_dir):
    path = run_dir / "result.json"
    data = json.loads(path.read_text())
    data["rows"][0]["correct"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="correctness disagrees"):
        _load_seed(run_dir)


def test_rejects_wrong_ids_even_when_row_count_matches(run_dir):
    path = run_dir / "result.json"
    result = json.loads(path.read_text())
    result["rows"][0]["example_id"] = "test:wrong-id"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="frozen held-out IDs"):
        _load_seed(run_dir)


def test_rejects_rewritten_frozen_split_with_stale_hash(run_dir):
    path = run_dir / "frozen_split.jsonl"
    frozen = json.loads(path.read_text())
    frozen["test"]["ids"][0] = "test:replacement"
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="hash disagrees"):
        _load_seed(run_dir)


def test_rejects_unknown_subject_with_unchanged_subject_count(run_dir):
    path = run_dir / "result.json"
    result = json.loads(path.read_text())
    subject = result["rows"][0]["subject"]
    for row in result["rows"]:
        if row["subject"] == subject:
            row["subject"] = "not_an_mmlu_subject"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="57 MMLU subjects"):
        _load_seed(run_dir)


def test_checkpoint_hashes_are_checked_against_files(tmp_path):
    files = {}
    for name in ("gdesigner_initial.pt", "gdesigner_pretest.pt"):
        content = name.encode()
        (tmp_path / name).write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    manifest = {"method": "gdesigner", "checkpoint_files": files}
    assert _validate_checkpoints(tmp_path, manifest) is True
    (tmp_path / "gdesigner_pretest.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing or corrupt"):
        _validate_checkpoints(tmp_path, manifest)


def test_legacy_checkpoint_gap_is_explicit_and_new_runtime_must_deliver(tmp_path):
    manifest = {"method": "gdesigner"}
    assert _validate_checkpoints(tmp_path, manifest) is False
    manifest["runtime_audit_version"] = "journal_checkpoints_v1"
    with pytest.raises(ValueError, match="missing its checkpoint"):
        _validate_checkpoints(tmp_path, manifest)


def test_aggregate_rejects_swapped_seed_directories(tmp_path, monkeypatch):
    import external_comparison.runners.aggregate_mmlu_v2 as audit
    monkeypatch.setattr(audit, "_load_seed", lambda _: {"method": "chain", "seed": 1})
    with pytest.raises(ValueError, match="artifact directory"):
        audit.aggregate(tmp_path, tmp_path / "summary", ("chain",))


def test_aggregate_rejects_duplicate_methods_before_loading(tmp_path):
    from external_comparison.runners.aggregate_mmlu_v2 import aggregate
    with pytest.raises(ValueError, match="nonempty subset"):
        aggregate(tmp_path, tmp_path / "summary", ("chain", "chain"))
