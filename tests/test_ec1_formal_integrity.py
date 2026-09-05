import hashlib
import json
from pathlib import Path

import pytest

from scripts.aggregate_ec1_native_formal import canonical_split, load


def seal(directory: Path) -> None:
    (directory / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  /remote/{p.name}\n"
                for p in sorted(directory.iterdir()) if p.name != "SHA256SUMS"),
        encoding="utf-8",
    )


@pytest.fixture
def completed(tmp_path):
    directory = tmp_path / "aflow" / "seed_0"
    directory.mkdir(parents=True)
    data = {
        "search_tasks": [f"HumanEval/{i}" for i in range(33)],
        "test_tasks": [f"HumanEval/{i}" for i in range(33, 164)],
        "search_fixture_source_sha256": "a" * 64,
        "test_fixture_source_sha256": "b" * 64,
        "public_test_sha256": "c" * 64,
    }
    manifest = {"method": "aflow", "seed": 0, "formal_result": True, "data": data}
    rows = [{"task_id": item, "passed": True} for item in data["test_tasks"]]
    calls = [{"split": split, "error": None, "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
             for split in ("search", "test")]
    summary = {"num_examples": 131, "score": 1.0, "model_errors": 0,
               "inference_calls": 1, "search_calls": 1, "inference_tokens": 5, "search_tokens": 5}
    for name, value in {
        "run_manifest.json": manifest,
        "result.json": {**manifest, "summary": summary, "rows": rows},
        "run_metrics.json": {"total_model_calls": 2, "total_tokens": 10, "wall_clock_seconds": 100},
    }.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    for name, values in (("test_outputs.jsonl", rows), ("calls.jsonl", calls)):
        (directory / name).write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")
    seal(directory)
    return tmp_path, directory


def test_valid_and_equivalent_native_split(completed):
    root, directory = completed
    assert load(root, "aflow", 0)["total_tokens"] == 10
    manifest = json.loads((directory / "run_manifest.json").read_text())
    data = manifest["data"]
    rpas = {key: value for key, value in data.items() if key.endswith("sha256")}
    rpas["split_manifest"] = {"task_ids": {"search": data["search_tasks"], "test": data["test_tasks"], "select": []}}
    assert canonical_split(manifest) == canonical_split(rpas)


def test_corrupt_checksum_rejected(completed):
    root, directory = completed
    with (directory / "result.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        load(root, "aflow", 0)


@pytest.mark.parametrize("field,value,expected", [
    ("score", 0.5, "score disagrees"),
    ("inference_tokens", 100, "accounting disagrees"),
    ("model_errors", 1, "model errors"),
])
def test_summary_cannot_override_evidence(completed, field, value, expected):
    root, directory = completed
    path = directory / "result.json"
    result = json.loads(path.read_text())
    result["summary"][field] = value
    path.write_text(json.dumps(result))
    seal(directory)
    with pytest.raises(ValueError, match=expected):
        load(root, "aflow", 0)


def test_same_count_wrong_heldout_ids_rejected(completed):
    root, directory = completed
    path = directory / "test_outputs.jsonl"
    path.write_text(path.read_text().replace("HumanEval/33", "HumanEval/0"))
    seal(directory)
    with pytest.raises(ValueError, match="frozen held-out"):
        load(root, "aflow", 0)


def test_missing_split_does_not_compare_equal():
    with pytest.raises(ValueError, match="split IDs"):
        canonical_split({})


def test_hashed_maas_is_not_a_native_baseline(completed):
    root, directory = completed
    destination = root / "maas" / "seed_0"
    destination.parent.mkdir()
    directory.rename(destination)
    for name in ("run_manifest.json", "result.json"):
        path = destination / name
        payload = json.loads(path.read_text())
        payload.update(method="maas", staged_compatibility_patch="offline deterministic hashed 3-gram embedding")
        path.write_text(json.dumps(payload))
    seal(destination)
    with pytest.raises(ValueError, match="not a native baseline"):
        load(root, "maas", 0)
