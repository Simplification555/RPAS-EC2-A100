import json

import pytest

from scripts.scir.ec2_pending_preflight import check_destination, check_model_files


def test_pending_destination_rejects_partial_and_completed_runs(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "environment.txt").write_text("fixture")
    check_destination(tmp_path)
    for name in ("execution_identity.json", "live_rows.jsonl", "result.json", "gdesigner_pretest.pt"):
        path = tmp_path / name
        path.write_text("fixture")
        with pytest.raises(ValueError, match="explicit archival"):
            check_destination(tmp_path)
        path.unlink()
    check_destination(tmp_path / "not_created")


def test_preflight_checks_every_model_shard(tmp_path):
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "part1", "b": "part2"}}))
    (tmp_path / "part1").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="part2"):
        check_model_files(tmp_path)
    (tmp_path / "part2").write_bytes(b"fixture")
    assert check_model_files(tmp_path) == 2
    index.write_text(json.dumps({"weight_map": {"a": "../outside"}}))
    with pytest.raises(ValueError, match="escapes"):
        check_model_files(tmp_path)


def test_formal_launcher_seals_only_stable_top_level_artifacts():
    from pathlib import Path

    source = Path("scripts/scir/ec2_formal_one_gpu.sbatch").read_text()
    assert 'find . -maxdepth 1 -type f ! -name SHA256SUMS' in source
    assert 'find "${job_root}" -type f' not in source
