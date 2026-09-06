import pytest

from external_comparison.runners import ec2_decode_probe as probe
from external_comparison.runners.ec2_v2 import OfficialGDesignerRuntime
from external_comparison.runners.mmlu import MMLUExample


def test_probe_cli_accepts_a_single_targeted_cap(monkeypatch):
    captured = {}
    monkeypatch.setattr(probe, "run", lambda args: captured.update(caps=args.caps, methods=args.methods))
    monkeypatch.setattr("sys.argv", ["ec2_decode_probe", "--data-dir", "data", "--gdesigner-root", "gd",
                                      "--output", "out", "--caps", "2048", "--methods", "full_connected"])
    monkeypatch.setattr(probe.asyncio, "run", lambda value: value)
    probe.main()
    assert captured == {"caps": [2048], "methods": ["full_connected"]}


def test_probe_reads_only_dev_and_freezes_same_ids(tmp_path, monkeypatch):
    def loader(path, split, **kwargs):
        assert split == "dev"
        assert kwargs == {"per_subject": 1, "seed": 2026}
        return [MMLUExample(f"dev:{i}", str(i), "q", ("a", "b", "c", "d"), "A") for i in range(57)]
    monkeypatch.setattr(probe, "load_mmlu_split", loader)
    a = probe.freeze_probe(tmp_path, tmp_path / "a")
    b = probe.freeze_probe(tmp_path, tmp_path / "b")
    assert a == b and len(a) == 8
    assert (tmp_path / "a/frozen_dev.jsonl").read_bytes() == (tmp_path / "b/frozen_dev.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        probe.freeze_probe(tmp_path, tmp_path / "a")


def test_probe_rejects_bad_size(tmp_path):
    with pytest.raises(ValueError):
        probe.freeze_probe(tmp_path, tmp_path / "a", 0)


def test_runtime_preserves_old_default_and_accepts_explicit_probe_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(OfficialGDesignerRuntime, "_load_official_modules", lambda self: None)
    assert OfficialGDesignerRuntime(tmp_path, seed=0).max_tokens == 256
    assert OfficialGDesignerRuntime(tmp_path, seed=0, max_tokens=1024).max_tokens == 1024
    with pytest.raises(ValueError):
        OfficialGDesignerRuntime(tmp_path, seed=0, max_tokens=0)


def test_summary_reports_missing_termination_and_tokens():
    result = probe.summarize([{"correct": True, "prediction": "A"}], [
        {"finish_reason": "length", "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        {"finish_reason": None, "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    ], 1.5)
    assert result["length_rate"] == 0.5
    assert result["missing_finish_reasons"] == 1
    assert result["total_tokens"] == 8
    assert result["wall_seconds"] == 1.5
