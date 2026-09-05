import hashlib

import pytest

from external_comparison.runners.native_ec1_driver import _install_maas_embedding_compat


def test_native_embedding_only_changes_model_location(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("modules.json", "config.json", "model.safetensors", "tokenizer.json"):
        (model / name).write_bytes(b"fixture")
    monkeypatch.setenv("RPAS_MAAS_EMBEDDING_MODEL", str(model))
    helper = tmp_path / "maas/ext/maas/models/utils.py"
    helper.parent.mkdir(parents=True)
    original = "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
    source = f"model = {original}\nother = {original}\ndef sample_operators(probs):\n    return probs\n"
    helper.write_text(source)
    description = _install_maas_embedding_compat(tmp_path)
    expected = source.replace(original, f"SentenceTransformer({str(model.resolve())!r}, local_files_only=True)")
    assert helper.read_text() == expected
    assert hashlib.sha256(b"fixture").hexdigest() in description
    with pytest.raises(RuntimeError, match="pristine"):
        _install_maas_embedding_compat(tmp_path)


def test_native_embedding_fails_closed_without_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("RPAS_MAAS_EMBEDDING_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="local all-MiniLM"):
        _install_maas_embedding_compat(tmp_path)
