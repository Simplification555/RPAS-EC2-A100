import pytest

from scripts.scir.qwen_decode_kernel_preflight import check, extract_function


def test_kernel_extraction_does_not_execute_module_or_decorators(tmp_path):
    source = tmp_path / "synthetic_kernel.py"
    source.write_text("raise RuntimeError('module must not run')\n@unknown_decorator\ndef kernel(x):\n    return x + 1\n")
    function = extract_function(source, "kernel", {})
    assert function(4) == 5


def test_kernel_audit_requires_allocated_gpu_before_importing_torch(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="allocated Slurm GPU"):
        check(tmp_path / "unused.py", tmp_path / "unused.json")
