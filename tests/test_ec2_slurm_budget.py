from pathlib import Path


def test_slurm_service_accepts_reflection_budget_without_changing_workers():
    wrapper = Path("scripts/scir/ec2_formal_one_gpu.sbatch").read_text()
    runner = Path("external_comparison/runners/ec2_v2.py").read_text()
    assert '[[ "${method}" != "rpas_comm" ]] || service_max_tokens=768' in wrapper
    assert '--max-new-tokens "${service_max_tokens}"' in wrapper
    assert "export RPAS_MMLU_MAX_TOKENS=256" in wrapper
    assert "MAX_TOKENS = 256" in runner
    assert "reflection_max_tokens=max(768, MAX_TOKENS)" in runner
    assert "RPAS_EC2_SERVICE_PORT" in wrapper
