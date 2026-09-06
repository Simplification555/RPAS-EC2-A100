from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ec1_launchers_use_cross_node_atomic_lock_directories() -> None:
    for relative in (
        "scripts/scir/ec1_livecodebench_seed0.sbatch",
        "scripts/scir/ec1_formal_one_gpu.sbatch",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "lock_dir=" in source
        assert 'mkdir "$lock_dir" 2>/dev/null ||' in source
        assert 'printf \'%s\\n\' "${SLURM_JOB_ID}" >"${lock_dir}/owner_job_id"' in source
        assert "flock" not in source
