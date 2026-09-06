import json
from pathlib import Path

from external_comparison.runners.native_rpas_ec1 import _native_operating_points
from external_comparison.runners.public_test_executor import PublicTestExecutor


def test_public_test_executor_runs_frozen_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "public.jsonl"
    fixture.write_text(
        json.dumps({"problem_id": "HumanEval/0", "entry_point": "f", "test": ["assert candidate(2) == 3"]}) + "\n",
        encoding="utf-8",
    )
    executor = PublicTestExecutor(fixture)
    assert executor.run("HumanEval/0", "f", "def f(x):\n    return x + 1").passed
    assert not executor.run("HumanEval/0", "f", "def f(x):\n    return x").passed


def test_ec1_rpas_single_service_config_has_no_other_gpu_labels() -> None:
    config = Path("experiments/phase2_humaneval_qwen35_9b_single_service.json").read_text(encoding="utf-8")
    assert "gpu6" not in config.lower()
    assert "gpu7" not in config.lower()
    payload = json.loads(config)
    assert list(payload["models"]) == ["qwen35_9b"]
    assert payload["defaults"]["dag_worker_models"] == ["qwen35_9b"] * 3


def test_maas_formal_launcher_uses_one_a100_endpoint_per_seed() -> None:
    launcher = Path("scripts/scir/maas_native_three_seed.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:a100-pcie-40gb:1" in launcher
    assert "#SBATCH --array=0-2%3" in launcher
    assert "one_endpoint_per_gpu" in launcher
    assert "native_minilm_bundle" not in launcher


def test_rpas_uses_native_quality_and_efficiency_operating_points() -> None:
    def row(candidate_id: str, score: float, tokens: float, *, valid: bool = True) -> dict:
        return {
            "candidate_id": candidate_id,
            "candidate": {"id": candidate_id},
            "score": score,
            "valid": valid,
            "is_valid_candidate": valid,
            "avg_total_tokens": tokens,
            "avg_calls": 1.0,
            "avg_inference_cost_usd": 0.0,
            "avg_cross_center_tokens": 0.0,
            "avg_network_latency_ms": 0.0,
            "avg_emulated_latency_ms": 0.0,
            "avg_errors": 0.0,
        }

    shortlist, policy, operating_points = _native_operating_points([
        row("quality", 0.90, 1000.0),
        row("efficient", 0.86, 100.0),
        row("invalid", 1.00, 1.0, valid=False),
    ])

    assert policy == "pareto_front_quality_band_cost_selection_shortlist.band=0.05"
    assert {item["candidate_id"] for item in shortlist} == {"quality", "efficient"}
    assert operating_points["quality"]["candidate_id"] == "quality"
    assert operating_points["efficiency"]["candidate_id"] == "efficient"
