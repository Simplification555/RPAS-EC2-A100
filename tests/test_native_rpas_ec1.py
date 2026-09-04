import json
from pathlib import Path

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
