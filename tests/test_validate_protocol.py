import json
from pathlib import Path

from external_comparison.runners.validate_protocol import validate_config


CONFIG = Path(__file__).resolve().parents[1] / "external_comparison" / "configs" / "ec1_livecodebench_v4.json"


def test_livecodebench_v4_config_passes_the_repository_validator() -> None:
    assert validate_config(CONFIG) == []


def test_livecodebench_v4_config_rejects_scientific_parameter_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["shared_inference"]["max_tokens"] = 4096
    payload["split_sizes"]["test"] = 128
    payload["rpas"]["rule_fallback"] = "allowed"
    target = tmp_path / "ec1_livecodebench_v4.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    errors = validate_config(target)
    assert any("split sizes" in error for error in errors)
    assert any("shared inference" in error for error in errors)
    assert any("rule fallback" in error for error in errors)
