from pathlib import Path


def test_shared_launcher_does_not_start_or_kill_model_service():
    source = Path("scripts/scir/ec2_shared_endpoint_packed.sh").read_text(encoding="utf-8")
    assert "--model /home" not in source
    assert 'kill "$service"' not in source
    assert "kill -TERM" not in source
    assert "guard_packed_allocation.sh" in source
    assert '"T $parent_start"' in source
    assert 'service_start="$(awk' in source


def test_shared_launcher_preserves_formal_budget_and_waits_for_activation():
    source = Path("scripts/scir/ec2_shared_endpoint_packed.sh").read_text(encoding="utf-8")
    assert "RPAS_MMLU_MAX_TOKENS=256" in source
    assert "--search-per-subject 1 --select-per-subject 1 --test-per-subject 10" in source
    assert source.index('[[ -f "$activation" ]] ||') < source.index("py -m external_comparison.runners.ec2_v2")
    assert '[[ ! -e "$result_dir" ]]' in source
    assert "flock -n 9" in source
