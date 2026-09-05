from pathlib import Path


def test_pending_bundle_waits_for_all_workers_and_never_kills_borrowed_service():
    source = Path("scripts/scir/ec2_pending_bundle_packed.sh").read_text()
    assert 'kill -TERM "$shared_service"' not in source
    assert 'kill -TERM "$parent"' not in source
    assert 'for pid in "${workers[@]}"; do wait "$pid" || status=1; done' in source
    assert 'trap stop_child EXIT' in source
    assert source.index('[[ -f "$activation" ]] || exit 2') < source.index('  scripts/serve_transformers')
    assert source.index('flock -n 7 && flock -n 8 && flock -n 9') < source.index('  scripts/serve_transformers')
    assert source.index('ec2_batch_service_preflight.py') < source.index('run_seed gdesigner 2')
    assert '"T $parent_start"' in source


def test_pending_bundle_keeps_exact_budgets_and_endpoint_ownership():
    source = Path("scripts/scir/ec2_pending_bundle_packed.sh").read_text()
    assert 'RPAS_MMLU_MAX_TOKENS=256 RPAS_EC2_FIXED_EVAL_CONCURRENCY=4' in source
    assert '--data-seed 2026 --search-per-subject 1 --select-per-subject 1 --test-per-subject 10' in source
    assert '--max-new-tokens 768 --max-batch-size 4' in source
    assert 'run_seed gdesigner 2 "$shared_port"' in source
    assert 'run_seed rpas_comm 1 "$port"' in source
    assert 'run_seed rpas_comm 2 "$port"' in source
    assert 'co_resident_operational_not_isolated' in source
    assert 'find . -maxdepth 1 -type f ! -name SHA256SUMS' in source
