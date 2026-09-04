import json
import os
import socket
from pathlib import Path

from experiments.phase2_wan_agent_search import acquire_cache_lock, cache_lock_is_stale
from scripts.aggregate_formal_track import compact_run_record, make_row


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compact_run_record_separates_search_and_test_costs(tmp_path: Path) -> None:
    run_dir = tmp_path / "aime" / "aime_2025" / "lan" / "wan_pareto" / "seed_0"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "mode": "wan_pareto",
            "search_seed": 0,
            "formal_result": False,
            "code_commit": "abc123",
            "config_sha256": "config",
            "model_manifest_sha256": "model",
            "runtime_cuda_visible_devices": "6",
            "dataset_splits": {
                "search": {"normalized_content_sha256": "search"},
                "selection": {"normalized_content_sha256": "select"},
                "test": {"normalized_content_sha256": "test"},
            },
        },
    )
    _write_json(
        run_dir / "result.json",
        {
            "mode": "wan_pareto",
            "seed": 0,
            "metadata": {
                "dataset": "aime",
                "test_name": "aime_2025",
                "dataset_manifest_sha256": "dataset",
                "search_size": 60,
                "selection_size": 30,
                "test_size": 30,
            },
            "seed_candidate_budget": 9,
            "new_candidate_budget": 24,
            "num_candidates": 33,
            "selection_policy": "frozen",
            "candidate_evaluation_overhead": {"calls": 100, "total_tokens": 1000},
            "search_overhead": {"controller_calls": 3, "controller_total_tokens": 50},
            "selected_test_rows": [
                {
                    "selected_rank": 0,
                    "test": {
                        "score": 0.5,
                        "correct": 15,
                        "num_examples": 30,
                        "valid_answer_rate": 1.0,
                        "valid_execution_rate": 1.0,
                        "truncated_unextractable_rate": 0.0,
                        "sum_calls": 30,
                        "sum_total_tokens": 300,
                        "sum_emulated_wall_latency_ms": 900,
                    },
                }
            ],
        },
    )

    record = compact_run_record(run_dir / "result.json")

    assert record["accuracy"] == 0.5
    assert record["test_calls"] == 30
    assert record["search_candidate_calls"] == 100
    assert record["search_controller_calls"] == 3
    assert record["new_candidates"] == 24
    assert record["split_hashes"]["test"] == "test"


def test_make_row_reports_search_and_test_totals() -> None:
    base = {
        "seed": 0,
        "accuracy": 0.5,
        "valid_answer_rate": 1.0,
        "valid_execution_rate": 1.0,
        "truncated_unextractable_rate": 0.0,
        "test_calls": 30,
        "test_tokens": 300,
        "test_latency_ms": 900,
        "search_candidate_calls": 100,
        "search_candidate_tokens": 1000,
        "search_controller_calls": 3,
        "search_controller_tokens": 50,
        "correct": 15,
        "test_examples": 30,
    }
    row = make_row("wan_pareto", [base, {**base, "seed": 1, "accuracy": 0.7}])

    assert row["n_seeds"] == 2
    assert row["accuracy_mean"] == 0.6
    assert row["search_calls_mean"] == 103
    assert row["total_calls_mean"] == 133
    assert row["total_tokens_mean"] == 1350


def test_cache_lock_reclaims_a_dead_local_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "result.lock"
    lock_path.write_text(
        json.dumps({"pid": 999_999_999, "hostname": socket.gethostname(), "created_at": 0}),
        encoding="utf-8",
    )

    assert cache_lock_is_stale(lock_path, stale_seconds=86_400)
    assert acquire_cache_lock(lock_path, stale_seconds=86_400)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["hostname"] == socket.gethostname()


def test_cache_lock_preserves_a_live_local_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "result.lock"
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": 0}),
        encoding="utf-8",
    )

    assert not cache_lock_is_stale(lock_path, stale_seconds=86_400)
    assert not acquire_cache_lock(lock_path, stale_seconds=86_400)
