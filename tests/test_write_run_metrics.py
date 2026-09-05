from __future__ import annotations

import json
import sys
from pathlib import Path

from external_comparison.runners import write_run_metrics


def test_write_run_metrics_records_totals_and_manifest(tmp_path: Path, monkeypatch) -> None:
    result = tmp_path / "result"
    job = tmp_path / "job"
    result.mkdir()
    job.mkdir()
    (result / "run_manifest.json").write_text(
        json.dumps({
            "method": "rpas",
            "dataset": "mmlu",
            "seed": 2,
            "model": "Qwen/Qwen3.5-9B",
            "communication_rounds": 1,
            "max_tokens": 256,
        }),
        encoding="utf-8",
    )
    (result / "calls.jsonl").write_text(
        json.dumps({
            "split": "search",
            "model": "Qwen/Qwen3.5-9B",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "wall_latency_ms": 12,
            "round": 1,
            "finish_reason": "stop",
        })
        + "\n",
        encoding="utf-8",
    )
    (result / "test_calls.jsonl").write_text(
        json.dumps({
            "split": "test",
            "model": "Qwen/Qwen3.5-9B",
            "prompt_tokens": 20,
            "completion_tokens": 6,
            "total_tokens": 26,
            "wall_latency_ms": 18,
            "round": 1,
            "finish_reason": "stop",
        })
        + "\n",
        encoding="utf-8",
    )
    (job / "environment.txt").write_text("run_started_at_epoch=100.0\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_run_metrics",
            "--result-dir",
            str(result),
            "--job-root",
            str(job),
            "--stage",
            "formal_seed",
        ],
    )

    assert write_run_metrics.main() == 0
    metrics = json.loads((result / "run_metrics.json").read_text(encoding="utf-8"))
    assert metrics["total_model_calls"] == 2
    assert metrics["prompt_tokens"] == 30
    assert metrics["completion_tokens"] == 11
    assert metrics["total_tokens"] == 41
    assert metrics["calls"]["by_phase_count"] == {"search": 1, "test": 1}
    assert metrics["rounds"]["configured"] == 1
    assert metrics["rounds"]["observed_max"] == 1
    assert metrics["parameters"]["manifest_snapshot"]["seed"] == 2
