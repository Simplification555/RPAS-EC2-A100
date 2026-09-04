from __future__ import annotations

from external_comparison.common.manifest import sha256_json
from external_comparison.common.pareto import efficiency_operating_point, pareto_frontier, quality_operating_point
from external_comparison.common.schema import CallRecord, CandidateRecord
from external_comparison.common.telemetry import summarize_calls


def test_call_schema_and_telemetry_accounting() -> None:
    call = CallRecord(
        run_id="run-1",
        method="rpas",
        dataset="humaneval",
        split="search",
        candidate_id="c1",
        agent="solver",
        model="test-model",
        site="center_a",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        inference_cost=0.01,
    )
    summary = summarize_calls([call.to_dict()])
    assert summary["calls"] == 1
    assert summary["total_tokens"] == 15
    assert summary["inference_cost"] == 0.01


def test_pareto_and_frozen_operating_points() -> None:
    records = [
        CandidateRecord("r", "rpas", "humaneval", "select", "a", 0, 0.80, True, 10, 100, 1.0).to_dict(),
        CandidateRecord("r", "rpas", "humaneval", "select", "b", 0, 0.79, True, 10, 50, 0.5).to_dict(),
        CandidateRecord("r", "rpas", "humaneval", "select", "c", 0, 0.80, True, 12, 120, 1.2).to_dict(),
    ]
    assert [row["candidate_id"] for row in pareto_frontier(records)] == ["a", "b"]
    assert quality_operating_point(records)["candidate_id"] == "a"
    assert efficiency_operating_point(records)["candidate_id"] == "b"


def test_manifest_hash_is_deterministic() -> None:
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
