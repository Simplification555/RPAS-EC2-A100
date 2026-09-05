import json

from scripts.audit_ec12_publication import common_budget_interval, telemetry_gaps


def test_incompatible_aflow_costs_cannot_share_a_ten_percent_budget():
    result = common_budget_interval([227054, 152135, 392888])
    assert result["status"] == "FAIL"
    assert result["minimum_feasible_budget"] > result["maximum_feasible_budget"]


def test_feasible_costs_do_not_prove_a_frozen_budget():
    assert common_budget_interval([100, 100, 100])["status"] == "NEEDS_FROZEN_BUDGET"
    assert common_budget_interval([])["status"] == "PENDING"


def test_missing_finish_reasons_are_unknown_not_zero_truncation(tmp_path):
    call = {"split": "test", "total_tokens": 1, "finish_reason": None}
    (tmp_path / "calls.jsonl").write_text(json.dumps(call) + "\n")
    (tmp_path / "test_outputs.jsonl").write_text(json.dumps({"task_id": "test:1"}) + "\n")
    result = telemetry_gaps(tmp_path)
    assert result["truncation_gate"] == "UNKNOWN"
    assert result["per_item_test_tokens_available"] is False
