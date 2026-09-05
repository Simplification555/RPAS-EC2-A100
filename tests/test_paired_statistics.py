import pytest

from external_comparison.common.paired_statistics import heldout_token_map, paired_token_interval, two_level_paired_bootstrap
from external_comparison.runners.aggregate_mmlu_v2 import _mcnemar


def fixture(value):
    return {seed: {f"q{i}": value for i in range(4)} for seed in range(3)}


def test_constant_difference_and_zero_difference():
    result = two_level_paired_bootstrap(fixture(1), fixture(0), repetitions=100)
    assert result["estimate"] == result["lower"] == result["upper"] == 1.0
    result = two_level_paired_bootstrap(fixture(0.5), fixture(0.5), repetitions=100)
    assert result["estimate"] == result["lower"] == result["upper"] == 0.0


def test_between_seed_uncertainty_is_not_lost():
    left = {seed: {f"q{i}": score for i in range(4)} for seed, score in enumerate((0.0, 0.5, 1.0))}
    result = two_level_paired_bootstrap(left, fixture(0), repetitions=1000)
    assert result["estimate"] == 0.5
    assert result["lower"] == 0 and result["upper"] == 1
    assert result == two_level_paired_bootstrap(left, fixture(0), repetitions=1000)


def test_missing_seed_or_item_is_never_silently_intersected():
    right = fixture(0)
    del right[2]
    with pytest.raises(ValueError, match="seeds"):
        two_level_paired_bootstrap(fixture(1), right)
    right = fixture(0)
    right[1]["other"] = right[1].pop("q0")
    with pytest.raises(ValueError, match="held-out IDs"):
        two_level_paired_bootstrap(fixture(1), right)


def test_nonfinite_values_are_rejected():
    left = fixture(1)
    left[0]["q0"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        two_level_paired_bootstrap(left, fixture(0))


def test_mcnemar_balanced_discordance_is_zero():
    assert _mcnemar({"a": 1, "b": 0}, {"a": 0, "b": 1})["chi_square_cc"] == 0.0


def test_token_interval_supports_large_values_without_weakening_score_validation():
    left, right = fixture(4096), fixture(1024)
    result = two_level_paired_bootstrap(left, right, repetitions=100, metric="tokens")
    assert result["estimate"] == result["lower"] == result["upper"] == 3072
    assert result["units"] == "tokens_per_query"
    with pytest.raises(ValueError, match="finite"):
        two_level_paired_bootstrap(left, right, repetitions=100)


def test_token_maps_sum_multiple_calls_and_exclude_search():
    calls = [{"split": phase, "example_id": item, "total_tokens": value}
             for phase, item, value in (("test", "a", 10), ("test", "a", 20),
                                         ("test", "b", 5), ("search", "a", 999))]
    result = heldout_token_map(calls, {"a", "b"})
    assert result["available"] is True
    assert result["item_tokens"] == {"a": 30, "b": 5}


def test_missing_token_ids_are_not_reconstructed_from_order():
    result = heldout_token_map([{"split": "test", "total_tokens": 5}], {"a"})
    assert result["available"] is False and result["item_tokens"] == {}
    left = {seed: result for seed in range(3)}
    paired = paired_token_interval(left, left)
    assert paired["available"] is False
    assert paired["reason"] == "per_item_token_attribution_incomplete"


def test_token_maps_require_all_expected_items():
    calls = [{"split": "test", "example_id": "a", "total_tokens": 5}]
    assert heldout_token_map(calls, {"a", "b"})["available"] is False
    calls[0]["total_tokens"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        heldout_token_map(calls, {"a"})


def test_rpas_literal_task_ids_in_agent_fields_are_preserved():
    call = {"split": "test", "method": "rpas", "dataset": "humaneval",
            "agent": "HumanEval/84:solver:0", "total_tokens": 12}
    result = heldout_token_map([call], {"HumanEval/84"})
    assert result["item_tokens"] == {"HumanEval/84": 12}
    assert result["identity_sources"] == ["recorded_humaneval_agent_task_prefix"]
    call["method"] = "aflow"
    assert heldout_token_map([call], {"HumanEval/84"})["available"] is False
