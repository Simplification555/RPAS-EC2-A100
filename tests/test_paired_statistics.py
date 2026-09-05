import pytest

from external_comparison.common.paired_statistics import two_level_paired_bootstrap
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
