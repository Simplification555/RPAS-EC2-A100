"""Protocol-defined nested paired bootstrap over search seeds and held-out IDs."""

from __future__ import annotations

import math
import random
from typing import Mapping


def two_level_paired_bootstrap(
    left: Mapping[int, Mapping[str, float]],
    right: Mapping[int, Mapping[str, float]],
    *,
    repetitions: int = 10_000,
    random_seed: int = 2026,
) -> dict:
    if type(repetitions) is not int or repetitions < 100:
        raise ValueError("At least 100 integer bootstrap repetitions are required")
    if set(left) != {0, 1, 2} or set(right) != {0, 1, 2}:
        raise ValueError("The formal paired statistic requires exactly matching seeds 0, 1, 2")
    item_ids = sorted(left[0])
    if not item_ids:
        raise ValueError("Held-out item IDs must be nonempty")
    expected_ids = set(item_ids)
    differences = []
    for seed in (0, 1, 2):
        if set(left[seed]) != expected_ids or set(right[seed]) != expected_ids:
            raise ValueError("Every seed and method must contain exactly the same held-out IDs; no intersection fallback")
        values = []
        for item in item_ids:
            a, b = left[seed][item], right[seed][item]
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
                   for value in (a, b)):
                raise ValueError("Item scores must be finite numbers in [0, 1]")
            values.append(float(a) - float(b))
        differences.append(values)
    rng = random.Random(random_seed)
    samples = []
    count = len(item_ids)
    for _ in range(repetitions):
        # Method pairing is preserved through differences. Each drawn seed
        # receives an independent with-replacement draw of its matched items.
        chosen_seeds = rng.choices(range(3), k=3)
        total = sum(sum(rng.choices(differences[seed], k=count)) for seed in chosen_seeds)
        samples.append(total / (3 * count))
    samples.sort()
    return {
        "available": True,
        "statistic": "mean_left_minus_right",
        "method": "two_level_paired_percentile_bootstrap",
        "outer_sampling": "three paired search seeds with replacement",
        "inner_sampling": "matched item IDs with replacement, independently for each sampled seed",
        "confidence_level": 0.95,
        "quantile_rule": "empirical percentile order statistics: floor(0.025*B), ceil(0.975*B)-1",
        "repetitions": repetitions,
        "random_seed": random_seed,
        "n_seeds": 3,
        "n_examples_per_seed": count,
        "estimate": sum(sum(row) for row in differences) / (3 * count),
        "lower": samples[math.floor(0.025 * repetitions)],
        "upper": samples[math.ceil(0.975 * repetitions) - 1],
    }
