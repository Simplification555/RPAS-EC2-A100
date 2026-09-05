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
    metric: str = "score",
) -> dict:
    if metric not in {"score", "tokens"}:
        raise ValueError("metric must be score or tokens")
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
            if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   or (metric == "score" and value > 1)
                   for value in (a, b)):
                raise ValueError("Item values must be finite nonnegative numbers; scores must be in [0, 1]")
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
        "metric": metric,
        "units": "tokens_per_query" if metric == "tokens" else "score_fraction",
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


def heldout_token_map(calls: list[dict], item_ids: set[str]) -> dict:
    """Require observed call-to-item attribution; never infer it from call order."""
    test_calls = [call for call in calls if call.get("split") == "test"]
    tokens = {item: 0 for item in item_ids}
    seen = set()
    identity_sources = set()
    for call in test_calls:
        item = call.get("example_id")
        source = "example_id"
        if not item and call.get("method") == "rpas" and call.get("dataset") == "humaneval":
            # humaneval._call_records stores the literal task ID in this field:
            # agent = f"{task_id}:{call.agent}:{index}". This is not order inference.
            agent = call.get("agent", "")
            fields = agent.split(":") if isinstance(agent, str) else []
            if len(fields) >= 3 and fields[-1].isdigit():
                item = fields[0]
                source = "recorded_humaneval_agent_task_prefix"
        if not isinstance(item, str) or item not in item_ids:
            return {"available": False, "reason": "missing_or_unknown_test_call_example_id", "item_tokens": {}}
        value = call.get("total_tokens")
        if type(value) is not int or value < 0:
            raise ValueError("Per-item tokens require nonnegative integer call usage")
        tokens[item] += value
        seen.add(item)
        identity_sources.add(source)
    if not item_ids or seen != item_ids:
        return {"available": False, "reason": "incomplete_test_call_item_coverage", "item_tokens": {}}
    return {"available": True, "reason": "observed_call_task_identity", "identity_sources": sorted(identity_sources),
            "item_tokens": tokens}


def paired_token_interval(left: dict[int, dict], right: dict[int, dict]) -> dict:
    if set(left) != {0, 1, 2} or set(right) != {0, 1, 2}:
        raise ValueError("Token interval requires all three paired seeds")
    missing = {side: {seed: row.get("reason") for seed, row in runs.items() if not row.get("available")}
               for side, runs in (("left", left), ("right", right))}
    if any(missing.values()):
        return {"available": False, "reason": "per_item_token_attribution_incomplete", "missing": missing}
    return two_level_paired_bootstrap(
        {seed: row["item_tokens"] for seed, row in left.items()},
        {seed: row["item_tokens"] for seed, row in right.items()}, metric="tokens",
    )
