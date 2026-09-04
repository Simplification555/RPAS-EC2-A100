"""Frozen constants shared by the external-comparison harness.

The repository's ``EXPERIMENT_PROTOCOL.md`` remains authoritative. Constants
here are deliberately small and explicit so that a run cannot silently use a
different budget or selection rule.
"""

from __future__ import annotations

from typing import Final

SEARCH_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MIN_SEARCH_SEEDS: Final[int] = 3
OPTIONAL_SEARCH_SEEDS: Final[int] = 5
REALIZED_BUDGET_FRACTIONS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75, 1.0)
RPAS_SHARED_SEED_COUNT: Final[int] = 9
RPAS_UNIQUE_CANDIDATE_BUDGET: Final[int] = 24
RPAS_MAX_ARCHIVE_SIZE: Final[int] = 33
SELECTION_SCORE_DELTA: Final[float] = 0.05

CONTROLLED_SEARCH_METHODS: Final[tuple[str, ...]] = (
    "random_as",
    "aflow_style",
    "adas_style",
    "rpas_quality",
    "rpas",
)

# EC-2 combines the external G-Designer executor with two explicitly labelled
# ablations.  The ablations are never aliases for an external baseline.
EC1_EXTERNAL_METHODS: Final[tuple[str, ...]] = ("aflow", "maas", "rpas")
EC2_EXTERNAL_METHODS: Final[tuple[str, ...]] = (
    "vanilla",
    "rpas_no_selection",
    "gdesigner",
    "rpas",
)
EXTERNAL_METHODS: Final[tuple[str, ...]] = EC1_EXTERNAL_METHODS + ("gdesigner",)
SEARCH_METHODS: Final[tuple[str, ...]] = CONTROLLED_SEARCH_METHODS

DATASETS: Final[tuple[str, ...]] = ("humaneval", "mmlu", "hotpotqa")

SELECTION_RULES: Final[dict[str, str]] = {
    "quality": "maximize score; tie-break total_tokens, total_calls, cost, cross_center_tokens, candidate_id",
    "efficiency": "minimize total_tokens among valid Pareto candidates within 0.05 score of the best",
}


def validate_shared_budget(*, realized_calls: int, realized_tokens: int) -> None:
    """Reject negative accounting values before they enter an aggregate."""

    if realized_calls < 0 or realized_tokens < 0:
        raise ValueError("realized calls and tokens must be non-negative")
