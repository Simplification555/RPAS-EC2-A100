"""Shared quality/cost operating-point calculations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def pareto_frontier(
    records: Iterable[Mapping[str, Any]],
    *,
    quality_key: str = "score",
    cost_key: str = "total_tokens",
) -> list[dict[str, Any]]:
    """Return valid records not dominated by higher quality and lower cost."""

    candidates = [dict(record) for record in records if record.get("valid", False) and record.get(quality_key) is not None]
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        dominated = any(
            other[quality_key] >= candidate[quality_key]
            and other[cost_key] <= candidate[cost_key]
            and (other[quality_key] > candidate[quality_key] or other[cost_key] < candidate[cost_key])
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (float(row[quality_key]), -float(row[cost_key])), reverse=True)


def quality_operating_point(records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Select Q using the frozen protocol's quality-first tie-break order."""

    valid = [dict(record) for record in records if record.get("valid", False) and record.get("score") is not None]
    if not valid:
        return None
    return max(
        valid,
        key=lambda row: (
            float(row["score"]),
            -int(row.get("total_tokens", 0)),
            -int(row.get("total_calls", 0)),
            -float(row.get("total_cost", 0.0)),
            -int(row.get("cross_center_tokens", 0)),
            str(row.get("candidate_id", "")),
        ),
    )


def efficiency_operating_point(records: Iterable[Mapping[str, Any]], delta: float = 0.05) -> dict[str, Any] | None:
    """Select the cheapest valid candidate within ``delta`` of best quality."""

    valid = [dict(record) for record in records if record.get("valid", False) and record.get("score") is not None]
    if not valid:
        return None
    best_score = max(float(row["score"]) for row in valid)
    eligible = [row for row in valid if float(row["score"]) >= best_score - delta]
    return min(
        eligible,
        key=lambda row: (
            int(row.get("total_tokens", 0)),
            int(row.get("total_calls", 0)),
            float(row.get("total_cost", 0.0)),
            str(row.get("candidate_id", "")),
        ),
    )

