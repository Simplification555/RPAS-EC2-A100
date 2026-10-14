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


def quality_operating_point(records):
    valid=[dict(x) for x in records if x.get("valid") and x.get("score") is not None]
    return min(valid,key=lambda x:(-float(x["score"]),float(x.get("avg_cost",x.get("total_tokens",0))),str(x.get("candidate_id","")))) if valid else None

def efficiency_operating_point(records,delta=0.05):
    valid=[dict(x) for x in records if x.get("valid") and x.get("score") is not None]
    q=quality_operating_point(valid)
    if q is None:return None
    frontier=pareto_frontier(valid,cost_key="avg_cost")
    eligible=[x for x in frontier if float(x["score"])>=float(q["score"])-delta]
    return min(eligible,key=lambda x:(float(x["avg_cost"]),-float(x["score"]),str(x.get("candidate_id","")))) if eligible else None
