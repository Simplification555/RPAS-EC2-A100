"""The narrow contract between a search policy and the shared runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Proposal:
    """A typed architecture proposal before validation and execution."""

    candidate_id: str
    architecture: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateObservation:
    """Only post-execution search evidence visible to an adapter."""

    candidate_id: str
    valid: bool
    score: float | None
    total_calls: int
    total_tokens: int
    total_cost: float
    split: str = "search"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SearchAdapter(Protocol):
    """A proposal policy; data loading, execution, scoring, and selection stay outside."""

    method_id: str

    def initialize(self, seed_archive: list[dict[str, Any]], search_space: dict[str, Any], rng: Any) -> None:
        ...

    def propose(self) -> Proposal:
        ...

    def observe(self, observation: CandidateObservation) -> None:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...

