"""Serializable records used by all comparison methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CallRecord:
    """One LLM call, with enough information to audit cost and latency."""

    run_id: str
    method: str
    dataset: str
    split: str
    candidate_id: str
    agent: str
    model: str
    site: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    inference_cost: float = 0.0
    model_latency_ms: float | None = None
    wall_latency_ms: float | None = None
    network_latency_ms: float = 0.0
    retry_count: int = 0
    finish_reason: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "retry_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRecord:
    """One evaluated architecture at one search or evaluation split."""

    run_id: str
    method: str
    dataset: str
    split: str
    candidate_id: str
    seed: int
    score: float | None
    valid: bool
    total_calls: int
    total_tokens: int
    total_cost: float
    observed_latency_ms: float | None = None
    cross_center_tokens: int = 0
    cross_center_messages: int = 0
    invalid_reason: str | None = None
    architecture: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("total_calls", "total_tokens", "cross_center_tokens", "cross_center_messages"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.valid and self.score is None:
            raise ValueError("a valid candidate must have a score")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    """Provenance record written next to every formal run."""

    run_id: str
    method: str
    dataset: str
    split: str
    seed: int
    protocol_version: str
    config_sha256: str
    protocol_sha256: str
    data_sha256: dict[str, str]
    source_sha256: dict[str, str]
    git_commit: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

