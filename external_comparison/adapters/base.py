"""Adapter contract for fair external comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterMetadata:
    name: str
    implementation_status: str
    native_search_description: str
    supports_search: bool
    supports_evaluation: bool


class ComparisonAdapter(ABC):
    """Every method must expose the same evaluation boundary and telemetry."""

    @property
    @abstractmethod
    def metadata(self) -> AdapterMetadata:
        raise NotImplementedError

    @abstractmethod
    def search(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Return candidate records; implementation must preserve native search."""

    @abstractmethod
    def evaluate(self, candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """Evaluate one candidate using the shared task/model/runtime boundary."""

