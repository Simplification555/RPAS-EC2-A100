"""Explicit placeholders for external methods not yet integrated."""

from __future__ import annotations

from typing import Any

from external_comparison.adapters.base import AdapterMetadata, ComparisonAdapter


class UnimplementedExternalAdapter(ComparisonAdapter):
    def __init__(self, name: str, native_search_description: str) -> None:
        self._metadata = AdapterMetadata(
            name=name,
            implementation_status="not_implemented",
            native_search_description=native_search_description,
            supports_search=False,
            supports_evaluation=False,
        )

    @property
    def metadata(self) -> AdapterMetadata:
        return self._metadata

    def search(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{self._metadata.name} adapter is intentionally not implemented yet")

    def evaluate(self, candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"{self._metadata.name} adapter is intentionally not implemented yet")

