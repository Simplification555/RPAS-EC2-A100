"""Read RPAS artifacts into the shared comparison schema.

This intentionally does not reimplement the RPAS search loop. The existing
phase-2 runner remains the source of truth; this adapter is the normalization
boundary for later aggregate/plot code.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from external_comparison.adapters.base import AdapterMetadata, ComparisonAdapter


class RPASArtifactAdapter(ComparisonAdapter):
    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="rpas",
            implementation_status="artifact_reader_ready",
            native_search_description="RPAS typed architecture search with reflection and Pareto selection",
            supports_search=False,
            supports_evaluation=True,
        )

    def search(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError("run the repository's frozen RPAS search first; this adapter does not duplicate it")

    def evaluate(self, candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("evaluation normalization is pending the final shared evaluator contract")

    @staticmethod
    def read_summary_csv(path: str | Path) -> list[dict[str, str]]:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

