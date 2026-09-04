"""Explicit registry for controlled-track adapter names."""

from __future__ import annotations

from experiments.search_adapters.fake import DeterministicFakeAdapter
from experiments.search_adapters.common_space import build_common_space_adapter


def build_adapter(method_id: str):
    if method_id == "fake":
        return DeterministicFakeAdapter()
    if method_id in {"random_as", "aflow_style", "adas_style", "rpas_quality", "rpas"}:
        return build_common_space_adapter(method_id)
    raise ValueError(f"adapter is not registered: {method_id}")
