"""Deterministic fake adapter used to test budget and leakage boundaries."""

from __future__ import annotations

import random
from typing import Any

from experiments.search_adapters.base import CandidateObservation, Proposal


class DeterministicFakeAdapter:
    """Produces reproducible typed proposals without a model or API call."""

    method_id = "fake"

    def __init__(self) -> None:
        self._rng = random.Random(0)
        self._next_id = 0
        self._observations: list[CandidateObservation] = []
        self._search_space: dict[str, Any] = {}

    def initialize(self, seed_archive: list[dict[str, Any]], search_space: dict[str, Any], rng: Any) -> None:
        del seed_archive
        self._search_space = dict(search_space)
        self._rng = rng

    def propose(self) -> Proposal:
        candidate_id = f"fake-{self._next_id}"
        self._next_id += 1
        topologies = self._search_space.get("topologies", ["single"])
        topology = topologies[self._next_id % len(topologies)]
        return Proposal(candidate_id=candidate_id, architecture={"topology": topology, "nonce": self._rng.randint(0, 10**6)})

    def observe(self, observation: CandidateObservation) -> None:
        if observation.split != "search":
            raise ValueError("adapters may observe search observations only")
        self._observations.append(observation)

    def state_dict(self) -> dict[str, Any]:
        return {"next_id": self._next_id, "observations": [item.__dict__ for item in self._observations]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._next_id = int(state["next_id"])
        self._observations = [CandidateObservation(**item) for item in state["observations"]]
