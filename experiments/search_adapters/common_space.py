"""Controlled common-space search policies used by the external runner.

These policies own proposal generation only.  They never load data, execute a
candidate, inspect D_select/D_test, or select a reported operating point.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from experiments.phase2_wan_agent_search import (
    candidate_key,
    make_random_architecture,
    mutate_candidate,
    select_parent,
    with_candidate_id,
)
from experiments.search_adapters.base import CandidateObservation, Proposal


def _row_from_observation(observation: CandidateObservation, architecture: dict[str, Any]) -> dict[str, Any]:
    row = {
        "candidate_id": observation.candidate_id,
        "candidate": architecture,
        "topology": architecture.get("topology", ""),
        "score": observation.score if observation.score is not None else 0.0,
        "avg_calls": observation.total_calls,
        "avg_total_tokens": observation.total_tokens,
        "avg_inference_cost_usd": observation.total_cost,
        "avg_cross_center_tokens": observation.diagnostics.get("cross_center_tokens", 0),
        "is_valid_candidate": observation.valid,
    }
    return row


class CommonSpaceAdapter:
    """Small stateful adapter base with the frozen proposal contract."""

    method_id = "common_space"
    mutation_mode = "random"

    def __init__(self) -> None:
        self._rng = random.Random(0)
        self._config: dict[str, Any] = {}
        self._candidates: dict[str, dict[str, Any]] = {}
        self._rows: dict[str, dict[str, Any]] = {}
        self._proposal_count = 0

    def initialize(self, seed_archive: list[dict[str, Any]], search_space: dict[str, Any], rng: Any) -> None:
        self._config = copy.deepcopy(search_space)
        self._rng = rng
        self._candidates = {
            str(candidate["id"]): copy.deepcopy(candidate)
            for candidate in seed_archive
            if isinstance(candidate, dict) and candidate.get("id")
        }
        self._rows = {}
        self._proposal_count = 0

    def _parent_rows(self) -> list[dict[str, Any]]:
        return list(self._rows.values())

    def register_candidate(self, candidate: dict[str, Any]) -> None:
        """Restore a candidate before replaying its search observation."""

        candidate_id = str(candidate.get("id", ""))
        if not candidate_id:
            raise ValueError("cannot register a candidate without an id")
        self._candidates[candidate_id] = copy.deepcopy(candidate)

    def candidate(self, candidate_id: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._candidates[candidate_id])
        except KeyError as exc:
            raise KeyError(f"unknown candidate: {candidate_id}") from exc

    def _choose_parent(self) -> dict[str, Any]:
        rows = self._parent_rows()
        if not rows:
            return self._rng.choice(list(self._candidates.values()))
        return self._select_parent(rows)[0]["candidate"]

    def _select_parent(self, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        return select_parent(
            rows,
            self._rng,
            self.mutation_mode,
            pareto_parent_prob=0.0,
            parent_score_band=0.10,
            parent_top_k=4,
        )

    def _mutate(self, parent: dict[str, Any], mode: str | None = None) -> dict[str, Any]:
        return mutate_candidate(
            parent,
            self._config,
            self._rng,
            parent_row=next(
                (row for row in self._rows.values() if row["candidate_id"] == parent.get("id")),
                None,
            ),
            mode=mode or self.mutation_mode,
        )

    def _proposal(self, candidate: dict[str, Any], **metadata: Any) -> Proposal:
        candidate = with_candidate_id(candidate)
        self._candidates[str(candidate["id"])] = copy.deepcopy(candidate)
        self._proposal_count += 1
        return Proposal(
            candidate_id=str(candidate["id"]),
            architecture=candidate,
            metadata={"method_id": self.method_id, "proposal_index": self._proposal_count, **metadata},
        )

    def propose(self) -> Proposal:
        return self._proposal(make_random_architecture(self._config, self._rng), strategy="random")

    def observe(self, observation: CandidateObservation) -> None:
        if observation.split != "search":
            raise ValueError("adapters may observe search observations only")
        candidate = self._candidates.get(observation.candidate_id)
        if candidate is None:
            raise KeyError(f"unknown candidate observed: {observation.candidate_id}")
        self._rows[observation.candidate_id] = _row_from_observation(observation, candidate)

    def state_dict(self) -> dict[str, Any]:
        return {
            "proposal_count": self._proposal_count,
            "candidates": copy.deepcopy(self._candidates),
            "rows": copy.deepcopy(self._rows),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._proposal_count = int(state.get("proposal_count", 0))
        self._candidates = copy.deepcopy(state.get("candidates", {}))
        self._rows = copy.deepcopy(state.get("rows", {}))


class RandomASAdapter(CommonSpaceAdapter):
    method_id = "random_as"

    def propose(self) -> Proposal:
        return self._proposal(make_random_architecture(self._config, self._rng), strategy="uniform_random")


class AFlowStyleMCTSAdapter(CommonSpaceAdapter):
    """MCTS-style common-space policy with UCB parent selection."""

    method_id = "aflow_style"

    def __init__(self) -> None:
        super().__init__()
        self._visits: dict[str, int] = {}

    def _select_parent(self, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        total_visits = max(1, sum(self._visits.values()))

        def ucb(row: dict[str, Any]) -> float:
            candidate_id = str(row["candidate_id"])
            visits = self._visits.get(candidate_id, 0)
            mean = float(row.get("score", 0.0))
            exploration = math.sqrt(math.log(total_visits + 1) / (visits + 1))
            return mean + 0.35 * exploration

        return max(rows, key=ucb), "ucb"

    def propose(self) -> Proposal:
        parent = self._choose_parent()
        return self._proposal(self._mutate(parent, mode="random"), strategy="mcts_ucb")

    def observe(self, observation: CandidateObservation) -> None:
        super().observe(observation)
        self._visits[observation.candidate_id] = self._visits.get(observation.candidate_id, 0) + 1

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["visits"] = dict(self._visits)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._visits = {str(key): int(value) for key, value in state.get("visits", {}).items()}


class ADASStyleMetaAgentAdapter(CommonSpaceAdapter):
    """Meta-agent-style policy represented by an auditable deterministic heuristic.

    The common-space track does not claim to reproduce ADAS's native code.  A
    future controller can replace ``_choose_parent`` while preserving the
    runner contract and artifact schema.
    """

    method_id = "adas_style"

    def _select_parent(self, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        ranked = sorted(
            rows,
            key=lambda row: (
                -float(row.get("score", 0.0)),
                int(row.get("avg_total_tokens", 0)),
                str(row.get("candidate_id", "")),
            ),
        )
        return self._rng.choice(ranked[: max(1, min(4, len(ranked)))]), "meta_agent_quality_band"

    def propose(self) -> Proposal:
        parent = self._choose_parent()
        return self._proposal(self._mutate(parent, mode="quality_only"), strategy="meta_agent_heuristic")


class RPASQualityAdapter(CommonSpaceAdapter):
    method_id = "rpas_quality"
    mutation_mode = "quality_only"

    def propose(self) -> Proposal:
        parent = self._choose_parent()
        return self._proposal(self._mutate(parent, mode="quality_only"), strategy="reflection_quality_only")


class RPASAdapter(CommonSpaceAdapter):
    method_id = "rpas"
    mutation_mode = "wan_pareto"

    def _select_parent(self, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        return select_parent(
            rows,
            self._rng,
            "wan_pareto",
            pareto_parent_prob=0.5,
            parent_score_band=0.10,
            parent_top_k=4,
        )

    def propose(self) -> Proposal:
        parent_row, source = self._select_parent(self._parent_rows())
        reflection = {"mode": "rule", "diagnosis": []}
        child = mutate_candidate(
            parent_row["candidate"],
            self._config,
            self._rng,
            parent_row=parent_row,
            mode="wan_pareto",
            reflection_plan=reflection,
        )
        return self._proposal(child, strategy="failure_aware_pareto_mutation", parent_source=source)


ADAPTERS = {
    "random_as": RandomASAdapter,
    "aflow_style": AFlowStyleMCTSAdapter,
    "adas_style": ADASStyleMetaAgentAdapter,
    "rpas_quality": RPASQualityAdapter,
    "rpas": RPASAdapter,
}


def build_common_space_adapter(method_id: str) -> CommonSpaceAdapter:
    try:
        return ADAPTERS[method_id]()
    except KeyError as exc:
        raise ValueError(f"common-space adapter is not registered: {method_id}") from exc
