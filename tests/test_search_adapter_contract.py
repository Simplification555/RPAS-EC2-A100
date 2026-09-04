from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from experiments.phase2_wan_agent_search import seed_architectures, select_parent, utility_score
from experiments.search_adapters.base import CandidateObservation
from experiments.search_adapters.fake import DeterministicFakeAdapter
from experiments.search_adapters.registry import build_adapter


def test_fake_adapter_receives_search_only() -> None:
    adapter = DeterministicFakeAdapter()
    adapter.initialize([], {"topologies": ["single"]}, random.Random(0))
    proposal = adapter.propose()
    adapter.observe(CandidateObservation(proposal.candidate_id, True, 1.0, 1, 1, 0.0))
    with pytest.raises(ValueError, match="search observations only"):
        adapter.observe(CandidateObservation(proposal.candidate_id, True, 1.0, 1, 1, 0.0, split="test"))


def test_fake_adapter_state_round_trip() -> None:
    first = DeterministicFakeAdapter()
    first.initialize([], {"topologies": ["single"]}, random.Random(4))
    proposal = first.propose()
    first.observe(CandidateObservation(proposal.candidate_id, True, 1.0, 1, 1, 0.0))

    second = DeterministicFakeAdapter()
    second.load_state_dict(first.state_dict())
    assert second.state_dict() == first.state_dict()


@pytest.mark.parametrize("method_id", ["random_as", "aflow_style", "adas_style", "rpas_quality", "rpas"])
def test_common_space_adapters_share_typed_contract(method_id: str) -> None:
    config = json.loads(open("experiments/phase2_wan_agent_config_qwen35_9b_homogeneous.json", encoding="utf-8").read())
    seeds = seed_architectures(config)
    adapter = build_adapter(method_id)
    adapter.initialize(seeds, config, random.Random(0))
    for seed in seeds:
        adapter.observe(CandidateObservation(seed["id"], True, 0.5, 1, 10, 0.0, diagnostics={"architecture": seed}))
    proposal = adapter.propose()
    assert proposal.candidate_id == proposal.architecture["id"]
    assert proposal.architecture["topology"] in config["allowed_topologies"]
    adapter.observe(
        CandidateObservation(
            proposal.candidate_id, True, 0.6, 1, 11, 0.0, diagnostics={"architecture": proposal.architecture}
        )
    )
    assert adapter.state_dict()["rows"][proposal.candidate_id]["score"] == 0.6


@pytest.mark.parametrize("mode", ["aflow_style", "adas_style"])
def test_formal_baseline_parent_policies_are_executable(mode: str) -> None:
    rows = [
        {
            "candidate_id": "a",
            "candidate": {"id": "a"},
            "score": 0.5,
            "avg_total_tokens": 100,
            "avg_calls": 1,
            "is_valid_candidate": True,
        },
        {
            "candidate_id": "b",
            "candidate": {"id": "b", "parent_id": "a"},
            "score": 0.6,
            "avg_total_tokens": 200,
            "avg_calls": 2,
            "is_valid_candidate": True,
        },
    ]
    selected, source = select_parent(
        rows,
        random.Random(0),
        mode,
        pareto_parent_prob=0.5,
        parent_score_band=0.05,
        parent_top_k=6,
    )
    assert selected["candidate_id"] in {"a", "b"}
    assert source in {"mcts_ucb", "meta_agent_quality_band"}
    assert utility_score(selected, mode) == selected["score"]


def test_legacy_launch_and_homogeneous_config_stay_within_gpu45() -> None:
    """Prevent non-EC launch paths from silently reopening retired GPU bindings."""
    runner = Path("experiments/run_formal_aime_track_a.sh").read_text(encoding="utf-8")
    assert 'GPU="${RPAS_CUDA_VISIBLE_DEVICES:-4}"' in runner
    assert '"${GPU}" != "4" && "${GPU}" != "5"' in runner

    config = json.loads(Path("experiments/phase2_wan_agent_config_qwen35_9b_homogeneous.json").read_text(encoding="utf-8"))
    model_names = set(config["models"])
    assert all("gpu6" not in name and "gpu7" not in name for name in model_names)
    assert all(not value["api_base"].startswith("http://127.0.0.1:801") for value in config["models"].values())
    defaults = config["defaults"]
    assert set(defaults["dag_worker_models"]).issubset(model_names)
    assert defaults["dag_aggregator_model"] in model_names
