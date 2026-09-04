from __future__ import annotations

import pytest

from external_comparison.runners.aggregate_mmlu_v2 import _load_seed
from external_comparison.runners.ec2_v2 import (
    AGENT_COUNT,
    BACKBONE,
    EC2_V2_PROTOCOL,
    MAX_TOKENS,
    ROLES,
    ROUNDS,
    _rpas_search_assignments,
    assert_v2_candidate,
    communication_candidate,
    topology_mask,
    validate_v2_manifest,
)
from external_comparison.runners.mmlu import MMLUExample


def _manifest(method: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_version": EC2_V2_PROTOCOL,
        "method": method,
        "backbone": BACKBONE,
        "agent_count": AGENT_COUNT,
        "roles": list(ROLES),
        "communication_rounds": ROUNDS,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "final_aggregator": "GDesigner.FinalRefer",
        "message_policy": "verbatim_official_gdesigner",
        "split_protocol": "dev_search__val_select__test_heldout",
    }
    if method == "gdesigner":
        payload.update(
            {
                "search_calls": 40,
                "search_tokens": 100,
                "gdesigner_training": {
                    "iterations": 10,
                    "initial_gcn_sha256": "a" * 64,
                    "trained_gcn_sha256": "b" * 64,
                },
            }
        )
    if method == "rpas_comm":
        payload.update(
            {
                "search_calls": 40,
                "search_tokens": 100,
                "rpas_reflection": {"reflection_calls": 1, "new_candidates": 1, "mutation_logs": 1, "rule_fallbacks": 0},
            }
        )
    return payload


def test_ec2_v2_topology_space_is_six_agent_dag() -> None:
    full = topology_mask("full_connected")
    chain = topology_mask("chain")
    assert len(full) == AGENT_COUNT
    assert sum(map(sum, full)) == 15
    assert sum(map(sum, chain)) == 5
    assert all(full[target][source] == 0 for source in range(AGENT_COUNT) for target in range(source + 1, AGENT_COUNT))


def test_ec2_v2_candidate_cannot_change_shared_execution_space() -> None:
    candidate = communication_candidate("chain")
    assert_v2_candidate(candidate)
    candidate["roles"] = ["other"] * AGENT_COUNT
    with pytest.raises(ValueError, match="role pool"):
        assert_v2_candidate(candidate)
    candidate = communication_candidate("chain")
    candidate["compression"] = "summary"
    with pytest.raises(ValueError, match="compression"):
        assert_v2_candidate(candidate)
    candidate = communication_candidate("chain")
    candidate["topology"] = "single"
    with pytest.raises(ValueError, match="may not leave"):
        assert_v2_candidate(candidate)


def test_ec2_v2_formal_manifest_gates() -> None:
    validate_v2_manifest(_manifest("gdesigner"))
    validate_v2_manifest(_manifest("rpas_comm"))
    single = _manifest("single_agent")
    single["agent_count"] = 1
    single["roles"] = [ROLES[0]]
    validate_v2_manifest(single)

    invalid_gdesigner = _manifest("gdesigner")
    training = invalid_gdesigner["gdesigner_training"]
    assert isinstance(training, dict)
    training["trained_gcn_sha256"] = training["initial_gcn_sha256"]
    with pytest.raises(ValueError, match="did not change"):
        validate_v2_manifest(invalid_gdesigner)

    invalid_rpas = _manifest("rpas_comm")
    invalid_rpas["rpas_reflection"] = {"reflection_calls": 0, "new_candidates": 0, "mutation_logs": 0, "rule_fallbacks": 0}
    with pytest.raises(ValueError, match="reflection"):
        validate_v2_manifest(invalid_rpas)


def test_rpas_search_assignments_match_official_training_query_budget() -> None:
    rows = [MMLUExample(str(index), "subject", "q", ("a", "b", "c", "d"), "A") for index in range(20)]
    assignments = _rpas_search_assignments(rows, candidate_count=4, seed=0)
    assert [len(assignment) for assignment in assignments] == [10, 10, 10, 10]
    assert sum(map(len, assignments)) == 40


def test_ec2_v2_aggregator_rejects_legacy_artifact(tmp_path) -> None:
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "run_manifest.json").write_text('{"protocol_version": "legacy"}', encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        _load_seed(run)
