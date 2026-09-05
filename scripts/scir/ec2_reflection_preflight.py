"""Validate a real topology-reflector call on synthetic, non-benchmark evidence."""

import argparse
import json
from pathlib import Path

from external_comparison.runners.ec2_v2 import _llm_topology_mutation, _reflection_context, communication_candidate


def check(repo_root: Path) -> dict:
    candidate = communication_candidate("full_connected")
    parent = {
        "candidate": candidate, "candidate_id": candidate["id"], "score": 0.5,
        "avg_calls": 7, "avg_total_tokens": 900, "avg_inter_agent_tokens": 600,
        "excluded_candidate_ids": [communication_candidate(name)["id"] for name in ("full_connected", "chain", "star")],
        "failure_examples": [{"id": "synthetic-only:0", "input": "Which integer is even? A: 2; B: 3; C: 5; D: 7",
                              "gold_answer": "A", "prediction": "B", "final_output_excerpt": "B"}],
    }
    config, models, profile = _reflection_context(repo_root)
    child, plan = _llm_topology_mutation(parent, config=config, models=models, profile=profile)
    assert child["topology"] == "layered"
    return {"formal_result": False, "synthetic_only": True, "status": "PASS_REAL_REFLECTION_ONLY",
            "selected_topology": child["topology"], "plan": plan}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preflight output must be new; never overwrite prior evidence")
    payload = check(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in payload.items() if key != "plan"}))
