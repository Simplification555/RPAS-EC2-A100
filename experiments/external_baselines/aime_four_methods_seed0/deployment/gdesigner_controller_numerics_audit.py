#!/usr/bin/env python3
"""Synthetic CPU audit; no model requests or benchmark row reads.

Run with the AIME GDesigner interpreter and the AIME package on PYTHONPATH.
Only in-memory Graph instances are changed. --full reproduces the previously
observed 20-dropout-seed comparisons; default runs only semantic mapping seed 0.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dense_to_sparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, default=Path("/data/jxc/models/all-MiniLM-L6-v2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(args.package / "experiments"), str(args.package / "upstream/GDesigner")]
    from native_external_methods import _register_gdesigner_prompt
    import GDesigner.prompt.gsm8k_prompt_set as native_prompt
    import GDesigner.agents.math_solver
    import GDesigner.agents.final_decision
    import GDesigner.graph.graph as graph_module
    from GDesigner.graph.graph import Graph

    torch.set_num_threads(1)
    model = SentenceTransformer(str(args.embedding), device="cpu")
    graph_module.get_sentence_embedding = lambda text: model.encode(text, convert_to_numpy=True)
    prompt_registration = _register_gdesigner_prompt()
    aime_roles = ["Algebraic Decomposer", "Independent Solver", "Consistency Checker", "Solution Synthesizer"]
    native_roles = ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"]
    mapping = {
        "Math Solver": "Independent Solver",
        "Mathematical Analyst": "Algebraic Decomposer",
        "Programming Expert": "Solution Synthesizer",
        "Inspector": "Consistency Checker",
    }
    semantic_edges = [(mapping[left], mapping[right]) for left, right in native_prompt.ROLE_CONNECTION]

    def make_graph(domain, node_roles, edges):
        torch.manual_seed(0)
        graph = Graph(
            domain=domain, llm_name="GPTChat", agent_names=["MathSolver"] * 4,
            decision_method="FinalRefer", optimized_spatial=True, optimized_temporal=False,
            fixed_spatial_masks=[[0 if i == j else 1 for j in range(4)] for i in range(4)],
            fixed_temporal_masks=[[0] * 4 for _ in range(4)],
            node_kwargs=[{"role": role} for role in node_roles],
        )
        indices = {role: i for i, role in enumerate(node_roles)}
        adjacency = torch.zeros(4, 4)
        for left, right in edges:
            adjacency[indices[left], indices[right]] = 1
        desired_edges = dense_to_sparse(adjacency)[0]
        if domain == "rpas_external" and edges == semantic_edges:
            assert torch.equal(graph.role_adj_matrix, desired_edges), "working shared helper must create the audited native semantic role graph"
        graph.role_adj_matrix = desired_edges
        return graph

    def probe(graph, seed):
        torch.manual_seed(seed)
        current = copy.deepcopy(graph)
        current.gcn.train()
        optimizer = torch.optim.Adam(current.gcn.parameters(), lr=0.1)
        before = {key: value.detach().clone() for key, value in current.gcn.state_dict().items()}
        features = current.construct_new_features("Compute 2 + 3. Return FINAL ANSWER: 5.")
        node_output = current.gcn(features, current.role_adj_matrix)
        vectors = current.mlp(node_output)
        gram = (vectors @ vectors.t()).flatten()
        span = gram.max() - gram.min()
        normalized = gram * 0.0 if float(span.detach()) == 0.0 else (gram - gram.min()) / span * 2 - 1
        current.spatial_logits = normalized
        log_prob = current.construct_spatial_connection()
        loss = -log_prob  # synthetic positive REINFORCE reward = 1
        optimizer.zero_grad()
        loss.backward()
        gradients = [parameter.grad for parameter in current.gcn.parameters() if parameter.grad is not None]
        gradient_l1 = sum(float(gradient.abs().sum()) for gradient in gradients)
        gradient_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        optimizer.step()
        parameter_delta = max(float((value - before[key]).abs().max()) for key, value in current.gcn.state_dict().items())
        return {
            "seed": seed, "reward": 1.0, "gcn_row_max_difference": float((node_output - node_output[0]).detach().abs().max()),
            "gram_span": float(span.detach()), "normalized_finite": bool(torch.isfinite(normalized).all()),
            "loss": float(loss.detach()), "gcn_gradient_finite": gradient_finite,
            "gcn_gradient_l1": gradient_l1, "gcn_parameter_max_delta": parameter_delta,
            "changed_parameter_tensors": sum(not torch.equal(value, before[key]) for key, value in current.gcn.state_dict().items()),
        }

    def scenario(domain, node_roles, edges, seeds):
        graph = make_graph(domain, node_roles, edges)
        edge, weight = gcn_norm(graph.role_adj_matrix, num_nodes=4, add_self_loops=True, dtype=torch.float32)
        matrix = torch.zeros(4, 4)
        matrix[edge[1], edge[0]] = weight
        return {
            "role_edges": len(edges), "role_connection": edges, "node_roles": node_roles,
            "normalized_propagation_matrix": matrix.tolist(), "propagation_rank": int(torch.linalg.matrix_rank(matrix)),
            "trials": [probe(graph, seed) for seed in seeds],
        }

    report = {
        "schema": "gdesigner_synthetic_cpu_controller_audit_v1", "model_calls": 0, "dtest_opened": False,
        "upstream_commit": "a6efcfa3b40bb4d9cbf46f883a95d62020bd8251", "torch": torch.__version__,
        "float_dtype": "float32", "mapping": mapping, "prompt_registration": prompt_registration,
        "synthetic_query": "Compute 2 + 3. Return FINAL ANSWER: 5.",
        "normalization": "unchanged min/max when span > 0; exact span == 0 maps to connected zero logits",
        "optimizer": "native Adam(graph.gcn.parameters(), lr=0.1)", "scenarios": {},
    }
    if args.full:
        report["scenarios"]["aime_complete_12"] = scenario("rpas_external", aime_roles, [(a, b) for a in aime_roles for b in aime_roles if a != b], range(20))
        report["scenarios"]["upstream_native_9"] = scenario("gsm8k", native_roles, native_prompt.ROLE_CONNECTION, range(20))
    report["scenarios"]["aime_semantic_native_9"] = scenario("rpas_external", aime_roles, semantic_edges, [0])
    semantic = report["scenarios"]["aime_semantic_native_9"]["trials"][0]
    assert semantic["normalized_finite"] and semantic["gcn_gradient_finite"]
    assert semantic["gram_span"] > 0 and semantic["gcn_gradient_l1"] > 0 and semantic["gcn_parameter_max_delta"] > 0
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
