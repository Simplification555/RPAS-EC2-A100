# EC-1B RPAS native-fidelity audit

## Verdict

EC-1B now uses `experiments.phase2_wan_agent_search.run_search` as the sole
search controller. The LiveCodeBench module supplies task loading, public-test
tool execution, private evaluation, and metric conversion only. It does not
select parents, generate mutations, maintain a Pareto archive, shortlist
candidates, or choose the final operating point.

This is a native-algorithm task adaptation, not a byte-identical execution of
the historical source file. Reports must use that precise description.

## Historical source evidence

- Read-only SCIR snapshot:
  `/home/jianbaizhao/RPAS/experiments/phase2_wan_agent_search.py`
- Snapshot SHA-256:
  `1b045d2f7e36f2a577be6271893356b7bdf0b6406f5b8b031c1e2d137c300a2f`
- A local read-only comparison copy was verified against the same SHA-256;
  its machine-specific path is intentionally omitted from the artifact.

The SCIR snapshot has no `.git` directory, so no commit is claimed for it.
Its content hash is the provenance anchor.

## Function-level comparison

AST hashes ignore line numbers and formatting. The following functions are
identical between the historical snapshot and the current core:

| Function | AST SHA-256 | Status |
| --- | --- | --- |
| `seed_architectures` | `b6584b62aa9ec1218b24df6a4b5ff04d06e15950183c456b644cbe4276611a12` | identical |
| `pareto_front` | `fc7888b91ce6f8dd4a8bd399fcb5bcdc0f15f314435c1b86748c9a54c367f07e` | identical |
| `select_rows_for_test` | `1635c5d3bfe940c472109a850e04fb58345c93b14dee6114b5a82dfcd745178f` | identical |
| `shortlist_rows_for_selection` | `11cc3d17c270c1aef1c1c7dab05d6decb0e97da933a8d7bb7aecacf023b40c12` | identical |
| `select_operating_points` | `5739442699fa8252842a48898fe0c9482903e129d64c383c83f117cd06ae01b4` | identical |

The full AST hashes of `select_parent` and `mutate_candidate` differ because
the current repository also supports later AFlow-style and ADAS-style modes.
Their `wan_pareto` branches used by EC-1B are unchanged from the historical
snapshot.

## EC-1B adapter boundary

The current core exposes an optional `candidate_evaluator` argument on
`run_search`. When it is absent, all existing experiments continue through
the original cached evaluator. When present, only candidate scoring is
delegated; these operations remain in `run_search`:

1. seed architecture queue and candidate budget
2. candidate validity gates
3. Pareto or score-band parent selection
4. LLM reflection and typed mutation planning
5. duplicate and invalid proposal rejection
6. mutation observation feedback
7. Pareto shortlist on `D_search`
8. reevaluation on disjoint `D_select`
9. native Q/E operating-point selection
10. frozen-candidate evaluation on `D_test`

The injected evaluator enforces the benchmark privacy boundary: `D_search`
executes and scores only the frozen public cases. Private cases are not opened
or executed on that split, so they cannot enter RPAS reflection or mutation
feedback. The evaluator uses private cases only for disjoint `D_select` and
the frozen final `D_test`.

`allow_rule_fallback=false` is also enforced when the reflector returns no
applicable typed mutation. The run aborts instead of silently creating a
rule-based child.

## Frozen controls

- mode: `wan_pareto`
- reflection: LLM
- rule fallback: forbidden
- Pareto-parent probability: `0.5`
- parent quality band: `0.05`
- parent top-k: `6`
- model pool: singleton Qwen3.5-9B
- site pool: singleton allocation-local GPU
- decoding temperature: `0`
- maximum generation tokens: `2048`
- thinking: disabled
- split sizes: search `64`, selection `64`, held-out test `256`

The seed-0 run uses four seed architectures and three generated candidates.
It is a pilot and must retain `formal_result=false` until the preregistered
multi-seed confirmation and release gates are complete.
