# RPAS Native Core Provenance Audit

## Authoritative source

The read-only RPAS source used for this audit is:

```text
$RPAS_ORIGINAL_ROOT/experiments/phase2_wan_agent_search.py
SHA-256: 1b045d2f7e36f2a577be6271893356b7bdf0b6406f5b8b031c1e2d137c300a2f
```

The duplicate top-level source in the original bundle has the same hash. The
original bundle has no usable Git metadata, so this content hash is the
authoritative provenance identifier.

## Native control flow

The original `run_search()` owns the complete Phase-2 method:

1. native seed architecture queue;
2. candidate contract and execution validity gates;
3. `select_parent()` with Pareto/front and top-score-band sampling;
4. LLM reflection and planned typed mutations;
5. mutation observation and proposal journaling;
6. `shortlist_rows_for_selection()` on `D_search`;
7. reevaluation on disjoint `D_select`;
8. `select_operating_points()` for frozen quality and efficiency points;
9. terminal evaluation after selection is frozen.

The external benchmark integration must not reproduce those steps in an
adapter loop. It may inject only a task evaluator and frozen task fixtures.

## Repository extension boundary

`experiments/phase2_wan_agent_search.py` retains the original controller and
adds a single optional `candidate_evaluator` boundary. When the hook is not
provided, the original `evaluate_candidate_cached()` path is unchanged. When
provided, search, parent selection, reflection, mutation, shortlisting, and
operating-point selection remain inside `run_search()`.

When rule fallback is disabled, an empty, duplicate, or invalid LLM mutation
plan triggers another native parent-selection/reflection attempt. The retries
are bounded and journaled; exhausting the bound fails closed. No rule-based
mutation is introduced by this recovery path.

## Benchmark bindings

| Experiment | Native entry point | External-only responsibility |
|---|---|---|
| EC-1B LiveCodeBench | `phase2_wan_agent_search.run_search` | typed LCB prompt, public/private execution evaluator, telemetry |
| EC-3 HotpotQA | `phase2_wan_agent_search.run_search` | provided-context F1 evaluator, telemetry, held-out lock |

EC-3 passes disjoint `D_calib` as the native terminal validity split during
pre-test execution. The frozen `D_test` file is not opened until the external
final-state gate is satisfied. The quality/efficiency selection is therefore
frozen by native RPAS before any held-out access.

## Fidelity assertions

Every new RPAS-Full artifact must record:

- original core SHA-256 and executed extended-core SHA-256;
- `native_control_flow=experiments.phase2_wan_agent_search.run_search`;
- `mode=wan_pareto` and LLM reflection with zero rule fallbacks;
- positive evaluated new-candidate and typed-mutation evidence;
- a constructed non-empty Pareto front;
- native shortlist policy and both Q/E candidate identifiers;
- frozen split/config/model provenance and complete call/token/time telemetry.

Pareto-front cardinality is not an outcome-independent fidelity condition. A
valid generated candidate may be dominated, so the final front need not be
larger than the seed-only front.
