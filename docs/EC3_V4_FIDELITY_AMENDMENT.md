# EC-3 V4 Fidelity Amendment

This amendment versions the EC-3 HotpotQA runner before any new formal
pretest or held-out execution. It does not modify, overwrite, reseal, or
relabel existing artifacts. Existing pretest directories remain historical
artifacts and cannot unlock `D_test` under this amendment.

## Scope

- The benchmark remains HotpotQA with the frozen `D_calib=40`,
  `D_search=120`, `D_select=80`, and `D_test=800` splits.
- The AFlow search core remains upstream-owned.
- The RPAS search core continues to use LLM reflection, typed mutation,
  candidate validity gates, and `wan_pareto`; no scoring objective, mutation
  rule, or model pool was changed.

## Corrected Execution Contract

All new EC-3 stages require executor `max_tokens=512` and meta/reflection
`max_tokens=4096`. The values are checked in the Slurm scripts, the RPAS
runner, the AFlow runner, and the final-state gate, and are recorded in the
environment and run manifests.

## Restored RPAS Selection

The external runner now calls the repository's native
`shortlist_rows_for_selection(..., mode="wan_pareto",
selection_strategy="quality_band_cost")` and
`select_operating_points()` functions. HotpotQA `answer_f1` is explicitly
recorded as the native selection `score`. Both quality and efficiency
operating points are written and hashed; held-out RPAS evaluation uses only
the frozen quality point.

## D_test Gate

Each of the six AFlow/RPAS seed states must have the same split hash and
non-zero pretest telemetry. RPAS states additionally require evidence of LLM
reflection, typed mutations, a constructed non-empty Pareto front, native shortlist
policy, and both operating points. The gate rejects missing or altered
evidence before it can create `d_test_unlock.json`.

The Pareto-front gate is structural rather than outcome-directed: a valid
mutation may be dominated, so the final front is not required to be larger
than the seed-only front.

This is a fidelity correction, not an outcome-directed adjustment: no
held-out output is used by search, selection, calibration, or this amendment.
