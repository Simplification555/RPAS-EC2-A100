# EC-3 HotpotQA V4F Seed-0 Gate Artifact

Status: `formal_result=false`; diagnostic one-seed gate result; `D_test` was
not opened.

This directory preserves the clean runs produced under code baseline
`447ea58` and split manifest
`1940fbef3792b691f60fe56c604d8e12f81e8f5ac71f444a7239462c0f3dd54c`.

## Result matrix

| Method | D_search | D_select | D_calib | D_select F1 / EM | D_calib F1 / EM | Calls | Tokens | Runner wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AFlow | 8 | 8 | 40 | 0.5596 / 0.5000 | 0.4516 / 0.4000 | 89 | 106,707 | 811.99 s |
| RPAS-Full | 8 | 8 | 40 | 0.7083 / 0.6250 | 0.8938 / 0.8750 | 166 | 267,945 | 989.74 s |

RPAS additionally required the clean distinct-candidate calibration job before
the pilot: 160 calls, 247,415 tokens, and 10:55 Slurm elapsed. Therefore its
observed end-to-end chain cost was 326 calls and 515,360 tokens. This overhead
is not hidden in the 166-call pilot row.

## Fidelity checks

- RPAS used repository-native `phase2_wan_agent_search.run_search`.
- Search created three accepted typed mutations and used six LLM reflection calls.
- Rule fallback count is zero; model error count is zero.
- Q and E were frozen before the calibration substitute.
- Both manifests record `d_test_accessed=false` and `formal_result=false`.
- Job-root SHA manifests for jobs 133521, 133531, and 133532 passed on SCIR.

The RPAS Pareto list has three raw entries but only two unique canonical IDs.
The duplicate is caused by model/site-equivalent native seed workflows under
the frozen singleton model and site pools. It does not change the selected Q/E
candidate, but the artifact must not be described as three distinct Pareto
candidates.

## Directory layout

- `aflow_clean_pilot/`: calls, selection, selected workflow, manifest, metrics.
- `rpas_clean_calibration/`: two distinct calibration candidates and calls.
- `rpas_clean_pilot/`: native search, reflection, mutation, Pareto, Q/E, calls.
- `job_audit/`: compact environment, command, status, health, and SCIR SHA records.
- `summary.json`: machine-readable audited summary.
- `SOURCE_SHA256SUMS`: hashes of copied SCIR files before path-only sanitization.
- `SHA256SUMS`: portable checksums for the complete handoff directory.

Public copies replace machine-specific absolute paths with `$REPO_ROOT`,
`$AFLOW_ROOT`, `$MODEL_ROOT`, and `$PYTHON_ENV`. No score, output, token count,
candidate, trace, command argument, or source code is changed by sanitization.

The saturation rule is triggered because Single/RPAS calibration F1 is 0.8938
and neither searched method establishes an improvement above 0.02. The V4
protocol therefore keeps HotpotQA `D_test` locked and requires a written
MuSiQue amendment or other explicit protocol decision.
