# EC-3 V4F Clean One-Seed Gate Result

Status: completed diagnostic gate; `formal_result=false`; no HotpotQA
`D_test` access.

## Audited result

| Method | Search/select/calib | D_select F1 | D_calib F1 | Calls | Tokens | SCIR job |
|---|---:|---:|---:|---:|---:|---:|
| AFlow | 8 / 8 / 40 | 0.5596 | 0.4516 | 89 | 106,707 | 133521 |
| RPAS-Full pilot only | 8 / 8 / 40 | 0.7083 | 0.8938 | 166 | 267,945 | 133532 |
| RPAS prerequisite calibration | - / - / 40 x 2 candidates | - | 0.8938 best | 160 | 247,415 | 133531 |

RPAS end-to-end observed cost, including its clean prerequisite calibration,
is 326 calls and 515,360 tokens. The runner wall times are 811.99 seconds for
AFlow and 989.74 seconds for the RPAS pilot; RPAS prerequisite calibration used
an additional 10:55 Slurm elapsed.

## Protocol evidence

Both methods used Qwen3.5-9B FP16, temperature 0, disabled thinking, executor
cap 512, meta cap 4096, the same frozen first-eight search/select fixtures, and
the complete 40-item calibration split. The split manifest SHA-256 is
`1940fbef3792b691f60fe56c604d8e12f81e8f5ac71f444a7239462c0f3dd54c`.

RPAS produced three accepted typed mutations from six LLM reflection calls,
constructed the Pareto/Q/E records, selected candidate `45c0167f778d`, used no
rule fallback, and recorded no model errors. AFlow recorded one optimizer call,
one generated workflow round, and no model errors.

The RPAS Pareto front contains three raw entries but two unique canonical IDs.
The duplicate comes from native seed workflows that collapse under the frozen
singleton model/site pools. This is disclosed rather than rewritten after the
run; selected Q/E is unchanged.

## Interpretation

This result proves the clean infrastructure and native RPAS control path are
executable. It does not estimate held-out generalization because `D_test` was
never opened, and the 8-item search/select fixtures are too small for a method
ranking claim.

The registered saturation gate is triggered: Single/RPAS calibration F1 is
0.8938 and no searched method establishes an improvement greater than 0.02.
Consequently HotpotQA `D_test` remains locked. Continuing EC-3 requires a
written MuSiQue amendment or an explicit protocol decision; silently testing
on HotpotQA would violate V4.

The portable artifact bundle is under `handoff_progress/ec3_seed_0/`.
