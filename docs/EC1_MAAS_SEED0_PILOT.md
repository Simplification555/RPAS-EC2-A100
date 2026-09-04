# EC-1 MaAS Seed-0 Pilot Delivery

## Status

This is a completed **pilot**, not a formal paper result.

- Method: MaAS, fresh random `MultiLayerController` followed by the official
  train/search branch and official held-out test branch.
- Upstream: `bingreeky/MaAS@987f3c1bc9a96e844fe090db3791446e3ef0f5c7`.
- Executor: frozen `Qwen/Qwen3.5-9B`, temperature `0`, max completion tokens
  `1024`, physical GPU `5`.
- Split: AFlow-derived frozen HumanEval `33` search tasks and `131` held-out
  test tasks. Hashes are recorded in the result manifest.
- Final score: `125 / 131 = 0.9541984733` Pass@1.
- Search accounting: `843` calls, `568,945` tokens, `9,329.455` seconds.
- Held-out inference accounting: `821` calls, `573,162` tokens, zero recorded
  model errors.

The pilot confirms the native fresh-train path functions end-to-end. It does
not establish a three-seed mean, a matched-budget comparison, or a main-table
claim. Do not compare this number against a search validation score.

## Delivered Artifact

`handoff_progress/ec1_seed_0/maas/completed_pilot/` contains the sanitized
result manifest, detailed test outputs, normalized telemetry, search record,
and CSV summary. Absolute paths were replaced by `$REPO_ROOT/`; no metric,
task ID, model output, call count, or hash was altered. The original local
workspace, log, cache, and credentials are not included.

The compact controller state at the parent directory remains available for
recovery and has SHA-256:

```text
cc51e468a6fce8c32053fff83de459c0e1d3f43c5adf038b3b0636e2e7f40bdf
```
