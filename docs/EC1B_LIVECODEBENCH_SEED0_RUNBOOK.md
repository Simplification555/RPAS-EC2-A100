# EC-1B LiveCodeBench release-v6 seed-0 runbook

This track implements the hard-code stress test in the V4 external benchmark
protocol. It replaces neither the historical HumanEval EC-1A artifacts nor
their provenance. Its only paper-safe status is a one-seed pilot until the
confirmatory multi-seed gate is separately completed.

## Frozen benchmark

- Dataset: `livecodebench/code_generation_lite`
- Revision: `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`
- Builder config: `release_v6`
- Official cumulative shards: `test.jsonl` through `test6.jsonl`
- Scenario: code generation
- Data seed: `2026`
- Split: calibration 20, search 64, selection 64, held-out test 256

The preparation script writes the exact problem IDs and every split SHA-256 to
`DATASET_MANIFEST.json`. It refuses to overwrite an existing bundle. Private
tests remain inside the evaluator and are redacted from RPAS reflection
payloads. Search, selection, and test IDs must be pairwise disjoint.

```bash
PYTHONPATH="$PWD" ./.uv-linux run --no-project --offline \
  --python /home/jianbaizhao/qwen_cuda126/bin/python \
  -m scripts.prepare_ec1_livecodebench \
  --source /home/jianbaizhao/.cache/rpas_lcb_source/test.jsonl \
           /home/jianbaizhao/.cache/rpas_lcb_source/test2.jsonl \
           /home/jianbaizhao/.cache/rpas_lcb_source/test3.jsonl \
           /home/jianbaizhao/.cache/rpas_lcb_source/test4.jsonl \
           /home/jianbaizhao/.cache/rpas_lcb_source/test5.jsonl \
           /home/jianbaizhao/.cache/rpas_lcb_source/test6.jsonl \
  --output-dir data/ec1_livecodebench_release_v6
```

## Method fidelity

`Single` is one direct deterministic Qwen call per held-out item. It has no
search and no public-test repair.

`AFlow` invokes the pinned upstream `Optimizer.optimize("Graph")`. The adapter
only supplies the LiveCodeBench task signature, stdin/stdout code-generation
contract, frozen data files, and the shared public-test evaluator. Search
workflows are reevaluated on `D_select`; the shared operating-point selector
chooses quality and efficiency candidates without consulting `D_test`.

`RPAS` calls the repository Phase-2 implementation for seed architectures,
candidate execution, Pareto parent sampling, LLM reflection, typed mutation,
Pareto shortlist, and operating-point selection. Its frozen settings are
`mode=wan_pareto`, `reflection_mode=llm`, no rule fallback,
`pareto_parent_prob=0.5`, `parent_score_band=0.05`, and `parent_top_k=6`.
The LCB runner does not reimplement these search decisions.

The seed-0 compute budget is deliberately a pilot budget of four seed
architectures and three new RPAS candidates. It must not be described as the
larger confirmatory budget from the main RPAS protocol.

## Submit

Run the fail-closed preflight before Slurm submission. The array requests one
A100 40 GB per method and allows all three methods to run concurrently.

```bash
PYTHONPATH="$PWD" ./.uv-linux run --no-project --offline \
  --python /home/jianbaizhao/qwen_cuda126/bin/python \
  -m external_comparison.runners.livecodebench_preflight \
  --data-dir data/ec1_livecodebench_release_v6 \
  --aflow-root /home/jianbaizhao/external_baselines/AFlow_formal_clean

sbatch scripts/scir/ec1_livecodebench_seed0.sbatch
```

Each method writes a run manifest, per-item outputs, model-call telemetry,
token totals, timing metadata, selection/search rows, and SHA-256 evidence.
The Slurm wrapper rejects wrong release provenance, wrong seed, overwritten
results, duplicate jobs, and any artifact that incorrectly claims formal
three-seed status.
