# RTX PRO 6000 MASBench seed-0 results — 2026-10-14

This archive contains all experiment evidence produced in the RTX PRO 6000 session: the completed ten-unit AFlow/MaAS batch, its failed strict-gate predecessor, the depth retry, and the earlier aborted attempt. The final batch completed at **2026-10-14 17:22:10 Asia/Shanghai**, after approximately 5 hours 2 minutes for the resumed selection/test batch. Search computation had already completed in the preceding run and is charged through the recorded search-resume provenance.

**Result scope: exploratory completion with relaxed quality gates.** All ten units completed their 24-example selection and 60-example test evaluations and passed the retained completeness/integrity checks. **None of the ten passed the original strict quality gate.** Failed or timed-out test examples remain in the 60-example denominator. These results must not be described as ten strictly passing formal runs or as an A100 measurement.

## Final test results

The table reports exact match at the quality operating point (Q). E has the same exact-match scores in this batch, but AFlow robustness selects a different E workflow with different component accuracy and token cost; consult the original aggregate for both points.

| Axis | AFlow exact match | MaAS exact match |
|---|---:|---:|
| breadth | 21/60 (35.00%) | 19/60 (31.67%) |
| depth | 18/60 (30.00%) | 15/60 (25.00%) |
| horizon | 9/60 (15.00%) | 7/60 (11.67%) |
| parallel | 7/60 (11.67%) | 10/60 (16.67%) |
| robustness | 1/60 (1.67%) | 0/60 (0.00%) |

Five-axis macro-average exact match: **AFlow 18.67%; MaAS 17.00%**. Q token cost per example averages **8,182.71 for AFlow** and **54,522.00 for MaAS**. These are observed D_test model-token counts, not USD costs; search usage is recorded separately in each run’s telemetry and resume provenance; cancelled dispatched requests without returned usage make affected cost totals lower bounds. This is one search seed, so no across-seed uncertainty or significance claim is available.

- [Original aggregate JSON](outputs/aflow_maas_completion_20261014/seed0_summary/aggregate.json) and [CSV](outputs/aflow_maas_completion_20261014/seed0_summary/aggregate.csv).
- [Per-axis failure and repair summary](failure_summary.csv).
- [Completed batch status](outputs/aflow_maas_completion_20261014/batch_status.json).

## Included evidence

| Path | Evidence |
|---|---|
| `outputs/aflow_maas_completion_20261014/` | Final 10-unit exploratory run, Q/E results, frozen selections, per-run quality gates, original native CSVs, all model-request ledgers and answer-repair traces, generated workflows/prompts and selected native controller checkpoints. |
| `outputs/aflow_maas_9b_ctx16k_seed0_pro6000_20261013/` | Original search and strict-gate attempt across both methods and all axes. AFlow had no eligible D_select candidate; most MaAS units stopped before D_test. MaAS breadth reached D_test (23/60 exact match) and failed the quality gate. Missing test results in these interrupted runs are not zero-accuracy measurements. |
| `outputs/retry_maas_depth_20261014d/` | Earlier MaAS depth retry: D_select 12/24 exact match, failed the strict quality gate, no D_test evaluation. |
| `archived_attempts/` | First aborted attempt and the code backup from before completion-policy integration. |
| `supporting_records/` | Original launcher/method/service logs, preflight report, hardware snapshot, frozen model revisions and weight hashes, dependency locks and earlier validation log. |
| `reproduction/` | Exact completed-run runner/scorer/context/repair source snapshots, local launch/setup/proxy scripts and regression tests. These are historical snapshots, not replacements for the repository's A100 configuration. |
| `dataset_provenance/` | Frozen dataset manifests and split hashes. Original benchmark texts and expected answers are also present in the native evaluator CSV evidence. |
| `artifact_manifest.json` | Inventory mapping each original package-relative filename to its exported filename, encoding, byte length and SHA-256 before/after compression. |

Raw `.jsonl`/`.log` files and large native CSV/JSON records are losslessly gzip-compressed. JSON manifests, scores and selection/quality decisions remain readable. Duplicate upstream baseline installations, unrelated upstream fixtures, bytecode and Git metadata are filtered; all generated MASBench workspaces and result records are retained. Foundation-model weights and virtual environments are represented by revisions/hashes and dependency records rather than copied. The two empty retry directories are recorded in the inventory. The original RTX5880 handoff packages produced no additional experiment results in this session and are not duplicated here.

Original result JSON is preserved byte-for-byte, including historical local absolute paths and failed statuses. The verifier resolves these paths via the artifact inventory. Do not run the original path-dependent quality gate against this relocated archive and mistake missing local paths for an experiment failure. Known syntax errors in generated candidate `.py` files are retained as evidence and must not be imported or treated as production source.

## Hardware, protocol and recovery

- RTX PRO 6000 Blackwell Workstation Edition, 96 GB class, driver 596.49; vLLM 0.18.1, serving Torch 2.10/CUDA 12.8, Python 3.12.15. MaAS control/embedding processes ran on CPU.
- Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16, TP=1, context 16,384, native output cap 6,144, thinking disabled, shared transport concurrency 16. Native role temperatures/overrides are retained. All completed requests require returned token usage and use the serving tokenizer/context audit.
- Dataset seed 2026, search seed 0, five MASBench axes, frozen 24/24/60 search/select/test split sizes. Each selection lock precedes that resumed run's D_test access.
- Native baselines ([AFlow](https://github.com/FoundationAgents/AFlow/tree/3f457218fc716093fe53f6df8a5d5e6379d66346) / [MaAS](https://github.com/bingreeky/MaAS/tree/987f3c1bc9a96e844fe090db3791446e3ef0f5c7)) are pinned to AFlow `3f457218fc716093fe53f6df8a5d5e6379d66346` and MaAS `987f3c1bc9a96e844fe090db3791446e3ef0f5c7`; baseline source hashes remain in run manifests.
- Reused original searches/controller checkpoints, without importing old selection/test scores. Original search calls/tokens are charged to the resumed unit. Historical test exposure is recorded (including the predecessor MaAS breadth run); this prevents a pristine confirmatory claim for the overall batch.
- Strict numerical quality thresholds were changed to recorded warnings for this explicitly requested completion run. Dataset/run identities, sample completeness, selection freeze, checkpoint/result/source hashes and usable call/token accounting still gate completion. The run uses a 600-second native sample cap and one MaAS execution attempt, so timeout rates and accuracy depend on this completion policy.
- The same gold-blind output recovery wraps D_select and D_test: deterministic conversion of explicit final answers, followed if needed by at most one extra model request using the original problem and original output. This extra request can recompute the solution, so this is an augmented method, not merely a presentation edit. Actual widths in these axes are at most six, using a 1,024-token repair cap and 120-second request timeout. The scorer never provides gold to repair. Successful and unsuccessful repairs, raw/final outputs and extra usage are audited. A native workflow exception/timeout is recorded as an execution failure rather than silently replaced by repair.

The major observed limitations are repeated/truncated reasoning, MaAS execution timeouts, low exact accuracy on robustness, and invalid generated AFlow interfaces. In AFlow parallel, round 2 passed unsupported `input`/`instruction` arguments to `ScEnsemble(solutions, problem)` and failed all 24 selection examples; it was excluded and Q/E selected round 4. The current static validator catches several missing names/result keys but does not yet validate every operator call signature. No corrected-candidate rerun is claimed here.

## Offline verification

From the repository root, without a GPU or inference endpoint:

```bash
uv run --no-project python results/masbench/pro6000_seed0_20261014/verify_artifacts.py
```

The standard-library verifier checks all 1,091 exported files against encoded and decoded hashes, checks the completed-run source snapshots/selection/result hashes, and independently recomputes all 20 Q/E test scores from the preserved 60-row native CSVs. It verifies the aggregate and macro averages. It writes [artifact_verification.json](validation/artifact_verification.json); this verification does not change or override any quality warning.

The original 32 regression tests were also run against the source snapshots; see [unit_tests.txt](validation/unit_tests.txt). Their environment requires OpenAI and the recorded method dependencies, whereas the offline archive verifier needs only Python.

```bash
PYTHONPATH="$PWD/results/masbench/pro6000_seed0_20261014/reproduction/repo:$PWD/results/masbench/pro6000_seed0_20261014/reproduction/repo/experiments" \
  uv run --no-project python -m unittest discover \
  -s results/masbench/pro6000_seed0_20261014/reproduction/repo/experiments -p 'test_*.py'
```

For an inference rerun, restore the exported files to the package-relative paths in `artifact_manifest.json`, place the snapshot scripts at the package root/`repo`, obtain the pinned official baselines, frozen benchmark inputs and model revisions, restore the recorded environments, and adjust the local endpoint/model paths for the new machine. The source launchers intentionally retain their original package layout and isolated-method assumptions; they are not a one-command A100 deployment.
