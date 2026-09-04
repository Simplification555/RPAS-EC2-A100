# RPAS Experiment Protocol

> Status: **FROZEN-CORE v1.0**  
> Freeze date: 2026-08-25  
> Scope: final paper experiments for RPAS  
> Execution status: formal runs are blocked until all gates in Section 14 pass

## 1. Purpose and scope

This document is the single source of truth for the final RPAS experiments. It freezes the research questions,
dataset partitions, search space, comparison methods, budgets, model settings, selection rules, metrics, and reporting
policy before formal experiments begin.

RPAS is evaluated as a **training-free, task-level multi-agent architecture search method**. It searches one static
architecture on development data and applies the validation-selected architecture to held-out examples. The current
paper does not claim per-query dynamic routing, model-weight training, or a real WAN deployment.

The central claim is:

> Reflective Pareto search can discover multi-agent architectures that achieve a better task-quality/resource-cost
> trade-off than fixed architectures, random search, and quality-only workflow search, especially on tasks whose
> structure benefits from decomposition, verification, parallelism, tools, or heterogeneous model assignment.

AIME is a control benchmark for detecting over-orchestration. MASBench and GAIA provide the primary positive evidence
for architecture adaptation. WAN profiles and heterogeneous model pools are deployment analyses, not the sole
motivation of the method.

## 2. Research questions

- **RQ1: Overall effectiveness.** Does RPAS improve the held-out quality/resource Pareto trade-off over handcrafted
  architectures and automated workflow-search baselines?
- **RQ2: Search mechanism.** Does failure-aware LLM reflection produce better candidates than random mutation,
  quality-only evolution, and MCTS/meta-agent search under the same candidate-evaluation budget?
- **RQ3: Task structure.** Does RPAS select different topologies for mathematical reasoning, MASBench structural axes,
  and tool-using GAIA tasks?
- **RQ4: Heterogeneous deployment.** Can RPAS assign models, roles, sites, and communication policies more efficiently
  when model capability, price, and placement differ?
- **RQ5: Over-orchestration.** Does RPAS retain a simple single-agent architecture when additional agents do not offer
  reliable quality gains?
- **RQ6: Robustness.** Are conclusions stable across search seeds, data partitions, and deployment profiles?

## 3. Experiment tracks

### 3.1 Main Track A: homogeneous architecture search

Use homogeneous Qwen3.5-9B replicas to isolate topology, role, token-budget, and communication-policy effects from model
capability differences.

Required benchmarks:

- AIME 2025 and AIME 2026.
- MASBench: breadth, depth, horizon, parallel, and robustness axes.
- GAIA 2023 validation Level 1, using the fixed controlled split in Section 5.3.

Required comparison methods are listed in Section 7. All methods use the common executor, role prompts, model pool,
tools, seed library, and validation/test protocol.

### 3.2 Main Track B: heterogeneous architecture search

Use a frozen three-tier model pool:

- `cheap`: low-cost/fast model for planning, decomposition, and routine tool interaction.
- `standard`: Qwen3.5-9B local backbone.
- `strong`: stronger paid API model for difficult solving, verification, or aggregation.

The exact provider model IDs, API versions, prices, rate limits, and endpoint configuration must be recorded in a
versioned model-pool manifest before the first formal heterogeneous run. Models may not be changed within one reported
experiment matrix.

Required benchmarks:

- MASBench parallel, depth, and robustness.
- GAIA Level 1 controlled split.
- AIME 2025 as an over-orchestration control only.

### 3.3 Deployment sensitivity track

Trace-based deployment emulation is evaluated separately from task quality. The main profiles are:

- `lan_homogeneous`.
- `wan_normal`.
- `wan_bandwidth_limited`.
- `wan_edge_hub`.

`wan_degraded` and `wan_lossy_unstable` are robustness/appendix profiles. The paper must call this mechanism
**trace-based WAN emulation**, not real-network benchmarking. No artificial sleep or stochastic packet drops are
injected.

Two analyses are required:

1. Post-hoc trace remapping: map the same execution traces to every profile without new model inference.
2. Profile-conditioned search: rerun RPAS on `lan_homogeneous`, `wan_bandwidth_limited`, and `wan_edge_hub` for
   MASBench-parallel and GAIA to test whether selected architectures change.

A small gpu15/gpu7 cross-node run is a sanity check only and is not used as the main WAN result.

## 4. Unit of search and leakage prevention

An architecture is searched at the dataset/axis level, not independently for each test query. Each benchmark is split
into three disjoint partitions:

- `D_search`: candidate generation and parent-selection feedback.
- `D_select`: final candidate ranking and primary architecture selection.
- `D_test`: one-time held-out evaluation.

Test labels and test scores must never influence reflection, mutation, parent selection, stopping, hyperparameter
selection, or the choice of the reported architecture. `selected_test_rows[0]` must always refer to the candidate chosen
using `D_select` before any test score is observed.

Dataset partitioning uses the fixed `data_seed=2026`. Search randomness uses seeds `0`, `1`, and `2`. Changing a search
seed must not change `D_search`, `D_select`, or `D_test`.

## 5. Datasets and fixed splits

### 5.1 AIME 2025 and AIME 2026

- Development source: local `aimo-validation-aime.jsonl`, 90 examples.
- `D_search`: 60 examples selected with `data_seed=2026`.
- `D_select`: the remaining 30 development examples.
- `D_test`: all 30 official examples from each of `aime_2025.jsonl` and `aime_2026.jsonl`.
- Metric: exact-match answer accuracy after the frozen answer normalizer.

Architecture search is performed once per method/search seed using the common AIMO development split. The same
validation-selected architecture is then evaluated on AIME 2025 and AIME 2026. The two years are not used to select
different architectures.

### 5.2 MASBench

- Source: official MASBench train and test splits.
- Axes: breadth, depth, horizon, parallel, and robustness.
- `D_search`: 24 examples per axis from the official train split.
- `D_select`: 24 disjoint examples per axis from the official train split.
- `D_test`: 60 examples per axis from the official test split, balanced across the available axis values.
- Metric: exact-match structured answer accuracy using `masbench_igsm_mod23_v1` or its explicitly versioned successor.

The exact selected IDs and axis-value counts must be stored in a dataset manifest. The official test split remains
isolated. Early outputs such as `masbench_depth_v4_sanity` and `masbench_dag_parallel_v5_sanity` are calibration only.

### 5.3 GAIA

- Source: GAIA 2023 public validation set.
- Primary scope: Level 1, all usable examples after the fixed integrity check.
- Known exclusion: corrupt zero-byte attachment `f918266a-b3e0-4914-865d-4faa564f1aef.py`.
- Expected usable Level-1 pool: 52 examples, subject to manifest verification.
- `D_search`: 24 examples.
- `D_select`: 12 examples.
- `D_test`: 16 examples.
- Split: fixed once with `data_seed=2026`, stratified by attachment/no-attachment and attachment type where possible.
- Metric: GAIA normalized exact-match accuracy.

All 52 usable Level-1 tasks must be attempted. Formal GAIA runs require local-file tools, Python/code execution,
spreadsheet/document/image/audio handling where required, and web/search access for tasks that require external
information. If the runtime cannot support the complete Level-1 pool, the result must be labeled
`GAIA-L1-supported-subset` and the fixed task manifest must be published; it must not be presented as the official GAIA
score.

The existing 24-task `gaia_adapter_calibration` result is a runtime calibration, not a paper result. Its 16.7% score
and verifier behavior must not be merged into the formal experiment tables.

## 6. Common architecture search space

All search methods operate on the same typed genome and executor.

### 6.1 Topologies

- `single`.
- `self_consistency`.
- `solver_verifier`.
- `planner_solver_verifier`.
- `debate`.
- `dag_decompose`.

### 6.2 Searchable fields

- Topology.
- Agent role and model assignment.
- Logical site placement.
- Per-role `max_tokens`.
- Self-consistency sample count.
- DAG worker count.
- Edge communication policy: `full`, `summary`, `final_only`, or `critic_brief`.

### 6.3 Shared seed library

Every automated search method begins with the same nine evaluated seed architectures:

1. `single_local`.
2. `single_strong_remote`.
3. `self_consistency_local`.
4. `solver_verifier_local`.
5. `solver_local_verifier_remote_summary`.
6. `solver_local_verifier_remote_final_only`.
7. `planner_solver_verifier_split`.
8. `debate_local_remote`.
9. `dag_decompose_three_workers`.

The seed evaluations are cached and shared across methods for the same dataset split, network profile, model manifest,
and search seed. A cache hit reuses the exact rollout result rather than rerunning the model. This is required to remove
mode-to-mode noise for identical architectures.

## 7. Comparison methods and paper-facing names

### 7.1 Non-search execution baselines

- **Single Agent**: strongest validation-selected single-agent configuration.
- **Self-Consistency**: fixed three-sample majority-vote architecture.
- **Solver-Verifier**: strongest validation-selected fixed solver-verifier seed.
- **Multi-Agent Debate**: fixed debate seed.
- **DAG Decomposition**: fixed decomposition/parallel-worker/aggregator seed.
- **Best Handcrafted**: one primary architecture selected from all nine handcrafted seeds using the common selector.

### 7.2 Automated search baselines

- **Random-AS**: random typed architecture generation/mutation in the common search space.
- **AFlow-style MCTS**: MCTS candidate search using the common RPAS executor and search space.
- **ADAS-style Meta-Agent Search**: an LLM meta-agent proposes executable architectures in the common search space.

If official external repositories can be run on a benchmark without changing their method, an additional
`Official implementation` comparison may be reported. A common-space reimplementation must be named `AFlow-style` or
`ADAS-style`; it must not be described as an exact official reproduction.

### 7.3 RPAS variants

- **RPAS-Quality**: LLM reflection and typed mutation, but parent/final selection uses quality only.
- **RPAS**: failure-aware LLM reflection, multi-proposal typed mutation, Pareto parent sampling, and Pareto final
  selection.

The old code names map as follows:

| Code mode | Paper-facing name | Category |
|---|---|---|
| `baselines` | Best Handcrafted plus fixed architecture rows | non-search baselines |
| `random` | Random-AS | automated search baseline |
| `quality_only` | RPAS-Quality | RPAS ablation |
| `wan_pareto` | RPAS | full method |

These four modes must not be presented as four external methods.

## 8. Fair search budget

All automated search methods receive exactly:

- Nine shared seed evaluations.
- Twenty-four **new unique executable candidate evaluations**.
- A maximum archive size of 33 candidates.
- The same `D_search` examples and evaluation concurrency.
- The same role prompts, tools, model pool, token pools, and candidate-validity checks.

Duplicate or invalid proposals do not consume the 24-candidate budget. The proposer may retry up to three times before
recording a proposal failure. Reflection/meta-agent/MCTS controller calls are not counted as candidate evaluations, but
their calls, tokens, wall time, and API cost must be reported separately as **search overhead**.

RPAS uses:

- LLM reflection.
- Up to three independent mutation proposals per reflection call.
- Up to three compact failure examples.
- `pareto_parent_prob=0.5`.
- `parent_score_band=0.05`.
- `parent_top_k=6`.

The formal code must expose seed count and new-candidate count separately. Using `max_candidates=16` is not permitted
for final runs because it yields only seven evolved candidates after the nine seeds.

## 9. Candidate evaluation and validity

Each candidate is executed on `D_search`. The executor records task predictions and a full model/tool/communication
trace. A candidate is invalid for selection if any of the following holds:

- Valid execution rate is below 99%.
- A context-window or protocol error affects more than 1% of examples.
- Output truncation prevents answer extraction on more than 5% of examples.
- The candidate violates the topology or tool contract.

Invalid candidates remain in diagnostic logs but are excluded from the Pareto front and primary selection. Errors may
not be traded for lower token cost.

After search, at most eight candidates are shortlisted: the full strict Pareto front if it has at most eight members;
otherwise the quality point, efficiency point, and six evenly distributed frontier points. These candidates are
reevaluated on `D_select`. Only `D_select` metrics determine the final architecture.

## 10. Pareto objectives and final selection

### 10.1 Primary objectives

- Maximize task score.
- Minimize total model tokens.
- Minimize model calls.
- Minimize monetary inference cost under the frozen model-price manifest.

### 10.2 Deployment objectives

- Minimize cross-center tokens.
- Minimize estimated network-only latency.
- Minimize expected retry overhead.

Observed wall latency is reported but is not a primary Pareto objective because it is sensitive to server load and
continuous batching. Deployment objectives are included in Track B and deployment experiments; they are secondary
diagnostics in homogeneous Track A.

### 10.3 Two pre-registered operating points

Every search method reports two candidates selected only on `D_select`:

- **Quality point (Q):** highest validation score; ties are broken by total tokens, calls, monetary cost, cross-center
  tokens, then candidate ID.
- **Efficiency point (E):** among valid Pareto candidates within `delta=0.05` absolute score of Q, minimize total
  tokens; ties are broken by calls, monetary cost, cross-center tokens, then candidate ID.

The main table reports both Q and E. RPAS-E is the primary quality-efficiency result. Comparisons at Q and E must use
the same selector for every search method. Test score must never be used to switch between Q and E.

## 11. Metrics and statistical reporting

### 11.1 Task quality

- Exact-match accuracy and number correct.
- Per-dataset/axis accuracy.
- GAIA level and attachment-type breakdown.
- MASBench axis-value breakdown.

### 11.2 Inference resources

- Average model calls.
- Average prompt, completion, and total tokens.
- Average requested maximum tokens and maxed-call rate.
- Monetary inference cost per task.
- Observed model wall latency per task.
- Tool calls, tool failures, and tool latency for GAIA.

### 11.3 Deployment resources

- Cross-center total/message/model tokens.
- Network-only latency estimate.
- Expected retry latency and failures.
- Emulated end-to-end latency.

### 11.4 Search resources and architecture behavior

- Unique candidate evaluations.
- Reflection/controller calls, tokens, wall time, and monetary cost.
- Number of valid/invalid/duplicate proposals.
- Pareto-front size, dominated hypervolume, and frontier coverage.
- Selected topology, role-model assignment, site placement, token budgets, and communication policies.
- Fraction of selected architectures that are single-agent versus multi-agent.

### 11.5 Repetition and uncertainty

- Formal search seeds: `0`, `1`, and `2`.
- Data split seed: fixed at `2026` for every search seed.
- Report mean and standard deviation across search seeds.
- Report 95% paired bootstrap confidence intervals over held-out examples with 10,000 resamples.
- Use paired comparisons because all methods share the same test examples.
- For AIME, additionally report `correct/30` and a Wilson binomial interval.

## 12. Main tables and figures

### Table 1: overall quality and efficiency

Rows: Single Agent, Best Handcrafted, Random-AS, AFlow-style MCTS, ADAS-style Meta-Agent Search, RPAS-Quality, and RPAS.

Columns per benchmark: test score, total tokens, calls, monetary cost, and observed wall latency. Report Q and E operating
points for automated search methods.

Benchmarks: AIME 2025, AIME 2026, five MASBench axes, and GAIA Level 1.

### Table 2: heterogeneous model assignment

Rows: Best Handcrafted, Random-AS, AFlow-style MCTS, RPAS-Quality, and RPAS.

Columns: score, tokens by model tier, calls by model tier, API cost, selected topology, and role-model assignment.

Benchmarks: MASBench depth/parallel/robustness and GAIA Level 1.

### Figure 1: quality-resource Pareto fronts

Plot held-out quality against total tokens, calls, and monetary cost. Use one panel each for AIME, MASBench, and GAIA.
Do not connect points produced by different test selections as if they were one optimization trajectory.

### Figure 2: architecture adaptation

Show selected topology and role/model placement by MASBench axis and GAIA attachment/tool category.

### Figure 3: deployment sensitivity

Show cross-center tokens and trace-emulated network latency under LAN, bandwidth-limited WAN, and edge-hub profiles,
including whether profile-conditioned RPAS changes architecture.

## 13. Ablation studies

Run ablations on MASBench depth, MASBench parallel, and GAIA Level 1 with seeds 0/1/2. AIME is not required for every
ablation.

- **No LLM reflection:** rule/random typed mutation only.
- **No failure examples:** reflector receives aggregate metrics and architecture but no failed rollouts.
- **Single proposal:** one mutation proposal instead of three.
- **No Pareto search:** RPAS-Quality.
- **No Pareto parent sampling:** parents selected only from the high-quality band.
- **Structure-only mutation:** disable model, site, and communication mutations.
- **No deployment objectives:** remove cross-center/network objectives in heterogeneous and WAN experiments.

The minimal paper ablation table must contain the first four ablations. The remaining three can be moved to the
appendix only if compute is constrained.

## 14. Formal-run gates

No output is labeled `formal` until every applicable gate passes:

- [ ] **G1: Repository gate.** Private Git repository initialized; clean commit recorded; no outputs, data, model
  weights, logs, caches, secrets, or absolute personal paths tracked.
- [ ] **G2: Protocol gate.** This file reviewed by all collaborators and tagged `experiment-protocol-v1.0`.
- [ ] **G3: Split gate.** Frozen dataset manifests contain task IDs, source hashes, split seed, and exclusion reasons.
- [ ] **G4: Executor gate.** AIME, MASBench, and GAIA use the common architecture genome, trace schema, validity checks,
  and final selector.
- [ ] **G5: GAIA gate.** Required tools are available; all formal GAIA tasks complete without unsupported-tool fallback.
- [ ] **G6: Heterogeneity gate.** Exact model IDs, API versions, prices, rate limits, and role eligibility are frozen in
  a versioned manifest.
- [ ] **G7: Baseline gate.** Random-AS, AFlow-style MCTS, and ADAS-style Meta-Agent Search pass shared-space fairness
  tests and use the same 24-new-candidate budget.
- [ ] **G8: Reproducibility gate.** Unit tests, lint, one end-to-end smoke per dataset, cache consistency, resume, and
  manifest-hash checks pass.
- [ ] **G9: Reporting gate.** Result files record code commit, protocol version, config hash, dataset-manifest hash,
  model-manifest hash, prompt protocol, search seed, data seed, and selection policy.

## 15. Existing results and their status

The following are not final paper results because they predate this protocol or use calibration-sized splits/budgets:

- All `phase1*` small-model GEPA boundary experiments.
- `rpas_aime_formal_v1`, `v2`, and `v3` runs.
- All `*_sanity`, `*_smoke`, `*_calibration`, and `_tmp_*` outputs.
- Existing MASBench depth/parallel calibration outputs.
- Existing GAIA readiness, tool, verifier, and adapter calibration outputs.

These results remain useful for engineering decisions, prompt calibration, runtime validation, and expected runtime
estimation. They must not be combined statistically with protocol-v1.0 formal results.

## 16. Freeze and change-control policy

The following items are frozen in v1.0: research questions, benchmark scope, three-way data isolation, data seed,
search seeds, common search space, nine seeds, 24-new-candidate budget, required baselines, two operating points,
primary metrics, ablations, and leakage policy.

Run-specific infrastructure values such as endpoint URLs, GPU IDs, and API credentials are not scientific variables and
belong in untracked environment files. Exact model IDs and prices are scientific variables and must be frozen in the
model manifest.

Any scientific change after the first formal run requires:

1. A new protocol version (`v1.1` for compatible clarification, `v2.0` for changed claims/budgets/splits/selectors).
2. A dated changelog entry explaining the reason and affected experiment cells.
3. Rerunning every affected method/dataset cell; selective reruns that favor RPAS are prohibited.
4. Preserving prior manifests and results under their original protocol version.

Bug fixes that change predictions, traces, cost accounting, candidate validity, or selection also require a protocol
revision and affected-cell reruns. Formatting-only or logging-only fixes do not.

## 17. Required artifact layout

Each formal run must write:

```text
outputs/formal_v1/<dataset>/<task_or_axis>/<deployment>/<method>/seed_<seed>/
  result.json
  summary.csv
  search_rows.jsonl
  selection_rows.jsonl
  search_checkpoint.json
  proposal_rows.jsonl
  search_overhead_rows.jsonl
  search_overhead.json
  selected_quality_candidate.json
  selected_efficiency_candidate.json
  test_outputs_<candidate_id>.json
  run_manifest.json
```

Cross-run summaries must be generated from these files and never manually edited. The paper table generator must reject
runs with mixed protocol versions, config hashes, dataset manifests, or model manifests.

## 18. Implementation gaps created by this freeze

Before formal runs, the code must be aligned with this protocol. Status as of 2026-08-27:

1. **Implemented; server gate pending.** Add distinct `D_search` and `D_select` evaluation paths.
2. **Implemented; server gate pending.** Separate `seed_candidates=9` from `new_candidate_budget=24`.
3. **Partially implemented.** The current four modes share the nine-seed archive and selector; external search
   baselines still need the same adapter contract.
4. **Implemented; test/server gate pending.** Exclude invalid candidates before Pareto construction and selection.
5. **Partially implemented.** Monetary model cost and reflection/controller search-overhead accounting are emitted;
   formal model prices and the local-GPU costing convention still need to be frozen.
6. **Implemented; test/server gate pending.** Emit both Q and E operating points and their candidate artifacts.
7. **Open.** Integrate the GAIA tool loop into the common architecture executor.
8. **Open.** Implement AFlow-style MCTS and ADAS-style Meta-Agent Search adapters.
9. **Partially implemented.** Every run emits normalized split, model, config, network, and protocol hashes; source-file
   hashes, a required clean code commit, and strict aggregation gates remain open.
10. **Open.** Add formal aggregation, confidence intervals, and protocol-consistency checks.

Until these ten items are complete, new runs are pilots even if they use large sample sizes.
