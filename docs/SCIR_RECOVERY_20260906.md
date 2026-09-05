# SCIR Recovery, 2026-09-06

This is an operations record, not a completed experiment report. Times below
are UTC+08. No aggregate is eligible for a paper table yet.

## Verified Failures And Recovery

- Slurm `132164_0.7` (MaAS seed 1) was CANCELLED with signal 15 when the
  AFlow parent allocation finished at approximately 00:50. Old logs did not
  establish continued execution. Earlier status messages claiming it was
  running were incorrect.
- Its interrupted workspace and telemetry are preserved under
  `outputs/scir_ec1_formal_v2/_interrupted/maas_seed_1_132164`.
  No complete `HumanEval_controller_sample4.pth` existed, so partial component
  weights were not presented as a completed search checkpoint.
- Independent MaAS seed 1 retry: `132375_4`, H100 on gpu01. Same model,
  dataset, seed, and search configuration; no reuse of the interrupted history.
- RPAS seed 1 already had a running process in `132166_2.1`. The newer
  V100 duplicate `132322_7` was cancelled to prevent two writers targeting
  the same output directory. Final artifacts require provenance checking.

## Allocation Lifecycle

The old submitted batch script does not wait for externally added Slurm steps.
Until they finish, `guard_packed_allocation.sh` pauses only its batch shell;
the model servers and runner child processes continue. It resumes the same
shell after the recorded worker PIDs finish, checking PID start times to
avoid PID reuse. Slurm time limits and allocated resources are unchanged.

- `132166_2.10` guards parent PID 1146820 while wrappers 1147633 (RPAS seed 1)
  and 1153518 (MaAS seed 0) run.
- `132375_4.4` guards parent PID 2687909 while wrapper 2688594 runs
  EC2 G-Designer seed 0 in `132375_4.2`.
- Guard logs are `outputs/scir_ec1_formal_v2/_jobs/packed_guard_<job>.log`.
- Parent batch state alone is not a seed-success signal. Inspect each step,
  wrapper, result, and checksum separately. Do not terminate a guard while
  its workers are active unless cancelling the whole allocation intentionally.
- New bundles should use a batch wrapper that directly starts and waits for
  all workers; this script is a recovery measure for existing allocations.

## Concurrency At 01:10

Eight GPU allocations contain eleven distinct experiment runners:

- EC1: AFlow 2, MaAS 0/1/2, RPAS 1.
- EC2: Full 0/1/2, Chain 0, Single 2 retry, G-Designer 0.

EC2 G-Designer seed 0 was removed from pending task `132330_9` before
starting the packed step. Remaining pending tasks are 7/8 and 10 through 14.
They are subject to the scheduler's user job limit. The H100 serving MaAS 1
and G-Designer 0 used about 38 GiB of its 80 GiB at the initial check.

## Artifact Integrity

`scripts/aggregate_ec1_native_formal.py` now normalizes the two native
manifest layouts to the actual frozen 33/131 task IDs and source hashes.
It validates checksums, row identity, score recomputation, model errors,
call phases, token sums, and metric totals before aggregation.

Eight focused regression tests cover normalization and rejection of corrupt
checksums, missing splits, wrong held-out IDs, false scores, model errors,
and inconsistent accounting. Local copies of AFlow seeds 0/1 and RPAS seeds
0/2 passed this artifact-integrity check. This does not certify algorithm
fidelity, baseline completeness, or the full protocol.

Previously running wrappers may still need seed-level `SHA256SUMS` generated
after all result/metrics files finish, because Slurm retains the submitted
script snapshot. Do not regenerate checksums to conceal changed predictions.
Do not overwrite held-out outputs to improve scores.

## Native MaAS Correction At 01:20

The actual upstream MaAS helper was inspected: it uses MiniLM both for
operator-description embeddings and its sentence encoder. Replacing these
with hashed features changes the controller input and is not a native result.
The original MiniLM weights already exist on SCIR. The compatibility helper
now changes only the two model-location arguments and enables offline loading;
all other upstream helper source, including sampling, is preserved verbatim.
The weight hash is recorded in the compatibility manifest.

Actual SCIR import/embedding smoke passed: the two encoder entry points return
finite 384-dimensional vectors that agree within 1e-6, and encoder weights
remain frozen. All 19 local runtime, embedding, metrics, and integrity tests passed.

- Stopped hashed MaAS seed 0 step `132166_2.4` and seed 2 job `132281_5`.
- Stopped only MaAS seed 1's runner/server in allocation `132375_4`.
  G-Designer seed 0 in `132375_4.2` remains active. This means the parent
  batch will eventually report MaAS failure; judge G-Designer using its own
  step and artifacts, not that parent exit code.
- Archived the hashed MaAS outputs to
  `outputs/scir_ec1_formal_v2/_interrupted/maas_hashed_non_native_20260906_0120`.
  Earlier interrupted seed 1 remains in its separate archive as well.
- Submitted `132391`, `scripts/scir/maas_native_three_seed.sbatch`:
  three independent seeds on one H100 80GB, 80GB host RAM, separate local
  endpoints, staggered model loading, and explicit waits for all children.
  Slurm confirmed RUNNING on gpu01 at 01:20:10; seed 0/1 preflight passed
  at the first check while remaining model services were starting.
- The aggregate audit now rejects manifests declaring hashed MaAS embeddings.

Do not claim complete EC1/EC2 results until every required method/seed and
the paper protocol (including the Single reference) is independently checked.

## EC2 Submission Correction At 01:31

- The RPAS-Comm reflector requests 768 tokens, but the previous batch service
  capped every response at 256. The updated wrapper permits 768 only for
  RPAS-Comm's service. Worker requests are still explicitly 256; no worker
  decoding budget or topology algorithm was changed.
- RPAS-Comm seed 0 is now running in `132375_4.10`, wrapper PID 2696165,
  service port 38712. Its environment records `service_max_tokens=768`.
  The old pending duplicate `132330_12` was cancelled before launch.
- Guard `132375_4.4` was replaced using KILL (not TERM, whose trap would
  resume the parent prematurely). Replacement guard `132375_4.13` watches
  both wrappers 2688594 and 2696165. The same parent PID 2687909 remains
  paused while these runners execute. New log: `packed_guard_132375_v2.log`.
- Pending EC2 indices 7/8, 10/11, 13/14 were resubmitted as array `132406`
  with the corrected batch snapshot, A100 PCIe 80GB, 4 CPUs and 40GB host RAM
  per task. SCIR rejected generic `gpu:1` during test-only preflight before
  any old tasks were cancelled. Typed A100 submission passed.
- At 01:31 there are 12 distinct runners in 8 allocations; remaining tasks
  are constrained by `QOSMaxJobsPerUserLimit`.

The user's later request to pack experiments supersedes the original
single-worker scheduling instruction. Co-resident wall-clock latency is
operational telemetry, not a controlled standalone method-speed comparison.
Do not present these timings as equivalent hardware-isolated latency results.

## RPAS Evaluator Repair At 01:45 UTC+08

The shared HumanEval extractor removed all source preceding the required
function, even when the full completion was valid Python. This discarded
imports and helper functions, causing false execution failures. The repaired
extractor preserves complete parseable programs. Imports, helpers, fenced
code and prose fallback have regression coverage.

Diagnostic-only rescoring of the original RPAS seed 0 raw completions changed
the passed count from 96 to 112 of 131. This is NOT a formal replacement:
the original search also used the broken extractor and was not replayed by
the diagnostic. The original outputs are untouched. The separate report is
`outputs/scir_ec1_formal_v2/_diagnostics/rpas_seed0_extraction_v2.json`, with
`formal_result=false` and the original result SHA-256.

All old RPAS seeds under `outputs/scir_ec1_formal_v2/rpas` are now ineligible
for formal aggregation. The aggregator requires
`code_extractor_version=preserve_complete_program_v2` for RPAS. Merely adding
this field to old manifests is forbidden; a full corrected search is required.

- Data preflight passed remotely: 164 official tasks, frozen disjoint 33/131
  fixtures, unchanged source hashes. No dataset download is necessary.
- New step `132166_2.13` owns two RPAS lanes on the existing H100. Lane zero
  runs seeds 0 then 2; lane one runs seed 1. It uses separate endpoints
  40332/40333 and independent result directories in
  `outputs/scir_ec1_rpas_extractor_v2`.
- Bundle PID 1160848 and new guard PID 1160858 were verified before retiring
  the old guard PID 1158203 with KILL and terminating only old RPAS step
  `132166.1`. Legacy runner/server PIDs were confirmed gone before activating
  the new bundle. AFlow seed 2 remains running in the original allocation.
- The batch parent PID 1146820 remains stopped intentionally while the guard
  watches the new bundle. The bundle waits for both lanes; do not terminate
  this guard or cancel the allocation while either AFlow or RPAS is active.
- Wrapper service cleanup now waits for server termination before reusing a
  port for the next seed. Output-root override prevents overwriting legacy
  artifacts. Initial endpoint smoke succeeded before starting search.

The complete EC1 table still requires the Single reference and the full
protocol audit, including search-budget comparability. Completed seed files
are not sufficient evidence for an accepted paper-ready comparison.

## Single Reference And Pending-Job Audit At 02:05 UTC+08

The missing EC1 direct Single reference now has a dedicated runner:
`external_comparison/runners/single_humaneval.py`. Each held-out item receives
one direct model request, temperature zero, 1024 completion tokens, no search,
and no public-test repair. The three seeds are independent deterministic
reference repetitions, not invented search seeds. Partial rows, raw outputs,
response token counts and latency are written after every completed item.
Transport failures or missing token usage fail closed, preserving diagnostics.
The client uses a 600-second timeout and zero hidden SDK retries; this policy
is recorded and must be compared with other methods' transport policies in the
final fidelity audit. It is not silently claimed to be identical to them.

Single runs sequentially on endpoint 42375 in step `132375_4.16`, sharing the
H100 with G-Designer seed 0 and RPAS-Comm seed 0. Wrapper PID 2706133 is covered
by separate guard step `132375_4.17`, PID 2706369, which also watches EC2 peer
wrappers 2688594 and 2696165. Old guard PID 2696431 was KILLed only after the
new guard was verified. Parent PID 2687909 remains intentionally stopped.
Single results are under `outputs/scir_ec1_single_reference`; initial real
requests completed successfully. Do not stop this allocation or its guard
until all three peer wrappers finish.

### Repairs For EC2 Jobs That Have Not Started

Code review found that the EC2 runtime retained most telemetry in memory,
discarded example/seed/iteration fields when materializing calls, and saved
only GCN hashes rather than reusable trained weights. The revised runtime:

- Journals completed calls, raw outputs, example results and reflection plans
  while running; refuses to overwrite an existing execution identity.
- Retains example ID, seed, phase and training iteration in calls.jsonl.
  Context variables are set before the unchanged upstream trainer creates
  each async task, preserving concurrent item attribution.
- Saves initial GCN/MLP and pre-test trained GCN/MLP checkpoints with hashes,
  and verifies that GCN values do not change during select/test evaluation.
- Freezes the selected RPAS-Comm topology before the held-out pass.
- Fixes reflector latency accounting to use observed_latency_ms rather than
  silently recording zero for the same completed call.

These are instrumentation/artifact repairs, not changes to the native
training loop, query sets, search budget, topology proposals or token caps.
The separate SCIR runtime preflight uses simulated LLM responses and
synthetic fixtures. It completed the actual pinned ten-iteration G-Designer
training loop, 280 simulated training calls and two checkpoint files. It is
strictly PASS_RUNTIME_ONLY, formal_result=false, not experiment evidence.

The six pending output directories (Chain/G-Designer/RPAS-Comm seeds 1/2)
were checked and do not contain earlier results. Running processes already
imported the old runtime and do NOT acquire these changes. In particular,
the active G-Designer seed 0 still has an unresolved checkpoint-delivery gap;
do not claim it is repaired retrospectively or restart it without a recovery
decision and preserving the existing work.

### Publication Gate Risk: Truncation

Read-only inspection of completed EC2 Single seeds 0 and 1 found, for each,
136 finish_reason=length calls out of 1140 (11.93%). Both have 570 held-out
rows and 100% parse-valid final answers, but the latter does not satisfy the
protocol's separate truncation <1% requirement. The existing aggregate does
not enforce all paper gates. These seeds must NOT be described as fully
paper-eligible on the strength of their manifest or parser success.

The frozen EC2 cap remains 256. Do not raise it, alter prompts using held-out
answers, silently relax the threshold, or report a changed protocol as the
original experiment. A documented protocol decision and independent
calibration would be needed to resolve an actual incompatibility. Other
remaining final audits include matched EC1 search budgets, three-seed paired
statistics, full telemetry completeness, and baseline fidelity.

### Verified Deployment And Real Reflection Preflight

At 02:06, the revised EC2 runtime was atomically deployed after two successful
SCIR native-loop preflights. The deployed and preflight source hashes match:
`536ded9d72a6d08148ba7d5295f1f92b85210d2a8a253c97b0d364abba8b7466`.
The old runtime is retained remotely as
`scripts/scir/ec2_v2_before_journal_20260906.py`. Pending array 132406 was
not cancelled or resubmitted. Its actual Slurm batch snapshot was read back:
normal requests retain cap 256, RPAS-Comm service permits reflector cap 768,
and the frozen per-subject budgets remain 1/1/10 with a 24-hour allocation.

A separate real-model synthetic preflight completed successfully at 02:11.
With only `layered` remaining legal, the existing reflector returned that
topology with parseable JSON, no fallback, no model error and finish=stop.
One call used 1549 prompt + 379 completion = 1928 total tokens; observed
latency was 277.13 seconds on the co-resident endpoint, including queue wait.
Evidence: `outputs/preflight_20260906/real_reflection.json` remotely and
`outputs/audit_20260906/real_reflection.json` locally. These calls are
preflight overhead, not part of Single's experiment telemetry or any table.

The local focused suite now has 37 passing tests. It additionally exercises
the entire RPAS-Comm path with a simulated model/runtime, checking exactly
40 D_search query executions, 228 D_select executions and 570 held-out
examples; the selected topology must be frozen before test evaluation.
This is regression evidence, not real-model benchmark performance.

## Bounded Fixed-Topology Batching Audit At 02:17 UTC+08

The EC2 service supports batches of four, but the harness evaluated held-out
questions sequentially. The fixed-graph path can therefore leave that batch
capacity unused. The proposed runtime evaluates independent fixed-graph
questions in groups of at most four, preserving fixture output order, each
question's copied graph, prompt contents, token cap, and candidate ordering.
On failure it cancels and joins outstanding siblings. Every completed item
still goes to the live journal. The selected concurrency is recorded in the
manifest; co-resident timings remain operational rather than isolated latency.

Concurrency is explicitly rejected for graphs with optimized spatial or
temporal edges. G-Designer training and its sampled select/test graph path
are unchanged. Chain and RPAS-Comm future runs use the bounded fixed path;
running processes retain their already-imported code.

The SCIR native-loop preflight passed again and compared serial versus
four-way evaluation using the same native fixed Chain graph. All 28 model
request payloads, ordered predictions and communication counts were equal
with simulated responses. Local coverage now includes 39 passing tests,
including out-of-order completion, the concurrency bound, sampled-graph
rejection and cancellation of siblings after a failed item. A separate real
model-service synthetic batching check is required before deployment.

### Truncation Attribution

For the existing serial EC2 Single runs, native Graph.arun executes the one
AnalyzeAgent and then FinalRefer. Each saved run has exactly 1140 calls for
570 questions. Inspecting this two-call ordering attributes all 136 length
terminations to the worker and zero to the final judge, for both seeds 0/1.
Old calls lack explicit agent/item metadata, so this attribution relies on
the verified serial executor layout rather than on a saved agent field.
This diagnostic does not change the original all-call 11.93% rate or waive
the protocol's truncation threshold.

### Fixed-Topology Batching Deployment

The strengthened native preflight used distinct synthetic question contents
and compared request payloads together with their ContextVar example IDs.
It passed, as did a real service check with four variable-length synthetic
prompts evaluated serially and concurrently. Real answers, stop reasons and
prompt-token counts agreed. Evidence is retained in
`outputs/preflight_20260906/real_batch_service.json` remotely and
`outputs/audit_20260906/real_batch_service.json` locally. This short smoke
does not prove a fourfold speedup or exact numerical equivalence for every
possible model input; do not report it as a throughput benchmark.

The approved runtime was atomically deployed at 02:22 with SHA-256
`a5de1792e8e6f6e98368d6b965ef18838b1a7eb956d730b95a79648636d22fe4`.
The preceding journal-only source is preserved remotely as
`scripts/scir/ec2_v2_before_batch_20260906.py`. Array 132406 remained pending
at deployment, with its queue priority unchanged. Chain and RPAS-Comm seeds
1/2 use four-item batches; G-Designer seeds 1/2 retain native serial sampled
evaluation. All currently running workers retain their old loaded code.

At the last check AFlow seed 2 had reached 89/131 held-out items. Its parent
batch is still deliberately stopped by the RPAS bundle guard, so AFlow
completion alone may not execute the original wrapper's cleanup/metrics
commands. Confirm the actual child runner has exited and result artifacts
are stable before collecting metrics or reclaiming that service's GPU memory.
Do not cancel allocation 132166, which also owns both corrected RPAS lanes.

## AFlow Completion And Three-Way RPAS At 02:40 UTC+08

All three AFlow seeds have completed and passed the seed artifact-integrity
loader. Scores are 115/131, 114/131 and 114/131 for seeds 0/1/2 respectively.
Mean accuracy is 0.8727735369; sample standard deviation is 0.0044072540.
Search calls/tokens are 237/227054, 255/152135, 359/392888; test calls/tokens
are 178/95388, 131/51759, 131/51759. This is a completed baseline, not a
completed EC1 matrix. The unequal search-token usage still requires the
protocol budget audit and cannot be labelled budget-matched automatically.

AFlow seed 2 metrics and seed-level checksums were generated while its
original parent was stopped. When the parent eventually resumes, its
post-processing may rewrite metrics. Reconcile and reseal metadata after
wrapper termination; do not alter predictions. A local copy of the seed-level
files is retained at `outputs/scir_ec1_formal_v2_aflow_seed2_20260906` and
passed SHA-256 verification after transfer.

Allocation 132166 now runs corrected RPAS seeds 0/1/2 on ports
40332/40333/40334, runners 1161180/1161313/1164802. Compute-node checks
confirmed all three live, healthy endpoints and about 57.2 GB GPU use.
Parent 1146820 remains stopped. Guard 1164899 watches wrappers 1160848 and
1164390; the earlier two-lane guard was replaced only after the new guard
was verified. Its log is `_jobs/guard_three_lanes_132166.log` in the corrected
RPAS output root. Never terminate an old guard with TERM during replacement,
because its EXIT trap can resume the parent prematurely.

The AFlow service was reclaimed inside a compute-node `srun` only after its
runners had exited. An earlier login-node PID check was invalid and corrected;
the third-lane activation preceded the corrected cleanup briefly. One overlap
step reported a cgroup memory allocation warning during that transition.
Subsequent compute-node checks confirmed the three RPAS runners and endpoints
remained live; this is not evidence of an experiment OOM termination.

Per-method/seed locks now prevent two wrappers from writing the same seed.
When the original sequential lane reaches RPAS seed 2, it waits for the new
lane and only reuses an audited complete result with the exact corrected root
and current configuration hash. Other existing results still reject overwrite.

EC1 Single seed 0 also completed: 113/131, exactly 131 direct model calls,
47423 total tokens and 2379.97 seconds operational wall time. It passed the
seed-integrity loader. Its local verified backup is
`outputs/scir_ec1_single_seed0_20260906`. Seed 1 was at 29/131 at 02:39;
seed 2 follows automatically. No public-test repair was used.

## Pending-Code And Aggregation Audit

The focused local suite passes 67 tests, covering HumanEval import preservation,
native adapters, MiniLM selection, run metrics, batching, split isolation,
checkpoints, duplicate writers and the paired statistics. This is regression
coverage, not a guarantee that every live workload will complete successfully.

EC1 aggregation now requires Single/AFlow/MaAS/RPAS with all three seeds,
including the separate corrected RPAS and Single roots. EC1 and EC2 report
10,000-repetition two-level paired bootstrap intervals: resample paired seeds,
then paired held-out IDs within each selected seed. Missing seeds or unmatched
IDs reject aggregation rather than silently taking their intersection.

EC2 integrity checks now reconstruct the frozen ID hash, require the actual
570 held-out IDs and the canonical 57 subjects, compare saved JSON/JSONL rows,
recompute accuracy and parser validity, and reconcile phase call/token totals.
The two completed older Single runs passed the actual-ID checks on SCIR; both
still fail the separate truncation gate. G-Designer checkpoint declarations
are checked against real nonempty files and hashes. Legacy absent checkpoints
are explicitly flagged, not retroactively invented or marked delivered.

Pending-runtime hardening publishes the shared split through a complete
temporary file and an atomic, no-overwrite hard link. Existing identical splits
are accepted; different splits reject startup. This avoids concurrent partial
JSON writes and has passed a real SCIR shared-filesystem check. Each seed still
retains its own frozen-split journal. Source identity is captured at import,
so a later deployment cannot change the final manifest's claimed runner hash.
Earlier live workers retain the old code; their final on-disk source hashes
may differ from their startup identity and require a provenance note.

Eight allocations remain running and array 132406 remains pending on the
per-user job limit, not missing dataset files. Pending work was neither
duplicated nor requeued. Runtime deployment is gated on a fresh SCIR native
training-loop preflight, including 10 iterations, checkpoint output and
serial/four-item request-equivalence checks with simulated responses.

### Deployment Confirmed At 02:45 UTC+08

The staged runtime passed the fresh SCIR preflight: ten native training
iterations, 280 simulated training calls, both checkpoint files and 28 identical
serial/four-item fixed-graph requests. No benchmark model requests were made
by this native-loop preflight. The exact tested runtime was deployed atomically
at 02:44:54 with SHA-256
`b44960f048c4ec610c9b574c41e3be03d8216bb52745ced8cdadaffeff684b51`.
The prior runtime is preserved as
`scripts/scir/ec2_v2_before_split_publish_20260906.py` remotely.

The deployed aggregation source SHA-256 is
`36e6e1870e3cdf3f41bd6caad26671dee674f3d228fdd7984138a47579863b8d`.
Shell syntax checks passed for the EC1/EC2 launchers, third-lane launcher and
EC1 aggregate launcher. Array 132406 was still pending during deployment;
no running worker was restarted and no queue position was discarded.

## Publication Inventory At 02:52 UTC+08

`scripts/audit_ec12_publication.py` now inventories all 27 required seed
results (12 EC1, 15 EC2). It is read-only and never declares publication
eligibility. The first SCIR snapshot is retained remotely at
`outputs/preflight_20260906/publication_gap_snapshot.json` and locally at
`outputs/audit_20260906/publication_gap_snapshot.json`.

Six seed artifact sets currently pass integrity checks. This is not six
publication-ready results. The snapshot establishes these concrete gaps:

- The three observed AFlow search costs cannot satisfy one +/-10% budget:
  the required budget lower bound is 357170.91 tokens, while its upper bound
  is 169038.89. The protocol's native-unit overshoot exception needs explicit
  adjudication, not automatic use to waive arbitrary cost differences.
- All 1291 AFlow calls across the three seeds lack recorded finish reasons
  and individual example IDs. Their truncation status is UNKNOWN, not zero.
  Per-item token intervals cannot be reconstructed from aggregate totals.
- EC1 Single seed 0 has three length terminations out of 131 calls (2.29%),
  exceeding the separate <1% generation-truncation gate.
- EC2 Single seeds 0/1 each have 136/1140 length terminations (11.93%).
- EC2 Single seed 2 contains an old result while its current retry remains
  live. Current environment startup postdates the old metrics, so the loader
  now rejects that stale completion. Do not overwrite or remove files while
  the runner is active; let its normal completion publish the new result.

The Slurm array task `132330_2` maps to actual allocation JobId `132376`.
Use the actual numeric JobId for `srun --jobid`; a message that array parent
132330 has expired does not mean its task has stopped. `scontrol show job`
and compute-node process inspection confirmed runner 124831 and endpoint
124192 alive on gpu13. The current task must not be restarted on the basis
of that parent-ID lookup error.

The V3.1 protocol also requires a token-difference confidence interval, not
only an accuracy interval. Both aggregators now support 10,000-repetition
two-level paired test-token intervals. Call-to-task attribution must be
observed. RPAS HumanEval's literal task prefix in the recorded agent field
is accepted with explicit provenance; call order or per-seed averages are
never substituted for missing item IDs. Unsupported intervals report
`available=false` and a reason rather than a numeric estimate.

The focused suite now passes 76 tests. SCIR also passed a separate 10,000-draw
constant-token-difference preflight. These aggregation changes are deployed;
no model parameters, token caps, candidate budgets or predictions changed.
Single seed 1 had reached 107/131 at the subsequent check.

A user decision was requested before any protocol revision or revised
experimental rerun. Until then, preserve and continue the active workloads;
do not silently relax gates, tune against held-out outcomes, or present the
current matrix as satisfying the original V3.1 specification. Older MaAS
cost-hook latency measures the accounting callback, not request wall time;
retain its run wall-clock as operational timing and audit per-call latency
before any latency claim. Co-resident runs are not isolated timing trials.

## Shared-Endpoint Chain Launch At 03:01 UTC+08

The Full endpoints already support four-request batching but their old serial
graph consumers leave spare capacity. Two pending Chain seeds now share
these existing endpoints, without a second Qwen instance or a new allocation.
This is authorized co-resident execution, not an isolated latency trial.
Both endpoints first passed the real synthetic serial/four-way consistency
preflight. Evidence: `outputs/preflight_20260906/shared_full0_batch.json`
and `shared_full1_batch.json`; those eight calls per endpoint are overhead,
not benchmark calls.

| Added Worker | Allocation | Endpoint Port | Wrapper | Runner | Parent | Guard |
|---|---|---|---|---|---|---|
| Chain seed 1 | 131866 = 131783_3 | 30166 | 1367520 | 1367804 | 1320268 | 1367619 |
| Chain seed 2 | 131909 = 131783_4 | 30209 | 1647412 | 1647558 | 1612721 | 1647459 |

Original Full runners 1320496 and 1613025 were verified live after activation.
Services 1320315 and 1612753 remain owned by their original batch shells.
Only those shells are deliberately stopped; do not stop the model services or
Full runners. Each independent guard resumes its parent after its added Chain
wrapper exits, so normal Full cleanup cannot kill an active shared consumer.
Guard logs are `_jobs/guard_shared_chain_seed{1,2}_{131866,131909}.log`
under `outputs/scir_ec2_formal` (seed 1 maps to 131866, seed 2 to 131909).

`ec2_shared_endpoint_packed.sh` validates the parent/service allocation and
PID start identities, refuses an existing destination, acquires a seed lock,
waits for an external guard activation, checks that the parent is stopped,
and never kills the shared service. It preserves cap 256, all 570 test items,
all 57 subjects, data seed 2026 and four-item fixed-graph batching. Artifacts
stay in the canonical EC2 output root. At 03:02 Chain seed 1 had written real
call records; seed 2's runner was live and loading from the shared filesystem.
GPU memory stayed near 20 GB rather than loading another 18 GB model.

### Queue Migration Incident And Recovery

After verifying array tasks 132406_7 and 132406_8 were PENDING, the command
`scancel --state=PENDING 132406_7 132406_8` cancelled the entire unmaterialized
array on this SCIR setup. `sacct` and `scontrol` confirmed cancellation of
the whole remaining array, not just the requested indices. Do not reuse this
filtered cancellation pattern for a partially selected array.

The other four never-started tasks were immediately restored as
`132507_[10-11,13-14%4]`: G-Designer seeds 1/2 and RPAS-Comm seeds 1/2.
They retain the same 4 CPUs, 40 GB host RAM, typed A100 PCIe 80 GB request,
24-hour limit and frozen experiment budgets. No running experiment was
cancelled and no partial benchmark work was lost, but the four tasks lost
their earlier queue age. This was reported to the user explicitly.

The focused suite now passes 78 tests. The shared launcher also passes
SCIR `bash -n`; the runtime it invokes is the previously native-preflighted
version with the unchanged protocol parameters.

At 03:04 both shared Chain runners were making real requests: seed 1 had
21 completed call records and seed 2 had 8 (excluding each run-start marker),
with no recorded errors. Neither had completed its first four-question batch
yet. Both Full runners and service processes remained live, and the parents
remained stopped under their verified guards. Single seed 1 reached 126/131.

At 03:05 EC1 Single seed 1 completed and passed the seed-integrity loader:
113/131, 131 direct calls, 47567 tokens, 1715.69 seconds operational wall
time, complete per-item token attribution. Its local backup at
`outputs/scir_ec1_single_seed1_20260906` passed SHA-256 verification.
Single seed 2 started automatically and reached 2/131. Chain seed 1 completed
its first four-question batch. These updates do not waive publication gates.

## Native EC1 Telemetry Repair At 03:19 UTC+08

Structural call-chain inspection found 11 consumers of the public
`native_common.call_record` helper, including EC2 and EC3. That helper was
left unchanged. The new native-only converter is used solely by AFlow,
MaAS and the native EC1 materializer, so legacy conversion remains identical.

New AFlow/MaAS driver invocations now observe the actual asynchronous OpenAI
response boundary, rather than a usage-history tail or a cost callback.
They persist response usage, finish reason, actual awaited duration, seed,
phase, task ID, observed workflow class identity, request cap/temperature,
configured timeout/retries and any error/cancellation. Task identity is carried
through ContextVars around the actual upstream `evaluate_problem`, preserving
concurrent example attribution. Operator class names are observed from native
Python operator frames when present; absent observations remain null.

The counter explicitly measures SDK `create` invocations. Internal HTTP retry
attempts are NOT individually observed: `retry_count=null` and
`transport_attempts_observed=false`, with the SDK retry limit recorded
separately. Do not equate this count to HTTP attempts or change retry policy
without the protocol decision. Streaming telemetry is explicitly unsupported;
the frozen local endpoint already rejects streaming. Missing response usage
creates an error record instead of being silently treated as measured zero.

Validation before deployment:

- 85 focused local tests passed, including concurrent identity, cancellation,
  actual await duration, operator frames, request/return preservation and
  identical legacy conversion fields.
- Both real upstream provider and HumanEval benchmark classes passed isolated
  synthetic-response preflights on SCIR. Each produced three correctly
  attributed calls with unchanged requests and upstream return values. No
  benchmark data or model endpoint was used in these two class preflights.
- A separate real endpoint synthetic check passed: 20 prompt + 2 completion
  tokens, finish=stop, 8337.56 ms observed latency. Evidence is retained in
  `outputs/preflight_20260906/native_telemetry_real_response.json` remotely and
  `outputs/audit_20260906/native_telemetry_real_response.json` locally. This
  call is preflight overhead, not part of a benchmark result.

The tested recorder was deployed with SHA-256
`bf734b6d723a0891ff2934075ca4fd95626483c751a17758f1da88b95ee00e36`;
the integrated driver has SHA-256
`47c0c725c910c74e223e4819537c7272c27a65bd6c204ee15b934776e260e8a7`.
The old driver and adapters are preserved remotely under `scripts/scir/`
with `before_response_telemetry` filenames. Existing processes retain their
loaded code; no live search was restarted and no old call evidence was
rewritten. A test-only resume may therefore mix preserved legacy search
records with new test records; `new_call_telemetry_schema` certifies only
new calls, not retroactive completeness of an entire historical run.

At 03:17 Single seed 2 had reached 69/131; the two shared Chain seeds had
completed 20 and 16 held-out rows. Array 132507 remained pending with its
four original experimental budgets unchanged.

### MaAS Seed 0 Long Tail

At the next check all three native MaAS driver processes were confirmed live
on the compute node. Seed 0 still showed batch 5 at 3/4 since approximately
02:39, while its API usage journal continued growing. Of 244 recorded calls,
the latest 30 all returned exactly 1024 completion tokens, consuming 56306
total tokens. The old journal has no task IDs or finish reasons, so this is
evidence of cap-saturating calls during a long remaining batch item, not proof
of a specific repeated prompt or an exact truncation rate. Do not describe
this as a dead process or restart it on that basis. Native retry limits and
the frozen completion cap were not changed. The outstanding protocol decision
must address this long-tail risk before a revised formal run is authorized.

Single seed 2 reached 95/131 at the same check. Native telemetry repair is
committed locally as `5599366`; no upstream GitHub push is implied.

## Pending-Task Audit And Baseline Completion At 03:45 UTC+08

The pending-task audit used the actual SCIR files, without model calls or
changing any benchmark parameters. `scripts/scir/ec2_pending_preflight.py`
verified all 171 MMLU source files against the existing frozen manifest,
57/57/570 split counts, all four Qwen safetensors shard files, local MiniLM
assets, pinned G-Designer revision, worker decoding settings and disabled
rule fallback. All four seed 1/2 G-Designer/RPAS-Comm destinations were
unoccupied at the audit snapshot. Weight contents were NOT rehashed, and
this readiness check is neither a lock nor a publication gate.

Evidence is preserved remotely under `outputs/preflight_20260906/` and
locally under `outputs/audit_20260906/`:

- `pending_inputs_0330.json`
- `pending_native_seed1_0330.json`
- `pending_native_seed2_0330.json`

The native runtime preflight now accepts an explicit seed. Seeds 1 and 2
each passed ten native GCN training iterations, 280 simulated training calls,
initial/pretest checkpoint output and equality of all 28 synthetic request
payloads between serial and four-item fixed-graph execution. These checks
ran CPU-only with simulated responses; they are not experimental outcomes.
The focused local regression suite passed 88 tests. SCIR `bash -n` passed.
Ruff was unavailable in the local offline environment; no lint pass is claimed.

### Result Sealing

The EC2 batch wrapper previously checksummed the still-active service log.
Service shutdown can append to that log after sealing, causing a bundle
checksum mismatch even when result artifacts are unchanged. The wrapper on
disk now seals stable top-level artifacts only, using relative paths, as the
shared-endpoint wrapper already does. Runtime logs are retained separately.

IMPORTANT: `scontrol write batch_script 132507 -` confirmed Slurm retained
the OLD wrapper snapshot. Replacing the file on disk does not update that
queued script. No job was cancelled or resubmitted to install this packaging
fix, so queue age was preserved. After these jobs finish, verify stable
artifacts and log finalization separately. A checksum-only log discrepancy
does not justify rerunning model inference; preserve the original manifest
and add a separately identified final bundle checksum after all writers exit.
The already-loaded model runners and model-service code were not modified.

### Completed Baselines

EC2 Single seed 2 finished with Slurm `COMPLETED`, exit `0:0`, elapsed
02:25:06. Its 570-row result passed the seed-integrity loader. The local
backup `outputs/scir_ec2_single_seed2_retry_20260906` passed all 14 original
SHA-256 checks, including its logs. This is the fresh retry, not the old
September 5 result.

| EC2 Single Seed | Accuracy | Calls | Total Tokens | All-Call Truncation |
|---|---|---|---|---|
| 0 | 0.80877193 | 1140 | 445810 | 11.9298% |
| 1 | 0.80701754 | 1140 | 444670 | 11.9298% |
| 2 | 0.80877193 | 1140 | 445765 | 12.1930% |

Mean accuracy is 0.80818713, sample standard deviation 0.00101290.
All three have complete parser-valid final outputs but FAIL the all-call
truncation gate. Per-item token attribution remains missing in these old
runtime instances. Do not claim complete EC2 or paper eligibility.

EC1 Single seed 2 also finished all 131 items and passed the integrity loader.
Its local backup `outputs/scir_ec1_single_seed2_20260906` passed all nine
SHA-256 checks. All three seeds score 113/131 = 0.86259542, sample standard
deviation zero. Seed 2 records 131 calls, 47423 tokens and 1980.62 seconds
operational wall time. Seeds 0/1 record 47423/47567 tokens respectively.
These are direct-reference repetitions, not a completed four-method EC1.

The released EC2 Single allocation allowed G-Designer seed 1 to start
automatically at 03:30:49: display `132507_10`, actual allocation `132532`,
gpu05 A100 PCIe 80 GB. Its model smoke check passed and native training was
making calls by 03:43. Remaining pending indices are 11, 13 and 14, still
limited by the per-user running-job cap. Eight allocations remain running.

After the EC1 Single wrapper exited, the guard on `132375` continued watching
the two remaining G-Designer/RPAS-Comm workers. Do not cancel that allocation
or assume its original parent has sole ownership. Shared Chain seeds 1/2
had 56/48 completed held-out rows at 03:43 and remained live. No live search
was restarted, no cap/retry policy was changed, and no publication gate was
waived during this audit.
