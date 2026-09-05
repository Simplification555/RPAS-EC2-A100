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
