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
