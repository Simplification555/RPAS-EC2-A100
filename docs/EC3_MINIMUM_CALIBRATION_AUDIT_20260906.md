# EC3 Minimum Calibration Audit

Status: diagnostic only; formal_result=false; D_test remains locked.
Evidence inspected on 2026-09-06 after jobs 133173 and 133174 completed.

## Frozen Evidence

Artifact root on SCIR: `$REPO_ROOT/outputs/ec3_hotpotqa_v3_v4d/calibration`.
Both methods used the same 40-question D_calib split and A100 PCIe 80GB.
Split manifest hash:
`1940fbef3792b691f60fe56c604d8e12f81e8f5ac71f444a7239462c0f3dd54c`.

| Calibration row | Answer F1 | Exact match | Valid answer rate |
|---|---:|---:|---:|
| AFlow initial round 1 | 0.451635 | 0.400 | 1.0 |
| AFlow generated round 2, explicit reevaluation | 0.787713 | 0.750 | 1.0 |
| RPAS seed single_local | 0.893831 | 0.875 | 1.0 |
| RPAS seed single_strong_remote | 0.893831 | 0.875 | 1.0 |

These scores come from calibration_rows.jsonl, not transient optimizer CSV
filenames or held-out evaluation. Calibration included repeated evaluations.

| Job/method | Calls | Prompt tokens | Completion tokens | Total tokens | Job elapsed |
|---|---:|---:|---:|---:|---|
| 133173 AFlow | 161 | 216687 | 11600 | 228287 | 00:22:46 |
| 133174 RPAS seeds | 80 | 121686 | 714 | 122400 | 00:05:57 |

AFlow recorded one optimizer call and one new workflow; the RPAS run evaluated
seed architectures only and performed no Full search. Both jobs recorded zero
model errors. AFlow's manifest reports zero truncation based on token counts,
but all 161 call records have a missing finish_reason. This is weaker evidence
than RPAS's 80 explicit stop reasons and must not be described as fully observed
termination telemetry.

## Duplicate Seed Defect

Both RPAS rows have candidate ID `a0711933a84a` and topology `single`.
The singleton model/site pool collapses the first two differently named native
seeds into one architecture. The old `seed_architectures(config)[:2]` therefore
did not test two distinct workflows. Its calibration passed execution checks,
but provides no evidence of multi-agent or RPAS-Full improvement.

The minimal correction selects the first two distinct native IDs in their
original order. No native seed, mutation, search objective, evaluator, prompt,
or frozen data split is changed. The selected IDs and policy are now logged.
Tests cover alias collapse, insufficient distinct seeds, and the actual frozen
singleton configuration.

## Next Decision

Run only the corrected two-workflow calibration into a new output root,
`$REPO_ROOT/outputs/ec3_calibration_unique_v4`; preserve all existing artifacts.
Do not expand to three seeds or unlock D_test from the current evidence.
Single F1 is above the V4 0.80 saturation threshold. A distinct workflow probe
is needed before assessing the 0.02 improvement threshold. A seed probe is
still not a searched RPAS-Full operating point, so it cannot establish a final
method comparison or justify a benchmark switch by itself.

## Distinct-Candidate Probe (133224)

The corrected probe completed its calls but failed the validity gate after
00:11:03. Single again scored F1 0.893831 (40 calls, 61200 tokens).
Self-consistency scored F1 0 with valid_answer_rate=0 (120 calls, 186219 tokens).
The saved output contains, for example, `### German` while the HotpotQA parser
requires `FINAL ANSWER: German`. This is a native aggregation serialization
defect, not evidence that all underlying model answers are wrong.

The repair adds HotpotQA to the existing dataset-specific majority-output
format branch. It leaves vote counts, tie-breaking, prompts, and grading
unchanged. Tests cover the HotpotQA contract and unchanged math formatting.
Original outputs remain untouched; a fresh diagnostic run is required. No
rescored diagnostic output may replace the failed run's original manifest.

## Corrective Minimum Runs Submitted

Source commit: `ba828a2` (including the contract correction in `7af0db8`).
Local regression checks: 59 passed. Remote checks: shell syntax, Python
compilation, and both runner CLI imports passed. Shell bundle line endings
were normalized to LF before deployment. Original remote source files were
backed up separately; previous output directories were not overwritten.

| Job | Scope | Hardware | Output root | Last observed state |
|---|---|---|---|---|
| 133270 | EC3 corrected unique-seed calibration, same 40 D_calib questions | A100 PCIe 80GB, gpu05 | `outputs/ec3_calibration_contract_v4` | RUNNING |
| 133273 | EC2 paired 256/1024-token decode probe, same 8 dev questions | A100 PCIe 40GB, gpu09 | `outputs/ec2_decode_probe/133273` | RUNNING |

These are diagnostics, not complete RPAS-Full/G-Designer searched-method
comparisons. Neither run authorizes D_test access or a three-seed campaign.
Each paired EC2 condition uses the same allocated device; its timing must not
be pooled with EC3 or historical runs on different hardware.

Separately, EC1 AFlow seed-0 job `133095_0` (allocation `133098`) completed
with exit code 0 after 01:57:21. Its runner log reports 131/131 evaluations
and score 0.87023. This is a preliminary log observation, not yet an audited
cross-method EC1 result.
