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
