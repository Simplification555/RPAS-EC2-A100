# MMLU-57x10 Main Experiment Delivery

## Scope and status

This delivery contains the completed controlled main experiment for
`MMLU-57x10`: all 57 MMLU subjects, 10 held-out test items per subject, and
570 test items per run. It is **not full MMLU**. All results have
`formal_result: false`, because the repository's formal-result gates are not
complete and the comparison is a controlled subset.

The completed matrix has four methods and three seeds (`0`, `1`, `2`), for 12
completed runs. Every run used Qwen/Qwen3.5-9B, temperature 0, a 256-token
cap, the same strict A/B/C/D answer parser, and the same frozen test split.

Only the two authorized devices were used:

| Logical GPU | UUID | Roles |
| --- | --- | --- |
| 4 | `GPU-37aba5d1-18eb-98e1-e471-3251edd974e6` | Vanilla, G-Designer, RPAS |
| 5 | `GPU-794996c3-b094-3224-b6bf-55c8f96cb7f6` | RPAS-no-selection, G-Designer, RPAS |

No other GPU was used for this experiment.

## Methods

| Label in tables | Implemented setting | Search status |
| --- | --- | --- |
| Vanilla | Fixed `single_local` direct-answer executor | No search |
| RPAS-no-selection | Fixed predeclared `solver_verifier_local` executor | No selection/search |
| G-Designer graph-executor setting | Official G-Designer graph executor with three `AnalyzeAgent` nodes, fixed graph masks, one communication round | No separately instrumented topology-search phase |
| RPAS controlled selection | Searches and selects among nine predefined architectures on 285 development examples | `RPAS_MMLU_NEW_CANDIDATES=0`; no reflective mutation candidates |

The G-Designer result must not be called a full official topology-training or
topology-search result. The RPAS result must not be called a complete
reflective architecture-search result.

## Main table

Values are means over three seeds. Accuracy intervals are seed-level 95%
intervals. `Search calls/tokens = 0*` means the adapter did not separately
instrument a search phase; it does not prove that no extra reasoning occurred.

| Method | Accuracy, 95% interval | Subject macro | Valid rate | Test calls | Search calls | Total calls | Test tokens | Search tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 81.99% [81.87, 82.10] | 81.99% | 100.00% | 570 | 0 | 570 | 141,833 | 0 | 141,833 |
| RPAS-no-selection | 81.52% [81.22, 81.82] | 81.52% | 99.30% | 1,140 | 0 | 1,140 | 326,846 | 0 | 326,846 |
| G-Designer graph-executor setting | 48.60% [47.29, 49.90] | 48.60% | 100.00% | 1,767.7 | 0* | 1,767.7 | 848,030.3 | 0* | 848,030.3 |
| RPAS controlled selection | 81.93% [81.93, 81.93] | 81.93% | 100.00% | 570 | 5,805 | 6,375 | 141,833 | 1,922,172.7 | 2,064,005.7 |

## Per-seed results

| Method | Seed | Accuracy | Valid rate | Test calls | Search calls | Test tokens | Search tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 0 | 82.11% | 100.00% | 570 | 0 | 141,833 | 0 |
| Vanilla | 1 | 81.93% | 100.00% | 570 | 0 | 141,833 | 0 |
| Vanilla | 2 | 81.93% | 100.00% | 570 | 0 | 141,833 | 0 |
| RPAS-no-selection | 0 | 81.23% | 99.30% | 1,140 | 0 | 326,846 | 0 |
| RPAS-no-selection | 1 | 81.75% | 99.30% | 1,140 | 0 | 326,846 | 0 |
| RPAS-no-selection | 2 | 81.58% | 99.30% | 1,140 | 0 | 326,846 | 0 |
| G-Designer graph-executor setting | 0 | 47.54% | 100.00% | 1,737 | 0* | 834,666 | 0* |
| G-Designer graph-executor setting | 1 | 49.82% | 100.00% | 1,777 | 0* | 852,206 | 0* |
| G-Designer graph-executor setting | 2 | 48.42% | 100.00% | 1,789 | 0* | 857,219 | 0* |
| RPAS controlled selection | 0 | 81.93% | 100.00% | 570 | 5,807 | 141,833 | 1,921,845 |
| RPAS controlled selection | 1 | 81.93% | 100.00% | 570 | 5,802 | 141,833 | 1,920,673 |
| RPAS controlled selection | 2 | 81.93% | 100.00% | 570 | 5,806 | 141,833 | 1,924,000 |

## Statistical analysis

Paired comparisons use the same 570 test IDs per seed, a 5,000-repetition
paired bootstrap, and continuity-corrected McNemar statistics. The generated
artifact records every seed-level value.

- RPAS versus Vanilla: seed 0 difference `-0.18` percentage points, bootstrap
  interval `[-0.53, 0.00]`; seeds 1 and 2 difference `0.00` percentage points.
  There is no evidence here that controlled RPAS selection improves over the
  Vanilla baseline.
- RPAS-no-selection versus Vanilla: differences range from `-0.88` to `-0.18`
  percentage points. All paired bootstrap intervals include zero.
- G-Designer graph-executor setting versus Vanilla: differences range from
  `-32.11` to `-34.56` percentage points. All paired bootstrap intervals are
  below zero; McNemar chi-square values are `139.76` to `154.28`.

The first two findings are particularly important for paper wording: this
experiment supports a controlled comparison against the present G-Designer
execution setting, but it does not support a claim that RPAS selection beats
single-model Vanilla inference or that it is computationally cheaper.

## Cost interpretation

RPAS has substantially larger recorded total cost than Vanilla because it
evaluates the nine predefined architectures on the 285-example development
split before testing. Its mean total-token count is approximately 2.06M,
compared with 0.142M for Vanilla. Therefore the following claim is unsupported
by these data: “RPAS is computationally cheaper.”

G-Designer's `search_calls=0` and `search_tokens=0` mean only that no separate
search stage was logged by the adapter. They are not a full accounting of any
topology-design, profile-encoding, or graph-setup work.

## Runtime fixes and reproducibility changes

The following fixes were required to obtain valid runs:

1. Updated the local Transformers stack to support Qwen3.5 and added a
   PyTorch fallback for the incompatible `causal_conv1d_update` CUDA wrapper.
2. Set the decoder-only tokenizer to left padding. The prior right-padding
   behavior produced invalid generations; the corrected smoke tests reached a
   valid-answer rate of 1.0.
3. Updated the G-Designer adapter's device guard from the obsolete GPU 6/7
   pair to the authorized GPU 4/5 pair.
4. Installed the G-Designer runtime dependencies required by its repository
   imports: `class-registry`, `shortuuid`, `wikipedia`, and `astunparse`.
5. Normalized G-Designer CSV subject IDs ending in `_test` during aggregation
   so paired statistics align with the shared MMLU test IDs.

## Artifacts and checks

Raw experiment artifacts remain local and are intentionally ignored by Git:

```text
outputs/mmlu_main_table_20260903/{vanilla,rpas_no_selection,gdesigner,rpas}/seed_{0,1,2}/
reports/mmlu_main_table_20260903/main_table.{json,csv}
```

The public repository includes this delivery note and
`docs/MMLU_MAIN_TABLE_GPU45.md`; it does not include model weights, MMLU CSV
content, prompt outputs, credentials, or the raw run artifacts.

Verification completed after aggregation:

```text
python -m pytest -q                  # 23 passed
python -m external_comparison.runners.validate_protocol \
  --config-dir external_comparison/configs --require-native  # all configs OK
```

## Paper-safe wording

> On the controlled MMLU-57x10 subset, RPAS controlled selection achieved
> 81.93% accuracy and the G-Designer graph-executor setting achieved 48.60%.
> RPAS incurred substantially higher recorded search cost. The controlled
> RPAS result was statistically indistinguishable from the Vanilla baseline
> under deterministic decoding and a fixed candidate set.

Do not write that these results are full MMLU, full official G-Designer
topology search, reflective mutation search, evidence of RPAS compute savings,
or evidence of a production/banking capability gain.
