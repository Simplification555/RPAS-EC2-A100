# MMLU-57x10 Main Table

This is a controlled `MMLU-57x10` subset: 57 subjects, 10 test items per
subject (570 items), Qwen/Qwen3.5-9B, temperature 0, and a 256-token cap.
All twelve runs were executed only on logical GPU 4 or logical GPU 5.
`formal_result` remains `false`; this is not a claim about full MMLU or
production banking performance.

| Method | Accuracy | Subject macro | Valid rate | Test calls | Search calls | Total calls | Test tokens | Search tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 81.99% | 81.99% | 100.00% | 570 | 0 | 570 | 141,833 | 0 |
| RPAS-no-selection | 81.52% | 81.52% | 99.30% | 1,140 | 0 | 1,140 | 326,846 | 0 |
| G-Designer graph-executor setting | 48.60% | 48.60% | 100.00% | 1,767.7 | 0* | 1,767.7 | 848,030.3 | 0* |
| RPAS controlled selection | 81.93% | 81.93% | 100.00% | 570 | 5,805 | 6,375 | 141,833 | 1,922,172.7 |

Values are means over seeds 0, 1, and 2. The accuracy 95% intervals and
per-seed paired bootstrap/McNemar statistics are in the generated
`reports/mmlu_main_table_20260903/main_table.json` artifact.

`RPAS controlled selection` evaluates nine predefined architectures; no new
reflective mutation candidates were enabled (`RPAS_MMLU_NEW_CANDIDATES=0`).
The G-Designer adapter reports test-graph calls only; its search/topology
cost is not separately instrumented. Therefore this table must not be used to
claim that RPAS is computationally cheaper.

\* A zero search counter means “not separately instrumented”, not “zero extra
reasoning work”.
