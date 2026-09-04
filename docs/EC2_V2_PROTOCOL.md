# EC-2 v2 Protocol

EC-2 v2 is the only EC-2 protocol eligible for a communication-topology main
table. Earlier EC-2 artifacts are development/debugging outputs and are
rejected by the v2 aggregator.

Each method uses the frozen Qwen/Qwen3.5-9B endpoint, temperature `0` (also
for the RPAS reflector), a
256-token cap, one communication round, the same `GDesigner.FinalRefer`
judge, and verbatim messages. The six worker roles are `Knowlegable Expert`,
`Critic`, `Mathematician`, `Psychologist`, `Historian`, and `Doctor`.
Compression is disabled because the pinned official G-Designer implementation
does not provide it.

`single_agent` is a separately labelled one-worker reference. The topology
competitors are fixed `full_connected`, fixed `chain`, official-trained
`gdesigner`, and `rpas_comm`. All topology competitors use the six-worker
pool. Data are fixed as MMLU `dev -> D_search`, `val -> D_select`, and
`test -> D_test`; source-file and split hashes are written to every run.

G-Designer is pinned to `yanweiyue/GDesigner@a6efcfa3b40bb4d9cbf46f883a95d62020bd8251`.
It runs the released 10-iteration, batch-size-4, learning-rate-0.1 MMLU
training loop. The initial and trained GCN checksums must differ. `D_select`
is an independently costed audit pass because that upstream loop does not
perform checkpoint selection.

RPAS-Comm starts in the same six-worker execution space and may mutate only
the topology. It uses an LLM reflector, typed topology mutations, and a
Pareto-aware `D_select` rule. Rule fallback, new-candidate count zero, missing
reflection calls, or missing mutation logs fail the formal gate. Its
`D_search` candidate evaluations are capped at 40 graph executions, matching
the released G-Designer train loop. `D_select` cost is reported separately;
this is not a claim of matched total search compute.

The main table reports accuracy, active worker edges/query, worker messages/
query, worker-to-worker tokens/query, FinalRefer input tokens/query, total
test tokens/query, and search tokens. Worker-output token metrics use the
local Qwen tokenizer; worker communication excludes the final judge input,
which is reported separately. No result is formal until every method completes
three protocol-valid seeds with the same split-manifest hash.

Run a pilot on exactly one approved physical GPU:

```bash
RPAS_CUDA_VISIBLE_DEVICES=4 bash experiments/run_ec2_mmlu_v2_a100.sh pilot
```

Use `5` instead of `4` for the other approved card. The script rejects every
other device identifier. After pilot review, run `formal`; add
`RPAS_EC2_V2_AGGREGATE=1` only after all methods and seeds have completed.
