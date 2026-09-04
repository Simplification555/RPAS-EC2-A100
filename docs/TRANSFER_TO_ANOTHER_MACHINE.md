# Transfer to Another Machine

This public repository is sufficient to reconstruct the reviewed experiment
infrastructure and its publicly redistributable fixed fixtures. It is not a
foundation-model checkpoint or a bundle of completed experiment outputs. The
only retained learned state is the small MaAS seed-0 controller used solely to
resume its interrupted pilot held-out test.

For the full execution order, recovery procedure, formal gates, and failure
handling, read [the continuation runbook](CONTINUATION_RUNBOOK_zh.md).

## Included

- EC-1 native AFlow, MaAS, and RPAS adapters; frozen HumanEval/AFlow fixtures;
  checksums in `data/ec1_humaneval/DATASET_MANIFEST.json`.
- EC-2 v2 implementation, protocol, and its official G-Designer integration.
- EC-3 V3 protocol, preflight/formal gates, and the fixed AFlow-derived
  HotpotQA validate/test fixtures with provenance in
  `data/ec3_hotpotqa/DATASET_MANIFEST.md`.
- A resumable EC-1 seed-0 pilot snapshot under `handoff_progress/`; it includes
  AFlow's generated workflow history and MaAS's compact trained controller so
  an interrupted held-out test does not retrain the controller.
- Tests, configuration snapshots, runners, and pinned external-source commits.

## Deliberately Excluded

- Qwen3.5-9B weights, tokenizer cache, and any other model artifact.
- API keys, environment files, logs, generated predictions, checkpoints,
  search workspaces, and result aggregates.
- EC-2 MMLU data. Obtain it according to its upstream license; do not use
  legacy outputs as paper-main-table results.
- EC-3 `D_calib=40`. Materialize it from official HotpotQA distractor `train`,
  verify its IDs are disjoint from the checked-in 1,000 fixture IDs, and freeze
  its hash before calibration. Do not replace it with arbitrary web data.

## Clean Setup

```bash
git clone https://github.com/ChiangYuhsin/RPAS-EC2-A100.git
cd RPAS-EC2-A100
bash scripts/bootstrap_external_comparison.sh --install-python-deps
```

The bootstrap script checks out the official sources exactly at:

| Baseline | Repository | Commit |
| --- | --- | --- |
| AFlow | `FoundationAgents/AFlow` | `3f457218fc716093fe53f6df8a5d5e6379d66346` |
| MaAS | `bingreeky/MaAS` | `987f3c1bc9a96e844fe090db3791446e3ef0f5c7` |
| G-Designer | `yanweiyue/GDesigner` | `a6efcfa3b40bb4d9cbf46f883a95d62020bd8251` |

Install the CUDA-compatible PyTorch distribution for the target driver before
launching the local Transformers service. The bootstrap script intentionally
does not choose a CUDA wheel, because that choice depends on the target
machine's driver and CUDA runtime.

Set the local model location. Never commit it or put credentials in a config:

```bash
export RPAS_MODEL_PATH=/absolute/path/to/Qwen3.5-9B
export RPAS_EC1_PYTHON="$PWD/.rpas-run/bin/python"
```

For the EC-1 service launcher, either make `models/Qwen3.5-9B` a symlink to
that location or invoke `scripts/serve_transformers_qwen35_openai.py` directly
with `--model "$RPAS_MODEL_PATH"`. The latter is shown below.

## GPU 4 and 5 Binding

Every launch must name one physical GPU. The EC-1 and EC-3 scripts reject any
GPU other than `4` or `5`; never set `CUDA_VISIBLE_DEVICES` to a list.

Terminal A, physical GPU 4:

```bash
CUDA_VISIBLE_DEVICES=4 .rpas-run/bin/python scripts/serve_transformers_qwen35_openai.py \
  --model "$RPAS_MODEL_PATH" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 29500 --max-new-tokens 1024
```

Terminal B, physical GPU 5:

```bash
CUDA_VISIBLE_DEVICES=5 .rpas-run/bin/python scripts/serve_transformers_qwen35_openai.py \
  --model "$RPAS_MODEL_PATH" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 29501 --max-new-tokens 1024
```

Then run exactly one worker against its matching resident service, for example:

```bash
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_EXTERNAL_API_BASE=http://127.0.0.1:29500/v1
export RPAS_EC1_SEED=0
bash scripts/run_ec1_native.sh aflow 4 \
  data/ec1_humaneval/official/humaneval.jsonl \
  data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
  outputs/ec1_native pilot
```

Run `python -m pytest -q` before dispatch. Use
`external_comparison.runners.ec1_preflight` before EC-1, and
`experiments/run_ec3_hotpotqa_v3.sh calibration` before EC-3. The latter will
remain blocked until the official, disjoint `D_calib` source is materialized.

## Result Integrity

The checked-in EC-1 artifacts are pilot infrastructure, not a completed
three-seed main table. The legacy EC-2 results are development-only; EC-2 v2
requires official G-Designer training and full-reflection RPAS-Comm. Do not
promote any run to a paper result until every protocol gate passes and the
frozen held-out test is first accessed only after final-state hashing.

## Restoring the Seed-0 Pilot

The snapshot preserves progress only; it is not a formal result. AFlow's
completed pilot result is retained for audit. To continue the interrupted MaAS
held-out test without repeating its completed controller training:

```bash
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_EXTERNAL_API_BASE=http://127.0.0.1:29501/v1
export RPAS_EC1_SEED=0
export RPAS_MAAS_TEST_ONLY=1
export RPAS_MAAS_CONTROLLER_PATH="$PWD/handoff_progress/ec1_seed_0/maas/HumanEval_controller_sample4.pth"
bash scripts/run_ec1_native.sh maas 5 \
  data/ec1_humaneval/official/humaneval.jsonl \
  data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
  outputs/ec1_native_resume pilot
```

This launches the official MaAS test branch against the preserved controller;
it does not call `Optimizer.optimize("Graph")`. Do not use that continuation
to replace a required clean formal seed.
