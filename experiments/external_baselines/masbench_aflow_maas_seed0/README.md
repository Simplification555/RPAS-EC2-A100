# Native AFlow + MaAS on MASBench (seed 0)

This package is a reproducible **single-seed, five-axis pilot** of the released AFlow and MaAS methods on the frozen Salesforce/MASBench data shipped here. It is not a three-seed paper table and it makes no across-seed variance/significance claim. The baseline implementations are fetched from their upstream repositories at pinned commits; only run-local compatibility/data/scoring adapters are patched. Each method/axis gets an isolated output tree.

## Locked protocol

- Methods: native AFlow and native MaAS; five axes: `breadth`, `depth`, `horizon`, `parallel`, `robustness`.
- One search seed only: `0` (pilot); frozen data seed: `2026`; each axis uses the canonical RPAS split of 24 `D_search`, 24 `D_select`, and 60 `D_test` examples.
- Model: `Qwen/Qwen3.5-9B`; context limit 8192; generation ceiling 6144; deterministic decoding; concurrency 8; vLLM TP=1, max-num-seqs=24, GPU memory utilization 0.92.
- Shared answer parser reports exact-match accuracy, component accuracy, and ordinal score 0/1/2 (wrong / partially correct / exact). Robustness explicitly preserves injected raw retrieval values while algebraic values remain in Z_23.
- AFlow uses its native MCTS/Graph optimizer with the fixed MATH/math operator configuration, sample 4, max rounds 8, and an 8-retry cap only for repeated/rejected mutations. MaAS uses its native Agentic Supernet/controller optimization, native MATH/math operators, one controller sample, one search round, batch 8, learning rate 0.01, no TextGrad, and its four native training repetitions. MaAS produces one native controller; its Q/E entries therefore reference the same selected controller rather than inventing a second method-specific policy. MaAS uses only the literal/format compatibility repair plus deterministic context guard; concise-v2 prompt rewrites are not applied. See `experiments/MAAS_HELPER_PROVENANCE.md`.
- The packaged pre-selection manifest contains only `D_search`/`D_select` IDs and file hashes; the optional builder reads only those two split files, never D_test or the per-axis manifests. The runner validates only `D_search`/`D_select` before search. It writes and hashes `selection_frozen.json` after the `D_select` decision, and only then reads `test_60.jsonl` and the full per-axis manifest. If a run is interrupted or selection is invalid, it fails closed and does not open D_test. Existing/partial run directories are never overwritten.

## Before using the RTX PRO 6000

The package was syntax/unit tested offline on the authoring machine, **not run on an RTX PRO 6000**. That GPU is a Blackwell, compute-capability 12.0 device according to [NVIDIA's CUDA GPU table](https://developer.nvidia.com/cuda/gpus). This is not proof that a particular PyTorch/vLLM/Qwen3.5 installation or these native method adapters work on that card. vLLM's [GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) describes supported CUDA/Linux environments, but actual compatibility still has to be established in the friend's exact driver/container.

Important: the upstream projects do not define a verified shared environment. This harness requires Python 3.10+; the upstream AFlow README's Python 3.9 suggestion alone is not sufficient for this wrapper. In particular, MaAS pins an old `torch==2.1.0+cu118` stack, while the pinned AFlow dependency set and current vLLM/PyTorch stack may also conflict. Do **not** blindly install either project's full requirements over the serving environment. Build a curated Linux environment/container with a PyTorch + vLLM build that supports the installed driver and SM 12.0, then install/test each adapter's dependencies without replacing that stack. The local preflight checks installed components, the local MaAS embedding config, and required vLLM flags but cannot replace a real model-load and one-request smoke test. No model or embedding weights are included; no implicit model downloads occur.

## Run on Linux

Unpack the ZIP into a clean directory, then set paths to locally available weights and a tested Python environment:

```bash
cd RPAS_MASBench_AFlow_MaAS_seed0
export RPAS_PYTHON="$HOME/venvs/rpas/bin/python"
export RPAS_VLLM="$HOME/venvs/rpas/bin/vllm"
export RPAS_MODEL_PATH="$HOME/models/Qwen3.5-9B"
export RPAS_MAAS_EMBEDDING_PATH="$HOME/models/all-MiniLM-L6-v2"
# If multiple GPUs are installed, expose just the intended RTX PRO 6000:
export CUDA_VISIBLE_DEVICES=0

bash scripts/setup_baselines.sh
# Run both methods sequentially on one GPU; no two method processes contend concurrently.
bash scripts/run_masbench_seed0.sh all
```

To run a single method, use `bash scripts/run_masbench_seed0.sh aflow` or `... maas`. `RPAS_OUTPUT_ROOT` and `RPAS_PORT` may be changed to a new location/port. A method-axis-seed directory is never overwritten: the runner skips it only if its stored gate and a fresh gate verification both pass; any partial/failed unit stops the run. If a unit was interrupted, preserve that root for diagnosis and use a fresh root for a clean rerun. Fully gated earlier units can safely be skipped if the same command is re-invoked. MaAS requires local Sentence-Transformers `all-MiniLM-L6-v2` weights.

The script launches vLLM on localhost, waits for the exact served model ID, evaluates each axis in order, runs the fail-closed quality gate after every run, and stops its own server on exit. `all` processes AFlow then MaAS serially, so only one GPU-backed service is used and the two methods do not contend concurrently. The runner records each native source commit/hash and exact run settings. It does not submit to a scheduler and has no global wall-time guarantee. It never resumes partial output directories: preserve and diagnose an interrupted directory, then use a fresh output root for a clean run.

The `all` runner automatically writes the aggregate after all 10 runs pass their individual gates. To revalidate an existing aggregate without overwriting it:

```bash
python experiments/aggregate_masbench_seed0.py \
  --root outputs/masbench_aflow_maas_seed0 --verify-existing
```

Aggregation refuses incomplete/failed/mismatched runs and writes `aggregate.json` and `aggregate.csv` in a new `seed0_summary` directory. The macro average is across the five axes for seed 0; it is **not** a standard deviation across seeds. Do not fill a three-seed mean±SD table from this package.

## Pre-run checks and tests

```bash
python -m py_compile experiments/*.py
PYTHONPATH="$PWD:$PWD/experiments:$PWD/external_comparison" \
  python -m unittest \
    experiments.test_native_masbench_external \
    experiments.test_masbench_answer_protocol \
    experiments.test_masbench_quality_gate \
    experiments.test_aggregate_masbench_seed0 -v
python experiments/masbench_runtime_smoke.py
```

`scripts/preflight_gpu.py` is also run automatically. Before the long run, the operator should manually start the same vLLM command and make one disposable local API request with a tiny prompt, then stop the server and use a fresh output root for the actual seed-0 jobs. The hardware/runtime compatibility test is intentionally left to the RTX PRO 6000 host.

## Source and data notes

- This is an external-baseline bundle: it does not include or run the RPAS search method. The benchmark task instruction is isolated in `experiments/masbench_task_prompt.py`; the RPAS optimizer file that originally held the same task-contract string is deliberately not included. `external_comparison/common/pareto.py` is the small generic Q/E operating-point selector, not an RPAS optimizer.

- AFlow: [FoundationAgents/AFlow](https://github.com/FoundationAgents/AFlow), pinned commit `3f457218fc716093fe53f6df8a5d5e6379d66346`.
- MaAS: [bingreeky/MaAS](https://github.com/bingreeky/MaAS), pinned commit `987f3c1bc9a96e844fe090db3791446e3ef0f5c7`.
- Third-party baseline code is cloned on the target machine, not copied into this ZIP. `scripts/setup_baselines.sh` refuses to replace an existing checkout unless it is already at the exact pinned revision.
- MASBench splits and manifests are byte-for-byte from [`JiangyueAnn/RPAS` at `e12f588`](https://github.com/JiangyueAnn/RPAS/tree/e12f58823be5f91a32f05f9af4d36e54838ffe59/data/masbench). All five official `search_24.jsonl`, `select_24.jsonl`, `test_60.jsonl`, and per-axis manifests are included and SHA-256 checked against the frozen manifest. The upstream dataset card declares Apache-2.0; users should retain attribution and comply with the [Salesforce/MASBench dataset card](https://huggingface.co/datasets/Salesforce/MASBench).
- This is a one-seed execution package, not evidence for the screenshot's three-seed mean±SD table. D_test is included because the RPAS canonical data branch contains it; do not use it for search, tuning, prompt changes, or candidate selection. Publicly sharing benchmark test examples may affect benchmark hygiene, even where the upstream dataset license permits redistribution.
- No separate license is asserted for this package's custom harness code. The method repositories and dataset keep their upstream licenses; see their linked source repositories/cards.
- Raw predictions and model outputs can contain benchmark material; keep result directories access-controlled.
