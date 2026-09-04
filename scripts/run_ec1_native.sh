#!/usr/bin/env bash
set -euo pipefail

# One job per invocation.  The caller must choose GPU 4 or GPU 5 explicitly.
if [[ $# -lt 5 ]]; then
  echo "usage: $0 <aflow|maas|rpas> <4|5> <humaneval.jsonl> <public_test.jsonl> <output_dir> [pilot|formal]" >&2
  exit 2
fi

method="$1"
gpu="$2"
dataset="$3"
public_test="$4"
output="$5"
run_kind="${6:-pilot}"
if [[ "$gpu" != "4" && "$gpu" != "5" ]]; then
  echo "EC-1 permits only GPU 4 or GPU 5" >&2
  exit 2
fi

export RPAS_EC1_GPU="$gpu"
export CUDA_VISIBLE_DEVICES="$gpu"
python_bin="${RPAS_EC1_PYTHON:-}"
if [[ -z "$python_bin" && -x ".rpas-run/bin/python" ]]; then
  python_bin=".rpas-run/bin/python"
fi
python_bin="${python_bin:-python}"
command=("$python_bin" -m external_comparison.runners.native_humaneval \
  --repo-root . \
  --method "$method" --seed "${RPAS_EC1_SEED:-0}" --dataset-path "$dataset" \
  --public-test-path "$public_test" --output-dir "$output" --run-kind "$run_kind")
validate_fixture="${RPAS_EC1_AFLOW_VALIDATE_PATH:-data/ec1_humaneval/aflow/humaneval_validate.jsonl}"
test_fixture="${RPAS_EC1_AFLOW_TEST_PATH:-data/ec1_humaneval/aflow/humaneval_test.jsonl}"
command+=(--aflow-validate-path "$validate_fixture" --aflow-test-path "$test_fixture")
if [[ "$run_kind" == "formal" ]]; then
  : "$validate_fixture"
  : "$test_fixture"
fi
"${command[@]}"
