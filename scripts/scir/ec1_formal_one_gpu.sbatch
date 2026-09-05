#!/usr/bin/env bash
# EC-1 formal HumanEval: one isolated method/seed per A100 task.
# Array indices: 0..8 = aflow, maas, rpas x seed 0..2.
# No task may read another seed's workspace or controller.
#SBATCH --job-name=rpas-ec1-formal
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Qwen3.5-9B FP16 plus the single-request endpoint fits comfortably on an
# A100 PCIe 40GB, which exposes more parallel slots than the 80GB pool.
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-8%6
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.err

set -euo pipefail

repo_root="/home/jianbaizhao/RPAS-EC2-A100"
model_path="/home/jianbaizhao/model/Qwen/Qwen3.5-9B"
python_bin="/home/jianbaizhao/qwen_cuda126/bin/python"
methods=(aflow maas rpas)
task_id="${SLURM_ARRAY_TASK_ID:?array task id is required}"
method="${methods[$((task_id / 3))]}"
seed="$((task_id % 3))"
gpu="${CUDA_VISIBLE_DEVICES:-}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "expected one allocated GPU token, got ${gpu:-<unset>}" >&2; exit 2; }

experiment_root="${repo_root}/outputs/scir_ec1_formal_v2"
job_root="${experiment_root}/_jobs/${method}/seed_${seed}_${SLURM_JOB_ID}"
port=$((29600 + SLURM_JOB_ID % 900))
dataset="${repo_root}/data/ec1_humaneval/official/humaneval.jsonl"
public_test="${repo_root}/data/ec1_humaneval/aflow/humaneval_public_test.jsonl"
validate_fixture="${repo_root}/data/ec1_humaneval/aflow/humaneval_validate.jsonl"
test_fixture="${repo_root}/data/ec1_humaneval/aflow/humaneval_test.jsonl"
mkdir -p "${job_root}/logs"
cd "${repo_root}"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export RPAS_SCIR_ALLOCATED_GPU=1
export RPAS_EC1_GPU="${gpu}"
export RPAS_EC1_SEED="${seed}"
export RPAS_EC1_PYTHON="${python_bin}"
export RPAS_AFLOW_ROOT="/home/jianbaizhao/external_baselines/AFlow_formal_clean"
export RPAS_MAAS_ROOT="/home/jianbaizhao/external_baselines/MaAS_formal_clean"
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:${port}/v1"
export RPAS_EXTERNAL_API_KEY=EMPTY
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_MODEL_DTYPE=float16
export RPAS_MAX_BATCH_SIZE=1
export RPAS_AFLOW_MAX_ROUNDS=2
export RPAS_AFLOW_SAMPLE=4
export RPAS_AFLOW_VALIDATION_ROUNDS=1
export RPAS_MAAS_SAMPLE=4
export RPAS_EC1_RPAS_SEED_CANDIDATES=4
export RPAS_EC1_RPAS_NEW_CANDIDATES=3
export GEPA_QWEN_DISABLE_THINKING=1
export GEPA_PHASE2_PROMPT_MODE=deliberate
export PYTHONUNBUFFERED=1

{
  date --iso-8601=seconds
  hostname
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_task_id=${task_id}"
  echo "method=${method}"
  echo "seed=${seed}"
  echo "cuda_visible_devices=${gpu}"
  echo "run_kind=formal"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
  sha256sum "${dataset}" "${public_test}" "${validate_fixture}" "${test_fixture}"
  "${python_bin}" -m pip show torch transformers litellm datasets sentence-transformers
} | tee "${job_root}/environment.txt"

"${python_bin}" -m external_comparison.runners.ec1_preflight \
  --dataset-path "${dataset}" \
  --public-test-path "${public_test}" \
  --aflow-validate-path "${validate_fixture}" \
  --aflow-test-path "${test_fixture}"
"${python_bin}" -m py_compile \
  external_comparison/runners/native_humaneval.py \
  external_comparison/runners/native_ec1_driver.py \
  external_comparison/runners/native_rpas_ec1.py

"${python_bin}" scripts/serve_transformers_qwen35_openai.py \
  --model "${model_path}" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "${port}" --max-new-tokens 1024 \
  --max-batch-size 1 --batch-wait-ms 0 --dtype float16 --stop-string '<<RPAS_END>>' \
  >"${job_root}/logs/service.log" 2>&1 &
service_pid=$!
cleanup() { kill "${service_pid}" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${port}/health" >"${job_root}/health.json" && break
  kill -0 "${service_pid}" 2>/dev/null || { tail -n 160 "${job_root}/logs/service.log" >&2; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:${port}/v1/models" >"${job_root}/models.json"
curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"Reply exactly: OK"}],"max_tokens":4,"temperature":0.0}' \
  >"${job_root}/smoke_completion.json"

bash scripts/run_ec1_native.sh "${method}" "${gpu}" "${dataset}" "${public_test}" \
  "${experiment_root}" formal >"${job_root}/logs/runner.log" 2>&1

test_result="${experiment_root}/${method}/seed_${seed}/result.json"
[[ -s "${test_result}" ]] || { echo "missing formal result: ${test_result}" >&2; exit 1; }
"${python_bin}" - "${test_result}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("formal_result") is not True:
    raise SystemExit("EC-1 formal result flag is not true")
PY
"${python_bin}" -m external_comparison.runners.write_run_metrics \
  --result-dir "${experiment_root}/${method}/seed_${seed}" --job-id "${SLURM_JOB_ID}" \
  --job-root "${job_root}" --stage formal_seed
echo "formal_result=true" >"${job_root}/job_status.txt"
find "${job_root}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"${job_root}/SHA256SUMS"
echo "EC-1 formal ${method} seed ${seed} completed: ${job_root}"
