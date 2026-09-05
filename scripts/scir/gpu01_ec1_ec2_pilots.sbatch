#!/usr/bin/env bash
#SBATCH --job-name=rpas-ec12-pilots
#SBATCH --partition=compute
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.err

set -euo pipefail

repo_root="/home/jianbaizhao/RPAS-EC2-A100"
model_path="/home/jianbaizhao/model/Qwen/Qwen3.5-9B"
server_python="/home/jianbaizhao/qwen_cuda126/bin/python"
runner_python="/home/jianbaizhao/qwen_cuda126/bin/python"
aflow_root="/home/jianbaizhao/external_baselines/AFlow"
gdesigner_root="/home/jianbaizhao/external_baselines/GDesigner"
mmlu_data="/home/jianbaizhao/RPAS/data/mmlu"
embedding_model="/home/jianbaizhao/model/sentence-transformers/all-MiniLM-L6-v2"
job_root="${repo_root}/outputs/scir_pilots/${SLURM_JOB_ID}"
mkdir -p "${job_root}" "${job_root}/logs"
cd "${repo_root}"

IFS=',' read -r -a gpu_tokens <<<"${CUDA_VISIBLE_DEVICES:-}"
if [[ "${#gpu_tokens[@]}" -ne 2 || -z "${gpu_tokens[0]}" || -z "${gpu_tokens[1]}" ]]; then
  echo "expected two Slurm-assigned GPU tokens, got ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 2
fi
if [[ "${gpu_tokens[0]}" == "${gpu_tokens[1]}" ]]; then
  echo "Slurm returned duplicate GPU tokens" >&2
  exit 2
fi

{
  date --iso-8601=seconds
  hostname
  echo "job_id=${SLURM_JOB_ID}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "protocol_amendment=SCIR_H100_FP16_SINGLE_GPU_20260905"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
  sha256sum data/ec1_humaneval/official/humaneval.jsonl
  sha256sum data/ec1_humaneval/aflow/humaneval_validate.jsonl
  sha256sum data/ec1_humaneval/aflow/humaneval_test.jsonl
  sha256sum data/ec1_humaneval/aflow/humaneval_public_test.jsonl
  git -C "${aflow_root}" rev-parse HEAD
  git -C "${gdesigner_root}" rev-parse HEAD
  git -C "${aflow_root}" status --porcelain --untracked-files=no
  git -C "${gdesigner_root}" status --porcelain --untracked-files=no
} | tee "${job_root}/environment.txt"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export RPAS_SCIR_PROTOCOL_AMENDMENT=1
export RPAS_EXTERNAL_API_KEY=EMPTY
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_MODEL_DTYPE=float16
export RPAS_MAX_BATCH_SIZE=1
export GEPA_QWEN_DISABLE_THINKING=1
export GEPA_PHASE2_PROMPT_MODE=deliberate

run_ec1() {
  local gpu="$1"
  local port="$2"
  local service_log="${job_root}/logs/ec1_service.log"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export RPAS_EC1_GPU="${gpu}"
  export RPAS_EC1_PYTHON="${runner_python}"
  export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:${port}/v1"
  export RPAS_EC1_SEED=0

  "${server_python}" scripts/serve_transformers_qwen35_openai.py \
    --model "${model_path}" --served-model-name Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port "${port}" --max-new-tokens 1024 \
    --max-batch-size 1 --batch-wait-ms 0 --dtype float16 --stop-string '<<RPAS_END>>' \
    >"${service_log}" 2>&1 &
  local service_pid=$!
  trap 'kill "${service_pid}" 2>/dev/null || true' RETURN
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${port}/health" >"${job_root}/ec1_health.json"; then break; fi
    if ! kill -0 "${service_pid}" 2>/dev/null; then tail -n 160 "${service_log}" >&2; return 1; fi
    sleep 5
  done
  curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null
  bash scripts/run_ec1_native.sh rpas "${gpu}" \
    data/ec1_humaneval/official/humaneval.jsonl \
    data/ec1_humaneval/aflow/humaneval_public_test.jsonl \
    "${job_root}/ec1_rpas_seed0" pilot
}

run_ec2() {
  local gpu="$1"
  local port="$2"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export RPAS_CUDA_VISIBLE_DEVICES="${gpu}"
  export RPAS_REPO_ROOT="${repo_root}"
  export RPAS_MODEL_PATH="${model_path}"
  export RPAS_MMLU_DATA_DIR="${mmlu_data}"
  export RPAS_GDESIGNER_ROOT="${gdesigner_root}"
  export RPAS_GDESIGNER_EMBEDDING_MODEL="${embedding_model}"
  export RPAS_PYTHON_BIN="${runner_python}"
  export RPAS_SERVER_PYTHON_BIN="${server_python}"
  export RPAS_SERVICE_PORT="${port}"
  export RPAS_SERVICE_LOG="${job_root}/logs/ec2_service.log"
  export RPAS_OUTPUT_DIR="${job_root}/ec2_mmlu_v2"
  bash experiments/run_ec2_mmlu_v2_a100.sh pilot
}

(run_ec1 "${gpu_tokens[0]}" 29690) >"${job_root}/logs/ec1_worker.log" 2>&1 &
ec1_pid=$!
# Avoid simultaneous 9B checkpoint materialization inside the 20 GB job cgroup.
sleep 180
(run_ec2 "${gpu_tokens[1]}" 29691) >"${job_root}/logs/ec2_worker.log" 2>&1 &
ec2_pid=$!

set +e
wait "${ec1_pid}"
ec1_status=$?
wait "${ec2_pid}"
ec2_status=$?
set -e

{
  echo "ec1_status=${ec1_status}"
  echo "ec2_status=${ec2_status}"
  echo "formal_result=false"
} | tee "${job_root}/job_status.txt"
find "${job_root}" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"${job_root}/SHA256SUMS"
if [[ "${ec1_status}" -ne 0 || "${ec2_status}" -ne 0 ]]; then
  exit 1
fi
echo "EC-1 and EC-2 seed-0 pilots completed: ${job_root}"
