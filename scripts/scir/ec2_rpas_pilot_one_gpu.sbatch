#!/usr/bin/env bash
# EC-2 v2 G-Designer and RPAS-Comm seed-0 pilot on one Slurm-allocated H100.
# This is a pilot artifact and is never a paper result.
#SBATCH --job-name=rpas-ec2-pilot
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.err

set -euo pipefail
repo_root="/home/jianbaizhao/RPAS-EC2-A100"
model_path="/home/jianbaizhao/model/Qwen/Qwen3.5-9B"
python_bin="/home/jianbaizhao/qwen_cuda126/bin/python"
gpu="${CUDA_VISIBLE_DEVICES:-}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "expected one Slurm GPU token, got ${gpu:-<unset>}" >&2; exit 2; }
job_root="${repo_root}/outputs/scir_ec2_pilot/${SLURM_JOB_ID}"
port=$((29790 + SLURM_JOB_ID % 1000))
mkdir -p "${job_root}/logs"
cd "${repo_root}"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export RPAS_SCIR_ALLOCATED_GPU=1
export CUDA_VISIBLE_DEVICES="${gpu}"
export RPAS_CUDA_VISIBLE_DEVICES="${gpu}"
export RPAS_REPO_ROOT="${repo_root}"
export RPAS_MODEL_PATH="${model_path}"
export RPAS_MMLU_DATA_DIR=/home/jianbaizhao/RPAS/data/mmlu
export RPAS_GDESIGNER_ROOT=/home/jianbaizhao/external_baselines/GDesigner
export RPAS_GDESIGNER_EMBEDDING_MODEL=/home/jianbaizhao/model/sentence-transformers/all-MiniLM-L6-v2
export RPAS_PYTHON_BIN="${python_bin}"
export RPAS_SERVER_PYTHON_BIN="${python_bin}"
export RPAS_SERVICE_PORT="${port}"
export RPAS_SERVICE_LOG="${job_root}/logs/service.log"
export RPAS_OUTPUT_DIR="${job_root}/result"
export RPAS_EXTERNAL_API_KEY=EMPTY
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_MMLU_MAX_TOKENS=256
export GEPA_QWEN_DISABLE_THINKING=1
export GEPA_PHASE2_PROMPT_MODE=deliberate

{
  date --iso-8601=seconds
  hostname
  echo "job_id=${SLURM_JOB_ID}"
  echo "cuda_visible_devices=${gpu}"
  echo "protocol_amendment=SCIR_ALLOCATED_GPU_PILOT_ONLY"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
  find /home/jianbaizhao/RPAS/data/mmlu -maxdepth 2 -type f -name '*.parquet' | wc -l
  "${python_bin}" -m pip show torch transformers litellm datasets sentence-transformers
  git -C /home/jianbaizhao/external_baselines/GDesigner rev-parse HEAD
} | tee "${job_root}/environment.txt"

"${python_bin}" -m py_compile external_comparison/runners/ec2_v2.py external_comparison/runners/aggregate_mmlu_v2.py
"${python_bin}" scripts/serve_transformers_qwen35_openai.py --model "${model_path}" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "${port}" --max-new-tokens 256 --max-batch-size 4 --batch-wait-ms 25 --dtype float16 \
  --stop-string '<<RPAS_END>>' >"${job_root}/logs/service.log" 2>&1 &
service_pid=$!
cleanup() { kill "${service_pid}" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${port}/health" >"${job_root}/health.json" && break
  kill -0 "${service_pid}" 2>/dev/null || { tail -n 160 "${job_root}/logs/service.log" >&2; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:${port}/v1/models" >"${job_root}/models.json"

for method in gdesigner rpas_comm; do
  "${python_bin}" -m external_comparison.runners.ec2_v2 --repo-root "${repo_root}" \
    --gdesigner-root "${RPAS_GDESIGNER_ROOT}" --data-dir "${RPAS_MMLU_DATA_DIR}" \
    --output-dir "${RPAS_OUTPUT_DIR}" --method "${method}" --seed 0 \
    --search-per-subject 1 --select-per-subject 1 --test-per-subject 1 \
    >>"${job_root}/logs/runner.log" 2>&1
done
echo "formal_result=false" >"${job_root}/job_status.txt"
find "${job_root}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"${job_root}/SHA256SUMS"
echo "EC-2 v2 G-Designer/RPAS-Comm seed-0 pilot completed: ${job_root}"
