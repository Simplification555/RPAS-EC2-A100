#!/usr/bin/env bash
# EC-3 formal held-out test: one method/seed per allocated GPU.
#SBATCH --job-name=ec3-formal-test
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.err

set -euo pipefail
repo_root="/home/jianbaizhao/RPAS-EC2-A100"
python_bin="/home/jianbaizhao/qwen_cuda126/bin/python"
model_path="/home/jianbaizhao/model/Qwen/Qwen3.5-9B"
aflow_root="/home/jianbaizhao/external_baselines/AFlow_formal_clean"
manifest="${repo_root}/data/ec3_hotpotqa/formal/hotpotqa_manifest.json"
methods=(rpas aflow)
task_id="${SLURM_ARRAY_TASK_ID:?array task id is required}"
method="${methods[$((task_id / 3))]}"
seed="$((task_id % 3))"
gpu="${CUDA_VISIBLE_DEVICES:-}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "expected one allocated GPU token" >&2; exit 2; }
root="${repo_root}/outputs/ec3_hotpotqa_v3"
job_root="${repo_root}/outputs/scir_ec3_test/${method}/seed_${seed}_${SLURM_JOB_ID}"
port=$((29900 + SLURM_JOB_ID % 800))
mkdir -p "${job_root}/logs" "${root}"
cd "${repo_root}"
run_started_at_epoch="$(date +%s.%N)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export RPAS_SCIR_ALLOCATED_GPU=1
export CUDA_VISIBLE_DEVICES="${gpu}"
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:${port}/v1"
export RPAS_EXTERNAL_API_KEY=EMPTY
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_EC3_EXECUTOR_MAX_TOKENS=512
export RPAS_EC3_META_MAX_TOKENS=4096
export GEPA_QWEN_DISABLE_THINKING=1
export PYTHONUNBUFFERED=1

{
  date --iso-8601=seconds
  echo "run_started_at_epoch=${run_started_at_epoch}"
  hostname
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_task_id=${task_id}"
  echo "method=${method}"
  echo "seed=${seed}"
  echo "cuda_visible_devices=${gpu}"
  echo "run_kind=formal_test"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
  sha256sum "${manifest}"
  "${python_bin}" -m pip show torch transformers litellm datasets sentence-transformers
} | tee "${job_root}/environment.txt"

"${python_bin}" scripts/serve_transformers_qwen35_openai.py --model "${model_path}" \
  --served-model-name Qwen/Qwen3.5-9B --host 127.0.0.1 --port "${port}" \
  --max-new-tokens 4096 --max-batch-size 1 --batch-wait-ms 0 --dtype float16 \
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
if [[ "${method}" == "aflow" ]]; then
  command=("${python_bin}" -m external_comparison.runners.native_ec3_aflow --repo-root "${repo_root}" --manifest "${manifest}" --output-root "${root}" --aflow-root "${aflow_root}" test --seed "${seed}")
else
  command=("${python_bin}" -m external_comparison.runners.ec3_v3 --repo-root "${repo_root}" --manifest "${manifest}" --output-root "${root}" --aflow-root "${aflow_root}" test --seed "${seed}")
fi
printf '%q ' "${command[@]}" >"${job_root}/runner_command.txt"
printf '\n' >>"${job_root}/runner_command.txt"
"${command[@]}" >"${job_root}/logs/runner.log" 2>&1
summary="${root}/${method}/seed_${seed}/test_summary.json"
[[ -s "${summary}" ]] || { echo "missing held-out summary: ${summary}" >&2; exit 1; }
"${python_bin}" -m external_comparison.runners.write_run_metrics \
  --result-dir "${root}/${method}/seed_${seed}" --job-id "${SLURM_JOB_ID}" --job-root "${job_root}" --stage formal_test
echo "formal_result=true; d_test_accessed=true" >"${job_root}/job_status.txt"
find "${job_root}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"${job_root}/SHA256SUMS"
echo "EC-3 formal test ${method} seed ${seed} completed: ${summary}"
