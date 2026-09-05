#!/usr/bin/env bash
#SBATCH --job-name=rpas-qwen-smoke
#SBATCH --partition=compute
#SBATCH --nodelist=gpu13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --mem=20G
#SBATCH --time=00:45:00
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.err

set -euo pipefail

repo_root="/home/jianbaizhao/RPAS-EC2-A100"
model_path="/home/jianbaizhao/model/Qwen/Qwen3.5-9B"
# Use the CUDA 12.6 environment: the CUDA 13 wheel used by the older smoke
# test dropped V100 (sm70) kernels and fails at generation time.
server_python="/home/jianbaizhao/qwen_cuda126/bin/python"
runner_python="/home/jianbaizhao/qwen_cuda126/bin/python"
port="29680"
artifact_dir="${repo_root}/outputs/scir_smoke/${SLURM_JOB_ID}"
mkdir -p "${artifact_dir}"
cd "${repo_root}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "expected exactly one Slurm-assigned GPU, got ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 2
fi

{
  date --iso-8601=seconds
  hostname
  echo "job_id=${SLURM_JOB_ID}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "protocol_amendment=SCIR_V100_FP16_SINGLE_GPU_20260905"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
} | tee "${artifact_dir}/environment.txt"

echo "server imports started $(date --iso-8601=seconds)" | tee -a "${artifact_dir}/environment.txt"
timeout 600 "${server_python}" -c \
  'import fastapi, torch, transformers, uvicorn; print(torch.__version__, transformers.__version__)' \
  | tee "${artifact_dir}/server_imports.txt"
echo "runner imports started $(date --iso-8601=seconds)" | tee -a "${artifact_dir}/environment.txt"
PYTHONPATH="${repo_root}" timeout 600 "${runner_python}" -c \
  'import torch; import external_comparison.runners.ec2_v2; import external_comparison.runners.native_rpas_ec1; print(torch.__version__)' \
  | tee "${artifact_dir}/runner_imports.txt"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${server_python}" \
  scripts/serve_transformers_qwen35_openai.py \
  --model "${model_path}" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "${port}" --max-new-tokens 64 \
  --max-batch-size 1 --batch-wait-ms 0 --dtype float16 \
  >"${artifact_dir}/server.log" 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${port}/health" >"${artifact_dir}/health.json"; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -n 160 "${artifact_dir}/server.log" >&2 || true
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${port}/health" | tee "${artifact_dir}/health_final.json"
curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"Reply with exactly: READY"}],"temperature":0,"max_tokens":16}' \
  >"${artifact_dir}/completion.json"
"${runner_python}" - "${artifact_dir}/completion.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["choices"][0]["message"]["content"].strip(), payload
assert payload["usage"]["total_tokens"] > 0, payload
print(json.dumps(payload, ensure_ascii=False))
PY
find "${artifact_dir}" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"${artifact_dir}/SHA256SUMS"
echo "SCIR Qwen smoke passed: ${artifact_dir}"
