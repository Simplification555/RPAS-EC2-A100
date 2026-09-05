#!/usr/bin/env bash
#SBATCH --job-name=rpas-qwen-h100-smoke
#SBATCH --partition=compute
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.err

set -euo pipefail

repo_root="/home/jianbaizhao/RPAS-EC2-A100"
model_path="/home/jianbaizhao/model/Qwen/Qwen3.5-9B"
python_bin="/home/jianbaizhao/qwen_cuda126/bin/python"
port="29680"
artifact_dir="${repo_root}/outputs/scir_h100_smoke/${SLURM_JOB_ID}"
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
  echo "protocol_amendment=SCIR_H100_FP16_SINGLE_GPU_20260905"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
} | tee "${artifact_dir}/environment.txt"

"${python_bin}" -c 'import fastapi, torch, transformers, uvicorn; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))' \
  | tee "${artifact_dir}/imports.txt"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${python_bin}" \
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

ready=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${port}/health" >"${artifact_dir}/health.json"; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -n 160 "${artifact_dir}/server.log" >&2 || true
    exit 1
  fi
  sleep 5
done
if [[ "${ready}" -ne 1 ]]; then
  echo "server did not become healthy" >&2
  tail -n 160 "${artifact_dir}/server.log" >&2 || true
  exit 1
fi

curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"Reply with exactly: READY"}],"temperature":0,"max_tokens":16}' \
  >"${artifact_dir}/completion.json"
"${python_bin}" - "${artifact_dir}/completion.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
content = payload["choices"][0]["message"]["content"].strip()
assert content, payload
assert payload["usage"]["total_tokens"] > 0, payload
print(json.dumps({"content": content, "usage": payload["usage"]}, ensure_ascii=False))
PY

find "${artifact_dir}" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"${artifact_dir}/SHA256SUMS"
echo "SCIR H100 Qwen smoke passed: ${artifact_dir}"
