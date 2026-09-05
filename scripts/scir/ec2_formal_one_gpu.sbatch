#!/usr/bin/env bash
# EC-2 v2 formal MMLU: one isolated topology/method and seed per A100 task.
# Array indices: 0..14 = five methods x seeds 0..2.
#SBATCH --job-name=rpas-ec2-formal
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# A100 PCIe 40GB is sufficient for the frozen Qwen3.5-9B FP16 endpoint and
# exposes otherwise idle 40GB slots while the 80GB pool is busy.
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-14%6
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.err

set -euo pipefail

repo_root="/home/jianbaizhao/RPAS-EC2-A100"
model_path="/home/jianbaizhao/model/Qwen/Qwen3.5-9B"
python_bin="/home/jianbaizhao/qwen_cuda126/bin/python"
data_dir="/home/jianbaizhao/RPAS/data/mmlu"
gdesigner_root="/home/jianbaizhao/external_baselines/GDesigner"
embedding_model="/home/jianbaizhao/model/sentence-transformers/all-MiniLM-L6-v2"
config_path="${repo_root}/experiments/phase2_mmlu_qwen35_9b.json"
methods=(single_agent full_connected chain gdesigner rpas_comm)
task_id="${SLURM_ARRAY_TASK_ID:?array task id is required}"
method="${methods[$((task_id / 3))]}"
seed="$((task_id % 3))"
gpu="${CUDA_VISIBLE_DEVICES:-}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "expected one allocated GPU token, got ${gpu:-<unset>}" >&2; exit 2; }
for path in "${model_path}" "${data_dir}" "${gdesigner_root}" "${embedding_model}"; do
  [[ -d "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
[[ -f "${config_path}" ]] || { echo "missing config: ${config_path}" >&2; exit 2; }

job_root="${repo_root}/outputs/scir_ec2_formal/${method}/seed_${seed}"
output_root="${repo_root}/outputs/scir_ec2_formal"
port=$((29700 + SLURM_JOB_ID % 900))
mkdir -p "${job_root}/logs" "${output_root}"
cd "${repo_root}"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export RPAS_SCIR_ALLOCATED_GPU=1
export RPAS_CUDA_VISIBLE_DEVICES="${gpu}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:${port}/v1"
export RPAS_EXTERNAL_API_KEY=EMPTY
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_MODEL_CONFIG="${config_path}"
export RPAS_TOKENIZER_PATH="${model_path}"
export RPAS_GDESIGNER_EMBEDDING_MODEL="${embedding_model}"
export RPAS_MMLU_MAX_TOKENS=256
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
  find "${data_dir}" -maxdepth 2 -type f -name '*.parquet' | sort
  sha256sum "${config_path}"
  "${python_bin}" -m pip show torch transformers litellm datasets sentence-transformers
  git -C "${gdesigner_root}" rev-parse HEAD
} | tee "${job_root}/environment.txt"

"${python_bin}" -m py_compile \
  external_comparison/runners/ec2_v2.py \
  external_comparison/runners/aggregate_mmlu_v2.py
"${python_bin}" -m external_comparison.runners.validate_protocol \
  --config-dir external_comparison/configs

"${python_bin}" scripts/serve_transformers_qwen35_openai.py \
  --model "${model_path}" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "${port}" --max-new-tokens 256 \
  --max-batch-size 4 --batch-wait-ms 25 --dtype float16 --stop-string '<<RPAS_END>>' \
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

"${python_bin}" -m external_comparison.runners.ec2_v2 \
  --repo-root "${repo_root}" --gdesigner-root "${gdesigner_root}" \
  --data-dir "${data_dir}" --output-dir "${output_root}" \
  --method "${method}" --seed "${seed}" --data-seed 2026 \
  --search-per-subject 1 --select-per-subject 1 --test-per-subject 10 \
  >"${job_root}/logs/runner.log" 2>&1

test_result="${output_root}/${method}/seed_${seed}/result.json"
[[ -s "${test_result}" ]] || { echo "missing formal result: ${test_result}" >&2; exit 1; }
"${python_bin}" - "${test_result}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("formal_result") is not False:
    raise SystemExit("EC-2 result must remain false until three-seed aggregate gate")
for key in ("calls", "search_trace", "split_manifest_sha256"):
    if key not in payload and key != "calls":
        raise SystemExit(f"missing EC-2 audit field: {key}")
PY
echo "formal_result=false; seed_complete=true" >"${job_root}/job_status.txt"
find "${job_root}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"${job_root}/SHA256SUMS"
echo "EC-2 formal ${method} seed ${seed} completed: ${job_root}"
