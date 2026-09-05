#!/usr/bin/env bash
# One EC-3 V3 method/stage per Slurm-allocated 32GB-or-larger GPU.
# The formal runner itself refuses D_test until ec3_formal_gate unlocks it.
#SBATCH --job-name=rpas-ec3-v3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --mem=40G
# AFlow search/select and held-out evaluation can exceed two hours on a
# serialized 9B endpoint; retain a six-hour ceiling for reproducible runs.
#SBATCH --time=06:00:00
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.err

set -euo pipefail

repo_root="/home/jianbaizhao/RPAS-EC2-A100"
model_path="/home/jianbaizhao/model/Qwen/Qwen3.5-9B"
python_bin="/home/jianbaizhao/qwen_cuda126/bin/python"
aflow_root="/home/jianbaizhao/external_baselines/AFlow"
manifest_path="${repo_root}/data/ec3_hotpotqa/formal/hotpotqa_manifest.json"
method="${RPAS_EC3_METHOD:?set RPAS_EC3_METHOD to aflow or rpas}"
stage="${RPAS_EC3_STAGE:?set RPAS_EC3_STAGE to calibration, pretest, test, or single-test}"
seed="${RPAS_EC3_SEED:-0}"
gpu="${CUDA_VISIBLE_DEVICES:-}"

[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "expected exactly one Slurm GPU token, got ${gpu:-<unset>}" >&2; exit 2; }
[[ "${method}" == "aflow" || "${method}" == "rpas" ]] || { echo "invalid EC-3 method: ${method}" >&2; exit 2; }
case "${stage}" in calibration|pretest|test|single-test) ;; *) echo "invalid EC-3 stage: ${stage}" >&2; exit 2 ;; esac
[[ "${stage}" != "single-test" || "${method}" == "rpas" ]] || { echo "single-test is owned by the RPAS runner" >&2; exit 2; }
[[ "${stage}" == "calibration" || "${seed}" =~ ^[012]$ ]] || { echo "EC-3 seed must be 0, 1, or 2" >&2; exit 2; }

gpu_memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
[[ "${gpu_memory_mib}" =~ ^[0-9]+$ ]] || { echo "unable to determine allocated GPU memory" >&2; exit 2; }
(( gpu_memory_mib >= 30000 )) || { echo "Qwen3.5-9B EC-3 requires at least 30GB, found ${gpu_memory_mib}MiB" >&2; exit 2; }

job_root="${repo_root}/outputs/scir_ec3_v3/${SLURM_JOB_ID}"
formal_root="${repo_root}/outputs/ec3_hotpotqa_v3"
port=$((29890 + SLURM_JOB_ID % 900))
mkdir -p "${job_root}/logs" "${formal_root}"
cd "${repo_root}"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export RPAS_SCIR_ALLOCATED_GPU=1
export CUDA_VISIBLE_DEVICES="${gpu}"
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:${port}/v1"
export RPAS_EXTERNAL_API_KEY=EMPTY
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_EC3_EXECUTOR_MAX_TOKENS="${RPAS_EC3_EXECUTOR_MAX_TOKENS:-256}"
export RPAS_EC3_META_MAX_TOKENS="${RPAS_EC3_META_MAX_TOKENS:-4096}"
export RPAS_CODE_COMMIT="${RPAS_CODE_COMMIT:-bundle_without_git_metadata}"
export GEPA_QWEN_DISABLE_THINKING=1
export PYTHONUNBUFFERED=1

{
  date --iso-8601=seconds
  hostname
  echo "job_id=${SLURM_JOB_ID}"
  echo "method=${method}"
  echo "stage=${stage}"
  echo "seed=${seed}"
  echo "cuda_visible_devices=${gpu}"
  echo "formal_root=${formal_root}"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
  sha256sum "${manifest_path}"
  # SCIR receives a source bundle rather than a Git working tree. Avoid a
  # failing Git probe inside this pipefail pipeline while still retaining a
  # reproducible provenance identifier in every artifact.
  if repo_commit="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null)"; then
    echo "repo_commit=${repo_commit}"
  else
    echo "bundle_code_commit=${RPAS_CODE_COMMIT}"
  fi
  git -C "${aflow_root}" rev-parse HEAD
  git -C "${aflow_root}" status --porcelain --untracked-files=no
  "${python_bin}" -m pip show torch transformers litellm pandas
} | tee "${job_root}/environment.txt"

"${python_bin}" -m py_compile \
  external_comparison/runners/ec3_v3.py \
  external_comparison/runners/native_ec3_aflow.py \
  external_comparison/runners/ec3_preflight.py \
  external_comparison/runners/ec3_formal_gate.py

"${python_bin}" scripts/serve_transformers_qwen35_openai.py \
  --model "${model_path}" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "${port}" --max-new-tokens 4096 \
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

common=(--repo-root "${repo_root}" --manifest "${manifest_path}" --aflow-root "${aflow_root}" --output-root "${formal_root}")
if [[ "${method}" == "aflow" ]]; then
  command=("${python_bin}" -m external_comparison.runners.native_ec3_aflow "${common[@]}" "${stage}")
else
  command=("${python_bin}" -m external_comparison.runners.ec3_v3 "${common[@]}" "${stage}")
fi
if [[ "${stage}" != "calibration" ]]; then
  command+=(--seed "${seed}")
fi
printf '%q ' "${command[@]}" >"${job_root}/runner_command.txt"
printf '\n' >>"${job_root}/runner_command.txt"
"${command[@]}" >"${job_root}/logs/runner.log" 2>&1

echo "formal_result=false" >"${job_root}/job_status.txt"
find "${job_root}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"${job_root}/SHA256SUMS"
echo "EC-3 ${method} ${stage} completed: ${job_root}"
