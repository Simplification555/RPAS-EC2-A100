#!/usr/bin/env bash
# EC-1B protocol-v4 one-seed pilot: Single, native AFlow, native RPAS-Full.
# Array indices: 0=Single, 1=AFlow, 2=RPAS-Full.
#SBATCH --job-name=rpas-ec1b-lcb
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.err

set -euo pipefail

repo_root="/home/jianbaizhao/RPAS-EC2-A100"
python_bin="/home/jianbaizhao/qwen_cuda126/bin/python"
model_path="/home/jianbaizhao/model/Qwen/Qwen3.5-9B"
data_dir="${repo_root}/data/ec1_livecodebench_release_v6"
output_root="${RPAS_EC1_LCB_OUTPUT_ROOT:-${repo_root}/outputs/ec1_livecodebench_v4_seed0}"
methods=(single aflow rpas)
method="${methods[${SLURM_ARRAY_TASK_ID:?array task ID is required}]}"
gpu="${CUDA_VISIBLE_DEVICES:-}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "expected one Slurm GPU token, got ${gpu:-unset}" >&2; exit 2; }
port="$((31000 + SLURM_JOB_ID % 2000))"
job_root="${output_root}/_jobs/${method}_${SLURM_JOB_ID}"

cd "$repo_root"
mkdir -p "$job_root/logs" "${output_root}/_locks" logs/slurm
lock_dir="${output_root}/_locks/${method}_seed_0.lockdir"
mkdir "$lock_dir" 2>/dev/null || { echo "duplicate ${method} seed 0 job" >&2; exit 2; }
printf '%s\n' "${SLURM_JOB_ID}" >"${lock_dir}/owner_job_id"
[[ ! -e "${output_root}/${method}/seed_0/result.json" ]] || { echo "completed result already exists" >&2; exit 2; }

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export RPAS_SCIR_ALLOCATED_GPU=1
export RPAS_EC1_GPU="$gpu"
export CUDA_VISIBLE_DEVICES="$gpu"
export RPAS_AFLOW_ROOT="/home/jianbaizhao/external_baselines/AFlow_formal_clean"
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:${port}/v1"
export RPAS_EXTERNAL_API_KEY=EMPTY
export RPAS_EXTERNAL_MODEL=Qwen/Qwen3.5-9B
export RPAS_MODEL_DTYPE=float16
export RPAS_MAX_BATCH_SIZE=4
export RPAS_EC1_LCB_CONCURRENCY=4
export RPAS_EC1_LCB_GENERATION_TIMEOUT_SECONDS=300
export RPAS_AFLOW_MAX_ROUNDS=2
export RPAS_AFLOW_SAMPLE=4
export RPAS_EC1_LCB_SEED_CANDIDATES=4
export RPAS_EC1_LCB_NEW_CANDIDATES=3
export GEPA_QWEN_DISABLE_THINKING=1
export PYTHONUNBUFFERED=1

{
  date --iso-8601=seconds
  hostname
  echo "job_id=${SLURM_JOB_ID}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
  echo "method=${method}"
  echo "seed=0"
  echo "dataset=livecodebench/code_generation_lite"
  echo "dataset_revision=0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
  echo "release_version=release_v6"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
  sha256sum "${data_dir}"/*.jsonl "${data_dir}/DATASET_MANIFEST.json"
  sha256sum experiments/phase2_wan_agent_search.py external_comparison/runners/*livecodebench*.py
} | tee "${job_root}/environment.txt"

./.uv-linux run --no-project --offline --python "$python_bin" -m external_comparison.runners.livecodebench_preflight \
  --data-dir "$data_dir" --aflow-root "$RPAS_AFLOW_ROOT"

./.uv-linux run --no-project --offline --python "$python_bin" scripts/serve_transformers_qwen35_openai.py \
  --model "$model_path" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "$port" --max-new-tokens 2048 \
  --max-batch-size 4 --batch-wait-ms 25 --dtype float16 --stop-string '<<RPAS_END>>' \
  >"${job_root}/logs/service.log" 2>&1 &
service_pid=$!
cleanup() {
  kill "$service_pid" 2>/dev/null || true
  wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 180); do
  curl -sf --max-time 5 "http://127.0.0.1:${port}/health" >"${job_root}/health.json" && break
  kill -0 "$service_pid" 2>/dev/null || { tail -n 160 "${job_root}/logs/service.log" >&2; exit 1; }
  sleep 5
done
curl -sf --max-time 10 "http://127.0.0.1:${port}/v1/models" >"${job_root}/models.json"

case "$method" in
  single)
    runner=external_comparison.runners.single_livecodebench
    ;;
  aflow)
    runner=external_comparison.runners.native_aflow_livecodebench
    ;;
  rpas)
    runner=external_comparison.runners.native_rpas_livecodebench
    ;;
esac
./.uv-linux run --no-project --offline --python "$python_bin" -m "$runner" \
  --data-dir "$data_dir" --output-dir "$output_root" --seed 0 \
  >"${job_root}/logs/runner.log" 2>&1

result="${output_root}/${method}/seed_0/result.json"
[[ -s "$result" ]] || { echo "missing result: $result" >&2; exit 1; }
./.uv-linux run --no-project --offline --python "$python_bin" - "$result" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("dataset") != "livecodebench" or payload.get("seed") != 0:
    raise SystemExit("result provenance mismatch")
if payload.get("formal_result") is not False:
    raise SystemExit("one-seed EC-1B pilot must not claim a formal result")
if payload.get("release_version") != "release_v6":
    raise SystemExit("release mismatch")
print(json.dumps(payload["summary"], sort_keys=True))
PY
./.uv-linux run --no-project --offline --python "$python_bin" -m external_comparison.runners.write_run_metrics \
  --result-dir "${output_root}/${method}/seed_0" --job-id "$SLURM_JOB_ID" --job-root "$job_root" --stage ec1b_seed0_pilot
echo "complete" >"${job_root}/job_status.txt"
