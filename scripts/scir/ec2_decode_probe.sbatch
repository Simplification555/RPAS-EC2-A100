#!/usr/bin/env bash
# Dev-only decode diagnostic; never unlocks or reads held-out data.
#SBATCH --job-name=ec2-decode-probe
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.err

set -euo pipefail
repo=/home/jianbaizhao/RPAS-EC2-A100
python_bin=/home/jianbaizhao/qwen_cuda126/bin/python
cd "$repo"
run=(./.uv-linux run --no-project --offline --python "$python_bin")
export RPAS_SCIR_ALLOCATED_GPU=1
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || exit 2
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export RPAS_TOKENIZER_PATH=/home/jianbaizhao/model/Qwen/Qwen3.5-9B
export RPAS_GDESIGNER_EMBEDDING_MODEL=/home/jianbaizhao/model/sentence-transformers/all-MiniLM-L6-v2
export RPAS_EXTERNAL_API_KEY=EMPTY
export GEPA_QWEN_DISABLE_THINKING=1
export PYTHONUNBUFFERED=1
port=$((31000 + SLURM_JOB_ID % 1000))
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:$port/v1"
job_root="$repo/outputs/ec2_decode_probe/$SLURM_JOB_ID"
mkdir -p "$job_root"
{
  date --iso-8601=seconds
  hostname
  echo "job_id=$SLURM_JOB_ID; formal_result=false; caps=256,1024; dev_count=8"
  nvidia-smi --query-gpu=uuid,name,memory.total --format=csv
  sha256sum external_comparison/runners/ec2_v2.py external_comparison/runners/ec2_decode_probe.py
} > "$job_root/environment.txt"
"${run[@]}" scripts/serve_transformers_qwen35_openai.py \
  --model "$RPAS_TOKENIZER_PATH" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "$port" --max-new-tokens 1024 \
  --max-batch-size 1 --batch-wait-ms 0 --dtype float16 \
  --stop-string '<<RPAS_END>>' > "$job_root/service.log" 2>&1 &
service_pid=$!
cleanup() { kill "$service_pid" 2>/dev/null || true; wait "$service_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for _ in $(seq 1 180); do
  if curl --max-time 2 -sf "http://127.0.0.1:$port/health" > "$job_root/health.json"; then ready=1; break; fi
  kill -0 "$service_pid" 2>/dev/null || { tail -n 30 "$job_root/service.log" >&2; exit 1; }
  sleep 5
done
[[ "$ready" == 1 ]] || { echo 'service readiness timeout' >&2; exit 1; }
"${run[@]}" -m external_comparison.runners.ec2_decode_probe \
  --data-dir /home/jianbaizhao/RPAS/data/mmlu \
  --gdesigner-root /home/jianbaizhao/external_baselines/GDesigner \
  --output "$job_root/result" --count 8 > "$job_root/runner.log" 2>&1
test -s "$job_root/result/summary.json"
echo 'probe_completed=true; formal_result=false'
