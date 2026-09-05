#!/usr/bin/env bash
# Three independent native MaAS seeds, each with a separate Qwen endpoint.
#SBATCH --job-name=ec1-maas-native3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.err

set -euo pipefail
cd /home/jianbaizhao/RPAS-EC2-A100
root="$PWD/outputs/scir_ec1_formal_v2"
mkdir -p "$root/_jobs" "$root/_locks"
exec 9>"$root/_locks/maas_native_three_seed.lock"
flock -n 9 || { echo 'Another native MaAS bundle owns the output.' >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || exit 2
memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[[ "$memory" -ge 75000 ]] || { echo 'Three endpoints require an allocated 80GB GPU.' >&2; exit 2; }
for seed in 0 1 2; do
  [[ ! -e "$root/maas/seed_$seed/_workspaces" && ! -e "$root/maas/seed_$seed/result.json" ]] || {
    echo "Refusing to overwrite existing MaAS seed $seed" >&2; exit 2;
  }
done
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
for seed in 0 1 2; do
  port=$((35000 + (SLURM_JOB_ID % 1000) * 3 + seed))
  env SLURM_ARRAY_TASK_ID="$((seed + 3))" RPAS_EC1_PACKED_SLOT=native_minilm_bundle \
    RPAS_EC1_SERVICE_PORT="$port" RPAS_EC1_REPLACE_WORKSPACE=0 \
    bash scripts/scir/ec1_formal_one_gpu.sbatch \
    >"$root/_jobs/maas_native_seed_${seed}_${SLURM_JOB_ID}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  # Stagger model loading to avoid simultaneous CPU staging-memory peaks.
  for _ in $(seq 1 180); do
    curl --max-time 2 -sf "http://127.0.0.1:$port/health" >/dev/null && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
done
status=0
for seed in 0 1 2; do
  rc=0
  wait "${pids[$seed]}" || rc=$?
  printf '%s seed=%s exit_code=%s\n' "$(date -Is)" "$seed" "$rc"
  [[ "$rc" -eq 0 ]] || status=1
done
pids=()
exit "$status"
