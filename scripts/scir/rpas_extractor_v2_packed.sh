#!/usr/bin/env bash
# Two independent endpoints; lane zero runs seeds 0 then 2, lane one seed 1.
# The activation marker allows retiring the legacy worker before loading models.
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: script <batch-shell-pid> <activation-marker>' >&2; exit 2; }
parent="$1"
activation="$2"
cd /home/jianbaizhao/RPAS-EC2-A100
root="$PWD/outputs/scir_ec1_rpas_extractor_v2"
mkdir -p "$root/_jobs" "$root/_locks"
exec 9>"$root/_locks/rpas_bundle.lock"
flock -n 9 || exit 2
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || exit 2
[[ ! -e "$activation" ]] || { echo 'Activation marker must be fresh.' >&2; exit 2; }
for seed in 0 1 2; do
  [[ ! -e "$root/rpas/seed_$seed" ]] || { echo "Existing seed $seed; refusing overwrite" >&2; exit 2; }
done
workers=()
cleanup() {
  for pid in "${workers[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
bash scripts/scir/guard_packed_allocation.sh "$parent" "$$" >"$root/_jobs/guard_${SLURM_JOB_ID}.log" 2>&1 &
guard_pid=$!
for _ in $(seq 1 20); do
  grep -q 'guarding job=' "$root/_jobs/guard_${SLURM_JOB_ID}.log" && break
  kill -0 "$guard_pid" 2>/dev/null || exit 2
  sleep 1
done
grep -q 'guarding job=' "$root/_jobs/guard_${SLURM_JOB_ID}.log" || exit 2
echo "bundle_pid=$$ guard_pid=$guard_pid waiting_for_activation=$activation"
for _ in $(seq 1 600); do
  [[ -f "$activation" ]] && break
  sleep 1
done
[[ -f "$activation" ]] || { echo 'Activation timed out without launching models.' >&2; exit 2; }
run_seed() {
  local seed="$1" port="$2"
  env SLURM_ARRAY_TASK_ID="$((seed + 6))" RPAS_EC1_PACKED_SLOT=extractor_v2_two_lane \
    RPAS_EC1_OUTPUT_ROOT="$root" RPAS_EC1_SERVICE_PORT="$port" RPAS_EC1_REPLACE_WORKSPACE=0 \
    bash scripts/scir/ec1_formal_one_gpu.sbatch \
    >"$root/_jobs/rpas_seed_${seed}_${SLURM_JOB_ID}.log" 2>&1
}
base_port=$((40000 + SLURM_JOB_ID % 1000 * 2))
(
  # A failed seed is retained; the independent next seed can still complete.
  lane_status=0
  run_seed 0 "$base_port" || lane_status=1
  run_seed 2 "$base_port" || lane_status=1
  exit "$lane_status"
) &
workers+=("$!")
healthy=0
for _ in $(seq 1 180); do
  if curl --max-time 2 -sf "http://127.0.0.1:$base_port/health" >/dev/null; then healthy=1; break; fi
  kill -0 "${workers[0]}" 2>/dev/null || break
  sleep 5
done
[[ "$healthy" -eq 1 ]] || { echo 'First endpoint did not become healthy.' >&2; exit 1; }
run_seed 1 "$((base_port + 1))" &
workers+=("$!")
status=0
for pid in "${workers[@]}"; do
  rc=0
  wait "$pid" || rc=$?
  printf '%s lane_pid=%s exit_code=%s\n' "$(date -Is)" "$pid" "$rc"
  [[ "$rc" -eq 0 ]] || status=1
done
workers=()
exit "$status"
