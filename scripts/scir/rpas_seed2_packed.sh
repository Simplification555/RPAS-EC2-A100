#!/usr/bin/env bash
# Wait for an external lifetime guard and confirmed release of the AFlow service.
set -euo pipefail
cd /home/jianbaizhao/RPAS-EC2-A100
[[ "${SLURM_JOB_ID:-}" == 132166 ]] || exit 2
root="$PWD/outputs/scir_ec1_rpas_extractor_v2"
activation="$root/_jobs/activate_seed2_132166"
[[ ! -e "$activation" ]] || { echo 'Refusing a stale activation marker.' >&2; exit 2; }
echo "third_lane_wrapper_pid=$$ waiting_for_activation=$activation"
for _ in $(seq 1 600); do
  [[ -f "$activation" ]] && break
  sleep 1
done
[[ -f "$activation" ]] || exit 2
exec env SLURM_ARRAY_TASK_ID=8 RPAS_EC1_PACKED_SLOT=extractor_v2_third_lane \
  RPAS_EC1_OUTPUT_ROOT="$root" RPAS_EC1_SERVICE_PORT=40334 RPAS_EC1_REPLACE_WORKSPACE=0 \
  bash scripts/scir/ec1_formal_one_gpu.sbatch
