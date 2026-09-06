#!/usr/bin/env bash
# Three independent native MaAS seeds. Each array element owns one Qwen
# endpoint on one GPU; do not co-reside multiple full 9B replicas on one card.
#SBATCH --job-name=ec1-maas-native3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
# The EC-1 matrix keeps every method on the same local A100 environment.
# One FP16 Qwen3.5-9B endpoint is within the 40 GB allocation.
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%A_%a.err

set -euo pipefail
repo_root="/home/jianbaizhao/RPAS-EC2-A100"
seed="${SLURM_ARRAY_TASK_ID:?array task id is required}"
[[ "${seed}" =~ ^[012]$ ]] || { echo "MaAS seed must be 0, 1, or 2" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || { echo "expected one allocated GPU" >&2; exit 2; }
export RPAS_EC1_OUTPUT_ROOT="${RPAS_EC1_OUTPUT_ROOT:-${repo_root}/outputs/scir_ec1_formal_v3}"
export RPAS_EC1_PACKED_SLOT="one_endpoint_per_gpu"
export RPAS_EC1_SERVICE_PORT="${RPAS_EC1_SERVICE_PORT:-$((35000 + SLURM_JOB_ID % 900))}"
export RPAS_EC1_REPLACE_WORKSPACE=0
cd "${repo_root}"
exec env SLURM_ARRAY_TASK_ID="$((seed + 3))" bash scripts/scir/ec1_formal_one_gpu.sbatch
