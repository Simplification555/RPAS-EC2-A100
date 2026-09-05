#!/usr/bin/env bash
#SBATCH --job-name=ec1-aggregate
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
set -euo pipefail
cd /home/jianbaizhao/RPAS-EC2-A100
/home/jianbaizhao/qwen_cuda126/bin/python scripts/aggregate_ec1_native_formal.py \
  --root outputs/scir_ec1_formal_v2 --output outputs/scir_ec1_formal_v2/aggregate
