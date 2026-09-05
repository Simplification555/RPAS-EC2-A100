#!/usr/bin/env bash
#SBATCH --job-name=ec2-aggregate
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
set -euo pipefail
cd /home/jianbaizhao/RPAS-EC2-A100
/home/jianbaizhao/qwen_cuda126/bin/python -m external_comparison.runners.aggregate_mmlu_v2 \
  --root outputs/scir_ec2_formal --output-dir outputs/scir_ec2_formal/aggregate
