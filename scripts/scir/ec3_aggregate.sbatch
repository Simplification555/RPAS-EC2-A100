#!/usr/bin/env bash
#SBATCH --job-name=ec3-aggregate
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
set -euo pipefail
cd /home/jianbaizhao/RPAS-EC2-A100
/home/jianbaizhao/qwen_cuda126/bin/python scripts/aggregate_ec3_formal.py --root outputs/ec3_hotpotqa_v3 --output outputs/ec3_hotpotqa_v3/aggregate
