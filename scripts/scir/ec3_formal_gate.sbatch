#!/usr/bin/env bash
# Unlock EC-3 D_test only after all six immutable pretest states succeed.
#SBATCH --job-name=ec3-gate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.out
#SBATCH --error=/home/jianbaizhao/RPAS-EC2-A100/logs/slurm/%x-%j.err

set -euo pipefail
repo_root="/home/jianbaizhao/RPAS-EC2-A100"
python_bin="/home/jianbaizhao/qwen_cuda126/bin/python"
cd "${repo_root}"
"${python_bin}" -m py_compile external_comparison/runners/ec3_formal_gate.py
"${python_bin}" -m external_comparison.runners.ec3_formal_gate unlock \
  --root "${repo_root}/outputs/ec3_hotpotqa_v3"
