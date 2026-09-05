#!/usr/bin/env bash
#SBATCH --job-name=ec1-aggregate
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
set -euo pipefail
cd /home/jianbaizhao/RPAS-EC2-A100
./.uv-linux run --no-project --offline --python /home/jianbaizhao/qwen_cuda126/bin/python -m scripts.aggregate_ec1_native_formal \
  --root outputs/scir_ec1_formal_v2 \
  --rpas-root outputs/scir_ec1_rpas_extractor_v2 \
  --single-root outputs/scir_ec1_single_reference \
  --output outputs/scir_ec1_formal_v2/aggregate
