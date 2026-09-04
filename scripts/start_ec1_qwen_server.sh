#!/usr/bin/env bash
set -euo pipefail

# Start exactly one EC-1-compatible Qwen service on an explicitly selected GPU.
# This launcher refuses to share a GPU with an existing compute process.
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <4|5> <port>" >&2
  exit 2
fi

gpu="$1"
port="$2"
if [[ "$gpu" != "4" && "$gpu" != "5" ]]; then
  echo "EC-1 permits only GPU 4 or GPU 5" >&2
  exit 2
fi
if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo "port must be an unprivileged TCP port" >&2
  exit 2
fi

active_pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {print $1}')"
if [[ -n "$active_pids" ]]; then
  echo "GPU $gpu already has compute process(es): $active_pids; refusing a second resident backbone" >&2
  exit 1
fi
if timeout 1 bash -c "</dev/tcp/127.0.0.1/$port" >/dev/null 2>&1; then
  echo "port $port is already in use" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$gpu"
python_bin="${RPAS_EC1_PYTHON:-}"
if [[ -z "$python_bin" && -x ".rpas-run/bin/python" ]]; then
  python_bin=".rpas-run/bin/python"
fi
python_bin="${python_bin:-python}"
exec "$python_bin" scripts/serve_transformers_qwen35_openai.py \
  --model models/Qwen3.5-9B \
  --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port "$port" \
  --max-new-tokens 1024 \
  --max-batch-size 4 \
  --batch-wait-ms 25 \
  --stop-string '<<RPAS_END>>'
