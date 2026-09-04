#!/usr/bin/env bash
# Run an EC-2 MMLU experiment on one explicitly selected local A100.
# The model is loaded once by vLLM; G-Designer and RPAS share that endpoint.

set -euo pipefail

# Each server is deliberately bound to exactly one authorized physical GPU.
export CUDA_VISIBLE_DEVICES="${RPAS_CUDA_VISIBLE_DEVICES:-4}"
if [ "${CUDA_VISIBLE_DEVICES}" != "4" ] && [ "${CUDA_VISIBLE_DEVICES}" != "5" ]; then
  echo "This runner requires RPAS_CUDA_VISIBLE_DEVICES=4 or 5; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${RPAS_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RPAS_PYTHON_BIN="${RPAS_PYTHON_BIN:-python}"
GDESIGNER_PYTHON_BIN="${RPAS_GDESIGNER_PYTHON_BIN:-${RPAS_PYTHON_BIN}}"
VLLM_BIN="${RPAS_VLLM_BIN:-vllm}"
MODEL_PATH="${RPAS_MODEL_PATH:?set RPAS_MODEL_PATH to the local Qwen/Qwen3.5-9B directory}"
DATA_DIR="${RPAS_MMLU_DATA_DIR:?set RPAS_MMLU_DATA_DIR to the frozen MMLU data directory}"
GDESIGNER_ROOT="${RPAS_GDESIGNER_ROOT:?set RPAS_GDESIGNER_ROOT to the G-Designer checkout}"
EMBEDDING_MODEL="${RPAS_MAAS_EMBEDDING_MODEL:?set RPAS_MAAS_EMBEDDING_MODEL to the local MiniLM directory}"
OUTPUT_DIR="${RPAS_OUTPUT_DIR:-${REPO_ROOT}/outputs/external_comparison/ec2_fixed_v5_a100}"
VLLM_PORT="${RPAS_VLLM_PORT:-29500}"
VLLM_LOG="${RPAS_VLLM_LOG:-${REPO_ROOT}/logs/vllm-ec2-a100.log}"
CONFIG_PATH="${RPAS_MODEL_CONFIG:-${REPO_ROOT}/experiments/phase2_mmlu_qwen35_9b.json}"

for required_dir in "${REPO_ROOT}" "${DATA_DIR}" "${MODEL_PATH}" "${GDESIGNER_ROOT}" "${EMBEDDING_MODEL}"; do
  if [ ! -d "${required_dir}" ]; then
    echo "required directory is missing: ${required_dir}" >&2
    exit 2
  fi
done
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "model config is missing: ${CONFIG_PATH}" >&2
  exit 2
fi
command -v "${RPAS_PYTHON_BIN}" >/dev/null
command -v "${GDESIGNER_PYTHON_BIN}" >/dev/null
command -v "${VLLM_BIN}" >/dev/null
command -v curl >/dev/null

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export GEPA_QWEN_DISABLE_THINKING=1
export GEPA_PHASE2_PROMPT_MODE=deliberate
export GEPA_QWEN35_9B_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export GEPA_QWEN35_9B_REMOTE_PROFILE_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export RPAS_MODEL_CONFIG="${CONFIG_PATH}"
export RPAS_GDESIGNER_ROOT="${GDESIGNER_ROOT}"
export RPAS_MAAS_EMBEDDING_MODEL="${EMBEDDING_MODEL}"
export RPAS_EXTERNAL_MODEL="Qwen/Qwen3.5-9B"
export RPAS_MMLU_MAX_TOKENS=256
export RPAS_MMLU_EVAL_CONCURRENCY="${RPAS_MMLU_EVAL_CONCURRENCY:-8}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${VLLM_LOG}")" "${REPO_ROOT}/logs"

"${RPAS_PYTHON_BIN}" -m external_comparison.runners.validate_protocol \
  --config-dir external_comparison/configs --require-native
"${RPAS_PYTHON_BIN}" -m external_comparison.runners.mmlu \
  --data-dir "${DATA_DIR}" --output "${OUTPUT_DIR}/split_manifest.json"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "Qwen/Qwen3.5-9B" --host 127.0.0.1 --port "${VLLM_PORT}" \
  --tensor-parallel-size 1 --max-model-len 32768 --reasoning-parser qwen3 --language-model-only \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    tail -n 200 "${VLLM_LOG}" || true
    exit 1
  fi
  sleep 5
done
if [ "${READY}" -ne 1 ]; then
  echo "vLLM startup timeout" >&2
  exit 1
fi
curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null

# Eight fixed examples catch parser, import, thinking, and truncation errors
# before the full 855-example method runs consume the A100 allocation.
SMOKE_OUTPUT_DIR="${OUTPUT_DIR}/_smoke"
export RPAS_NATIVE_SAMPLE_LIMIT=8
for method in gdesigner rpas; do
  if [ "${method}" = "rpas" ]; then
    METHOD_PYTHON="${RPAS_PYTHON_BIN}"
  else
    METHOD_PYTHON="${GDESIGNER_PYTHON_BIN}"
  fi
  "${METHOD_PYTHON}" -m external_comparison.runners.native_mmlu \
    --repo-root "${REPO_ROOT}" --data-dir "${DATA_DIR}" --method "${method}" --seed 0 \
    --output-dir "${SMOKE_OUTPUT_DIR}"
done
unset RPAS_NATIVE_SAMPLE_LIMIT

for method in gdesigner rpas; do
  for seed in 0 1 2; do
    if [ "${method}" = "rpas" ]; then
      METHOD_PYTHON="${RPAS_PYTHON_BIN}"
    else
      METHOD_PYTHON="${GDESIGNER_PYTHON_BIN}"
    fi
    RPAS_METHOD="${method}" RPAS_SEARCH_SEED="${seed}" RPAS_OUTPUT_DIR="${OUTPUT_DIR}" \
      PYTHONPATH="${REPO_ROOT}" "${METHOD_PYTHON}" -m external_comparison.runners.native_mmlu \
      --repo-root "${REPO_ROOT}" --data-dir "${DATA_DIR}" --method "${method}" --seed "${seed}" \
      --output-dir "${OUTPUT_DIR}"
  done
done

echo "EC-2 A100 run completed: ${OUTPUT_DIR}"
