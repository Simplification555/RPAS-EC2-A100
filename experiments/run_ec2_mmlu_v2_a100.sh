#!/usr/bin/env bash
# Run one EC-2 v2 workload against one resident Qwen service on GPU 4 or GPU 5.
# This script deliberately never selects, probes, or starts any other GPU.

set -euo pipefail

STAGE="${1:-}"
if [[ "${STAGE}" != "pilot" && "${STAGE}" != "formal" ]]; then
  echo "usage: $0 {pilot|formal}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${RPAS_CUDA_VISIBLE_DEVICES:-}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "4" && "${CUDA_VISIBLE_DEVICES}" != "5" ]]; then
  echo "set RPAS_CUDA_VISIBLE_DEVICES to exactly 4 or 5" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${RPAS_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${RPAS_PYTHON_BIN:-${REPO_ROOT}/.rpas-run/bin/python}"
MODEL_PATH="${RPAS_MODEL_PATH:?set RPAS_MODEL_PATH to the local Qwen3.5-9B directory}"
DATA_DIR="${RPAS_MMLU_DATA_DIR:?set RPAS_MMLU_DATA_DIR to frozen MMLU data}"
GDESIGNER_ROOT="${RPAS_GDESIGNER_ROOT:?set RPAS_GDESIGNER_ROOT to the pinned G-Designer checkout}"
EMBEDDING_MODEL="${RPAS_GDESIGNER_EMBEDDING_MODEL:-${RPAS_MAAS_EMBEDDING_MODEL:-}}"
[[ -n "${EMBEDDING_MODEL}" ]] || { echo "set RPAS_GDESIGNER_EMBEDDING_MODEL to local all-MiniLM-L6-v2 files" >&2; exit 2; }
OUTPUT_DIR="${RPAS_OUTPUT_DIR:-${REPO_ROOT}/outputs/external_comparison/ec2_mmlu_v2}"
if [[ "${CUDA_VISIBLE_DEVICES}" == "4" ]]; then
  DEFAULT_SERVICE_PORT=29500
else
  DEFAULT_SERVICE_PORT=29501
fi
SERVICE_PORT="${RPAS_SERVICE_PORT:-${DEFAULT_SERVICE_PORT}}"
SERVICE_LOG="${RPAS_SERVICE_LOG:-${REPO_ROOT}/logs/transformers-ec2-v2-gpu${CUDA_VISIBLE_DEVICES}.log}"
CONFIG_PATH="${RPAS_MODEL_CONFIG:-${REPO_ROOT}/experiments/phase2_mmlu_qwen35_9b.json}"
METHODS=( ${RPAS_EC2_V2_METHODS:-single_agent full_connected chain gdesigner rpas_comm} )
SEEDS=( ${RPAS_EC2_V2_SEEDS:-0 1 2} )

for required_dir in "${REPO_ROOT}" "${MODEL_PATH}" "${DATA_DIR}" "${GDESIGNER_ROOT}" "${EMBEDDING_MODEL}"; do
  [[ -d "${required_dir}" ]] || { echo "required directory is missing: ${required_dir}" >&2; exit 2; }
done
[[ -f "${CONFIG_PATH}" ]] || { echo "model config is missing: ${CONFIG_PATH}" >&2; exit 2; }
command -v "${UV_BIN}" >/dev/null
[[ -x "${PYTHON_BIN}" ]] || { echo "Python runtime is missing: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${REPO_ROOT}/scripts/serve_transformers_qwen35_openai.py" ]] || {
  echo "Transformers OpenAI service is missing" >&2
  exit 2
}
command -v curl >/dev/null

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RPAS_EXTERNAL_API_BASE="http://127.0.0.1:${SERVICE_PORT}/v1"
export RPAS_EXTERNAL_API_KEY="${RPAS_EXTERNAL_API_KEY:-EMPTY}"
export RPAS_EXTERNAL_MODEL="Qwen/Qwen3.5-9B"
export RPAS_MODEL_CONFIG="${CONFIG_PATH}"
export RPAS_TOKENIZER_PATH="${RPAS_TOKENIZER_PATH:-${MODEL_PATH}}"
export RPAS_GDESIGNER_EMBEDDING_MODEL="${EMBEDDING_MODEL}"
export RPAS_MMLU_MAX_TOKENS=256
export GEPA_QWEN_DISABLE_THINKING=1
export GEPA_PHASE2_PROMPT_MODE=deliberate

"${UV_BIN}" run python -m py_compile \
  external_comparison/runners/ec2_v2.py \
  external_comparison/runners/aggregate_mmlu_v2.py
"${UV_BIN}" run python -m external_comparison.runners.validate_protocol \
  --config-dir external_comparison/configs

mkdir -p "${OUTPUT_DIR}" "$(dirname "${SERVICE_LOG}")"
SERVICE_PID=""
if curl -sf "http://127.0.0.1:${SERVICE_PORT}/health" >/dev/null; then
  echo "Reusing resident EC-2 backend on GPU ${CUDA_VISIBLE_DEVICES}, port ${SERVICE_PORT}."
else
  # vLLM cannot execute Qwen3_5ForConditionalGeneration in this environment.
  # This service uses the official Transformers implementation and binds exactly
  # the selected physical GPU through CUDA_VISIBLE_DEVICES.
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" \
    scripts/serve_transformers_qwen35_openai.py --model "${MODEL_PATH}" \
    --served-model-name "Qwen/Qwen3.5-9B" --host 127.0.0.1 --port "${SERVICE_PORT}" \
    --max-new-tokens 256 --max-batch-size 4 --batch-wait-ms 25 --stop-string '<<RPAS_END>>' \
    >"${SERVICE_LOG}" 2>&1 &
  SERVICE_PID=$!
fi

cleanup() {
  if [[ -n "${SERVICE_PID}" ]]; then
    kill "${SERVICE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${SERVICE_PORT}/health" >/dev/null; then
    break
  fi
  if [[ -n "${SERVICE_PID}" ]] && ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
    tail -n 120 "${SERVICE_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${SERVICE_PORT}/v1/models" >/dev/null

if [[ "${STAGE}" == "pilot" ]]; then
  METHODS=(gdesigner rpas_comm)
  SEEDS=(0)
  SEARCH_PER_SUBJECT=1
  SELECT_PER_SUBJECT=1
  TEST_PER_SUBJECT=1
  OUTPUT_DIR="${OUTPUT_DIR}/pilot"
else
  SEARCH_PER_SUBJECT=1
  SELECT_PER_SUBJECT=1
  TEST_PER_SUBJECT=10
fi

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${UV_BIN}" run python -m external_comparison.runners.ec2_v2 \
      --repo-root "${REPO_ROOT}" --gdesigner-root "${GDESIGNER_ROOT}" --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" --method "${method}" --seed "${seed}" \
      --search-per-subject "${SEARCH_PER_SUBJECT}" --select-per-subject "${SELECT_PER_SUBJECT}" \
      --test-per-subject "${TEST_PER_SUBJECT}"
  done
done

if [[ "${STAGE}" == "formal" && "${RPAS_EC2_V2_AGGREGATE:-0}" == "1" ]]; then
  "${UV_BIN}" run python -m external_comparison.runners.aggregate_mmlu_v2 \
    --root "${OUTPUT_DIR}" --output-dir "${OUTPUT_DIR}/aggregate"
fi

echo "EC-2 v2 ${STAGE} workload completed on physical GPU ${CUDA_VISIBLE_DEVICES}: ${OUTPUT_DIR}"
