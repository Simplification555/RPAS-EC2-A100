#!/usr/bin/env bash
# Run one or more frozen AIME Track-A search seeds on one selected A100.

set -euo pipefail

GPU="${RPAS_CUDA_VISIBLE_DEVICES:-4}"
if [[ "${GPU}" != "4" && "${GPU}" != "5" ]]; then
  echo "RPAS_CUDA_VISIBLE_DEVICES must be 4 or 5; got ${GPU}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${RPAS_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_PATH="${RPAS_MODEL_PATH:?set RPAS_MODEL_PATH to Qwen3.5-9B}"
VLLM_BIN="${RPAS_VLLM_BIN:-vllm}"
PYTHON_BIN="${RPAS_PYTHON_BIN:-python}"
INFERENCE_BACKEND="${RPAS_INFERENCE_BACKEND:-transformers}"
PORT="${RPAS_VLLM_PORT:-29600}"
DATA_DIR="${RPAS_AIME_DATA_DIR:-${REPO_ROOT}/data/formal_v1/aime}"
CONFIG="${RPAS_FORMAL_CONFIG:-${REPO_ROOT}/experiments/phase2_formal_track_a_qwen35_9b.json}"
OUTPUT_ROOT="${RPAS_FORMAL_OUTPUT_DIR:-${REPO_ROOT}/outputs/formal_v1}"
SEEDS="${RPAS_SEEDS:-0}"
METHODS="${RPAS_METHODS:-baselines random aflow_style adas_style quality_only wan_pareto}"
LOG_DIR="${RPAS_LOG_DIR:-${REPO_ROOT}/logs/formal_aime_gpu${GPU}}"
RPC_BASE_PATH="${RPAS_VLLM_RPC_BASE_PATH:-/dev/shm/vllm-rpas-gpu${GPU}}"

for required in "${MODEL_PATH}" "${DATA_DIR}" "${CONFIG}"; do
  [[ -e "${required}" ]] || { echo "missing required path: ${required}" >&2; exit 2; }
done
command -v "${VLLM_BIN}" >/dev/null

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${RPC_BASE_PATH}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TMPDIR="${RPAS_TMPDIR:-/dev/shm}"
export VLLM_RPC_BASE_PATH="${RPC_BASE_PATH}"
# This inference stack is PyTorch-only. Avoid importing an unrelated system
# TensorFlow build that is incompatible with the user-level NumPy runtime.
export USE_TF=0
export GEPA_QWEN_DISABLE_THINKING=1
export GEPA_QWEN35_9B_LOCAL_API_BASE="http://127.0.0.1:${PORT}/v1"
export GEPA_CODE_COMMIT="$(git rev-parse HEAD)"

if [[ "${INFERENCE_BACKEND}" == "transformers" ]]; then
  "${PYTHON_BIN}" scripts/serve_transformers_qwen35_openai.py \
    --model "${MODEL_PATH}" --served-model-name "Qwen/Qwen3.5-9B" \
    --host 127.0.0.1 --port "${PORT}" --max-new-tokens 4096 \
    --max-batch-size "${RPAS_BATCH_SIZE:-8}" --batch-wait-ms "${RPAS_BATCH_WAIT_MS:-25}" \
    --stop-string "${RPAS_STOP_STRING:-<<RPAS_END>>}" \
    >"${LOG_DIR}/inference.log" 2>&1 &
elif [[ "${INFERENCE_BACKEND}" == "vllm" ]]; then
  "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name "Qwen/Qwen3.5-9B" --host 127.0.0.1 --port "${PORT}" \
    --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    >"${LOG_DIR}/inference.log" 2>&1 &
else
  echo "RPAS_INFERENCE_BACKEND must be transformers or vllm; got ${INFERENCE_BACKEND}" >&2
  exit 2
fi
VLLM_PID=$!
cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    tail -n 200 "${LOG_DIR}/inference.log" >&2 || true
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    NEW_CANDIDATE_BUDGET=24
    if [[ "${method}" == "baselines" ]]; then
      NEW_CANDIDATE_BUDGET=0
    fi
    python experiments/phase2_wan_agent_search.py \
      --config "${CONFIG}" --mode "${method}" --network-profile lan_homogeneous \
      --dataset aime --data-dir "${DATA_DIR}" --aime-test-file aime_2025.jsonl \
      --search-size 60 --selection-size 30 --test-size 30 --search-examples 60 \
      --seed-candidates 9 --new-candidate-budget "${NEW_CANDIDATE_BUDGET}" --selection-shortlist-size 8 \
      --test-top-k 1 --eval-concurrency "${RPAS_EVAL_CONCURRENCY:-8}" \
      --seed "${seed}" --data-seed 2026 --reflection-mode llm \
      --dataset-manifest "${DATA_DIR}/dataset_manifest.json" --output-dir "${OUTPUT_ROOT}" \
      2>&1 | tee "${LOG_DIR}/${method}_seed${seed}.log"

    PRIMARY_RUN="${OUTPUT_ROOT}/aime/aime_2025/lan_homogeneous/${method}/seed_${seed}"
    python scripts/evaluate_frozen_aime_candidates.py \
      --primary-run "${PRIMARY_RUN}" --config "${CONFIG}" --data-dir "${DATA_DIR}" \
      --test-file aime_2026.jsonl \
      --output "${PRIMARY_RUN}/secondary_test_aime_2026.json" \
      --eval-concurrency "${RPAS_EVAL_CONCURRENCY:-8}" \
      2>&1 | tee "${LOG_DIR}/${method}_seed${seed}_aime2026.log"
  done
done
