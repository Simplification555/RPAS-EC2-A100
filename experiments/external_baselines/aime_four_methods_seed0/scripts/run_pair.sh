#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/experiments"
PAIR="${1:-}"
DATA="${AIME_DATA_DIR:-}"
MODEL="${AIME_MODEL_PATH:-}"
OUT="${AIME_OUTPUT_DIR:-$ROOT/outputs/seed0}"
PORT="${AIME_PORT:-18080}"
VLLM="${AIME_VLLM_BIN:-vllm}"
COMMON_PYTHON="${AIME_PYTHON:-python3}"
AFLOW_PYTHON="${AIME_AFLOW_PYTHON:-$COMMON_PYTHON}"
MAAS_PYTHON="${AIME_MAAS_PYTHON:-$COMMON_PYTHON}"
ADAS_PYTHON="${AIME_ADAS_PYTHON:-$COMMON_PYTHON}"
GDESIGNER_PYTHON="${AIME_GDESIGNER_PYTHON:-$COMMON_PYTHON}"
EMBEDDING="${AIME_MINILM_PATH:-}"
UPSTREAM="$ROOT/upstream"

case "$PAIR" in
  aflow_maas) METHODS=(aflow maas) ;;
  adas_gdesigner) METHODS=(adas gdesigner) ;;
  diagnostics) METHODS=(aflow maas adas gdesigner); export AIME_DIAGNOSTICS_ONLY=1 ;;
  *) echo "usage: bash scripts/run_pair.sh {aflow_maas|adas_gdesigner|diagnostics}" >&2; exit 2 ;;
esac

[[ -n "$DATA" && -d "$DATA" ]] || { echo "Set AIME_DATA_DIR to the locally authorized frozen AIME data directory" >&2; exit 2; }
[[ -n "$MODEL" && -f "$MODEL/config.json" && -f "$MODEL/tokenizer_config.json" ]] || { echo "Set AIME_MODEL_PATH to local Qwen3.5-9B weights/tokenizer" >&2; exit 2; }
for file in aimo-validation-aime.jsonl aime_2025.jsonl aime_2026.jsonl; do
  [[ -f "$DATA/$file" ]] || { echo "Missing required local AIME file: $DATA/$file" >&2; exit 2; }
done
# Only the validation pool is opened before candidate selection. The test files
# are existence-checked above, but their bytes/rows are not read by this script.
[[ "$(wc -l < "$DATA/aimo-validation-aime.jsonl")" -eq 90 ]] || { echo "Validation split must contain exactly 90 lines" >&2; exit 2; }
for executable in "$COMMON_PYTHON" "$AFLOW_PYTHON" "$MAAS_PYTHON" "$ADAS_PYTHON" "$GDESIGNER_PYTHON" "$VLLM" curl; do
  command -v "$executable" >/dev/null || { echo "Required executable not found: $executable" >&2; exit 2; }
done
command -v setsid >/dev/null || { echo "setsid is required to cleanly stop the vLLM process group" >&2; exit 2; }
[[ -d "$UPSTREAM/AFlow/.git" && -d "$UPSTREAM/MaAS/.git" && -d "$UPSTREAM/ADAS/.git" && -d "$UPSTREAM/GDesigner/.git" ]] || {
  echo "Run scripts/setup_upstreams.sh first" >&2; exit 2;
}
[[ -n "$EMBEDDING" && -d "$EMBEDDING" ]] || { echo "Set AIME_MINILM_PATH to local all-MiniLM-L6-v2 weights for MaAS/G-Designer" >&2; exit 2; }

CONTEXT=8192
OUTPUT=6144
CONCURRENCY=8
MAX_SEQS=24
GPU_UTIL=0.92
DATA_SEED=2026
SEED=0
mkdir -p "$OUT" "$ROOT/logs"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export AIME_TOKENIZER_PATH="$MODEL" AIME_MINILM_PATH="$EMBEDDING" AIME_MAAS_EMBEDDING_PATH="$EMBEDDING"
export AIME_AFLOW_SOURCE="$UPSTREAM/AFlow" AIME_MAAS_SOURCE="$UPSTREAM/MaAS"
export AIME_ADAS_SOURCE="$UPSTREAM/ADAS" AIME_GDESIGNER_SOURCE="$UPSTREAM/GDesigner"
export AIME_MAX_MODEL_LEN="$CONTEXT" AIME_MAX_NUM_SEQS="$MAX_SEQS" AIME_TP_SIZE=1
export AIME_PAIR="$PAIR"
export AIME_AFLOW_REQUEST_TIMEOUT_S="${AIME_AFLOW_REQUEST_TIMEOUT_S:-600}"
export AIME_AFLOW_SAMPLE_TIMEOUT_S="${AIME_AFLOW_SAMPLE_TIMEOUT_S:-1200}"
export AIME_AFLOW_CODE_TIMEOUT_S="${AIME_AFLOW_CODE_TIMEOUT_S:-30}"
export AIME_MAAS_REQUEST_TIMEOUT_S="${AIME_MAAS_REQUEST_TIMEOUT_S:-600}"
export AIME_MAAS_SAMPLE_TIMEOUT_S="${AIME_MAAS_SAMPLE_TIMEOUT_S:-1500}"
export AIME_MAAS_SAMPLE_RETRIES="${AIME_MAAS_SAMPLE_RETRIES:-3}"
export AIME_MAAS_CODE_TIMEOUT_S="${AIME_MAAS_CODE_TIMEOUT_S:-60}"
export AIME_EXTERNAL_REQUEST_TIMEOUT_S="${AIME_EXTERNAL_REQUEST_TIMEOUT_S:-600}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
AIME_CACHE_DIR="${AIME_CACHE_DIR:-/data/jxc/cache/rpas}"
AIME_TEMP_DIR="${AIME_TEMP_DIR:-/data/jxc/tmp/rpas/aime}"
export HF_HOME="$AIME_CACHE_DIR/huggingface" XDG_CACHE_HOME="$AIME_CACHE_DIR/xdg"
export TORCH_HOME="$AIME_CACHE_DIR/torch" TRITON_CACHE_DIR="$AIME_CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$AIME_CACHE_DIR/torchinductor" VLLM_CACHE_ROOT="$AIME_CACHE_DIR/vllm"
export CUDA_CACHE_PATH="$AIME_CACHE_DIR/cuda" TMPDIR="$AIME_TEMP_DIR"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH" "$TMPDIR"
AIME_EXISTING_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
export NO_PROXY="${AIME_EXISTING_NO_PROXY:+$AIME_EXISTING_NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
[[ -n "${AIME_MODEL_VERIFICATION_REPORT:-}" && -f "$AIME_MODEL_VERIFICATION_REPORT" ]] || {
  echo "Set AIME_MODEL_VERIFICATION_REPORT to a fresh scripts/verify_local_models.py report" >&2; exit 2;
}
export RPAS_SERVE_PYTHON="$COMMON_PYTHON" RPAS_AFLOW_PYTHON="$AFLOW_PYTHON" RPAS_MAAS_PYTHON="$MAAS_PYTHON"
export RPAS_VLLM="$VLLM" RPAS_MODEL_PATH="$MODEL" RPAS_MAAS_EMBEDDING_PATH="$EMBEDDING"
export RPAS_MODEL_VERIFICATION_REPORT="$AIME_MODEL_VERIFICATION_REPORT"
export RPAS_MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export RPAS_MAAS_EMBEDDING_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41

EXISTING=0
for method in "${METHODS[@]}"; do
  run_dir="$OUT/$method/seed_$SEED"
  if [[ -e "$run_dir" ]]; then
    EXISTING=$((EXISTING + 1))
    if [[ "$method" == aflow || "$method" == maas ]]; then
      if [[ "$method" == aflow ]]; then CHECK_PYTHON="$AFLOW_PYTHON"; else CHECK_PYTHON="$MAAS_PYTHON"; fi
      "$CHECK_PYTHON" "$CODE/aime_quality_gate.py" --run-dir "$run_dir" --method "$method" --seed "$SEED"
    else
      "$ADAS_PYTHON" "$CODE/aggregate_external_methods.py" --method "$method" --check-run "$run_dir" --dataset aime --seed "$SEED"
    fi
    echo "REUSE_VERIFIED $method seed=$SEED"
  fi
done

# Refuse a partially existing pair rather than overwriting or mixing runs.
if [[ "$EXISTING" == "${#METHODS[@]}" ]]; then
  echo "PAIR_ALREADY_COMPLETE pair=$PAIR; all outputs passed their quality gates"
  exit 0
fi
if [[ "$EXISTING" -gt 0 ]]; then
  echo "One pair member already has output; use a fresh AIME_OUTPUT_DIR to avoid mixing runs." >&2
  exit 3
fi
[[ ! -e "$OUT/run_session.json" ]] || { echo "This output root has a previous service attempt; choose a fresh output root." >&2; exit 3; }
for method in "${METHODS[@]}"; do
  run_dir="$OUT/$method/seed_$SEED"
  if [[ -e "$run_dir" ]]; then
    echo "Existing run detected at $run_dir; choose a fresh AIME_OUTPUT_DIR for any retry." >&2
    exit 3
  fi
done

python_for() {
  case "$1" in
    aflow) printf '%s' "$AFLOW_PYTHON" ;;
    maas) printf '%s' "$MAAS_PYTHON" ;;
    adas) printf '%s' "$ADAS_PYTHON" ;;
    gdesigner) printf '%s' "$GDESIGNER_PYTHON" ;;
  esac
}

"$COMMON_PYTHON" - "$PORT" <<'PY'
import socket, sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Match the server's reuse policy: recently closed connections can remain
    # in TIME_WAIT. An active listener still makes this bind fail.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
"$COMMON_PYTHON" "$ROOT/scripts/preflight_gpu.py" --role serve
for method in "${METHODS[@]}"; do
  if [[ "$method" == aflow || "$method" == maas ]]; then
    "$(python_for "$method")" "$ROOT/scripts/preflight_gpu.py" --role "$method"
  fi
done
LOG="$(mktemp "$ROOT/logs/vllm-${PAIR}-${PORT}-XXXXXXXX.log")"
GPU_LOG="$(mktemp "$ROOT/logs/gpu-${PAIR}-${PORT}-XXXXXXXX.csv")"
setsid "$VLLM" serve "$MODEL" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len "$CONTEXT" --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens 16384 --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching --reasoning-parser qwen3 --language-model-only \
  >"$LOG" 2>&1 &
VLLM_PID=$!
nvidia-smi --id "$GPU_ID" --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv -lms 1000 >"$GPU_LOG" 2>&1 &
GPU_PID=$!
cleanup() {
  set +e
  kill -- "-$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  kill "$GPU_PID" 2>/dev/null || true
  wait "$GPU_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
SESSION="${LOG%.log}.json"
export AIME_RUN_SESSION_RECORD="$SESSION"
"$COMMON_PYTHON" "$ROOT/scripts/record_run_session.py" --root "$ROOT" --output "$SESSION" \
  --server-pid "$VLLM_PID" --gpu-monitor-pid "$GPU_PID" --server-log "$LOG" --gpu-log "$GPU_LOG" --port "$PORT"
cp "$SESSION" "$OUT/run_session.json"

READY=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | "$COMMON_PYTHON" -c 'import json,sys; sys.exit(0 if any(x.get("id")=="Qwen/Qwen3.5-9B" for x in json.load(sys.stdin).get("data",[])) else 1)' 2>/dev/null; then READY=1; break; fi
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -n 200 "$LOG" >&2; exit 1; }
  sleep 5
done
[[ "$READY" == 1 ]] || { echo "Timed out waiting for vLLM; inspect $LOG" >&2; tail -n 200 "$LOG" >&2; exit 1; }

# A disposable endpoint smoke test; it is not part of any search/test metric.
"$COMMON_PYTHON" - "$PORT" <<'PY'
import sys
from openai import OpenAI
client = OpenAI(base_url=f"http://127.0.0.1:{sys.argv[1]}/v1", api_key="EMPTY", timeout=30, max_retries=0)
r = client.chat.completions.create(model="Qwen/Qwen3.5-9B", messages=[{"role":"user","content":"Return only OK."}], max_tokens=16, temperature=0, extra_body={"chat_template_kwargs":{"enable_thinking":False}})
assert r.choices and (r.choices[0].message.content or "").strip()
print("MODEL_ENDPOINT_SMOKE_PASS", flush=True)
PY

if [[ "${AIME_DIAGNOSTICS_ONLY:-0}" == 1 ]]; then
  DIAG="${AIME_DIAGNOSTIC_DIR:-$(mktemp -d "$TMPDIR/aime-native-XXXXXXXX")}"
  echo "SYNTHETIC_DIAGNOSTIC_ROOT $DIAG"
  for method in "${METHODS[@]}"; do
    echo "DIAGNOSTIC_START method=$method"
    "$(python_for "$method")" -u "$ROOT/scripts/diagnose_native_methods.py" \
      --method "$method" --baseline-root "$UPSTREAM" --endpoint "http://127.0.0.1:$PORT/v1" \
      --run-dir "$DIAG/$method" --max-tokens "$OUTPUT"
    echo "DIAGNOSTIC_PASS method=$method"
  done
  echo "AIME_NATIVE_DIAGNOSTICS_COMPLETE"
  exit 0
fi

for method in "${METHODS[@]}"; do
  PY="$(python_for "$method")"
  run_dir="$OUT/$method/seed_$SEED"
  export AIME_RUN_ID="aime:$PAIR:$method:seed_$SEED"
  export MASBENCH_RUN_ID="$AIME_RUN_ID"
  echo "START method=$method seed=$SEED data_seed=$DATA_SEED context=$CONTEXT output=$OUTPUT concurrency=$CONCURRENCY run_dir=$run_dir"
  if [[ "$method" == aflow || "$method" == maas ]]; then
    "$PY" -u "$CODE/native_aime_formal.py" --method "$method" --baseline-root "$UPSTREAM" \
      --data-dir "$DATA" --run-dir "$run_dir" --endpoint "http://127.0.0.1:$PORT/v1" \
      --data-seed "$DATA_SEED" --seed "$SEED" --search-size 60 --selection-size 30 --test-size 30 \
      --max-tokens "$OUTPUT" --concurrency "$CONCURRENCY" --aflow-max-rounds 8 \
      --maas-samples 1 --maas-batch-size 8
    "$PY" -u "$CODE/aime_quality_gate.py" --run-dir "$run_dir" --method "$method" --seed "$SEED"
  else
    "$PY" -u "$CODE/native_external_methods.py" --dataset aime --method "$method" \
      --seed "$SEED" --data-dir "$DATA" --run-dir "$run_dir" \
      --endpoint "http://127.0.0.1:$PORT/v1" --adas-generations 30
    "$PY" -u "$CODE/aggregate_external_methods.py" --method "$method" \
      --check-run "$run_dir" --dataset aime --seed "$SEED"
  fi
  cp "$SESSION" "$run_dir/run_session.json"
  echo "PASS method=$method seed=$SEED"
done

echo "AIME_PAIR_COMPLETE pair=$PAIR seed=$SEED"
