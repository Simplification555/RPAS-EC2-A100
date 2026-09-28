#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${RPAS_PYTHON:-python3}"
VLLM="${RPAS_VLLM:-vllm}"
MODEL_PATH="${RPAS_MODEL_PATH:-}"
METHOD="${1:-all}"
PORT="${RPAS_PORT:-8000}"
OUT_ROOT="${RPAS_OUTPUT_ROOT:-$ROOT/outputs/masbench_aflow_maas_seed0}"
DATA_DIR="$ROOT/data/masbench"

case "$METHOD" in aflow|maas|all) ;; *) echo "usage: bash scripts/run_masbench_seed0.sh [aflow|maas|all]" >&2; exit 2 ;; esac
[[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]] || { echo "Set RPAS_MODEL_PATH to local Qwen3.5-9B weights" >&2; exit 2; }
[[ -d "$ROOT/external_baselines/AFlow" && -d "$ROOT/external_baselines/MaAS" ]] || {
  echo "Run bash scripts/setup_baselines.sh first" >&2; exit 2;
}
[[ -d "$DATA_DIR" ]] || { echo "Packaged MASBench data directory missing" >&2; exit 2; }


export RPAS_PYTHON="$PYTHON" RPAS_VLLM="$VLLM" RPAS_MODEL_PATH="$MODEL_PATH"
"$PYTHON" "$ROOT/scripts/preflight_gpu.py" --method "$METHOD"
export PYTHONPATH="$ROOT:$ROOT/experiments:$ROOT/external_comparison${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export RPAS_AFLOW_SOURCE="$ROOT/external_baselines/AFlow"
export RPAS_MAAS_SOURCE="$ROOT/external_baselines/MaAS"
export RPAS_MAAS_EMBEDDING_PATH="${RPAS_MAAS_EMBEDDING_PATH:-}"
export RPAS_MAX_MODEL_LEN=8192 RPAS_MAX_NUM_SEQS=24 RPAS_TP_SIZE=1 RPAS_GPU_MEMORY_UTILIZATION=0.92

mkdir -p "$ROOT/logs" "$OUT_ROOT"
SERVER_LOG="$ROOT/logs/vllm_seed0_${PORT}.log"
"$VLLM" serve "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 24 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --language-model-only >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PYTHON" - "$PORT" "$SERVER_PID" "$SERVER_LOG" <<'PY'
import json, sys, time, urllib.error, urllib.request
port, pid, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
url = f"http://127.0.0.1:{port}/v1/models"
for attempt in range(180):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.loads(response.read().decode())
        if any(item.get("id") == "Qwen/Qwen3.5-9B" for item in data.get("data", [])):
            print("VLLM_MODEL_READY", flush=True)
            break
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        pass
    try:
        import os
        os.kill(pid, 0)
    except OSError:
        print(f"vLLM exited early; inspect {log}", file=sys.stderr)
        raise SystemExit(1)
    time.sleep(5)
else:
    print(f"vLLM did not become ready within 15 minutes; inspect {log}", file=sys.stderr)
    raise SystemExit(1)
PY

for selected_method in aflow maas; do
  if [[ "$METHOD" != all && "$METHOD" != "$selected_method" ]]; then continue; fi
  if [[ "$selected_method" == maas && -z "$RPAS_MAAS_EMBEDDING_PATH" ]]; then
    echo "Set RPAS_MAAS_EMBEDDING_PATH for MaAS" >&2; exit 2
  fi
  for axis in breadth depth horizon parallel robustness; do
    run_dir="$OUT_ROOT/$selected_method/$axis/seed_0"
    if [[ -e "$run_dir" ]]; then
      if [[ -f "$run_dir/quality_gate.json" ]] && "$PYTHON" - "$run_dir" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
from masbench_quality_gate import validate_run
run_dir, root = Path(sys.argv[1]), Path(sys.argv[2])
try:
    stored = json.loads((run_dir / "quality_gate.json").read_text())
    current = validate_run(run_dir, root)
    if stored.get("passed") is True and current.get("passed") is True:
        print("ALREADY_GATED_PASS")
        raise SystemExit(0)
    print(json.dumps(current, indent=2), file=sys.stderr)
except Exception as exc:
    print(f"existing run cannot be verified: {type(exc).__name__}: {exc}", file=sys.stderr)
raise SystemExit(1)
PY
      then
        echo "SKIP already complete and gated: $selected_method/$axis/seed_0"
        continue
      fi
      echo "Refusing to overwrite partial or failed run: $run_dir" >&2
      exit 3
    fi

    args=(--method "$selected_method" --axis "$axis" --seed 0
      --rpas-root "$ROOT" --data-dir "$DATA_DIR" --run-dir "$run_dir"
      --endpoint "http://127.0.0.1:$PORT/v1" --data-seed 2026
      --max-tokens 6144 --concurrency 8 --max-rounds 8
      --maas-samples 1 --maas-batch-size 8)
    echo "START $selected_method/$axis seed=0"
    "$PYTHON" "$ROOT/experiments/native_masbench_external.py" "${args[@]}"
    "$PYTHON" "$ROOT/experiments/masbench_quality_gate.py" --run-dir "$run_dir" --rpas-root "$ROOT"
  done
done

if [[ "$METHOD" == all ]]; then
  if [[ -e "$OUT_ROOT/seed0_summary" ]]; then
    if [[ -f "$OUT_ROOT/seed0_summary/aggregate.json" && -f "$OUT_ROOT/seed0_summary/aggregate.csv" ]] \
      && "$PYTHON" "$ROOT/experiments/aggregate_masbench_seed0.py" --root "$OUT_ROOT" --verify-existing; then
      echo "SKIP aggregation; existing summary revalidated at $OUT_ROOT/seed0_summary"
    else
      echo "Refusing to trust or overwrite an incomplete/stale summary: $OUT_ROOT/seed0_summary" >&2
      exit 4
    fi
  else
    "$PYTHON" "$ROOT/experiments/aggregate_masbench_seed0.py" --root "$OUT_ROOT"
  fi
fi
echo "MASBENCH_SEED0_RUN_COMPLETE"
