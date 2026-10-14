#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
[[ "${WSL_DISTRO_NAME:-}" == zyc111 ]] || { echo "Run only inside WSL zyc111"; exit 2; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec 9>run_task.lock
flock -n 9 || { echo 'Another task launcher holds the lock'; exit 2; }
export RPAS_MODEL_PATH="${RPAS_MODEL_PATH:-$ROOT/models/Qwen3.5-9B}"
export RPAS_MAAS_EMBEDDING_PATH="${RPAS_MAAS_EMBEDDING_PATH:-$ROOT/models/all-MiniLM-L6-v2}"
export MODEL_PATH="$RPAS_MODEL_PATH"
export RPAS_PYTHON="${RPAS_PYTHON:-$ROOT/.venv-method/bin/python}"
export VLLM_BIN="${VLLM_BIN:-$ROOT/.venv-serve/bin/vllm}"
export PORT="${PORT:-8000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_NUM_SEQS=16
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export RPAS_ENDPOINT="http://127.0.0.1:$PORT/v1"
export PYTHONPATH="$ROOT/repo:$ROOT/repo/experiments:$ROOT/repo/external_comparison"
export RPAS_TOKENIZER_PATH="$RPAS_MODEL_PATH"
export OPENAI_API_KEY=EMPTY
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
# A free port prevents accidentally attaching this run to a different service.
"$RPAS_PYTHON" - "$PORT" <<'PYPORT'
import socket,sys
with socket.socket() as s:s.bind(('127.0.0.1',int(sys.argv[1])))
PYPORT
[[ -d "$MODEL_PATH" && -d "$RPAS_MAAS_EMBEDDING_PATH" ]] || { echo 'Model directories missing; run setup.sh or set model paths'; exit 2; }
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv > hardware_at_start.csv
"$RPAS_PYTHON" -m pip freeze > method_requirements_at_start.txt
if [[ -x "$ROOT/.venv-serve/bin/python" ]]; then "$ROOT/.venv-serve/bin/python" -m pip freeze > serving_requirements_at_start.txt; fi
"$RPAS_PYTHON" check_native_imports.py > native_import_before_load.log 2>&1
export VLLM_SOCKET="$ROOT/vllm.sock"
"$RPAS_PYTHON" - "$VLLM_SOCKET" <<'PYSOCKET'
import os,stat,sys
from pathlib import Path
p=Path(sys.argv[1])
if p.exists():
    info=p.stat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid!=os.getuid():raise SystemExit("Refusing non-owned/non-socket backend path")
    p.unlink()
PYSOCKET
"$RPAS_PYTHON" -u local_api_proxy.py --socket "$VLLM_SOCKET" --port "$PORT" > local_api_proxy.log 2>&1 &
PROXY_PID=$!
bash GPU教程/start_vllm.sh > vllm.log 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > server.pid
cleanup() { kill "$SERVER_PID" "$PROXY_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; wait "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$RPAS_PYTHON" - "$SERVER_PID" "$RPAS_ENDPOINT" <<'PYREADY'
import os,sys,time,json
from urllib.request import urlopen
pid=int(sys.argv[1]);url=sys.argv[2]+'/models'
for _ in range(180):
    os.kill(pid,0)
    try:
        with urlopen(url,timeout=3) as r: data=json.load(r)
        if any(x.get('id')=='Qwen/Qwen3.5-9B' for x in data.get('data',[])):break
    except Exception: pass
    time.sleep(5)
else:raise SystemExit('Service not ready after 15min; inspect vllm.log')
PYREADY
"$RPAS_PYTHON" run_aligned_batch.py
