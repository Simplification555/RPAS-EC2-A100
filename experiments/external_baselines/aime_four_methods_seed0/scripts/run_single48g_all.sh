#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"
case "$STAGE" in all|diagnostics) ;; *) echo "usage: $0 [all|diagnostics]" >&2; exit 2 ;; esac
: "${AIME_MODEL_PATH:?Set the verified local Qwen3.5-9B path}"
: "${AIME_MINILM_PATH:?Set the verified local MiniLM path}"
: "${AIME_MODEL_VERIFICATION_REPORT:?Set the model verification report path}"
: "${AIME_PYTHON:?Set the serving environment Python}"
: "${AIME_VLLM_BIN:?Set the serving environment vLLM}"
: "${AIME_AFLOW_PYTHON:?Set AFlow Python}"
: "${AIME_MAAS_PYTHON:?Set MaAS Python}"
: "${AIME_ADAS_PYTHON:?Set ADAS Python}"
: "${AIME_GDESIGNER_PYTHON:?Set G-Designer Python}"
export AIME_DATA_DIR="${AIME_DATA_DIR:-$ROOT/data}"
AIME_OUTPUT_ROOT="${AIME_OUTPUT_ROOT:-$ROOT/outputs/aime_four_methods_seed0_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "$AIME_OUTPUT_ROOT" ]] || { echo "Choose a fresh AIME_OUTPUT_ROOT" >&2; exit 3; }
mkdir -p "$AIME_OUTPUT_ROOT"
"$AIME_PYTHON" - "$AIME_OUTPUT_ROOT/launch_record.json" "$ROOT" "$STAGE" <<'PY'
import datetime, hashlib, json, os, pathlib, subprocess, sys
path, root, stage = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
names = ("AIME_MODEL_PATH", "AIME_MINILM_PATH", "AIME_MODEL_VERIFICATION_REPORT",
         "AIME_PYTHON", "AIME_VLLM_BIN", "AIME_AFLOW_PYTHON", "AIME_MAAS_PYTHON",
         "AIME_ADAS_PYTHON", "AIME_GDESIGNER_PYTHON", "AIME_DIAGNOSTIC_DIR", "CUDA_VISIBLE_DEVICES")
record = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "source_root": str(root), "source_commit": subprocess.check_output(
              ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
          "stage": stage, "settings": {name: os.environ.get(name, "") for name in names},
          "launcher_sha256": hashlib.sha256((root / "scripts/run_single48g_all.sh").read_bytes()).hexdigest(),
          "protocol": {"search_seed": 0, "data_seed": 2026, "D_search": 60, "D_select": 30,
                       "D_test_each_year": 30, "context": 8192, "max_tokens": 6144, "concurrency": 8},
          "note": "Shared AIME adapter protocol; one seed only. Diagnostics use synthetic questions."}
with path.open("x") as f:
    json.dump(record, f, indent=2); f.write("\n")
PY
echo "AIME_ALL_START $(date -Iseconds) output=$AIME_OUTPUT_ROOT"
export AIME_OUTPUT_DIR="$AIME_OUTPUT_ROOT/diagnostics"
bash "$ROOT/scripts/run_pair.sh" diagnostics
if [[ "$STAGE" == diagnostics ]]; then
  echo "AIME_DIAGNOSTICS_FINISHED $(date -Iseconds)"
  exit 0
fi
for pair in aflow_maas adas_gdesigner; do
  export AIME_OUTPUT_DIR="$AIME_OUTPUT_ROOT/$pair"
  echo "AIME_PAIR_START $(date -Iseconds) pair=$pair"
  bash "$ROOT/scripts/run_pair.sh" "$pair"
  echo "AIME_PAIR_PASS $(date -Iseconds) pair=$pair"
done
"$AIME_PYTHON" "$ROOT/scripts/aggregate_aime_seed0.py" --root "$AIME_OUTPUT_ROOT"
echo "AIME_FOUR_METHODS_SEED0_COMPLETE $(date -Iseconds) output=$AIME_OUTPUT_ROOT"
