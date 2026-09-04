#!/usr/bin/env bash
# Prepare a clean, pinned external-comparison checkout on another machine.
# This script never downloads model weights and never starts an experiment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline_root="${RPAS_BASELINE_ROOT:-${repo_root}/external_baselines}"
python_bin="${PYTHON_BIN:-python3}"

AFLOW_REPO="https://github.com/FoundationAgents/AFlow.git"
AFLOW_COMMIT="3f457218fc716093fe53f6df8a5d5e6379d66346"
MAAS_REPO="https://github.com/bingreeky/MaAS.git"
MAAS_COMMIT="987f3c1bc9a96e844fe090db3791446e3ef0f5c7"
GDESIGNER_REPO="https://github.com/yanweiyue/GDesigner.git"
GDESIGNER_COMMIT="a6efcfa3b40bb4d9cbf46f883a95d62020bd8251"

usage() {
  cat <<'EOF'
usage: bash scripts/bootstrap_external_comparison.sh [--install-python-deps]

Checks out the exact official baseline source revisions required by EC-1/EC-2.
With --install-python-deps, creates .rpas-run through uv and installs the
Python packages used by the local OpenAI-compatible Qwen server. Install a
CUDA-compatible PyTorch wheel separately before using the server.
EOF
}

install_python_deps=0
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi
if [[ $# -eq 1 ]]; then
  [[ "$1" == "--install-python-deps" ]] || { usage >&2; exit 2; }
  install_python_deps=1
fi

checkout_pinned() {
  local destination="$1" repository="$2" revision="$3"
  if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
    echo "refusing to replace non-Git path: $destination" >&2
    exit 2
  fi
  if [[ ! -d "$destination/.git" ]]; then
    git clone "$repository" "$destination"
  fi
  git -C "$destination" fetch --tags origin
  git -C "$destination" checkout --detach "$revision"
  if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
    echo "baseline checkout is dirty: $destination" >&2
    exit 2
  fi
  local actual
  actual="$(git -C "$destination" rev-parse HEAD)"
  [[ "$actual" == "$revision" ]] || { echo "revision mismatch in $destination" >&2; exit 2; }
}

mkdir -p "$baseline_root"
checkout_pinned "$baseline_root/AFlow" "$AFLOW_REPO" "$AFLOW_COMMIT"
checkout_pinned "$baseline_root/MaAS" "$MAAS_REPO" "$MAAS_COMMIT"
checkout_pinned "$baseline_root/GDesigner" "$GDESIGNER_REPO" "$GDESIGNER_COMMIT"

if [[ "$install_python_deps" == "1" ]]; then
  command -v uv >/dev/null || { echo "uv is required for --install-python-deps" >&2; exit 2; }
  cd "$repo_root"
  uv venv --python "$python_bin" .rpas-run
  uv pip install --python .rpas-run/bin/python -e '.[full,test]'
  uv pip install --python .rpas-run/bin/python \
    'transformers>=5.16,<5.17' 'fastapi>=0.136,<0.137' \
    'uvicorn>=0.49,<0.50' 'openai>=2.44,<2.45'
fi

echo "Pinned external baselines are ready in: $baseline_root"
echo "Next: read docs/TRANSFER_TO_ANOTHER_MACHINE.md and set RPAS_MODEL_PATH."
