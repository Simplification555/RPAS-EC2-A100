#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
[[ "${WSL_DISTRO_NAME:-}" == zyc111 ]] || { echo 'This launcher requires WSL zyc111'; exit 2; }
UV="$ROOT/.tools/uv/bin/uv"
PY="$ROOT/.tools/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12"
export UV_CACHE_DIR="$ROOT/.tools/cache"
export UV_HTTP_TIMEOUT=600
export UV_HTTP_CONNECT_TIMEOUT=60
export UV_CONCURRENT_DOWNLOADS=4
export UV_HTTP_RETRIES=6
if [[ ! -x .venv-serve/bin/python ]]; then "$UV" venv --seed --python "$PY" .venv-serve; fi
"$UV" pip install --python .venv-serve/bin/python 'vllm==0.18.1' 'openai>=1.82,<2' 'huggingface_hub<1' --index-url https://pypi.tuna.tsinghua.edu.cn/simple
.venv-serve/bin/python -m pip check
.venv-serve/bin/python -m pip freeze > serving_requirements.lock.txt
if [[ ! -x .venv-method/bin/python ]]; then "$UV" venv --seed --python "$PY" .venv-method; fi
"$UV" pip install --python .venv-method/bin/python 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install --python .venv-method/bin/python -r requirements-method.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
.venv-method/bin/python -m pip check
.venv-method/bin/python -m pip freeze > method_requirements.lock.txt
bash repo/scripts/setup_baselines.sh
PYTHONPATH="$ROOT/repo:$ROOT/repo/experiments:$ROOT/repo/external_comparison" .venv-method/bin/python -m unittest discover -s repo/experiments -p 'test_*.py' -v
printf 'PRO6000_ENV_READY\n'
