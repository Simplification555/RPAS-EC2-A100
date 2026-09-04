#!/usr/bin/env bash
# Validate one EC-3 V3 worker binding before calibration or formal dispatch.
# Actual search/test drivers must call this launcher first and remain bound to
# the same single physical GPU for their whole process.

set -euo pipefail

stage="${1:-}"
if [[ "${stage}" != "calibration" && "${stage}" != "formal" ]]; then
  echo "usage: $0 {calibration|formal}" >&2
  exit 2
fi
gpu="${RPAS_EC3_GPU:-}"
if [[ "${gpu}" != "4" && "${gpu}" != "5" ]]; then
  echo "set RPAS_EC3_GPU to exactly 4 or 5" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-${gpu}}" != "${gpu}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must match RPAS_EC3_GPU exactly" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${RPAS_REPO_ROOT:-$(cd "${script_dir}/.." && pwd)}"
python_bin="${RPAS_EC3_PYTHON:-${repo_root}/.rpas-run/bin/python}"
manifest="${RPAS_EC3_MANIFEST:?set RPAS_EC3_MANIFEST to frozen hotpotqa_manifest.json}"
aflow_root="${RPAS_AFLOW_ROOT:?set RPAS_AFLOW_ROOT to FoundationAgents/AFlow@3f457218 checkout}"
if [[ "${gpu}" == "4" ]]; then
  endpoint_default="http://127.0.0.1:29500/v1"
else
  endpoint_default="http://127.0.0.1:29501/v1"
fi
export CUDA_VISIBLE_DEVICES="${gpu}"
export RPAS_EXTERNAL_API_BASE="${RPAS_EXTERNAL_API_BASE:-${endpoint_default}}"

[[ -x "${python_bin}" ]] || { echo "missing EC-3 python runtime: ${python_bin}" >&2; exit 2; }
[[ -f "${manifest}" ]] || { echo "missing frozen EC-3 manifest: ${manifest}" >&2; exit 2; }
[[ -d "${aflow_root}" ]] || { echo "missing AFlow source: ${aflow_root}" >&2; exit 2; }

cd "${repo_root}"
"${python_bin}" -m external_comparison.runners.ec3_preflight \
  --manifest "${manifest}" --aflow-root "${aflow_root}" --expected-endpoint "${RPAS_EXTERNAL_API_BASE}" \
  --output "${RPAS_EC3_PREFLIGHT_OUTPUT:-${repo_root}/outputs/external_comparison/ec3_hotpotqa_v3/preflight_gpu${gpu}.json}"

echo "EC-3 V3 ${stage} preflight passed on physical GPU ${gpu}."
