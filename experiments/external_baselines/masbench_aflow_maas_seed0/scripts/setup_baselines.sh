#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external_baselines"
mkdir -p "$DEST"

clone_pinned() {
  local name="$1" url="$2" commit="$3" target="$DEST/$1"
  if [[ -e "$target" ]]; then
    if git -C "$target" rev-parse --is-inside-work-tree >/dev/null 2>&1 && \
       [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]]; then
      echo "$name already at pinned commit $commit"
      return
    fi
    echo "Refusing to overwrite existing non-matching baseline: $target" >&2
    exit 2
  fi
  git clone --no-checkout "$url" "$target"
  git -C "$target" checkout --detach "$commit"
  [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]]
  echo "$name checked out at $commit"
}

clone_pinned AFlow https://github.com/FoundationAgents/AFlow.git 3f457218fc716093fe53f6df8a5d5e6379d66346
clone_pinned MaAS https://github.com/bingreeky/MaAS.git 987f3c1bc9a96e844fe090db3791446e3ef0f5c7
