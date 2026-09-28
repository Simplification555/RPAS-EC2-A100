#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/upstream"
mkdir -p "$DEST"

clone_pinned() {
  local name="$1" url="$2" commit="$3" target="$DEST/$1"
  if [[ -e "$target" ]]; then
    [[ -d "$target/.git" ]] || { echo "Refusing non-Git upstream path: $target" >&2; exit 2; }
    [[ "$(git -C "$target" remote get-url origin)" == "$url" ]] || { echo "Wrong origin at $target" >&2; exit 2; }
    [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]] || { echo "Wrong pinned commit at $target" >&2; exit 2; }
    [[ -z "$(git -C "$target" status --porcelain --untracked-files=normal)" ]] || { echo "Dirty upstream at $target" >&2; exit 2; }
    echo "$name already verified at $commit"
    return
  fi
  git clone --no-checkout "$url" "$target"
  git -C "$target" checkout --detach "$commit"
  [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]]
  [[ -z "$(git -C "$target" status --porcelain --untracked-files=normal)" ]]
  echo "$name checked out and verified at $commit"
}

clone_pinned AFlow https://github.com/FoundationAgents/AFlow.git 3f457218fc716093fe53f6df8a5d5e6379d66346
clone_pinned MaAS https://github.com/bingreeky/MaAS.git 987f3c1bc9a96e844fe090db3791446e3ef0f5c7
clone_pinned ADAS https://github.com/ShengranHu/ADAS.git 2702bee8fefda42255efc5be9f60e3bd3db96ae4
clone_pinned GDesigner https://github.com/yanweiyue/GDesigner.git a6efcfa3b40bb4d9cbf46f883a95d62020bd8251

echo "UPSTREAMS_VERIFIED"
