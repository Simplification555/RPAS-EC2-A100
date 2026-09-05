#!/usr/bin/env bash
# Emergency guard for an already-submitted batch shell with external srun steps.
# New submissions should wait for every owned worker in their batch wrapper.
set -euo pipefail

[[ $# -ge 2 ]] || { echo "usage: $0 <batch-shell-pid> <worker-wrapper-pid>..." >&2; exit 2; }
parent="$1"
shift
[[ "$parent" =~ ^[0-9]+$ && -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ -r "/proc/$parent/environ" ]] || exit 2
tr '\0' '\n' <"/proc/$parent/environ" | grep -Fx "SLURM_JOB_ID=$SLURM_JOB_ID" >/dev/null
tr '\0' ' ' <"/proc/$parent/cmdline" | grep -F '/slurm_script' >/dev/null
parent_start="$(awk '{print $22}' "/proc/$parent/stat")"
declare -A starts
for pid in "$@"; do
  [[ "$pid" =~ ^[0-9]+$ ]] || exit 2
  tr '\0' '\n' <"/proc/$pid/environ" | grep -Fx "SLURM_JOB_ID=$SLURM_JOB_ID" >/dev/null
  starts[$pid]="$(awk '{print $22}' "/proc/$pid/stat")"
done
resume() {
  if [[ -r "/proc/$parent/stat" ]] && [[ "$(awk '{print $22}' "/proc/$parent/stat")" == "$parent_start" ]]; then
    kill -CONT "$parent" 2>/dev/null || true
  fi
}
trap resume EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
kill -STOP "$parent"
printf '%s guarding job=%s parent=%s workers=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$parent" "$*"
while :; do
  active=0
  for pid in "$@"; do
    if [[ -r "/proc/$pid/stat" ]]; then
      read -r state started < <(awk '{print $3, $22}' "/proc/$pid/stat") || continue
      if [[ "$started" == "${starts[$pid]}" && "$state" != Z ]]; then
        active=$((active + 1))
      fi
    fi
  done
  printf '%s remaining_workers=%s\n' "$(date -Is)" "$active"
  [[ "$active" -gt 0 ]] || break
  sleep 30
done
echo 'Workers exited; resuming batch shell for normal accounting and cleanup.'
