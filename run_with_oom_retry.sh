#!/usr/bin/env bash
# Rerun a training command when it dies of CUDA out-of-memory.
#
# The 4090 is shared with the team: two V14.2 runs were lost on 2026-09-21
# when another job grew to 16 GiB.  train_v14_2.sh resumes from latest.pt
# (saved every 500 steps), so a rerun loses at most those steps.  Any other
# failure stops the loop -- a real bug must not be retried forever.
#
#   run_with_oom_retry.sh <log file> <command> [args...]
#
# OOM_RETRY_DELAY (seconds, default 3) and OOM_MAX_RETRIES (default 100).
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <log file> <command> [args...]" >&2
  exit 2
fi
LOG="$1"; shift
DELAY="${OOM_RETRY_DELAY:-3}"
MAX_RETRIES="${OOM_MAX_RETRIES:-100}"
OOM_PATTERN="CUDA out of memory|OutOfMemoryError|CUDA error: out of memory"

mkdir -p "$(dirname -- "${LOG}")"
touch "${LOG}"
attempt=0
while true; do
  attempt=$((attempt + 1))
  first_line=$(( $(wc -l < "${LOG}" 2>/dev/null || echo 0) + 1 ))
  echo "[oom-retry] attempt ${attempt} at $(date '+%F %T'): $*" >> "${LOG}"
  "$@" >> "${LOG}" 2>&1
  status=$?
  if [[ ${status} -eq 0 ]]; then
    echo "[oom-retry] finished at $(date '+%F %T') after ${attempt} attempt(s)" >> "${LOG}"
    exit 0
  fi
  # Only this attempt's lines: an old OOM higher up must not trigger a retry.
  if ! tail -n "+${first_line}" "${LOG}" | grep -qE "${OOM_PATTERN}"; then
    echo "[oom-retry] exit ${status} without CUDA OOM at $(date '+%F %T'); not retrying" >> "${LOG}"
    exit "${status}"
  fi
  if (( attempt > MAX_RETRIES )); then
    echo "[oom-retry] CUDA OOM again; giving up after ${attempt} attempts" >> "${LOG}"
    exit "${status}"
  fi
  echo "[oom-retry] CUDA OOM (exit ${status}); rerunning in ${DELAY}s from latest.pt" >> "${LOG}"
  sleep "${DELAY}"
done
