#!/usr/bin/env bash
# openWakeWord baseline on the v5 Phase-2 data (senior review 28/09/2026): align -> features ->
# train -> score, each step logged; stops at the first failure.
set -uo pipefail
cd /media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk
PY=$PWD/v5/python_v5.sh
L=logs/v5/oww; mkdir -p $L
log() { echo "[oww] $* $(date '+%F %T')" >> logs/v17/lanes.log; }
for stage in ${@:-align features train score}; do
  log "$stage start"
  if ! bash run_with_oom_retry.sh $L/$stage.log $PY -u v5/oww_baseline.py $stage; then
    log "$stage FAILED"; exit 1
  fi
  log "$stage done"
done
log "queue done"
