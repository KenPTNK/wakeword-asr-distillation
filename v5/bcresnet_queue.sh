#!/usr/bin/env bash
# BC-ResNet-8 baseline on the v5 Phase-2 data (29/09/2026): features -> train -> score; stops at the
# first failure.  Uses oww_v5/align.json (phrase ends) from oww_queue.sh.
set -uo pipefail
cd /media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk
PY=$PWD/v5/python_v5.sh
L=logs/v5/bcresnet; mkdir -p $L
log() { echo "[bcresnet] $* $(date '+%F %T')" >> logs/v17/lanes.log; }
for stage in ${@:-features train score}; do
  log "$stage start"
  if ! bash run_with_oom_retry.sh $L/$stage.log $PY -u v5/bcresnet_baseline.py $stage; then
    log "$stage FAILED"; exit 1
  fi
  log "$stage done"
done
log "queue done"
