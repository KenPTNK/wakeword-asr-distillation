#!/usr/bin/env bash
# ICIIT v5: false accepts of every system on the sealed multi-domain 100 h set at all thresholds
# (v5/v5_longfa.py), for FRR at 0.1 FA/h on hk15 + sealed.  Starts after the eval queue, i.e.
# after the one-shot final gate of the deployment threshold.  One GPU job at a time.
set -uo pipefail
cd /media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk
PY=$PWD/v5/python_v5.sh; REALPY=/home/ml4u/.conda/envs/asr/bin/python
C=checkpoints/student_L5_H192_M64; O=logs/v5; K=logs/v5/kws; F=logs/v5/longfa
mkdir -p $F
log() { echo "[v5-longfa] $* $(date "+%F %T")" >> logs/v17/lanes.log; }
until grep -q "\[v5-eval\] done" logs/v17/lanes.log; do sleep 120; done
log "start"
e() { bash run_with_oom_retry.sh $F/run.log "$@"; }
for m in v5_prop v5_prop_s2027 v5_prop_s2028 v5_kd v5_kd_s2027 v5_kd_s2028 v5_ctc v5_ctc_s2027 \
         v5_ctc_s2028 v5_whole v5_whole_s2027 v5_whole_s2028 v5_v2 v5_random v5_v11; do
  [[ -s $F/$m.npz ]] && continue
  e $PY -u v5/v5_longfa.py --student ${C}_$m/best_longaudio_recall.pt --scores $O/scores/$m.npz --out $F/$m.npz
  log "$m"
done
for n in dscnn_w64 dscnn_w192 dscnn_w64_noctx dscnn_w192_noctx; do
  [[ -s $F/$n.npz ]] && continue
  ck=$($REALPY -c "import json; print(json.load(open('$K/kws_baseline_$n.json'))['chosen'])")
  e $PY -u v5/v5_longfa.py --dscnn $K/$n/$ck --scores $O/scores/$n.npz --out $F/$n.npz
  log "$n"
done
e $PY -u v5/v5_longfa.py --teacher --scores $O/teacher_cascade.npz \
  --stage1 ${C}_v5_prop/best_longaudio_recall.pt --out $F/teacher.npz
log "done"
