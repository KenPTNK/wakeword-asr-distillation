#!/usr/bin/env bash
# ICIIT v5, parallel worker: the three remaining DS-CNN baselines (small models) next to the main
# queue's Conformer runs.  Each: train -> `kws_baseline.py evaluate` (frozen selection rule on the
# selection speakers, writes "chosen") -> per-clip test scores.  The main queue skips these names
# because their score files exist (placeholders, overwritten here).
set -uo pipefail
cd /media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk
PY=$PWD/v5/python_v5.sh; REALPY=/home/ml4u/.conda/envs/asr/bin/python
O=logs/v5; K=logs/v5/kws; S=logs/v5/scores; REF=/home/ml4u/datasets/v5_ref/unified_manifest.jsonl
HK15=multi_phone_threshold_analysis/hk15_24h/audio_16k_mono.wav
log() { echo "[v5b] $* $(date "+%F %T")" >> logs/v17/lanes.log; }
log "start"
for spec in "192 yes" "64 no" "192 no"; do
  set -- $spec
  w=$1; name=dscnn_w$1; extra=(--context-aug)
  [[ $2 == no ]] && { name=${name}_noctx; extra=(); }
  log "$name start"
  bash run_with_oom_retry.sh $O/$name.log $PY -u kws_baseline.py train --name $name --width $w \
    --manifest $REF --out $K "${extra[@]}"
  bash run_with_oom_retry.sh $O/$name.log $PY -u kws_baseline.py evaluate --name $name --manifest $REF \
    --out $K --positive-manifest $O/positives_v5.jsonl --audio $HK15
  ck=$($REALPY -c "import json; print(json.load(open('$K/kws_baseline_$name.json'))['chosen'])")
  bash run_with_oom_retry.sh $O/$name.log $PY -u v5/v5_dump_scores.py --dscnn $K/$name/$ck --out $S/$name.npz
  log "$name scored ($ck)"
done
log "done"
