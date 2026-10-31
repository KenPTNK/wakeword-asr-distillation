#!/usr/bin/env bash
# Conventional DS-CNN KWS baselines (ctctknk-docs/33 §3q): same data, phase-2 v2 augmentation,
# frozen selection rule, test once. Waits for the phase-1 ablation queue to free the GPU.
set -euo pipefail
cd /media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk
PY=/home/ml4u/.conda/envs/asr/bin/python
M=checkpoints/student_L5_H192_M64_v17_t6_lr1e-4_h0/unified_manifest.jsonl
OUT=logs/v17/kws_baseline
mkdir -p $OUT
until grep -q "ablation queue ended" logs/v17/lanes.log 2>/dev/null; do sleep 120; done
for W in 64 192; do
  $PY -u kws_baseline.py train --name dscnn_w$W --width $W --manifest $M --out $OUT --context-aug
  $PY -u kws_baseline.py evaluate --name dscnn_w$W --manifest $M --out $OUT \
    --positive-manifest logs/v14_2/hk15_eval/positives_pos8.jsonl \
    --audio multi_phone_threshold_analysis/hk15_24h/audio_16k_mono.wav
done
echo "kws baselines done $(date '+%F %T')"
