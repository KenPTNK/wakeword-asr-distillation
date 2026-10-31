#!/usr/bin/env bash
# Phase 3a: three runs in sequence (ctctknk-docs/30 §5).
#   1 mixed  KD->CTC ladder 0.9:0.1 .. 0.1:0.9, each stage until the validation loss plateaus
#   2 kl     KD only
#   3 ctc    CTC only
# Each through run_with_oom_retry.sh: rerun 3 s after a CUDA OOM, resume from latest.pt.
# WANDB_API_KEY must already be in the environment (never written to disk).
set -uo pipefail
cd /media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk
PY=/home/ml4u/.conda/envs/asr/bin/python
L=logs/v16; mkdir -p $L
COMMON=(--workers 5 --validate-every 2000 --validate-utterances 1500 --steps 150000
        --wandb-mode online --wandb-project parakeet-vietnamese-distillation)
run() {  # name log extra...
  local name=$1 log=$2; shift 2
  echo "[v16] $name start $(date '+%F %T')" >> $L/lanes.log
  bash run_with_oom_retry.sh "$log" $PY -u parakeet_v16_phase1_train.py \
    --run-name "$name" "${COMMON[@]}" "$@"
  echo "[v16] $name exit $? $(date '+%F %T')" >> $L/lanes.log
}
run v16_phase1_mixed_trial2 $L/mixed.log \
  --stages 0.9:0.1,0.7:0.3,0.5:0.5,0.3:0.7,0.1:0.9 --patience 3 --min-delta 0.005 \
  --stage-min-steps 6000 --stage-max-steps 30000 --wandb-tags v16,phase1,64x7,lr5e4-cosine-fp32,mixed-ladder
run v16_phase1_kl_trial2 $L/kl.log \
  --stages 1:0 --patience 5 --min-delta 0.005 --stage-min-steps 6000 --stage-max-steps 150000 \
  --wandb-tags v16,phase1,64x7,lr5e4-cosine-fp32,kl-only
run v16_phase1_ctc_trial2 $L/ctc.log \
  --stages 0:1 --patience 5 --min-delta 0.005 --stage-min-steps 6000 --stage-max-steps 150000 \
  --wandb-tags v16,phase1,64x7,lr5e4-cosine-fp32,ctc-only
echo "[v16] all runs ended $(date '+%F %T')" >> $L/lanes.log
