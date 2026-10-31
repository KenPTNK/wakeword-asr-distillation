#!/usr/bin/env bash
# ICIIT v5: every paper system retrained on the v5 speaker split (v5/make_split.py: 10 test /
# 5 selection / 37 train, stratified by gender and accent; 048-054 added) with the recipes of
# paper_rev2_queue.sh / paper_abl_v1.sh / v17_lanes_v2.sh / kws_baseline*.sh unchanged.
# One GPU job at a time (shared with vLLM). Each run: train -> keep only the selected checkpoint
# -> per-clip test scores (v5/v5_dump_scores.py).
set -uo pipefail
cd /media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk
REALPY=/home/ml4u/.conda/envs/asr/bin/python
export CTCTKNK_PYTHON=$PWD/v5/python_v5.sh V14_FREQ_MASK_MIN_BIN=6
export V14_NEGATIVE_MANIFEST=/home/ml4u/datasets/v17_negatives_manifest.jsonl
export CHANNEL_NEGATIVE_MANIFEST=/home/ml4u/datasets/v17_channel_negatives_manifest.jsonl
export V14_2_RATIOS=1 V14_2_STAGE_STEPS=10000
export WAKEWORD_SHARDED_MANIFEST=/home/ml4u/datasets/v5_wakeword_channel_manifest.jsonl
export SPEAKER_SPLIT_FILE=$PWD/v5/v5_trainer_splits.json
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY=$($REALPY -c 'import netrc; print(netrc.netrc().authenticators("api.wandb.ai")[2])')
  export WANDB_API_KEY
fi
E=/home/ml4u/datasets/ctctknk_3a/checkpoints
T6=$E/v16_phase1_ladder10_trial6/encoder3a_student_weights.pt
CTC=$E/v16_phase1_ctc_trial5_full/encoder3a_student_weights.pt
LAD=$E/v16_phase1_ladder10_trial5/encoder3a_student_weights.pt
V11=$PWD/checkpoints/student_L5_H192_M64_v11_blockB1_adaptive_kdctc_streaming_chunk640ms_cappedmax_nang2_khieu4_nangkhieu2_y2_oi2_cap4_reserve_target_tokens_unbounded_infore2_full_cappedmax_epochcontroller_trial1/student_final_weights.pt
SH="WakewordV12=0.25,InfoRE2=0.30,VLSP=0.15,LSVSC=0.12,ViMD=0.10,MUSAN=0.08"
C=checkpoints/student_L5_H192_M64; L=logs/v5; S=logs/v5/scores; K=logs/v5/kws
REF=/home/ml4u/datasets/v5_ref/unified_manifest.jsonl
mkdir -p $L $S $K
log() { echo "[v5] $* $(date "+%F %T")" >> logs/v17/lanes.log; }
keep_selected() {  # new v5 run dirs only: keep the selected checkpoint and small metadata
  local d=$1
  [[ $d == ${C}_v5_* ]] || return 0
  test -s $d/best_longaudio_recall.pt || return 0
  find $d -mindepth 1 -maxdepth 1 ! -name best_longaudio_recall.pt ! -name best_longaudio_state.json \
    ! -name run_config.json ! -name last_launch.json ! -name training_metrics.jsonl \
    ! -name speaker_splits.json ! -name initialization_audit.json -exec rm -rf {} +
}
run() {  # name init encoder_lr head_only_steps policy seed context_aug
  local name=$1 init=$2 lr=$3 head=$4 policy=$5 seed=$6 ctx=$7 seedarg=()
  [[ -n "$seed" ]] && seedarg=(--seed "$seed")
  [[ -s ${C}_$name/best_longaudio_recall.pt && -s $S/$name.npz ]] && { log "$name skip (done)"; return; }
  log "$name start"
  bash run_with_oom_retry.sh $L/$name.log env V14_2_RUN_NAME=student_L5_H192_M64_$name \
    V14_INIT_CHECKPOINT=$init V14_2_ENCODER_LR=$lr V14_2_HEAD_ONLY_STEPS=$head \
    V17_TARGET_POLICY=$policy V17_CONTEXT_AUG=$ctx WANDB_TAGS=v5,v5-split,targets-$policy \
    bash train_v17.sh --v14-negative-source-shares "$SH" --v14-2-in-context-selection "${seedarg[@]}"
  log "$name exit $?"
  keep_selected ${C}_$name
  $CTCTKNK_PYTHON -u v5/v5_dump_scores.py --student ${C}_$name/best_longaudio_recall.pt \
    --out $S/$name.npz >> $L/$name.log 2>&1
  log "$name scored"
}
dscnn() {  # width context(yes/no)
  local w=$1 ctx=$2 name=dscnn_w$1 extra=()
  [[ $ctx == no ]] && name=${name}_noctx || extra=(--context-aug)
  [[ -s $S/$name.npz ]] && { log "$name skip (done)"; return; }
  log "$name start"
  bash run_with_oom_retry.sh $L/$name.log $CTCTKNK_PYTHON -u kws_baseline.py train --name $name \
    --width $w --manifest $REF --out $K "${extra[@]}"
  local ck=$($REALPY -c "import json; print(json.load(open('$K/kws_baseline_$name.json'))['chosen'])")
  $CTCTKNK_PYTHON -u v5/v5_dump_scores.py --dscnn $K/$name/$ck --out $S/$name.npz >> $L/$name.log 2>&1
  log "$name scored"
}
log "queue start"
run v5_prop           $T6  0.0001 0    words ""   0
run v5_ctc            $CTC 0.0001 0    words ""   0
run v5_kd             $LAD 0.0001 0    words ""   0
run v5_whole          $T6  0.0001 0    whole ""   0
run v5_v2             $T6  0.0001 0    words ""   1
run v5_random         random 0.001 0   words ""   0
run v5_v11            $V11 0.0001 0    words ""   0
dscnn 64 yes
run v5_prop_s2027     $T6  0.0001 0    words 2027 0
run v5_ctc_s2027      $CTC 0.0001 0    words 2027 0
run v5_kd_s2027       $LAD 0.0001 0    words 2027 0
run v5_whole_s2027    $T6  0.0001 0    whole 2027 0
run v5_prop_s2028     $T6  0.0001 0    words 2028 0
run v5_ctc_s2028      $CTC 0.0001 0    words 2028 0
run v5_kd_s2028       $LAD 0.0001 0    words 2028 0
run v5_whole_s2028    $T6  0.0001 0    whole 2028 0
dscnn 192 yes
dscnn 64 no
dscnn 192 no
run v5_earlier        $V11 0.00001 2000 words ""  0
log "queue done"
