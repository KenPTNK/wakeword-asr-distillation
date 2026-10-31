#!/usr/bin/env bash
# V17 launcher: train_v14_2.sh calling parakeet_v17_train.py (word targets, [64,7]); ctctknk-docs/33.
# V14.2 = V14 + three measured corrections.
#
#   * checkpoints chosen by recall at <= 1 false accept/hour on *continuous
#     YouTube audio* with the automaton reset after each hit — the metric the
#     product is judged by.  Selection uses three shorter clips (4.87 h);
#     hk15_24h (11.76 h) is never touched so the final number stays honest.
#   * channel augmentation: a 300-3400 Hz band drops recall from 0.948 to
#     0.222, so every source (positives and all negatives) gets copies through
#     band limits and measured room responses -- precomputed on disk by
#     prepare_v14_2_channel_shards.py (room 0.15, band 0.25), never live.
#   * one eight-speaker evaluation split that contains women and two northern
#     speakers, instead of four male speakers of whom one dominated the metric.
#     (The first V14.2 runs silently fell back to a reshuffled 4/4 split; the
#     trainer now honours the file.)
#
# Prepare the copies first (CPU only, ~40 GB):
#   python prepare_v14_2_channel_shards.py --output-root "${CHANNEL_ROOT}"
# Check the whole data path without touching the GPU:
#   bash train_v14_2.sh --audit-only
#
# Dropped from V14.1: confusion-class sampling and the rebalanced source
# shares.  Both lost to plain V14 on long audio (0.87 vs 0.93).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
CHECKPOINT_ROOT="${V11_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
SOURCE_RUN="student_L5_H192_M64_v11_blockB1_adaptive_kdctc_streaming_chunk640ms_cappedmax_nang2_khieu4_nangkhieu2_y2_oi2_cap4_reserve_target_tokens_unbounded_infore2_full_cappedmax_epochcontroller_trial1"

CTCTKNK_DATA_ROOT="${CTCTKNK_DATA_ROOT:-/home/ml4u/datasets/ctctknk_data_nvme}"
CHANNEL_ROOT="${CHANNEL_ROOT:-/home/ml4u/datasets/v14_2_channel_16k}"
# Originals + positive/negative WakewordV12 channel copies (same families).
# V14_2_CHANNEL=off trains on the originals only (the ablation that measures
# what the channel copies contribute); anything else uses the copies.
CHANNEL_MODE="${V14_2_CHANNEL:-shards}"
if [[ "${CHANNEL_MODE}" == "off" ]]; then
  WAKEWORD_SHARDED_MANIFEST="${WAKEWORD_SHARDED_MANIFEST:-/home/ml4u/datasets/wakeword_v12_16k_sharded/wakeword_v12_16k_sharded_manifest.jsonl}"
  CHANNEL_ARGS=()
else
  WAKEWORD_SHARDED_MANIFEST="${WAKEWORD_SHARDED_MANIFEST:-${CHANNEL_ROOT}/v14_2_wakeword_channel_manifest.jsonl}"
  CHANNEL_NEGATIVE_MANIFEST="${CHANNEL_NEGATIVE_MANIFEST:-${CHANNEL_ROOT}/v14_2_channel_negatives_manifest.jsonl}"
  CHANNEL_ARGS=(--v14-2-channel-negative-manifest "${CHANNEL_NEGATIVE_MANIFEST}")
fi
V14_NEGATIVE_MANIFEST="${V14_NEGATIVE_MANIFEST:-/home/ml4u/datasets/v14_negatives_16k/v14_negatives_16k_manifest.jsonl}"
SPEAKER_SPLIT_FILE="${SPEAKER_SPLIT_FILE:-${PROJECT_ROOT}/v14_2_speaker_splits.json}"
LONG_AUDIO_DIR="${LONG_AUDIO_DIR:-${PROJECT_ROOT}/multi_phone_threshold_analysis}"

# Selection corpus: everything except hk15_24h, which stays sealed.
SELECT_AUDIO=(
  "${LONG_AUDIO_DIR}/2_ngay_1_dem_s3_ep60/audio_16k_mono.wav"
  "${LONG_AUDIO_DIR}/have_a_sip_256/audio_16k_mono.wav"
  "${LONG_AUDIO_DIR}/have_a_sip_52/audio_16k_mono.wav"
)

RATIOS="${V14_2_RATIOS:-1,8}"
STAGE_STEPS="${V14_2_STAGE_STEPS:-6000,14000}"
VALIDATION_EVERY="${V14_2_VALIDATION_EVERY:-2000}"
SELECT_FA="${V14_2_SELECT_FA:-1.0}"
# mean fill above bin 6 won on long audio (0.9480 vs 0.9368 for bin 21); the
# gap is inside the noise floor, so either is defensible.
V14_FREQ_MASKS="${V14_FREQ_MASKS:-2}"
V14_FREQ_WIDTH="${V14_FREQ_WIDTH:-8}"
V14_FREQ_MASK_FILL="${V14_FREQ_MASK_FILL:-mean}"
V14_FREQ_MASK_MIN_BIN="${V14_FREQ_MASK_MIN_BIN:-6}"

TOKENIZER_MODEL="${TOKENIZER_MODEL:-${PROJECT_ROOT}/checkpoints/parakeet_teacher_tokenizer.model}"
INIT_CHECKPOINT="${V14_INIT_CHECKPOINT:-${CHECKPOINT_ROOT}/${SOURCE_RUN}/student_final_weights.pt}"
# V14_INIT_CHECKPOINT=random: the from-scratch arm of ctctknk-docs/26 phase 2.
# A random encoder cannot be fine-tuned at 1e-5 behind 2,000 head-only steps,
# so that arm sets these three (the pretrained arms keep the defaults).
ENCODER_LR="${V14_2_ENCODER_LR:-0.00001}"
DECODER_LR="${V14_2_DECODER_LR:-0.0001}"
HEAD_ONLY_STEPS="${V14_2_HEAD_ONLY_STEPS:-2000}"
if [[ "${INIT_CHECKPOINT}" == "random" ]]; then
  INIT_ARGS=(--allow-random-init)
else
  INIT_ARGS=(--init-checkpoint "${INIT_CHECKPOINT}")
fi
# The mask position goes in the run name: the two positions are compared as
# two runs, so they must not write into the same directory.
# A new name: the *_channelaug_split8_* directories hold latest.pt from the
# live-augmentation, reshuffled-split runs, and this script resumes latest.pt.
RUN_NAME="${V14_2_RUN_NAME:-student_L5_H192_M64_v17_trial1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_USERNAME="${WANDB_USERNAME:-khactrungkiennguyen}"
WANDB_ENTITY="${WANDB_ENTITY:-khactrungkiennguyen-ai-tech-lab}"
WANDB_PROJECT="${WANDB_PROJECT:-parakeet-vietnamese-distillation}"
WANDB_TAGS="${WANDB_TAGS:-v14.2,longaudio-selection,channel-shards,split8,reduced-vocabulary,pure-ctc,streaming,chunk-640ms,curriculum1to8,two-stage,minbin-${V14_FREQ_MASK_MIN_BIN}}"

fail() { echo "ERROR: $*" >&2; exit 2; }

PYTHON="${CTCTKNK_PYTHON:-}"
[[ -z "${PYTHON}" ]] && PYTHON="$(command -v python || true)"
[[ -z "${PYTHON}" && -x /home/ml4u/.conda/envs/asr/bin/python ]] \
  && PYTHON=/home/ml4u/.conda/envs/asr/bin/python
[[ -n "${PYTHON}" ]] || fail "No python found; set CTCTKNK_PYTHON"

[[ -d "${CTCTKNK_DATA_ROOT}" ]] || fail "Missing CTCTKNK_DATA_ROOT: ${CTCTKNK_DATA_ROOT}"
[[ -s "${WAKEWORD_SHARDED_MANIFEST}" ]] || fail "Missing wakeword manifest: ${WAKEWORD_SHARDED_MANIFEST} (run prepare_v14_2_channel_shards.py)"
if [[ "${CHANNEL_MODE}" != "off" ]]; then
  [[ -s "${CHANNEL_NEGATIVE_MANIFEST}" ]] || fail "Missing channel negative manifest: ${CHANNEL_NEGATIVE_MANIFEST}"
fi
[[ -s "${V14_NEGATIVE_MANIFEST}" ]] || fail "Missing negative manifest"
[[ -s "${SPEAKER_SPLIT_FILE}" ]] || fail "Missing speaker split: ${SPEAKER_SPLIT_FILE}"
[[ -s "${TOKENIZER_MODEL}" ]] || fail "Missing tokenizer"
[[ "${INIT_CHECKPOINT}" == "random" || -s "${INIT_CHECKPOINT}" ]] || fail "Missing initialization checkpoint"
for clip in "${SELECT_AUDIO[@]}"; do
  [[ -s "${clip}" ]] || fail "Missing selection audio: ${clip}"
  [[ "${clip}" == *hk15_24h* ]] && fail "hk15_24h must stay out of selection"
done

"${PYTHON}" -u "${SCRIPT_DIR}/validate_v14_2_training.py" >/dev/null \
  || fail "validate_v14_2_training.py failed; run it directly to see which check broke"

RUN_DIRECTORY="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
if [[ -s "${RUN_DIRECTORY}/student_final_weights.pt" ]]; then
  echo "V14.2 already finished: ${RUN_DIRECTORY}/student_final_weights.pt"
  exit 0
fi
if [[ -s "${RUN_DIRECTORY}/latest.pt" ]]; then
  echo "Resuming V14.2: ${RUN_DIRECTORY}/latest.pt"
else
  echo "Starting V14.2 from: ${INIT_CHECKPOINT}"
fi
echo "Init         : ${INIT_CHECKPOINT} (encoder lr ${ENCODER_LR}, head lr ${DECODER_LR}, head-only ${HEAD_ONLY_STEPS} steps)"
echo "Curriculum   : ratios ${RATIOS} for ${STAGE_STEPS} steps"
echo "Selection    : recall at <= ${SELECT_FA} FA/h on ${#SELECT_AUDIO[@]} long clips"
if [[ "${CHANNEL_MODE}" == "off" ]]; then
  echo "Channel      : OFF (originals only, ablation)"
else
  echo "Channel      : precomputed copies in ${CHANNEL_ROOT} (live augmentation off)"
fi
echo "Speakers     : ${SPEAKER_SPLIT_FILE}"

"${PYTHON}" -u "${SCRIPT_DIR}/parakeet_v17_train.py" \
  --v14-2-stage-steps "${STAGE_STEPS}" \
  --v14-2-select-audio "${SELECT_AUDIO[@]}" \
  --v14-2-select-fa-per-hour "${SELECT_FA}" \
  "${CHANNEL_ARGS[@]}" \
  --v14-negative-manifest "${V14_NEGATIVE_MANIFEST}" \
  --v14-freq-masks "${V14_FREQ_MASKS}" \
  --v14-freq-width "${V14_FREQ_WIDTH}" \
  --v14-freq-mask-fill "${V14_FREQ_MASK_FILL}" \
  --v14-freq-mask-min-bin "${V14_FREQ_MASK_MIN_BIN}" \
  --speaker-split-file "${SPEAKER_SPLIT_FILE}" \
  --project-root "${PROJECT_ROOT}" \
  --data-root "${CTCTKNK_DATA_ROOT}" \
  --wakeword-manifest "${WAKEWORD_SHARDED_MANIFEST}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  "${INIT_ARGS[@]}" \
  --run-name "${RUN_NAME}" \
  --negative-positive-ratios "${RATIOS}" \
  --speaker-split-seed 2026 \
  --new-data-negative-share 0.25 \
  --hard-negative-share 0.75 \
  --negative-slots-per-epoch 80000 \
  --target-encoder-frames-per-update 2000 \
  --encoder-learning-rate "${ENCODER_LR}" \
  --decoder-learning-rate "${DECODER_LR}" \
  --head-only-optimizer-steps "${HEAD_ONLY_STEPS}" \
  --warmup-steps 1000 \
  --validation-amp-dtype fp32 \
  --test-amp-dtype fp32 \
  --validation-every-steps "${VALIDATION_EVERY}" \
  --stage-lr-decay 0.80 \
  --num-workers 6 \
  --log-every-steps 10 \
  --plot-every-steps 1000 \
  --save-every-steps 500 \
  --no-phrase-score-audit \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-expected-username "${WANDB_USERNAME}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-tags "${WANDB_TAGS}" \
  --shard-row-group-size 64 \
  --shard-locality-block-size 64 \
  "$@"

if [[ -s "${RUN_DIRECTORY}/student_final_weights.pt" ]]; then
  echo "V14.2 finished: ${RUN_DIRECTORY}/student_final_weights.pt"
  echo "Exported from : ${RUN_DIRECTORY}/best_longaudio_recall.pt"
  echo "Final gate    : score it on hk15_24h, which selection never saw:"
  echo "  python evaluate_long_audio_streaming.py --checkpoints v14.2=<ckpt> \\"
  echo "    --audio ${LONG_AUDIO_DIR}/hk15_24h/audio_16k_mono.wav \\"
  echo "    --positive-manifest ${RUN_DIRECTORY}/unified_manifest.jsonl"
fi
