#!/usr/bin/env bash
# ICIIT v5: remaining paper measurements on the v5 split, after v5_queue.sh has trained and
# scored every system. One GPU job at a time. Waits for "[v5] queue done".
set -uo pipefail
cd /media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk
PY=$PWD/v5/python_v5.sh; REALPY=/home/ml4u/.conda/envs/asr/bin/python
C=checkpoints/student_L5_H192_M64; O=logs/v5; K=logs/v5/kws
HK15=multi_phone_threshold_analysis/hk15_24h/audio_16k_mono.wav
NOISE=/media/ml4u/KINGSTON/ctctknk/audio_shards_3a/MS-SNSD/noise/noise_test/0000.parquet
REF=/home/ml4u/datasets/v5_ref
log() { echo "[v5-eval] $* $(date "+%F %T")" >> logs/v17/lanes.log; }
until grep -q "\[v5\] queue done" logs/v17/lanes.log; do sleep 120; done
log "start"
selection_threshold() {  # the frozen rule's operating threshold at the selected checkpoint
  $REALPY - "$1" <<'PYEOF'
import json, sys
d = sys.argv[1]
step = json.load(open(f"{d}/best_longaudio_state.json"))["optimizer_step"]
thr = None
for line in open(f"{d}/training_metrics.jsonl"):
    r = json.loads(line)
    if r.get("optimizer_step") == step and "la_ic_selection_threshold" in r:
        thr = float(r["la_ic_selection_threshold"])
print(thr)
PYEOF
}
chosen() { $REALPY -c "import json; print(json.load(open('$K/kws_baseline_$1.json'))['chosen'])"; }
THR=$(selection_threshold ${C}_v5_prop); THR2=$(selection_threshold ${C}_v5_v2)
echo "v5_prop $THR v5_v2 $THR2" > $O/deploy_thresholds.txt
e() { bash run_with_oom_retry.sh $O/eval.log "$@"; }
e $PY -u v5/v5_dump_scores.py --student ${C}_v5_prop/best_longaudio_recall.pt --extra-thresholds $THR \
  --out $O/scores/v5_prop_deploy.npz
log "deploy scores"
e $PY -u v5/v5_teacher.py --out $O/teacher_cascade.npz
log "teacher"
for m in v5_prop v5_v11 v5_earlier; do
  e $PY -u v5/v5_isolated.py --student ${C}_$m/best_longaudio_recall.pt --label $m --out-dir $O/isolated
done
for n in dscnn_w64_noctx dscnn_w64; do
  e $PY -u v5/v5_isolated.py --dscnn $K/$n/$(chosen $n) --label $n --out-dir $O/isolated
done
log "isolated"
e $PY -u evaluate_baselines.py --skip-teacher --student ${C}_v5_prop/best_longaudio_recall.pt \
  --run-dir $REF --positive-manifest $O/positives_v5.jsonl --audio $HK15 \
  --det-out $O/det_v5_prop.csv --out $O/baselines
e $PY -u evaluate_in_context.py --run-dir ${C}_v5_prop --checkpoint ${C}_v5_prop/best_longaudio_recall.pt \
  --det-csv $O/det_v5_prop.csv --audio $HK15 --positive-manifest $O/positives_v5.jsonl \
  --noise-parquet $NOISE --near-miss-manifest $REF/unified_manifest.jsonl --role test \
  --output-dir $O/conditions
log "conditions"
e $PY -u v5/v5_dump_scores.py --student ${C}_v5_prop/best_longaudio_recall.pt --fake-int8 \
  --out $O/scores/v5_prop_int8.npz
e $PY -u measure_channel_robustness.py --checkpoint ${C}_v5_prop/best_longaudio_recall.pt \
  --manifest $O/positives_v5.jsonl --negative-audio $HK15 --fixed-threshold=$THR --output-dir $O/channel
log "int8+channel"
e $PY -u final_gate.py --checkpoint ${C}_v5_prop/best_longaudio_recall.pt --threshold $THR \
  --out $O/final_gate_v5_prop.json
log "gate prop"
e $PY -u final_gate.py --checkpoint ${C}_v5_v2/best_longaudio_recall.pt --threshold $THR2 \
  --out $O/final_gate_v5_v2.json
log "done"
