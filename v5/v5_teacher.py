#!/usr/bin/env python3
"""v5: per-clip teacher / cascade scores on the v5 test speakers (derived from paper_v4_teacher.py).

evaluate_baselines.py and teacher_textmatch.py saved only summary tables for the original four
test speakers.  To pool them with speakers 048-054 this re-runs exactly their scoring on both clip
sets (original test clips: embedding salts 1/2 as the frozen protocol; 048-054: salts 7/8 as
eval_newspk.py) and saves per-clip values, prefixed old_ / new_:
  s_pos/s_neg     stage-1 student (paper model, seed 2026 step 8k), in-context window best
  tf_pos/tf_neg   teacher, posteriors folded to 5 classes, in-context window best
  tv_pos/tv_neg   teacher folded score on the cascade verify window [clip-3 s, clip+0.64 s]
  tt_pos/tt_neg   teacher greedy text of the same window contains the phrase (bool)
and, shared by both sets (same hk15 stream): the stage-1 threshold at 10 FA/h (grid of the
original test positives, as evaluate_baselines.cascade), the teacher-folded scores of the verify
windows around every stage-1 trigger (cascade FA at any verifier threshold), and teacher-folded
hk15 FA events on a threshold grid.  The text-match FA on hk15 (0 events) does not depend on the
clips and is taken from teacher_text.json.  Nothing in the frozen protocol is modified.

usage: v5/python_v5.sh -u v5/v5_teacher.py --out logs/v5/teacher_cascade.npz
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

import evaluate_baselines as base
import evaluate_wake_suite as suite
import incontext_protocol as ic
import parakeet_v12_train as v12
import streaming_protocol as protocol
import teacher_textmatch as ttm

RUN_DIR = Path("/home/ml4u/datasets/v5_ref")
STUDENT = Path("checkpoints/student_L5_H192_M64_v5_prop/best_longaudio_recall.pt")
POS = Path("logs/v5/positives_v5.jsonl")
AUDIO = Path("multi_phone_threshold_analysis/hk15_24h/audio_16k_mono.wav")
STAGE1_TARGET = 10.0
GRID = 256


def old_clips():
    pos_waves, _ = suite.ev.load_dev_positives(POS)
    labels = suite.positive_paths(POS)
    keep = [i for i, lab in enumerate(labels) if ic.role_of(lab) == "test"]
    neg_waves, neg_labels = suite.load_hard_negatives(RUN_DIR / "unified_manifest.jsonl")
    nkeep = [i for i, lab in enumerate(neg_labels) if ic.role_of(lab) == "test"]
    return ([pos_waves[i] for i in keep], [labels[i] for i in keep],
            [neg_waves[i] for i in nkeep], [neg_labels[i] for i in nkeep])


def verify_cut(streams, windows):
    rate = protocol.SAMPLE_RATE
    before, after = int(base.VERIFY_BEFORE * rate), int(base.VERIFY_AFTER * rate)
    return [s[max(0, a - before):min(len(s), b + after)] for s, (a, b) in zip(streams, windows)]


def free_gpu():
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    v12.install_runtime_imports()
    import soundfile as sf
    long_wave, rate = sf.read(str(AUDIO), dtype="int16")
    hours = len(long_wave) / rate / 3600

    o_pos, o_pos_label, o_neg, o_neg_label = old_clips()
    streams = {"test": (base.test_streams(o_pos, [long_wave], 1), base.test_streams(o_neg, [long_wave], 2))}
    del o_pos, o_neg
    out = {"hours": hours, "test_pos_label": np.array(o_pos_label), "test_neg_label": np.array(o_neg_label)}

    # stage 1: the paper model, exactly as evaluate_baselines.single_detector / cascade
    forward = base.student_forward(STUDENT)
    for name, (ps, ns) in streams.items():
        out[f"{name}_s_pos"] = ic.window_best(forward, *ps)
        out[f"{name}_s_neg"] = ic.window_best(forward, *ns)
    finite = out["test_s_pos"][np.isfinite(out["test_s_pos"])]
    s_thr = np.linspace(float(finite.min()), float(finite.max()), GRID)
    s_probs, s_events = base.long_events(forward, long_wave, s_thr)
    per_hour = np.asarray(s_events) / hours
    index = int(np.argmin(np.abs(per_hour - STAGE1_TARGET)))
    tau1 = float(s_thr[index])
    frames = base.event_frames(s_probs, tau1)
    before, after = int(base.VERIFY_BEFORE * rate), int(base.VERIFY_AFTER * rate)
    pieces = []
    for frame in frames:
        end = (frame // protocol.CHUNK_FRAMES + 1) * protocol.CHUNK_FRAMES * protocol.FRAME_SAMPLES
        pieces.append(long_wave[max(0, end - before):min(len(long_wave), end + after)])
    out.update(tau1=tau1, stage1_fa_per_hour=float(per_hour[index]), stage1_events=len(frames))
    print(f"stage 1: tau1={tau1:.5f} at {per_hour[index]:.3f} FA/h, {len(frames)} triggers",
          flush=True)
    del forward, s_probs
    free_gpu()

    # teacher, folded posteriors: alone (in context) and as cascade verifier
    teacher = base.TeacherForward()
    for name, (ps, ns) in streams.items():
        out[f"{name}_tf_pos"] = ic.window_best(teacher, *ps)
        out[f"{name}_tf_neg"] = ic.window_best(teacher, *ns)
        out[f"{name}_tv_pos"] = base.verify_windows(teacher, verify_cut(*ps))
        out[f"{name}_tv_neg"] = base.verify_windows(teacher, verify_cut(*ns))
        print(f"teacher folded done: {name}", flush=True)
    out["t_long"] = base.verify_windows(teacher, pieces)
    grids = []
    for name in streams:
        finite = out[f"{name}_tf_pos"][np.isfinite(out[f"{name}_tf_pos"])]
        grids.append(np.linspace(float(finite.min()), float(finite.max()), GRID))
    tf_thr = np.unique(np.concatenate(grids))
    _, tf_events = base.long_events(teacher, long_wave, tf_thr)
    out.update(tf_thr=tf_thr, tf_events=np.asarray(tf_events))
    del teacher
    free_gpu()

    # teacher, greedy text + exact phrase match on the same verify windows
    text = ttm.TeacherText()
    for name, (ps, ns) in streams.items():
        out[f"{name}_tt_pos"] = np.array([ttm.hit(t) for t in text.decode(verify_cut(*ps))])
        out[f"{name}_tt_neg"] = np.array([ttm.hit(t) for t in text.decode(verify_cut(*ns))])
        print(f"text {name}: FRR {1 - out[f'{name}_tt_pos'].mean():.4f}, "
              f"NM {out[f'{name}_tt_neg'].mean():.4f}", flush=True)
    del text
    free_gpu()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
