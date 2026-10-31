#!/usr/bin/env python3
"""v5: false accepts on the sealed multi-domain set (sealed_longaudio_v1: 105 videos, 75 channels,
9 categories, ~100 h) at every threshold of a model's hk15 grid, so FA/h can be measured on
hk15 + sealed (~112 h) and FRR reported at 0.1 FA/h with ~11 false accepts.

Same streaming as the hk15 counts of dump_scores.py (protocol.stream_probabilities, reset
automaton, 1 s refractory).  As in final_gate.py, audio covered by VSV-1100 training clips
(dedup coverage masks) is excluded: the probability stream is cut at the masks and the masked
time is removed from the hours.  Run only after the one-shot final gate of the deployment
threshold (v5_eval_queue.sh).

  --student CKPT | --dscnn CKPT  --scores NPZ   events at the npz thresholds, per file
  --teacher --scores TEACHER_NPZ --stage1 CKPT  teacher folded events at its grid; cascade
        (stage 1 at the npz tau1): folded verifier scores and text hits of every trigger;
        teacher text match alone (4 s windows, 2 s hop, hits within 3 s merged)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_baselines as base
import parakeet_v12_train as v12
import streaming_protocol as protocol

SEALED = Path("/home/ml4u/datasets/sealed_longaudio_v1")
COVERAGE = Path("/home/ml4u/datasets/dedup/coverage.json")


def sealed_files():
    manifest = {}
    for line in open(SEALED / "manifest.jsonl", encoding="utf-8"):
        row = json.loads(line)
        manifest[row["audio"]] = row
    masks = json.loads(COVERAGE.read_text(encoding="utf-8")) if COVERAGE.is_file() else {}
    for path in sorted((SEALED / "audio").glob("*.wav")):
        meta = manifest.get(f"audio/{path.name}", {})
        yield path, meta.get("category", "unknown"), masks.get(str(path), {}).get("mask", [])


def segments(n_frames, mask, seconds):
    """Unmasked frame ranges and unmasked hours."""
    fps = protocol.FRAMES_PER_SECOND
    cuts = sorted((max(0, int(s * fps)), min(n_frames, int(np.ceil(e * fps)))) for s, e in mask)
    out, start = [], 0
    for a, b in cuts:
        if a > start:
            out.append((start, a))
        start = max(start, b)
    if start < n_frames:
        out.append((start, n_frames))
    masked = sum(max(0.0, min(e, seconds) - s) for s, e in mask)
    return out, (seconds - masked) / 3600


def count(probabilities, segs, thresholds):
    total = np.zeros(len(thresholds), dtype=np.int64)
    for a, b in segs:
        if b - a > 1:
            total += np.asarray(protocol.batched_events(probabilities[a:b][None], thresholds)[0])
    return total


def detector_mode(forward, thresholds, out):
    import soundfile as sf
    events, hours, names, cats = [], [], [], []
    for path, category, mask in sealed_files():
        wave, rate = sf.read(str(path), dtype="int16")
        probs = protocol.stream_probabilities(forward, wave)
        segs, h = segments(probs.shape[0], mask, len(wave) / rate)
        events.append(count(probs, segs, thresholds))
        hours.append(h)
        names.append(path.name)
        cats.append(category)
        print(f"{path.name} {category}: {h:.2f} h", flush=True)
    np.savez_compressed(out, thresholds=thresholds, events=np.array(events), hours=np.array(hours),
                        files=np.array(names), categories=np.array(cats))


def teacher_mode(scores, stage1, out):
    import soundfile as sf
    import torch
    import teacher_textmatch as ttm
    d = np.load(scores)
    tf_thr, tau1 = d["tf_thr"], float(d["tau1"])
    rate = protocol.SAMPLE_RATE
    before, after = int(base.VERIFY_BEFORE * rate), int(base.VERIFY_AFTER * rate)
    files = list(sealed_files())
    # stage 1 (student) triggers at tau1, outside the masks
    forward = base.student_forward(stage1)
    pieces, piece_file, hours, names, cats = [], [], [], [], []
    for k, (path, category, mask) in enumerate(files):
        wave, _ = sf.read(str(path), dtype="int16")
        probs = protocol.stream_probabilities(forward, wave)
        segs, h = segments(probs.shape[0], mask, len(wave) / rate)
        hours.append(h)
        names.append(path.name)
        cats.append(category)
        for a, b in segs:
            for frame in base.event_frames(probs[a:b], tau1):
                end = ((a + frame) // protocol.CHUNK_FRAMES + 1) * protocol.CHUNK_FRAMES * protocol.FRAME_SAMPLES
                pieces.append(wave[max(0, end - before):min(len(wave), end + after)])
                piece_file.append(k)
    del forward
    torch.cuda.empty_cache()
    print(f"stage 1: {len(pieces)} triggers", flush=True)
    # teacher, folded posteriors: alone (every threshold) and as verifier of the triggers
    teacher = base.TeacherForward()
    tf_events = []
    for path, category, mask in files:
        wave, _ = sf.read(str(path), dtype="int16")
        probs = protocol.stream_probabilities(teacher, wave)
        segs, _ = segments(probs.shape[0], mask, len(wave) / rate)
        tf_events.append(count(probs, segs, tf_thr))
        print(f"teacher folded {path.name}", flush=True)
    t_long = base.verify_windows(teacher, pieces)
    del teacher
    torch.cuda.empty_cache()
    # teacher text: cascade verification and text match alone
    text = ttm.TeacherText()
    cascade_hits = np.array([ttm.hit(t) for t in text.decode(pieces)]) if pieces else np.zeros(0, bool)
    win, hop = int(ttm.WINDOW_S * rate), int(ttm.HOP_S * rate)
    text_events = []
    for path, category, mask in files:
        wave, _ = sf.read(str(path), dtype="int16")
        starts = list(range(0, max(1, len(wave) - win + 1), hop))
        texts = []
        for i in range(0, len(starts), 2048):
            texts += text.decode([wave[s:s + win] for s in starts[i:i + 2048]])
        times = [(s + win / 2) / rate for s, t in zip(starts, texts) if ttm.hit(t)]
        times = [t for t in times if not any(a <= t <= b for a, b in mask)]
        text_events.append(ttm.merged_events(times))
        print(f"teacher text {path.name}: {text_events[-1]}", flush=True)
    np.savez_compressed(out, tf_thr=tf_thr, tf_events=np.array(tf_events), tau1=tau1,
                        t_long=np.asarray(t_long), cascade_hits=cascade_hits,
                        trigger_file=np.array(piece_file), text_events=np.array(text_events),
                        hours=np.array(hours), files=np.array(names), categories=np.array(cats))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path)
    parser.add_argument("--dscnn", type=Path)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--stage1", type=Path)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    v12.install_runtime_imports()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.teacher:
        teacher_mode(args.scores, args.stage1, args.out)
    else:
        if args.dscnn is not None:
            import kws_baseline
            forward = kws_baseline.load_forward(args.dscnn)[0]
        else:
            forward = base.student_forward(args.student)
        detector_mode(forward, np.load(args.scores)["thresholds"], args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
