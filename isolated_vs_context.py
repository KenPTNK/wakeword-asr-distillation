#!/usr/bin/env python3
"""Isolated-clip vs in-context scoring on the frozen test protocol (ICIIT revision).

For one detector, with thresholds fixed by FA/h on the continuous hk15 stream:
  * in-context FRR / near-miss FA  -- positives embedded after 20.48 s of hk15 (the protocol);
  * isolated FRR / near-miss FA     -- the same clips scored alone from a cold (reset) encoder,
                                       as clip benchmarks do (clip + 0.64 s of silence);
  * cold-start FA per restart       -- 2,000 random 5.12 s hk15 segments, each scored from a
                                       cold encoder (what a restart/wake-up of the device sees).
Writes one JSON per detector.  Test speakers only; nothing in the frozen protocol changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_baselines as base
import evaluate_wake_suite as suite
import incontext_protocol as ic
import parakeet_v12_train as v12
import streaming_protocol as protocol

RUN_DIR = Path("checkpoints/student_L5_H192_M64_v17_t6_lr1e-4_h0")
POS = Path("logs/v14_2/hk15_eval/positives_pos8.jsonl")
AUDIO = Path("multi_phone_threshold_analysis/hk15_24h/audio_16k_mono.wav")
COLD_SEGMENTS = 2000
COLD_SECONDS = 5.12
SEED = 20260926


def cold_streams(clips):
    tail = np.zeros(int(ic.TAIL_SECONDS * protocol.SAMPLE_RATE), dtype=np.int16)
    streams = [np.concatenate([c.astype(np.int16), tail]) for c in clips]
    return streams, [(0, len(c)) for c in clips]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path)
    parser.add_argument("--dscnn", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/v17/paper_final/isolated"))
    args = parser.parse_args()
    v12.install_runtime_imports()
    import soundfile as sf
    if args.dscnn is not None:
        import kws_baseline
        forward = kws_baseline.load_forward(args.dscnn)[0]
    else:
        forward = base.student_forward(args.student)

    pos_waves, _ = suite.ev.load_dev_positives(POS)
    labels = suite.positive_paths(POS)
    pos = [w for w, lab in zip(pos_waves, labels) if ic.role_of(lab) == "test"]
    neg_waves, neg_labels = suite.load_hard_negatives(RUN_DIR / "unified_manifest.jsonl")
    neg = [w for w, lab in zip(neg_waves, neg_labels) if ic.role_of(lab) == "test"]
    long_wave, rate = sf.read(str(AUDIO), dtype="int16")
    hours = len(long_wave) / rate / 3600

    ctx_pos = ic.window_best(forward, *base.test_streams(pos, [long_wave], 1))
    ctx_neg = ic.window_best(forward, *base.test_streams(neg, [long_wave], 2))
    iso_pos = ic.window_best(forward, *cold_streams(pos))
    iso_neg = ic.window_best(forward, *cold_streams(neg))
    rng = np.random.default_rng(SEED)
    seg = int(COLD_SECONDS * rate)
    starts = rng.integers(0, len(long_wave) - seg, COLD_SEGMENTS)
    cold = [long_wave[s:s + seg] for s in starts]
    cold_best = ic.window_best(forward, cold, [(0, seg)] * len(cold))

    finite = np.concatenate([ctx_pos, iso_pos])
    finite = finite[np.isfinite(finite)]
    thresholds = np.linspace(float(finite.min()), float(finite.max()), 256)
    _, events = base.long_events(forward, long_wave, thresholds)
    per_hour = np.asarray(events) / hours
    rows = {}
    for limit in base.FA_TARGETS:
        allowed = per_hour <= limit
        if not allowed.any():
            rows[f"{limit:g}"] = None
            continue
        # operating threshold = the in-context protocol's choice (max in-context recall)
        recalls = np.array([(ctx_pos >= t).mean() for t in thresholds])
        t = float(thresholds[int(np.flatnonzero(allowed)[np.argmax(recalls[allowed])])])
        rows[f"{limit:g}"] = {
            "threshold": t, "fa_per_hour": float(per_hour[thresholds == t][0]),
            "fa_events": int(np.asarray(events)[thresholds == t][0]),
            "frr_in_context": float((ctx_pos < t).mean()),
            "frr_isolated": float((iso_pos < t).mean()),
            "near_miss_in_context": float((ctx_neg >= t).mean()),
            "near_miss_isolated": float((iso_neg >= t).mean()),
            "cold_start_fa_per_restart": float((cold_best >= t).mean()),
        }
    report = {"label": args.label, "checkpoint": str(args.student or args.dscnn),
              "n_pos": len(pos), "n_neg": len(neg), "cold_segments": COLD_SEGMENTS,
              "cold_seconds": COLD_SECONDS, "operating_points": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.label}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(rows, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
