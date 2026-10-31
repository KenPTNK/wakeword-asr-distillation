#!/usr/bin/env python3
"""Per-clip scores on the frozen test protocol, for clustered statistics (ICIIT revision).

Re-runs exactly the scoring of ``evaluate_baselines.single_detector`` (same embedded test
streams, same seeds, same hk15 FA stream) and saves, instead of only the FRR table:
  pos_best / pos_label / pos_family   -- best in-window score per test positive
  neg_best / neg_label                -- best in-window score per test near miss
  thresholds / events                 -- hk15 FA events per threshold (256-point grid + extras)
so FRR at any threshold, per speaker, per original utterance and with a cluster bootstrap
can be computed offline.  Nothing in the frozen protocol is modified.

usage: python dump_scores.py --student CKPT --out X.npz [--extra-thresholds -4.0402 ...]
       python dump_scores.py --dscnn logs/v17/kws_baseline/dscnn_w64/ckpt_016000.pt --out Y.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import evaluate_baselines as base
import evaluate_wake_suite as suite
import incontext_protocol as ic
import parakeet_v12_train as v12

RUN_DIR = Path("checkpoints/student_L5_H192_M64_v17_t6_lr1e-4_h0")
POS = Path("logs/v14_2/hk15_eval/positives_pos8.jsonl")
AUDIO = Path("multi_phone_threshold_analysis/hk15_24h/audio_16k_mono.wav")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path)
    parser.add_argument("--dscnn", type=Path)
    parser.add_argument("--fake-int8", action="store_true")
    parser.add_argument("--thresholds", type=int, default=256)
    parser.add_argument("--extra-thresholds", type=float, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    v12.install_runtime_imports()
    import soundfile as sf
    if args.dscnn is not None:
        import kws_baseline
        forward = kws_baseline.load_forward(args.dscnn)[0]
    else:
        forward = base.student_forward(args.student, args.fake_int8)

    pos_waves, _ = suite.ev.load_dev_positives(POS)
    labels = suite.positive_paths(POS)
    keep = [i for i, lab in enumerate(labels) if ic.role_of(lab) == "test"]
    pos = [pos_waves[i] for i in keep]
    pos_label = [labels[i] for i in keep]
    families = base.positive_augmentations(POS)
    pos_family = [families[i] for i in keep]
    neg_waves, neg_labels = suite.load_hard_negatives(RUN_DIR / "unified_manifest.jsonl")
    nkeep = [i for i, lab in enumerate(neg_labels) if ic.role_of(lab) == "test"]
    neg = [neg_waves[i] for i in nkeep]
    neg_label = [neg_labels[i] for i in nkeep]
    long_wave, rate = sf.read(str(AUDIO), dtype="int16")
    hours = len(long_wave) / rate / 3600

    pos_streams, pos_windows = base.test_streams(pos, [long_wave], 1)
    neg_streams, neg_windows = base.test_streams(neg, [long_wave], 2)
    pos_best = ic.window_best(forward, pos_streams, pos_windows)
    neg_best = ic.window_best(forward, neg_streams, neg_windows)
    finite = pos_best[np.isfinite(pos_best)]
    grid = np.linspace(float(finite.min()), float(finite.max()), args.thresholds)
    thresholds = np.unique(np.concatenate([grid, np.asarray(args.extra_thresholds, float)]))
    _, events = base.long_events(forward, long_wave, thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pos_best=pos_best, pos_label=np.array(pos_label),
                        pos_family=np.array(pos_family), neg_best=neg_best,
                        neg_label=np.array(neg_label), thresholds=thresholds,
                        events=np.asarray(events), hours=hours)
    table = base.frr_table(pos_best, neg_best, thresholds, np.asarray(events) / hours,
                           np.asarray(events))
    print({k: (None if v is None else round(v["frr"], 4)) for k, v in table.items()}, flush=True)
    print(f"wrote {args.out}: {len(pos_best)} positives, {len(neg_best)} near misses, "
          f"{len(thresholds)} thresholds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
