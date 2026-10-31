#!/usr/bin/env python3
"""Augment speakers 048-054 exactly like the original test speakers (ICIIT v4).

The original test clips carry 20 variants per recording (team recipe; parameters confirmed by
measuring every family against its original on the held-out speakers, check_aug.py): orig; pitch
+2/-2/+4/-4 semitones; time stretch 0.8/0.9/1.1/1.2x; WHAM! noise at 12/10/5 dB SNR (full-clip
power); overlap with another random recording of the corpus, both at half amplitude (measured
target gain 0.50); volume x0.5; and pitch +2 / stretch 1.1 / pitch -2 / stretch 0.9 each plus
noise at 8 dB.  No reverberation (the recipe has none).  Noise comes from the WHAM! test split.
Overlap speech comes from original training-speaker recordings WITHOUT the wake phrase: an
interferer containing the phrase would turn a near miss into a positive (first attempt: near-miss
overlap clips were accepted 45 % of the time vs 11 % in the original test data).

Positives: transcripts containing the phrase (parakeet_v12_train.contains_word_sequence).  Near
misses: other recordings whose transcript yields a non-empty partial target, i.e. the
``is_hard_negative`` rule of the frozen protocol; the rest are not test clips.

usage: python augment_newspk.py --out /home/ml4u/datasets/wakeword_newspk_048_054_aug
       python augment_newspk.py --all --out /home/ml4u/datasets/wakeword_newspk_048_054_aug_all
       (--all keeps every recording, also negatives with an empty target: training data, v5 split)
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import zlib
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

import parakeet_v12_train as v12

RAW = Path("/home/ml4u/datasets/wakeword_newspk_048_054_raw")
WHAM = Path("/media/ml4u/Samsung_T52/wham_noise/tt")
SHARDS = "/home/ml4u/datasets/wakeword_v12_16k_sharded/"
MANIFEST = Path("checkpoints/student_L5_H192_M64_v17_t6_lr1e-4_h0/unified_manifest.jsonl")
TOKENIZER = Path("checkpoints/parakeet_teacher_tokenizer.model")
SR = 16000
OVERLAP_POOL = 300


def add_noise(y, noise, snr_db, rng):
    if len(noise) < len(y):
        noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(y) + 1))
    n = noise[start:start + len(y)]
    scale = np.sqrt(np.mean(y ** 2) / max(np.mean(n ** 2), 1e-12) / 10 ** (snr_db / 10))
    return y + scale * n


def overlap(y, other, rng):
    if len(other) >= len(y):
        start = int(rng.integers(0, len(other) - len(y) + 1))
        o = other[start:start + len(y)]
    else:
        o = np.zeros_like(y)
        start = int(rng.integers(0, len(y) - len(other) + 1))
        o[start:start + len(other)] = other
    return 0.5 * y + 0.5 * o


def variants(y, rng, noise_files, pool):
    wham = lambda: sf.read(noise_files[int(rng.integers(len(noise_files)))], dtype="float64")[0][:, 0]
    other = lambda: pool[int(rng.integers(len(pool)))]
    pitch = lambda k: librosa.effects.pitch_shift(y, sr=SR, n_steps=k)
    speed = lambda r: librosa.effects.time_stretch(y, rate=r)
    return {
        "orig": y,
        "pitch_p2": pitch(2), "pitch_n2": pitch(-2), "pitch_p4": pitch(4), "pitch_n4": pitch(-4),
        "speed_08": speed(0.8), "speed_09": speed(0.9), "speed_11": speed(1.1), "speed_12": speed(1.2),
        "noise_light": add_noise(y, wham(), 12, rng), "noise_med": add_noise(y, wham(), 10, rng),
        "noise_heavy": add_noise(y, wham(), 5, rng),
        "overlap1": overlap(y, other(), rng), "overlap2": overlap(y, other(), rng),
        "overlap3": overlap(y, other(), rng),
        "vol_low": 0.5 * y,
        "pitch_noise": add_noise(pitch(2), wham(), 8, rng),
        "speed_noise": add_noise(speed(1.1), wham(), 8, rng),
        "pitch_n2_noise": add_noise(pitch(-2), wham(), 8, rng),
        "speed_09_noise": add_noise(speed(0.9), wham(), 8, rng),
    }


def overlap_pool(rng):
    rows = [json.loads(line) for line in MANIFEST.open(encoding="utf-8")]
    rows = [r for r in rows if r.get("source_dataset") == "WakewordV12" and r["split"] == "train"
            and r["augmentation"] == "orig" and not r["is_positive"]]
    files, pool = {}, []
    for i in rng.choice(len(rows), OVERLAP_POOL, replace=False):
        r = rows[int(i)]
        path = r["audio_path"] if r["audio_path"].startswith("/") else SHARDS + r["audio_path"]
        f = files.setdefault(path, pq.ParquetFile(path))
        group, offset = divmod(int(r["audio_row"]), 64)
        blob = f.read_row_group(group, columns=["audio"]).column(0)[offset].as_py()
        pool.append(sf.read(io.BytesIO(blob), dtype="float64")[0])
    return pool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="also negatives with an empty target")
    args = parser.parse_args()
    v12.install_runtime_imports()
    tokenizer, _, old_to_reduced = v12.build_reduced_tokenizer(TOKENIZER)
    noise_files = sorted(glob.glob(str(WHAM / "*.wav")))
    pool = overlap_pool(np.random.default_rng(20260927))
    counts = {"positive": 0, "near_miss": 0, "skipped": 0}
    for path in sorted(RAW.rglob("*.wav")):
        text = path.stem.split("_", 1)[1].rsplit("_", 1)[0]
        positive = v12.contains_word_sequence(v12.normalize_text(text))
        partial = v12.transcript_target_ids(text, tokenizer, old_to_reduced)
        if not positive and not partial and not args.all:
            counts["skipped"] += 1
            continue
        counts["positive" if positive else "near_miss"] += 1
        y, rate = sf.read(str(path), dtype="float64")
        assert rate == SR and y.ndim == 1, path
        rng = np.random.default_rng(zlib.crc32(str(path.relative_to(RAW)).encode()))
        out_dir = args.out / path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for family, wave in variants(y, rng, noise_files, pool).items():
            pcm = np.clip(np.round(wave * 32767.0), -32768, 32767).astype(np.int16)
            sf.write(str(out_dir / f"{path.stem}_{family}.wav"), pcm, SR, subtype="PCM_16")
    print(json.dumps(counts), flush=True)
    (args.out / "augment_counts.json").write_text(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
