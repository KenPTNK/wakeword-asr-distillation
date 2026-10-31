#!/usr/bin/env python3
"""Compact teacher logits to what the KD loss reads: top-64 + logsumexp (ctctknk-docs/30).

``kd_kl_sum`` (V8/V11) and ``kd_kl_mean`` (V15) only use, per frame, the 64
largest teacher logits, their token ids, and the log-sum-exp over all 1,025
classes (for the "everything else" bucket).  Storing exactly that shrinks the
full fp16 logits (2,050 B/frame, 173 GB on two exFAT USB disks with slow random
reads) to ~260 B/frame (~22 GB) on the internal NVMe (ext4).

For every ``<name>.data.npy`` under each ``--src`` root, writes under ``--dst``
(same relative path):

    <name>.top64v.npy   float16 [frames, 64]  values, sorted descending (col 0 = argmax)
    <name>.top64i.npy   int16   [frames, 64]  token ids
    <name>.lse.npy      float32 [frames]      logsumexp over all 1,025 logits
    <name>.lseT2.npy    float32 [frames]      logsumexp over logits / 2 (KD temperature)
    <name>.done

    python compact_teacher_logits.py --src /media/ml4u/KINGSTON/ctctknk/teacher_logits_full \
        "/media/ml4u/Extreme SSD/ctctknk/teacher_logits_part2" \
        --dst /home/ml4u/datasets/ctctknk_3a/teacher_top64
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

TOPK = 64
CLASSES = 1025
BLOCK_FRAMES = 200_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs="+", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--temperatures", nargs="*", type=float, default=[2.0])
    # Reading exFAT on the USB disks is bound by per-file kernel work, not bandwidth:
    # 4 processes on different files read 56 MB/s where one reads 20 MB/s (22/09).
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--block-frames", type=int, default=BLOCK_FRAMES)
    return parser.parse_args(argv)


def save(path: Path, array: np.ndarray) -> None:
    temporary = path.parent / (path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def compact(source: Path, stem: Path, temperatures, device,
            block_frames: int = BLOCK_FRAMES) -> int:
    logits = np.load(source, mmap_mode="r")
    if logits.ndim != 2 or logits.shape[1] != CLASSES:
        raise RuntimeError(f"{source}: shape {logits.shape}")
    frames = logits.shape[0]
    values = np.empty((frames, TOPK), dtype=np.float16)
    indices = np.empty((frames, TOPK), dtype=np.int16)
    lse = np.empty(frames, dtype=np.float32)
    tempered = {t: np.empty(frames, dtype=np.float32) for t in temperatures}
    for start in range(0, frames, block_frames):
        block = torch.from_numpy(np.asarray(logits[start:start + block_frames]))
        block = block.to(device).float()
        top = torch.topk(block, k=TOPK, dim=-1, sorted=True)
        end = start + block.shape[0]
        values[start:end] = top.values.half().cpu().numpy()
        indices[start:end] = top.indices.to(torch.int16).cpu().numpy()
        lse[start:end] = torch.logsumexp(block, dim=-1).cpu().numpy()
        for t, out in tempered.items():
            out[start:end] = torch.logsumexp(block / t, dim=-1).cpu().numpy()
    stem.parent.mkdir(parents=True, exist_ok=True)
    save(Path(f"{stem}.top64v.npy"), values)
    save(Path(f"{stem}.top64i.npy"), indices)
    save(Path(f"{stem}.lse.npy"), lse)
    for t, out in tempered.items():
        save(Path(f"{stem}.lseT{t:g}.npy"), out)
    Path(f"{stem}.done").write_text(time.strftime("%F %T") + "\n")
    return frames


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = []
    for root in args.src:
        for path in sorted(glob.glob(str(root / "logits" / "**" / "*.data.npy"), recursive=True)):
            source = Path(path)
            relative = source.relative_to(root / "logits")
            stem = args.dst / relative.parent / relative.name.replace(".data.npy", "")
            jobs.append((source, stem))
    started, done_frames = time.time(), 0
    for number, (source, stem) in enumerate(jobs, 1):
        if (number - 1) % args.num_shards != args.shard or Path(f"{stem}.done").is_file():
            continue
        frames = compact(source, stem, args.temperatures, device, args.block_frames)
        done_frames += frames
        print(f"[{number}/{len(jobs)}] {stem.relative_to(args.dst)}: {frames:,} frames | "
              f"{done_frames / 12.5 / 3600:.1f} h in {(time.time() - started) / 60:.1f} min",
              flush=True)
    print("all files done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
