#!/usr/bin/env python3
"""Put every train audio shard of ``splits_3a.jsonl`` on the NVMe (ctctknk-docs/31).

Training read VSV-1100 (130 GB) and FOSD from KINGSTON (exFAT over USB, ~22 MB/s
when another job reads it too) and the rest from Samsung_T52, so the GPU sat
idle waiting for audio.  The NVMe has ~128 GB free, less than the 202 GB of train
audio, hence:

* InfoRE2: already on the NVMe (``ctctknk_data_nvme``, 111/111 files same size).
* VSV-1100: WAV re-encoded as FLAC (lossless) into a Parquet file with the same
  row groups and row order, so ``audio_row`` stays valid; checked sample-exact.
* Everything else: copied byte for byte.

Then ``--write-manifest`` writes ``splits_3a_nvme.jsonl``: the same rows with
``audio_path`` pointing at the NVMe copy wherever one exists (the original
manifest is left untouched).

    python prepare_nvme_audio_3a.py --copy --shard 0 --num-shards 4
    python prepare_nvme_audio_3a.py --write-manifest
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

MANIFEST = Path("/home/ml4u/datasets/ctctknk_3a/splits_3a.jsonl")
NVME_MANIFEST = Path("/home/ml4u/datasets/ctctknk_3a/splits_3a_nvme.jsonl")
NVME_ROOT = Path("/home/ml4u/datasets/ctctknk_3a/audio")
INFORE2_SOURCE = "/media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk_data/audio_shards/InfoRE2"
INFORE2_NVME = "/home/ml4u/datasets/ctctknk_data_nvme/audio_shards/InfoRE2"
FLAC_DATASETS = {"VSV-1100"}
MIN_FREE_GB = 10.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args(argv)


def nvme_path(source: str) -> str:
    """``/media/ml4u/<disk>/rest`` -> ``NVME_ROOT/<disk>/rest``; InfoRE2 -> old copy."""
    if source.startswith(INFORE2_SOURCE):
        return INFORE2_NVME + source[len(INFORE2_SOURCE):]
    parts = Path(source).parts
    if parts[:3] != ("/", "media", "ml4u"):
        raise ValueError(f"unexpected audio path {source}")
    return str(NVME_ROOT.joinpath(*parts[3:]))


def train_files() -> dict[str, str]:
    """audio_path -> dataset, for every file holding at least one train row."""
    files: dict[str, str] = {}
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["role"] == "train":
                files[row["audio_path"]] = row["dataset"]
    return files


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def wav_to_flac(payload: bytes) -> bytes:
    import soundfile as sf

    data, rate = sf.read(io.BytesIO(payload), dtype="int16", always_2d=False)
    out = io.BytesIO()
    sf.write(out, data, rate, format="FLAC", subtype="PCM_16")
    return out.getvalue()


def same_audio(wav: bytes, flac: bytes) -> bool:
    import soundfile as sf

    a, ra = sf.read(io.BytesIO(wav), dtype="int16", always_2d=False)
    b, rb = sf.read(io.BytesIO(flac), dtype="int16", always_2d=False)
    return ra == rb and np.array_equal(a, b)


def reencode_flac(source: str, target: str) -> None:
    """Row group by row group, same sizes and order; first row of each group checked."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(source)
    partial = target + ".partial"
    writer = None
    try:
        for group in range(reader.metadata.num_row_groups):
            table = reader.read_row_group(group)
            audio = table.column("audio").combine_chunks()
            payloads = audio.field("bytes").to_pylist()
            flacs = [wav_to_flac(p) for p in payloads]
            if not same_audio(payloads[0], flacs[0]):
                raise RuntimeError(f"{source} group {group}: FLAC differs from WAV")
            new_audio = pa.StructArray.from_arrays(
                [pa.array(flacs, type=pa.binary()), audio.field("path")], names=["bytes", "path"])
            table = table.set_column(table.schema.get_field_index("audio"), "audio", new_audio)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="NONE")
            writer.write_table(table, row_group_size=table.num_rows)
    finally:
        if writer is not None:
            writer.close()
    check = pq.ParquetFile(partial).metadata
    groups = [reader.metadata.row_group(i).num_rows for i in range(reader.metadata.num_row_groups)]
    if [check.row_group(i).num_rows for i in range(check.num_row_groups)] != groups:
        raise RuntimeError(f"{partial}: row groups differ from {source}")
    os.replace(partial, target)


def copy_file(source: str, target: str) -> None:
    partial = target + ".partial"
    shutil.copyfile(source, partial)
    if os.path.getsize(partial) != os.path.getsize(source):
        raise RuntimeError(f"{partial}: size differs from {source}")
    os.replace(partial, target)


def copy(options) -> int:
    files = sorted(train_files().items())[options.shard::options.num_shards]
    for number, (source, dataset) in enumerate(files, 1):
        target = nvme_path(source)
        if os.path.isfile(target):
            continue
        if free_gb(NVME_ROOT.parent) < MIN_FREE_GB:
            raise SystemExit(f"NVMe below {MIN_FREE_GB} GB free; stopping")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        if dataset in FLAC_DATASETS:
            reencode_flac(source, target)
        else:
            copy_file(source, target)
        print(f"[{options.shard}] {number}/{len(files)} {dataset} "
              f"{os.path.getsize(source) / 1e9:.2f} -> {os.path.getsize(target) / 1e9:.2f} GB "
              f"{target}", flush=True)
    return 0


def write_manifest() -> int:
    missing = sorted(p for p in train_files() if not os.path.isfile(nvme_path(p)))
    if missing:
        raise SystemExit(f"{len(missing)} train files not on the NVMe yet, e.g. {missing[0]}")
    moved = 0
    with MANIFEST.open(encoding="utf-8") as src, \
            NVME_MANIFEST.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            target = nvme_path(row["audio_path"])
            if os.path.isfile(target):
                row["audio_path"] = target
                moved += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {NVME_MANIFEST}: {moved:,} rows point at the NVMe", flush=True)
    return 0


def main(argv=None) -> int:
    options = parse_args(argv)
    NVME_ROOT.mkdir(parents=True, exist_ok=True)
    if options.copy:
        return copy(options)
    if options.write_manifest:
        return write_manifest()
    print("nothing to do: pass --copy or --write-manifest", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
