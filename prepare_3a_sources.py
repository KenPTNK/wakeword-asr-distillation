#!/usr/bin/env python3
"""Convert the non-Parquet 3a sources (ctctknk-docs/30) to the shard layout.

Output: <out>/<Dataset>/<config>/<split>/NNNN.parquet with columns
``audio`` {bytes, path} (original WAV bytes, original sample rate),
``transcription`` and ``id`` — the layout ``generate_teacher_logits.py`` reads.

* GigaSpeech2 vi dev/test: ``<split>.tsv`` (id, text) + ``<split>.tar.gz`` (WAVs).
* VIVOS test only (blind test for Parakeet): ``vivos/test/prompts.txt`` + waves.
* MS-SNSD noise_train / noise_test: every WAV cut into 10 s pieces with an empty
  transcript (the teacher drops clips > 20 s; Parakeet trains on MS-SNSD as
  noise-only utterances).

    python prepare_3a_sources.py --raw /media/ml4u/KINGSTON/ctctknk/raw_datasets \
        --out /media/ml4u/KINGSTON/ctctknk/audio_shards_3a
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROWS_PER_SHARD = 2000
NOISE_PIECE_SECONDS = 10.0
MIN_NOISE_PIECE_SECONDS = 1.0


class ShardWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.index = 0
        self.total = 0

    def add(self, audio: bytes, name: str, text: str, key: str) -> None:
        self.rows.append({"audio": {"bytes": audio, "path": name},
                          "transcription": text, "id": key})
        if len(self.rows) >= ROWS_PER_SHARD:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows)
        target = self.directory / f"{self.index:04d}.parquet"
        temporary = target.parent / (target.name + ".tmp")
        pq.write_table(table, temporary, row_group_size=100)
        temporary.replace(target)
        self.total += len(self.rows)
        self.index += 1
        self.rows = []


def gigaspeech2(raw: Path, out: Path) -> None:
    base = raw / "speechcolab__gigaspeech2" / "data" / "vi"
    for split in ("dev", "test"):
        texts = {}
        for line in (base / f"{split}.tsv").read_text(encoding="utf-8").splitlines():
            key, _, text = line.partition("\t")
            texts[key.strip()] = text.strip()
        writer = ShardWriter(out / "GigaSpeech2" / "vi" / split)
        missing = 0
        with tarfile.open(base / f"{split}.tar.gz", "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                key = Path(member.name).stem
                if key not in texts:
                    missing += 1
                    continue
                writer.add(archive.extractfile(member).read(), member.name, texts.pop(key), key)
        writer.flush()
        print(f"GigaSpeech2/{split}: {writer.total} rows, {missing} WAVs without text, "
              f"{len(texts)} texts without WAV", flush=True)


def vivos(raw: Path, out: Path) -> None:
    archive_path = raw / "AILAB-VNUHCM__vivos" / "data" / "vivos.tar.gz"
    texts: dict[str, str] = {}
    waves: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or "/test/" not in member.name:
                continue
            if member.name.endswith("prompts.txt"):
                for line in archive.extractfile(member).read().decode("utf-8").splitlines():
                    key, _, text = line.partition(" ")
                    texts[key.strip()] = text.strip()
            elif member.name.endswith(".wav"):
                waves[Path(member.name).stem] = archive.extractfile(member).read()
    writer = ShardWriter(out / "VIVOS" / "default" / "test")
    for key in sorted(waves):
        if key in texts:
            writer.add(waves[key], f"{key}.wav", texts[key], key)
    writer.flush()
    print(f"VIVOS/test: {writer.total} rows ({len(waves)} WAVs, {len(texts)} prompts)",
          flush=True)


def ms_snsd(raw: Path, out: Path) -> None:
    for split in ("noise_train", "noise_test"):
        writer = ShardWriter(out / "MS-SNSD" / "noise" / split)
        seconds = 0.0
        for path in sorted((raw / "MS-SNSD" / split).glob("*.wav")):
            data, rate = sf.read(str(path), dtype="int16", always_2d=False)
            piece = int(NOISE_PIECE_SECONDS * rate)
            for number, start in enumerate(range(0, len(data), piece)):
                chunk = data[start:start + piece]
                if len(chunk) < MIN_NOISE_PIECE_SECONDS * rate:
                    continue
                buffer = io.BytesIO()
                sf.write(buffer, chunk, rate, format="WAV", subtype="PCM_16")
                writer.add(buffer.getvalue(), f"{path.stem}_{number:03d}.wav", "",
                           f"{path.stem}_{number:03d}")
                seconds += len(chunk) / rate
        writer.flush()
        print(f"MS-SNSD/{split}: {writer.total} pieces, {seconds / 3600:.2f} h", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=["gigaspeech2", "vivos", "ms_snsd"])
    args = parser.parse_args()
    steps = {"gigaspeech2": gigaspeech2, "vivos": vivos, "ms_snsd": ms_snsd}
    for name in args.only:
        steps[name](args.raw, args.out)
    print("conversion done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
