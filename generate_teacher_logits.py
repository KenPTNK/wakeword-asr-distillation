#!/usr/bin/env python3
"""Parakeet-CTC-0.6B-vi logits for EVERY row of the ASR corpora (ctctknk-docs/29).

The logits used so far (``ctctknk_data/teacher_logits``) were computed on Colab
and cover 279,349 of the 462,813 rows: part of InfoRE2 and VLSP, and none of the
test/validation splits.  This recomputes all of them on the server, one output
per Parquet file, so a run can stop and resume at any file.

Format = the old one: **raw logits** (before log-softmax), float16,
[frames, 1025] (1,024 SentencePiece pieces + blank), one row per 80 ms frame.
``--verify N`` first recomputes N rows that already have old logits and
compares frame counts, argmax agreement and value error.

Output (``--out-dir``):
    logits/<dataset>/<config>/<split>/<file>.data.npy   concatenated frames
    index/<same path>.jsonl                             one line per row:
        dataset, audio_relpath, audio_row, split, teacher_file, teacher_offset,
        teacher_length, teacher_dtype, duration_seconds, text
    index/<same path>.done                              written last

    python generate_teacher_logits.py --audio-root ctctknk_data/audio_shards \
        --out-dir /media/ml4u/KINGSTON/ctctknk/teacher_logits_full --verify 64
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CLASSES = 1025
DEFAULT_MODEL = ("/home/ml4u/.cache/huggingface/hub/models--nvidia--parakeet-ctc-0.6b-Vietnamese/"
                 "snapshots/b0493142b49458810324e3db8be9e8e07b4ebc17/parakeet-ctc-0.6b-vi.nemo")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="top-level folders under --audio-root (default: all)")
    parser.add_argument("--batch-seconds", type=float, default=400.0,
                        help="padded audio seconds per forward pass")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--verify", type=int, default=0,
                        help="rows compared with the old logits before generating")
    parser.add_argument("--old-manifest", type=Path, default=None)
    parser.add_argument("--old-logits", type=Path, default=None,
                        help="teacher_logits_uncompressed/ (shard_*.data.npy)")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-done-in", nargs="*", type=Path, default=[],
                        help="other output dirs whose index/**.done also count as finished "
                             "(resume into a new --out-dir without recomputing)")
    return parser.parse_args(argv)


class Teacher:
    def __init__(self, model_path: Path, precision: str):
        import torch
        import nemo.collections.asr as nemo_asr

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nemo_asr.models.ASRModel.restore_from(str(model_path),
                                                           map_location=self.device)
        self.model.eval()
        self.dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]

    def logits(self, waves: list[np.ndarray]) -> list[np.ndarray]:
        """Raw logits per wave: preprocessor → encoder → decoder conv, no log-softmax."""
        torch = self.torch
        lengths = [len(w) for w in waves]
        batch = np.zeros((len(waves), max(lengths)), dtype=np.float32)
        for row, wave in enumerate(waves):
            batch[row, :len(wave)] = wave
        audio = torch.from_numpy(batch).to(self.device)
        sizes = torch.tensor(lengths, device=self.device)
        with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=self.dtype or torch.float32,
                enabled=self.dtype is not None):
            features, feature_lengths = self.model.preprocessor(input_signal=audio,
                                                                length=sizes)
            encoded, encoded_lengths = self.model.encoder(audio_signal=features,
                                                          length=feature_lengths)
            raw = self.model.decoder.decoder_layers(encoded).transpose(1, 2).float()
        if raw.shape[-1] != CLASSES:
            raise RuntimeError(f"decoder gives {raw.shape[-1]} classes, expected {CLASSES}")
        return [raw[row, :int(encoded_lengths[row])].cpu().numpy().astype(np.float16)
                for row in range(len(waves))]


def decode(audio_object) -> np.ndarray:
    import parakeet_v12_train as v12

    return np.asarray(v12.decode_audio_object(audio_object, SAMPLE_RATE), dtype=np.float32)


def text_column(names) -> str:
    for name in ("transcription", "text", "sentence"):
        if name in names:
            return name
    raise RuntimeError(f"no transcript column in {names}")


def batches_by_length(waves, seconds: float):
    """Index batches of similar length whose padded size stays under ``seconds``."""
    order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
    batch, longest = [], 0
    for index in order:
        grown = max(longest, len(waves[index]))
        if batch and grown * (len(batch) + 1) > seconds * SAMPLE_RATE:
            yield batch
            batch, grown = [], len(waves[index])
        batch.append(index)
        longest = grown
    if batch:
        yield batch


def locate_row(reader, row: int) -> tuple[int, int]:
    start = 0
    for group in range(reader.metadata.num_row_groups):
        size = reader.metadata.row_group(group).num_rows
        if row < start + size:
            return group, row - start
        start += size
    raise IndexError(row)


CHUNK_ROWS = 2000


def dataset_of(relative: Path) -> str:
    """Top folder, minus a Hugging Face ``owner__`` prefix (NhutP__VSV-1100 -> VSV-1100)."""
    return relative.parts[0].split("__")[-1]


def split_of(relative: Path) -> str:
    """The split folder, or for the HF layout ``data/<split>-NNNNN-of-NNNNN.parquet``
    the file-name prefix."""
    if relative.parts[-2] == "data":
        return relative.name.split("-")[0]
    return relative.parts[-2]


def chunk_row_groups(reader) -> list[list[int]]:
    chunks, current, size = [], [], 0
    for group in range(reader.metadata.num_row_groups):
        current.append(group)
        size += reader.metadata.row_group(group).num_rows
        if size >= CHUNK_ROWS:
            chunks.append(current)
            current, size = [], 0
    if current:
        chunks.append(current)
    return chunks


def process_file(teacher: Teacher, path: Path, audio_root: Path, out_dir: Path,
                 seconds: float) -> tuple[int, float]:
    """Logits for one Parquet file, read and written in chunks of ~CHUNK_ROWS rows
    (a VSV-1100 file holds ~40 h of audio), one ``.partNNN.data.npy`` per chunk."""
    import pyarrow.parquet as pq

    relative = path.relative_to(audio_root)
    stem = relative.with_suffix("")
    index_path = out_dir / "index" / relative.with_suffix(".jsonl")
    done_path = out_dir / "index" / relative.with_suffix(".done")
    if done_path.is_file():
        return 0, 0.0
    reader = pq.ParquetFile(str(path))
    column = text_column(reader.schema_arrow.names)
    records, row_base, hours = [], 0, 0.0
    for part, chunk in enumerate(chunk_row_groups(reader)):
        table = reader.read_row_groups(chunk, columns=["audio", column])
        waves = [decode(value) for value in table.column("audio").to_pylist()]
        texts = [str(value or "") for value in table.column(column).to_pylist()]
        del table
        outputs: list[np.ndarray | None] = [None] * len(waves)
        for batch in batches_by_length(waves, seconds):
            for index, value in zip(batch, teacher.logits([waves[i] for i in batch])):
                outputs[index] = value
        offsets = np.cumsum([0] + [len(o) for o in outputs])
        logits_path = out_dir / "logits" / f"{stem}.part{part:03d}.data.npy"
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = logits_path.parent / (logits_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, np.concatenate(outputs, axis=0))
        temporary.replace(logits_path)
        teacher_file = str(logits_path.relative_to(out_dir)).replace("\\", "/")
        for row, (wave, text) in enumerate(zip(waves, texts)):
            records.append({
                "dataset": dataset_of(relative),
                "audio_relpath": str(relative).replace("\\", "/"),
                "audio_row": row_base + row, "split": split_of(relative),
                "teacher_file": teacher_file, "teacher_offset": int(offsets[row]),
                "teacher_length": int(len(outputs[row])), "teacher_dtype": "float16",
                "duration_seconds": round(len(wave) / SAMPLE_RATE, 3), "text": text})
        row_base += len(waves)
        hours += sum(len(w) for w in waves) / SAMPLE_RATE / 3600
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    done_path.write_text(time.strftime("%F %T") + "\n")
    return row_base, hours


def verify(teacher: Teacher, args) -> bool:
    """Recompute rows that have old logits and compare."""
    import pyarrow.parquet as pq

    rows = [json.loads(line) for line in args.old_manifest.open(encoding="utf-8")]
    rng = np.random.default_rng(0)
    picked = [rows[int(i)] for i in rng.choice(len(rows), size=args.verify, replace=False)]
    # The Colab logits carry one padding frame past the model's valid length on
    # some rows (InfoRE2), and 44.1 kHz sources were resampled differently there;
    # both are tolerated: at most one frame of difference, argmax >= 99.5 %.
    frames_equal = agree = total = 0
    worst = 0.0
    for row in picked:
        reader = pq.ParquetFile(str(args.audio_root / row["audio_relpath"]))
        group, offset = locate_row(reader, int(row["audio_row"]))
        audio = reader.read_row_group(group, columns=["audio"]).column("audio")[offset].as_py()
        new = teacher.logits([decode(audio)])[0].astype(np.float32)
        old_all = np.load(args.old_logits / row["teacher_shard"].replace(".npz", ".data.npy"),
                          mmap_mode="r")
        old = np.asarray(old_all[row["teacher_offset"]:row["teacher_offset"]
                                 + row["teacher_length"]], dtype=np.float32)
        frames_equal += int(abs(len(new) - len(old)) <= 1)
        size = min(len(new), len(old))
        agree += int((new[:size].argmax(-1) == old[:size].argmax(-1)).sum())
        total += size
        worst = max(worst, float(np.abs(new[:size] - old[:size]).max()))
    rate = agree / max(total, 1)
    print(f"[verify] {len(picked)} rows: frame count within 1 {frames_equal}/{len(picked)}, "
          f"argmax agreement {rate:.5f}, max |Δlogit| {worst:.3f}", flush=True)
    return frames_equal == len(picked) and rate > 0.995


def main(argv=None) -> int:
    args = parse_args(argv)
    import parakeet_v12_train as v12

    v12.install_runtime_imports()
    teacher = Teacher(args.model, args.precision)
    if args.verify:
        if not verify(teacher, args):
            print("[verify] FAILED: new logits do not reproduce the old ones", flush=True)
            return 1
        if args.verify_only:
            return 0
    files = sorted(Path(p) for p in glob.glob(str(args.audio_root / "**" / "*.parquet"),
                                              recursive=True))
    if args.datasets:
        files = [f for f in files if f.relative_to(args.audio_root).parts[0] in args.datasets]
    finished_elsewhere = [f for f in files if any(
        (other / "index" / f.relative_to(args.audio_root).with_suffix(".done")).is_file()
        for other in args.skip_done_in)]
    if finished_elsewhere:
        print(f"skipping {len(finished_elsewhere)} files finished in {args.skip_done_in}",
              flush=True)
        files = [f for f in files if f not in finished_elsewhere]
    started, rows_done, hours_done = time.time(), 0, 0.0
    for number, path in enumerate(files, 1):
        rows, hours = process_file(teacher, path, args.audio_root, args.out_dir,
                                   args.batch_seconds)
        rows_done, hours_done = rows_done + rows, hours_done + hours
        elapsed = time.time() - started
        print(f"[{number}/{len(files)}] {path.relative_to(args.audio_root)}: "
              f"{rows} rows, {hours:.2f} h | total {rows_done:,} rows {hours_done:.1f} h "
              f"in {elapsed / 60:.1f} min", flush=True)
    print("all files done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
