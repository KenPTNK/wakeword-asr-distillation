#!/usr/bin/env python3
"""Prepare V14 extra negatives as mono-16k PCM16 Parquet shards on fast storage.

Why pre-shard: the source corpora live on an exFAT USB drive, and ViMD / part
of VLSP are 44.1-48 kHz stereo.  Read raw, every training epoch would decode,
down-mix and ``soxr_hq``-resample each sample on the CPU.  Doing it once here
leaves training with a plain 16 kHz WAV decode.

Sources
-------
* LSVSC, VLSP, ViMD : rows of ``manifests/aligned_manifest.jsonl`` (phase-1
  aligned train rows) -- none of them was used by V11 or V12.2.
* VLSP (unused)     : ``unused_non_infore2_candidates_eligible.jsonl``
  (39,315 VLSP-train rows never aligned; hash-deduplicated by the earlier audit).
* MUSAN             : ``music/`` and ``noise/`` only, cut into fixed segments.
  No speech, so the empty CTC target is correct for them.

Output layout (``--output-root``)::

    shards/<SOURCE>-<NNNNN>.parquet   audio: binary (16 kHz mono PCM16 WAV), row groups of 64
    parts/<SOURCE>-<NNNNN>.jsonl      one record per shard row (written last = unit done)
    v14_negatives_16k_manifest.jsonl  concatenation of every part, in unit order
    summary.json

A unit (one source Parquet file, or one block of MUSAN files) is skipped when
its part file exists, so the job is resumable.  Source data is never modified.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

TARGET_SR = 16_000
ROW_GROUP_SIZE = 64
MIN_SECONDS = 0.3
MAX_SECONDS = 60.0
MUSAN_SEGMENT_SECONDS = 6.0
MUSAN_MIN_LAST_SECONDS = 2.0
MUSAN_FILES_PER_UNIT = 40
ASR_SOURCES = ("LSVSC", "VLSP", "ViMD")
TRANSCRIPT_COLUMNS = ("transcription", "text", "normalized_text", "raw_transcription", "sentence")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT.parent / "ctctknk_data")
    parser.add_argument("--output-root", type=Path,
                        default=Path("/home/ml4u/datasets/v14_negatives_16k"))
    parser.add_argument("--musan-root", type=Path, default=None,
                        help="extracted MUSAN directory (contains music/ and noise/)")
    parser.add_argument("--sources", default="LSVSC,VLSP,ViMD",
                        help="comma list from LSVSC,VLSP,ViMD,MUSAN")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="drop utterances longer than this; transcripts cannot be split "
                             "without word timestamps")
    parser.add_argument("--merge-only", action="store_true",
                        help="only rebuild the manifest and summary from existing parts")
    return parser.parse_args(argv)


# ============================================================
# Work units
# ============================================================

def asr_units(data_root: Path, sources):
    """Group requested rows by source Parquet file, keeping row order."""
    by_file = collections.defaultdict(list)
    with (data_root / "manifests" / "aligned_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["dataset"] not in sources:
                continue
            by_file[(record["dataset"], record["audio_relpath"])].append({
                "uid": f"{record['dataset']}:{record['sample_id']}",
                "audio_row": int(record["audio_row"]),
                "utterance_index": str(record["sample_id"]),
                "origin": "aligned_manifest",
            })
    if "VLSP" in sources:
        eligible = (data_root / "manifests" / "unused_non_infore2_audit"
                    / "unused_non_infore2_candidates_eligible.jsonl")
        with eligible.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["dataset"] != "VLSP" or not record.get("eligible", False):
                    continue
                by_file[("VLSP", record["audio_relpath"])].append({
                    "uid": f"VLSP:u{record['candidate_index']}",
                    "audio_row": int(record["audio_row"]),
                    "utterance_index": f"u{record['candidate_index']}",
                    "origin": "unused_vlsp_train",
                })
    units = []
    counters = collections.Counter()
    for (source, relpath), rows in sorted(by_file.items()):
        rows.sort(key=lambda item: item["audio_row"])
        rows_seen = [row["audio_row"] for row in rows]
        if len(rows_seen) != len(set(rows_seen)):
            raise RuntimeError(f"duplicate rows requested from {relpath}")
        units.append({
            "kind": "asr",
            "source": source,
            "name": f"{source}-{counters[source]:05d}",
            "input": str(data_root / "audio_shards" / relpath),
            "input_relpath": relpath,
            "rows": rows,
        })
        counters[source] += 1
    return units


def musan_units(musan_root: Path):
    files = sorted(
        path for category in ("music", "noise")
        for path in (musan_root / category).rglob("*.wav")
    )
    if not files:
        raise FileNotFoundError(f"no MUSAN music/noise wav under {musan_root}")
    units = []
    for start in range(0, len(files), MUSAN_FILES_PER_UNIT):
        units.append({
            "kind": "musan",
            "source": "MUSAN",
            "name": f"MUSAN-{start // MUSAN_FILES_PER_UNIT:05d}",
            "files": [str(path) for path in files[start:start + MUSAN_FILES_PER_UNIT]],
            "musan_root": str(musan_root),
        })
    return units


# ============================================================
# Worker
# ============================================================

def _bind_decoder():
    """Bind only what v12.decode_audio_object needs (no NeMo import)."""
    import librosa
    import numpy
    import soundfile
    import torch

    import parakeet_v12_train as v12

    v12.np, v12.sf, v12.librosa, v12.torch = numpy, soundfile, librosa, torch
    torch.set_num_threads(1)
    return v12


def _pcm16_wav_bytes(waveform) -> bytes:
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, waveform, TARGET_SR, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def _write_unit(output_root: Path, name: str, payloads, records):
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_rel = f"shards/{name}.parquet"
    shard_path = output_root / shard_rel
    temporary = shard_path.with_suffix(".parquet.tmp")
    table = pa.table({"audio": pa.array(payloads, type=pa.binary())})
    pq.write_table(table, temporary, row_group_size=ROW_GROUP_SIZE, compression="none")
    os.replace(temporary, shard_path)
    part_path = output_root / "parts" / f"{name}.jsonl"
    part_tmp = part_path.with_suffix(".jsonl.tmp")
    with part_tmp.open("w", encoding="utf-8") as handle:
        for row, record in enumerate(records):
            record["audio_relpath"] = shard_rel
            record["audio_row"] = row
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(part_tmp, part_path)


def _process_asr(unit, v12, stats, payloads, records):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(unit["input"])
    names = parquet.schema_arrow.names
    text_column = next((c for c in TRANSCRIPT_COLUMNS if c in names), None)
    if text_column is None:
        raise RuntimeError(f"no transcript column in {unit['input']}: {names}")
    wanted = {row["audio_row"]: row for row in unit["rows"]}
    stats["requested"] += len(wanted)
    offset = 0
    for group in range(parquet.metadata.num_row_groups):
        group_rows = parquet.metadata.row_group(group).num_rows
        hits = [r for r in range(offset, offset + group_rows) if r in wanted]
        if hits:
            table = parquet.read_row_group(group, columns=["audio", text_column])
            audio = table.column("audio")
            texts = table.column(text_column)
            for row in hits:
                meta = wanted[row]
                transcript = texts[row - offset].as_py()
                if transcript is None or not str(transcript).strip():
                    stats["skip_empty_transcript"] += 1
                    continue
                if v12.contains_word_sequence(v12.normalize_text(transcript)):
                    stats["skip_contains_wake_phrase"] += 1
                    continue
                waveform = v12.decode_audio_object(audio[row - offset].as_py()).numpy()
                seconds = len(waveform) / TARGET_SR
                if not MIN_SECONDS <= seconds <= unit.get("max_seconds", MAX_SECONDS):
                    stats["skip_duration"] += 1
                    continue
                payload = _pcm16_wav_bytes(waveform)
                payloads.append(payload)
                records.append({
                    "uid": meta["uid"],
                    "source_dataset": unit["source"],
                    "duration_seconds": round(seconds, 4),
                    "transcript": str(transcript),
                    "speaker_id": None,
                    "utterance_index": meta["utterance_index"],
                    "origin": {"list": meta["origin"], "audio_relpath": unit["input_relpath"],
                               "audio_row": row},
                    "pcm16_sha1": hashlib.sha1(payload).hexdigest(),
                })
                stats["kept"] += 1
                stats["kept_seconds"] += seconds
            del audio, texts, table
        offset += group_rows
    found = sum(1 for row in wanted if row < offset)
    if found != len(wanted):
        raise IndexError(f"{len(wanted) - found} requested rows exceed {unit['input']}")


def _process_musan(unit, v12, stats, payloads, records):
    root = Path(unit["musan_root"])
    segment = int(MUSAN_SEGMENT_SECONDS * TARGET_SR)
    minimum_last = int(MUSAN_MIN_LAST_SECONDS * TARGET_SR)
    for file_name in unit["files"]:
        path = Path(file_name)
        waveform = v12.decode_audio_object(str(path)).numpy()
        relative = path.relative_to(root).as_posix()
        track = relative.rsplit(".", 1)[0]
        stats["requested"] += 1
        for index, start in enumerate(range(0, len(waveform), segment)):
            piece = waveform[start:start + segment]
            if len(piece) < segment and len(piece) < minimum_last:
                stats["skip_short_tail"] += 1
                continue
            payload = _pcm16_wav_bytes(piece)
            payloads.append(payload)
            records.append({
                "uid": f"MUSAN:{track}:{index}",
                "source_dataset": "MUSAN",
                "duration_seconds": round(len(piece) / TARGET_SR, 4),
                "transcript": "",
                "speaker_id": f"musan:{track}",
                "utterance_index": str(index),
                "origin": {"list": "musan", "file": relative, "start_seconds": start / TARGET_SR},
                "pcm16_sha1": hashlib.sha1(payload).hexdigest(),
            })
            stats["kept"] += 1
            stats["kept_seconds"] += len(piece) / TARGET_SR


def process_unit(unit: dict, output_root: str) -> dict:
    output_root = Path(output_root)
    v12 = _bind_decoder()
    stats = collections.Counter()
    payloads, records = [], []
    if unit["kind"] == "asr":
        _process_asr(unit, v12, stats, payloads, records)
    else:
        _process_musan(unit, v12, stats, payloads, records)
    if payloads:
        _write_unit(output_root, unit["name"], payloads, records)
    else:
        (output_root / "parts" / f"{unit['name']}.jsonl").write_text("", encoding="utf-8")
    return {"name": unit["name"], "source": unit["source"], "stats": dict(stats)}


# ============================================================
# Merge
# ============================================================

def merge(output_root: Path, unit_names):
    manifest = output_root / "v14_negatives_16k_manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    totals = collections.defaultdict(collections.Counter)
    uids = set()
    with temporary.open("w", encoding="utf-8") as out:
        for name in unit_names:
            with (output_root / "parts" / f"{name}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record["uid"] in uids:
                        raise RuntimeError(f"duplicate uid {record['uid']}")
                    uids.add(record["uid"])
                    counter = totals[record["source_dataset"]]
                    counter["rows"] += 1
                    counter["seconds"] += record["duration_seconds"]
                    out.write(line)
    os.replace(temporary, manifest)
    summary = {
        source: {"rows": int(c["rows"]), "hours": round(c["seconds"] / 3600, 2)}
        for source, c in sorted(totals.items())
    }
    summary["_total"] = {
        "rows": sum(v["rows"] for v in summary.values()),
        "hours": round(sum(v["hours"] for v in summary.values()), 2),
    }
    return manifest, summary


def main(argv=None):
    args = parse_args(argv)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(sources) - set(ASR_SOURCES) - {"MUSAN"}
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    output_root = args.output_root
    (output_root / "shards").mkdir(parents=True, exist_ok=True)
    (output_root / "parts").mkdir(parents=True, exist_ok=True)

    units = asr_units(args.data_root, [s for s in sources if s in ASR_SOURCES])
    if "MUSAN" in sources:
        if args.musan_root is None:
            raise ValueError("--musan-root is required for MUSAN")
        units += musan_units(args.musan_root)
    for unit in units:
        unit["max_seconds"] = float(args.max_seconds)

    done ={p.stem for p in (output_root / "parts").glob("*.jsonl")}
    pending = [u for u in units if u["name"] not in done]
    print(f"units: total={len(units)} already_done={len(units) - len(pending)} "
          f"pending={len(pending)} workers={args.workers}", flush=True)

    grand = collections.Counter()
    if pending and not args.merge_only:
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_unit, u, str(output_root)) for u in pending]
            for finished, future in enumerate(as_completed(futures), 1):
                result = future.result()
                grand.update(result["stats"])
                print(f"[{finished}/{len(pending)}] {result['name']:>16s} "
                      f"kept={result['stats'].get('kept', 0):6d} "
                      f"elapsed={(time.time() - started) / 60:6.1f}m", flush=True)
        print("run stats:", dict(grand), flush=True)

    all_names = sorted(p.stem for p in (output_root / "parts").glob("*.jsonl"))
    manifest, summary = merge(output_root, all_names)
    summary["_last_run_stats"] = {k: (round(v, 2) if isinstance(v, float) else v)
                                  for k, v in grand.items()}
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("manifest:", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
