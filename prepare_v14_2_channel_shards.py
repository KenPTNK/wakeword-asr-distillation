#!/usr/bin/env python3
"""Precompute V14.2 channel-augmented copies (room reverb + band limit) as 16 kHz shards.

Why on disk: the live path (``parakeet_v14_2_channel.install``) replaced a
hashed 30 % of samples by a reverberant version for the whole run, so their
clean versions never existed during training, and it could only hold 2,000 of
the 60,218 usable impulse responses in RAM.  Here every copy is ADDED beside
its original and draws its own response from the full set.
``ctctknk-docs/20`` has the reasoning and the numbers.

What gets a copy
----------------
Every source the sampler draws from, at one shared rate, so the augmented
share of negative draws equals that of positive draws (otherwise the model can
learn "reverberant / narrowband => wake word"):

* WakewordV12 (positives and negatives), **train speakers only** according to
  ``--speaker-split-file``.  A family (speaker, transcript, utterance) of
  ``n`` variants gets ``round(n * p/(1-p))`` copies in the SAME family, split
  across room / band / room+band by largest remainder.
* LSVSC, VLSP, ViMD, MUSAN (``v14_negatives_16k``) and InfoRE2 (the aligned
  manifest; all of it is training data in V12.2+).  Each row is its own
  family, so a copy becomes a NEW family (``utterance_index`` suffixed): a
  copy joined to a one-row family would only get half of that family's draws.
  Each row gets ``floor(q)`` copies plus one with probability ``frac(q)``.

The plan is a pure function of the inputs, the probabilities and ``--salt``
(blake2b of the uid), so a rerun reproduces it bit for bit.

Output layout (``--output-root``)::

    shards/<SOURCE>-<NNNNN>.parquet           audio: binary WAV, row groups of 64
    parts/<SOURCE>-<NNNNN>.jsonl              one record per row (written last = unit done)
    plan.json                                 configuration lock for resuming
    v14_2_channel_negatives_manifest.jsonl    -> --v14-2-channel-negative-manifest
    v14_2_wakeword_channel_manifest.jsonl     -> --wakeword-manifest (originals + copies)
    summary.json

Nothing under the source datasets is modified.
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
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import parakeet_v14_2_channel as channel  # noqa: E402

TARGET_SR = 16_000
ROW_GROUP_SIZE = 64
WAKEWORD = "WakewordV12"
PREPARED_NEGATIVES = ("LSVSC", "VLSP", "ViMD", "MUSAN")
ALL_SOURCES = (WAKEWORD, "InfoRE2") + PREPARED_NEGATIVES
NEGATIVE_MANIFEST_NAME = "v14_2_channel_negatives_manifest.jsonl"
WAKEWORD_MANIFEST_NAME = "v14_2_wakeword_channel_manifest.jsonl"
BYTES_PER_HOUR = {"PCM_16": TARGET_SR * 2 * 3600, "FLOAT": TARGET_SR * 4 * 3600}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wakeword-manifest", type=Path, default=Path(
        "/home/ml4u/datasets/wakeword_v12_16k_sharded/wakeword_v12_16k_sharded_manifest.jsonl"))
    parser.add_argument("--negative-manifest", type=Path, default=Path(
        "/home/ml4u/datasets/v14_negatives_16k/v14_negatives_16k_manifest.jsonl"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("/home/ml4u/datasets/ctctknk_data_nvme"),
                        help="holds manifests/aligned_manifest.jsonl and audio_shards/ (InfoRE2)")
    parser.add_argument("--speaker-split-file", type=Path,
                        default=PROJECT_ROOT / "v14_2_speaker_splits.json")
    parser.add_argument("--rir-root", type=Path, default=channel.DEFAULT_RIR_ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=Path("/home/ml4u/datasets/v14_2_channel_16k"))
    parser.add_argument("--room-probability", type=float, default=channel.ROOM_PROBABILITY)
    parser.add_argument("--band-probability", type=float, default=channel.BAND_PROBABILITY)
    parser.add_argument("--salt", type=int, default=0)
    parser.add_argument("--sources", default=",".join(ALL_SOURCES))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--plan-only", action="store_true",
                        help="print what would be written (rows, hours, GB) and exit")
    parser.add_argument("--max-units", type=int, default=0,
                        help="process at most this many pending units (smoke tests)")
    parser.add_argument("--merge-only", action="store_true",
                        help="only rebuild the manifests and summary from existing parts")
    return parser.parse_args(argv)


# ============================================================
# Planning (main process, deterministic)
# ============================================================

def _speaker(value) -> str:
    return unicodedata.normalize("NFKC", str(value))


def _copy_job(origin: dict, index: int, kind: str, salt: int, rir_count: int) -> dict:
    uid = channel.copy_uid(origin["uid"], index, kind)
    return {
        "uid": uid,
        "index": index,
        "kind": kind,
        "rir_index": channel.pick_index(rir_count, salt, "rir", uid) if "room" in kind else None,
        "band_index": (channel.pick_index(len(channel.BANDS), salt, "band", uid)
                       if "band" in kind else None),
    }


def plan_wakeword(manifest: Path, splits: dict[str, str], config: dict, rir_count: int,
                  normalize) -> tuple[list[dict], dict]:
    """Copies for train-speaker WakewordV12 families, grouped by input shard."""
    families: dict[tuple, list[dict]] = collections.defaultdict(list)
    skipped = collections.Counter()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            speaker = _speaker(record["speaker_id"])
            if speaker not in splits:
                raise RuntimeError(f"speaker {speaker} missing from the split file")
            if splits[speaker] != "train":
                skipped[splits[speaker]] += 1
                continue
            key = (speaker, normalize(str(record["transcript"])), str(record["utterance_index"]))
            families[key].append(record)
    weights = channel.kind_weights(config["room_probability"], config["band_probability"])
    salt = config["salt"]
    by_input: dict[str, list[dict]] = collections.defaultdict(list)
    for key in sorted(families):
        variants = sorted(families[key],
                          key=lambda r: (channel.hash_fraction(salt, "base", r["uid"]), r["uid"]))
        count = channel.copies_for_family(len(variants), config["room_probability"],
                                          config["band_probability"])
        for index, kind in enumerate(channel.split_kinds(count, weights)):
            base = variants[index % len(variants)]
            job = _copy_job(base, index, kind, salt, rir_count)
            job["base"] = base
            source_path = (manifest.parent / str(base["audio_path"])).resolve()
            by_input[str(source_path)].append(job)
    units = [{"kind": "wakeword", "source": WAKEWORD, "input": path,
              "jobs": sorted(jobs, key=lambda j: int(j["base"]["audio_row"]))}
             for path, jobs in sorted(by_input.items())]
    return units, {"families": len(families), "skipped_non_train_rows": dict(skipped)}


def _single_row_jobs(rows: list[dict], config: dict, rir_count: int) -> list[dict]:
    weights = channel.kind_weights(config["room_probability"], config["band_probability"])
    jobs = []
    for row in rows:
        copies = channel.copies_for_row(row["uid"], config["salt"],
                                        config["room_probability"], config["band_probability"])
        for index in range(copies):
            kind = channel.pick_kind(f"{row['uid']}|{index}", config["salt"], weights)
            job = _copy_job(row, index, kind, config["salt"], rir_count)
            job["origin"] = row
            jobs.append(job)
    return jobs


def plan_prepared_negatives(manifest: Path, sources, config: dict, rir_count: int) -> list[dict]:
    """LSVSC / VLSP / ViMD / MUSAN rows of the V14 negative shards."""
    by_input: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["source_dataset"] in sources:
                path = str((manifest.parent / record["audio_relpath"]).resolve())
                by_input[(record["source_dataset"], path)].append(record)
    units = []
    for (source, path), rows in sorted(by_input.items()):
        jobs = _single_row_jobs(sorted(rows, key=lambda r: int(r["audio_row"])), config, rir_count)
        if jobs:
            units.append({"kind": "prepared", "source": source, "input": path, "jobs": jobs})
    return units


def plan_infore2(data_root: Path, config: dict, rir_count: int) -> list[dict]:
    """InfoRE2 rows of the aligned manifest (every one is training data in V12.2+)."""
    audio_root = (data_root / "audio_shards").resolve()
    by_input: dict[str, list[dict]] = collections.defaultdict(list)
    with (data_root / "manifests" / "aligned_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record.get("dataset", "")).strip().lower() != "infore2":
                continue
            path = (audio_root / str(record["audio_relpath"])).resolve()
            by_input[str(path)].append({
                "uid": f"InfoRE2:{int(record['sample_id'])}",
                "audio_row": int(record["audio_row"]),
                "utterance_index": str(record["sample_id"]),
                "duration_seconds": int(record["teacher_length"]) * 0.08,
            })
    units = []
    for path, rows in sorted(by_input.items()):
        jobs = _single_row_jobs(sorted(rows, key=lambda r: r["audio_row"]), config, rir_count)
        if jobs:
            units.append({"kind": "infore2", "source": "InfoRE2", "input": path, "jobs": jobs})
    return units


def name_units(units: list[dict]) -> list[dict]:
    counters = collections.Counter()
    for unit in units:
        unit["name"] = f"{unit['source']}-{counters[unit['source']]:05d}"
        counters[unit["source"]] += 1
    return units


def job_seconds(unit: dict, job: dict) -> float:
    row = job.get("base") or job.get("origin")
    return float(row.get("duration_seconds") or 0.0)


def describe_plan(units: list[dict]) -> dict:
    table = collections.defaultdict(collections.Counter)
    for unit in units:
        subtype = "FLOAT" if unit["kind"] == "wakeword" else "PCM_16"
        for job in unit["jobs"]:
            label = unit["source"]
            if unit["kind"] == "wakeword":
                label += " positive" if job["base"]["is_positive"] else " negative"
            counter = table[label]
            counter["copies"] += 1
            counter[job["kind"]] += 1
            seconds = job_seconds(unit, job)
            counter["seconds"] += seconds
            counter["bytes"] += seconds / 3600 * BYTES_PER_HOUR[subtype]
    return {label: {"copies": int(c["copies"]),
                    "room": int(c["room"]), "band": int(c["band"]),
                    "room+band": int(c["room+band"]),
                    "hours": round(c["seconds"] / 3600, 2),
                    "gigabytes": round(c["bytes"] / 1e9, 2)}
            for label, c in sorted(table.items())}


# ============================================================
# Worker
# ============================================================

_WORKER: dict[str, object] = {}


def _init_worker(rir_paths: list[str]) -> None:
    import librosa
    import numpy
    import soundfile
    import torch

    import parakeet_v12_train as v12

    # Bind only what v12.decode_audio_object needs (no NeMo import).
    v12.np, v12.sf, v12.librosa, v12.torch = numpy, soundfile, librosa, torch
    torch.set_num_threads(1)
    _WORKER.update(v12=v12, rir_paths=rir_paths, rirs={})


def _rir(index: int):
    """The response at ``index``, or the next usable one (deterministic)."""
    paths, cache = _WORKER["rir_paths"], _WORKER["rirs"]
    for offset in range(len(paths)):
        position = (index + offset) % len(paths)
        if position not in cache:
            cache[position] = channel.read_rir(Path(paths[position]))
        if cache[position] is not None:
            return position, cache[position]
    raise RuntimeError("no usable impulse response")


def _wav_bytes(waveform, subtype: str) -> bytes:
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, waveform, TARGET_SR, subtype=subtype, format="WAV")
    return buffer.getvalue()


def _augment(wave, job: dict) -> tuple[object, dict]:
    rir = None
    detail = {"kind": job["kind"]}
    if job["rir_index"] is not None:
        position, rir = _rir(int(job["rir_index"]))
        detail["rir"] = Path(_WORKER["rir_paths"][position]).name
    band = channel.BANDS[job["band_index"]] if job["band_index"] is not None else None
    if band is not None:
        detail["band_hz"] = list(band)
    return channel.apply_kind(wave, job["kind"], rir, band), detail


def _read_rows(path: str, rows: set[int], columns: list[str]) -> dict[int, dict]:
    """Only the row groups that hold a requested row."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    found, offset = {}, 0
    for group in range(parquet.metadata.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        hits = [r for r in range(offset, offset + size) if r in rows]
        if hits:
            table = parquet.read_row_group(group, columns=columns)
            for row in hits:
                found[row] = {c: table.column(c)[row - offset].as_py() for c in columns}
        offset += size
    missing = rows - set(found)
    if missing:
        raise IndexError(f"{len(missing)} rows beyond {path}, e.g. {sorted(missing)[:3]}")
    return found


def _wakeword_record(job: dict, payload: bytes, wave) -> dict:
    base = job["base"]
    record = dict(base)
    record.update({
        "uid": job["uid"],
        "source_kind": "parquet",
        "augmentation": f"{base.get('augmentation') or 'orig'}+ch:{job['kind']}",
        "is_augmented": True,
        "provenance_kind": "v14_2_channel_augmented",
        "origin_uid": base["uid"],
        "sample_rate": TARGET_SR,
        "prepared_audio_subtype": "FLOAT",
        "prepared_audio_encoding": "wav_float32",
        "prepared_audio_md5": hashlib.md5(payload).hexdigest(),
        "prepared_audio_size_bytes": len(payload),
        "prepared_pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
        "shard_row_group_size": ROW_GROUP_SIZE,
        "md5": None,
        "google_drive_object_id": None,
    })
    return record


def _process_wakeword(unit: dict, payloads: list, records: list) -> None:
    import numpy as np
    import soundfile as sf

    rows = _read_rows(unit["input"], {int(j["base"]["audio_row"]) for j in unit["jobs"]},
                      ["audio"])
    for job in unit["jobs"]:
        data, rate = sf.read(io.BytesIO(rows[int(job["base"]["audio_row"])]["audio"]),
                             dtype="float32", always_2d=False)
        if rate != TARGET_SR:
            raise RuntimeError(f"{unit['input']}: {rate} Hz")
        wave, detail = _augment(np.asarray(data if data.ndim == 1 else data.mean(axis=1)), job)
        payload = _wav_bytes(wave, "FLOAT")
        record = _wakeword_record(job, payload, wave)
        record["channel_augmentation"] = detail
        payloads.append(payload)
        records.append(record)


def _negative_record(unit: dict, job: dict, origin: dict, transcript: str,
                     seconds: float, payload: bytes, detail: dict) -> dict:
    return {
        "uid": job["uid"],
        "source_dataset": unit["source"],
        "duration_seconds": round(seconds, 4),
        "transcript": transcript,
        "speaker_id": origin.get("speaker_id"),
        # A new family: joined to a one-row family, a copy would get half its draws.
        "utterance_index": f"{origin['utterance_index']}{channel.CHANNEL_UID_TAG}{job['index']}",
        "origin_uid": origin["uid"],
        "channel_augmentation": detail,
        "pcm16_sha1": hashlib.sha1(payload).hexdigest(),
    }


def _process_single_rows(unit: dict, payloads: list, records: list) -> None:
    v12 = _WORKER["v12"]
    text_column = "transcription" if unit["kind"] == "infore2" else None
    columns = ["audio"] + ([text_column] if text_column else [])
    rows = _read_rows(unit["input"], {int(j["origin"]["audio_row"]) for j in unit["jobs"]},
                      columns)
    for job in unit["jobs"]:
        origin = job["origin"]
        row = rows[int(origin["audio_row"])]
        transcript = str(row[text_column] if text_column else origin.get("transcript") or "")
        if unit["source"] != "MUSAN" and not transcript.strip():
            raise RuntimeError(f"empty transcript for {origin['uid']}")
        if v12.contains_word_sequence(v12.normalize_text(transcript)):
            raise RuntimeError(f"negative source contains the wake phrase: {origin['uid']}")
        wave = v12.decode_audio_object(row["audio"]).numpy()
        augmented, detail = _augment(wave, job)
        payload = _wav_bytes(augmented, "PCM_16")
        payloads.append(payload)
        records.append(_negative_record(unit, job, origin, transcript,
                                        len(augmented) / TARGET_SR, payload, detail))


def _write_unit(output_root: Path, name: str, payloads: list, records: list,
                absolute_paths: bool) -> None:
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
            if absolute_paths:
                record["audio_path"] = str(shard_path.resolve())
            else:
                record["audio_relpath"] = shard_rel
            record["audio_row"] = row
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(part_tmp, part_path)


def process_unit(unit: dict, output_root: str) -> dict:
    payloads: list[bytes] = []
    records: list[dict] = []
    if unit["kind"] == "wakeword":
        _process_wakeword(unit, payloads, records)
    else:
        _process_single_rows(unit, payloads, records)
    _write_unit(Path(output_root), unit["name"], payloads, records,
                absolute_paths=unit["kind"] == "wakeword")
    kinds = collections.Counter(r["channel_augmentation"]["kind"] for r in records)
    return {"name": unit["name"], "rows": len(records), "kinds": dict(kinds)}


# ============================================================
# Merge
# ============================================================

def _share_by_family(originals: dict[tuple, int], copies: dict[tuple, int]) -> float:
    """Mean over families of copies/(originals+copies): the sampler's view."""
    if not originals:
        return 0.0
    return sum(copies.get(k, 0) / (n + copies.get(k, 0)) for k, n in originals.items()) / len(originals)


def merge(args, config: dict, unit_names: list[str], splits: dict[str, str], normalize) -> dict:
    root = args.output_root
    negatives = root / NEGATIVE_MANIFEST_NAME
    wakeword = root / WAKEWORD_MANIFEST_NAME
    uids: set[str] = set()
    per_source = collections.defaultdict(collections.Counter)
    wake_orig = collections.defaultdict(collections.Counter)
    wake_copy = collections.defaultdict(collections.Counter)

    with wakeword.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as out:
        with args.wakeword_manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["audio_path"] = str((args.wakeword_manifest.parent
                                            / str(record["audio_path"])).resolve())
                uids.add(record["uid"])
                speaker = _speaker(record["speaker_id"])
                if splits.get(speaker) == "train":
                    label = "positive" if record["is_positive"] else "negative"
                    key = (speaker, normalize(str(record["transcript"])),
                           str(record["utterance_index"]))
                    wake_orig[label][key] += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        for name in (n for n in unit_names if n.startswith(WAKEWORD)):
            for line in (root / "parts" / f"{name}.jsonl").open(encoding="utf-8"):
                record = json.loads(line)
                _claim(uids, record["uid"])
                label = "positive" if record["is_positive"] else "negative"
                key = (_speaker(record["speaker_id"]), normalize(str(record["transcript"])),
                       str(record["utterance_index"]))
                wake_copy[label][key] += 1
                per_source[f"{WAKEWORD} {label}"]["copies"] += 1
                per_source[f"{WAKEWORD} {label}"]["seconds"] += float(record["duration_seconds"])
                out.write(line if line.endswith("\n") else line + "\n")
    os.replace(wakeword.with_suffix(".jsonl.tmp"), wakeword)

    with negatives.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as out:
        for name in (n for n in unit_names if not n.startswith(WAKEWORD)):
            for line in (root / "parts" / f"{name}.jsonl").open(encoding="utf-8"):
                record = json.loads(line)
                _claim(uids, record["uid"])
                per_source[record["source_dataset"]]["copies"] += 1
                per_source[record["source_dataset"]]["seconds"] += float(record["duration_seconds"])
                out.write(line if line.endswith("\n") else line + "\n")
    os.replace(negatives.with_suffix(".jsonl.tmp"), negatives)

    summary = {label: {"copies": int(c["copies"]), "hours": round(c["seconds"] / 3600, 2)}
               for label, c in sorted(per_source.items())}
    for label in ("positive", "negative"):
        key = f"{WAKEWORD} {label}"
        summary.setdefault(key, {})["augmented_draw_share"] = round(
            _share_by_family(wake_orig[label], wake_copy[label]), 4)
    return {"config": config, "sources": summary,
            "manifests": {"wakeword": str(wakeword), "negatives": str(negatives)}}


def _claim(uids: set[str], uid: str) -> None:
    if uid in uids:
        raise RuntimeError(f"duplicate uid {uid}")
    uids.add(uid)


# ============================================================
# Main
# ============================================================

def lock_config(output_root: Path, config: dict) -> None:
    """Refuse to resume parts written under a different plan."""
    path = output_root / "plan.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError(f"{path} was written with a different configuration:\n"
                               f"  existing={previous}\n  requested={config}\n"
                               "use a new --output-root")
        return
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_plan(args, sources, splits, normalize, rir_count):
    config = {
        "room_probability": float(args.room_probability),
        "band_probability": float(args.band_probability),
        "augmented_fraction": round(channel.augmented_fraction(args.room_probability,
                                                               args.band_probability), 6),
        "salt": int(args.salt),
        "sources": sorted(sources),
        "speaker_split_sha1": hashlib.sha1(args.speaker_split_file.read_bytes()).hexdigest(),
        "rir_count": rir_count,
        "bands_hz": [list(b) for b in channel.BANDS],
    }
    units, wake_info = [], {}
    if WAKEWORD in sources:
        wake_units, wake_info = plan_wakeword(args.wakeword_manifest, splits, config,
                                              rir_count, normalize)
        units += wake_units
    prepared = [s for s in sources if s in PREPARED_NEGATIVES]
    if prepared:
        units += plan_prepared_negatives(args.negative_manifest, prepared, config, rir_count)
    if "InfoRE2" in sources:
        units += plan_infore2(args.data_root, config, rir_count)
    return config, name_units(units), wake_info


def main(argv=None):
    args = parse_args(argv)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    if set(sources) != set(ALL_SOURCES):
        print("WARNING: not every source gets copies; the augmented share of negative "
              "draws will not match the positives (ctctknk-docs/20 section 5)", flush=True)

    import parakeet_v12_train as v12

    splits = {_speaker(k): v for k, v in
              json.loads(args.speaker_split_file.read_text(encoding="utf-8")).items()}
    rir_paths = [str(p) for p in channel.usable_rir_paths(args.rir_root)]
    config, units, wake_info = build_plan(args, sources, splits, v12.normalize_text,
                                          len(rir_paths))
    plan = describe_plan(units)
    print(json.dumps({"config": config, "wakeword": wake_info, "plan": plan,
                      "total_gigabytes": round(sum(v["gigabytes"] for v in plan.values()), 2)},
                     indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return 0

    output_root = args.output_root
    (output_root / "shards").mkdir(parents=True, exist_ok=True)
    (output_root / "parts").mkdir(parents=True, exist_ok=True)
    lock_config(output_root, config)

    done = {p.stem for p in (output_root / "parts").glob("*.jsonl")}
    pending = [u for u in units if u["name"] not in done]
    if args.max_units:
        pending = pending[:args.max_units]
    print(f"units: total={len(units)} done={len(done)} running={len(pending)} "
          f"workers={args.workers}", flush=True)
    if pending and not args.merge_only:
        started = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(rir_paths,)) as pool:
            futures = [pool.submit(process_unit, u, str(output_root)) for u in pending]
            for finished, future in enumerate(as_completed(futures), 1):
                result = future.result()
                print(f"[{finished}/{len(pending)}] {result['name']:>18s} rows={result['rows']:6d} "
                      f"{result['kinds']} elapsed={(time.time() - started) / 60:6.1f}m", flush=True)

    names = [u["name"] for u in units if (output_root / "parts" / f"{u['name']}.jsonl").is_file()]
    if len(names) < len(units):
        print(f"{len(units) - len(names)} units still pending: manifests NOT written "
              "(they must never describe a partial dataset)", flush=True)
        return 0
    summary = merge(args, config, names, splits, v12.normalize_text)
    summary["planned"] = plan
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
