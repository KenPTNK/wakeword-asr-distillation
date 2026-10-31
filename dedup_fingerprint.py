#!/usr/bin/env python3
"""Is any training utterance cut from a program we measure false accepts on?

VSV-1100 (phase-1 training of the paper encoder) was scraped from social media
without source ids, and the FA test programs (hk15_24h, 2N1D, Have a Sip, the
sealed 100 h set) are YouTube uploads.  A training clip taken from one of them
would make its FA/h optimistic.  This matches audio, not metadata, with
landmark fingerprints (constellation hashes, Wang 2003):

* audio -> 8 kHz, STFT (512 / hop 128 = 16 ms), log magnitude;
* spectral peaks = local maxima in a 15 x 15 neighbourhood, strongest
  ``PEAKS_PER_SECOND`` kept;
* each peak paired with up to ``FAN_OUT`` later peaks (1..40 frames, |df| <= 60
  bins) -> 24-bit hash (f1, df, dt) with the anchor time;
* a query clip matches a program when >= ``MIN_VOTES`` of its hashes agree on
  one time offset (+-1 frame) in that program.

Subcommands:
    index      fingerprint the programs -> index.npz (sorted hashes, times, ids)
    calibrate  sensitivity on degraded snippets of the programs, and false
               matches on read speech that cannot be in them (InfoRE2)
    query      scan parquet shards of a dataset, write matches.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RATE = 8000
N_FFT = 512
HOP = 128
PEAKS_PER_SECOND = 25
NEIGHBOURHOOD = 15
FAN_OUT = 5
MAX_DT = 40
MAX_DF = 60
MIN_VOTES = 25                          # calibrated 2026-09-25: InfoRE2 false max 19, true p50 121
CHUNK_SECONDS = 600
OFFSET_BASE = 10_000_000


def to_8k(wave: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import resample_poly
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate != RATE:
        wave = resample_poly(wave, RATE, rate).astype(np.float32)
    return wave


def peaks(wave8k: np.ndarray):
    from scipy.ndimage import maximum_filter
    from scipy.signal import stft
    if wave8k.size < N_FFT:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    _, _, spec = stft(wave8k, fs=RATE, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None,
                      padded=False)
    mag = np.log(np.abs(spec) + 1e-6)
    local = (mag == maximum_filter(mag, size=NEIGHBOURHOOD)) & (mag > mag.mean())
    freqs, times = np.nonzero(local)
    keep = int(PEAKS_PER_SECOND * wave8k.size / RATE) + 1
    if freqs.size > keep:
        strongest = np.argsort(mag[freqs, times])[-keep:]
        freqs, times = freqs[strongest], times[strongest]
    order = np.argsort(times, kind="stable")
    return freqs[order].astype(np.int32), times[order].astype(np.int32)


def hashes(wave8k: np.ndarray):
    """(hash uint32, anchor frame int32) pairs, vectorised over look-ahead k."""
    freqs, times = peaks(wave8k)
    n = freqs.size
    out_h, out_t, made = [], [], np.zeros(n, np.int32)
    for k in range(1, 4 * FAN_OUT + 1):
        if k >= n:
            break
        dt = times[k:] - times[:-k]
        df = freqs[k:] - freqs[:-k]
        ok = (dt >= 1) & (dt <= MAX_DT) & (np.abs(df) <= MAX_DF) & (made[:-k] < FAN_OUT)
        anchor = np.flatnonzero(ok)
        made[anchor] += 1
        out_h.append((freqs[anchor].astype(np.uint32) << 15)
                     | ((df[anchor] + MAX_DF).astype(np.uint32) << 7) | dt[anchor].astype(np.uint32))
        out_t.append(times[anchor])
    if not out_h:
        return np.zeros(0, np.uint32), np.zeros(0, np.int32)
    return np.concatenate(out_h), np.concatenate(out_t).astype(np.int32)


def file_hashes(path: Path):
    import soundfile as sf
    info = sf.info(str(path))
    block = int(CHUNK_SECONDS * info.samplerate)
    all_h, all_t = [], []
    offset_frames = 0
    for chunk in sf.blocks(str(path), blocksize=block, dtype="float32"):
        wave = to_8k(chunk, info.samplerate)
        h, t = hashes(wave)
        all_h.append(h)
        all_t.append(t + offset_frames)
        offset_frames += wave.size // HOP
    return np.concatenate(all_h), np.concatenate(all_t)


def _index_one(item):
    program_id, path = item
    h, t = file_hashes(Path(path))
    return program_id, h, t


def build_index(args) -> int:
    programs = []
    for spec in args.programs:
        path = Path(spec)
        programs.extend(sorted(path.glob("*.wav")) if path.is_dir() else [path])
    names = [str(p) for p in programs]
    parts = []
    with ProcessPoolExecutor(args.workers) as pool:
        for program_id, h, t in pool.map(_index_one, list(enumerate(names))):
            parts.append((h, t, np.full(h.size, program_id, np.int32)))
            print(f"indexed {names[program_id]}: {h.size:,} hashes", flush=True)
    h = np.concatenate([p[0] for p in parts])
    t = np.concatenate([p[1] for p in parts])
    ids = np.concatenate([p[2] for p in parts])
    order = np.argsort(h, kind="stable")
    np.savez(args.index, hashes=h[order], times=t[order], ids=ids[order],
             names=np.array(names))
    print(f"index: {len(names)} programs, {h.size:,} hashes -> {args.index}", flush=True)
    return 0


class Index:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.hashes, self.times, self.ids = data["hashes"], data["times"], data["ids"]
        self.names = [str(n) for n in data["names"]]

    def match(self, wave8k: np.ndarray):
        """Best (votes, program_id, offset_frames) for one query clip."""
        qh, qt = hashes(wave8k)
        if qh.size == 0:
            return 0, -1, 0
        lo = np.searchsorted(self.hashes, qh, side="left")
        hi = np.searchsorted(self.hashes, qh, side="right")
        counts = hi - lo
        if counts.sum() == 0:
            return 0, -1, 0
        starts = np.repeat(lo, counts)
        within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        rows = starts + within
        offsets = self.times[rows] - np.repeat(qt, counts)
        keys = self.ids[rows].astype(np.int64) * OFFSET_BASE + (offsets.astype(np.int64)
                                                                   + OFFSET_BASE // 2)
        values, votes = np.unique(keys, return_counts=True)
        best = int(np.argmax(votes))
        near = np.abs(values - values[best]) <= 1        # +-1 frame of jitter
        program = int(values[best] // OFFSET_BASE)
        offset = int(values[best] % OFFSET_BASE) - OFFSET_BASE // 2
        return int(votes[near].sum()), program, offset


_INDEX: Index | None = None


def _init_worker(index_path: str) -> None:
    global _INDEX
    _INDEX = Index(Path(index_path))


def _query_shard(task):
    import pyarrow.parquet as pq
    import soundfile as sf
    path, rows_wanted = task
    wanted = set(rows_wanted) if rows_wanted is not None else None
    table = pq.ParquetFile(path)
    found, scanned, row = [], 0, 0
    for group in range(table.num_row_groups):
        column = table.read_row_group(group, columns=["audio"]).column("audio").to_pylist()
        for audio in column:
            current = row
            row += 1
            if wanted is not None and current not in wanted:
                continue
            wave, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            votes, program, offset = _INDEX.match(to_8k(wave, rate))
            scanned += 1
            if votes >= MIN_VOTES:
                found.append({"key": f"{path}#{current}", "program": _INDEX.names[program],
                              "offset_s": offset * HOP / RATE, "votes": votes})
    return path, scanned, found


def query(args) -> int:
    rows_by_file: dict[str, list[int]] | None = None
    if args.split_manifest:
        rows_by_file = {}
        with open(args.split_manifest, encoding="utf-8") as handle:
            for line in handle:
                if args.dataset not in line:
                    continue
                record = json.loads(line)
                if record.get("dataset") != args.dataset or record.get("role") not in args.roles:
                    continue
                rows_by_file.setdefault(record["audio_path"], []).append(int(record["audio_row"]))
    files = sorted(rows_by_file) if rows_by_file is not None else sorted(
        str(p) for p in Path(args.parquet_dir).glob("*.parquet"))
    tasks = [(f, rows_by_file[f] if rows_by_file is not None else None) for f in files]
    total = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out, ProcessPoolExecutor(
            args.workers, initializer=_init_worker, initargs=(str(args.index),)) as pool:
        for path, scanned, found in pool.map(_query_shard, tasks):
            total += scanned
            for row in found:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{Path(path).name}: scanned {scanned:,}, matches {len(found)} "
                  f"(total scanned {total:,})", flush=True)
    return 0


def calibrate(args) -> int:
    import pyarrow.parquet as pq
    import soundfile as sf
    index = Index(args.index)
    rng = np.random.default_rng(20260925)
    hits = trials = 0
    true_votes = []
    for name in rng.choice(index.names, size=args.snippets, replace=True):
        info = sf.info(name)
        length = int(args.snippet_seconds * info.samplerate)
        start = int(rng.integers(0, info.frames - length))
        wave, rate = sf.read(name, start=start, frames=length, dtype="float32")
        noise = rng.normal(0, np.sqrt(np.mean(wave ** 2)) / 10 ** (args.snr_db / 20), wave.size)
        degraded = (wave + noise) * rng.uniform(0.3, 1.5)
        votes, program, offset = index.match(to_8k(degraded, rate))
        trials += 1
        expected = start / rate * RATE / HOP
        correct = index.names[program] == name and abs(offset - expected) <= 2
        true_votes.append(votes if correct else 0)
        hits += votes >= MIN_VOTES and correct
    print(f"sensitivity: {hits}/{trials} = {hits / trials:.3f} "
          f"({args.snippet_seconds:g} s snippets, SNR {args.snr_db:g} dB, gain 0.3-1.5)",
          flush=True)
    false = scanned = 0
    max_votes = []
    for path in sorted(Path(args.negative_dir).glob("*.parquet"))[:args.negative_files]:
        column = pq.read_table(path, columns=["audio"]).column("audio").to_pylist()
        for audio in column[:args.negatives_per_file]:
            wave, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            votes, _, _ = index.match(to_8k(wave, rate))
            scanned += 1
            max_votes.append(votes)
            false += votes >= MIN_VOTES
    print(f"false matches on {args.negative_dir}: {false}/{scanned}; best-offset votes "
          f"p50/p99/max = {np.percentile(max_votes, 50):.0f}/"
          f"{np.percentile(max_votes, 99):.0f}/{max(max_votes)}", flush=True)
    true_votes, max_votes = np.array(true_votes), np.array(max_votes)
    print("true-match votes p1/p5/p50 = "
          + "/".join(f"{np.percentile(true_votes, p):.0f}" for p in (1, 5, 50)), flush=True)
    for threshold in (12, 20, 25, 30, 40, 60):
        print(f"  min_votes={threshold}: sensitivity {(true_votes >= threshold).mean():.3f}, "
              f"false-match rate {(max_votes >= threshold).mean():.4f}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("index")
    p.add_argument("--programs", nargs="+", required=True, help="wav files or directories")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("calibrate")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--snippets", type=int, default=300)
    p.add_argument("--snippet-seconds", type=float, default=4.0)
    p.add_argument("--snr-db", type=float, default=15.0)
    p.add_argument("--negative-dir", required=True)
    p.add_argument("--negative-files", type=int, default=3)
    p.add_argument("--negatives-per-file", type=int, default=500)
    p = sub.add_parser("query")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--parquet-dir", default=None)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--dataset", default="VSV-1100")
    p.add_argument("--roles", nargs="+", default=["train"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-votes", type=int, default=MIN_VOTES)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if getattr(args, "min_votes", None) is not None:
        globals()["MIN_VOTES"] = int(args.min_votes)   # workers inherit it through fork
    return {"index": build_index, "calibrate": calibrate, "query": query}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
