#!/usr/bin/env python3
"""Negative manifest for V17 (ctctknk-docs/33): V14's negatives + the 3a corpora.

V14 trains the wake head with negatives from VLSP, LSVSC, ViMD and MUSAN
(``v14_negatives_16k``).  V17 keeps those rows unchanged and adds every *train*
row of the phase-3a split whose corpus V14 never had: VSV-1100, FOSD,
CommonVoice20, Vietmed, fleurs, vais1000 and MS-SNSD (audio on the NVMe copy,
``splits_3a_nvme.jsonl``; InfoRE2 is loaded separately by the V12.2 path, and
VLSP/LSVSC stay as V14 prepared them).

``parakeet_v14_train.load_extra_negatives`` requires every shard to sit below the
manifest's directory, so the manifest is written to ``/home/ml4u/datasets`` and
paths are relative to it.  The 3a transcript is the number-normalised ``norm``
(numbers spoken as words, like the targets of 3a).  A row whose transcript
contains the whole wake phrase would be a positive, not a negative: none exist in
the 3a train split (checked 24/09), and the script refuses to write one.

    python build_v17_negatives.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/ml4u/datasets")
V14_MANIFEST = ROOT / "v14_negatives_16k" / "v14_negatives_16k_manifest.jsonl"
SPLITS_3A = ROOT / "ctctknk_3a" / "splits_3a_nvme.jsonl"
OUT = ROOT / "v17_negatives_manifest.jsonl"
# The new sources alone: input of prepare_v14_2_channel_shards.py (channel copies).
NEW_ONLY = ROOT / "v17_new_negatives_manifest.jsonl"
NEW_DATASETS = ("VSV-1100", "FOSD", "CommonVoice20", "Vietmed", "fleurs", "vais1000", "MS-SNSD")
# VSV-1100 has 1.02 M rows; the sampler draws ~16 k of them per epoch at a 0.20
# share, and every row needs ~0.57 channel copies on the NVMe (V14.2's copy rate),
# so a fixed hashed subset is used.  The other new sources are kept whole.
SUBSET_ROWS = {"VSV-1100": 50_000}
WAKE_WORDS = ("năng", "khiếu", "ơi")


def has_wake_phrase(text: str) -> bool:
    words = text.split()
    return any(tuple(words[i:i + 3]) == WAKE_WORDS for i in range(len(words) - 2))


def v14_rows():
    base = V14_MANIFEST.parent.relative_to(ROOT)
    with V14_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                record["audio_relpath"] = str(base / record["audio_relpath"])
                yield record


def hash_rank(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def subset_keys() -> dict[str, set]:
    """The SUBSET_ROWS lowest-hash train keys of each subsetted source (deterministic)."""
    ranked: dict[str, list] = collections.defaultdict(list)
    with SPLITS_3A.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["role"] == "train" and row["dataset"] in SUBSET_ROWS:
                ranked[row["dataset"]].append((hash_rank(row["key"]), row["key"]))
    return {name: {key for _, key in sorted(rows)[:SUBSET_ROWS[name]]}
            for name, rows in ranked.items()}


def split_3a_rows():
    keep = subset_keys()
    with SPLITS_3A.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["role"] != "train" or row["dataset"] not in NEW_DATASETS:
                continue
            if row["dataset"] in keep and row["key"] not in keep[row["dataset"]]:
                continue
            audio = Path(row["audio_path"])
            if ROOT not in audio.parents:
                raise RuntimeError(f"{row['key']}: audio outside {ROOT}: {audio}")
            if has_wake_phrase(row["norm"]):
                raise RuntimeError(f"{row['key']}: 3a transcript holds the wake phrase")
            yield {"uid": f"3a:{row['key']}", "source_dataset": row["dataset"],
                   "duration_seconds": float(row["duration"]), "transcript": row["norm"],
                   "speaker_id": None, "utterance_index": row["key"],
                   "audio_relpath": str(audio.relative_to(ROOT)),
                   "audio_row": int(row["audio_row"])}


def main() -> int:
    counts: collections.Counter = collections.Counter()
    hours: collections.Counter = collections.Counter()
    partial = OUT.with_suffix(".jsonl.partial")
    new_partial = NEW_ONLY.with_suffix(".jsonl.partial")
    with partial.open("w", encoding="utf-8") as out, \
            new_partial.open("w", encoding="utf-8") as new_out:
        for is_new, source in ((False, v14_rows()), (True, split_3a_rows())):
            for record in source:
                line = json.dumps(record, ensure_ascii=False) + "\n"
                out.write(line)
                if is_new:
                    new_out.write(line)
                counts[record["source_dataset"]] += 1
                hours[record["source_dataset"]] += record["duration_seconds"] / 3600
    partial.replace(OUT)
    new_partial.replace(NEW_ONLY)
    for name in sorted(counts):
        print(f"{name:14s} {counts[name]:>9,} rows {hours[name]:8.1f} h")
    print(f"total {sum(counts.values()):,} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
