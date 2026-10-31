#!/usr/bin/env python3
"""Fixed train / validation / test split for phase 3a (ctctknk-docs/30 §3).

Merges the teacher-logit indices of every output dir (the logits live on two
disks) into one manifest, one JSON line per utterance with absolute paths:

    key, dataset, source_split, role, reason, audio_path, audio_row,
    logits_path, top64_stem, teacher_offset, teacher_length, duration, text, norm

``norm`` is ``vi_number_norm.normalize_for_asr(text)``: numbers read as words
the way the teacher says them, lower case, no punctuation — the CTC target and
the scoring reference.  ``top64_stem`` + ``.top64v/.top64i/.lse/.lseT2.npy`` is
the compact copy of the logits on the NVMe (``compact_teacher_logits.py``).

Roles (mirroring Parakeet-CTC-0.6B-vi, whose in-domain / blind test sets we
want to compare against):

* train       original train splits, <= 20 s (the teacher's max_duration),
              non-empty text (MS-SNSD noise pieces keep their empty text);
              InfoRE2 / VLSP / VSV-1100 minus a 0.5 % hash hold-out; FOSD
              minus a 5 % hash test.  Train utterances whose text (>= 4 words)
              also occurs in a validation / test set are counted and kept
              (``--drop-eval-text-overlap`` removes them).
* validation  Vietmed dev + cv (Parakeet's validation), LSVSC / FLEURS /
              Common Voice validation, the 0.5 % hold-outs.
* test_in     in-domain tests: LSVSC, FLEURS, Vietmed, Common Voice test,
              FOSD 5 % hold-out (no official FOSD split exists).
* dev_blind / test_blind   never trained on: ViMD partial-valid / partial-test,
              GigaSpeech2 dev / test, VIVOS test.
* noise_test  MS-SNSD noise_test pieces (false words on noise only).
* excluded    everything else, with the reason (ViMD train is kept blind;
              ViMD partial-train duplicates ViMD_full).

    python build_splits_3a.py --out /home/ml4u/datasets/ctctknk_3a
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path

import vi_number_norm

TOP64_ROOT = Path("/home/ml4u/datasets/ctctknk_3a/teacher_top64")
SAMSUNG_AUDIO = Path("/media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk_data/audio_shards")
KINGSTON = Path("/media/ml4u/KINGSTON/ctctknk")
INDEX_DIRS = [KINGSTON / "teacher_logits_full",
              Path("/media/ml4u/Extreme SSD/ctctknk/teacher_logits_part2")]
CONVERTED = {"GigaSpeech2", "VIVOS", "MS-SNSD"}
NAMES = {"fpt_fosd": "FOSD", "common-voice-corpus-20": "CommonVoice20"}
MAX_TRAIN_SECONDS = 20.0
MIN_SECONDS = 0.1
HOLDOUT_PER_MILLE = {"InfoRE2": 5, "VLSP": 5, "VSV-1100": 5}
FOSD_TEST_PER_MILLE = 50
LEAK_MIN_WORDS = 4

ROLE_OF = {  # (dataset, source split) -> role before the train filters
    ("LSVSC", "train"): "train", ("LSVSC", "validation"): "validation",
    ("LSVSC", "test"): "test_in",
    ("fleurs", "train"): "train", ("fleurs", "validation"): "validation",
    ("fleurs", "test"): "test_in",
    ("Vietmed", "train"): "train", ("Vietmed", "dev"): "validation",
    ("Vietmed", "cv"): "validation", ("Vietmed", "test"): "test_in",
    ("CommonVoice20", "train"): "train", ("CommonVoice20", "validation"): "validation",
    ("CommonVoice20", "test"): "test_in",
    ("InfoRE2", "train"): "train", ("VLSP", "train"): "train",
    ("VSV-1100", "train"): "train", ("FOSD", "train"): "train",
    ("vais1000", "train"): "train",
    ("ViMD", "partial-valid"): "dev_blind", ("ViMD", "partial-test"): "test_blind",
    ("GigaSpeech2", "dev"): "dev_blind", ("GigaSpeech2", "test"): "test_blind",
    ("VIVOS", "test"): "test_blind",
    ("MS-SNSD", "noise_train"): "train", ("MS-SNSD", "noise_test"): "noise_test",
}
EXCLUDED = {("ViMD_full", "data"): "vimd_kept_blind",
            ("ViMD", "partial-train"): "duplicate_of_vimd_full"}
EVAL_ROLES = {"validation", "test_in", "dev_blind", "test_blind"}


def normalize(text: str) -> str:
    """Numbers read as words, lower case, letters/digits only (``vi_number_norm``)."""
    return vi_number_norm.normalize_for_asr(text)


def top64_stem(logits_file: str) -> str:
    """logits/<rel>.data.npy -> TOP64_ROOT/<rel> (``compact_teacher_logits.py`` layout)."""
    relative = logits_file[len("logits/"):] if logits_file.startswith("logits/") else logits_file
    return str(TOP64_ROOT / relative.replace(".data.npy", ""))


def per_mille(*parts) -> int:
    digest = hashlib.sha1(("v3a:" + ":".join(map(str, parts))).encode()).hexdigest()
    return int(digest[:8], 16) % 1000


def audio_root(relpath: str) -> Path:
    top = relpath.split("/")[0]
    if top in CONVERTED:
        return KINGSTON / "audio_shards_3a"
    if "__" in top:
        return KINGSTON / "raw_datasets"
    return SAMSUNG_AUDIO


def assign(row: dict) -> tuple[str, str]:
    dataset, split = row["dataset"], row["source_split"]
    if (dataset, split) in EXCLUDED:
        return "excluded", EXCLUDED[(dataset, split)]
    role = ROLE_OF.get((dataset, split))
    if role is None:
        return "excluded", "unmapped_split"
    if row["duration"] < MIN_SECONDS:
        return "excluded", "under_0.1s"
    if role != "train":
        return role, ""
    if dataset in HOLDOUT_PER_MILLE and \
            per_mille(dataset, row["key"]) < HOLDOUT_PER_MILLE[dataset]:
        return "validation", "holdout_0.5pct"
    if dataset == "FOSD" and per_mille(dataset, row["key"]) < FOSD_TEST_PER_MILLE:
        return "test_in", "fosd_holdout_5pct"
    if row["duration"] > MAX_TRAIN_SECONDS:
        return "excluded", "over_20s"
    if not row["norm"] and dataset != "MS-SNSD":
        return "excluded", "empty_text"
    return "train", ""


def load_rows() -> list[dict]:
    rows, seen = [], set()
    for index_dir in INDEX_DIRS:
        pattern = str(index_dir / "index" / "**" / "*.jsonl")
        for index in sorted(glob.glob(pattern, recursive=True)):
            if not Path(index).with_suffix(".done").is_file():
                continue
            with open(index, encoding="utf-8") as handle:
                for line in handle:
                    r = json.loads(line)
                    key = f"{r['audio_relpath']}#{r['audio_row']}"
                    if key in seen:
                        raise SystemExit(f"duplicate utterance across index dirs: {key}")
                    seen.add(key)
                    rows.append({
                        "key": key, "dataset": NAMES.get(r["dataset"], r["dataset"]),
                        "source_split": r["split"],
                        "audio_path": str(audio_root(r["audio_relpath"]) / r["audio_relpath"]),
                        "audio_row": int(r["audio_row"]),
                        "logits_path": str(index_dir / r["teacher_file"]),
                        "top64_stem": top64_stem(r["teacher_file"]),
                        "teacher_offset": int(r["teacher_offset"]),
                        "teacher_length": int(r["teacher_length"]),
                        "duration": float(r["duration_seconds"]),
                        "text": r["text"], "norm": normalize(r["text"])})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drop-eval-text-overlap", action="store_true",
                        help="also drop train utterances whose text occurs in a validation/test "
                             "set (off: the team keeps them; the count is still reported)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    for row in rows:
        row["role"], row["reason"] = assign(row)

    # Optional (off since 2026-09-22, the team's choice: train drops only > 20 s):
    # drop training utterances whose text also appears in an evaluation set.
    eval_texts = {r["norm"] for r in rows
                  if r["role"] in EVAL_ROLES and len(r["norm"].split()) >= LEAK_MIN_WORDS}
    leaked = 0
    for row in rows:
        if row["role"] == "train" and row["norm"] in eval_texts:
            leaked += 1
            if args.drop_eval_text_overlap:
                row["role"], row["reason"] = "excluded", "text_in_eval_set"

    fields = ("key", "dataset", "source_split", "role", "reason", "audio_path", "audio_row",
              "logits_path", "top64_stem", "teacher_offset", "teacher_length", "duration",
              "text", "norm")
    manifest = args.out / "splits_3a.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({k: row[k] for k in fields}, ensure_ascii=False) + "\n")

    table = collections.defaultdict(lambda: [0, 0.0])
    reasons = collections.Counter()
    by_role = collections.defaultdict(lambda: [0, 0.0])
    for row in rows:
        for cell in (table[(row["role"], row["dataset"], row["source_split"])],
                     by_role[row["role"]]):
            cell[0] += 1
            cell[1] += row["duration"] / 3600
        if row["reason"]:
            reasons[(row["role"], row["dataset"], row["reason"])] += 1
    summary = {
        "rows": len(rows), "leaked_train_texts_removed": leaked,
        "by_role": {k: {"rows": v[0], "hours": round(v[1], 2)} for k, v in sorted(by_role.items())},
        "by_role_dataset_split": [{"role": k[0], "dataset": k[1], "split": k[2],
                                   "rows": v[0], "hours": round(v[1], 2)}
                                  for k, v in sorted(table.items())],
        "reasons": [{"role": k[0], "dataset": k[1], "reason": k[2], "rows": v}
                    for k, v in sorted(reasons.items())],
        "rules": {"max_train_seconds": MAX_TRAIN_SECONDS, "holdout_per_mille": HOLDOUT_PER_MILLE,
                  "fosd_test_per_mille": FOSD_TEST_PER_MILLE, "leak_min_words": LEAK_MIN_WORDS,
                  "index_dirs": [str(d) for d in INDEX_DIRS]},
    }
    (args.out / "splits_3a_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    action = "removed" if args.drop_eval_text_overlap else "kept"
    print(f"{len(rows):,} utterances -> {manifest}; train texts also in an eval set "
          f"({action}): {leaked:,}")
    for role, (count, hours) in sorted(by_role.items()):
        print(f"  {role:12s} {count:9,d} rows {hours:8.1f} h")
    print("\nrole/dataset/split:")
    for (role, dataset, split), (count, hours) in sorted(table.items()):
        print(f"  {role:12s} {dataset:14s} {split:14s} {count:9,d} {hours:8.1f} h")
    print("\nreasons:")
    for (role, dataset, reason), count in sorted(reasons.items()):
        print(f"  {role:12s} {dataset:14s} {reason:24s} {count:9,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
