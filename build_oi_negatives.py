#!/usr/bin/env python3
"""Negative manifests with an extra "OiHard" source (ctctknk-docs/33 §3s).

The final model's false accepts on hk15 are almost all vocatives ending in "ơi"
("anh em ơi", "Sếp ơi"...).  Only 1,063 phase-2 negatives contain the word "ơi"
without the wake phrase, and at the V14.2 source shares they are almost never
drawn.  This writes copies of the manifests where those rows (and their channel
copies) also appear under ``source_dataset="OiHard"`` with ``OiHard:``-prefixed
uids, so ``--v14-negative-source-shares ...,OiHard=0.05`` draws them directly.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"(?<!\w)ơi(?!\w)")
WAKE = "năng khiếu"


def is_oi(record: dict) -> bool:
    text = (record.get("transcript") or "").lower()
    return bool(PATTERN.search(text)) and WAKE not in text


def relabel(record: dict) -> dict:
    out = dict(record)
    out["uid"] = "OiHard:" + record["uid"]
    out["source_dataset"] = "OiHard"
    if "origin_uid" in record:
        out["origin_uid"] = "OiHard:" + record["origin_uid"]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--channel", type=Path, required=True)
    parser.add_argument("--out-negatives", type=Path, required=True)
    parser.add_argument("--out-channel", type=Path, required=True)
    args = parser.parse_args(argv)
    chosen = set()
    extra = []
    with args.negatives.open(encoding="utf-8") as src, \
            args.out_negatives.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            record = json.loads(line)
            if is_oi(record):
                chosen.add(record["uid"])
                extra.append(relabel(record))
        for record in extra:
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    extra = []
    with args.channel.open(encoding="utf-8") as src, \
            args.out_channel.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            record = json.loads(line)
            if record.get("origin_uid") in chosen:
                extra.append(relabel(record))
        for record in extra:
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"OiHard: {len(chosen):,} originals, {len(extra):,} channel copies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
