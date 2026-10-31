"""Constrained test metrics for the v4 paper tables (ICIIT 2027).

"FRR at x FA/h" = lowest FRR over thresholds with FA/h <= x AND near-miss accepts <= 10 %
(the rule of Tables 1-2).  For every npz in a score directory: FRR at 0.1/0.5/1/2 FA/h, NM and
FA events at each operating point, and at 1 FA/h the per-speaker FRR, the FRR on original
recordings, the speaker-averaged FRR and a cluster-bootstrap CI (analyze_scores.cluster_boot).

usage: python v4_tables.py v1_8k/merged > v1_8k/merged_stats.json      (11 speakers)
       python v4_tables.py v1_8k/scores                                 (original 4, check)
"""
import json
import sys
from pathlib import Path

import numpy as np

import analyze_scores as az

FA_TARGETS = (0.1, 0.5, 1.0, 2.0)
NM_CAP = 0.10


def constrained(pos, neg, thresholds, per_hour, limit):
    ok = (per_hour <= limit) & np.array([(neg >= t).mean() <= NM_CAP for t in thresholds])
    if not ok.any():
        return None
    recalls = np.array([(pos >= t).mean() for t in thresholds])
    return int(np.flatnonzero(ok)[np.argmax(recalls[ok])])


def analyse(path: Path, detail: bool = True) -> dict:
    d = np.load(path)
    pos, neg = d["pos_best"], d["neg_best"]
    thr, ev, hours = d["thresholds"], d["events"], float(d["hours"])
    per_hour = ev / hours
    keys = [az.utterance_key(str(x)) for x in d["pos_label"]]
    spk = np.array([k[0] for k in keys])
    utt = np.array([k[1] for k in keys])
    orig = d["pos_family"] == "orig"
    out = {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "n_orig": int(orig.sum()),
           "n_speakers": len(set(spk)), "levels": {}}
    for limit in FA_TARGETS:
        i = constrained(pos, neg, thr, per_hour, limit)
        if i is None:
            out["levels"][f"{limit:g}"] = None
            continue
        t = thr[i]
        miss = pos < t
        row = {"threshold": float(t), "fa_events": int(ev[i]), "fa_per_hour": float(per_hour[i]),
               "frr": float(miss.mean()), "near_miss_fa": float((neg >= t).mean())}
        if detail and limit == 1.0:
            by = {s: float(miss[spk == s].mean()) for s in sorted(set(spk))}
            row.update(frr_by_speaker=by, frr_speaker_mean=float(np.mean(list(by.values()))),
                       frr_orig=float(miss[orig].mean()),
                       frr_ci95_cluster=az.cluster_boot(miss, spk, utt))
        out["levels"][f"{limit:g}"] = row
    return out


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    report = {p.stem: analyse(p) for p in sorted(folder.glob("*.npz"))}
    print(json.dumps(report, indent=1))
