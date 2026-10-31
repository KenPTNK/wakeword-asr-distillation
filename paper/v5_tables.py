"""v5 tables: FRR at <= x FA/h with near-miss accepts <= 10 % on the v5 test speakers, with false
accepts counted on hk15 only (v5/scores/<run>.npz) and on hk15 + the sealed multi-domain 100 h set
(v5/longfa/<run>.npz from v5_longfa.py, same thresholds).  Per level: FRR, near-miss accepts, FA
events; at the level used for the text also the cluster CI, per-speaker FRR and FRR on originals.

usage: python v5_tables.py > v5/v5_stats.json
"""
import json
from pathlib import Path

import numpy as np

import analyze_scores as az
import v4_tables as vt

ROOT = Path(__file__).resolve().parent / "v5"
LEVELS = (0.1, 0.5, 1.0, 2.0)


def table(pos, neg, thr, events, hours, keys):
    spk = np.array([k[0] for k in keys])
    utt = np.array([k[1] for k in keys])
    out = {}
    for limit in LEVELS:
        i = vt.constrained(pos, neg, thr, events / hours, limit)
        if i is None:
            out[f"{limit:g}"] = None
            continue
        miss = pos < thr[i]
        out[f"{limit:g}"] = {
            "threshold": float(thr[i]), "fa_events": int(events[i]), "fa_per_hour": float(events[i] / hours),
            "frr": float(miss.mean()), "near_miss_fa": float((neg >= thr[i]).mean()),
            "frr_ci95_cluster": az.cluster_boot(miss, spk, utt),
            "frr_by_speaker": {s: float(miss[spk == s].mean()) for s in sorted(set(spk))}}
    return out


def analyse(name):
    d = np.load(ROOT / "scores" / f"{name}.npz")
    keys = [az.utterance_key(str(x)) for x in d["pos_label"]]
    orig = d["pos_family"] == "orig"
    pos, neg, thr = d["pos_best"], d["neg_best"], d["thresholds"]
    out = {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "n_orig": int(orig.sum()),
           "hk15": table(pos, neg, thr, d["events"].astype(float), float(d["hours"]), keys)}
    long_path = ROOT / "longfa" / f"{name}.npz"
    if long_path.is_file():
        f = np.load(long_path)
        assert np.allclose(f["thresholds"], thr), name
        events = d["events"].astype(float) + f["events"].sum(0)
        hours = float(d["hours"]) + float(f["hours"].sum())
        out["long"] = table(pos, neg, thr, events, hours, keys)
        out["long_hours"] = hours
        out["sealed_by_category_events"] = {
            c: f["events"][f["categories"] == c].sum(0).tolist() for c in sorted(set(f["categories"]))}
        out["sealed_by_category_hours"] = {
            c: float(f["hours"][f["categories"] == c].sum()) for c in sorted(set(f["categories"]))}
    for key in ("hk15", "long"):
        if key in out:
            for level in out[key].values():
                if level is not None:
                    level["frr_orig"] = float((pos < level["threshold"])[orig].mean())
    return out


if __name__ == "__main__":
    names = sorted(p.stem for p in (ROOT / "scores").glob("*.npz")
                   if not p.stem.startswith(("old_", "pipeline")) and "deploy" not in p.stem)
    report = {}
    for n in names:
        try:
            report[n] = analyse(n)
        except (ValueError, OSError) as err:        # placeholder files of runs still in progress
            report[n] = {"error": str(err)}
    print(json.dumps(report, indent=1))
