"""Clustered statistics from per-clip test scores (dump_scores.py npz files).

For each npz: FRR at the nominal FA/h levels (same rule as evaluate_baselines.frr_table),
the FA events behind each level, exact Poisson CIs for FA/h, per-speaker FRR, and a bootstrap
CI that resamples ORIGINAL utterances (a recording and all its perturbed copies move together)
within each test speaker.  Also FRR/hk15-FA at any extra threshold stored in the npz.

usage: python analyze_scores.py scores/*.npz > stats.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

FA_TARGETS = (0.1, 0.5, 1.0, 2.0)
BOOT = 2000
RNG = np.random.default_rng(20260926)


def poisson_ci(k: int, hours: float):
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    hi = chi2.ppf(0.975, 2 * (k + 1)) / 2
    return lo / hours, hi / hours


def utterance_key(label: str):
    name = Path(label).stem.split("_")
    return name[0], name[2]                       # (speaker, utterance index)


def operating_index(pos, thresholds, per_hour, limit):
    allowed = per_hour <= limit
    if not allowed.any():
        return None
    recalls = np.array([(pos >= t).mean() for t in thresholds])
    return int(np.flatnonzero(allowed)[np.argmax(recalls[allowed])])


def cluster_boot(miss, speakers, utts):
    """95 % CI of the FRR, resampling original utterances within speakers."""
    groups = {}
    for i, key in enumerate(zip(speakers, utts)):
        groups.setdefault(key, []).append(i)
    by_speaker = {}
    for (spk, _), idx in groups.items():
        by_speaker.setdefault(spk, []).append(np.array(idx))
    stats = []
    for _ in range(BOOT):
        chosen = []
        for clusters in by_speaker.values():
            pick = RNG.integers(0, len(clusters), len(clusters))
            chosen.extend(clusters[j] for j in pick)
        idx = np.concatenate(chosen)
        stats.append(miss[idx].mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def analyse(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    pos, neg = d["pos_best"], d["neg_best"]
    thresholds, events, hours = d["thresholds"], d["events"], float(d["hours"])
    per_hour = events / hours
    keys = [utterance_key(str(x)) for x in d["pos_label"]]
    speakers = np.array([k[0] for k in keys])
    utts = np.array([k[1] for k in keys])
    fam = d["pos_family"]
    out = {"n_pos": int(len(pos)), "n_originals": len(set(keys)), "n_neg": int(len(neg)),
           "hours": hours, "levels": {}}
    for limit in FA_TARGETS:
        i = operating_index(pos, thresholds, per_hour, limit)
        if i is None:
            out["levels"][f"{limit:g}"] = None
            continue
        t = thresholds[i]
        miss = pos < t
        orig = fam == "orig"
        out["levels"][f"{limit:g}"] = {
            "threshold": float(t), "fa_events": int(events[i]), "fa_per_hour": float(per_hour[i]),
            "fa_per_hour_ci95": poisson_ci(int(events[i]), hours),
            "frr": float(miss.mean()), "frr_ci95_cluster": cluster_boot(miss, speakers, utts),
            "frr_orig": float(miss[orig].mean()), "n_orig": int(orig.sum()),
            "near_miss_fa": float((neg >= t).mean()),
            "frr_by_speaker": {s: float(miss[speakers == s].mean()) for s in sorted(set(speakers))},
        }
    extras = {}
    grid_n = 256
    if len(thresholds) > grid_n:
        # thresholds that are not on the linspace grid were added with --extra-thresholds
        lin = np.linspace(thresholds.min(), thresholds.max(), grid_n)
        for i, t in enumerate(thresholds):
            if np.min(np.abs(lin - t)) > 1e-9:
                miss = pos < t
                extras[f"{t:.4f}"] = {
                    "fa_events": int(events[i]), "fa_per_hour": float(per_hour[i]),
                    "fa_per_hour_ci95": poisson_ci(int(events[i]), hours),
                    "frr": float(miss.mean()), "frr_ci95_cluster": cluster_boot(miss, speakers, utts),
                    "near_miss_fa": float((neg >= t).mean()),
                    "frr_by_speaker": {s: float(miss[speakers == s].mean())
                                       for s in sorted(set(speakers))}}
    out["extra_thresholds"] = extras
    return out


if __name__ == "__main__":
    report = {Path(p).stem: analyse(Path(p)) for p in sys.argv[1:]}
    print(json.dumps(report, indent=1))
