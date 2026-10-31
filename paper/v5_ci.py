"""v5: FRR confidence intervals of the Distilled Conformer (first run) at the 0.1 FA/h operating point
on hk15 + sealed set, with recordings identified by (speaker, text, index) -- analyze_scores.utterance_key
drops the text and merges different sentences that share an index.
  within: resample recordings within each test speaker (speakers fixed)
  two_level: resample speakers, then recordings within each drawn speaker
usage: python v5_ci.py > v5/v5_ci.json"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "v5"
BOOT = 10000
RNG = np.random.default_rng(0)


def recording_key(label):
    name = Path(label).stem.split("_")
    return name[0], name[1] + "#" + name[2]


def clusters_by_speaker(labels):
    groups = {}
    for i, key in enumerate(map(recording_key, labels)):
        groups.setdefault(key, []).append(i)
    by_speaker = {}
    for (spk, _), idx in groups.items():
        by_speaker.setdefault(spk, []).append(np.array(idx))
    return by_speaker


def boot(miss, by_speaker, two_level):
    speakers = list(by_speaker)
    stats = []
    for _ in range(BOOT):
        drawn = RNG.choice(speakers, len(speakers)) if two_level else speakers
        chosen = []
        for spk in drawn:
            clusters = by_speaker[spk]
            chosen.extend(clusters[j] for j in RNG.integers(0, len(clusters), len(clusters)))
        stats.append(miss[np.concatenate(chosen)].mean())
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


stats = json.loads((ROOT / "v5_stats.json").read_text(encoding="utf-8"))
out = {}
for run in ("v5_prop", "v5_prop_s2027", "v5_prop_s2028"):
    d = np.load(ROOT / "scores" / f"{run}.npz")
    labels = [str(x) for x in d["pos_label"]]
    by_speaker = clusters_by_speaker(labels)
    thr = stats[run]["long"]["0.1"]["threshold"]
    miss = d["pos_best"] < thr
    out[run] = {"frr": float(miss.mean()), "threshold": thr,
                "n_recordings": sum(len(c) for c in by_speaker.values()), "n_speakers": len(by_speaker),
                "ci95_within_speakers": boot(miss, by_speaker, False),
                "ci95_two_level": boot(miss, by_speaker, True)}
print(json.dumps(out, indent=1))
