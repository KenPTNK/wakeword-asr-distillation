"""Fig. 2 (v5): DET curves (FRR vs FA/h on hk15 + the sealed multi-domain 100 h set, 111.5 h) under
the deployment constraint NM <= 10 %, on the 10 test speakers of the v5 split.

Computed from the per-clip test scores in _results/v5/scores (v5_dump_scores.py) and the sealed-set
events in _results/v5/longfa (v5_longfa.py).  Only thresholds whose near-miss accept rate is <= 10 % are used, the
same constraint as the frozen selection rule, so a detector that fires on partial matches cannot
look good.  Teacher text matching is a single operating point (v5_teacher_table.py,
v5_teacher_stats.json).  The grey
band marks FA rates that rest on <= 4 false-accept events.  Run: python figures/make_fig2.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent / "v5"
SCORES = ROOT / "scores"
LONG = ROOT / "longfa"
NEAR_MISS_CAP = 0.10


def load(stem):
    """Per-clip scores with false accepts counted on hk15 + sealed (same thresholds)."""
    d = np.load(SCORES / f"{stem}.npz")
    f = np.load(LONG / f"{stem}.npz")
    events = d["events"] + f["events"].sum(0)
    hours = float(d["hours"]) + float(f["hours"].sum())
    return d["pos_best"], d["neg_best"], d["thresholds"], events, hours


def det(stem):
    """(FA/h, FRR) pairs over thresholds that keep near-miss accepts <= the cap."""
    pos, neg, thr, ev, hours = load(stem)
    rows = sorted((e / hours, (pos < t).mean()) for t, e in zip(thr, ev)
                  if e > 0 and (neg >= t).mean() <= NEAR_MISS_CAP)
    best, pareto = 1.0, []
    for fa, frr in rows:                      # lower envelope: best FRR reachable at <= fa
        best = min(best, frr)
        pareto.append((fa, best))
    return np.array(pareto).T if pareto else (np.array([]), np.array([]))


def main():
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.33, 2.25))
    ax.axvspan(0.003, 4.5 / 111.5, color="0.92", linewidth=0, zorder=0)
    ax.text(0.0034, 0.03, "$\\leq$4 FA events", fontsize=5.5, color="0.35")
    # Distinct line styles AND markers so the figure also reads in greyscale (ACM template).
    # Our detector: one curve (first run, designated in advance), drawn first and thickest.
    fa, frr = det("v5_prop")
    ax.step(fa, frr, "-", where="post", color="#D55E00", linewidth=2.2, zorder=5,
            label="Distilled Conformer (ours)")
    ax.annotate("near-miss cap\nreached", xy=(fa[-1], frr[-1]), xytext=(0.16, 0.06), fontsize=5.5,
                color="#D55E00", arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.6))
    curves = [
        ("bcresnet", "BC-ResNet-8", "#0072B2", "-.", "h"),
        ("oww_rnn", "openWakeWord", "#882255", "--", "P"),
        ("dscnn_w64", "DS-CNN 137k", "#E69F00", ":", "o"),
    ]
    for stem, label, color, style, marker in curves:
        fa, frr = det(stem)
        if len(fa):
            ax.step(fa, frr, style, where="post", color=color, linewidth=1.0, label=label,
                    marker=marker, markersize=3, markevery=max(1, len(fa) // 5))
        else:                                  # only zero-FA thresholds meet the cap
            pos, neg, thr, _, _ = load(stem)
            ok = [(pos < t).mean() for t in thr if (neg >= t).mean() <= NEAR_MISS_CAP]
            ax.plot([0.0035], [min(ok)], "s", color=color, markersize=4,
                    label=f"{label} (0 FA)")
    t = json.loads((ROOT / "v5_teacher_stats.json").read_text(encoding="utf-8"))
    ax.plot([0.0035], [t["teacher_text"]["frr"]], "*", color="0.3", markersize=5.5,
            label="Teacher 600M, text match (0 FA)")
    ax.set_xscale("log")
    ax.set_xlim(0.003, 4)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0.01, 0.1, 0.5, 1, 2])
    ax.set_xticklabels(["0.01", "0.1", "0.5", "1", "2"])
    ax.set_xlabel("False accepts per hour on 111.5 h of audio (FA/h)")
    ax.set_ylabel("False rejection rate (FRR)")
    ax.grid(True, which="major", linewidth=0.3, alpha=0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.01), fontsize=5.6, frameon=False,
              ncol=2, handlelength=2.3, columnspacing=0.8, borderaxespad=0.1)
    fig.tight_layout(pad=0.2)
    out = Path(__file__).resolve().parent / "fig2_det.pdf"
    fig.savefig(out)
    print("->", out)


if __name__ == "__main__":
    main()
