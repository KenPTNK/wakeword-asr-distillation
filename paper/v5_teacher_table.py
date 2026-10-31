"""v5: teacher (folded / text match) and cascade rows on the v5 test speakers, false accepts on hk15
(v5/teacher_cascade.npz, v5/cascade_text_hk15.json; teacher text alone has 0 hk15 events, a clip-
independent result of teacher_textmatch.py) plus the sealed 100 h set (v5/longfa/teacher.npz).
Also the first-stage recall of the Distilled Conformer (run 1) at 10/30/100 FA/h on hk15.
usage: python v5_teacher_table.py > v5/v5_teacher_stats.json"""
import json
from pathlib import Path

import numpy as np

import v4_tables as vt

ROOT = Path(__file__).resolve().parent / "v5"
GRID = 256


def table(pos, neg, thr, events, hours, cap):
    per_hour = events / hours
    out = {}
    for limit in vt.FA_TARGETS:
        if cap:
            i = vt.constrained(pos, neg, thr, per_hour, limit)
        else:
            ok = per_hour <= limit
            rec = np.array([(pos >= t).mean() for t in thr])
            i = int(np.flatnonzero(ok)[np.argmax(rec[ok])]) if ok.any() else None
        out[f"{limit:g}"] = None if i is None else {
            "frr": float((pos < thr[i]).mean()), "near_miss_fa": float((neg >= thr[i]).mean()),
            "fa_events": int(events[i])}
    return out


d = np.load(ROOT / "teacher_cascade.npz")
f = np.load(ROOT / "longfa" / "teacher.npz")
ctext = json.loads((ROOT / "cascade_text_hk15.json").read_text())
hours = float(d["hours"]) + float(f["hours"].sum())
tau1 = float(d["tau1"])
assert abs(tau1 - float(f["tau1"])) < 1e-9
s_pos, s_neg = d["test_s_pos"], d["test_s_neg"]
# teacher folded
tf_events = d["tf_events"].astype(float) + f["tf_events"].sum(0)
teacher = table(d["test_tf_pos"], d["test_tf_neg"], d["tf_thr"], tf_events, hours, cap=True)
# cascade, folded verifier
tv_pos, tv_neg = d["test_tv_pos"], d["test_tv_neg"]
grid = np.linspace(float(np.min(tv_pos)), float(np.max(tv_pos)), GRID)
t_long = np.concatenate([d["t_long"], f["t_long"]])
c_events = np.array([float((t_long >= g).sum()) for g in grid])
c_pos = np.where(s_pos >= tau1, tv_pos, -np.inf)
c_neg = np.where(s_neg >= tau1, tv_neg, -np.inf)
# text match
tt_pos, tt_neg = d["test_tt_pos"], d["test_tt_neg"]
report = {
    "hours": hours, "tau1": tau1,
    "teacher_folded_nm10": teacher,
    "cascade_folded": table(c_pos, c_neg, grid, c_events, hours, cap=False),
    "cascade_folded_nm10": table(c_pos, c_neg, grid, c_events, hours, cap=True),
    "teacher_text": {"frr": float(1 - tt_pos.mean()), "near_miss_fa": float(tt_neg.mean()),
                     "fa_events": int(f["text_events"].sum()), "fa_per_hour": float(f["text_events"].sum()) / hours},
    "cascade_text": {"frr": float(1 - ((s_pos >= tau1) & tt_pos).mean()),
                     "near_miss_fa": float(((s_neg >= tau1) & tt_neg).mean()),
                     "fa_events": int(ctext["text_fa_events"] + f["cascade_hits"].sum()),
                     "fa_per_hour": float(ctext["text_fa_events"] + f["cascade_hits"].sum()) / hours},
    "stage1_triggers_sealed": int(len(f["t_long"])), "stage1_triggers_hk15": int(len(d["t_long"])),
}
p = np.load(ROOT / "scores" / "v5_prop.npz")
per_hour = p["events"] / float(p["hours"])
report["stage1_recall"] = {}
for target in (10.0, 30.0, 100.0):
    i = int(np.argmin(np.abs(per_hour - target)))
    report["stage1_recall"][f"{target:g}"] = {"fa_per_hour": float(per_hour[i]),
                                               "recall": float((p["pos_best"] >= p["thresholds"][i]).mean())}
print(json.dumps(report, indent=1))
