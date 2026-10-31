#!/usr/bin/env python3
"""Diagnostic (paper review): why teacher text matching misses unperturbed test clips.
Same streams / verify windows / decoder / matcher as v5_teacher.py; unperturbed ('orig') positives only.
Decodes (a) the in-context verify window [clip-3 s, clip+0.64 s] and (b) the clip alone."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "v5")
import v5_teacher as vt  # noqa: E402
import evaluate_baselines as base  # noqa: E402
import teacher_textmatch as ttm  # noqa: E402
import parakeet_v12_train as v12  # noqa: E402


def main():
    v12.install_runtime_imports()
    import soundfile as sf
    long_wave, _ = sf.read(str(vt.AUDIO), dtype="int16")
    pos, labels, _, _ = vt.old_clips()
    orig = [i for i, lab in enumerate(labels) if "_orig" in Path(str(lab)).stem]
    streams, windows = base.test_streams(pos, [long_wave], 1)
    ctx = vt.verify_cut([streams[i] for i in orig], [windows[i] for i in orig])
    alone = [pos[i] for i in orig]
    text = ttm.TeacherText()
    t_ctx, t_alone = text.decode(ctx), text.decode(alone)
    h_ctx = np.array([ttm.hit(t) for t in t_ctx]); h_alone = np.array([ttm.hit(t) for t in t_alone])
    out = {"n": len(orig), "frr_context": float(1 - h_ctx.mean()), "frr_alone": float(1 - h_alone.mean()),
           "examples": [{"label": str(labels[i]), "context": a, "alone": b}
                        for i, a, b in list(zip(orig, t_ctx, t_alone))[:60]]}
    Path("logs/v5/teacher_diag.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n", "frr_context", "frr_alone")}))


if __name__ == "__main__":
    main()
