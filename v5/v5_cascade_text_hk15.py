"""v5: cascade with text-match verifier on hk15 -- stage 1 = the v5 Distilled Conformer (run 1) at
the 10 FA/h threshold of teacher_cascade.npz; the teacher greedy-decodes each trigger window
[-3 s, +0.64 s] and a false accept is a window whose text contains the phrase."""
import json
import sys
from pathlib import Path

import numpy as np

import evaluate_baselines as base
import parakeet_v12_train as v12
import streaming_protocol as protocol
import teacher_textmatch as ttm

v12.install_runtime_imports()
import soundfile as sf  # noqa: E402

d = np.load("logs/v5/teacher_cascade.npz")
tau1 = float(d["tau1"])
wave, rate = sf.read("multi_phone_threshold_analysis/hk15_24h/audio_16k_mono.wav", dtype="int16")
forward = base.student_forward(Path("checkpoints/student_L5_H192_M64_v5_prop/best_longaudio_recall.pt"))
probs = protocol.stream_probabilities(forward, wave)
frames = base.event_frames(probs, tau1)
before, after = int(base.VERIFY_BEFORE * rate), int(base.VERIFY_AFTER * rate)
pieces = []
for frame in frames:
    end = (frame // protocol.CHUNK_FRAMES + 1) * protocol.CHUNK_FRAMES * protocol.FRAME_SAMPLES
    pieces.append(wave[max(0, end - before):min(len(wave), end + after)])
del forward
hits = [ttm.hit(t) for t in ttm.TeacherText().decode(pieces)]
out = {"tau1": tau1, "stage1_events": len(frames), "text_fa_events": int(sum(hits)),
       "hours": len(wave) / rate / 3600}
Path("logs/v5/cascade_text_hk15.json").write_text(json.dumps(out, indent=1))
print(out)
