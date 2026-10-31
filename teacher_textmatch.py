#!/usr/bin/env python3
"""Stronger teacher baselines for the ICIIT revision: ASR text matching instead of folding.

The folded-posterior teacher of evaluate_baselines.py keeps only four of 1,024 SentencePiece
units, which a reviewer can call a strawman.  Here the teacher (Parakeet-CTC-0.6B-vi, full
context, offline) greedy-decodes text and a window counts as a detection when the normalised
transcript contains "năng khiếu ơi":
  * teacher_text    -- in-context test positives / near misses: window [clip start - 3.0 s,
                       clip end + 0.64 s]; FA: hk15 decoded in 4 s windows with a 2 s hop,
                       matches closer than 3 s merged into one event;
  * cascade_text    -- the student at thresholds giving 10 / 30 / 100 FA/h on hk15 wakes the
                       teacher, which decodes [trigger - 3.0 s, trigger + 0.64 s] and must
                       find the phrase (same verify window as evaluate_baselines.cascade).
Same test clips, embedding seeds and hk15 stream as the frozen protocol.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

import evaluate_baselines as base
import evaluate_wake_suite as suite
import incontext_protocol as ic
import parakeet_v12_train as v12
import streaming_protocol as protocol

RUN_DIR = Path("checkpoints/student_L5_H192_M64_v17_t6_lr1e-4_h0")
POS = Path("logs/v14_2/hk15_eval/positives_pos8.jsonl")
AUDIO = Path("multi_phone_threshold_analysis/hk15_24h/audio_16k_mono.wav")
PHRASE = "năng khiếu ơi"
WINDOW_S, HOP_S, MERGE_S = 4.0, 2.0, 3.0
BATCH = 32


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class TeacherText:
    def __init__(self):
        import nemo.collections.asr as nemo_asr
        import torch
        self.torch = torch
        self.device = torch.device("cuda")
        self.model = nemo_asr.models.ASRModel.restore_from(base.TEACHER_PATH,
                                                           map_location=self.device).eval()
        self.blank = self.model.decoder.num_classes_with_blank - 1

    def decode(self, pieces):
        torch = self.torch
        texts = []
        for start in range(0, len(pieces), BATCH):
            group = pieces[start:start + BATCH]
            width = max(len(p) for p in group)
            block = np.zeros((len(group), width), dtype=np.float32)
            for row, p in enumerate(group):
                block[row, :len(p)] = p.astype(np.float32) / 32768.0
            audio = torch.from_numpy(block).to(self.device)
            sizes = torch.tensor([len(p) for p in group], device=self.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats, flen = self.model.preprocessor(input_signal=audio, length=sizes)
                enc, elen = self.model.encoder(audio_signal=feats, length=flen)
                logits = self.model.decoder.decoder_layers(enc).transpose(1, 2).float()
            best = logits.argmax(-1).cpu().numpy()
            for row, n in enumerate(elen.cpu().numpy()):
                ids, prev = [], -1
                for token in best[row, :n]:
                    if token != prev and token != self.blank:
                        ids.append(int(token))
                    prev = token
                texts.append(self.model.tokenizer.ids_to_text(ids) if ids else "")
        return texts


def hit(text: str) -> bool:
    return PHRASE in normalise(text)


def clip_windows(streams, windows):
    rate = protocol.SAMPLE_RATE
    before, after = int(base.VERIFY_BEFORE * rate), int(base.VERIFY_AFTER * rate)
    return [s[max(0, a - before):min(len(s), b + after)] for s, (a, b) in zip(streams, windows)]


def merged_events(times):
    events, last = 0, -1e9
    for t in sorted(times):
        if t - last > MERGE_S:
            events += 1
        last = t
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path,
                        default=RUN_DIR / "validation_snapshots/step_008000.pt")
    parser.add_argument("--out", type=Path,
                        default=Path("logs/v17/paper_final/teacher_text/teacher_text.json"))
    args = parser.parse_args()
    v12.install_runtime_imports()
    import soundfile as sf
    pos_waves, _ = suite.ev.load_dev_positives(POS)
    labels = suite.positive_paths(POS)
    pos = [w for w, lab in zip(pos_waves, labels) if ic.role_of(lab) == "test"]
    neg_waves, neg_labels = suite.load_hard_negatives(RUN_DIR / "unified_manifest.jsonl")
    neg = [w for w, lab in zip(neg_waves, neg_labels) if ic.role_of(lab) == "test"]
    long_wave, rate = sf.read(str(AUDIO), dtype="int16")
    hours = len(long_wave) / rate / 3600
    pos_streams = base.test_streams(pos, [long_wave], 1)
    neg_streams = base.test_streams(neg, [long_wave], 2)

    # student (stage 1) scores, exactly as the protocol
    forward = base.student_forward(args.student)
    s_pos = ic.window_best(forward, *pos_streams)
    s_neg = ic.window_best(forward, *neg_streams)
    finite = s_pos[np.isfinite(s_pos)]
    s_thr = np.linspace(float(finite.min()), float(finite.max()), 256)
    s_probs, s_events = base.long_events(forward, long_wave, s_thr)
    del forward

    teacher = TeacherText()
    t_pos = np.array([hit(t) for t in teacher.decode(clip_windows(*pos_streams))])
    t_neg = np.array([hit(t) for t in teacher.decode(clip_windows(*neg_streams))])
    win, hop = int(WINDOW_S * rate), int(HOP_S * rate)
    starts = list(range(0, max(1, len(long_wave) - win + 1), hop))
    long_texts = []
    for i in range(0, len(starts), 2048):
        long_texts += teacher.decode([long_wave[s:s + win] for s in starts[i:i + 2048]])
    match_times = [(s + win / 2) / rate for s, t in zip(starts, long_texts) if hit(t)]
    fa_examples = [t for t in long_texts if hit(t)][:40]
    t_events = merged_events(match_times)
    report = {"phrase": PHRASE, "hours": hours, "n_pos": len(pos), "n_neg": len(neg),
              "teacher_text": {"frr": float(1 - t_pos.mean()), "near_miss_fa": float(t_neg.mean()),
                               "fa_events": t_events, "fa_per_hour": t_events / hours,
                               "window_s": WINDOW_S, "hop_s": HOP_S,
                               "fa_examples": fa_examples},
              "cascade_text": []}
    per_hour = np.asarray(s_events) / hours
    before, after = int(base.VERIFY_BEFORE * rate), int(base.VERIFY_AFTER * rate)
    for target in base.STAGE1_FA_PER_HOUR:
        index = int(np.argmin(np.abs(per_hour - target)))
        tau1 = float(s_thr[index])
        frames = base.event_frames(s_probs, tau1)
        pieces = []
        for frame in frames:
            end = (frame // protocol.CHUNK_FRAMES + 1) * protocol.CHUNK_FRAMES * protocol.FRAME_SAMPLES
            pieces.append(long_wave[max(0, end - before):min(len(long_wave), end + after)])
        verified = [hit(t) for t in teacher.decode(pieces)] if pieces else []
        fa = int(sum(verified))
        report["cascade_text"].append({
            "stage1_fa_per_hour": float(per_hour[index]), "stage1_events": len(frames),
            "teacher_duty_cycle": len(frames) * (base.VERIFY_BEFORE + base.VERIFY_AFTER) / (hours * 3600),
            "frr": float(1 - ((s_pos >= tau1) & t_pos).mean()),
            "near_miss_fa": float(((s_neg >= tau1) & t_neg).mean()),
            "fa_events": fa, "fa_per_hour": fa / hours})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "teacher_text"}, indent=1,
                     ensure_ascii=False))
    print(json.dumps({k: v for k, v in report["teacher_text"].items() if k != "fa_examples"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
