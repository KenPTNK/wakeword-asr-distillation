#!/usr/bin/env python3
"""Baselines on the frozen test protocol (ctctknk-docs/33 §3j-3k, §3m).

Same test set and embedding seeds as ``reselect_in_context.py``: test-speaker
positives and near misses embedded in hk15_24h (20.48 s context, -20 dB), FA/h
streamed on hk15_24h.  Detectors:

* ``student``  — a phase-2 checkpoint, deployment context [64, 7] (reference);
* ``teacher``  — Parakeet-CTC-0.6B-vi, full context (offline upper bound).  Its
  1,025 outputs are folded to the student's five classes: the four wake pieces
  (▁năng ▁kh iếu ▁ơi) keep their probability, everything else goes to blank;
  the same keyword automaton then scores it;
* ``cascade``  — Siri-style two stages: the student at a low threshold wakes the
  teacher, which verifies ``VERIFY_BEFORE`` s before .. ``VERIFY_AFTER`` s after
  each stage-1 trigger; a detection needs both.  Reported with the teacher's
  duty cycle (fraction of audio time it runs).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import evaluate_long_audio_streaming as longaudio
import evaluate_wake_suite as suite
import incontext_protocol as ic
import parakeet_v12_train as v12
import streaming_protocol as protocol

TEACHER_PATH = ("/home/ml4u/.cache/huggingface/hub/models--nvidia--parakeet-ctc-0.6b-Vietnamese/"
                "snapshots/b0493142b49458810324e3db8be9e8e07b4ebc17/parakeet-ctc-0.6b-vi.nemo")
TEACHER_WAKE_IDS = (435, 20, 558, 645)          # ▁năng ▁kh iếu ▁ơi in the teacher vocabulary
TEST_SEED = 20260929                            # reselect_in_context SEED + 1 (test streams)
FA_TARGETS = (0.1, 0.5, 1.0, 2.0)
STAGE1_FA_PER_HOUR = (10.0, 30.0, 100.0)        # stage-1 operating points of the cascade
VERIFY_BEFORE = 3.0
VERIFY_AFTER = 0.64
BLOCK = 1000


def wilson(successes: int, total: int, z: float = 1.96):
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class TeacherForward:
    """``forward(block, lengths) -> (5-class log-probs, lengths)`` for the teacher."""

    def __init__(self, path: str = TEACHER_PATH):
        import nemo.collections.asr as nemo_asr
        import torch
        self.torch = torch
        self.device = torch.device("cuda")
        self.model = nemo_asr.models.ASRModel.restore_from(path, map_location=self.device).eval()
        self.ids = torch.tensor(TEACHER_WAKE_IDS, device=self.device)

    def __call__(self, block: np.ndarray, lengths):
        torch = self.torch
        audio = torch.from_numpy(np.ascontiguousarray(block)).to(self.device)
        sizes = torch.tensor(list(lengths), device=self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features, feature_lengths = self.model.preprocessor(input_signal=audio, length=sizes)
            encoded, encoded_lengths = self.model.encoder(audio_signal=features,
                                                          length=feature_lengths)
            raw = self.model.decoder.decoder_layers(encoded).transpose(1, 2).float()
        probs = torch.softmax(raw, dim=-1)
        wake = probs.index_select(-1, self.ids)
        rest = (1.0 - wake.sum(-1, keepdim=True)).clamp_min(1e-8)
        folded = torch.cat([wake, rest], dim=-1)          # blank = class 4 = CTC_BLANK_ID
        return torch.log(folded.clamp_min(1e-8)), encoded_lengths


def fake_int8(model) -> int:
    """Per-output-channel symmetric int8 weights (quantise then dequantise in place).

    Every weight with >= 2 dims (linear, conv, embedding); biases and norms stay
    float, as in a typical int8 deployment.  Returns the number of tensors changed.
    """
    import torch
    changed = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim < 2:
                continue
            flat = parameter.reshape(parameter.shape[0], -1)
            scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
            parameter.copy_((torch.round(flat / scale).clamp(-127, 127) * scale)
                            .reshape(parameter.shape))
            changed += 1
    return changed


def student_forward(checkpoint: Path, int8: bool = False):
    args = argparse.Namespace(segment_seconds=protocol.SEGMENT_SECONDS,
                              left_context=protocol.LEFT_CONTEXT_SECONDS,
                              att_left_frames=protocol.ATT_CONTEXT[0], batch_utterances=64)
    runner = longaudio.Runner(checkpoint, args)             # Runner fixes [64, 7]
    if int8:
        print(f"fake int8: {fake_int8(runner.model)} weight tensors quantised", flush=True)
    return runner._forward


def test_streams(clips, contexts, salt: int):
    """The embedded test streams exactly as incontext_protocol.measure builds them."""
    streams, windows = [], []
    for index, start in enumerate(range(0, len(clips), BLOCK)):
        s, w = ic.embed(clips[start:start + BLOCK], contexts, TEST_SEED * 1000 + salt * 100 + index)
        streams.extend(s)
        windows.extend(w)
    return streams, windows


def long_events(forward, wave, thresholds):
    probabilities = protocol.stream_probabilities(forward, wave)
    return probabilities, protocol.batched_events(probabilities[None], thresholds)[0]


def event_frames(probabilities: np.ndarray, threshold: float):
    """Frames at which the reset automaton fires (single threshold)."""
    classes = probabilities.shape[1]
    matrices, states = protocol.transition_matrices(classes)
    matched_start = len(tuple(v12.WAKE_SEQUENCE)) * classes
    initial = np.zeros(states)
    initial[int(v12.CTC_BLANK_ID)] = 1.0
    state = initial.copy()
    tiny = np.finfo(np.float64).tiny
    skip = int(round(protocol.REFRACTORY_SECONDS * protocol.FRAMES_PER_SECOND))
    blocked, fired = 0, []
    for frame in range(probabilities.shape[0]):
        state = state @ np.einsum("c,csu->su", probabilities[frame].astype(np.float64), matrices)
        state /= max(state.sum(), tiny)
        score = (math.log(max(state[matched_start:].sum(), tiny))
                 - math.log(max(state[:matched_start].sum(), tiny)))
        if blocked <= 0 and score >= threshold:
            fired.append(frame)
            state = initial.copy()
            blocked = skip
        blocked -= 1
    return fired


def positive_augmentations(manifest: Path) -> list[str]:
    """Augmentation family per positive, in ``load_dev_positives`` order.

    The evaluation positives are 540 original recordings plus synthetic variants
    (speed, pitch, noise, overlap, volume); reporting only the pooled FRR hides
    that 95 % of the clips are synthetic.
    """
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (record["source_dataset"] == "WakewordV12" and record["split"] == "validation"
                    and record["is_positive"]):
                family = (record.get("augmentation") or "orig").split("_")[0]
                family = "overlap" if family.startswith("overlap") else family
                rows.append((record["audio_path"], record.get("audio_row") or 0, family))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows]


def by_family(pos_best, families, table):
    """FRR per augmentation family at each operating threshold of ``table``."""
    out = {}
    families = np.asarray(families)
    for limit, row in table.items():
        if row is None:
            continue
        hits = pos_best >= row["threshold"]
        out[limit] = {}
        for family in sorted(set(families.tolist())):
            chosen = hits[families == family]
            out[limit][family] = {"n": int(chosen.size), "frr": 1 - float(chosen.mean()),
                                  "frr_ci95": wilson(int((~chosen).sum()), int(chosen.size))}
    return out


def frr_table(pos_best, neg_best, thresholds, per_hour, events):
    out = {}
    recalls = np.array([(pos_best >= t).mean() for t in thresholds])
    for limit in FA_TARGETS:
        allowed = per_hour <= limit
        if not allowed.any():
            out[f"{limit:g}"] = None
            continue
        index = int(np.flatnonzero(allowed)[np.argmax(recalls[allowed])])
        hits = pos_best >= thresholds[index]
        near = neg_best >= thresholds[index]
        out[f"{limit:g}"] = {"threshold": float(thresholds[index]),
                             "fa_per_hour": float(per_hour[index]),
                             "fa_events": int(events[index]),
                             "frr": 1 - float(hits.mean()),
                             "frr_ci95": wilson(int((~hits).sum()), len(hits)),
                             "near_miss_fa": float(near.mean())}
    return out


def single_detector(name, forward, long_wave, hours, streams, count):
    (pos_streams, pos_windows), (neg_streams, neg_windows) = streams
    pos_best = ic.window_best(forward, pos_streams, pos_windows)
    neg_best = ic.window_best(forward, neg_streams, neg_windows)
    finite = pos_best[np.isfinite(pos_best)]
    thresholds = np.linspace(float(finite.min()), float(finite.max()), count)
    probabilities, events = long_events(forward, long_wave, thresholds)
    result = {"detector": name,
              "frr_at_fa": frr_table(pos_best, neg_best, thresholds, events / hours, events)}
    return result, pos_best, neg_best, probabilities, thresholds, events


def verify_windows(teacher, pieces):
    return (ic.window_best(teacher, pieces, [(0, len(p)) for p in pieces])
            if pieces else np.zeros(0))


def cascade(teacher, long_wave, hours, streams, stage1, count):
    """Stage 1 = the student at thresholds giving ``STAGE1_FA_PER_HOUR`` on hk15."""
    _, s_pos, s_neg, s_probs, s_thr, s_events = stage1
    rate = protocol.SAMPLE_RATE
    before, after = int(VERIFY_BEFORE * rate), int(VERIFY_AFTER * rate)

    def around_clip(stream_set):
        cut = []
        for stream, (start, end) in zip(*stream_set):
            cut.append(stream[max(0, start - before):min(len(stream), end + after)])
        return cut

    t_pos = verify_windows(teacher, around_clip(streams[0]))
    t_neg = verify_windows(teacher, around_clip(streams[1]))
    per_hour = s_events / hours
    rows = []
    for target in STAGE1_FA_PER_HOUR:
        index = int(np.argmin(np.abs(per_hour - target)))
        tau1 = float(s_thr[index])
        frames = event_frames(s_probs, tau1)
        hop = protocol.FRAME_SAMPLES
        pieces = []
        for frame in frames:
            end = (frame // protocol.CHUNK_FRAMES + 1) * protocol.CHUNK_FRAMES * hop
            pieces.append(long_wave[max(0, end - before):min(len(long_wave), end + after)])
        t_long = verify_windows(teacher, pieces)
        grid = np.linspace(float(np.min(t_pos)), float(np.max(t_pos)), count)
        fa_events = np.array([int((t_long >= g).sum()) for g in grid])
        pos_best = np.where(s_pos >= tau1, t_pos, -np.inf)
        neg_best = np.where(s_neg >= tau1, t_neg, -np.inf)
        rows.append({"stage1_threshold": tau1, "stage1_fa_per_hour": float(per_hour[index]),
                     "stage1_events": len(frames),
                     "teacher_duty_cycle": len(frames) * (VERIFY_BEFORE + VERIFY_AFTER)
                                           / (hours * 3600),
                     "stage1_recall": float((s_pos >= tau1).mean()),
                     "stage1_near_miss_fa": float((s_neg >= tau1).mean()),
                     "frr_at_fa": frr_table(pos_best, neg_best, grid, fa_events / hours,
                                            fa_events)})
        print(json.dumps(rows[-1], default=float), flush=True)
    return {"detector": "cascade", "verify_window_s": [VERIFY_BEFORE, VERIFY_AFTER],
            "operating_points": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="for the near-miss manifest")
    parser.add_argument("--positive-manifest", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--thresholds", type=int, default=256)
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--fake-int8", action="store_true",
                        help="evaluate the student with per-channel int8 weights")
    parser.add_argument("--det-out", type=Path, default=None,
                        help="write the student's hk15 DET grid (threshold, fa_per_hour, "
                             "fa_events) as CSV, for evaluate_in_context.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    v12.install_runtime_imports()
    import soundfile as sf
    args.out.mkdir(parents=True, exist_ok=True)
    pos_waves, _ = suite.ev.load_dev_positives(args.positive_manifest)
    labels = suite.positive_paths(args.positive_manifest)
    pos = [w for w, lab in zip(pos_waves, labels) if ic.role_of(lab) == "test"]
    families = [f for f, lab in zip(positive_augmentations(args.positive_manifest), labels)
                if ic.role_of(lab) == "test"]
    neg_waves, neg_labels = suite.load_hard_negatives(args.run_dir / "unified_manifest.jsonl")
    neg = [w for w, lab in zip(neg_waves, neg_labels) if ic.role_of(lab) == "test"]
    long_wave, rate = sf.read(str(args.audio), dtype="int16")
    hours = len(long_wave) / rate / 3600
    print(f"test: {len(pos):,} positives, {len(neg):,} near misses, {hours:.2f} h", flush=True)
    streams = (test_streams(pos, [long_wave], 1), test_streams(neg, [long_wave], 2))
    report = {"protocol": ic.PROTOCOL_NAME, "test_speakers": ic.TEST_SPEAKERS, "detectors": []}
    stage1 = single_detector("student_int8" if args.fake_int8 else "student",
                             student_forward(args.student, args.fake_int8), long_wave, hours,
                             streams, args.thresholds)
    stage1[0]["by_augmentation"] = by_family(stage1[1], families, stage1[0]["frr_at_fa"])
    report["detectors"].append({**stage1[0], "checkpoint": str(args.student)})
    if args.det_out is not None:
        thresholds, events = stage1[4], stage1[5]
        with args.det_out.open("w", encoding="utf-8") as handle:
            handle.write("threshold,fa_per_hour,fa_events\n")
            for threshold, count in zip(thresholds, events):
                handle.write(f"{threshold:.5f},{count / hours:.4f},{int(count)}\n")
    print("student", json.dumps(stage1[0]["frr_at_fa"], default=float), flush=True)
    if not args.skip_teacher:
        teacher = TeacherForward()
        alone = single_detector("teacher", teacher, long_wave, hours, streams, args.thresholds)
        alone[0]["by_augmentation"] = by_family(alone[1], families, alone[0]["frr_at_fa"])
        report["detectors"].append(alone[0])
        print("teacher", json.dumps(alone[0]["frr_at_fa"], default=float), flush=True)
        report["detectors"].append(cascade(teacher, long_wave, hours, streams, stage1,
                                           args.thresholds))
    name = args.student.parent.name if args.student.parent.name != "validation_snapshots" \
        else args.student.parent.parent.name + "_" + args.student.stem
    name += "_int8" if args.fake_int8 else ""
    (args.out / f"baselines_{name}.json").write_text(json.dumps(report, indent=1, default=float),
                                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
