#!/usr/bin/env python3
"""One streaming protocol for every place that scores the student on long audio.

Before this module the trainer, ``evaluate_long_audio_streaming.py`` and the
channel meter each simulated streaming their own way, and all of them shared
one bug (``ctctknk-docs/25`` §5.2): a segment was cut at its nominal end, so
the last chunk of every segment lost its 7-frame look-ahead.  On the V14.2 bin6
checkpoint that fabricated 140 of 320 false-accept events, all in the last
0.64 s of each 10 s cold-start segment, and once every 120 s in streaming mode.

The protocol (``ctctknk-docs/26`` phase 1):

* **Attention context ``[64, 7]``** — 64 frames (5.12 s) of left context, the
  deployment setting (``ctctknk-docs/24`` §3.1: no recall lost).  Training
  keeps ``[-1, 7]``; only scoring switches, via ``limited_context``.
* **Exact segmenting.**  With chunked attention, causal convolutions and causal
  subsampling, a frame depends on at most 5 layers x (64 + 7 + 8) frames of
  history (~32 s) and on audio up to the end of its own 0.64 s chunk.  So each
  segment starts on the stream's chunk grid, carries 32 s of history, and
  carries 1.28 s of audio *after* its end; both margins are computed and
  dropped.  The result equals one forward pass over the whole file
  (``validate_streaming_protocol.py`` checks it numerically).
* **Automaton reset after each detection**, refractory 1 s — unchanged.
* **Cold start as its own number**: false accepts in the first 3 s after the
  encoder restarts, per restart, with and without a 2 s pre-roll (audio the
  device runs through the encoder before it starts scoring,
  ``ctctknk-docs/25`` §5.3).  Restarts are placed every 10 s of the corpus.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

import parakeet_v12_train as v12

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280                # 8 x 10 ms mel hop = one encoder frame (80 ms)
CHUNK_FRAMES = 8                     # 0.64 s attention chunk
CHUNK_SAMPLES = FRAME_SAMPLES * CHUNK_FRAMES
FRAMES_PER_SECOND = SAMPLE_RATE / FRAME_SAMPLES

ATT_CONTEXT = (64, 7)                # deployment: 5.12 s left, chunk look-ahead
SEGMENT_SECONDS = 128.0              # 200 chunks
LEFT_CONTEXT_SECONDS = 32.0          # >= 5 layers x (64 + 7 + 8) frames
LOOKAHEAD_PAD_SECONDS = 1.28         # two chunks after the segment end
REFRACTORY_SECONDS = 1.0

COLD_WINDOW_SECONDS = 3.0
COLD_SPACING_SECONDS = 10.0
PREROLL_SECONDS = 2.0
COLD_PREROLLS = (0.0, PREROLL_SECONDS)

PROTOCOL_NAME = "stream64x7_exactseg_v1"


def _samples(seconds: float, unit: int) -> int:
    """Seconds -> samples, refusing values off the ``unit`` grid."""
    samples = int(round(seconds * SAMPLE_RATE))
    if samples % unit:
        raise ValueError(f"{seconds} s is not a multiple of {unit / SAMPLE_RATE:g} s")
    return samples


def describe() -> dict:
    return {"name": PROTOCOL_NAME, "att_context": list(ATT_CONTEXT),
            "segment_seconds": SEGMENT_SECONDS, "left_context_seconds": LEFT_CONTEXT_SECONDS,
            "lookahead_pad_seconds": LOOKAHEAD_PAD_SECONDS,
            "refractory_seconds": REFRACTORY_SECONDS,
            "cold_window_seconds": COLD_WINDOW_SECONDS,
            "cold_spacing_seconds": COLD_SPACING_SECONDS,
            "cold_prerolls_seconds": list(COLD_PREROLLS)}


# ------------------------------------------------------------------ context

@contextmanager
def limited_context(model, att_context=ATT_CONTEXT):
    """Score with ``att_context`` and restore the training context afterwards.

    NeMo's forward reads ``encoder.att_context_size`` whenever it is not
    sampling among several training contexts; setting the attribute directly
    avoids ``set_default_att_context_size``'s warning on every validation.
    """
    encoder = model.encoder
    previous = list(encoder.att_context_size)
    encoder.att_context_size = list(att_context)
    encoder.setup_streaming_params()
    try:
        yield model
    finally:
        encoder.att_context_size = previous
        encoder.setup_streaming_params()


# ------------------------------------------------------------------ streaming

def stream_probabilities(forward, wave: np.ndarray,
                         segment_seconds: float = SEGMENT_SECONDS,
                         left_seconds: float = LEFT_CONTEXT_SECONDS,
                         pad_seconds: float = LOOKAHEAD_PAD_SECONDS) -> np.ndarray:
    """Frame posteriors of one long int16 file, equal to a single forward pass.

    ``forward(block[None], [length]) -> (logits, lengths)`` as in
    ``evaluate_long_audio_streaming.Runner._forward``.
    """
    import torch

    segment = _samples(segment_seconds, CHUNK_SAMPLES)
    left = _samples(left_seconds, CHUNK_SAMPLES)
    pad = _samples(pad_seconds, FRAME_SAMPLES)
    pieces = []
    position = 0
    while position < len(wave):
        start = max(0, position - left)
        end = min(len(wave), position + segment + pad)
        block = wave[start:end].astype(np.float32) / 32767.0
        logits, lengths = forward(block[None, :], [len(block)])
        frames = int(lengths[0])
        first = (position - start) // FRAME_SAMPLES
        last = frames if position + segment >= len(wave) else first + segment // FRAME_SAMPLES
        if last > frames:
            raise RuntimeError(f"segment at {position}: needs frame {last}, model gave {frames}")
        pieces.append(torch.softmax(logits[0, first:last].float(), dim=-1).cpu().numpy())
        position += segment
    return np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 5))


def restart_points(length: int, preroll: float, spacing: float = COLD_SPACING_SECONDS,
                   window: float = COLD_WINDOW_SECONDS,
                   pad: float = LOOKAHEAD_PAD_SECONDS) -> list[int]:
    """Restart sample positions whose full input (pre-roll .. window + pad) fits."""
    step = int(round(spacing * SAMPLE_RATE))
    need_before = int(round(preroll * SAMPLE_RATE))
    need_after = int(round((window + pad) * SAMPLE_RATE)) + FRAME_SAMPLES
    first = step * max(1, -(-need_before // step))
    return list(range(first, length - need_after + 1, step))


def cold_window_frames(window: float = COLD_WINDOW_SECONDS) -> int:
    """Frames whose start time lies inside the first ``window`` seconds."""
    return int(np.ceil(window * FRAMES_PER_SECOND - 1e-9))


def cold_probabilities(forward, waves, preroll: float, batch: int = 64,
                       window: float = COLD_WINDOW_SECONDS,
                       spacing: float = COLD_SPACING_SECONDS) -> np.ndarray:
    """Posteriors of the first ``window`` s after each restart: [restarts, frames, classes].

    The encoder starts ``preroll`` s before the restart point with no history
    (a cold start); scoring begins at the restart point.
    """
    import torch

    before = _samples(preroll, FRAME_SAMPLES)
    keep = cold_window_frames(window)
    after = keep * FRAME_SAMPLES + _samples(LOOKAHEAD_PAD_SECONDS, FRAME_SAMPLES)
    blocks = [(wave, point) for wave in waves
              for point in restart_points(len(wave), preroll, spacing, window)]
    out = []
    first = before // FRAME_SAMPLES
    for index in range(0, len(blocks), batch):
        chunk = blocks[index:index + batch]
        audio = np.stack([wave[point - before:point + after] for wave, point in chunk])
        audio = audio.astype(np.float32) / 32767.0
        logits, lengths = forward(audio, [audio.shape[1]] * len(chunk))
        if int(lengths.min()) < first + keep:
            raise RuntimeError("cold-start block shorter than its scored window")
        out.append(torch.softmax(logits[:, first:first + keep].float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, keep, 5))


# ------------------------------------------------------- automaton with reset

def transition_matrices(class_count: int):
    transitions = v12.keyword_automaton_transitions(
        v12.WAKE_SEQUENCE, class_count, v12.CTC_BLANK_ID)
    state_count = (len(tuple(v12.WAKE_SEQUENCE)) + 1) * class_count
    matrices = np.zeros((class_count, state_count, state_count), dtype=np.float64)
    for class_index, destinations in enumerate(transitions):
        for source, destination in enumerate(destinations):
            matrices[class_index, source, destination] = 1.0
    return matrices, state_count


def batched_events(probabilities: np.ndarray, thresholds: np.ndarray,
                   refractory: float = REFRACTORY_SECONDS) -> np.ndarray:
    """Events per (sequence, threshold), reset after each detection.

    ``probabilities`` is [sequences, frames, classes]; every sequence starts
    from the automaton's initial state.  Same recursion as
    ``evaluate_long_audio_streaming.running_events``, vectorised over sequences.
    """
    count, frames, class_count = probabilities.shape
    matrices, state_count = transition_matrices(class_count)
    matched_start = len(tuple(v12.WAKE_SEQUENCE)) * class_count
    initial = np.zeros(state_count, dtype=np.float64)
    initial[int(v12.CTC_BLANK_ID)] = 1.0
    state = np.broadcast_to(initial, (count, len(thresholds), state_count)).copy()
    blocked = np.zeros((count, len(thresholds)), dtype=np.int64)
    events = np.zeros((count, len(thresholds)), dtype=np.int64)
    skip = max(1, int(round(refractory * FRAMES_PER_SECOND)))
    tiny = np.finfo(np.float64).tiny
    for frame in range(frames):
        combined = np.einsum("wc,csu->wsu", probabilities[:, frame].astype(np.float64), matrices)
        state = np.einsum("wts,wsu->wtu", state, combined)
        state /= np.maximum(state.sum(axis=2, keepdims=True), tiny)
        matched = state[:, :, matched_start:].sum(axis=2)
        unmatched = state[:, :, :matched_start].sum(axis=2)
        score = np.log(np.maximum(matched, tiny)) - np.log(np.maximum(unmatched, tiny))
        fired = (score >= thresholds[None, :]) & (blocked <= 0)
        if fired.any():
            events += fired
            state[fired] = initial
            blocked[fired] = skip
        blocked = np.maximum(blocked - 1, 0)
    return events


# ------------------------------------------------------------------ metrics

def recall_at(per_hour: np.ndarray, recalls: np.ndarray, limit: float) -> float:
    allowed = per_hour <= limit
    return float(recalls[allowed].max()) if allowed.any() else 0.0


def operating_index(per_hour: np.ndarray, recalls: np.ndarray, limit: float) -> int:
    """Lowest threshold (highest recall) whose streaming FA/h stays within ``limit``;
    the highest threshold when none does."""
    allowed = np.flatnonzero(per_hour <= limit)
    if not len(allowed):
        return len(per_hour) - 1
    return int(allowed[np.argmax(recalls[allowed])])


def cold_key(preroll: float) -> str:
    return f"la_cold_fa_per_start_preroll{preroll:g}".replace(".", "p")


# ------------------------------------------------------------------ one measurement

def measure(runner, waves, positive_waves, threshold_count: int = 128,
            refractory: float = REFRACTORY_SECONDS) -> dict:
    """Everything the selection and the reports need, as raw arrays.

    ``runner`` provides ``model``, ``_forward`` and ``utterance_scores`` (see
    ``evaluate_long_audio_streaming.Runner``).  Thresholds run from the lowest
    to the highest positive score, so recall 1.0 stays measurable.
    """
    hours = sum(len(wave) for wave in waves) / SAMPLE_RATE / 3600
    with limited_context(runner.model):
        positives = runner.utterance_scores(positive_waves)
        thresholds = np.linspace(float(positives.min()), float(positives.max()),
                                 threshold_count)
        events = np.zeros(len(thresholds), dtype=np.int64)
        for wave in waves:
            probabilities = stream_probabilities(runner._forward, wave)
            events += batched_events(probabilities[None], thresholds, refractory)[0]
        cold, starts = {}, {}
        for preroll in COLD_PREROLLS:
            windows = cold_probabilities(runner._forward, waves, preroll)
            cold[preroll] = batched_events(windows, thresholds, refractory).sum(axis=0)
            starts[preroll] = len(windows)
    return {"hours": hours, "positives": positives, "thresholds": thresholds,
            "events": events, "cold_events": cold, "cold_starts": starts}


def summarize(measured: dict, select_fa: float = 1.0,
              fa_targets=(0.5, 1.0, 2.0), recall_targets=(0.80, 0.90)) -> dict:
    """``la_*`` metrics from ``measure``.  The cold-start numbers are false
    accepts per restart at the streaming operating threshold (the one reaching
    the best recall within ``select_fa`` FA/h), i.e. what a shipped device with
    that threshold would do right after its encoder restarts."""
    thresholds, positives = measured["thresholds"], measured["positives"]
    per_hour = measured["events"] / measured["hours"]
    recalls = np.array([(positives >= t).mean() for t in thresholds])
    summary = {"la_protocol": PROTOCOL_NAME, "la_hours": round(measured["hours"], 2),
               "la_positives": int(len(positives))}
    for limit in fa_targets:
        summary["la_recall_at_fa_" + f"{limit:g}".replace(".", "p")] = \
            recall_at(per_hour, recalls, limit)
    for target in recall_targets:
        reached = recalls >= target
        summary["la_fa_per_hour_at_recall_" + f"{target:.2f}".replace(".", "p")] = (
            float(per_hour[reached].min()) if reached.any() else -1.0)
    zero = measured["events"] == 0
    summary["la_recall_at_zero_fa"] = float(recalls[zero].max()) if zero.any() else 0.0
    index = operating_index(per_hour, recalls, select_fa)
    summary["la_selection_recall"] = recall_at(per_hour, recalls, select_fa)
    summary["la_selection_threshold"] = float(thresholds[index])
    summary["la_selection_limit_per_hour"] = select_fa
    for preroll, events in measured["cold_events"].items():
        starts = max(1, measured["cold_starts"][preroll])
        summary[cold_key(preroll)] = float(events[index]) / starts
    summary["la_cold_starts"] = int(measured["cold_starts"][COLD_PREROLLS[0]])
    return summary
