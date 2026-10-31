#!/usr/bin/env python3
"""V14.2: V14, minus what did not work, plus what the measurements demanded.

V14.1 added four things and lost to plain V14 on the only metric that matters
(recall at <= 1 false accept/hour on 16.6 h of real YouTube audio: 0.87 vs
0.93).  So V14.2 starts from V14 and keeps only what evidence supports.

Dropped from V14.1
------------------
* **Confusion-class sampling of hard negatives.**  A run with it switched off
  beat every run with it on, and the false accepts it was meant to target do
  not exist: of 43 false-accept events across two corpora, none came from the
  classes it prioritised, and more than half came from sentences containing no
  wake fragment at all.  The errors are acoustic, not lexical.
* **The re-weighted negative source shares.**  The old V14 weights scored at
  least as well.

Kept from V14.1
---------------
* A fixed per-stage schedule, retuned: two stages, 1:1 then 8:1, the peak held
  for 14,000 of the 20,000 steps.  Peaks of 4 and 24 both lost clearly; 32 is
  impossible (a batch must hold one positive, so the ratio cannot exceed
  ``--max-utterances-per-batch`` - 1).

New in V14.2
------------
1. **Selection on the deployment metric.**  Every previous generation chose
   checkpoints by a proxy — development F1, then recall on isolated LSVSC
   clips — and both proxies pointed the wrong way.  Here the criterion is
   recall at a false-accept budget measured on continuous YouTube audio with
   the automaton reset after each detection, exactly as the product runs.
   Three shorter clips (4.87 h) drive selection; ``hk15_24h`` (11.76 h) is
   never touched, so the final number stays honest.
2. **Channel augmentation, precomputed on disk.**  A 300-3400 Hz telephone
   band drops recall from 0.948 to 0.222.  Copies made by
   ``prepare_v14_2_channel_shards.py`` (room 0.15, band 0.25) are *added*
   beside every source: positives through ``--wakeword-manifest``, negatives
   through ``--v14-2-channel-negative-manifest``.  The audit refuses a run in
   which the augmented share of negative draws differs from that of positive
   draws, and refuses live augmentation on top of the copies
   (``ctctknk-docs/20``).  The live path of ``parakeet_v14_2_channel`` is
   still available for comparison but defaults to off.
3. **A speaker split chosen by hand** (``--speaker-split-file``): validation
   and test merged into one eight-speaker set that, unlike the old four,
   contains women and more than one northern speaker.  The first V14.2 runs
   did NOT get it: the V12.2 loader re-shuffles speakers after reading the
   file, and its audit insists on 4 validation + 4 test speakers.  Both are
   replaced here when a split file is given.

Everything else — architecture, five-token vocabulary, V11 initialisation,
optimizer, shards, MUSAN, frequency-only SpecAugment, source shares, negative
slots per epoch, curriculum — is V14 untouched.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

import evaluate_long_audio_streaming as longaudio
import parakeet_v12_2_sharded_train as previous
import parakeet_v12_train as v12
import parakeet_v14_2_channel as channel
import parakeet_v14_train as v14
import incontext_protocol as incontext
import streaming_protocol as protocol

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_FORMAT_VERSION = 11
TRAINER_VARIANT = "v14.2_reduced_streaming_pure_ctc_longaudio_selection_channelshards"
# The V14 weights, restored: V14.1's rebalance was not supported by any run.
DEFAULT_SOURCE_SHARES = (
    "WakewordV12=0.25,InfoRE2=0.30,VLSP=0.15,LSVSC=0.12,ViMD=0.10,MUSAN=0.08"
)
DEFAULT_STAGE_STEPS = "6000,14000"
BEST_FILE = "best_longaudio_recall.pt"
BEST_STATE = "best_longaudio_state.json"
FA_TARGETS = (0.5, 1.0, 2.0)
RECALL_TARGETS = (0.80, 0.90)
CHANNEL_PROVENANCE = "v14_2_channel_augmented"
# Since 2026-09-22 (ctctknk-docs/28): one protocol (streaming_protocol) at the
# deployment context [64, 7], segment ends with their look-ahead, cold start as
# false accepts per restart.  The old rule averaged streaming recall with a
# "cold recall" whose segments had lost their look-ahead.
SELECTION_RULE = "stream64_recall_at_fa_then_zero_fa_then_cold_preroll"
# Opt-in since 2026-09-25 (ctctknk-docs/33 §3j): recall with each positive inside
# continuous audio (warm encoder), selection speakers only, near-miss FA capped.
# Frozen 2026-09-25 (§3k): the rule lives in incontext_protocol; refuse a changed file.
IN_CONTEXT_SELECTION_RULE = incontext.SELECTION_RULE
IN_CONTEXT_PROTOCOL_MD5 = "c6f26f5d3faa49ac4ade5b20b23e34fb"
# v5 speaker split (27/09/2026): v5/proto/incontext_protocol.py on PYTHONPATH -- same rule,
# new selection/test speaker lists.  Either frozen file is accepted; any other edit is refused.
IN_CONTEXT_PROTOCOL_MD5_ACCEPTED = {IN_CONTEXT_PROTOCOL_MD5, "319bdeaccb8708ccf7147e6fa15d45a9"}
IN_CONTEXT_SEED = 20260925
SNAPSHOT_DIR = "validation_snapshots"
# Hard and easy pools are compared only when both are this large.
MIN_POOL_FOR_PARITY = 500

_V14_TRAIN = v14.train_v12
_V14_SIGNATURE = v14.training_signature
_V14_AUDIT = v14.audit_samples
_V14_LOAD_INFORE2 = v14.load_infore2_samples
_V12_EVALUATE = v12.evaluate
_V12_APPEND = v12.RunArtifacts.append

STAGE_STEPS: tuple[int, ...] = ()
SELECT_AUDIO: list[Path] = []
SELECT_FA = 1.0
REFRACTORY = 1.0
THRESHOLD_COUNT = 128
PARTIAL = False
IN_CONTEXT = False
CHANNEL_NEGATIVES: Path | None = None
CHANNEL_TOLERANCE = 0.02
SPLIT_FILE: dict[str, str] | None = None
HARD_NEGATIVE_SHARE = 0.75

_SIGNATURE: dict[str, object] = {}
_LAST_STUDENT = None
_EVAL_POSITIVES: list = []
_EVAL_NEAR_MISSES: list = []
_CACHE: dict[str, object] = {}
_CHANNEL_REPORT: dict[str, object] = {}


def fa_key(limit: float) -> str:
    return "la_recall_at_fa_" + f"{limit:g}".replace(".", "p")


def recall_key(target: float) -> str:
    return "la_fa_per_hour_at_recall_" + f"{target:.2f}".replace(".", "p")


# ============================================================
# Options
# ============================================================

def option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--v14-2-stage-steps", default=DEFAULT_STAGE_STEPS)
    parser.add_argument("--v14-2-select-audio", nargs="+", type=Path, required=True,
                        help="long audio used to choose checkpoints; keep the "
                             "largest file out of this list for the final test")
    parser.add_argument("--v14-2-select-fa-per-hour", type=float, default=1.0)
    parser.add_argument("--v14-2-refractory", type=float, default=1.0)
    parser.add_argument("--v14-2-thresholds", type=int, default=128,
                        help="threshold grid from the lowest to the highest positive score")
    parser.add_argument("--v14-2-channel-negative-manifest", type=Path, default=None,
                        help="negative copies from prepare_v14_2_channel_shards.py; pair it "
                             "with that script's wakeword manifest as --wakeword-manifest")
    parser.add_argument("--v14-2-channel-tolerance", type=float, default=0.02,
                        help="largest allowed gap between the augmented share of positive "
                             "and negative draws")
    # Live channel augmentation (the first V14.2 runs).  Off by default: with
    # precomputed copies it would augment twice, and the audit refuses that.
    parser.add_argument("--v14-2-rir-root", type=Path, default=channel.DEFAULT_RIR_ROOT)
    parser.add_argument("--v14-2-room-probability", type=float, default=0.0)
    parser.add_argument("--v14-2-band-probability", type=float, default=0.0)
    parser.add_argument("--v14-2-channel-salt", type=int, default=0)
    parser.add_argument("--v14-2-partial", action="store_true")
    parser.add_argument("--v14-2-in-context-selection", action="store_true",
                        help="select on in-context recall of the selection speakers "
                             "(incontext_protocol); test speakers are never scored")
    return parser


def passthrough_parser() -> argparse.ArgumentParser:
    """The base-trainer options V14.2 needs to read before the base trainer does."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--speaker-split-file", type=Path, default=None)
    parser.add_argument("--hard-negative-share", type=float, default=0.75)
    return parser


def configure(options: argparse.Namespace) -> None:
    global STAGE_STEPS, SELECT_AUDIO, SELECT_FA, REFRACTORY, THRESHOLD_COUNT, PARTIAL
    global IN_CONTEXT, SELECTION_RULE
    global CHANNEL_NEGATIVES, CHANNEL_TOLERANCE
    steps = tuple(int(x) for x in str(options.v14_2_stage_steps).split(",") if x.strip())
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("--v14-2-stage-steps needs positive integers")
    missing = [p for p in options.v14_2_select_audio if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"selection audio not found: {missing}")
    if options.v14_2_select_fa_per_hour <= 0:
        raise ValueError("--v14-2-select-fa-per-hour must be positive")
    manifest = options.v14_2_channel_negative_manifest
    if manifest is not None and not Path(manifest).is_file():
        raise FileNotFoundError(f"channel negative manifest not found: {manifest}")
    live = options.v14_2_room_probability > 0 or options.v14_2_band_probability > 0
    if manifest is not None and live:
        raise ValueError("precomputed channel copies AND live channel augmentation: that "
                         "augments twice; set --v14-2-room/band-probability to 0")
    if options.v14_2_channel_tolerance <= 0:
        raise ValueError("--v14-2-channel-tolerance must be positive")
    STAGE_STEPS = steps
    SELECT_AUDIO = [Path(p) for p in options.v14_2_select_audio]
    SELECT_FA = float(options.v14_2_select_fa_per_hour)
    REFRACTORY = float(options.v14_2_refractory)
    THRESHOLD_COUNT = int(options.v14_2_thresholds)
    PARTIAL = bool(options.v14_2_partial)
    IN_CONTEXT = bool(getattr(options, "v14_2_in_context_selection", False))
    if IN_CONTEXT:
        digest = hashlib.md5(Path(incontext.__file__).read_bytes()).hexdigest()
        if digest not in IN_CONTEXT_PROTOCOL_MD5_ACCEPTED:
            raise RuntimeError(f"incontext_protocol.py changed since the rule was frozen "
                               f"({digest} != {IN_CONTEXT_PROTOCOL_MD5}); add a new rule "
                               "instead of editing a frozen one")
        SELECTION_RULE = IN_CONTEXT_SELECTION_RULE
    CHANNEL_NEGATIVES = Path(manifest).resolve() if manifest is not None else None
    CHANNEL_TOLERANCE = float(options.v14_2_channel_tolerance)


def configure_passthrough(options: argparse.Namespace) -> None:
    global SPLIT_FILE, HARD_NEGATIVE_SHARE
    HARD_NEGATIVE_SHARE = float(options.hard_negative_share)
    if options.speaker_split_file is None:
        SPLIT_FILE = None
        return
    raw = json.loads(Path(options.speaker_split_file).read_text(encoding="utf-8"))
    SPLIT_FILE = {unicodedata.normalize("NFKC", str(k)): str(v) for k, v in raw.items()}


# ============================================================
# Speaker split: the file, not a reshuffle
# ============================================================

def keep_split_file(samples, split_seed):
    """Stands in for V12.2's ``reshuffle_speaker_splits`` when a split file is given.

    The V12 loader has already applied the file; V12.2 then drew a fresh
    37/4/4 split and silently discarded it.
    """
    splits = {str(sample.speaker_id): sample.split for sample in samples}
    print("V14.2 speaker split from --speaker-split-file (V12.2 reshuffle disabled):",
          dict(collections.Counter(splits.values())), flush=True)
    return list(samples), splits


def split_file_audit(samples, speaker_splits, require_test: bool = False) -> None:
    """V12.2's protocol audit, minus its fixed 4 validation + 4 test speakers."""
    has_test = any(sample.split == "test" for sample in samples)
    previous._ORIGINAL_AUDIT_SAMPLES(samples, speaker_splits, require_test=has_test)
    infore2_development = [s for s in samples
                           if s.source_dataset == "InfoRE2" and s.split != "train"]
    if infore2_development:
        raise RuntimeError("All InfoRE2 rows were exposed to V11 and must remain in training; "
                           f"found {len(infore2_development)} non-train rows")
    by_split: dict[str, set[str]] = collections.defaultdict(set)
    labels: dict[str, set[bool]] = collections.defaultdict(set)
    for sample in samples:
        if sample.source_dataset != v14.WAKEWORD_SOURCE:
            continue
        speaker = str(sample.speaker_id)
        by_split[sample.split].add(speaker)
        labels[speaker].add(bool(sample.is_positive))
        if SPLIT_FILE is not None and SPLIT_FILE.get(speaker) != sample.split:
            raise RuntimeError(f"{sample.uid}: split {sample.split!r} but the split file "
                               f"says {SPLIT_FILE.get(speaker)!r}")
    splits = list(by_split)
    for index, first in enumerate(splits):
        for second in splits[index + 1:]:
            if by_split[first] & by_split[second]:
                raise RuntimeError(f"speakers in both {first} and {second}: "
                                   f"{sorted(by_split[first] & by_split[second])}")
    one_label = sorted(s for split in splits if split != "train"
                       for s in by_split[split] if len(labels[s]) < 2)
    print("V14.2 split audit:",
          {split: len(speakers) for split, speakers in sorted(by_split.items())},
          f"test={'yes' if has_test else 'no (development only)'}",
          f"evaluation speakers with one label only={one_label or 'none'}",
          "infore2_nontrain=0 disjoint=yes", flush=True)


# ============================================================
# Channel copies
# ============================================================

def load_infore2_with_channel(*args, **kwargs):
    """V14's loader (InfoRE2 + prepared negatives) plus the negative channel copies."""
    samples = _V14_LOAD_INFORE2(*args, **kwargs)
    if CHANNEL_NEGATIVES is None:
        return samples
    tokenizer = args[1] if len(args) > 1 else kwargs["tokenizer"]
    old_to_reduced = args[2] if len(args) > 2 else kwargs["old_to_reduced"]
    copies = v14.load_extra_negatives(CHANNEL_NEGATIVES, tokenizer, old_to_reduced)
    strays = [s.uid for s in copies if not channel.is_channel_copy(s.uid)]
    if strays:
        raise RuntimeError(f"{CHANNEL_NEGATIVES} holds rows that are not channel copies: "
                           f"{strays[:3]}")
    copies = [replace(s, provenance_kind=CHANNEL_PROVENANCE, augmentation="channel")
              for s in copies]
    print(f"V14.2 channel negative copies: {len(copies):,} from {CHANNEL_NEGATIVES}", flush=True)
    return list(samples) + copies


def origin_uid(uid: str) -> str:
    return str(uid).split(channel.CHANNEL_UID_TAG, 1)[0]


def expected_copy_share(pool) -> float:
    """Expected share of draws on copies, following speaker -> family -> variant."""
    by_speaker: dict[str, dict[tuple, list[bool]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for sample in pool:
        speaker = (str(sample.speaker_id) if sample.speaker_id is not None
                   else f"__{sample.source_dataset}__")
        by_speaker[speaker][previous.family_key(sample)].append(
            channel.is_channel_copy(sample.uid))
    if not by_speaker:
        return 0.0
    per_speaker = [np.mean([np.mean(flags) for flags in families.values()])
                   for families in by_speaker.values()]
    return float(np.mean(per_speaker))


def negative_draw_share(train_negatives) -> tuple[float, dict]:
    """Slot-weighted copy share over the sources the V14 sampler draws from."""
    pools: dict[tuple[str, str], list] = collections.defaultdict(list)
    for sample in train_negatives:
        pools[(sample.source_dataset, "hard" if sample.is_hard_negative else "easy")].append(sample)
    total, table = 0.0, {}
    for source, share in v14.SOURCE_SHARES.items():
        if share <= 0:
            continue
        hard, easy = pools[(source, "hard")], pools[(source, "easy")]
        hard_weight = HARD_NEGATIVE_SHARE if hard and easy else (1.0 if hard else 0.0)
        shares = {"hard": expected_copy_share(hard), "easy": expected_copy_share(easy)}
        total += share * (hard_weight * shares["hard"] + (1.0 - hard_weight) * shares["easy"])
        table[source] = {"hard": round(shares["hard"], 4), "easy": round(shares["easy"], 4),
                         "hard_rows": len(hard), "easy_rows": len(easy)}
        if min(len(hard), len(easy)) >= MIN_POOL_FOR_PARITY and \
                abs(shares["hard"] - shares["easy"]) > CHANNEL_TOLERANCE:
            raise RuntimeError(f"{source}: copies take {shares['hard']:.3f} of hard draws but "
                               f"{shares['easy']:.3f} of easy draws")
    return total, table


def audit_channel(samples) -> dict:
    """Refuse a run whose channel copies would teach 'channel => label'."""
    copies = [s for s in samples if channel.is_channel_copy(s.uid)]
    if copies and channel.live_active():
        raise RuntimeError("precomputed channel copies AND live channel augmentation: "
                           "that augments twice")
    if not copies:
        if CHANNEL_NEGATIVES is not None:
            raise RuntimeError("--v14-2-channel-negative-manifest given but no copy loaded")
        return {}
    train = {s.uid for s in samples if s.split == "train" and not channel.is_channel_copy(s.uid)}
    leaked = [s.uid for s in copies if s.split != "train"]
    orphans = [s.uid for s in copies if origin_uid(s.uid) not in train]
    if leaked or orphans:
        raise RuntimeError(f"channel copies outside train: {leaked[:3]}; copies whose "
                           f"original is not a training sample: {orphans[:3]}")
    positive_pool = [s for s in samples if s.split == "train" and s.is_positive]
    negatives = [s for s in samples if s.split == "train" and not s.is_positive]
    positive = expected_copy_share(positive_pool)
    negative, table = negative_draw_share(negatives)
    if positive == 0.0 or negative == 0.0:
        raise RuntimeError(f"copies on one class only: positive share {positive:.3f}, "
                           f"negative share {negative:.3f}")
    if abs(positive - negative) > CHANNEL_TOLERANCE:
        raise RuntimeError(f"augmented share of positive draws {positive:.4f} vs negative "
                           f"draws {negative:.4f} differs by more than {CHANNEL_TOLERANCE}; "
                           "the model could learn 'channel => wake word' (ctctknk-docs/20 §5)")
    report = {"copies": len(copies), "positive_draw_share": round(positive, 4),
              "negative_draw_share": round(negative, 4), "by_source": table,
              "negative_manifest": str(CHANNEL_NEGATIVES) if CHANNEL_NEGATIVES else None,
              "negative_manifest_sha1": (hashlib.sha1(CHANNEL_NEGATIVES.read_bytes()).hexdigest()
                                         if CHANNEL_NEGATIVES else None)}
    print(f"V14.2 channel audit: copies={len(copies):,} positive_share={positive:.4f} "
          f"negative_share={negative:.4f} (tolerance {CHANNEL_TOLERANCE})", flush=True)
    for source, row in sorted(table.items()):
        print(f"  {source:12s} hard={row['hard']:.4f} ({row['hard_rows']:,} rows) "
              f"easy={row['easy']:.4f} ({row['easy_rows']:,} rows)", flush=True)
    return report


# ============================================================
# Fixed curriculum
# ============================================================

def staged_schedule_observe(state, metric, optimizer_step, stage_count, *args, **kwargs):
    if state.complete:
        return "already_complete"
    state.validation_observations_in_stage += 1
    metric = float(metric)
    if state.best_metric is None or metric < state.best_metric:
        state.best_metric = metric
    budget = STAGE_STEPS[min(state.stage_index, len(STAGE_STEPS) - 1)]
    if optimizer_step - state.stage_start_optimizer_step < budget:
        return "held"
    if state.stage_index + 1 < stage_count:
        state.stage_index += 1
        state.epochs_in_stage = 0
        state.best_metric = None
        state.bad_validations = 0
        state.validation_observations_in_stage = 0
        state.stage_start_optimizer_step = int(optimizer_step)
        state.metric_ema = None
        state.best_detection_metric = None
        state.bad_detection_validations = 0
        return "advanced"
    state.complete = True
    return "complete"


# ============================================================
# Selection on real long audio
# ============================================================

def _load_selection_audio():
    import soundfile as sf

    if "waves" in _CACHE:
        return _CACHE["waves"], _CACHE["hours"]
    waves, hours = [], 0.0
    for path in SELECT_AUDIO:
        wave, rate = sf.read(str(path), dtype="int16")
        if rate != longaudio.SAMPLE_RATE:
            raise RuntimeError(f"{path}: expected 16 kHz, got {rate}")
        if wave.ndim > 1:
            wave = wave.mean(axis=1).astype(np.int16)
        waves.append(wave)
        hours += len(wave) / longaudio.SAMPLE_RATE / 3600
    _CACHE["waves"], _CACHE["hours"] = waves, hours
    print(f"[long-audio] selection corpus: {len(waves)} files, {hours:.2f} h", flush=True)
    return waves, hours


def _positive_waves():
    if "positives" in _CACHE:
        return _CACHE["positives"]
    dataset = v12.UnifiedAudioDataset(_EVAL_POSITIVES)
    waves = []
    for index in range(len(_EVAL_POSITIVES)):
        audio = dataset[index]["audio"]
        audio = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        waves.append(np.clip(audio * 32767.0, -32768, 32767).astype(np.int16))
    _CACHE["positives"] = waves
    print(f"[long-audio] evaluation positives: {len(waves):,}", flush=True)
    return waves


def _near_miss_waves():
    if "near_misses" in _CACHE:
        return _CACHE["near_misses"]
    dataset = v12.UnifiedAudioDataset(_EVAL_NEAR_MISSES)
    waves = []
    for index in range(len(_EVAL_NEAR_MISSES)):
        audio = dataset[index]["audio"]
        audio = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        waves.append(np.clip(audio * 32767.0, -32768, 32767).astype(np.int16))
    _CACHE["near_misses"] = waves
    print(f"[long-audio] in-context near misses: {len(waves):,}", flush=True)
    return waves


def _in_context_metrics(runner, waves, hours) -> dict:
    """``la_ic_*``: in-context recall / near-miss FA on the FA/h grid of ``waves``."""
    with protocol.limited_context(runner.model):
        measured = incontext.measure(runner._forward, waves, _positive_waves(),
                                     _near_miss_waves(), IN_CONTEXT_SEED)
        finite = measured["positive_best"][np.isfinite(measured["positive_best"])]
        thresholds = np.linspace(float(finite.min()), float(finite.max()), THRESHOLD_COUNT)
        events = np.zeros(len(thresholds), dtype=np.int64)
        for wave in waves:
            probabilities = protocol.stream_probabilities(runner._forward, wave)
            events += protocol.batched_events(probabilities[None], thresholds, REFRACTORY)[0]
    summary = incontext.summarize(measured, thresholds, events / hours, SELECT_FA)
    return {"la_" + key: value for key, value in summary.items()
            if isinstance(value, (int, float))}


def compute_long_audio_metrics(student, device) -> dict:
    """Recall against false accepts on the selection audio, ``streaming_protocol``.

    Streaming at the deployment context ``[64, 7]`` (segment ends keep their
    look-ahead), plus false accepts in the first 3 s after an encoder restart,
    with and without a 2 s pre-roll, at the streaming operating threshold.
    The grid starts at the lowest positive score so recall 1.0 stays measurable.
    """
    import torch

    started = time.time()
    waves, hours = _load_selection_audio()
    positives = _positive_waves()

    # Reuse the long-audio scorer's forward and utterance scoring without
    # letting it load a checkpoint: the student being trained is in memory.
    runner = longaudio.Runner.__new__(longaudio.Runner)
    runner.torch = torch
    runner.model = student
    runner.device = device
    runner.step = None
    runner.args = argparse.Namespace(batch_utterances=32)

    was_training = student.training
    student.eval()
    try:
        measured = protocol.measure(runner, waves, positives, THRESHOLD_COUNT, REFRACTORY)
        extra = _in_context_metrics(runner, waves, hours) if IN_CONTEXT else {}
    finally:
        student.train(was_training)
    summary = protocol.summarize(measured, SELECT_FA, FA_TARGETS, RECALL_TARGETS)
    summary["la_selection_score"] = summary["la_selection_recall"]
    if IN_CONTEXT:
        summary.update(extra)
        summary["la_selection_score"] = extra["la_ic_selection_score"]
    summary["la_seconds"] = round(time.time() - started, 1)
    return summary


def evaluate_with_long_audio(student, loader, device, *args, **kwargs):
    global _LAST_STUDENT
    metrics, preview = _V12_EVALUATE(student, loader, device, *args, **kwargs)
    if kwargs.get("include_full_detection_metrics"):
        return metrics, preview                   # the held-out test stays untouched
    metrics.update(compute_long_audio_metrics(student, device))
    _LAST_STUDENT = student
    shown = " ".join(f"R@FA<={limit:g}={metrics[fa_key(limit)]:.4f}" for limit in FA_TARGETS)
    print(f"[long-audio] {protocol.PROTOCOL_NAME} {shown} "
          f"FA/h@R0.90={metrics[recall_key(0.90)]:.2f} "
          f"R@0FA={metrics['la_recall_at_zero_fa']:.4f} "
          f"cold FA/start={metrics[protocol.cold_key(0.0)]:.4f} "
          f"(pre-roll {protocol.PREROLL_SECONDS:g}s: "
          f"{metrics[protocol.cold_key(protocol.PREROLL_SECONDS)]:.4f}) "
          f"score={metrics['la_selection_score']:.4f} ({metrics['la_seconds']:.0f}s)", flush=True)
    return metrics, preview


def _selection_key(record) -> tuple[float, ...]:
    """Streaming recall at the FA budget, then recall at zero false accepts,
    then fewer false accepts per cold start with the 2 s pre-roll the device
    uses, then without it (``ctctknk-docs/28`` §3)."""
    if IN_CONTEXT:
        return (float(record["la_selection_score"]),
                -float(record.get("la_ic_selection_near_miss_fa", 1.0)),
                -float(record.get(protocol.cold_key(protocol.PREROLL_SECONDS), 1e9)),
                -float(record.get(protocol.cold_key(0.0), 1e9)))
    return (float(record["la_selection_score"]), float(record["la_recall_at_zero_fa"]),
            -float(record.get(protocol.cold_key(protocol.PREROLL_SECONDS), 1e9)),
            -float(record.get(protocol.cold_key(0.0), 1e9)))


def _snapshot(run_directory: Path, record: dict, step: int) -> None:
    """Keep every validated student so a later criterion can re-choose without retraining."""
    target = run_directory / SNAPSHOT_DIR / f"step_{step:06d}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    v12.atomic_torch_save({
        "format_version": v12.CHECKPOINT_FORMAT_VERSION,
        "optimizer_step": step,
        "validation_metrics": {k: v for k, v in record.items()
                               if k not in {"batch_plan_audit", "curriculum_state"}},
        "vocabulary": list(v12.REDUCED_VOCABULARY),
        "ctc_blank_id": v12.CTC_BLANK_ID,
        "student_state_dict": _LAST_STUDENT.state_dict(),
    }, target)


def artifacts_append(self, record, preview=None):
    _V12_APPEND(self, record, preview)
    if record.get("kind") != "validation" or "la_selection_score" not in record:
        return
    step = int(record["optimizer_step"])
    if getattr(self, "wandb_run", None) is not None:
        payload = {"optimizer_step": step}
        for key, value in record.items():
            if key.startswith("la_") and isinstance(value, (int, float)):
                payload[f"validation/longaudio/{key[3:]}"] = value
        self.wandb_run.log(payload)

    run_directory = Path(self.run_directory)
    _snapshot(run_directory, record, step)
    state_path = run_directory / BEST_STATE
    state = json.loads(state_path.read_text()) if state_path.is_file() else None
    if state is not None and state.get("rule") != SELECTION_RULE:
        state = None                  # a state written under an older rule never wins ties
    key = _selection_key(record)
    if state is not None and tuple(state["key"]) >= key:
        return
    audit_path = run_directory / "initialization_audit.json"
    payload = {
        "format_version": v12.CHECKPOINT_FORMAT_VERSION,
        "trainer_variant": _SIGNATURE.get("trainer_variant", TRAINER_VARIANT),
        "criterion": SELECTION_RULE,
        "criterion_value": float(record["la_selection_score"]),
        "optimizer_step": step,
        "completed_epochs": int(record.get("epoch", 0)),
        "stage_index": int(record.get("stage_index_before_observation", 0)),
        "signature": dict(_SIGNATURE),
        "validation_metrics": {k: v for k, v in record.items()
                               if k not in {"batch_plan_audit", "curriculum_state"}},
        "vocabulary": list(v12.REDUCED_VOCABULARY),
        "ctc_blank_id": v12.CTC_BLANK_ID,
        "initialization_audit": (json.loads(audit_path.read_text())
                                 if audit_path.is_file() else None),
        "student_state_dict": _LAST_STUDENT.state_dict(),
    }
    target = run_directory / BEST_FILE
    v12.atomic_torch_save(payload, target)
    v12.atomic_write_text(state_path, json.dumps(
        {"key": list(key), "rule": SELECTION_RULE, "optimizer_step": step}) + "\n")
    self.update_best_summary("la_selection_score",
                             float(record["la_selection_score"]), step, target)
    print(f"Saved best la_selection_score={record['la_selection_score']:.6f} "
          f"(streaming recall at FA/h<={SELECT_FA:g}, R@0FA "
          f"{record['la_recall_at_zero_fa']:.4f}) at step {step:,}: {target}", flush=True)


# ============================================================
# Trainer hooks
# ============================================================

def train_v12(*args, **kwargs):
    trainer_args = args[0] if args else kwargs["args"]
    ratios = args[1] if len(args) > 1 else kwargs["ratios"]
    samples = args[2] if len(args) > 2 else kwargs["samples"]
    if len(ratios) != len(STAGE_STEPS):
        raise ValueError(f"{len(ratios)} curriculum ratios but {len(STAGE_STEPS)} "
                         "stage lengths (--v14-2-stage-steps)")
    every = int(getattr(trainer_args, "validation_every_steps", 0))
    if every <= 0 or any(step % every for step in STAGE_STEPS):
        raise ValueError("every --v14-2-stage-steps entry must be a multiple of "
                         f"--validation-every-steps ({every})")
    cap = int(getattr(trainer_args, "max_utterances_per_batch", 32))
    if max(ratios) > cap - 1:
        raise ValueError(f"ratio {max(ratios)} needs every batch to hold one positive; "
                         f"the cap is --max-utterances-per-batch - 1 = {cap - 1}")
    _EVAL_POSITIVES.clear()
    _EVAL_POSITIVES.extend(s for s in samples if s.split == "validation" and s.is_positive
                           and s.source_dataset == v14.WAKEWORD_SOURCE)
    if not _EVAL_POSITIVES:
        raise RuntimeError("no WakewordV12 validation positives for long-audio selection")
    if IN_CONTEXT:
        # Selection speakers only; the test half (incontext.TEST_SPEAKERS) is never scored.
        # Near misses = every WakewordV12 validation negative of those speakers (the
        # Sample carries no hard-negative flag; 496 of 2,124 are milder phrases).
        _EVAL_POSITIVES[:] = [s for s in _EVAL_POSITIVES
                              if incontext.role_of(s.original_path or "") == "selection"]
        _EVAL_NEAR_MISSES[:] = [
            s for s in samples if s.split == "validation" and not s.is_positive
            and s.source_dataset == v14.WAKEWORD_SOURCE
            and incontext.role_of(s.original_path or "") == "selection"
            and not channel.is_channel_copy(s.uid)]
        if not _EVAL_POSITIVES or not _EVAL_NEAR_MISSES:
            raise RuntimeError("in-context selection: no selection-speaker positives or "
                               "near misses in the validation split")
        print(f"V14.2: in-context selection on speakers {list(incontext.SELECTION_SPEAKERS)}: "
              f"{len(_EVAL_POSITIVES):,} positives, {len(_EVAL_NEAR_MISSES):,} near misses",
              flush=True)
    if any(channel.is_channel_copy(s.uid) for s in _EVAL_POSITIVES):
        raise RuntimeError("channel copies among the evaluation positives")
    trainer_args.final_selection_checkpoint = BEST_FILE
    # V14.2 merges the old validation and test speakers into one eight-speaker
    # evaluation set, so there is no test split left to run at the end. Detect
    # that rather than making the caller remember a flag; the honest final
    # number comes from the long audio that selection never touches.
    if not any(sample.split == "test" for sample in samples):
        trainer_args.development_only = True
        print("V14.2: split has no test speakers — final test step skipped; "
              "the untouched long-audio file is the final gate", flush=True)
    if PARTIAL:
        trainer_args.development_only = True
    print(f"V14.2: ratios={list(ratios)} stage_steps={list(STAGE_STEPS)} "
          f"select=recall@FA/h<={SELECT_FA:g} on {len(SELECT_AUDIO)} long files | "
          f"{'PARTIAL' if PARTIAL else 'full run'}", flush=True)
    return _V14_TRAIN(*args, **kwargs)


def training_signature(args, ratios, manifest_sha1, tokenizer_path, old_piece_ids):
    signature = _V14_SIGNATURE(args, ratios, manifest_sha1, tokenizer_path, old_piece_ids)
    signature.update({
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "trainer_variant": TRAINER_VARIANT + ("_partial" if PARTIAL else ""),
        "v14_2": {
            "curriculum": {"ratios": [int(r) for r in ratios],
                           "stage_steps": list(STAGE_STEPS),
                           "controller": "fixed_optimizer_steps_per_stage"},
            "selection": {"metric": SELECTION_RULE,
                          "protocol": protocol.describe(),
                          "threshold_grid": f"{THRESHOLD_COUNT} from min to max positive score",
                          "snapshots": SNAPSHOT_DIR,
                          "limit_per_hour": SELECT_FA,
                          "corpus": [p.name for p in SELECT_AUDIO],
                          "refractory_seconds": REFRACTORY,
                          "detector": "ctc_keyword_log_odds, reset after each hit",
                          "checkpoint": BEST_FILE},
            "speaker_split": ("split_file" if SPLIT_FILE is not None
                              else "v12_2_reshuffle"),
            "dropped_from_v14_1": ["confusion_class_sampling", "rebalanced_source_shares"],
            "partial": PARTIAL,
        },
    })
    signature.update(channel.signature(dict(_CHANNEL_REPORT) or None))
    if PARTIAL:
        signature["final_test_executed"] = False
    _SIGNATURE.clear()
    _SIGNATURE.update(signature)
    return signature


def audit_samples(samples, speaker_splits, require_test: bool = False) -> None:
    _V14_AUDIT(samples, speaker_splits, require_test=require_test)
    _CHANNEL_REPORT.clear()
    _CHANNEL_REPORT.update(audit_channel(samples))
    mode = channel.signature(dict(_CHANNEL_REPORT) or None)["channel_augmentation"]["mode"]
    print("V14.2 protocol audit:",
          f"stage_steps={','.join(map(str, STAGE_STEPS))}",
          f"select=recall@FA/h<={SELECT_FA:g}",
          f"selection_corpus={[p.name for p in SELECT_AUDIO]}",
          f"channel={mode}",
          f"speaker_split={'file' if SPLIT_FILE is not None else 'v12.2 reshuffle'}",
          f"partial={'yes' if PARTIAL else 'no'}", flush=True)


def apply_patches() -> None:
    v14.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v14.TRAINER_VARIANT = TRAINER_VARIANT
    v14.train_v12 = train_v12
    v14.training_signature = training_signature
    v14.audit_samples = audit_samples
    # v14.main assigns ``previous.load_infore2_samples = load_infore2_samples``
    # by global name, so this reaches the base trainer.
    v14.load_infore2_samples = load_infore2_with_channel
    if SPLIT_FILE is not None:
        previous.reshuffle_speaker_splits = keep_split_file
        v14._V12_2_AUDIT_SAMPLES = split_file_audit
    v12.curriculum_observe_steps = staged_schedule_observe
    v12.evaluate = evaluate_with_long_audio
    v12.RunArtifacts.append = artifacts_append
    # The class-based sampler of V14.1 is deliberately NOT installed.


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    options, remaining = option_parser().parse_known_args(argv)
    passthrough, _ = passthrough_parser().parse_known_args(remaining)
    if not any(arg.startswith("--v14-negative-source-shares") for arg in remaining):
        remaining += ["--v14-negative-source-shares", DEFAULT_SOURCE_SHARES]
    configure(options)
    configure_passthrough(passthrough)
    v12.install_runtime_imports()
    report = channel.install(rir_root=options.v14_2_rir_root,
                             room_probability=options.v14_2_room_probability,
                             band_probability=options.v14_2_band_probability,
                             salt=options.v14_2_channel_salt)
    print(f"V14.2 live channel augmentation: {report} | precomputed negatives: "
          f"{CHANNEL_NEGATIVES or 'none'}", flush=True)
    apply_patches()
    return v14.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
