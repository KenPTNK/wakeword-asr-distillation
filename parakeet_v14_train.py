#!/usr/bin/env python3
"""V14: V12.2 protocol + frequency-only SpecAugment + extra ASR/non-speech negatives.

What changes relative to V12.2 (and nothing else)
-------------------------------------------------
1. Extra negatives.  Rows from a prepared manifest
   (``prepare_v14_negative_16k_shards.py``: LSVSC, VLSP, ViMD, MUSAN music/noise)
   are added to the training split.  They are already mono 16 kHz PCM16 WAV in
   Parquet shards on fast storage, so training never resamples them.  Targets
   come from the same V12 projection as InfoRE2; MUSAN rows (no speech) have the
   empty target.  Every row must be a negative.

2. Per-source negative allocation.  The V12.2 sampler knows only two negative
   sources, ``"infore2"`` and ``"new"`` (= everything that is not InfoRE2).  Added
   corpora would silently be counted as ``"new"`` and share the 25 % budget meant
   for WakewordV12 negatives.  V14 replaces it with an explicit share per
   ``source_dataset``; within each source the V12.2 hard/easy split and
   family-balanced selection are kept.  The WakewordV12 share must equal
   ``--new-data-negative-share`` so the two settings cannot disagree.

3. Frequency masking.  Between preprocessor and encoder, training only:
   ``--v14-freq-masks`` bands, each 0..``--v14-freq-width`` mel bins wide, over the
   utterance's valid frames.  No time masking (a time mask can erase a whole
   syllable of a 1-1.5 s wake clip while the target still says [0, 1, 2, 3]).
   The preprocessor uses ``normalize="NA"``: measured log-mel values have mean
   -9.6 and 0.0 sits at the 99.9th percentile, so a literal-zero mask writes a
   loud band.  ``--v14-freq-mask-fill`` chooses mean (default), floor (utterance
   minimum, ~-16.6 = silence) or zero; ``--v14-freq-mask-min-bin`` keeps masks
   above a mel bin (6 ~ 300 Hz protects F0, 21 ~ 1 kHz protects the resolved
   harmonics that carry Vietnamese tone).  The mask has no parameters: V14
   checkpoints load in V12.2 tooling.

4. Partial mode (``--v14-partial-stage-steps N``) for cheap configuration
   sweeps: every curriculum stage lasts exactly N optimizer steps (no plateau
   controller, so all configurations get identical budgets) and the final
   held-out test is skipped, so sweeps never look at the test split.
   Everything else -- data, sampler, curriculum ratios, optimizer, validation,
   best-F1 export -- is the full V14 path.

Initialization, architecture, vocabulary, curriculum, optimizer, validation and
checkpoint selection are exactly V12.2.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

import parakeet_v12_2_sharded_train as previous
import parakeet_v12_train as v12


CHECKPOINT_FORMAT_VERSION = 9
TRAINER_VARIANT = (
    "v14_reduced_streaming_pure_ctc_freqmask_extra_negatives_sharded16k_reshuffled_val4_test4"
)
WAKEWORD_SOURCE = "WakewordV12"
DEFAULT_SOURCE_SHARES = (
    "WakewordV12=0.25,InfoRE2=0.30,VLSP=0.15,LSVSC=0.12,ViMD=0.10,MUSAN=0.08"
)

_V12_2_LOAD_INFORE2 = previous.load_infore2_samples
_V12_2_AUDIT_SAMPLES = previous.audit_samples
_V12_2_TRAINING_SIGNATURE = previous.training_signature
_V12_BUILD_STUDENT = v12.build_student_class

FILL_MODES = ("mean", "floor", "zero")

_V12_2_TRAIN = previous.train_v12

NEGATIVE_MANIFEST: Path | None = None
SOURCE_SHARES: dict[str, float] = {}
FREQ_MASKS = 2
FREQ_WIDTH = 8
FREQ_FILL = "mean"
FREQ_MIN_BIN = 0
PARTIAL_STAGE_STEPS = 0
_EXTRA_SUMMARY: dict[str, dict[str, float]] = {}


# ============================================================
# Options
# ============================================================

def option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v14-negative-manifest", type=Path, default=None)
    parser.add_argument("--v14-negative-source-shares", default=DEFAULT_SOURCE_SHARES)
    parser.add_argument("--v14-freq-masks", type=int, default=2)
    parser.add_argument("--v14-freq-width", type=int, default=8)
    parser.add_argument("--v14-freq-mask-fill", choices=FILL_MODES, default="mean",
                        help="mean: utterance mean; floor: utterance minimum (silence); "
                             "zero: literal 0.0 (loud under normalize='NA')")
    parser.add_argument("--v14-freq-mask-min-bin", type=int, default=0,
                        help="lowest mel bin a mask may start at (6 ~ 300 Hz, 21 ~ 1 kHz)")
    parser.add_argument("--v14-partial-stage-steps", type=int, default=0,
                        help="partial mode: fixed optimizer steps per curriculum stage, "
                             "no plateau controller, final test skipped (0 = off)")
    return parser


def parse_source_shares(text: str) -> dict[str, float]:
    shares: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name = name.strip()
        if not name or not value:
            raise ValueError(f"bad share entry {item!r}; expected NAME=FRACTION")
        if name in shares:
            raise ValueError(f"duplicate share for {name}")
        share = float(value)
        if not 0.0 <= share <= 1.0:
            raise ValueError(f"share for {name} must be in [0, 1]")
        shares[name] = share
    if not math.isclose(sum(shares.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"negative source shares must sum to 1, got {sum(shares.values()):.6f}")
    if WAKEWORD_SOURCE not in shares:
        raise ValueError(f"{WAKEWORD_SOURCE} must have an explicit share")
    return shares


def configure(options: argparse.Namespace) -> None:
    global NEGATIVE_MANIFEST, SOURCE_SHARES, FREQ_MASKS, FREQ_WIDTH
    global FREQ_FILL, FREQ_MIN_BIN, PARTIAL_STAGE_STEPS
    if options.v14_negative_manifest is None:
        raise ValueError("--v14-negative-manifest is required")
    manifest = options.v14_negative_manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if options.v14_freq_masks < 0 or options.v14_freq_width < 0:
        raise ValueError("frequency mask count and width must be non-negative")
    if options.v14_freq_width >= v12.STUDENT_N_MELS:
        raise ValueError(f"--v14-freq-width must be < {v12.STUDENT_N_MELS} mel bins")
    min_bin = int(getattr(options, "v14_freq_mask_min_bin", 0))
    if not 0 <= min_bin <= v12.STUDENT_N_MELS - max(options.v14_freq_width, 1):
        raise ValueError(
            f"--v14-freq-mask-min-bin must leave room for a {options.v14_freq_width}-bin "
            f"mask inside {v12.STUDENT_N_MELS} mel bins"
        )
    fill = getattr(options, "v14_freq_mask_fill", "mean")
    if fill not in FILL_MODES:
        raise ValueError(f"--v14-freq-mask-fill must be one of {FILL_MODES}")
    partial = int(getattr(options, "v14_partial_stage_steps", 0))
    if partial < 0:
        raise ValueError("--v14-partial-stage-steps must be non-negative")
    NEGATIVE_MANIFEST = manifest
    SOURCE_SHARES = parse_source_shares(options.v14_negative_source_shares)
    FREQ_MASKS = int(options.v14_freq_masks)
    FREQ_WIDTH = int(options.v14_freq_width)
    FREQ_FILL = fill
    FREQ_MIN_BIN = min_bin
    PARTIAL_STAGE_STEPS = partial


# ============================================================
# Extra negatives
# ============================================================

def load_extra_negatives(manifest: Path, tokenizer, old_to_reduced) -> list[v12.Sample]:
    root = manifest.parent
    samples: list[v12.Sample] = []
    checked_shards: set[Path] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            shard = (root / record["audio_relpath"]).resolve()
            if shard not in checked_shards:
                if root not in shard.parents:
                    raise RuntimeError(f"shard escapes manifest root at line {line_number}")
                if not shard.is_file():
                    raise FileNotFoundError(shard)
                checked_shards.add(shard)
            transcript = str(record.get("transcript") or "")
            normalized = v12.normalize_text(transcript)
            reduced = (
                v12.transcript_target_ids(normalized, tokenizer, old_to_reduced)
                if normalized else ()
            )
            if v12.contains_word_sequence(normalized):
                raise RuntimeError(
                    f"extra negative contains the wake phrase at line {line_number}: {transcript!r}"
                )
            samples.append(
                v12.Sample(
                    uid=str(record["uid"]),
                    source_dataset=str(record["source_dataset"]),
                    source_kind="parquet",
                    audio_path=str(shard),
                    audio_row=int(record["audio_row"]),
                    speaker_id=record.get("speaker_id"),
                    transcript=transcript,
                    utterance_index=str(record["utterance_index"]),
                    duration_seconds=float(record["duration_seconds"]),
                    reduced_token_ids=tuple(reduced),
                    is_positive=False,
                    split="train",
                    provenance_kind="v14_prepared_negative",
                )
            )
    if not samples:
        raise RuntimeError(f"no rows in {manifest}")
    return samples


def load_infore2_samples(*args, **kwargs):
    samples = _V12_2_LOAD_INFORE2(*args, **kwargs)
    tokenizer = args[1] if len(args) > 1 else kwargs["tokenizer"]
    old_to_reduced = args[2] if len(args) > 2 else kwargs["old_to_reduced"]
    extra = load_extra_negatives(NEGATIVE_MANIFEST, tokenizer, old_to_reduced)
    summary: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for sample in extra:
        counter = summary[sample.source_dataset]
        counter["rows"] += 1
        counter["seconds"] += sample.duration_seconds
        counter["hard"] += int(sample.is_hard_negative)
    _EXTRA_SUMMARY.clear()
    _EXTRA_SUMMARY.update({
        source: {"rows": int(c["rows"]), "hours": round(c["seconds"] / 3600, 3),
                 "hard_negative": int(c["hard"])}
        for source, c in sorted(summary.items())
    })
    return list(samples) + extra


# ============================================================
# Per-source negative allocation
# ============================================================

def allocate_slots(shares: dict[str, float], total: int) -> dict[str, int]:
    """Largest-remainder rounding so the counts sum exactly to ``total``."""
    raw = {name: share * total for name, share in shares.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    leftover = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:leftover]:
        counts[name] += 1
    return counts


def stratified_family_negative_sequence(
    samples: Sequence[v12.Sample],
    negative_slots: int,
    new_data_share: float,
    hard_negative_share: float,
    seed: int,
) -> list[int]:
    if not math.isclose(float(new_data_share), SOURCE_SHARES.get(WAKEWORD_SOURCE, -1.0),
                         abs_tol=1e-9):
        raise RuntimeError(
            f"--new-data-negative-share={new_data_share} disagrees with the V14 "
            f"{WAKEWORD_SOURCE} share {SOURCE_SHARES.get(WAKEWORD_SOURCE)}"
        )
    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.split != "train" or sample.is_positive:
            continue
        difficulty = "hard" if sample.is_hard_negative else "easy"
        groups[(sample.source_dataset, difficulty)].append(index)

    active = {name: share for name, share in SOURCE_SHARES.items() if share > 0}
    missing = [name for name in active if not (groups[(name, "hard")] or groups[(name, "easy")])]
    if missing:
        raise RuntimeError(f"negative sources with a share but no training rows: {missing}")

    selected: list[int] = []
    for source, source_count in allocate_slots(active, int(negative_slots)).items():
        hard_pool = groups[(source, "hard")]
        easy_pool = groups[(source, "easy")]
        if hard_pool and easy_pool:
            hard_count = int(round(source_count * hard_negative_share))
        elif hard_pool:
            hard_count = source_count
        else:
            hard_count = 0
        selected.extend(previous.family_balanced_sequence(
            samples, hard_pool, hard_count, seed, f"{source}-hard"))
        selected.extend(previous.family_balanced_sequence(
            samples, easy_pool, source_count - hard_count, seed, f"{source}-easy"))
    random.Random(seed + 9_187).shuffle(selected)
    return selected


# ============================================================
# Frequency-only SpecAugment
# ============================================================

def frequency_mask(features, lengths, mask_count: int, max_width: int,
                   fill: str = "mean", min_bin: int = 0):
    """Mask ``mask_count`` mel bands over each utterance's valid frames.

    ``features`` is [batch, mel, time].  A band of 0..``max_width`` bins starts
    at or above ``min_bin``.  ``fill`` selects the value written into it:
    ``mean`` (utterance mean over valid frames), ``floor`` (utterance minimum,
    i.e. the silence floor) or ``zero`` (literal 0.0).  Padded frames are left
    untouched.
    """
    import torch

    if mask_count <= 0 or max_width <= 0:
        return features
    if fill not in FILL_MODES:
        raise ValueError(f"unknown fill {fill!r}")
    batch, mel, frames = features.shape
    masked = features.clone()
    valid = (torch.arange(frames, device=features.device).unsqueeze(0)
             < lengths.to(features.device).unsqueeze(1))
    for item in range(batch):
        time_valid = valid[item]
        if not bool(time_valid.any()):
            continue
        region = features[item][:, time_valid]
        if fill == "mean":
            value = region.mean()
        elif fill == "floor":
            value = region.min()
        else:
            value = torch.zeros((), dtype=features.dtype, device=features.device)
        for _ in range(mask_count):
            width = int(torch.randint(0, max_width + 1, (1,)))
            if width == 0:
                continue
            start = int(torch.randint(min_bin, mel - width + 1, (1,)))
            band = masked[item, start:start + width]
            band[:, time_valid] = value
    return masked


# ============================================================
# Partial (proxy) runs
# ============================================================

def fixed_schedule_observe_steps(state, metric, optimizer_step, stage_count, *args, **kwargs):
    """Advance exactly every ``PARTIAL_STAGE_STEPS`` optimizer steps.

    Replaces the plateau controller in partial runs so every configuration
    spends the same number of steps in each curriculum stage.  Best-metric
    bookkeeping is kept for logging only.
    """
    if state.complete:
        return "already_complete"
    state.validation_observations_in_stage += 1
    metric = float(metric)
    if state.best_metric is None or metric < state.best_metric:
        state.best_metric = metric
    detection = kwargs.get("detection_metric")
    if detection is not None and (state.best_detection_metric is None
                                  or float(detection) > state.best_detection_metric):
        state.best_detection_metric = float(detection)
    if optimizer_step - state.stage_start_optimizer_step < PARTIAL_STAGE_STEPS:
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


def train_v12(*args, **kwargs):
    trainer_args = args[0] if args else kwargs["args"]
    if PARTIAL_STAGE_STEPS > 0:
        every = int(getattr(trainer_args, "validation_every_steps", 0))
        if every <= 0 or PARTIAL_STAGE_STEPS % every:
            raise ValueError(
                "--v14-partial-stage-steps must be a positive multiple of "
                f"--validation-every-steps (got {PARTIAL_STAGE_STEPS} and {every})"
            )
        # The data audit already ran with the held-out test split intact; only
        # the final test evaluation is suppressed, so proxy runs never see it.
        trainer_args.development_only = True
        print(f"V14 PARTIAL: {PARTIAL_STAGE_STEPS:,} steps per stage, "
              "plateau controller disabled, final test skipped")
    return _V12_2_TRAIN(*args, **kwargs)


def build_student_class():
    base = _V12_BUILD_STUDENT()

    class _StudentASRFreqMask(base):
        def forward(self, input_signal, input_signal_length):
            processed, processed_lengths = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
            if self.training:
                processed = frequency_mask(processed, processed_lengths, FREQ_MASKS,
                                           FREQ_WIDTH, FREQ_FILL, FREQ_MIN_BIN)
            encoded, encoded_lengths = self.encoder(
                audio_signal=processed, length=processed_lengths
            )
            return self.decoder(encoder_output=encoded), encoded_lengths

    _StudentASRFreqMask.__name__ = "_StudentASRFreqMask"
    return _StudentASRFreqMask


# ============================================================
# Audit and signature
# ============================================================

def _required_frames(ids) -> int:
    ids = tuple(int(v) for v in ids)
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def _encoder_frames(seconds: float) -> int:
    hop = int(round(v12.TARGET_SAMPLE_RATE * v12.FEATURE_HOP_SECONDS))
    mel = int(round(seconds * v12.TARGET_SAMPLE_RATE)) // hop + 1
    return math.ceil(mel / v12.SUBSAMPLING_FACTOR)


def audit_samples(samples, speaker_splits, require_test: bool = False) -> None:
    _V12_2_AUDIT_SAMPLES(samples, speaker_splits, require_test=require_test)
    uids = [sample.uid for sample in samples]
    if len(uids) != len(set(uids)):
        raise RuntimeError("duplicate uid across sources")

    counters: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for sample in samples:
        if sample.split != "train" or sample.is_positive:
            continue
        counter = counters[sample.source_dataset]
        counter["rows"] += 1
        counter["seconds"] += sample.duration_seconds
        counter["hard"] += int(sample.is_hard_negative)
        counter["empty"] += int(not sample.reduced_token_ids)
    print("=" * 88)
    print("V14 TRAINING NEGATIVES BY SOURCE")
    print("=" * 88)
    for source, counter in sorted(counters.items()):
        share = SOURCE_SHARES.get(source, 0.0)
        print(f"{source:12s} rows={counter['rows']:8,d} hours={counter['seconds'] / 3600:8.2f} "
              f"hard={counter['hard']:7,d} empty={counter['empty']:8,d} share={share:5.2f}")
    unused = sorted(source for source in counters if SOURCE_SHARES.get(source, 0.0) == 0.0)
    missing = sorted(source for source, share in SOURCE_SHARES.items()
                     if share > 0 and source not in counters)
    if missing:
        raise RuntimeError(f"sources with a share but no training negatives: {missing}")

    extra_positive = [s.uid for s in samples
                      if s.provenance_kind == "v14_prepared_negative" and s.is_positive]
    if extra_positive:
        raise RuntimeError(f"extra negatives labeled positive: {extra_positive[:5]}")
    infeasible = [s.uid for s in samples if s.reduced_token_ids
                  and _required_frames(s.reduced_token_ids) > _encoder_frames(s.duration_seconds)]
    if infeasible:
        raise RuntimeError(f"{len(infeasible)} CTC-infeasible samples, e.g. {infeasible[:5]}")
    print(
        "V14 protocol audit:",
        f"extra_sources={','.join(sorted(_EXTRA_SUMMARY))}",
        f"unshared_sources={','.join(unused) or 'none'}",
        f"freq_masks={FREQ_MASKS}x{FREQ_WIDTH}",
        f"fill={FREQ_FILL}",
        f"min_bin={FREQ_MIN_BIN}",
        "time_masks=0",
        f"partial_stage_steps={PARTIAL_STAGE_STEPS or 'off'}",
        "extra_negatives_all_negative=yes",
        "ctc_feasible=yes",
    )


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def training_signature(args, ratios, manifest_sha1, tokenizer_path, old_piece_ids):
    signature = _V12_2_TRAINING_SIGNATURE(
        args, ratios, manifest_sha1, tokenizer_path, old_piece_ids
    )
    signature.update({
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "trainer_variant": TRAINER_VARIANT,
        "spec_augment": {
            "type": "frequency_mask_only",
            "freq_masks": FREQ_MASKS,
            "freq_width_max_bins": FREQ_WIDTH,
            "time_masks": 0,
            "fill": {"mean": "utterance_mean_over_valid_frames",
                     "floor": "utterance_minimum_over_valid_frames",
                     "zero": "literal_zero"}[FREQ_FILL],
            "min_start_bin": FREQ_MIN_BIN,
            "applied": "training_forward_only",
        },
        "partial": (
            {"stage_steps": PARTIAL_STAGE_STEPS,
             "controller": "fixed_optimizer_steps_per_stage",
             "final_test": "skipped"}
            if PARTIAL_STAGE_STEPS else None
        ),
        "extra_negatives": {
            "manifest": str(NEGATIVE_MANIFEST),
            "manifest_sha1": _file_sha1(NEGATIVE_MANIFEST),
            "sources": dict(_EXTRA_SUMMARY),
        },
        "negative_source_shares": dict(SOURCE_SHARES),
        "training_sampler": "class_then_source_share_then_hard_easy_then_family",
    })
    if PARTIAL_STAGE_STEPS:
        signature["final_test_executed"] = False
        signature["trainer_variant"] = TRAINER_VARIANT + "_partial"
    return signature


# ============================================================
# Wiring
# ============================================================

def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    options, remaining = option_parser().parse_known_args(argv)
    configure(options)
    previous.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    previous.TRAINER_VARIANT = TRAINER_VARIANT
    previous.load_infore2_samples = load_infore2_samples
    previous.stratified_family_negative_sequence = stratified_family_negative_sequence
    previous.audit_samples = audit_samples
    previous.training_signature = training_signature
    previous.train_v12 = train_v12
    v12.build_student_class = build_student_class
    if PARTIAL_STAGE_STEPS:
        v12.curriculum_observe_steps = fixed_schedule_observe_steps
    return previous.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
