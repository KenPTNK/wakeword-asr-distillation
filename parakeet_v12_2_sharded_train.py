#!/usr/bin/env python3
"""V12.2 development trainer for the reduced-vocabulary streaming KWS model.

This variant keeps the V11 L5/H192/M64 architecture and pure CTC objective,
but corrects the V12.1 experiment protocol:

* a new deterministic speaker shuffle replaces every earlier V12/V12.1 split;
* InfoRE2 validation rows already seen by V11 are returned to training;
* wakeword sampling is speaker/family first, then rotates augmentation variants;
* every CTC sequence loss uses one input-frame normalization;
* validation is class/speaker/family balanced and runs in FP32 by default;
* optimization supports decoder-only warm-up, discriminative LRs, and a fixed
  encoder-frame target per optimizer update;
* final export selects global validation F1 and runs one complete held-out test
  audit only after training has converged.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import parakeet_v12_1_sharded_train as v12_1_sharded
import parakeet_v12_train as v12


CHECKPOINT_FORMAT_VERSION = 6
TRAINER_VARIANT = "v12.2_reduced_streaming_pure_ctc_sharded16k_reshuffled_val4_test4"

_BASE_PARSE_ARGS = v12_1_sharded.parse_args
_BASE_TRAINING_SIGNATURE = v12_1_sharded.training_signature
_ORIGINAL_LOAD_WAKEWORD = v12.load_prepared_wakeword_samples
_ORIGINAL_LOAD_INFORE2 = v12.load_infore2_samples
_ORIGINAL_AUDIT_SAMPLES = v12.audit_samples
_ORIGINAL_PURE_CTC = v12.pure_ctc_per_sample
_ORIGINAL_TRAIN = v12.train_v12

_NEGATIVE_SLOTS_PER_EPOCH = 80_000
V12_2_SPEAKER_SPLITS: dict[str, str] = {}


def option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--negative-slots-per-epoch", type=int, default=80_000)
    parser.add_argument("--speaker-split-seed", type=int, default=2026)
    parser.add_argument(
        "--target-encoder-frames-per-update", type=int, default=2_000
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--decoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-only-optimizer-steps", type=int, default=2_000)
    parser.add_argument(
        "--validation-amp-dtype",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
    )
    parser.add_argument(
        "--test-amp-dtype",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
    )
    parser.add_argument("--curriculum-ema-alpha", type=float, default=0.40)
    parser.add_argument("--detection-veto-patience", type=int, default=6)
    parser.add_argument(
        "--detection-min-absolute-improvement", type=float, default=0.0005
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    global _NEGATIVE_SLOTS_PER_EPOCH
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = option_parser()
    options, remaining = parser.parse_known_args(raw)
    if "--help" in raw or "-h" in raw:
        try:
            _BASE_PARSE_ARGS(remaining)
        except SystemExit:
            print("\nV12.2 protocol options:")
            print(parser.format_help())
            raise
    args = _BASE_PARSE_ARGS(remaining)
    for name, value in vars(options).items():
        setattr(args, name, value)
    if args.negative_slots_per_epoch < 1:
        raise ValueError("--negative-slots-per-epoch must be positive")
    if args.speaker_split_seed < 0:
        raise ValueError("--speaker-split-seed must be non-negative")
    if args.target_encoder_frames_per_update < 1:
        raise ValueError("--target-encoder-frames-per-update must be positive")
    if args.encoder_learning_rate <= 0.0 or args.decoder_learning_rate <= 0.0:
        raise ValueError("Discriminative learning rates must be positive")
    if args.head_only_optimizer_steps < 0:
        raise ValueError("--head-only-optimizer-steps must be non-negative")
    if not 0.0 < args.curriculum_ema_alpha <= 1.0:
        raise ValueError("--curriculum-ema-alpha must be in (0, 1]")
    if args.detection_veto_patience < 0:
        raise ValueError("--detection-veto-patience must be non-negative")
    if args.detection_min_absolute_improvement < 0.0:
        raise ValueError("--detection-min-absolute-improvement must be non-negative")

    # These are protocol invariants, not optional convenience flags.
    args.development_only = False
    args.final_selection_checkpoint = "best_f1.pt"
    args.restore_stage_best = False
    _NEGATIVE_SLOTS_PER_EPOCH = int(args.negative_slots_per_epoch)
    v12.CURRICULUM_EMA_ALPHA = float(args.curriculum_ema_alpha)
    return args


def family_key(sample: v12.Sample) -> tuple[str, ...]:
    if sample.source_dataset == "InfoRE2":
        return (sample.source_dataset, sample.uid)
    return (
        sample.source_dataset,
        str(sample.speaker_id),
        v12.normalize_text(sample.transcript),
        str(sample.utterance_index),
    )


def family_balanced_sequence(
    samples: Sequence[v12.Sample],
    pool: Sequence[int],
    count: int,
    seed: int,
    label: str,
) -> list[int]:
    """Select speaker -> family -> rotating augmentation variant."""
    if count <= 0 or not pool:
        return []
    by_speaker: dict[str, dict[tuple[str, ...], list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for index in pool:
        sample = samples[int(index)]
        speaker = (
            str(sample.speaker_id)
            if sample.speaker_id is not None
            else f"__{sample.source_dataset}__"
        )
        by_speaker[speaker][family_key(sample)].append(int(index))
    speakers = sorted(by_speaker)
    if not speakers:
        return []
    # Sorting the complete InfoRE2 family set inside the selection loop made
    # every epoch-plan rebuild effectively O(selected * families log families).
    # The family membership is immutable for this call, so sort each speaker
    # exactly once. This preserves the previous deterministic order and RNG.
    families_by_speaker = {
        speaker: sorted(families)
        for speaker, families in by_speaker.items()
    }
    label_hash = int.from_bytes(v12.stable_digest(label)[:8], "big")
    speaker_cursor = 0
    speaker_cycle = -1
    speaker_order: list[str] = []
    family_cursors = collections.defaultdict(int)
    family_orders: dict[str, list[tuple[str, ...]]] = {}
    variant_cursors = collections.defaultdict(int)
    variant_orders: dict[tuple[str, tuple[str, ...], int], list[int]] = {}
    output: list[int] = []
    while len(output) < count:
        current_speaker_cycle = speaker_cursor // len(speakers)
        if current_speaker_cycle != speaker_cycle:
            speaker_cycle = current_speaker_cycle
            speaker_order = list(speakers)
            random.Random(seed + label_hash + speaker_cycle * 1_000_003).shuffle(
                speaker_order
            )
        speaker = speaker_order[speaker_cursor % len(speakers)]
        speaker_cursor += 1

        families = families_by_speaker[speaker]
        family_position = family_cursors[speaker]
        family_cycle = family_position // len(families)
        family_order_key = f"{speaker}|{family_cycle}"
        if family_order_key not in family_orders:
            order = list(families)
            random.Random(
                seed
                + label_hash
                + int.from_bytes(v12.stable_digest(family_order_key)[:8], "big")
            ).shuffle(order)
            family_orders[family_order_key] = order
        family = family_orders[family_order_key][family_position % len(families)]
        family_cursors[speaker] += 1

        variants = by_speaker[speaker][family]
        variant_position = variant_cursors[(speaker, family)]
        if len(variants) == 1:
            output.append(variants[0])
            variant_cursors[(speaker, family)] += 1
            continue
        variant_cycle = variant_position // len(variants)
        variant_key = (speaker, family, variant_cycle)
        if variant_key not in variant_orders:
            order = list(variants)
            random.Random(
                seed
                + label_hash
                + int.from_bytes(v12.stable_digest(*variant_key)[:8], "big")
            ).shuffle(order)
            variant_orders[variant_key] = order
        output.append(variant_orders[variant_key][variant_position % len(variants)])
        variant_cursors[(speaker, family)] += 1
    return output


def stratified_family_negative_sequence(
    samples: Sequence[v12.Sample],
    negative_slots: int,
    new_data_share: float,
    hard_negative_share: float,
    seed: int,
) -> list[int]:
    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.split != "train" or sample.is_positive:
            continue
        source = "infore2" if sample.source_dataset == "InfoRE2" else "new"
        difficulty = "hard" if sample.is_hard_negative else "easy"
        groups[(source, difficulty)].append(index)
    available_sources = [
        source
        for source in ("new", "infore2")
        if groups[(source, "hard")] or groups[(source, "easy")]
    ]
    if not available_sources:
        raise RuntimeError("No negative training samples")
    if len(available_sources) == 1:
        source_counts = {available_sources[0]: negative_slots}
    else:
        new_count = int(round(negative_slots * new_data_share))
        source_counts = {"new": new_count, "infore2": negative_slots - new_count}
    selected: list[int] = []
    for source, source_count in source_counts.items():
        hard_pool = groups[(source, "hard")]
        easy_pool = groups[(source, "easy")]
        if hard_pool and easy_pool:
            hard_count = int(round(source_count * hard_negative_share))
        elif hard_pool:
            hard_count = source_count
        else:
            hard_count = 0
        easy_count = source_count - hard_count
        selected.extend(
            family_balanced_sequence(
                samples, hard_pool, hard_count, seed, f"{source}-hard"
            )
        )
        selected.extend(
            family_balanced_sequence(
                samples, easy_pool, easy_count, seed, f"{source}-easy"
            )
        )
    random.Random(seed + 9_187).shuffle(selected)
    return selected


def build_epoch_batch_plan(
    samples: Sequence[v12.Sample],
    negative_positive_ratio: int,
    max_utterances_per_batch: int,
    max_padded_audio_seconds: float,
    new_data_negative_share: float,
    hard_negative_share: float,
    seed: int,
    epoch: int,
    shard_aware_batching: bool = False,
    shard_row_group_size: int = 64,
    shard_locality_block_size: int = 64,
) -> list[list[int]]:
    positive_pool = [
        index
        for index, sample in enumerate(samples)
        if sample.split == "train" and sample.is_positive
    ]
    negative_pool = [
        index
        for index, sample in enumerate(samples)
        if sample.split == "train" and not sample.is_positive
    ]
    if not positive_pool or not negative_pool:
        raise RuntimeError("Training needs both positive and negative pools")
    negative_slots = int(_NEGATIVE_SLOTS_PER_EPOCH)
    positive_slots = max(1, math.ceil(negative_slots / negative_positive_ratio))
    batch_size = int(max_utterances_per_batch)
    total_slots = negative_slots + positive_slots
    batch_count = math.ceil(total_slots / batch_size)
    if positive_slots < batch_count or negative_slots < batch_count:
        raise RuntimeError("Cannot keep both classes in every training batch")
    epoch_seed = seed + int(epoch) * 1_000_003 + int(negative_positive_ratio) * 65_537
    base_size, larger_count = divmod(total_slots, batch_count)
    batch_sizes = [base_size + int(i < larger_count) for i in range(batch_count)]
    random.Random(epoch_seed + 11).shuffle(batch_sizes)
    capacities = [size - 2 for size in batch_sizes]
    extra_positive_total = positive_slots - batch_count
    capacity_total = sum(capacities)
    ideals = (
        [capacity * extra_positive_total / capacity_total for capacity in capacities]
        if capacity_total
        else [0.0] * batch_count
    )
    extra_positive_counts = [int(math.floor(value)) for value in ideals]
    remainder = extra_positive_total - sum(extra_positive_counts)
    order = sorted(
        range(batch_count),
        key=lambda i: (
            -(ideals[i] - extra_positive_counts[i]),
            v12.stable_digest("v12.2-batch-remainder", epoch_seed, i),
        ),
    )
    for index in order[:remainder]:
        extra_positive_counts[index] += 1
    positive_counts = [1 + value for value in extra_positive_counts]
    negative_counts = [
        size - positive for size, positive in zip(batch_sizes, positive_counts)
    ]
    positives = family_balanced_sequence(
        samples, positive_pool, positive_slots, epoch_seed + 17, "positive"
    )
    negatives = stratified_family_negative_sequence(
        samples,
        negative_slots,
        new_data_negative_share,
        hard_negative_share,
        epoch_seed,
    )
    extra_positives = family_balanced_sequence(
        samples,
        positive_pool,
        len(positive_pool),
        epoch_seed + 29,
        "padding-positive",
    )
    if shard_aware_batching:
        positives = v12.order_by_storage_locality(
            positives,
            samples,
            epoch_seed + 31,
            "positive",
            shard_row_group_size,
            shard_locality_block_size,
        )
        negatives = v12.order_by_storage_locality(
            negatives,
            samples,
            epoch_seed + 37,
            "negative",
            shard_row_group_size,
            shard_locality_block_size,
        )
        extra_positives = v12.order_by_storage_locality(
            extra_positives,
            samples,
            epoch_seed + 41,
            "padding-positive",
            shard_row_group_size,
            shard_locality_block_size,
        )
    positive_cycle = itertools.cycle(extra_positives)
    plan: list[list[int]] = []
    positive_cursor = 0
    negative_cursor = 0
    for batch_index, (positive_count, negative_count) in enumerate(
        zip(positive_counts, negative_counts)
    ):
        batch = positives[positive_cursor : positive_cursor + positive_count]
        batch += negatives[negative_cursor : negative_cursor + negative_count]
        positive_cursor += positive_count
        negative_cursor += negative_count
        random.Random(epoch_seed + batch_index * 7_919).shuffle(batch)
        plan.extend(
            v12._split_to_padding_budget(
                batch, samples, max_padded_audio_seconds, positive_cycle
            )
        )
    if not plan or any(
        not any(samples[index].is_positive for index in batch) for batch in plan
    ):
        raise RuntimeError("V12.2 planner violated the positive-batch invariant")
    if positive_cursor != len(positives) or negative_cursor != len(negatives):
        raise RuntimeError("V12.2 planner did not consume every allocated slot")
    return plan


def reshuffle_speaker_splits(
    samples: Sequence[v12.Sample],
    split_seed: int,
) -> tuple[list[v12.Sample], dict[str, str]]:
    """Create a fresh 37/4/4 split using only dual-label val/test speakers."""
    labels_by_speaker: dict[str, set[bool]] = collections.defaultdict(set)
    for sample in samples:
        if sample.speaker_id is None:
            raise RuntimeError("WakewordV12 sample is missing speaker_id")
        labels_by_speaker[str(sample.speaker_id)].add(bool(sample.is_positive))
    speakers = sorted(labels_by_speaker)
    eligible = [speaker for speaker in speakers if labels_by_speaker[speaker] == {False, True}]
    if len(eligible) < 8:
        raise RuntimeError(
            "Need at least eight dual-label speakers for a 4-validation/4-test split; "
            f"found {len(eligible)}"
        )
    random.Random(int(split_seed)).shuffle(eligible)
    validation_speakers = set(eligible[:4])
    test_speakers = set(eligible[4:8])
    speaker_splits = {speaker: "train" for speaker in speakers}
    for speaker in validation_speakers:
        speaker_splits[speaker] = "validation"
    for speaker in test_speakers:
        speaker_splits[speaker] = "test"
    remapped = [
        replace(sample, split=speaker_splits[str(sample.speaker_id)])
        for sample in samples
    ]
    print(
        "V12.2 fresh speaker split:",
        f"seed={int(split_seed)}",
        f"train={len(speakers) - 8}",
        f"validation={sorted(validation_speakers)}",
        f"test={sorted(test_speakers)}",
    )
    return remapped, speaker_splits


def load_prepared_wakeword_samples(*args, **kwargs):
    global V12_2_SPEAKER_SPLITS
    samples, _embedded_splits = _ORIGINAL_LOAD_WAKEWORD(*args, **kwargs)
    trainer_args = args[3] if len(args) > 3 else kwargs["args"]
    remapped, speaker_splits = reshuffle_speaker_splits(
        samples,
        int(trainer_args.speaker_split_seed),
    )
    V12_2_SPEAKER_SPLITS = dict(speaker_splits)
    return remapped, speaker_splits


def load_infore2_samples(*args, **kwargs):
    samples = _ORIGINAL_LOAD_INFORE2(*args, **kwargs)
    return [
        replace(sample, split="train") if sample.split == "validation" else sample
        for sample in samples
    ]


def audit_samples(
    samples: Sequence[v12.Sample],
    speaker_splits: dict[str, str],
    require_test: bool = False,
) -> None:
    """Run the base audit, then enforce the immutable V12.2 data protocol."""
    _ORIGINAL_AUDIT_SAMPLES(
        samples,
        speaker_splits,
        require_test=require_test,
    )
    if not require_test:
        raise RuntimeError("V12.2 requires a held-out test split")
    test_samples = [sample for sample in samples if sample.split == "test"]
    if not test_samples:
        raise RuntimeError("V12.2 held-out test split is empty")
    infore2_development = [
        sample
        for sample in samples
        if sample.source_dataset == "InfoRE2" and sample.split != "train"
    ]
    if infore2_development:
        raise RuntimeError(
            "All InfoRE2 rows were exposed to V11 and must remain in training; "
            f"found {len(infore2_development)} non-train rows"
        )
    validation_speakers = {
        str(sample.speaker_id)
        for sample in samples
        if sample.source_dataset != "InfoRE2" and sample.split == "validation"
    }
    test_speakers = {
        str(sample.speaker_id)
        for sample in test_samples
        if sample.source_dataset != "InfoRE2"
    }
    train_speakers = {
        str(sample.speaker_id)
        for sample in samples
        if sample.source_dataset != "InfoRE2" and sample.split == "train"
    }
    if len(validation_speakers) != 4 or len(test_speakers) != 4:
        raise RuntimeError(
            "V12.2 fresh split must contain exactly 4 validation and 4 test speakers"
        )
    if validation_speakers & test_speakers or validation_speakers & train_speakers or test_speakers & train_speakers:
        raise RuntimeError("V12.2 speaker split is not disjoint")
    for split, expected_speakers in (
        ("validation", validation_speakers),
        ("test", test_speakers),
    ):
        for label in (False, True):
            observed = {
                str(sample.speaker_id)
                for sample in samples
                if sample.source_dataset != "InfoRE2"
                and sample.split == split
                and bool(sample.is_positive) == label
            }
            if observed != expected_speakers:
                raise RuntimeError(
                    f"Every {split} speaker must contain label={label}; "
                    f"expected={sorted(expected_speakers)} actual={sorted(observed)}"
                )
    print(
        "V12.2 protocol audit:",
        "validation_speakers=4",
        "test_speakers=4",
        "infore2_nontrain=0",
        f"validation_ids={','.join(sorted(validation_speakers))}",
        f"test_ids={','.join(sorted(test_speakers))}",
        "both_labels=yes",
    )


def pure_ctc_per_input_frame(*args, **kwargs):
    _legacy_normalized, raw_losses = _ORIGINAL_PURE_CTC(*args, **kwargs)
    student_lengths = args[1] if len(args) > 1 else kwargs["student_lengths"]
    normalizers = student_lengths.to(raw_losses.device, dtype=raw_losses.dtype).clamp_min(1)
    return raw_losses / normalizers, raw_losses


def development_sample_weights(
    samples: Sequence[v12.Sample],
) -> dict[str, float]:
    groups: dict[bool, dict[str, dict[tuple[str, ...], list[v12.Sample]]]] = (
        collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list))
        )
    )
    for sample in samples:
        if sample.split != "validation":
            continue
        if sample.source_dataset == "InfoRE2":
            raise RuntimeError("InfoRE2 must not enter V12.2 development selection")
        speaker = str(sample.speaker_id)
        groups[bool(sample.is_positive)][speaker][family_key(sample)].append(sample)
    weights: dict[str, float] = {}
    for label in (False, True):
        speakers = groups[label]
        if not speakers:
            raise RuntimeError(f"Development set has no label={label} samples")
        speaker_weight = 1.0 / len(speakers)
        for speaker, families in speakers.items():
            family_weight = speaker_weight / len(families)
            for variants in families.values():
                sample_weight = family_weight / len(variants)
                for sample in variants:
                    weights[sample.uid] = sample_weight
    return weights


def train_v12(*args, **kwargs):
    samples = args[2] if len(args) > 2 else kwargs["samples"]
    v12.EVALUATION_SAMPLE_WEIGHTS = development_sample_weights(samples)
    positive_weight = sum(
        weight
        for sample in samples
        if sample.is_positive
        for weight in [v12.EVALUATION_SAMPLE_WEIGHTS.get(sample.uid, 0.0)]
    )
    negative_weight = sum(
        weight
        for sample in samples
        if not sample.is_positive
        for weight in [v12.EVALUATION_SAMPLE_WEIGHTS.get(sample.uid, 0.0)]
    )
    print(
        "V12.2 development weights:",
        f"samples={len(v12.EVALUATION_SAMPLE_WEIGHTS):,}",
        f"positive_total={positive_weight:.6f}",
        f"negative_total={negative_weight:.6f}",
    )
    if not math.isclose(positive_weight, 1.0, abs_tol=1e-9):
        raise RuntimeError("Positive development weights do not sum to one")
    if not math.isclose(negative_weight, 1.0, abs_tol=1e-9):
        raise RuntimeError("Negative development weights do not sum to one")
    return _ORIGINAL_TRAIN(*args, **kwargs)


def training_signature(
    args: argparse.Namespace,
    ratios: Sequence[int],
    manifest_sha1: str,
    tokenizer_path: Path,
    old_piece_ids: Sequence[int],
) -> dict[str, object]:
    signature = _BASE_TRAINING_SIGNATURE(
        args, ratios, manifest_sha1, tokenizer_path, old_piece_ids
    )
    signature.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_variant": TRAINER_VARIANT,
            "data_protocol": "fresh_seeded_speaker_shuffle_train37_validation4_test4",
            "speaker_split_seed": int(args.speaker_split_seed),
            "speaker_splits": dict(sorted(V12_2_SPEAKER_SPLITS.items())),
            "infore2_validation_policy": "return_all_preexposed_rows_to_train",
            "heldout_test_policy": "evaluate_once_after_training_converges",
            "training_sampler": "class_then_speaker_then_family_then_variant",
            "negative_slots_per_epoch": int(args.negative_slots_per_epoch),
            "nonempty_target_loss": "ctc_nll_normalized_per_input_frame",
            "empty_negative_target_loss": "all_blank_ctc_nll_normalized_per_input_frame",
            "target_encoder_frames_per_update": int(
                args.target_encoder_frames_per_update
            ),
            "encoder_learning_rate": float(args.encoder_learning_rate),
            "decoder_learning_rate": float(args.decoder_learning_rate),
            "head_only_optimizer_steps": int(args.head_only_optimizer_steps),
            "validation_amp_dtype": str(args.validation_amp_dtype),
            "test_amp_dtype": str(args.test_amp_dtype),
            "validation_weighting": "class_speaker_family_balanced",
            "curriculum_metric": "ema_of_balanced_macro_ctc_per_input_frame",
            "curriculum_ema_alpha": float(args.curriculum_ema_alpha),
            "detection_veto_metric": "balanced_development_f1",
            "detection_veto_patience": int(args.detection_veto_patience),
            "detection_min_absolute_improvement": float(
                args.detection_min_absolute_improvement
            ),
            "stage_handoff": "plateau_endpoint_without_optimizer_rewind",
            "final_selection": "global_best_validation_f1",
            "final_test_executed": True,
        }
    )
    return signature


def main(argv: Sequence[str] | None = None) -> int:
    v12.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v12.parse_args = parse_args
    v12.training_signature = training_signature
    v12.load_prepared_wakeword_samples = load_prepared_wakeword_samples
    v12.load_infore2_samples = load_infore2_samples
    v12.audit_samples = audit_samples
    v12.build_epoch_batch_plan = build_epoch_batch_plan
    v12.pure_ctc_per_sample = pure_ctc_per_input_frame
    v12.train_v12 = train_v12
    return v12.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
