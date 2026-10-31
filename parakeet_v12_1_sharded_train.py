#!/usr/bin/env python3
"""V12.1 trainer variant for prepared mono-16k shard-local wakeword data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import parakeet_v12_1_train as v12_1
import parakeet_v12_train as v12


CHECKPOINT_FORMAT_VERSION = 3
TRAINER_VARIANT = "v12.1_reduced_streaming_pure_ctc_sharded16k"


def option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--no-shard-aware-batching",
        dest="shard_aware_batching",
        action="store_false",
        default=True,
        help="Ablation only: disable bounded row-group locality ordering.",
    )
    parser.add_argument("--shard-row-group-size", type=int, default=64)
    parser.add_argument("--shard-locality-block-size", type=int, default=64)
    parser.add_argument(
        "--initial-curriculum-stage",
        type=int,
        default=0,
        help=(
            "For a new weights-only warm start, begin at this ratio-stage index. "
            "Optimizer and plateau state intentionally start fresh."
        ),
    )
    parser.add_argument(
        "--allow-nonprepared-wakeword",
        dest="require_prepared_wakeword_16khz",
        action="store_false",
        default=True,
        help="Ablation only: permit file-backed or non-16k wakeword records.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = option_parser()
    options, remaining = parser.parse_known_args(raw)
    if "--help" in raw or "-h" in raw:
        try:
            v12_1.parse_args(remaining)
        except SystemExit:
            print("\nV12.1 sharded-data options:")
            print(parser.format_help())
            raise
    args = v12_1.parse_args(remaining)
    for name, value in vars(options).items():
        setattr(args, name, value)
    args.required_prepared_audio_subtype = "FLOAT"
    if args.shard_row_group_size < 1:
        raise ValueError("--shard-row-group-size must be positive")
    if args.shard_locality_block_size < 1:
        raise ValueError("--shard-locality-block-size must be positive")
    ratios = tuple(
        int(value.strip()) for value in args.negative_positive_ratios.split(",")
    )
    if not 0 <= args.initial_curriculum_stage < len(ratios):
        raise ValueError("--initial-curriculum-stage is outside the configured stages")
    return args


def training_signature(
    args: argparse.Namespace,
    ratios: Sequence[int],
    manifest_sha1: str,
    tokenizer_path: Path,
    old_piece_ids: Sequence[int],
) -> dict[str, object]:
    signature = v12_1.training_signature(
        args, ratios, manifest_sha1, tokenizer_path, old_piece_ids
    )
    signature.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_variant": TRAINER_VARIANT,
            "wakeword_storage_layout": "parquet_wav_float32_mono_16khz",
            "wakeword_pre_resampled": True,
            "prepared_audio_subtype": str(args.required_prepared_audio_subtype),
            "shard_aware_batching": bool(args.shard_aware_batching),
            "shard_row_group_size": int(args.shard_row_group_size),
            "shard_locality_block_size": int(args.shard_locality_block_size),
            "require_prepared_wakeword_16khz": bool(
                args.require_prepared_wakeword_16khz
            ),
            "initial_curriculum_stage": int(args.initial_curriculum_stage),
        }
    )
    return signature


def main(argv: Sequence[str] | None = None) -> int:
    v12.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v12.parse_args = parse_args
    v12.training_signature = training_signature
    return v12.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
