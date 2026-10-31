#!/usr/bin/env python3
"""V12.1 controller and phrase-score audit on top of the audited V12 trainer.

V12.1 deliberately reuses V12's dataset adapter, reduced vocabulary, pure CTC
loss, and V11 L5/H192/M64 chunk-causal model.  Its behavioral differences are:

* a stage has no epoch cap and continues while macro CTC materially improves;
* a plateau restores the complete stage-best optimization state before handoff;
* the next stage uses a configurable multiplicative learning-rate decay;
* global and per-stage best-F1 checkpoints are retained;
* validation can audit a CTC-path phrase score without tuning on held-out test.

The default detector remains the audited V12 greedy detector.  Phrase-score
detection is opt-in and requires an explicitly frozen validation threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import parakeet_v12_train as v12


CHECKPOINT_FORMAT_VERSION = 2
TRAINER_VARIANT = "v12.1_reduced_streaming_pure_ctc"

_v12_parse_args = v12.parse_args
_v12_training_signature = v12.training_signature


def _v12_1_option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage-lr-decay",
        type=float,
        default=0.70,
        help=(
            "Multiply the base LR by this factor after each stage-best restore. "
            "Default: 0.70."
        ),
    )
    parser.add_argument(
        "--no-restore-stage-best",
        dest="restore_stage_best",
        action="store_false",
        default=True,
        help="Ablation only: hand the plateau endpoint to the next stage.",
    )
    parser.add_argument(
        "--detection-mode",
        choices=("greedy", "phrase_score"),
        default="greedy",
        help="Detector used for validation and the one-shot final test.",
    )
    parser.add_argument(
        "--phrase-score-threshold",
        type=float,
        default=None,
        help=(
            "Frozen validation-derived CTC phrase-score threshold. Required only "
            "with --detection-mode phrase_score; never fitted on test."
        ),
    )
    parser.add_argument(
        "--no-phrase-score-audit",
        dest="phrase_score_audit",
        action="store_false",
        default=True,
        help="Disable validation-only phrase-score threshold sweeps.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    option_parser = _v12_1_option_parser()
    v12_1_options, remaining = option_parser.parse_known_args(raw)
    if "--help" in raw or "-h" in raw:
        try:
            _v12_parse_args(remaining)
        except SystemExit:
            print("\nV12.1-specific options:")
            print(option_parser.format_help())
            raise
    args = _v12_parse_args(remaining)
    for name, value in vars(v12_1_options).items():
        setattr(args, name, value)
    if not 0.0 < float(args.stage_lr_decay) <= 1.0:
        raise ValueError("--stage-lr-decay must be in (0, 1]")
    if args.detection_mode == "phrase_score" and args.phrase_score_threshold is None:
        raise ValueError(
            "--detection-mode phrase_score requires --phrase-score-threshold "
            "selected on validation"
        )
    return args


def training_signature(
    args: argparse.Namespace,
    ratios: Sequence[int],
    manifest_sha1: str,
    tokenizer_path: Path,
    old_piece_ids: Sequence[int],
) -> dict[str, object]:
    signature = _v12_training_signature(
        args,
        ratios,
        manifest_sha1,
        tokenizer_path,
        old_piece_ids,
    )
    signature.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_variant": TRAINER_VARIANT,
            "stage_has_epoch_cap": False,
            "stage_handoff": "restore_full_stage_best_optimization_state",
            "stage_lr_decay": float(args.stage_lr_decay),
            "detection_mode": str(args.detection_mode),
            "phrase_score_threshold": args.phrase_score_threshold,
            "phrase_score_audit": bool(args.phrase_score_audit),
            "phrase_score_definition": (
                "log_P_collapsed_path_contains_wake_minus_log_P_not_contains"
            ),
        }
    )
    return signature


def main(argv: Sequence[str] | None = None) -> int:
    # The imported module owns all heavy runtime globals.  Patching only these
    # versioned entry points keeps the V12 implementation reusable while giving
    # V12.1 an incompatible resume signature and its own checkpoint directories.
    v12.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v12.parse_args = parse_args
    v12.training_signature = training_signature
    return v12.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
