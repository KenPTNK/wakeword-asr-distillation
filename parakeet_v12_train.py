#!/usr/bin/env python3
"""V12 reduced-vocabulary, chunk-causal, pure-CTC wake-word trainer.

V12 keeps the V11 L5/H192/M64 streaming architecture but reduces the decoder
from the 1,024 Vietnamese SentencePiece pieces plus blank to exactly four
pieces plus blank::

    0: ▁năng    1: ▁kh    2: iếu    3: ▁ơi    4: CTC blank

Transcripts are still encoded by the original Parakeet SentencePiece model.
All token IDs outside the four-piece vocabulary are discarded.  Consequently
``không nha`` can have target ``[1]`` while ``nắng`` has an empty CTC target.

Training combines the local InfoRE2 aligned Parquet corpus with a speaker-folder
dataset.  Every training batch contains at least one true wake-phrase positive.
The negative:positive curriculum defaults to 1:1, 2:1, 4:1, 8:1, and 10:1;
advancement is driven by an epoch-spaced validation plateau and is fully
checkpoint-resumable.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import importlib.metadata as importlib_metadata
import io
import itertools
import json
import math
import os
import platform
import random
import re
import signal
import socket
import sys
import tempfile
import time
import unicodedata
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


TARGET_SAMPLE_RATE = 16_000
FEATURE_HOP_SECONDS = 0.01
SUBSAMPLING_FACTOR = 8
STREAMING_CHUNK_SECONDS = 0.64
STREAMING_CHUNK_MEL_FRAMES = 64
STREAMING_CHUNK_ENCODER_FRAMES = 8

STUDENT_N_LAYERS = 5
STUDENT_D_MODEL = 192
STUDENT_N_MELS = 64
STUDENT_N_HEADS = 8
STUDENT_FF_EXPANSION = 4
STUDENT_CONV_KERNEL = 9

REDUCED_PIECES = ("▁năng", "▁kh", "iếu", "▁ơi")
REDUCED_VOCABULARY = REDUCED_PIECES
REDUCED_OUTPUT_DIM = len(REDUCED_VOCABULARY) + 1
CTC_BLANK_ID = len(REDUCED_VOCABULARY)
WAKE_SEQUENCE = tuple(range(len(REDUCED_VOCABULARY)))
TARGET_PHRASE = "năng khiếu ơi"
TRANSCRIPT_COLUMN_PRIORITY = (
    "transcription",
    "text",
    "normalized_text",
    "raw_transcription",
)
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
CHECKPOINT_FORMAT_VERSION = 1


# Heavy lab-only dependencies are imported by install_runtime_imports(), after
# argparse has handled --help.  This makes the CLI and structural tests usable
# on a laptop that does not have the ASR environment installed.
np = None
pa = None
pq = None
sf = None
spm = None
torch = None
nn = None
F = None
librosa = None
AudioToMelSpectrogramPreprocessor = None
ConformerEncoder = None
ConvASRDecoder = None

# Optional hooks used by later audited trainer variants.  Empty/default values
# preserve the exact V12/V12.1 behavior.
EVALUATION_SAMPLE_WEIGHTS: dict[str, float] = {}
CURRICULUM_EMA_ALPHA = 1.0


@dataclass(frozen=True)
class Sample:
    uid: str
    source_dataset: str
    source_kind: str
    audio_path: str
    audio_row: int | None
    speaker_id: str | None
    transcript: str
    utterance_index: str
    duration_seconds: float
    reduced_token_ids: tuple[int, ...]
    is_positive: bool
    split: str
    original_path: str | None = None
    google_drive_object_id: str | None = None
    augmentation: str | None = None
    audio_md5: str | None = None
    sample_rate: int | None = None
    provenance_kind: str | None = None

    @property
    def is_hard_negative(self) -> bool:
        return not self.is_positive and bool(self.reduced_token_ids)


@dataclass
class CurriculumState:
    stage_index: int = 0
    epochs_in_stage: int = 0
    best_metric: float | None = None
    bad_validations: int = 0
    complete: bool = False
    validation_observations_in_stage: int = 0
    stage_start_optimizer_step: int = 0
    metric_ema: float | None = None
    best_detection_metric: float | None = None
    bad_detection_validations: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train V12: V11 chunk-causal architecture, four target pieces, "
            "pure CTC, and validation-driven positive/negative curriculum."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root used for checkpoints and logs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="CTCTKNK root containing manifests/aligned_manifest.jsonl and audio_shards/.",
    )
    parser.add_argument(
        "--wakeword-data-root",
        type=Path,
        required=False,
        help="Folder whose immediate subfolders are speaker IDs.",
    )
    parser.add_argument(
        "--wakeword-manifest",
        type=Path,
        default=None,
        help=(
            "Prepared wakeword_v12_manifest.jsonl containing raw and recovered "
            "Drive-duplicate variants. Preferred over filename-only folder loading."
        ),
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=None,
        help=(
            "Original 1,024-piece Parakeet SentencePiece .model. Defaults to "
            "PROJECT_ROOT/checkpoints/parakeet_teacher_tokenizer.model."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "V11/V10 student .pt used for weights-only initialization. "
            "The 1,025-row decoder is remapped to four pieces plus blank."
        ),
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Permit a new run without --init-checkpoint (not recommended).",
    )
    parser.add_argument("--run-name", default="v12_reduced_ctc_streaming")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--audio-extensions",
        default=",".join(DEFAULT_AUDIO_EXTENSIONS),
        help="Comma-separated extensions accepted in the speaker dataset.",
    )
    parser.add_argument("--new-validation-speaker-fraction", type=float, default=0.10)
    parser.add_argument("--new-test-speaker-fraction", type=float, default=0.10)
    parser.add_argument(
        "--speaker-split-file",
        type=Path,
        default=None,
        help="Optional JSON object mapping speaker ID to train/validation/test.",
    )
    parser.add_argument(
        "--infore2-validation-samples",
        type=int,
        default=4096,
        help="Deterministic InfoRE2 rows held out for validation.",
    )
    parser.add_argument(
        "--infore2-train-fraction",
        type=float,
        default=1.0,
        help="Deterministic fraction of non-validation InfoRE2 rows used for training.",
    )
    parser.add_argument(
        "--negative-positive-ratios",
        default="1,2,4,8,10",
        help="Comma-separated negative:positive curriculum stages.",
    )
    parser.add_argument(
        "--stage-min-epochs",
        type=int,
        default=2,
        help="Minimum completed epochs in a ratio stage before it may advance.",
    )
    parser.add_argument(
        "--stage-plateau-patience",
        type=int,
        default=3,
        help="Consecutive non-improving epoch validations required to advance/finish.",
    )
    parser.add_argument("--stage-min-relative-improvement", type=float, default=0.002)
    parser.add_argument("--stage-min-absolute-improvement", type=float, default=1e-4)
    parser.add_argument(
        "--validation-every-steps",
        type=int,
        default=0,
        help=(
            "Run validation and observe the curriculum every N optimizer steps. "
            "Zero preserves the original epoch-end validation controller."
        ),
    )
    parser.add_argument(
        "--stage-min-optimizer-steps",
        type=int,
        default=0,
        help=(
            "With step validation, require at least this many optimizer steps in "
            "a stage before it may advance."
        ),
    )
    parser.add_argument(
        "--stage-min-validations",
        type=int,
        default=1,
        help=(
            "With step validation, require this many validation observations in "
            "a stage before it may advance."
        ),
    )
    parser.add_argument(
        "--new-data-negative-share",
        type=float,
        default=0.25,
        help="Target share of negative slots drawn from the new speaker dataset.",
    )
    parser.add_argument(
        "--hard-negative-share",
        type=float,
        default=0.50,
        help="Within each negative source, target share with a non-empty reduced target.",
    )
    parser.add_argument("--max-utterances-per-batch", type=int, default=32)
    parser.add_argument("--max-padded-audio-seconds", type=float, default=512.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=10,
        help="Write/print durable train metrics every N optimizer steps.",
    )
    parser.add_argument(
        "--plot-every-steps",
        type=int,
        default=10,
        help="Refresh loss_curve.png/svg every N optimizer steps and every validation.",
    )
    parser.add_argument(
        "--validation-preview-samples",
        type=int,
        default=64,
        help="Maximum balanced positive/negative prediction rows retained per validation.",
    )
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Build/audit the unified manifest and batch plans without constructing the GPU model.",
    )
    parser.add_argument(
        "--skip-infore2",
        action="store_true",
        help="Audit/debug only the new speaker dataset.",
    )
    parser.add_argument(
        "--wandb-project",
        default="parakeet-vietnamese-distillation",
        help="Weights & Biases project name.",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-expected-username",
        default=None,
        help="Required online: refuse a credential belonging to a different account.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--wandb-tags",
        default="v12,wakeword,reduced-vocabulary,pure-ctc,streaming,curriculum",
    )
    parser.add_argument(
        "--no-wandb-backfill",
        action="store_true",
        help="Do not upload existing durable metric history on first W&B connection.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    ratios = tuple(int(value.strip()) for value in args.negative_positive_ratios.split(","))
    if not ratios or any(value < 1 for value in ratios):
        raise ValueError("--negative-positive-ratios must contain positive integers")
    if tuple(sorted(ratios)) != ratios:
        raise ValueError("--negative-positive-ratios must be non-decreasing")
    if args.stage_min_epochs < 1:
        raise ValueError("--stage-min-epochs must be >= 1")
    if args.stage_plateau_patience < 2:
        raise ValueError("--stage-plateau-patience must be >= 2")
    if args.validation_every_steps < 0:
        raise ValueError("--validation-every-steps must be >= 0")
    if args.stage_min_optimizer_steps < 0:
        raise ValueError("--stage-min-optimizer-steps must be >= 0")
    if args.stage_min_validations < 1:
        raise ValueError("--stage-min-validations must be >= 1")
    for name in (
        "new_validation_speaker_fraction",
        "new_test_speaker_fraction",
        "new_data_negative_share",
        "hard_negative_share",
        "infore2_train_fraction",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.new_validation_speaker_fraction + args.new_test_speaker_fraction >= 1.0:
        raise ValueError("New-data validation + test speaker fractions must be < 1")
    if args.max_utterances_per_batch < 2:
        raise ValueError("--max-utterances-per-batch must be >= 2")
    if args.max_padded_audio_seconds <= 0.0:
        raise ValueError("--max-padded-audio-seconds must be positive")
    if args.log_every_steps <= 0:
        raise ValueError("--log-every-steps must be positive")
    if args.plot_every_steps <= 0:
        raise ValueError("--plot-every-steps must be positive")
    if args.validation_preview_samples < 0:
        raise ValueError("--validation-preview-samples must be non-negative")
    if args.infore2_validation_samples < 0:
        raise ValueError("--infore2-validation-samples must be non-negative")
    if args.max_optimizer_steps < 0 or args.max_epochs < 0:
        raise ValueError("Safety caps must be non-negative")
    return ratios


def install_runtime_imports() -> None:
    global np, pa, pq, sf, spm, torch, nn, F, librosa
    global AudioToMelSpectrogramPreprocessor, ConformerEncoder, ConvASRDecoder

    import librosa as _librosa
    import numpy as _np
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    import sentencepiece as _spm
    import soundfile as _sf
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from nemo.collections.asr.modules import (
        AudioToMelSpectrogramPreprocessor as _AudioToMelSpectrogramPreprocessor,
        ConformerEncoder as _ConformerEncoder,
        ConvASRDecoder as _ConvASRDecoder,
    )

    np, pa, pq, sf, spm = _np, _pa, _pq, _sf, _spm
    torch, nn, F, librosa = _torch, _nn, _F, _librosa
    AudioToMelSpectrogramPreprocessor = _AudioToMelSpectrogramPreprocessor
    ConformerEncoder = _ConformerEncoder
    ConvASRDecoder = _ConvASRDecoder


def normalize_text(text: object) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    normalized: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        normalized.append(character if category.startswith(("L", "N")) else " ")
    return " ".join("".join(normalized).split())


def contains_word_sequence(normalized_text: str, phrase: str = TARGET_PHRASE) -> bool:
    words = normalized_text.split()
    target = normalize_text(phrase).split()
    width = len(target)
    return any(words[start : start + width] == target for start in range(len(words) - width + 1))


def contains_subsequence(sequence: Sequence[int], target: Sequence[int]) -> bool:
    width = len(target)
    return width > 0 and any(
        tuple(sequence[start : start + width]) == tuple(target)
        for start in range(len(sequence) - width + 1)
    )


def project_token_ids(
    original_ids: Iterable[int], old_to_reduced: dict[int, int]
) -> tuple[int, ...]:
    return tuple(old_to_reduced[value] for value in original_ids if value in old_to_reduced)


# Versioned policy: historical trainers retain their original targets.
TARGET_PROJECTION = "original_sentencepiece_then_exact_id_filter"


def transcript_target_ids(text, tokenizer, old_to_reduced):
    if TARGET_PROJECTION == "normalized_text_substrings_v1":
        # Scan in text order, retaining repeated occurrences and accents.
        # Whitespace never joins two fragments into a synthetic match.
        return tuple(
            {"năng": 0, "kh": 1, "iếu": 2, "ơi": 3}[match.group()]
            for match in re.finditer("năng|kh|iếu|ơi", normalize_text(text))
        )
    return project_token_ids(tokenizer.encode(text, out_type=int), old_to_reduced)


def stable_digest(*parts: object) -> bytes:
    return hashlib.sha1(":".join(str(part) for part in parts).encode("utf-8")).digest()


def stable_fraction(*parts: object) -> float:
    return int.from_bytes(stable_digest(*parts)[:8], "big") / float(2**64)


def parse_wakeword_filename(path: Path, speaker_id: str) -> tuple[str, str, str, str]:
    """Parse ``speaker_transcript_stt[_augmentation].ext`` without renaming it."""
    stem = unicodedata.normalize("NFKC", path.stem)
    prefix = speaker_id + "_"
    if not stem.startswith(prefix):
        raise ValueError(
            f"Filename speaker does not match parent folder: {path} (expected {speaker_id!r})"
        )
    fields = stem[len(prefix) :].split("_")
    if len(fields) < 2:
        raise ValueError(f"Filename must end in _STT: {path}")
    transcript, utterance_index = fields[:2]
    augmentation = "_".join(fields[2:]) if len(fields) > 2 else "base"
    if not transcript.strip() or not utterance_index.strip():
        raise ValueError(f"Empty transcript or STT in filename: {path}")
    if not re.fullmatch(r"[0-9]+", utterance_index):
        raise ValueError(f"STT must contain only digits: {path}")
    if len(fields) > 2 and not augmentation.strip():
        raise ValueError(f"Empty augmentation suffix in filename: {path}")
    return speaker_id, transcript, utterance_index, augmentation


def locate_data_root(args: argparse.Namespace) -> Path | None:
    if args.skip_infore2:
        return None
    if args.data_root is not None:
        candidate = args.data_root.expanduser().resolve()
    elif os.environ.get("CTCTKNK_DATA_ROOT"):
        candidate = Path(os.environ["CTCTKNK_DATA_ROOT"]).expanduser().resolve()
    else:
        raise ValueError("Pass --data-root or set CTCTKNK_DATA_ROOT")
    required = (
        candidate / "manifests" / "aligned_manifest.jsonl",
        candidate / "audio_shards",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"InfoRE2 data root is incomplete: {missing}")
    return candidate


def locate_tokenizer(args: argparse.Namespace) -> Path:
    candidate = (
        args.tokenizer_model.expanduser()
        if args.tokenizer_model is not None
        else args.project_root.expanduser()
        / "checkpoints"
        / "parakeet_teacher_tokenizer.model"
    ).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Original Parakeet tokenizer not found: {candidate}. "
            "Pass --tokenizer-model PATH."
        )
    return candidate


def build_reduced_tokenizer(tokenizer_path: Path):
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    old_ids: list[int] = []
    for piece in REDUCED_PIECES:
        token_id = int(tokenizer.piece_to_id(piece))
        if (
            token_id < 0
            or token_id == int(tokenizer.unk_id())
            or tokenizer.id_to_piece(token_id) != piece
        ):
            raise RuntimeError(f"Required exact piece is absent from the tokenizer: {piece!r}")
        old_ids.append(token_id)
    if len(set(old_ids)) != len(old_ids):
        raise RuntimeError(f"Reduced pieces map to duplicate old IDs: {old_ids}")
    old_to_reduced = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
    target_ids = tuple(
        int(value) for value in tokenizer.encode(TARGET_PHRASE, out_type=int)
    )
    projected = project_token_ids(target_ids, old_to_reduced)
    if projected != WAKE_SEQUENCE:
        pieces = tuple(tokenizer.id_to_piece(value) for value in target_ids)
        raise RuntimeError(
            "Target phrase does not reduce to the required four-token sequence: "
            f"full_ids={target_ids} pieces={pieces} projected={projected}"
        )
    return tokenizer, tuple(old_ids), old_to_reduced


def transcript_column_for_schema(column_names: Sequence[str]) -> str | None:
    for column in TRANSCRIPT_COLUMN_PRIORITY:
        if column in column_names:
            return column
    return None


def audio_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        duration = float(info.frames) / float(info.samplerate)
    except Exception:
        duration = float(librosa.get_duration(path=str(path)))
    if not math.isfinite(duration) or duration <= 0.0:
        raise RuntimeError(f"Invalid audio duration for {path}: {duration}")
    return duration


def load_speaker_split_file(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    with path.expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("--speaker-split-file must contain a JSON object")
    aliases = {"val": "validation", "valid": "validation"}
    result = {}
    for speaker, raw_split in payload.items():
        split = aliases.get(str(raw_split).lower(), str(raw_split).lower())
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split for speaker {speaker!r}: {raw_split!r}")
        result[str(speaker)] = split
    return result


def deterministic_speaker_splits(
    speaker_positive: dict[str, bool],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    speakers = sorted(
        speaker_positive,
        key=lambda speaker: stable_digest("speaker-split", seed, speaker),
    )
    count = len(speakers)
    if count == 0:
        return {}
    validation_count = (
        max(1, int(round(count * validation_fraction)))
        if validation_fraction > 0.0 and count >= 2
        else 0
    )
    test_count = (
        max(1, int(round(count * test_fraction)))
        if test_fraction > 0.0 and count >= 3
        else 0
    )
    while validation_count + test_count >= count:
        if test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
    result = {speaker: "train" for speaker in speakers}
    for speaker in speakers[:test_count]:
        result[speaker] = "test"
    for speaker in speakers[test_count : test_count + validation_count]:
        result[speaker] = "validation"

    # If enough positive speakers exist, make train and validation useful for
    # wake-word learning/evaluation without breaking speaker disjointness.
    positive_speakers = [speaker for speaker in speakers if speaker_positive[speaker]]
    desired_positive_splits = ["train"]
    if validation_count:
        desired_positive_splits.append("validation")
    if test_count:
        desired_positive_splits.append("test")
    required_splits = desired_positive_splits[: len(positive_speakers)]
    if positive_speakers:
        protected_splits: set[str] = set()
        for desired_split in required_splits:
            if any(
                result[speaker] == desired_split and speaker_positive[speaker]
                for speaker in speakers
            ):
                protected_splits.add(desired_split)
                continue
            donor = next(
                speaker
                for speaker in positive_speakers
                if result[speaker] != desired_split
                and (
                    result[speaker] not in protected_splits
                    or sum(
                        result[other] == result[speaker] and speaker_positive[other]
                        for other in speakers
                    )
                    > 1
                )
            )
            receiver = next(
                speaker
                for speaker in speakers
                if result[speaker] == desired_split and not speaker_positive[speaker]
            )
            result[donor], result[receiver] = result[receiver], result[donor]
            protected_splits.add(desired_split)
    return result


def load_new_speaker_samples(
    root: Path,
    extensions: set[str],
    tokenizer,
    old_to_reduced: dict[int, int],
    args: argparse.Namespace,
) -> tuple[list[Sample], dict[str, str]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    split_override = load_speaker_split_file(args.speaker_split_file)
    pending: list[dict[str, object]] = []
    speaker_positive: dict[str, bool] = {}
    seen_paths: set[str] = set()

    speaker_directories = sorted(path for path in root.iterdir() if path.is_dir())
    if not speaker_directories:
        raise RuntimeError(f"No speaker subfolders found under {root}")
    for speaker_directory in speaker_directories:
        speaker_id = unicodedata.normalize("NFKC", speaker_directory.name)
        speaker_positive.setdefault(speaker_id, False)
        for audio_path in sorted(path for path in speaker_directory.rglob("*") if path.is_file()):
            if audio_path.suffix.lower() not in extensions:
                continue
            parsed_speaker, transcript, utterance_index, augmentation = parse_wakeword_filename(
                audio_path, speaker_id
            )
            relative_path = audio_path.relative_to(root).as_posix()
            if relative_path in seen_paths:
                raise RuntimeError(f"Duplicate local path: {relative_path}")
            seen_paths.add(relative_path)
            normalized = normalize_text(transcript)
            if not normalized:
                raise RuntimeError(f"Transcript becomes empty after normalization: {audio_path}")
            full_ids = tuple(int(value) for value in tokenizer.encode(normalized, out_type=int))
            reduced = transcript_target_ids(normalized, tokenizer, old_to_reduced)
            positive = contains_word_sequence(normalized)
            if positive and not contains_subsequence(reduced, WAKE_SEQUENCE):
                pieces = tuple(tokenizer.id_to_piece(value) for value in full_ids)
                raise RuntimeError(
                    f"Positive transcript lost the wake sequence: {audio_path}; "
                    f"pieces={pieces}, reduced={reduced}"
                )
            speaker_positive[speaker_id] |= positive
            pending.append(
                {
                    "audio_path": audio_path.resolve(),
                    "speaker_id": speaker_id,
                    "transcript": transcript,
                    "utterance_index": utterance_index,
                    "augmentation": augmentation,
                    "duration_seconds": audio_duration(audio_path),
                    "reduced_token_ids": reduced,
                    "is_positive": positive,
                }
            )
    if not pending:
        raise RuntimeError(
            f"No supported audio found under {root}; extensions={sorted(extensions)}"
        )

    if split_override is None:
        speaker_splits = deterministic_speaker_splits(
            speaker_positive,
            args.new_validation_speaker_fraction,
            args.new_test_speaker_fraction,
            args.seed,
        )
    else:
        missing = sorted(set(speaker_positive) - set(split_override))
        extra = sorted(set(split_override) - set(speaker_positive))
        if missing or extra:
            raise RuntimeError(
                f"Speaker split file mismatch: missing={missing}, unknown={extra}"
            )
        speaker_splits = split_override

    samples = []
    for record in pending:
        speaker_id = str(record["speaker_id"])
        utterance_index = str(record["utterance_index"])
        audio_path = Path(record["audio_path"])
        relative_path = audio_path.relative_to(root).as_posix()
        samples.append(
            Sample(
                uid=f"WakewordNew:{speaker_id}:{utterance_index}:{relative_path}",
                source_dataset="WakewordNew",
                source_kind="file",
                audio_path=str(audio_path),
                audio_row=None,
                speaker_id=speaker_id,
                transcript=str(record["transcript"]),
                utterance_index=utterance_index,
                duration_seconds=float(record["duration_seconds"]),
                reduced_token_ids=tuple(record["reduced_token_ids"]),
                is_positive=bool(record["is_positive"]),
                split=speaker_splits[speaker_id],
                original_path=relative_path,
                augmentation=str(record["augmentation"]),
                provenance_kind="raw_unique_path",
            )
        )
    return samples, speaker_splits


def load_prepared_wakeword_samples(
    manifest_path: Path,
    tokenizer,
    old_to_reduced: dict[int, int],
    args: argparse.Namespace,
) -> tuple[list[Sample], dict[str, str]]:
    """Load the audited raw+Drive-variant manifest without parsing ID filenames."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    records: list[dict[str, object]] = []
    seen_uids: set[str] = set()
    seen_audio_locations: set[tuple[str, str, int | None]] = set()
    speaker_positive: dict[str, bool] = {}
    embedded_splits: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {
                "uid",
                "audio_path",
                "speaker_id",
                "transcript",
                "utterance_index",
                "augmentation",
                "duration_seconds",
                "is_positive",
                "split",
            }
            missing = sorted(required - set(record))
            if missing:
                raise RuntimeError(
                    f"Prepared wakeword manifest line {line_number} is missing {missing}"
                )
            uid = str(record["uid"])
            source_kind = str(record.get("source_kind", "file"))
            if source_kind not in {"file", "parquet"}:
                raise RuntimeError(
                    f"Unsupported prepared source_kind {source_kind!r}: {uid}"
                )
            unresolved_audio_path = Path(str(record["audio_path"])).expanduser()
            if not unresolved_audio_path.is_absolute():
                unresolved_audio_path = manifest_path.parent / unresolved_audio_path
            audio_path = str(unresolved_audio_path.resolve())
            audio_row = None
            if source_kind == "parquet":
                if record.get("audio_row") is None:
                    raise RuntimeError(
                        f"Prepared Parquet sample is missing audio_row: {uid}"
                    )
                audio_row = int(record["audio_row"])
                if audio_row < 0:
                    raise RuntimeError(f"Negative audio_row for {uid}: {audio_row}")
                configured_row_group_size = getattr(args, "shard_row_group_size", None)
                recorded_row_group_size = record.get("shard_row_group_size")
                if (
                    configured_row_group_size is not None
                    and recorded_row_group_size is not None
                    and int(configured_row_group_size) != int(recorded_row_group_size)
                ):
                    raise RuntimeError(
                        f"Shard row-group mismatch for {uid}: "
                        f"trainer={configured_row_group_size} "
                        f"manifest={recorded_row_group_size}"
                    )
            if uid in seen_uids:
                raise RuntimeError(f"Duplicate prepared-manifest UID: {uid}")
            location = (source_kind, audio_path, audio_row)
            if location in seen_audio_locations:
                raise RuntimeError(
                    f"Duplicate prepared-manifest audio location: {location}"
                )
            seen_uids.add(uid)
            seen_audio_locations.add(location)
            if not Path(audio_path).is_file():
                raise FileNotFoundError(audio_path)
            if bool(getattr(args, "require_prepared_wakeword_16khz", False)):
                if source_kind != "parquet":
                    raise RuntimeError(
                        f"Sharded V12.1 requires source_kind='parquet': {uid}"
                    )
                if int(record.get("sample_rate", 0)) != TARGET_SAMPLE_RATE:
                    raise RuntimeError(
                        f"Sharded V12.1 requires {TARGET_SAMPLE_RATE} Hz audio: {uid}"
                    )
                expected_subtype = getattr(
                    args, "required_prepared_audio_subtype", None
                )
                if (
                    expected_subtype is not None
                    and str(record.get("prepared_audio_subtype", ""))
                    != str(expected_subtype)
                ):
                    raise RuntimeError(
                        f"Sharded V12.1 requires prepared subtype "
                        f"{expected_subtype}: {uid}"
                    )
            speaker_id = unicodedata.normalize("NFKC", str(record["speaker_id"]))
            transcript = str(record["transcript"])
            normalized = normalize_text(transcript)
            if not normalized:
                raise RuntimeError(f"Transcript normalizes to empty: {uid}")
            positive = contains_word_sequence(normalized)
            if positive != bool(record["is_positive"]):
                raise RuntimeError(
                    f"Prepared positive label disagrees with transcript: {uid}; "
                    f"manifest={record['is_positive']} recomputed={positive}"
                )
            split = str(record["split"])
            if split not in {"train", "validation", "test"}:
                raise RuntimeError(f"Invalid prepared split {split!r}: {uid}")
            previous_split = embedded_splits.setdefault(speaker_id, split)
            if previous_split != split:
                raise RuntimeError(
                    f"Speaker leakage in prepared manifest: {speaker_id} has "
                    f"{previous_split!r} and {split!r}"
                )
            speaker_positive[speaker_id] = speaker_positive.get(speaker_id, False) or positive
            record = dict(record)
            record["source_kind"] = source_kind
            record["audio_path"] = audio_path
            record["audio_row"] = audio_row
            record["speaker_id"] = speaker_id
            record["transcript"] = transcript
            record["normalized_transcript"] = normalized
            record["recomputed_positive"] = positive
            records.append(record)
    if not records:
        raise RuntimeError(f"Prepared wakeword manifest is empty: {manifest_path}")

    split_override = load_speaker_split_file(args.speaker_split_file)
    if split_override is not None:
        missing = sorted(set(speaker_positive) - set(split_override))
        extra = sorted(set(split_override) - set(speaker_positive))
        if missing or extra:
            raise RuntimeError(
                f"Speaker split file mismatch: missing={missing}, unknown={extra}"
            )
        speaker_splits = split_override
    else:
        speaker_splits = embedded_splits

    samples: list[Sample] = []
    for record in records:
        normalized = str(record["normalized_transcript"])
        full_ids = tuple(int(value) for value in tokenizer.encode(normalized, out_type=int))
        reduced = transcript_target_ids(normalized, tokenizer, old_to_reduced)
        positive = bool(record["recomputed_positive"])
        if positive and not contains_subsequence(reduced, WAKE_SEQUENCE):
            pieces = tuple(tokenizer.id_to_piece(value) for value in full_ids)
            raise RuntimeError(
                f"Positive prepared transcript lost the wake sequence: uid={record['uid']}; "
                f"pieces={pieces}, reduced={reduced}"
            )
        speaker_id = str(record["speaker_id"])
        samples.append(
            Sample(
                uid=str(record["uid"]),
                source_dataset=str(record.get("source_dataset", "WakewordV12")),
                source_kind=str(record["source_kind"]),
                audio_path=str(record["audio_path"]),
                audio_row=(
                    int(record["audio_row"])
                    if record.get("audio_row") is not None
                    else None
                ),
                speaker_id=speaker_id,
                transcript=str(record["transcript"]),
                utterance_index=str(record["utterance_index"]),
                duration_seconds=float(record["duration_seconds"]),
                reduced_token_ids=reduced,
                is_positive=positive,
                split=speaker_splits[speaker_id],
                original_path=(
                    str(record["original_path"]) if record.get("original_path") is not None else None
                ),
                google_drive_object_id=(
                    str(record["google_drive_object_id"])
                    if record.get("google_drive_object_id") is not None
                    else None
                ),
                augmentation=str(record["augmentation"]),
                audio_md5=(
                    str(record["prepared_audio_md5"])
                    if record.get("prepared_audio_md5") is not None
                    else (
                        str(record["md5"])
                        if record.get("md5") is not None
                        else None
                    )
                ),
                sample_rate=(
                    int(record["sample_rate"]) if record.get("sample_rate") is not None else None
                ),
                provenance_kind=(
                    str(record["provenance_kind"])
                    if record.get("provenance_kind") is not None
                    else None
                ),
            )
        )
    return samples, speaker_splits


@dataclass(frozen=True)
class _InfoRow:
    source_index: int
    audio_path: Path
    audio_row: int
    duration_seconds: float


def _read_parquet_transcripts(rows: Sequence[_InfoRow]) -> Iterator[tuple[_InfoRow, str]]:
    rows_by_path: dict[Path, list[_InfoRow]] = defaultdict(list)
    for row in rows:
        rows_by_path[row.audio_path].append(row)
    for parquet_path in sorted(rows_by_path):
        selected = sorted(rows_by_path[parquet_path], key=lambda item: item.audio_row)
        selected_rows = [item.audio_row for item in selected]
        if len(selected_rows) != len(set(selected_rows)):
            raise RuntimeError(f"Duplicate InfoRE2 Parquet rows in {parquet_path}")
        parquet = pq.ParquetFile(parquet_path)
        transcript_column = transcript_column_for_schema(parquet.schema_arrow.names)
        if transcript_column is None:
            raise RuntimeError(
                f"No transcript column in {parquet_path}; columns={parquet.schema_arrow.names}"
            )
        pointer = 0
        row_group_start = 0
        for group_index in range(parquet.metadata.num_row_groups):
            group_rows = parquet.metadata.row_group(group_index).num_rows
            row_group_end = row_group_start + group_rows
            selection_start = pointer
            while pointer < len(selected_rows) and selected_rows[pointer] < row_group_end:
                if selected_rows[pointer] < row_group_start:
                    raise RuntimeError(f"InfoRE2 row order moved backward in {parquet_path}")
                pointer += 1
            if pointer > selection_start:
                table = parquet.read_row_group(group_index, columns=[transcript_column])
                column = table.column(transcript_column)
                for position in range(selection_start, pointer):
                    local_offset = selected_rows[position] - row_group_start
                    transcript = column[local_offset].as_py()
                    if transcript is None or not str(transcript).strip():
                        raise RuntimeError(
                            f"Empty InfoRE2 transcript: {parquet_path} row={selected_rows[position]}"
                        )
                    yield selected[position], str(transcript)
                del column, table
            row_group_start = row_group_end
            if pointer == len(selected):
                break
        if pointer != len(selected):
            raise IndexError(
                f"{len(selected) - pointer} requested rows exceed {parquet_path}"
            )


def load_infore2_samples(
    data_root: Path,
    tokenizer,
    old_to_reduced: dict[int, int],
    args: argparse.Namespace,
) -> list[Sample]:
    manifest_path = data_root / "manifests" / "aligned_manifest.jsonl"
    audio_root = (data_root / "audio_shards").resolve()
    rows: list[_InfoRow] = []
    seen_ids: set[int] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            dataset = str(record.get("dataset", "")).strip().lower()
            if dataset != "infore2":
                continue
            source_index = int(record["sample_id"])
            if source_index in seen_ids:
                raise RuntimeError(f"Duplicate InfoRE2 sample_id {source_index}")
            seen_ids.add(source_index)
            relative = Path(str(record["audio_relpath"]))
            audio_path = (audio_root / relative).resolve()
            if audio_path != audio_root and audio_root not in audio_path.parents:
                raise RuntimeError(
                    f"InfoRE2 manifest path escapes audio root at line {line_number}: {relative}"
                )
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            teacher_frames = int(record["teacher_length"])
            if teacher_frames <= 0:
                raise RuntimeError(f"Invalid teacher length at manifest line {line_number}")
            rows.append(
                _InfoRow(
                    source_index=source_index,
                    audio_path=audio_path,
                    audio_row=int(record["audio_row"]),
                    duration_seconds=teacher_frames * 0.08,
                )
            )
    if not rows:
        raise RuntimeError(f"No InfoRE2 rows in {manifest_path}")

    validation_ids = {
        row.source_index
        for row in sorted(
            rows,
            key=lambda item: stable_digest("infore2-validation", args.seed, item.source_index),
        )[: min(args.infore2_validation_samples, len(rows))]
    }
    samples = []
    for row, transcript in _read_parquet_transcripts(rows):
        normalized = normalize_text(transcript)
        full_ids = tuple(int(value) for value in tokenizer.encode(normalized, out_type=int))
        reduced = transcript_target_ids(normalized, tokenizer, old_to_reduced)
        positive = contains_word_sequence(normalized)
        if positive and not contains_subsequence(reduced, WAKE_SEQUENCE):
            raise RuntimeError(
                f"InfoRE2 positive lost wake sequence: sample_id={row.source_index}, "
                f"transcript={transcript!r}, reduced={reduced}"
            )
        if row.source_index in validation_ids:
            split = "validation"
        elif stable_fraction("infore2-train", args.seed, row.source_index) < args.infore2_train_fraction:
            split = "train"
        else:
            split = "unused"
        samples.append(
            Sample(
                uid=f"InfoRE2:{row.source_index}",
                source_dataset="InfoRE2",
                source_kind="parquet",
                audio_path=str(row.audio_path),
                audio_row=row.audio_row,
                speaker_id=None,
                transcript=transcript,
                utterance_index=str(row.source_index),
                duration_seconds=row.duration_seconds,
                reduced_token_ids=reduced,
                is_positive=positive,
                split=split,
            )
        )
    return samples


def manifest_record(sample: Sample) -> dict[str, object]:
    return {
        **asdict(sample),
        "reduced_token_ids": list(sample.reduced_token_ids),
        "reduced_token_pieces": [REDUCED_PIECES[value] for value in sample.reduced_token_ids],
        "is_hard_negative": sample.is_hard_negative,
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_unified_manifest(
    samples: Sequence[Sample], path: Path, protect_existing: bool = False
) -> str:
    digest = hashlib.sha1()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            for sample in sorted(samples, key=lambda item: item.uid):
                line = json.dumps(
                    manifest_record(sample), ensure_ascii=False, sort_keys=True
                ) + "\n"
                encoded = line.encode("utf-8")
                digest.update(encoded)
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_sha1 = digest.hexdigest()
        if path.is_file() and protect_existing:
            existing_digest = hashlib.sha1()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(4 * 1024**2), b""):
                    existing_digest.update(block)
            if existing_digest.hexdigest() != manifest_sha1:
                raise RuntimeError(
                    f"Dataset changed for an existing run: {path}. "
                    "Use a new --run-name; the old provenance manifest was preserved."
                )
        os.replace(temporary, path)
        return manifest_sha1
    finally:
        temporary.unlink(missing_ok=True)


def audit_samples(
    samples: Sequence[Sample],
    speaker_splits: dict[str, str],
    *,
    require_test: bool = True,
) -> None:
    if not samples:
        raise RuntimeError("Unified dataset is empty")
    print("=" * 88)
    print("V12 UNIFIED DATASET AUDIT")
    print("=" * 88)
    counters: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for sample in samples:
        key = (sample.source_dataset, sample.split)
        counters[key]["samples"] += 1
        counters[key]["positive"] += int(sample.is_positive)
        counters[key]["hard_negative"] += int(sample.is_hard_negative)
        counters[key]["empty_target"] += int(not sample.reduced_token_ids)
        counters[key]["hours"] += sample.duration_seconds / 3600.0
    for key in sorted(counters):
        values = counters[key]
        print(
            f"{key[0]:12s} {key[1]:10s} "
            f"samples={int(values['samples']):8,d} "
            f"positive={int(values['positive']):7,d} "
            f"hard-neg={int(values['hard_negative']):7,d} "
            f"empty={int(values['empty_target']):8,d} "
            f"hours={values['hours']:10.2f}"
        )
    if speaker_splits:
        split_counts = defaultdict(int)
        for split in speaker_splits.values():
            split_counts[split] += 1
        print("Speaker-disjoint split:", dict(sorted(split_counts.items())))

    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    test = [sample for sample in samples if sample.split == "test"]
    if not any(sample.is_positive for sample in train):
        raise RuntimeError("Training split contains no full wake-phrase positive")
    if not any(not sample.is_positive for sample in train):
        raise RuntimeError("Training split contains no negative")
    if not validation:
        raise RuntimeError("Validation split is empty")
    if not any(sample.is_positive for sample in validation):
        raise RuntimeError(
            "Validation has no positive. Add positive speakers or provide --speaker-split-file."
        )
    if not any(not sample.is_positive for sample in validation):
        raise RuntimeError("Validation has no negative")
    if require_test:
        if not test:
            raise RuntimeError("Test split is empty")
        if not any(sample.is_positive for sample in test):
            raise RuntimeError("Test split has no positive")
        if not any(not sample.is_positive for sample in test):
            raise RuntimeError("Test split has no negative")


def cycle_sample(
    pool: Sequence[int], count: int, seed: int, label: str
) -> list[int]:
    if count <= 0:
        return []
    if not pool:
        return []
    result: list[int] = []
    cycle = 0
    while len(result) < count:
        values = list(pool)
        random.Random(seed + cycle * 104_729 + int.from_bytes(stable_digest(label)[:4], "big")).shuffle(values)
        result.extend(values[: count - len(result)])
        cycle += 1
    return result


def storage_locality_key(sample: Sample, row_group_size: int) -> tuple[str, str, int]:
    """Return a deterministic storage block used only to order selected samples."""
    if sample.source_kind == "parquet" and sample.audio_row is not None:
        return (
            "parquet",
            sample.audio_path,
            int(sample.audio_row) // int(row_group_size),
        )
    # Independent files intentionally remain independent locality groups.
    return ("file", sample.audio_path, 0)


def order_by_storage_locality(
    selected: Sequence[int],
    samples: Sequence[Sample],
    seed: int,
    label: str,
    row_group_size: int,
    block_size: int,
) -> list[int]:
    """Cluster a bounded number of selections from the same Parquet row group.

    Counts and membership are unchanged.  Bounded round-robin blocks avoid
    placing every replay of one shard next to itself while allowing batched
    ``__getitems__`` reads to serve most of a batch from one or two row groups.
    """
    if not selected:
        return []
    if row_group_size < 1 or block_size < 1:
        raise ValueError("Storage-locality sizes must be positive")
    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index in selected:
        grouped[storage_locality_key(samples[int(index)], row_group_size)].append(
            int(index)
        )
    label_digest = int.from_bytes(stable_digest(label)[:8], "big")
    keys = list(grouped)
    random.Random(seed + label_digest).shuffle(keys)
    for group_index, key in enumerate(keys):
        random.Random(seed + label_digest + (group_index + 1) * 104_729).shuffle(
            grouped[key]
        )
    cursors = {key: 0 for key in keys}
    output: list[int] = []
    round_index = 0
    active = keys
    while active:
        round_keys = list(active)
        random.Random(seed + label_digest + round_index * 1_000_003).shuffle(round_keys)
        next_active: list[tuple[str, str, int]] = []
        for key in round_keys:
            start = cursors[key]
            end = min(start + block_size, len(grouped[key]))
            output.extend(grouped[key][start:end])
            cursors[key] = end
            if end < len(grouped[key]):
                next_active.append(key)
        active = next_active
        round_index += 1
    if len(output) != len(selected):
        raise RuntimeError("Storage-locality ordering changed sample count")
    expected_counts: dict[int, int] = defaultdict(int)
    output_counts: dict[int, int] = defaultdict(int)
    for index in selected:
        expected_counts[int(index)] += 1
    for index in output:
        output_counts[int(index)] += 1
    if output_counts != expected_counts:
        raise RuntimeError("Storage-locality ordering changed sample membership")
    return output


def stratified_negative_sequence(
    samples: Sequence[Sample],
    negative_slots: int,
    new_data_share: float,
    hard_negative_share: float,
    seed: int,
) -> list[int]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.split != "train" or sample.is_positive:
            continue
        source = "infore2" if sample.source_dataset == "InfoRE2" else "new"
        difficulty = "hard" if sample.is_hard_negative else "easy"
        groups[(source, difficulty)].append(index)

    available_sources = [
        source for source in ("new", "infore2")
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
        if source_count <= 0:
            continue
        hard_pool = groups[(source, "hard")]
        easy_pool = groups[(source, "easy")]
        if hard_pool and easy_pool:
            hard_count = int(round(source_count * hard_negative_share))
        elif hard_pool:
            hard_count = source_count
        else:
            hard_count = 0
        easy_count = source_count - hard_count
        selected.extend(cycle_sample(hard_pool, hard_count, seed, f"{source}-hard"))
        selected.extend(cycle_sample(easy_pool, easy_count, seed, f"{source}-easy"))
    random.Random(seed + 9187).shuffle(selected)
    return selected


def _split_to_padding_budget(
    batch: list[int],
    samples: Sequence[Sample],
    max_padded_audio_seconds: float,
    positive_cycle: Iterator[int],
) -> list[list[int]]:
    if not batch:
        return []
    padded = max(samples[index].duration_seconds for index in batch) * len(batch)
    if padded <= max_padded_audio_seconds or len(batch) == 1:
        if not any(samples[index].is_positive for index in batch):
            batch.insert(0, next(positive_cycle))
        return [batch]
    positives = sorted(
        (index for index in batch if samples[index].is_positive),
        key=lambda index: samples[index].duration_seconds,
        reverse=True,
    )
    negatives = sorted(
        (index for index in batch if not samples[index].is_positive),
        key=lambda index: samples[index].duration_seconds,
        reverse=True,
    )
    # Split each class independently so the requested batch ratio survives
    # duration-based subdivision as closely as integer counts allow.
    left = positives[::2] + negatives[::2]
    right = positives[1::2] + negatives[1::2]
    for group in (left, right):
        if group and not any(samples[index].is_positive for index in group):
            group.insert(0, next(positive_cycle))
    return (
        _split_to_padding_budget(left, samples, max_padded_audio_seconds, positive_cycle)
        + _split_to_padding_budget(right, samples, max_padded_audio_seconds, positive_cycle)
    )


def build_epoch_batch_plan(
    samples: Sequence[Sample],
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
    batch_size = max_utterances_per_batch
    epoch_seed = seed + int(epoch) * 1_000_003 + int(negative_positive_ratio) * 65_537
    # Use every negative once and choose the epoch-level positive count before
    # forming batches.  A fixed per-batch rounded count makes B=32 turn 8:1
    # into 7:1 and 4:1 into 4.333:1.  Proportional integer allocation below
    # alternates neighboring batch compositions while preserving the requested
    # global ratio and at least one item of each class in every base batch.
    negative_slots = len(negative_pool)
    positive_slots = max(1, math.ceil(negative_slots / negative_positive_ratio))
    total_slots = negative_slots + positive_slots
    batch_count = math.ceil(total_slots / batch_size)
    if positive_slots < batch_count or negative_slots < batch_count:
        raise RuntimeError(
            "Batch size is too small to keep positive and negative examples in every batch"
        )

    base_size, larger_batch_count = divmod(total_slots, batch_count)
    batch_sizes = [base_size + int(index < larger_batch_count) for index in range(batch_count)]
    random.Random(epoch_seed + 11).shuffle(batch_sizes)
    capacities = [size - 2 for size in batch_sizes]
    extra_positive_total = positive_slots - batch_count
    extra_capacity_total = sum(capacities)
    if extra_capacity_total == 0:
        extra_positive_counts = [0] * batch_count
    else:
        ideals = [capacity * extra_positive_total / extra_capacity_total for capacity in capacities]
        extra_positive_counts = [int(math.floor(value)) for value in ideals]
        remainder = extra_positive_total - sum(extra_positive_counts)
        order = sorted(
            range(batch_count),
            key=lambda index: (
                -(ideals[index] - extra_positive_counts[index]),
                stable_digest("batch-ratio-remainder", epoch_seed, index),
            ),
        )
        for index in order[:remainder]:
            extra_positive_counts[index] += 1
    positive_counts = [1 + value for value in extra_positive_counts]
    negative_counts = [
        size - positive_count for size, positive_count in zip(batch_sizes, positive_counts)
    ]
    if (
        sum(positive_counts) != positive_slots
        or sum(negative_counts) != negative_slots
        or any(value < 1 for value in positive_counts)
        or any(value < 1 for value in negative_counts)
        or any(p + n > batch_size for p, n in zip(positive_counts, negative_counts))
    ):
        raise RuntimeError("Proportional batch allocation invariant failed")

    negatives = stratified_negative_sequence(
        samples,
        negative_slots,
        new_data_negative_share,
        hard_negative_share,
        epoch_seed,
    )
    positives = cycle_sample(
        positive_pool,
        positive_slots,
        epoch_seed + 17,
        "positive",
    )
    # Padding splits may need extra repeated positives. This is deliberate:
    # the no-positive-batch invariant is stronger than the requested ratio.
    extra_positives = cycle_sample(
        positive_pool,
        len(positive_pool),
        epoch_seed + 29,
        "padding-positive",
    )
    if shard_aware_batching:
        positives = order_by_storage_locality(
            positives,
            samples,
            epoch_seed + 31,
            "positive",
            shard_row_group_size,
            shard_locality_block_size,
        )
        negatives = order_by_storage_locality(
            negatives,
            samples,
            epoch_seed + 37,
            "negative",
            shard_row_group_size,
            shard_locality_block_size,
        )
        extra_positives = order_by_storage_locality(
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
    for batch_index in range(batch_count):
        positive_count = positive_counts[batch_index]
        negative_count = negative_counts[batch_index]
        batch = positives[positive_cursor : positive_cursor + positive_count]
        batch += negatives[negative_cursor : negative_cursor + negative_count]
        positive_cursor += positive_count
        negative_cursor += negative_count
        random.Random(epoch_seed + batch_index * 7919).shuffle(batch)
        plan.extend(
            _split_to_padding_budget(
                batch,
                samples,
                max_padded_audio_seconds,
                positive_cycle,
            )
        )
    if not plan or any(not any(samples[index].is_positive for index in batch) for batch in plan):
        raise RuntimeError("Batch planner violated the at-least-one-positive invariant")
    if positive_cursor != len(positives) or negative_cursor != len(negatives):
        raise RuntimeError("Batch planner did not consume every allocated slot")
    return plan


def curriculum_observe(
    state: CurriculumState,
    metric: float,
    stage_count: int,
    minimum_epochs: int,
    patience: int,
    relative_improvement: float,
    absolute_improvement: float,
) -> str:
    if state.complete:
        return "already_complete"
    state.epochs_in_stage += 1
    if state.best_metric is None:
        improved = True
    else:
        required = max(
            absolute_improvement,
            relative_improvement * abs(float(state.best_metric)),
        )
        improved = metric < float(state.best_metric) - required
    if improved:
        state.best_metric = float(metric)
        state.bad_validations = 0
        return "improved"
    state.bad_validations += 1
    if state.epochs_in_stage < minimum_epochs or state.bad_validations < patience:
        return "held"
    if state.stage_index + 1 < stage_count:
        state.stage_index += 1
        state.epochs_in_stage = 0
        state.best_metric = None
        state.bad_validations = 0
        return "advanced"
    state.complete = True
    return "complete"


def curriculum_observe_steps(
    state: CurriculumState,
    metric: float,
    optimizer_step: int,
    stage_count: int,
    minimum_optimizer_steps: int,
    minimum_validations: int,
    patience: int,
    relative_improvement: float,
    absolute_improvement: float,
    detection_metric: float | None = None,
    detection_patience: int = 0,
    detection_absolute_improvement: float = 0.0,
) -> str:
    """Observe a step-scheduled CTC metric with a long recovery window.

    The monitored quantity is an EMA of validation macro CTC loss (an alpha of
    one preserves the legacy raw-loss controller).  A stage can advance only
    after both its minimum step/observation guards and a complete run of
    consecutive non-improving CTC validations.  When enabled, the detection
    veto additionally requires F1 to have stopped improving for its own
    patience window.  This gives a fluctuating loss time to recover without an
    epoch cap and prevents a handoff while detection quality is still rising.
    """
    if state.complete:
        return "already_complete"
    if optimizer_step < state.stage_start_optimizer_step:
        raise ValueError(
            "optimizer_step precedes the recorded curriculum stage start"
        )
    state.validation_observations_in_stage += 1
    if detection_metric is not None:
        if (
            state.best_detection_metric is None
            or float(detection_metric)
            > float(state.best_detection_metric) + float(detection_absolute_improvement)
        ):
            state.best_detection_metric = float(detection_metric)
            state.bad_detection_validations = 0
        else:
            state.bad_detection_validations += 1
    alpha = float(CURRICULUM_EMA_ALPHA)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("CURRICULUM_EMA_ALPHA must be in (0, 1]")
    if state.metric_ema is None:
        state.metric_ema = float(metric)
    else:
        state.metric_ema = alpha * float(metric) + (1.0 - alpha) * float(
            state.metric_ema
        )
    observed_metric = float(state.metric_ema)
    if state.best_metric is None:
        improved = True
    else:
        required = max(
            absolute_improvement,
            relative_improvement * abs(float(state.best_metric)),
        )
        improved = observed_metric < float(state.best_metric) - required
    if improved:
        state.best_metric = observed_metric
        state.bad_validations = 0
        return "improved"
    state.bad_validations += 1
    steps_in_stage = optimizer_step - state.stage_start_optimizer_step
    if (
        steps_in_stage < minimum_optimizer_steps
        or state.validation_observations_in_stage < minimum_validations
        or state.bad_validations < patience
        or (
            detection_patience > 0
            and state.bad_detection_validations < detection_patience
        )
    ):
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


def mono_float32(value):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 2:
        if tensor.shape[0] <= 8:
            tensor = tensor.mean(dim=0)
        elif tensor.shape[1] <= 8:
            tensor = tensor.mean(dim=1)
        else:
            raise RuntimeError(f"Cannot identify channel axis: {tuple(tensor.shape)}")
    elif tensor.ndim != 1:
        tensor = tensor.reshape(-1)
    tensor = tensor.contiguous()
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise RuntimeError("Decoded audio is empty or non-finite")
    return tensor


def decode_audio_object(audio_object, target_sample_rate: int = TARGET_SAMPLE_RATE):
    sample_rate = target_sample_rate
    if isinstance(audio_object, dict) and "audio" in audio_object and len(audio_object) == 1:
        return decode_audio_object(audio_object["audio"], target_sample_rate)
    if hasattr(audio_object, "get_all_samples"):
        decoded = audio_object.get_all_samples()
        waveform = mono_float32(decoded.data)
        sample_rate = int(decoded.sample_rate)
    elif isinstance(audio_object, (bytes, bytearray, memoryview)):
        # Python bytes expose a text ``decode()`` method.  They must be handled
        # before the generic audio-extension ``decode()`` protocol below, or a
        # binary WAV/FLAC payload is incorrectly interpreted as UTF-8 text.
        data, sample_rate = sf.read(
            io.BytesIO(bytes(audio_object)), dtype="float32", always_2d=False
        )
        waveform = mono_float32(data)
    elif hasattr(audio_object, "decode"):
        decoded = audio_object.decode()
        waveform = mono_float32(decoded.data if hasattr(decoded, "data") else decoded)
        sample_rate = int(getattr(decoded, "sample_rate", target_sample_rate))
    elif isinstance(audio_object, dict) and "array" in audio_object:
        waveform = mono_float32(audio_object["array"])
        sample_rate = int(
            audio_object.get("sampling_rate", audio_object.get("sample_rate", target_sample_rate))
        )
    elif isinstance(audio_object, dict) and audio_object.get("bytes") is not None:
        data, sample_rate = sf.read(
            io.BytesIO(audio_object["bytes"]), dtype="float32", always_2d=False
        )
        waveform = mono_float32(data)
    elif isinstance(audio_object, dict) and audio_object.get("path"):
        data, sample_rate = sf.read(
            audio_object["path"], dtype="float32", always_2d=False
        )
        waveform = mono_float32(data)
    elif isinstance(audio_object, (str, os.PathLike)):
        try:
            data, sample_rate = sf.read(
                str(audio_object), dtype="float32", always_2d=False
            )
        except Exception:
            data, sample_rate = librosa.load(str(audio_object), sr=None, mono=True)
        waveform = mono_float32(data)
    else:
        waveform = mono_float32(audio_object)
    if int(sample_rate) != int(target_sample_rate):
        resampled = librosa.resample(
            y=waveform.numpy(),
            orig_sr=int(sample_rate),
            target_sr=int(target_sample_rate),
            res_type="soxr_hq",
        )
        waveform = torch.from_numpy(np.ascontiguousarray(resampled, dtype=np.float32))
    return waveform.contiguous()


class UnifiedAudioDataset:
    """Map dataset with a batched fast path for grouped Parquet row-group reads."""

    def __init__(self, samples: Sequence[Sample]):
        self.samples = tuple(samples)
        self._parquet_cache_pid: int | None = None
        self._parquet_cache: OrderedDict[str, object] = OrderedDict()
        self._row_group_boundaries: dict[str, tuple[int, ...]] = {}
        self._decoded_wakeword_row_groups: OrderedDict[
            tuple[str, int], tuple[object, ...]
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _ensure_process_cache(self) -> None:
        process_id = os.getpid()
        if self._parquet_cache_pid == process_id:
            return
        self._parquet_cache.clear()
        self._row_group_boundaries.clear()
        self._decoded_wakeword_row_groups.clear()
        self._parquet_cache_pid = process_id

    def _parquet(self, path: str):
        self._ensure_process_cache()
        reader = self._parquet_cache.pop(path, None)
        if reader is None:
            reader = pq.ParquetFile(path)
        self._parquet_cache[path] = reader
        while len(self._parquet_cache) > 4:
            self._parquet_cache.popitem(last=False)
        return reader

    def _boundaries(self, path: str, reader) -> tuple[int, ...]:
        boundaries = self._row_group_boundaries.get(path)
        if boundaries is None:
            cumulative = 0
            values = []
            for group_index in range(reader.metadata.num_row_groups):
                cumulative += reader.metadata.row_group(group_index).num_rows
                values.append(cumulative)
            boundaries = tuple(values)
            self._row_group_boundaries[path] = boundaries
        return boundaries

    def _read_parquet_samples(self, indexed_samples: Sequence[tuple[int, Sample]]):
        by_path: dict[str, list[tuple[int, Sample]]] = defaultdict(list)
        for output_position, sample in indexed_samples:
            by_path[sample.audio_path].append((output_position, sample))
        result: dict[int, dict[str, object]] = {}
        for path, selected in by_path.items():
            reader = self._parquet(path)
            if "audio" not in reader.schema_arrow.names:
                raise RuntimeError(f"Missing audio column in {path}")
            boundaries = self._boundaries(path, reader)
            by_group: dict[int, list[tuple[int, Sample]]] = defaultdict(list)
            for output_position, sample in selected:
                assert sample.audio_row is not None
                group_index = next(
                    (index for index, end in enumerate(boundaries) if sample.audio_row < end),
                    None,
                )
                if group_index is None:
                    raise IndexError(f"Audio row {sample.audio_row} exceeds {path}")
                by_group[group_index].append((output_position, sample))
            for group_index, group_samples in by_group.items():
                start = 0 if group_index == 0 else boundaries[group_index - 1]
                cache_wakeword_group = all(
                    sample.source_dataset != "InfoRE2"
                    for _output_position, sample in group_samples
                )
                cache_key = (path, group_index)
                decoded_group = None
                if cache_wakeword_group:
                    decoded_group = self._decoded_wakeword_row_groups.pop(
                        cache_key, None
                    )
                if decoded_group is None:
                    table = reader.read_row_group(group_index, columns=["audio"])
                    column = table.column("audio")
                    if cache_wakeword_group:
                        decoded_group = tuple(
                            decode_audio_object(value) for value in column.to_pylist()
                        )
                        self._decoded_wakeword_row_groups[cache_key] = decoded_group
                        while len(self._decoded_wakeword_row_groups) > 2:
                            self._decoded_wakeword_row_groups.popitem(last=False)
                    else:
                        for output_position, sample in group_samples:
                            waveform = decode_audio_object(
                                column[int(sample.audio_row) - start].as_py()
                            )
                            result[output_position] = self._item(sample, waveform)
                    del column, table
                else:
                    self._decoded_wakeword_row_groups[cache_key] = decoded_group
                if decoded_group is not None:
                    for output_position, sample in group_samples:
                        waveform = decoded_group[int(sample.audio_row) - start]
                        result[output_position] = self._item(sample, waveform)
        return result

    @staticmethod
    def _item(sample: Sample, waveform) -> dict[str, object]:
        return {
            "sample": sample,
            "audio": waveform,
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        if sample.source_kind == "file":
            return self._item(sample, decode_audio_object(sample.audio_path))
        return self._read_parquet_samples([(0, sample)])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, object]]:
        output: list[dict[str, object] | None] = [None] * len(indices)
        parquet_samples: list[tuple[int, Sample]] = []
        for output_position, index in enumerate(indices):
            sample = self.samples[int(index)]
            if sample.source_kind == "file":
                output[output_position] = self._item(
                    sample, decode_audio_object(sample.audio_path)
                )
            else:
                parquet_samples.append((output_position, sample))
        if parquet_samples:
            for output_position, value in self._read_parquet_samples(parquet_samples).items():
                output[output_position] = value
        if any(value is None for value in output):
            raise RuntimeError("Batched audio loader left an unresolved item")
        return list(output)


def collate_audio_items(items: Sequence[dict[str, object]]) -> dict[str, object]:
    if not items:
        raise RuntimeError("Cannot collate an empty batch")
    samples = [item["sample"] for item in items]
    waveforms = [item["audio"] for item in items]
    max_audio = max(int(waveform.numel()) for waveform in waveforms)
    audio = torch.zeros(len(items), max_audio, dtype=torch.float32)
    audio_lengths = torch.empty(len(items), dtype=torch.long)
    target_lengths = torch.empty(len(items), dtype=torch.long)
    required_frames = torch.empty(len(items), dtype=torch.long)
    concatenated_targets: list[int] = []
    for index, (sample, waveform) in enumerate(zip(samples, waveforms)):
        audio_length = int(waveform.numel())
        token_ids = tuple(int(value) for value in sample.reduced_token_ids)
        repeated_neighbors = sum(
            left == right for left, right in zip(token_ids, token_ids[1:])
        )
        audio[index, :audio_length] = waveform
        audio_lengths[index] = audio_length
        target_lengths[index] = len(token_ids)
        required_frames[index] = len(token_ids) + repeated_neighbors
        concatenated_targets.extend(token_ids)
    return {
        "audio": audio,
        "audio_lengths": audio_lengths,
        "ctc_targets": torch.tensor(concatenated_targets, dtype=torch.long),
        "ctc_target_lengths": target_lengths,
        "ctc_required_frames": required_frames,
        "samples": samples,
        "positive_count": sum(sample.is_positive for sample in samples),
    }


StudentASR = None  # Rebound to the runtime nn.Module class by build_student_class().


def build_student_class():
    class _StudentASR(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocabulary = list(REDUCED_VOCABULARY)
            self.preprocessor = AudioToMelSpectrogramPreprocessor(
                sample_rate=TARGET_SAMPLE_RATE,
                window_size=0.025,
                window_stride=FEATURE_HOP_SECONDS,
                window="hann",
                normalize="NA",
                features=STUDENT_N_MELS,
                n_fft=512,
                log=True,
                dither=0.0,
                pad_to=0,
                frame_splicing=1,
                pad_value=0.0,
                nb_augmentation_prob=0.0,
            )
            self.encoder = ConformerEncoder(
                feat_in=STUDENT_N_MELS,
                n_layers=STUDENT_N_LAYERS,
                d_model=STUDENT_D_MODEL,
                feat_out=-1,
                subsampling="dw_striding",
                subsampling_factor=SUBSAMPLING_FACTOR,
                subsampling_conv_channels=256,
                causal_downsampling=True,
                ff_expansion_factor=STUDENT_FF_EXPANSION,
                self_attention_model="rel_pos",
                n_heads=STUDENT_N_HEADS,
                att_context_size=[-1, STREAMING_CHUNK_ENCODER_FRAMES - 1],
                att_context_style="chunked_limited",
                xscaling=True,
                untie_biases=True,
                pos_emb_max_len=5000,
                conv_kernel_size=STUDENT_CONV_KERNEL,
                conv_norm_type="layer_norm",
                conv_context_size="causal",
                dropout=0.1,
                dropout_pre_encoder=0.1,
                dropout_emb=0.0,
                dropout_att=0.1,
                stochastic_depth_drop_prob=0.0,
                stochastic_depth_mode="linear",
                stochastic_depth_start_layer=1,
                use_pytorch_sdpa=True,
            )
            self.decoder = ConvASRDecoder(
                feat_in=STUDENT_D_MODEL,
                num_classes=len(self.vocabulary),
                vocabulary=self.vocabulary,
                add_blank=True,
            )

        def forward(self, input_signal, input_signal_length):
            processed, processed_lengths = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
            encoded, encoded_lengths = self.encoder(
                audio_signal=processed, length=processed_lengths
            )
            return self.decoder(encoder_output=encoded), encoded_lengths

    return _StudentASR


def strip_state_prefixes(state: dict[str, object]) -> dict[str, object]:
    result = dict(state)
    for prefix in ("module.", "model.", "student."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix) :]: value for key, value in result.items()}
    return result


def checkpoint_student_state(payload: object) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a state_dict or a dictionary payload")
    metadata = payload
    for key in ("student_state_dict", "state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return strip_state_prefixes(candidate), metadata
    if payload and all(isinstance(key, str) for key in payload):
        return strip_state_prefixes(payload), {}
    raise RuntimeError("Cannot locate student weights in checkpoint")


def initialize_from_checkpoint(
    student,
    checkpoint_path: Path,
    old_piece_ids: Sequence[int],
) -> dict[str, object]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state, metadata = checkpoint_student_state(payload)
    target_state = student.state_dict()

    # A V12 checkpoint already has the same head and can be loaded directly.
    if set(source_state) == set(target_state) and all(
        tuple(source_state[key].shape) == tuple(target_state[key].shape)
        for key in target_state
    ):
        student.load_state_dict(source_state, strict=True)
        return {
            "path": str(checkpoint_path),
            "mode": "exact_v12",
            "source_optimizer_step": metadata.get("optimizer_step"),
            "remapped_keys": [],
        }

    source_output_rows = None
    signature = metadata.get("signature", {}) if isinstance(metadata, dict) else {}
    if isinstance(signature, dict) and signature.get("output_dim") is not None:
        source_output_rows = int(signature["output_dim"])
    if source_output_rows is None:
        candidate_rows = sorted(
            {
                int(value.shape[0])
                for key, value in source_state.items()
                if key in target_state
                and hasattr(value, "shape")
                and len(value.shape) >= 1
                and value.shape[0] > REDUCED_OUTPUT_DIM
                and tuple(value.shape[1:]) == tuple(target_state[key].shape[1:])
                and target_state[key].shape[0] == REDUCED_OUTPUT_DIM
            }
        )
        if len(candidate_rows) == 1:
            source_output_rows = candidate_rows[0]
    if source_output_rows is None or source_output_rows <= max(old_piece_ids):
        raise RuntimeError(
            "Cannot infer the old decoder output dimension for vocabulary remapping"
        )
    source_blank_id = (
        int(signature.get("ctc_blank_id"))
        if isinstance(signature, dict) and signature.get("ctc_blank_id") is not None
        else source_output_rows - 1
    )
    source_rows = tuple(int(value) for value in old_piece_ids) + (source_blank_id,)
    remapped_keys: list[str] = []
    transferred: dict[str, object] = {}
    missing: list[str] = []
    incompatible: list[str] = []
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            missing.append(key)
            continue
        if tuple(source_value.shape) == tuple(target_value.shape):
            transferred[key] = source_value
            continue
        can_remap = (
            len(source_value.shape) >= 1
            and int(source_value.shape[0]) == source_output_rows
            and int(target_value.shape[0]) == REDUCED_OUTPUT_DIM
            and tuple(source_value.shape[1:]) == tuple(target_value.shape[1:])
        )
        if can_remap:
            transferred[key] = source_value[list(source_rows)].clone()
            remapped_keys.append(key)
        else:
            incompatible.append(
                f"{key}: source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
            )
    if missing or incompatible or not remapped_keys:
        raise RuntimeError(
            "Checkpoint is not architecture-compatible with V11 L5/H192/M64: "
            f"missing={missing[:10]}, incompatible={incompatible[:10]}, "
            f"remapped_keys={remapped_keys}"
        )
    student.load_state_dict(transferred, strict=True)
    return {
        "path": str(checkpoint_path),
        "mode": "v11_decoder_row_remap",
        "source_optimizer_step": metadata.get("optimizer_step"),
        "source_output_dim": source_output_rows,
        "source_blank_id": source_blank_id,
        "source_rows": list(source_rows),
        "target_rows": list(range(REDUCED_OUTPUT_DIM)),
        "remapped_keys": remapped_keys,
    }


def split_concatenated_targets(targets, target_lengths) -> list[object]:
    result = []
    offset = 0
    for length in target_lengths.detach().cpu().tolist():
        length = int(length)
        result.append(targets[offset : offset + length])
        offset += length
    if offset != int(targets.numel()):
        raise RuntimeError("Concatenated CTC target length mismatch")
    return result


def pure_ctc_per_sample(
    student_output,
    student_lengths,
    ctc_targets,
    ctc_target_lengths,
    ctc_required_frames,
):
    """Exact pure CTC, including mathematically explicit empty-target loss."""
    input_lengths_cpu = student_lengths.detach().cpu().long()
    target_lengths_cpu = ctc_target_lengths.detach().cpu().long()
    required_cpu = ctc_required_frames.detach().cpu().long()
    infeasible = input_lengths_cpu.lt(required_cpu)
    if torch.any(infeasible):
        raise RuntimeError(
            "Audio is too short for its reduced CTC target: "
            f"input={input_lengths_cpu.tolist()} required={required_cpu.tolist()}"
        )
    with torch.autocast(device_type=student_output.device.type, enabled=False):
        log_probs_btc = F.log_softmax(student_output.float(), dim=-1)
        batch_size, max_time, _classes = log_probs_btc.shape
        loss_values: list[object | None] = [None] * batch_size
        target_parts = split_concatenated_targets(ctc_targets, ctc_target_lengths)
        nonempty_indices = [
            index for index, length in enumerate(target_lengths_cpu.tolist()) if int(length) > 0
        ]
        empty_indices = [
            index for index, length in enumerate(target_lengths_cpu.tolist()) if int(length) == 0
        ]
        if nonempty_indices:
            index_tensor = torch.tensor(
                nonempty_indices, device=student_output.device, dtype=torch.long
            )
            selected_log_probs = log_probs_btc.index_select(0, index_tensor).transpose(0, 1).contiguous()
            selected_targets = torch.cat([target_parts[index] for index in nonempty_indices])
            selected_input_lengths = input_lengths_cpu[nonempty_indices]
            selected_target_lengths = target_lengths_cpu[nonempty_indices]
            selected_losses = F.ctc_loss(
                selected_log_probs,
                selected_targets,
                selected_input_lengths,
                selected_target_lengths,
                blank=CTC_BLANK_ID,
                reduction="none",
                zero_infinity=False,
            )
            for selected_position, batch_index in enumerate(nonempty_indices):
                loss_values[batch_index] = selected_losses[selected_position]
        for index in empty_indices:
            valid_time = int(input_lengths_cpu[index])
            loss_values[index] = -log_probs_btc[index, :valid_time, CTC_BLANK_ID].sum()
        if any(value is None for value in loss_values):
            raise RuntimeError("CTC loss did not cover every batch item")
        losses = torch.stack(loss_values)
        if not torch.isfinite(losses).all():
            raise RuntimeError("Pure CTC produced NaN/Inf")
        input_lengths_device = input_lengths_cpu.to(student_output.device)
        target_lengths_device = target_lengths_cpu.to(student_output.device)
        normalizers = torch.where(
            target_lengths_device.gt(0),
            target_lengths_device,
            input_lengths_device,
        ).to(dtype=losses.dtype).clamp_min(1)
        normalized = losses / normalizers
    return normalized, losses


def greedy_token_sequences(student_output, student_lengths) -> list[tuple[int, ...]]:
    predictions = student_output.argmax(dim=-1).detach().cpu()
    sequences = []
    for row, length in zip(predictions, student_lengths.detach().cpu().tolist()):
        collapsed: list[int] = []
        previous = None
        for value in row[: int(length)].tolist():
            token_id = int(value)
            if token_id != previous and token_id != CTC_BLANK_ID:
                collapsed.append(token_id)
            previous = token_id
        sequences.append(tuple(collapsed))
    return sequences


def keyword_automaton_transitions(
    pattern: Sequence[int], class_count: int, blank_id: int
) -> tuple[tuple[int, ...], ...]:
    """Build a CTC-collapse-aware DFA for ``collapsed_path contains pattern``.

    A state is ``(matched_prefix_length, previous_raw_class)``.  Remembering the
    previous raw class is required because CTC only emits a repeated non-blank
    token after an intervening blank.  The fully matched prefix is absorbing.
    The returned table is indexed as ``[current_class][flat_source_state]`` and
    contains the flat destination state.
    """
    target = tuple(int(value) for value in pattern)
    if not target:
        raise ValueError("Keyword pattern must be non-empty")
    if not 0 <= int(blank_id) < int(class_count):
        raise ValueError("blank_id must be a valid output class")
    if any(value == blank_id or not 0 <= value < class_count for value in target):
        raise ValueError("Keyword pattern must contain valid non-blank classes")

    def advance(prefix_length: int, emitted: int) -> int:
        if prefix_length == len(target):
            return prefix_length
        candidate = target[:prefix_length] + (int(emitted),)
        for width in range(min(len(target), len(candidate)), -1, -1):
            if width == 0 or candidate[-width:] == target[:width]:
                return width
        raise AssertionError("Unreachable keyword automaton transition")

    state_count = (len(target) + 1) * class_count
    transitions: list[tuple[int, ...]] = []
    for current_class in range(class_count):
        destinations: list[int] = []
        for flat_state in range(state_count):
            prefix_length, previous_class = divmod(flat_state, class_count)
            if prefix_length == len(target):
                next_prefix = prefix_length
            elif current_class == blank_id or current_class == previous_class:
                next_prefix = prefix_length
            else:
                next_prefix = advance(prefix_length, current_class)
            destinations.append(next_prefix * class_count + current_class)
        transitions.append(tuple(destinations))
    return tuple(transitions)


def ctc_keyword_log_odds(
    student_output,
    student_lengths,
    pattern: Sequence[int] = WAKE_SEQUENCE,
    blank_id: int = CTC_BLANK_ID,
):
    """Return streaming-compatible log odds that a collapsed CTC path matched.

    This forward recursion sums *all* CTC paths.  It can be carried chunk by
    chunk because every update depends only on the previous DFA distribution
    and the current frame.  The trainer evaluates complete utterances in one
    call, while deployment can preserve the same state across streaming chunks.
    """
    class_count = int(student_output.shape[-1])
    transitions = keyword_automaton_transitions(pattern, class_count, blank_id)
    batch_size = int(student_output.shape[0])
    state_count = (len(tuple(pattern)) + 1) * class_count
    probabilities = F.softmax(student_output.float(), dim=-1)
    state = probabilities.new_zeros((batch_size, state_count))
    state[:, int(blank_id)] = 1.0
    transition_tensors = [
        torch.tensor(values, dtype=torch.long, device=student_output.device)
        for values in transitions
    ]
    lengths = student_lengths.to(device=student_output.device, dtype=torch.long)
    maximum_time = int(lengths.max().item()) if batch_size else 0
    for time_index in range(maximum_time):
        updated = torch.zeros_like(state)
        for class_index, destinations in enumerate(transition_tensors):
            contribution = state * probabilities[:, time_index, class_index].unsqueeze(1)
            updated.scatter_add_(
                1,
                destinations.unsqueeze(0).expand(batch_size, -1),
                contribution,
            )
        active = lengths.gt(time_index).unsqueeze(1)
        state = torch.where(active, updated, state)
    matched_start = len(tuple(pattern)) * class_count
    matched_probability = state[:, matched_start:].sum(dim=1)
    unmatched_probability = state[:, :matched_start].sum(dim=1)
    epsilon = torch.finfo(state.dtype).tiny
    return torch.log(matched_probability.clamp_min(epsilon)) - torch.log(
        unmatched_probability.clamp_min(epsilon)
    )


def phrase_score_threshold_audit(
    scores: Sequence[float],
    labels: Sequence[bool],
    max_curve_points: int = 257,
) -> dict[str, object]:
    """Sweep a validation-only phrase-score threshold without touching test."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("Phrase-score audit requires equally sized non-empty inputs")
    ranked = sorted(
        ((float(score), bool(label)) for score, label in zip(scores, labels)),
        key=lambda item: item[0],
        reverse=True,
    )
    positive_count = sum(label for _score, label in ranked)
    negative_count = len(ranked) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Phrase-score audit requires both positive and negative samples")

    threshold_above_max = math.nextafter(ranked[0][0], math.inf)
    candidates: list[dict[str, float | int]] = []

    def add_candidate(threshold: float, tp: int, fp: int) -> None:
        detection = binary_detection_metrics(
            true_positives=tp,
            false_positives=fp,
            true_negatives=negative_count - fp,
            false_negatives=positive_count - tp,
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "recall": float(detection["recall"]),
                "precision": float(detection["precision"]),
                "f1_score": float(detection["f1_score"]),
                "accuracy": float(detection["accuracy"]),
                "true_positives": int(detection["true_positives"]),
                "false_positives": int(detection["false_positives"]),
            }
        )

    add_candidate(threshold_above_max, 0, 0)
    true_positives = 0
    false_positives = 0
    cursor = 0
    while cursor < len(ranked):
        score = ranked[cursor][0]
        while cursor < len(ranked) and ranked[cursor][0] == score:
            if ranked[cursor][1]:
                true_positives += 1
            else:
                false_positives += 1
            cursor += 1
        add_candidate(score, true_positives, false_positives)

    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            float(candidates[index]["f1_score"]),
            float(candidates[index]["precision"]),
            float(candidates[index]["recall"]),
            float(candidates[index]["threshold"]),
        ),
    )
    if len(candidates) <= max_curve_points:
        selected_indices = list(range(len(candidates)))
    else:
        selected_indices = {
            round(index * (len(candidates) - 1) / (max_curve_points - 1))
            for index in range(max_curve_points)
        }
        selected_indices.add(best_index)
        selected_indices = sorted(selected_indices)
    positive_scores = sorted(
        score for score, label in ranked if label
    )
    negative_scores = sorted(
        score for score, label in ranked if not label
    )

    def percentile(values: Sequence[float], fraction: float) -> float:
        position = round(fraction * (len(values) - 1))
        return float(values[position])

    return {
        "score_definition": "log_P_collapsed_path_contains_wake_minus_log_P_not_contains",
        "positive_samples": positive_count,
        "negative_samples": negative_count,
        "best": dict(candidates[best_index]),
        "positive_score_p05": percentile(positive_scores, 0.05),
        "positive_score_median": percentile(positive_scores, 0.50),
        "negative_score_median": percentile(negative_scores, 0.50),
        "negative_score_p95": percentile(negative_scores, 0.95),
        "curve": [candidates[index] for index in selected_indices],
    }


def write_phrase_score_audit_plot(
    run_directory: Path, audit_record: dict[str, object]
) -> None:
    """Write a dependency-free SVG and optional PNG for the latest PR sweep."""
    curve = audit_record.get("curve")
    best = audit_record.get("best")
    if not isinstance(curve, list) or not curve or not isinstance(best, dict):
        raise ValueError("Invalid phrase-score audit record")
    points = [
        (float(item["recall"]), float(item["precision"]))
        for item in curve
        if isinstance(item, dict)
    ]
    width, height = 900, 650
    left, right, top, bottom = 85, 35, 65, 75
    plot_width = width - left - right
    plot_height = height - top - bottom

    def xy(recall: float, precision: float) -> tuple[float, float]:
        return (
            left + min(max(recall, 0.0), 1.0) * plot_width,
            top + (1.0 - min(max(precision, 0.0), 1.0)) * plot_height,
        )

    polyline = " ".join(
        f"{xy(recall, precision)[0]:.2f},{xy(recall, precision)[1]:.2f}"
        for recall, precision in points
    )
    best_x, best_y = xy(float(best["recall"]), float(best["precision"]))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="35" text-anchor="middle" font-size="22" font-family="Arial">Latest validation CTC phrase-score precision-recall sweep</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        f'<polyline points="{polyline}" fill="none" stroke="#4C78A8" stroke-width="3"/>',
        f'<circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="7" fill="#E45756"/>',
        f'<text x="{best_x + 10:.2f}" y="{best_y - 10:.2f}" font-size="14" font-family="Arial">best F1={float(best["f1_score"]):.4f}, threshold={float(best["threshold"]):.4f}</text>',
        f'<text x="450" y="625" text-anchor="middle" font-size="17" font-family="Arial">Recall</text>',
        f'<text x="22" y="325" text-anchor="middle" font-size="17" font-family="Arial" transform="rotate(-90 22 325)">Precision</text>',
        '</svg>',
    ]
    svg_path = run_directory / "phrase_score_pr_curve_latest.svg"
    png_path = run_directory / "phrase_score_pr_curve_latest.png"
    atomic_write_text(svg_path, "\n".join(svg) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 6), dpi=140)
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color="#4C78A8",
            linewidth=2,
        )
        axis.scatter(
            [float(best["recall"])],
            [float(best["precision"])],
            color="#E45756",
            s=55,
            label=(
                f"best F1={float(best['f1_score']):.4f}, "
                f"threshold={float(best['threshold']):.4f}"
            ),
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Latest validation CTC phrase-score sweep")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        temporary = png_path.with_name(png_path.name + ".tmp")
        figure.savefig(temporary, format="png", bbox_inches="tight")
        plt.close(figure)
        os.replace(temporary, png_path)
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        print(f"[phrase-audit] PNG refresh failed: {type(exc).__name__}: {exc}")


def collate_training_audio_items(items: Sequence[dict[str, object]]) -> dict[str, object]:
    batch = collate_audio_items(items)
    if int(batch["positive_count"]) < 1:
        raise RuntimeError(
            "Training batch has no full wake-phrase positive; this is a hard V12 invariant"
        )
    return batch


def make_loader(
    dataset,
    batch_plan: Sequence[Sequence[int]],
    num_workers: int,
    training: bool,
    loader_seed: int,
):
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(loader_seed))
    kwargs = {
        "dataset": dataset,
        "batch_sampler": list(batch_plan),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "collate_fn": collate_training_audio_items if training else collate_audio_items,
        "generator": loader_generator,
    }
    if num_workers > 0:
        kwargs.update({"prefetch_factor": 1, "persistent_workers": False})
    return torch.utils.data.DataLoader(**kwargs)


def validation_batch_plan(
    samples: Sequence[Sample],
    max_utterances_per_batch: int,
    max_padded_seconds: float,
    split: str = "validation",
    shard_aware_batching: bool = False,
    shard_row_group_size: int = 64,
) -> list[list[int]]:
    if shard_aware_batching:
        # Half-second duration buckets retain most padding efficiency while
        # grouping adjacent Parquet rows for fewer row-group reads.
        indices = sorted(
            (index for index, sample in enumerate(samples) if sample.split == split),
            key=lambda index: (
                int(samples[index].duration_seconds / 0.5),
                storage_locality_key(samples[index], shard_row_group_size),
                samples[index].duration_seconds,
                samples[index].uid,
            ),
        )
    else:
        indices = sorted(
            (index for index, sample in enumerate(samples) if sample.split == split),
            key=lambda index: samples[index].duration_seconds,
        )
    plan: list[list[int]] = []
    current: list[int] = []
    current_max = 0.0
    for index in indices:
        duration = samples[index].duration_seconds
        candidate_max = max(current_max, duration)
        if current and (
            len(current) + 1 > max_utterances_per_batch
            or candidate_max * (len(current) + 1) > max_padded_seconds
        ):
            plan.append(current)
            current = []
            current_max = 0.0
        current.append(index)
        current_max = max(current_max, duration)
    if current:
        plan.append(current)
    return plan


def batch_plan_audit(samples: Sequence[Sample], plan: Sequence[Sequence[int]]) -> dict[str, float]:
    counters = defaultdict(float)
    for batch in plan:
        if not batch:
            raise RuntimeError("Batch plan contains an empty batch")
        positives = sum(samples[index].is_positive for index in batch)
        if positives < 1:
            raise RuntimeError("Training batch plan contains a no-positive batch")
        counters["batches"] += 1
        counters["samples"] += len(batch)
        counters["positive"] += positives
        counters["negative"] += len(batch) - positives
        counters["hard_negative"] += sum(samples[index].is_hard_negative for index in batch)
        counters["new_negative"] += sum(
            not samples[index].is_positive
            and samples[index].source_dataset != "InfoRE2"
            for index in batch
        )
        counters["padded_hours"] += (
            max(samples[index].duration_seconds for index in batch) * len(batch) / 3600.0
        )
    counters["negative_positive_ratio"] = counters["negative"] / max(
        counters["positive"], 1
    )
    counters["hard_negative_share"] = counters["hard_negative"] / max(
        counters["negative"], 1
    )
    counters["new_negative_share"] = counters["new_negative"] / max(
        counters["negative"], 1
    )
    return dict(counters)


def batch_storage_audit(
    samples: Sequence[Sample],
    plan: Sequence[Sequence[int]],
    row_group_size: int,
) -> dict[str, float]:
    distinct_counts = [
        len(
            {
                storage_locality_key(samples[index], row_group_size)
                for index in batch
            }
        )
        for batch in plan
        if batch
    ]
    if not distinct_counts:
        return {
            "storage_groups_per_batch": 0.0,
            "max_storage_groups_per_batch": 0.0,
            "batches_with_at_most_two_storage_groups_share": 0.0,
        }
    return {
        "storage_groups_per_batch": sum(distinct_counts) / len(distinct_counts),
        "max_storage_groups_per_batch": float(max(distinct_counts)),
        "batches_with_at_most_two_storage_groups_share": sum(
            value <= 2 for value in distinct_counts
        )
        / len(distinct_counts),
    }


def binary_detection_metrics(
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
) -> dict[str, int | float]:
    """Return explicit wake-word confusion counts and derived rates."""
    tp = int(true_positives)
    fp = int(false_positives)
    tn = int(true_negatives)
    fn = int(false_negatives)
    if min(tp, fp, tn, fn) < 0:
        raise ValueError("Confusion-matrix counts must be non-negative")

    positive_count = tp + fn
    negative_count = tn + fp
    predicted_positive_count = tp + fp
    recall = tp / positive_count if positive_count else 0.0
    false_reject_rate = fn / positive_count if positive_count else 0.0
    false_accept_rate = fp / negative_count if negative_count else 0.0
    precision = tp / predicted_positive_count if predicted_positive_count else 0.0
    total_count = tp + fp + tn + fn
    accuracy = (tp + tn) / total_count if total_count else 0.0
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "recall": recall,
        "false_reject_rate": false_reject_rate,
        "false_accept_rate": false_accept_rate,
        "precision": precision,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


def weighted_binary_detection_metrics(
    true_positives: float,
    false_positives: float,
    true_negatives: float,
    false_negatives: float,
) -> dict[str, float]:
    """Binary metrics for deterministic speaker/family sample weights."""
    tp = float(true_positives)
    fp = float(false_positives)
    tn = float(true_negatives)
    fn = float(false_negatives)
    if min(tp, fp, tn, fn) < 0.0:
        raise ValueError("Weighted confusion values must be non-negative")
    positive_count = tp + fn
    negative_count = tn + fp
    predicted_positive_count = tp + fp
    recall = tp / positive_count if positive_count else 0.0
    false_reject_rate = fn / positive_count if positive_count else 0.0
    false_accept_rate = fp / negative_count if negative_count else 0.0
    precision = tp / predicted_positive_count if predicted_positive_count else 0.0
    total_count = tp + fp + tn + fn
    accuracy = (tp + tn) / total_count if total_count else 0.0
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "recall": recall,
        "false_reject_rate": false_reject_rate,
        "false_accept_rate": false_accept_rate,
        "precision": precision,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


def evaluate(
    student,
    loader,
    device,
    amp_dtype_name: str,
    preview_limit: int = 0,
    include_full_detection_metrics: bool = False,
    detection_mode: str = "greedy",
    phrase_score_threshold: float | None = None,
    collect_phrase_score_audit: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if detection_mode not in {"greedy", "phrase_score"}:
        raise ValueError(f"Unsupported detection mode: {detection_mode}")
    if detection_mode == "phrase_score" and phrase_score_threshold is None:
        raise ValueError("phrase_score detection requires a fixed validation threshold")
    student.eval()
    totals = defaultdict(float)
    audit_scores: list[float] = []
    audit_labels: list[bool] = []
    preview: list[dict[str, object]] = []
    preview_positive_limit = (int(preview_limit) + 1) // 2
    preview_negative_limit = int(preview_limit) // 2
    preview_positive_count = 0
    preview_negative_count = 0
    amp_enabled = amp_dtype_name != "fp32" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    with torch.inference_mode():
        for batch in loader:
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
            targets = batch["ctc_targets"].to(device, non_blocking=True)
            target_lengths = batch["ctc_target_lengths"].to(device, non_blocking=True)
            required_frames = batch["ctc_required_frames"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output, output_lengths = student(audio, audio_lengths)
                normalized_losses, raw_losses = pure_ctc_per_sample(
                    output,
                    output_lengths,
                    targets,
                    target_lengths,
                    required_frames,
                )
            predictions = greedy_token_sequences(output, output_lengths)
            need_phrase_scores = collect_phrase_score_audit or detection_mode == "phrase_score"
            phrase_scores = (
                ctc_keyword_log_odds(output, output_lengths).detach().cpu().tolist()
                if need_phrase_scores
                else [None] * len(predictions)
            )
            for index, (sample, prediction, phrase_score) in enumerate(
                zip(batch["samples"], predictions, phrase_scores)
            ):
                normalized_loss = float(normalized_losses[index].item())
                sample_weight = float(EVALUATION_SAMPLE_WEIGHTS.get(sample.uid, 1.0))
                weighting_applied = sample.uid in EVALUATION_SAMPLE_WEIGHTS
                if sample_weight <= 0.0 or not math.isfinite(sample_weight):
                    raise ValueError(f"Invalid evaluation weight for {sample.uid}: {sample_weight}")
                totals["samples"] += 1
                totals["normalized_loss_sum"] += normalized_loss
                totals["weighted_samples"] += sample_weight
                totals["weighted_normalized_loss_sum"] += normalized_loss * sample_weight
                totals["weighting_applied"] += int(weighting_applied)
                totals["raw_loss_sum"] += float(raw_losses[index].item())
                totals["exact_match"] += prediction == sample.reduced_token_ids
                greedy_triggered = contains_subsequence(prediction, WAKE_SEQUENCE)
                triggered = (
                    float(phrase_score) >= float(phrase_score_threshold)
                    if detection_mode == "phrase_score"
                    else greedy_triggered
                )
                if need_phrase_scores:
                    audit_scores.append(float(phrase_score))
                    audit_labels.append(bool(sample.is_positive))
                include_preview = (
                    sample.is_positive and preview_positive_count < preview_positive_limit
                ) or (
                    not sample.is_positive and preview_negative_count < preview_negative_limit
                )
                if include_preview:
                    preview.append(
                        {
                            "uid": sample.uid,
                            "source_dataset": sample.source_dataset,
                            "speaker_id": sample.speaker_id,
                            "transcript": sample.transcript,
                            "is_positive": sample.is_positive,
                            "target_token_ids": list(sample.reduced_token_ids),
                            "target_pieces": " ".join(
                                REDUCED_VOCABULARY[value]
                                for value in sample.reduced_token_ids
                            ),
                            "prediction_token_ids": list(prediction),
                            "prediction_pieces": " ".join(
                                REDUCED_VOCABULARY[value]
                                for value in prediction
                                if 0 <= value < len(REDUCED_VOCABULARY)
                            ),
                            "triggered": triggered,
                            "greedy_triggered": greedy_triggered,
                            "phrase_score": phrase_score,
                            "exact_match": prediction == sample.reduced_token_ids,
                            "normalized_ctc_loss": normalized_loss,
                            "duration_seconds": sample.duration_seconds,
                        }
                    )
                    if sample.is_positive:
                        preview_positive_count += 1
                    else:
                        preview_negative_count += 1
                if sample.is_positive:
                    totals["positive_samples"] += 1
                    totals["positive_loss_sum"] += normalized_loss
                    totals["true_accepts"] += triggered
                    totals["weighted_positive_samples"] += sample_weight
                    totals["weighted_positive_loss_sum"] += normalized_loss * sample_weight
                    totals["weighted_true_accepts"] += sample_weight * int(triggered)
                else:
                    totals["negative_samples"] += 1
                    totals["negative_loss_sum"] += normalized_loss
                    totals["negative_hours"] += sample.duration_seconds / 3600.0
                    totals["false_accepts"] += triggered
                    totals["weighted_negative_samples"] += sample_weight
                    totals["weighted_negative_loss_sum"] += normalized_loss * sample_weight
                    totals["weighted_false_accepts"] += sample_weight * int(triggered)
                    if sample.is_hard_negative:
                        totals["hard_negative_samples"] += 1
                        totals["hard_negative_false_accepts"] += triggered
            del audio, audio_lengths, targets, target_lengths, required_frames, output
    use_weights = totals["weighting_applied"] > 0
    if use_weights:
        positive_loss = totals["weighted_positive_loss_sum"] / max(
            totals["weighted_positive_samples"], 1e-12
        )
        negative_loss = totals["weighted_negative_loss_sum"] / max(
            totals["weighted_negative_samples"], 1e-12
        )
    else:
        positive_loss = totals["positive_loss_sum"] / max(totals["positive_samples"], 1)
        negative_loss = totals["negative_loss_sum"] / max(totals["negative_samples"], 1)
    macro_loss = 0.5 * (positive_loss + negative_loss)
    true_positives = int(totals["true_accepts"])
    false_positives = int(totals["false_accepts"])
    positive_samples = int(totals["positive_samples"])
    negative_samples = int(totals["negative_samples"])
    if use_weights:
        weighted_true_positives = totals["weighted_true_accepts"]
        weighted_false_positives = totals["weighted_false_accepts"]
        detection = weighted_binary_detection_metrics(
            true_positives=weighted_true_positives,
            false_positives=weighted_false_positives,
            true_negatives=(
                totals["weighted_negative_samples"] - weighted_false_positives
            ),
            false_negatives=(
                totals["weighted_positive_samples"] - weighted_true_positives
            ),
        )
    else:
        detection = binary_detection_metrics(
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=negative_samples - false_positives,
            false_negatives=positive_samples - true_positives,
        )
    metrics = {
        "ctc_loss": (
            totals["weighted_normalized_loss_sum"]
            / max(totals["weighted_samples"], 1e-12)
            if use_weights
            else totals["normalized_loss_sum"] / max(totals["samples"], 1)
        ),
        "positive_ctc_loss": positive_loss,
        "negative_ctc_loss": negative_loss,
        "macro_ctc_loss": macro_loss,
        "recall": detection["recall"],
        "precision": detection["precision"],
        "f1_score": detection["f1_score"],
        "accuracy": detection["accuracy"],
    }
    if use_weights:
        metrics["evaluation_weighting"] = "class_speaker_family_balanced"
    if include_full_detection_metrics:
        metrics.update(
            {
                **detection,
                "false_accepts_per_hour": false_positives
                / max(totals["negative_hours"], 1e-12),
                "false_accepts": false_positives,
                "negative_hours": totals["negative_hours"],
                "hard_negative_false_accept_rate": totals[
                    "hard_negative_false_accepts"
                ]
                / max(totals["hard_negative_samples"], 1),
                "reduced_exact_match_rate": totals["exact_match"]
                / max(totals["samples"], 1),
                "samples": int(totals["samples"]),
                "positive_samples": positive_samples,
                "negative_samples": negative_samples,
            }
        )
    if collect_phrase_score_audit:
        metrics["phrase_score_audit"] = phrase_score_threshold_audit(
            audit_scores,
            audit_labels,
        )
    student.train()
    return metrics, preview


def architecture_signature() -> dict[str, object]:
    return {
        "layers": STUDENT_N_LAYERS,
        "hidden_size": STUDENT_D_MODEL,
        "n_mels": STUDENT_N_MELS,
        "heads": STUDENT_N_HEADS,
        "subsampling_factor": SUBSAMPLING_FACTOR,
        "streaming_chunk_seconds": STREAMING_CHUNK_SECONDS,
        "streaming_chunk_mel_frames": STREAMING_CHUNK_MEL_FRAMES,
        "streaming_chunk_encoder_frames": STREAMING_CHUNK_ENCODER_FRAMES,
        "attention_context_size": (-1, STREAMING_CHUNK_ENCODER_FRAMES - 1),
        "attention_context_style": "chunked_limited",
        "causal_downsampling": True,
        "convolution_context": "causal",
        "convolution_norm": "layer_norm",
        "preprocessor_normalize": "NA",
    }


def training_signature(
    args: argparse.Namespace,
    ratios: Sequence[int],
    manifest_sha1: str,
    tokenizer_path: Path,
    old_piece_ids: Sequence[int],
) -> dict[str, object]:
    signature = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "trainer_variant": "v12_reduced_streaming_pure_ctc",
        "training_objective": "pure_ctc",
        "manifest_sha1": manifest_sha1,
        "tokenizer_path": str(tokenizer_path),
        "old_piece_ids": tuple(int(value) for value in old_piece_ids),
        "vocabulary": tuple(REDUCED_VOCABULARY),
        "output_dim": REDUCED_OUTPUT_DIM,
        "ctc_blank_id": CTC_BLANK_ID,
        "wake_sequence": WAKE_SEQUENCE,
        "target_projection": TARGET_PROJECTION,
        "empty_negative_target_loss": "exact_all_blank_path_normalized_per_input_frame",
        "nonempty_target_loss": "ctc_nll_normalized_per_target_token",
        "negative_positive_ratios": tuple(int(value) for value in ratios),
        "stage_min_epochs": int(args.stage_min_epochs),
        "stage_plateau_patience": int(args.stage_plateau_patience),
        "stage_min_relative_improvement": float(args.stage_min_relative_improvement),
        "stage_min_absolute_improvement": float(args.stage_min_absolute_improvement),
        "new_data_negative_share": float(args.new_data_negative_share),
        "hard_negative_share": float(args.hard_negative_share),
        "max_utterances_per_batch": int(args.max_utterances_per_batch),
        "max_padded_audio_seconds": float(args.max_padded_audio_seconds),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_steps": int(args.warmup_steps),
        "amp_dtype": args.amp_dtype,
        "seed": int(args.seed),
        **architecture_signature(),
    }
    validation_every_steps = int(getattr(args, "validation_every_steps", 0))
    if validation_every_steps > 0:
        signature.update(
            {
                "curriculum_observation_unit": "optimizer_step_validation",
                "validation_every_steps": validation_every_steps,
                "stage_min_optimizer_steps": int(
                    getattr(args, "stage_min_optimizer_steps", 0)
                ),
                "stage_min_validations": int(
                    getattr(args, "stage_min_validations", 1)
                ),
            }
        )
    return signature


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_checkpoint(
    student,
    optimizer,
    scheduler,
    scaler,
    signature: dict[str, object],
    curriculum: CurriculumState,
    optimizer_step: int,
    epoch: int,
    next_batch_index: int,
    best_validation_metric: float | None,
    best_metrics: dict[str, float],
    initialization_audit: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "saved_unix_time": time.time(),
        "signature": signature,
        "optimizer_step": int(optimizer_step),
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "curriculum_state": asdict(curriculum),
        "best_validation_metric": best_validation_metric,
        "best_metrics": dict(best_metrics),
        "initialization_audit": initialization_audit,
        "vocabulary": list(REDUCED_VOCABULARY),
        "ctc_blank_id": CTC_BLANK_ID,
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
    }


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def write_run_information(
    run_directory: Path,
    args: argparse.Namespace,
    signature: dict[str, object],
    samples: Sequence[Sample],
) -> dict[str, object]:
    split_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    positive_by_split: dict[str, int] = defaultdict(int)
    negative_by_split: dict[str, int] = defaultdict(int)
    for sample in samples:
        split_counts[sample.split] += 1
        source_counts[sample.source_dataset] += 1
        if sample.is_positive:
            positive_by_split[sample.split] += 1
        else:
            negative_by_split[sample.split] += 1
    launch = {
        "unix_time": time.time(),
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "arguments": json_safe(vars(args)),
    }
    config = {
        "format_version": 1,
        "run_name": args.run_name,
        "run_directory": str(run_directory),
        "training_signature": json_safe(signature),
        "dataset": {
            "samples": len(samples),
            "samples_by_split": dict(sorted(split_counts.items())),
            "samples_by_source": dict(sorted(source_counts.items())),
            "positive_by_split": dict(sorted(positive_by_split.items())),
            "negative_by_split": dict(sorted(negative_by_split.items())),
        },
        "first_recorded_launch": launch,
    }
    config_path = run_directory / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("training_signature") != config["training_signature"]:
            raise RuntimeError(f"Existing run_config.json has a different signature: {config_path}")
        config = existing
    else:
        atomic_write_text(
            config_path,
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    atomic_write_text(
        run_directory / "last_launch.json",
        json.dumps(launch, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    environment = {
        "unix_time": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": getattr(torch, "__version__", None),
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "packages": {
            name: package_version(name)
            for name in (
                "nemo_toolkit",
                "numpy",
                "pyarrow",
                "sentencepiece",
                "soundfile",
                "librosa",
                "wandb",
            )
        },
    }
    atomic_write_text(
        run_directory / "runtime_environment.json",
        json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return config


class RunArtifacts:
    """Resume-safe local metrics, plots, validation previews, and W&B."""

    def __init__(
        self,
        run_directory: Path,
        args: argparse.Namespace,
        run_config: dict[str, object],
        authoritative_step: int,
    ) -> None:
        self.run_directory = run_directory
        self.args = args
        self.metrics_path = run_directory / "training_metrics.jsonl"
        self.svg_path = run_directory / "loss_curve.svg"
        self.png_path = run_directory / "loss_curve.png"
        self.preview_path = run_directory / "validation_preview_latest.jsonl"
        self.test_metrics_path = run_directory / "test_metrics.json"
        self.test_preview_path = run_directory / "test_prediction_preview.jsonl"
        self.wandb_run_id_path = run_directory / "wandb_run_id.txt"
        self.wandb_identity_path = run_directory / "wandb_identity.json"
        self.wandb_backfill_path = run_directory / "wandb_history_backfilled.txt"
        self.records = self._load_and_prune_metrics(authoritative_step)
        self.wandb = None
        self.wandb_run = None
        self.finished = False
        if self.records:
            self.update_loss_plot()
        self._initialize_wandb(run_config, authoritative_step)
        atexit.register(self.finish)

    @staticmethod
    def _record_step(record: dict[str, object]) -> int | None:
        value = record.get("optimizer_step", record.get("step"))
        return None if value is None else int(value)

    def _load_and_prune_metrics(self, authoritative_step: int) -> list[dict[str, object]]:
        loaded: list[dict[str, object]] = []
        if self.metrics_path.is_file():
            with self.metrics_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Ignoring incomplete metrics line {line_number}: {self.metrics_path}")
                        continue
                    step = self._record_step(record)
                    if step is None:
                        continue
                    if record.get("kind") == "epoch_validation":
                        record["kind"] = "validation"
                    record["optimizer_step"] = step
                    if step <= int(authoritative_step):
                        loaded.append(record)
        atomic_write_jsonl(self.metrics_path, loaded)
        print(
            "Metrics logger     :",
            self.metrics_path,
            f"({len(loaded):,} records through step {authoritative_step:,})",
        )
        return loaded

    def _initialize_wandb(
        self, run_config: dict[str, object], authoritative_step: int
    ) -> None:
        if self.args.wandb_mode == "disabled":
            print("W&B tracking       : disabled")
            return
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "W&B SDK is not installed. Run: python -m pip install --upgrade wandb"
            ) from exc
        self.wandb = wandb
        if self.args.wandb_mode == "online":
            if not os.environ.get("WANDB_API_KEY", "").strip():
                raise RuntimeError(
                    "W&B online requires WANDB_API_KEY in this process; refusing an implicit credential."
                )
            if not self.args.wandb_expected_username or not self.args.wandb_entity:
                raise RuntimeError(
                    "W&B online requires --wandb-expected-username and --wandb-entity."
                )
            api = wandb.Api()
            viewer = api.viewer
            username = getattr(viewer, "username", None) or getattr(viewer, "_attrs", {}).get(
                "username"
            )
            if username != self.args.wandb_expected_username:
                raise RuntimeError(
                    "W&B credential mismatch: "
                    f"expected={self.args.wandb_expected_username!r} actual={username!r}"
                )
            identity = {
                "username": str(username),
                "entity": str(self.args.wandb_entity),
                "project": str(self.args.wandb_project),
            }
        else:
            identity = {
                "username": "offline",
                "entity": str(self.args.wandb_entity or "offline"),
                "project": str(self.args.wandb_project),
            }
        if self.wandb_run_id_path.is_file():
            run_id = self.wandb_run_id_path.read_text(encoding="utf-8").strip()
            if not run_id or not self.wandb_identity_path.is_file():
                raise RuntimeError("Incomplete saved W&B identity in checkpoint directory")
            saved_identity = json.loads(self.wandb_identity_path.read_text(encoding="utf-8"))
            if saved_identity != {**identity, "run_id": run_id}:
                raise RuntimeError(
                    f"Saved W&B identity does not match this launch: {saved_identity}"
                )
        else:
            run_id = hashlib.sha1(os.urandom(32)).hexdigest()[:16]
            atomic_write_text(self.wandb_run_id_path, run_id + "\n")
            atomic_write_text(
                self.wandb_identity_path,
                json.dumps(
                    {**identity, "run_id": run_id},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
        tags = [value.strip() for value in self.args.wandb_tags.split(",") if value.strip()]
        self.wandb_run = wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            id=run_id,
            name=self.args.run_name,
            group="parakeet-v12-reduced-ctc",
            job_type="train",
            tags=tags,
            config=json_safe(run_config),
            resume="allow",
            mode=self.args.wandb_mode,
            force=self.args.wandb_mode == "online",
            dir=str(self.run_directory),
        )
        self.wandb_run.define_metric("optimizer_step")
        self.wandb_run.define_metric("train/*", step_metric="optimizer_step")
        self.wandb_run.define_metric("validation/*", step_metric="optimizer_step")
        self.wandb_run.define_metric("test/*", step_metric="optimizer_step")
        for name in (
            "validation/macro_ctc_loss",
            "validation/ctc_loss",
        ):
            self.wandb_run.define_metric(name, summary="min")
        for name in (
            "validation/recall",
            "validation/precision",
            "validation/f1_score",
            "validation/accuracy",
        ):
            self.wandb_run.define_metric(name, summary="max")
        self.wandb_run.summary["latest_checkpoint_path"] = str(
            self.run_directory / "latest.pt"
        )
        self.wandb_run.summary["current_optimizer_step"] = int(authoritative_step)
        print("W&B tracking       :", self.args.wandb_mode)
        print("W&B project        :", self.args.wandb_project)
        print("W&B run ID         :", run_id)
        if getattr(self.wandb_run, "url", None):
            print("W&B run URL        :", self.wandb_run.url)
        if not self.args.no_wandb_backfill and not self.wandb_backfill_path.is_file():
            print(f"W&B backfill       : {len(self.records):,} durable records")
            for record in self.records:
                self._wandb_log(record)
            atomic_write_text(
                self.wandb_backfill_path,
                f"run_id={run_id}\nrecords={len(self.records)}\nthrough_step={authoritative_step}\n",
            )

    def _wandb_log(
        self,
        record: dict[str, object],
        preview: Sequence[dict[str, object]] | None = None,
    ) -> None:
        if self.wandb_run is None:
            return
        kind = str(record["kind"])
        prefix = "validation" if kind == "validation" else "test" if kind == "test" else "train"
        step = int(record["optimizer_step"])
        payload: dict[str, object] = {"optimizer_step": step}
        ignored = {"kind", "optimizer_step", "unix_time", "batch_plan_audit", "curriculum_state"}
        validation_tracked = {
            "epoch",
            "stage_index_before_observation",
            "stage_index_after_observation",
            "negative_positive_ratio",
            "train_ctc_loss",
            "epoch_wall_seconds",
            "ctc_loss",
            "positive_ctc_loss",
            "negative_ctc_loss",
            "macro_ctc_loss",
            "recall",
            "precision",
            "f1_score",
            "accuracy",
        }
        for key, value in record.items():
            if key in ignored:
                continue
            if kind == "validation" and key not in validation_tracked:
                continue
            if isinstance(value, (bool, int, float)):
                payload[f"{prefix}/{key}"] = value
        for nested_name in ("batch_plan_audit", "curriculum_state"):
            nested = record.get(nested_name)
            if isinstance(nested, dict):
                for key, value in nested.items():
                    if isinstance(value, (bool, int, float)):
                        payload[f"{prefix}/{nested_name}/{key}"] = value
        if preview and self.wandb is not None:
            columns = [
                "uid",
                "source_dataset",
                "speaker_id",
                "transcript",
                "is_positive",
                "target_pieces",
                "prediction_pieces",
                "triggered",
                "greedy_triggered",
                "phrase_score",
                "exact_match",
                "normalized_ctc_loss",
            ]
            payload[f"{prefix}/prediction_preview"] = self.wandb.Table(
                columns=columns,
                data=[[row.get(column) for column in columns] for row in preview],
            )
        self.wandb_run.log(payload)
        self.wandb_run.summary["current_optimizer_step"] = step

    def append(
        self,
        record: dict[str, object],
        preview: Sequence[dict[str, object]] | None = None,
    ) -> None:
        append_jsonl(self.metrics_path, record)
        self.records.append(record)
        if preview is not None:
            atomic_write_jsonl(self.preview_path, list(preview))
        self._wandb_log(record, preview)
        if record["kind"] == "validation" or int(record["optimizer_step"]) % int(
            self.args.plot_every_steps
        ) == 0:
            self.update_loss_plot()

    def record_test(
        self,
        record: dict[str, object],
        preview: Sequence[dict[str, object]],
    ) -> None:
        """Persist and upload the one-shot final held-out test evaluation."""
        atomic_write_text(
            self.test_metrics_path,
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_jsonl(self.test_preview_path, list(preview))
        self.records = [item for item in self.records if item.get("kind") != "test"]
        atomic_write_jsonl(self.metrics_path, self.records)
        append_jsonl(self.metrics_path, record)
        self.records.append(record)
        self._wandb_log(record, preview)

    @staticmethod
    def _moving_average(values: Sequence[float], window: int = 25) -> list[float]:
        output: list[float] = []
        running = 0.0
        for index, value in enumerate(values):
            running += value
            if index >= window:
                running -= values[index - window]
            output.append(running / min(index + 1, window))
        return output

    def _plot_series(self) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        train: list[tuple[int, float]] = []
        validation: list[tuple[int, float]] = []
        for record in self.records:
            step = self._record_step(record)
            if step is None:
                continue
            if record.get("kind") == "train":
                value = record.get("window_ctc_loss", record.get("ctc_loss"))
                if value is not None and math.isfinite(float(value)):
                    train.append((step, float(value)))
            elif record.get("kind") in {"validation", "epoch_validation"}:
                value = record.get("macro_ctc_loss")
                if value is not None and math.isfinite(float(value)):
                    validation.append((step, float(value)))
        return sorted(train), sorted(validation)

    def update_loss_plot(self) -> None:
        train, validation = self._plot_series()
        if not train and not validation:
            return
        all_points = train + validation
        min_step = min(point[0] for point in all_points)
        max_step = max(point[0] for point in all_points)
        min_loss = min(point[1] for point in all_points)
        max_loss = max(point[1] for point in all_points)
        if min_step == max_step:
            max_step += 1
        if min_loss == max_loss:
            padding = max(abs(min_loss) * 0.05, 1e-6)
            min_loss -= padding
            max_loss += padding
        width, height = 1200, 700
        left, right, top, bottom = 95, 35, 70, 80
        plot_width = width - left - right
        plot_height = height - top - bottom

        def xy(point: tuple[int, float]) -> tuple[float, float]:
            x = left + (point[0] - min_step) * plot_width / (max_step - min_step)
            y = top + (max_loss - point[1]) * plot_height / (max_loss - min_loss)
            return x, y

        train_smoothed = list(
            zip(
                (point[0] for point in train),
                self._moving_average([point[1] for point in train]),
            )
        )
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="600" y="36" text-anchor="middle" font-size="24" font-family="Arial">V12 reduced-vocabulary pure CTC loss</text>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        ]
        if train_smoothed:
            points = " ".join(f"{xy(point)[0]:.2f},{xy(point)[1]:.2f}" for point in train_smoothed)
            svg.append(f'<polyline points="{points}" fill="none" stroke="#1F5A94" stroke-width="3"/>')
        if validation:
            points = " ".join(f"{xy(point)[0]:.2f},{xy(point)[1]:.2f}" for point in validation)
            svg.append(f'<polyline points="{points}" fill="none" stroke="#E45756" stroke-width="2"/>')
            for point in validation:
                x, y = xy(point)
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#E45756"/>')
        svg.extend(
            [
                f'<text x="600" y="675" text-anchor="middle" font-size="17" font-family="Arial">Optimizer step</text>',
                f'<text x="25" y="350" text-anchor="middle" font-size="17" font-family="Arial" transform="rotate(-90 25 350)">Normalized CTC loss</text>',
                '</svg>',
            ]
        )
        atomic_write_text(self.svg_path, "\n".join(svg) + "\n")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(11, 6), dpi=140)
            if train:
                axis.plot(
                    [point[0] for point in train],
                    [point[1] for point in train],
                    color="#4C78A8",
                    alpha=0.22,
                    linewidth=1,
                    label="Train window CTC",
                )
                axis.plot(
                    [point[0] for point in train_smoothed],
                    [point[1] for point in train_smoothed],
                    color="#1F5A94",
                    linewidth=2,
                    label="Train moving average (25)",
                )
            if validation:
                axis.plot(
                    [point[0] for point in validation],
                    [point[1] for point in validation],
                    color="#E45756",
                    marker="o",
                    linewidth=1.5,
                    label="Validation macro CTC",
                )
            axis.set_title("V12 reduced-vocabulary pure CTC loss")
            axis.set_xlabel("Optimizer step")
            axis.set_ylabel("Normalized CTC loss")
            axis.grid(True, alpha=0.25)
            axis.legend()
            figure.tight_layout()
            temporary = self.png_path.with_name(self.png_path.name + ".tmp")
            figure.savefig(temporary, format="png", bbox_inches="tight")
            plt.close(figure)
            os.replace(temporary, self.png_path)
        except ModuleNotFoundError:
            print("[metrics] matplotlib unavailable; loss_curve.svg remains available")
        except Exception as exc:
            print(f"[metrics] PNG refresh failed; JSONL/SVG continue: {type(exc).__name__}: {exc}")

    def update_best_summary(
        self, metric_name: str, value: float, optimizer_step: int, path: Path
    ) -> None:
        if self.wandb_run is None:
            return
        key = metric_name.replace("_", "-")
        self.wandb_run.summary[f"best/{key}/value"] = float(value)
        self.wandb_run.summary[f"best/{key}/optimizer_step"] = int(optimizer_step)
        self.wandb_run.summary[f"best/{key}/checkpoint"] = str(path)

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        if self.wandb_run is not None:
            self.wandb_run.finish()


def assert_resume_signature(saved: object, current: dict[str, object]) -> None:
    if saved != current:
        saved_keys = set(saved) if isinstance(saved, dict) else set()
        current_keys = set(current)
        differences = {
            key: {
                "saved": saved.get(key) if isinstance(saved, dict) else None,
                "current": current.get(key),
            }
            for key in sorted(saved_keys | current_keys)
            if not isinstance(saved, dict) or saved.get(key) != current.get(key)
        }
        raise RuntimeError(
            "latest.pt is incompatible with this V12 configuration: "
            + json.dumps(differences, ensure_ascii=False, default=str)
        )


def train_v12(
    args: argparse.Namespace,
    ratios: Sequence[int],
    samples: Sequence[Sample],
    signature: dict[str, object],
    old_piece_ids: Sequence[int],
    run_directory: Path,
) -> None:
    global StudentASR
    StudentASR = build_student_class()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    latest_path = run_directory / "latest.pt"
    best_path = run_directory / "best.pt"
    final_path = run_directory / "student_final_weights.pt"
    student = StudentASR().to(device)
    initialization_audit = None

    resume_payload = None
    if latest_path.is_file():
        resume_payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        assert_resume_signature(resume_payload.get("signature"), signature)
        student.load_state_dict(resume_payload["student_state_dict"], strict=True)
        print("Resuming exact checkpoint:", latest_path)
    elif args.init_checkpoint is not None:
        initialization_audit = initialize_from_checkpoint(
            student, args.init_checkpoint, old_piece_ids
        )
        print("Initialized from checkpoint:")
        print(json.dumps(initialization_audit, ensure_ascii=False, indent=2))
    elif not args.allow_random_init:
        raise ValueError(
            "A new V12 run requires --init-checkpoint. Use --allow-random-init only "
            "for an intentional from-scratch ablation."
        )
    else:
        initialization_audit = {"mode": "random_initialization"}
        print("WARNING: trainer is starting from random initialization")

    initial_curriculum_stage = int(
        getattr(args, "initial_curriculum_stage", 0)
    )
    if not 0 <= initial_curriculum_stage < len(ratios):
        raise ValueError(
            f"Initial curriculum stage {initial_curriculum_stage} is outside "
            f"[0, {len(ratios) - 1}]"
        )
    curriculum = CurriculumState(stage_index=initial_curriculum_stage)
    stage_lr_decay = float(getattr(args, "stage_lr_decay", 1.0))
    restore_stage_best = bool(getattr(args, "restore_stage_best", False))

    encoder_learning_rate = getattr(args, "encoder_learning_rate", None)
    decoder_learning_rate = getattr(args, "decoder_learning_rate", None)
    head_only_optimizer_steps = int(getattr(args, "head_only_optimizer_steps", 0))
    if encoder_learning_rate is None and decoder_learning_rate is None:
        optimizer_groups = [
            {
                "params": list(student.parameters()),
                "lr": float(args.learning_rate),
                "role": "all",
            }
        ]
    else:
        encoder_lr = float(
            args.learning_rate if encoder_learning_rate is None else encoder_learning_rate
        )
        decoder_lr = float(
            args.learning_rate if decoder_learning_rate is None else decoder_learning_rate
        )
        decoder_parameters = list(student.decoder.parameters())
        decoder_parameter_ids = {id(parameter) for parameter in decoder_parameters}
        encoder_parameters = [
            parameter
            for parameter in student.parameters()
            if id(parameter) not in decoder_parameter_ids
        ]
        optimizer_groups = [
            {"params": encoder_parameters, "lr": encoder_lr, "role": "encoder"},
            {"params": decoder_parameters, "lr": decoder_lr, "role": "decoder"},
        ]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    def lr_lambda(step: int, role: str = "all") -> float:
        warmup_factor = (
            1.0
            if args.warmup_steps <= 0
            else min(1.0, max(step + 1, 1) / float(args.warmup_steps))
        )
        if role == "encoder" and step < head_only_optimizer_steps:
            return 0.0
        return warmup_factor * stage_lr_decay ** int(curriculum.stage_index)

    group_roles = [str(group.get("role", "all")) for group in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            (lambda step, role=role: lr_lambda(step, role)) for role in group_roles
        ],
    )
    scaler_enabled = args.amp_dtype == "fp16" and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    optimizer_step = 0
    epoch = 0
    next_batch_index = 0
    best_validation_metric = None
    best_metrics = {
        "macro_ctc_loss": float("inf"),
        "ctc_loss": float("inf"),
        "recall": float("-inf"),
        "f1_score": float("-inf"),
        **{
            f"stage_{stage_index}_f1_score": float("-inf")
            for stage_index in range(len(ratios))
        },
    }
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload.get("scaler_state_dict", {}))
        optimizer_step = int(resume_payload["optimizer_step"])
        epoch = int(resume_payload["epoch"])
        next_batch_index = int(resume_payload["next_batch_index"])
        curriculum = CurriculumState(**resume_payload["curriculum_state"])
        best_validation_metric = resume_payload.get("best_validation_metric")
        for name, value in resume_payload.get("best_metrics", {}).items():
            if name in best_metrics:
                best_metrics[name] = float(value)
        initialization_audit = resume_payload.get("initialization_audit")
        restore_rng_state(resume_payload.get("rng_state"))
        del resume_payload

    encoder_trainable = {"value": True}

    def synchronize_encoder_trainability() -> None:
        should_train = optimizer_step >= head_only_optimizer_steps
        if encoder_trainable["value"] == should_train:
            return
        for parameter in student.encoder.parameters():
            parameter.requires_grad_(should_train)
        encoder_trainable["value"] = should_train
        print(
            "Encoder fine-tuning:",
            "enabled" if should_train else "frozen for decoder warm-up",
            f"(step={optimizer_step:,})",
        )

    # Force the first synchronization even though the default state is True.
    encoder_trainable["value"] = not (optimizer_step >= head_only_optimizer_steps)
    synchronize_encoder_trainability()

    def apply_current_stage_learning_rate() -> float:
        active_step = int(getattr(scheduler, "last_epoch", optimizer_step))
        learning_rates = []
        for base_lr, group, role in zip(
            scheduler.base_lrs, optimizer.param_groups, group_roles
        ):
            learning_rate = float(base_lr) * lr_lambda(active_step, role)
            group["lr"] = learning_rate
            learning_rates.append(learning_rate)
        scheduler._last_lr = learning_rates
        return learning_rates[0]

    apply_current_stage_learning_rate()

    if initialization_audit is not None:
        atomic_write_text(
            run_directory / "initialization_audit.json",
            json.dumps(
                initialization_audit,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    run_config = write_run_information(run_directory, args, signature, samples)
    artifacts = RunArtifacts(run_directory, args, run_config, optimizer_step)

    dataset = UnifiedAudioDataset(samples)
    validation_plan = validation_batch_plan(
        samples,
        args.max_utterances_per_batch,
        args.max_padded_audio_seconds,
        shard_aware_batching=bool(getattr(args, "shard_aware_batching", False)),
        shard_row_group_size=int(getattr(args, "shard_row_group_size", 64)),
    )
    validation_loader = make_loader(
        dataset,
        validation_plan,
        args.num_workers,
        training=False,
        loader_seed=args.seed + 900_001,
    )
    development_only = bool(getattr(args, "development_only", False))
    test_plan = validation_batch_plan(
        samples,
        args.max_utterances_per_batch,
        args.max_padded_audio_seconds,
        split="test",
        shard_aware_batching=bool(getattr(args, "shard_aware_batching", False)),
        shard_row_group_size=int(getattr(args, "shard_row_group_size", 64)),
    )
    if not test_plan and not development_only:
        raise RuntimeError("Test split is empty; final held-out evaluation is required")
    test_loader = (
        None
        if not test_plan
        else make_loader(
            dataset,
            test_plan,
            args.num_workers,
            training=False,
            loader_seed=args.seed + 900_002,
        )
    )
    amp_enabled = args.amp_dtype != "fp32" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    stop_requested = {"value": False}

    def request_stop(signum, _frame):
        stop_requested["value"] = True
        print(f"Received signal {signum}; will checkpoint after this optimizer step")

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), request_stop)

    def save_latest(current_epoch: int, current_next_batch: int) -> None:
        payload = build_checkpoint(
            student,
            optimizer,
            scheduler,
            scaler,
            signature,
            curriculum,
            optimizer_step,
            current_epoch,
            current_next_batch,
            best_validation_metric,
            best_metrics,
            initialization_audit,
        )
        atomic_torch_save(payload, latest_path)
        del payload
        print(
            f"Saved {latest_path} (step={optimizer_step:,}, epoch={current_epoch + 1}, "
            f"next_batch={current_next_batch:,})"
        )

    def save_metric_best(
        path: Path,
        metric_name: str,
        metric_value: float,
        completed_epoch: int,
        validation_metrics: dict[str, object],
        stage_index: int | None = None,
    ) -> None:
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_variant": signature.get(
                "trainer_variant", "v12_reduced_streaming_pure_ctc"
            ),
            "criterion": metric_name,
            "criterion_value": float(metric_value),
            "optimizer_step": optimizer_step,
            "completed_epochs": completed_epoch,
            "stage_index": (
                int(curriculum.stage_index) if stage_index is None else int(stage_index)
            ),
            "signature": signature,
            "validation_metrics": dict(validation_metrics),
            "vocabulary": list(REDUCED_VOCABULARY),
            "ctc_blank_id": CTC_BLANK_ID,
            "initialization_audit": initialization_audit,
            "student_state_dict": student.state_dict(),
        }
        atomic_torch_save(payload, path)
        del payload
        artifacts.update_best_summary(metric_name, metric_value, optimizer_step, path)
        print(f"Saved best {metric_name}={metric_value:.8f}: {path}")

    validation_every_steps = int(getattr(args, "validation_every_steps", 0))

    def run_validation_observation(
        *,
        epoch_display: int,
        checkpoint_epoch: int,
        checkpoint_next_batch: int,
        ratio: int,
        train_ctc_loss: float,
        train_wall_seconds: float,
        plan_audit: dict[str, object],
        trigger: str,
    ) -> str:
        nonlocal best_validation_metric
        metrics, validation_preview = evaluate(
            student,
            validation_loader,
            device,
            str(getattr(args, "validation_amp_dtype", args.amp_dtype)),
            args.validation_preview_samples,
            detection_mode=getattr(args, "detection_mode", "greedy"),
            phrase_score_threshold=getattr(args, "phrase_score_threshold", None),
            collect_phrase_score_audit=bool(
                getattr(args, "phrase_score_audit", False)
            ),
        )
        phrase_score_audit = metrics.pop("phrase_score_audit", None)
        observed_stage = curriculum.stage_index
        observed_stage_start_step = curriculum.stage_start_optimizer_step
        observed_validation_count = (
            curriculum.validation_observations_in_stage + 1
            if validation_every_steps > 0
            else curriculum.epochs_in_stage + 1
        )
        if validation_every_steps > 0:
            action = curriculum_observe_steps(
                curriculum,
                metrics["macro_ctc_loss"],
                optimizer_step,
                len(ratios),
                int(getattr(args, "stage_min_optimizer_steps", 0)),
                int(getattr(args, "stage_min_validations", 1)),
                args.stage_plateau_patience,
                args.stage_min_relative_improvement,
                args.stage_min_absolute_improvement,
                detection_metric=metrics["f1_score"],
                detection_patience=int(
                    getattr(args, "detection_veto_patience", 0)
                ),
                detection_absolute_improvement=float(
                    getattr(args, "detection_min_absolute_improvement", 0.0)
                ),
            )
        else:
            action = curriculum_observe(
                curriculum,
                metrics["macro_ctc_loss"],
                len(ratios),
                args.stage_min_epochs,
                args.stage_plateau_patience,
                args.stage_min_relative_improvement,
                args.stage_min_absolute_improvement,
            )
        steps_in_stage = optimizer_step - observed_stage_start_step
        print(
            f"[validation] trigger={trigger} step={optimizer_step:,} "
            f"epoch={epoch_display} stage={observed_stage} "
            f"macro_ctc={metrics['macro_ctc_loss']:.6f} "
            f"pos_ctc={metrics['positive_ctc_loss']:.6f} "
            f"neg_ctc={metrics['negative_ctc_loss']:.6f} "
            f"recall={100 * metrics['recall']:.2f}% "
            f"precision={100 * metrics['precision']:.2f}% "
            f"F1={100 * metrics['f1_score']:.2f}% "
            f"accuracy={100 * metrics['accuracy']:.2f}% "
            f"action={action} ctc_bad={curriculum.bad_validations} "
            f"f1_bad={curriculum.bad_detection_validations}"
        )
        record = {
            "kind": "validation",
            "unix_time": time.time(),
            "epoch": epoch_display,
            "optimizer_step": optimizer_step,
            "validation_trigger": trigger,
            "stage_index_before_observation": observed_stage,
            "stage_index_after_observation": curriculum.stage_index,
            "stage_start_optimizer_step": observed_stage_start_step,
            "optimizer_steps_in_stage": steps_in_stage,
            "validation_observations_in_stage": observed_validation_count,
            "negative_positive_ratio": ratio,
            "curriculum_action": action,
            "curriculum_state": asdict(curriculum),
            "train_ctc_loss": train_ctc_loss,
            "train_interval_wall_seconds": train_wall_seconds,
            "batch_plan_audit": plan_audit,
            **metrics,
        }
        if action in {"advanced", "complete"}:
            endpoint_path = run_directory / f"stage_{observed_stage}_endpoint.pt"
            endpoint_payload = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "trainer_variant": signature.get(
                    "trainer_variant", "v12_reduced_streaming_pure_ctc"
                ),
                "criterion": "stage_endpoint",
                "optimizer_step": optimizer_step,
                "completed_epochs": checkpoint_epoch,
                "stage_index": observed_stage,
                "signature": signature,
                "validation_metrics": dict(metrics),
                "vocabulary": list(REDUCED_VOCABULARY),
                "ctc_blank_id": CTC_BLANK_ID,
                "initialization_audit": initialization_audit,
                "student_state_dict": student.state_dict(),
            }
            atomic_torch_save(endpoint_payload, endpoint_path)
            del endpoint_payload
            print("Saved stage endpoint:", endpoint_path)

        if action in {"advanced", "complete"} and restore_stage_best:
            record["stage_restore_checkpoint"] = str(
                run_directory / f"stage_{observed_stage}_best.pt"
            )
        artifacts.append(record, validation_preview)
        if isinstance(phrase_score_audit, dict):
            phrase_audit_record = {
                "kind": "validation_phrase_score_audit",
                "unix_time": time.time(),
                "epoch": epoch_display,
                "optimizer_step": optimizer_step,
                "stage_index": observed_stage,
                "active_detection_metrics": {
                    name: float(metrics[name])
                    for name in ("recall", "precision", "f1_score", "accuracy")
                },
                **phrase_score_audit,
            }
            append_jsonl(
                run_directory / "phrase_score_validation_audit.jsonl",
                phrase_audit_record,
            )
            write_phrase_score_audit_plot(run_directory, phrase_audit_record)
            phrase_best = phrase_score_audit["best"]
            print(
                "[phrase-audit] "
                f"best_threshold={float(phrase_best['threshold']):.6f} "
                f"recall={100 * float(phrase_best['recall']):.2f}% "
                f"precision={100 * float(phrase_best['precision']):.2f}% "
                f"F1={100 * float(phrase_best['f1_score']):.2f}% "
                f"accuracy={100 * float(phrase_best['accuracy']):.2f}% "
                "(validation diagnostic only)"
            )

        metric_specs = (
            ("macro_ctc_loss", "best_macro_ctc.pt", True),
            ("ctc_loss", "best_ctc.pt", True),
            ("recall", "best_positive_recall.pt", False),
            ("f1_score", "best_f1.pt", False),
        )
        metric_improvements: dict[str, bool] = {}
        for metric_name, filename, minimize in metric_specs:
            value = float(metrics[metric_name])
            previous = float(best_metrics[metric_name])
            improved = value < previous if minimize else value > previous
            metric_improvements[metric_name] = improved
            if improved:
                best_metrics[metric_name] = value
                save_metric_best(
                    run_directory / filename,
                    metric_name,
                    value,
                    checkpoint_epoch,
                    metrics,
                    stage_index=observed_stage,
                )

        stage_f1_key = f"stage_{observed_stage}_f1_score"
        stage_f1_value = float(metrics["f1_score"])
        if stage_f1_value > float(best_metrics[stage_f1_key]):
            best_metrics[stage_f1_key] = stage_f1_value
            save_metric_best(
                run_directory / f"stage_{observed_stage}_best_f1.pt",
                "f1_score",
                stage_f1_value,
                checkpoint_epoch,
                metrics,
                stage_index=observed_stage,
            )

        if metric_improvements["macro_ctc_loss"]:
            best_validation_metric = float(metrics["macro_ctc_loss"])

        if action == "improved":
            stage_path = run_directory / f"stage_{observed_stage}_best.pt"
            stage_payload = build_checkpoint(
                student,
                optimizer,
                scheduler,
                scaler,
                signature,
                curriculum,
                optimizer_step,
                checkpoint_epoch,
                checkpoint_next_batch,
                best_validation_metric,
                best_metrics,
                initialization_audit,
            )
            stage_payload["validation_metrics"] = metrics
            stage_payload["stage_index"] = observed_stage
            atomic_torch_save(stage_payload, stage_path)
            del stage_payload
            print("Saved stage best:", stage_path)
        if metric_improvements["macro_ctc_loss"]:
            best_payload = build_checkpoint(
                student,
                optimizer,
                scheduler,
                scaler,
                signature,
                curriculum,
                optimizer_step,
                checkpoint_epoch,
                checkpoint_next_batch,
                best_validation_metric,
                best_metrics,
                initialization_audit,
            )
            best_payload["validation_metrics"] = metrics
            atomic_torch_save(best_payload, best_path)
            del best_payload
            print("Saved global best:", best_path)

        if action in {"advanced", "complete"} and restore_stage_best:
            stage_path = run_directory / f"stage_{observed_stage}_best.pt"
            if not stage_path.is_file():
                raise FileNotFoundError(
                    f"Cannot hand off stage {observed_stage}: missing {stage_path}"
                )
            stage_best = torch.load(stage_path, map_location="cpu", weights_only=False)
            student.load_state_dict(stage_best["student_state_dict"], strict=True)
            optimizer.load_state_dict(stage_best["optimizer_state_dict"])
            scheduler.load_state_dict(stage_best["scheduler_state_dict"])
            scaler.load_state_dict(stage_best.get("scaler_state_dict", {}))
            restored_step = int(stage_best["optimizer_step"])
            restored_metric = float(stage_best["validation_metrics"]["macro_ctc_loss"])
            del stage_best
            active_learning_rate = apply_current_stage_learning_rate()
            append_jsonl(
                run_directory / "stage_transition_history.jsonl",
                {
                    "unix_time": time.time(),
                    "action": action,
                    "completed_epoch": checkpoint_epoch,
                    "current_optimizer_step": optimizer_step,
                    "source_stage_index": observed_stage,
                    "source_checkpoint": str(stage_path),
                    "source_optimizer_step": restored_step,
                    "source_macro_ctc_loss": restored_metric,
                    "next_stage_index": curriculum.stage_index,
                    "next_negative_positive_ratio": int(
                        ratios[curriculum.stage_index]
                    ),
                    "stage_lr_decay": stage_lr_decay,
                    "active_learning_rate": active_learning_rate,
                },
            )
            print(
                f"[stage-restore] source_stage={observed_stage} "
                f"source_step={restored_step:,} macro_ctc={restored_metric:.6f} "
                f"next_stage={curriculum.stage_index} lr={active_learning_rate:.3e}"
            )
        elif action in {"advanced", "complete"}:
            active_learning_rate = apply_current_stage_learning_rate()
            append_jsonl(
                run_directory / "stage_transition_history.jsonl",
                {
                    "unix_time": time.time(),
                    "action": action,
                    "completed_epoch": checkpoint_epoch,
                    "current_optimizer_step": optimizer_step,
                    "source_stage_index": observed_stage,
                    "source_checkpoint": str(endpoint_path),
                    "source_optimizer_step": optimizer_step,
                    "source_macro_ctc_loss": float(metrics["macro_ctc_loss"]),
                    "next_stage_index": curriculum.stage_index,
                    "next_negative_positive_ratio": int(
                        ratios[curriculum.stage_index]
                    ),
                    "stage_lr_decay": stage_lr_decay,
                    "active_learning_rate": active_learning_rate,
                    "handoff": "endpoint_without_rewind",
                },
            )
            print(
                f"[stage-endpoint] source_stage={observed_stage} "
                f"source_step={optimizer_step:,} "
                f"next_stage={curriculum.stage_index} lr={active_learning_rate:.3e}"
            )
        return action

    print("=" * 88)
    variant_name = str(signature.get("trainer_variant", ""))
    trainer_label = (
        "V12.2"
        if variant_name.startswith("v12.2")
        else "V12.1" if variant_name.startswith("v12.1") else "V12"
    )
    print(f"{trainer_label} PURE-CTC STREAMING TRAINING")
    print("=" * 88)
    print("Architecture       :", architecture_signature())
    print("Reduced vocabulary :", list(enumerate(REDUCED_VOCABULARY)))
    print("CTC blank          :", CTC_BLANK_ID)
    print("Curriculum ratios  :", list(ratios), "negative:positive")
    if validation_every_steps > 0:
        print(
            "Stage controller   :",
            f"metric=macro_ctc_loss validation_every_steps={validation_every_steps:,} "
            f"min_stage_steps={args.stage_min_optimizer_steps:,} "
            f"min_validations={args.stage_min_validations} "
            f"patience={args.stage_plateau_patience} "
            f"relative_improvement={args.stage_min_relative_improvement} "
            f"absolute_improvement={args.stage_min_absolute_improvement}",
        )
    else:
        print(
            "Stage controller   :",
            f"min_epochs={args.stage_min_epochs}, patience={args.stage_plateau_patience}, "
            f"relative_improvement={args.stage_min_relative_improvement}",
        )
    if restore_stage_best:
        print(
            "Stage handoff      : restore stage-best model/optimizer/scheduler/scaler; "
            f"LR decay={stage_lr_decay:.4f}"
        )
    print("Detection mode    :", getattr(args, "detection_mode", "greedy"))
    print("Parameters         :", f"{sum(value.numel() for value in student.parameters()):,}")

    while not curriculum.complete:
        if args.max_epochs and epoch >= args.max_epochs:
            print("Reached --max-epochs safety cap; pausing without marking convergence")
            save_latest(epoch, next_batch_index)
            return
        ratio = int(ratios[curriculum.stage_index])
        plan_build_started = time.perf_counter()
        full_plan = build_epoch_batch_plan(
            samples,
            ratio,
            args.max_utterances_per_batch,
            args.max_padded_audio_seconds,
            args.new_data_negative_share,
            args.hard_negative_share,
            args.seed,
            epoch,
            shard_aware_batching=bool(
                getattr(args, "shard_aware_batching", False)
            ),
            shard_row_group_size=int(getattr(args, "shard_row_group_size", 64)),
            shard_locality_block_size=int(
                getattr(args, "shard_locality_block_size", 64)
            ),
        )
        plan_build_seconds = time.perf_counter() - plan_build_started
        if next_batch_index > len(full_plan):
            raise RuntimeError(
                f"Resume batch index {next_batch_index} exceeds epoch plan {len(full_plan)}"
            )
        plan_audit_started = time.perf_counter()
        plan_audit = batch_plan_audit(samples, full_plan)
        if bool(getattr(args, "shard_aware_batching", False)):
            plan_audit.update(
                batch_storage_audit(
                    samples,
                    full_plan,
                    int(getattr(args, "shard_row_group_size", 64)),
                )
            )
        plan_audit_seconds = time.perf_counter() - plan_audit_started
        print(
            f"[epoch {epoch + 1}] stage={curriculum.stage_index} target_neg:pos={ratio}:1 "
            f"batches={int(plan_audit['batches']):,} actual_neg:pos="
            f"{plan_audit['negative_positive_ratio']:.3f}:1 "
            f"hard_neg={100 * plan_audit['hard_negative_share']:.1f}% "
            f"new_neg={100 * plan_audit['new_negative_share']:.1f}%"
            f" plan_build={plan_build_seconds:.2f}s"
            f" plan_audit={plan_audit_seconds:.2f}s"
            + (
                f" storage_groups/batch={plan_audit['storage_groups_per_batch']:.2f}"
                if "storage_groups_per_batch" in plan_audit
                else ""
            )
        )
        train_loader = make_loader(
            dataset,
            full_plan[next_batch_index:],
            args.num_workers,
            training=True,
            loader_seed=args.seed + epoch * 1_000_003,
        )
        student.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        epoch_start = time.time()
        log_window_start = time.time()
        log_window_loss_sum = 0.0
        log_window_samples = 0
        log_window_audio_seconds = 0.0
        log_window_positive = 0
        log_window_negative = 0
        transition_action: str | None = None
        target_encoder_frames = int(
            getattr(args, "target_encoder_frames_per_update", 0)
        )
        accumulated_frames = 0
        accumulated_samples = 0
        accumulated_microbatches = 0
        accumulated_loss_sum = 0.0
        accumulated_positive = 0
        accumulated_audio_seconds = 0.0
        optimizer.zero_grad(set_to_none=True)
        for local_batch_index, batch in enumerate(train_loader):
            global_batch_index = next_batch_index + local_batch_index
            if int(batch["positive_count"]) < 1:
                raise RuntimeError("No-positive training batch reached the optimizer")
            if accumulated_samples == 0:
                synchronize_encoder_trainability()
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
            targets = batch["ctc_targets"].to(device, non_blocking=True)
            target_lengths = batch["ctc_target_lengths"].to(device, non_blocking=True)
            required_frames = batch["ctc_required_frames"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output, output_lengths = student(audio, audio_lengths)
                normalized_losses, _raw_losses = pure_ctc_per_sample(
                    output,
                    output_lengths,
                    targets,
                    target_lengths,
                    required_frames,
                )
                loss_sum = normalized_losses.sum()
                backward_loss = (
                    loss_sum
                    if target_encoder_frames > 0
                    else loss_sum / max(int(normalized_losses.numel()), 1)
                )
            scaler.scale(backward_loss).backward()
            batch_size = len(batch["samples"])
            batch_positive = int(batch["positive_count"])
            batch_audio_seconds = sum(
                float(sample.duration_seconds) for sample in batch["samples"]
            )
            batch_loss_sum = float(loss_sum.detach().item())
            batch_frames = int(output_lengths.detach().sum().item())
            accumulated_frames += batch_frames
            accumulated_samples += batch_size
            accumulated_microbatches += 1
            accumulated_loss_sum += batch_loss_sum
            accumulated_positive += batch_positive
            accumulated_audio_seconds += batch_audio_seconds
            epoch_loss_sum += batch_loss_sum
            epoch_samples += batch_size
            is_last_batch = global_batch_index + 1 >= len(full_plan)
            apply_update = (
                target_encoder_frames <= 0
                or accumulated_frames >= target_encoder_frames
                or is_last_batch
                or stop_requested["value"]
            )
            if not apply_update:
                del (
                    audio,
                    audio_lengths,
                    targets,
                    target_lengths,
                    required_frames,
                    output,
                    output_lengths,
                    normalized_losses,
                    loss_sum,
                    backward_loss,
                    _raw_losses,
                )
                continue
            scaler.unscale_(optimizer)
            if target_encoder_frames > 0:
                gradient_scale = 1.0 / max(accumulated_samples, 1)
                for parameter in student.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(gradient_scale)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), args.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite gradient norm: {gradient_norm}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer_step += 1
            next_batch_after = global_batch_index + 1
            loss_value = accumulated_loss_sum / max(accumulated_samples, 1)
            update_samples = accumulated_samples
            update_positive = accumulated_positive
            update_frames = accumulated_frames
            update_microbatches = accumulated_microbatches
            update_audio_seconds = accumulated_audio_seconds
            log_window_loss_sum += accumulated_loss_sum
            log_window_samples += update_samples
            log_window_audio_seconds += update_audio_seconds
            log_window_positive += update_positive
            log_window_negative += update_samples - update_positive
            accumulated_frames = 0
            accumulated_samples = 0
            accumulated_microbatches = 0
            accumulated_loss_sum = 0.0
            accumulated_positive = 0
            accumulated_audio_seconds = 0.0
            optimizer.zero_grad(set_to_none=True)

            if optimizer_step % args.log_every_steps == 0:
                log_wall_seconds = max(time.time() - log_window_start, 1e-9)
                window_loss = log_window_loss_sum / max(log_window_samples, 1)
                train_record = {
                    "kind": "train",
                    "unix_time": time.time(),
                    "optimizer_step": optimizer_step,
                    "epoch": epoch + 1,
                    "stage_index": curriculum.stage_index,
                    "negative_positive_ratio_target": ratio,
                    "batch_index": next_batch_after,
                    "batches_in_epoch": len(full_plan),
                    "ctc_loss": loss_value,
                    "window_ctc_loss": window_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "decoder_learning_rate": float(optimizer.param_groups[-1]["lr"]),
                    "gradient_norm": float(gradient_norm.item()),
                    "encoder_frames_in_update": update_frames,
                    "microbatches_in_update": update_microbatches,
                    "window_samples": log_window_samples,
                    "window_positive_samples": log_window_positive,
                    "window_negative_samples": log_window_negative,
                    "window_audio_seconds": log_window_audio_seconds,
                    "window_wall_seconds": log_wall_seconds,
                    "samples_per_second": log_window_samples / log_wall_seconds,
                    "audio_realtime_factor": log_window_audio_seconds / log_wall_seconds,
                }
                artifacts.append(train_record)
                print(
                    f"[train] step={optimizer_step:,} epoch={epoch + 1} "
                    f"batch={next_batch_after:,}/{len(full_plan):,} "
                    f"loss={loss_value:.6f} window={window_loss:.6f} "
                    f"enc_lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"head_lr={optimizer.param_groups[-1]['lr']:.3e} "
                    f"frames={update_frames:,} micro={update_microbatches} "
                    f"pos={update_positive}/{update_samples}"
                )
                log_window_start = time.time()
                log_window_loss_sum = 0.0
                log_window_samples = 0
                log_window_audio_seconds = 0.0
                log_window_positive = 0
                log_window_negative = 0
            if optimizer_step % args.save_every_steps == 0:
                save_latest(epoch, next_batch_after)
            del (
                audio,
                audio_lengths,
                targets,
                target_lengths,
                required_frames,
                output,
                output_lengths,
                normalized_losses,
                loss_sum,
                backward_loss,
                _raw_losses,
            )

            if stop_requested["value"]:
                save_latest(epoch, next_batch_after)
                return
            if (
                validation_every_steps > 0
                and optimizer_step % validation_every_steps == 0
            ):
                action = run_validation_observation(
                    epoch_display=epoch + 1,
                    checkpoint_epoch=epoch,
                    checkpoint_next_batch=next_batch_after,
                    ratio=ratio,
                    train_ctc_loss=epoch_loss_sum / max(epoch_samples, 1),
                    train_wall_seconds=time.time() - epoch_start,
                    plan_audit=plan_audit,
                    trigger="optimizer_steps",
                )
                if action in {"advanced", "complete"}:
                    transition_action = action
                    break
                save_latest(epoch, next_batch_after)
            if args.max_optimizer_steps and optimizer_step >= args.max_optimizer_steps:
                print(
                    "Reached --max-optimizer-steps safety cap; pausing without "
                    "marking convergence"
                )
                save_latest(epoch, next_batch_after)
                return

        if validation_every_steps > 0:
            # A stage transition intentionally abandons the remainder of the old
            # ratio plan and starts a fresh deterministic plan for the new stage.
            # Otherwise a fully consumed plan advances the epoch seed normally;
            # validation remains tied strictly to global optimizer-step multiples.
            next_batch_index = 0
            epoch += 1
            if transition_action is None:
                curriculum.epochs_in_stage += 1
            save_latest(epoch, 0)
            continue

        # Original epoch-end validation path (V12 and legacy V12.1 runs).
        # An epoch is committed only after its complete deterministic plan.
        next_batch_index = 0
        completed_epoch = epoch + 1
        metrics, validation_preview = evaluate(
            student,
            validation_loader,
            device,
            str(getattr(args, "validation_amp_dtype", args.amp_dtype)),
            args.validation_preview_samples,
            detection_mode=getattr(args, "detection_mode", "greedy"),
            phrase_score_threshold=getattr(args, "phrase_score_threshold", None),
            collect_phrase_score_audit=bool(
                getattr(args, "phrase_score_audit", False)
            ),
        )
        phrase_score_audit = metrics.pop("phrase_score_audit", None)
        observed_stage = curriculum.stage_index
        action = curriculum_observe(
            curriculum,
            metrics["macro_ctc_loss"],
            len(ratios),
            args.stage_min_epochs,
            args.stage_plateau_patience,
            args.stage_min_relative_improvement,
            args.stage_min_absolute_improvement,
        )
        print(
            f"[validation] epoch={completed_epoch} stage={observed_stage} "
            f"macro_ctc={metrics['macro_ctc_loss']:.6f} "
            f"pos_ctc={metrics['positive_ctc_loss']:.6f} "
            f"neg_ctc={metrics['negative_ctc_loss']:.6f} "
            f"recall={100 * metrics['recall']:.2f}% "
            f"precision={100 * metrics['precision']:.2f}% "
            f"F1={100 * metrics['f1_score']:.2f}% "
            f"accuracy={100 * metrics['accuracy']:.2f}% "
            f"action={action} bad={curriculum.bad_validations}"
        )
        record = {
            "kind": "validation",
            "unix_time": time.time(),
            "epoch": completed_epoch,
            "optimizer_step": optimizer_step,
            "stage_index_before_observation": observed_stage,
            "stage_index_after_observation": curriculum.stage_index,
            "negative_positive_ratio": ratio,
            "curriculum_action": action,
            "curriculum_state": asdict(curriculum),
            "train_ctc_loss": epoch_loss_sum / max(epoch_samples, 1),
            "epoch_wall_seconds": time.time() - epoch_start,
            "batch_plan_audit": plan_audit,
            **metrics,
        }
        if action in {"advanced", "complete"} and restore_stage_best:
            record["stage_restore_checkpoint"] = str(
                run_directory / f"stage_{observed_stage}_best.pt"
            )
        artifacts.append(record, validation_preview)
        if isinstance(phrase_score_audit, dict):
            phrase_audit_record = {
                "kind": "validation_phrase_score_audit",
                "unix_time": time.time(),
                "epoch": completed_epoch,
                "optimizer_step": optimizer_step,
                "stage_index": observed_stage,
                "active_detection_metrics": {
                    name: float(metrics[name])
                    for name in ("recall", "precision", "f1_score", "accuracy")
                },
                **phrase_score_audit,
            }
            append_jsonl(
                run_directory / "phrase_score_validation_audit.jsonl",
                phrase_audit_record,
            )
            write_phrase_score_audit_plot(run_directory, phrase_audit_record)
            phrase_best = phrase_score_audit["best"]
            print(
                "[phrase-audit] "
                f"best_threshold={float(phrase_best['threshold']):.6f} "
                f"recall={100 * float(phrase_best['recall']):.2f}% "
                f"precision={100 * float(phrase_best['precision']):.2f}% "
                f"F1={100 * float(phrase_best['f1_score']):.2f}% "
                "(validation diagnostic only)"
            )

        metric_specs = (
            ("macro_ctc_loss", "best_macro_ctc.pt", True),
            ("ctc_loss", "best_ctc.pt", True),
            ("recall", "best_positive_recall.pt", False),
            ("f1_score", "best_f1.pt", False),
        )
        metric_improvements: dict[str, bool] = {}
        for metric_name, filename, minimize in metric_specs:
            value = float(metrics[metric_name])
            previous = float(best_metrics[metric_name])
            improved = value < previous if minimize else value > previous
            metric_improvements[metric_name] = improved
            if improved:
                best_metrics[metric_name] = value
                save_metric_best(
                    run_directory / filename,
                    metric_name,
                    value,
                    completed_epoch,
                    metrics,
                    stage_index=observed_stage,
                )

        stage_f1_key = f"stage_{observed_stage}_f1_score"
        stage_f1_value = float(metrics["f1_score"])
        if stage_f1_value > float(best_metrics[stage_f1_key]):
            best_metrics[stage_f1_key] = stage_f1_value
            save_metric_best(
                run_directory / f"stage_{observed_stage}_best_f1.pt",
                "f1_score",
                stage_f1_value,
                completed_epoch,
                metrics,
                stage_index=observed_stage,
            )

        if metric_improvements["macro_ctc_loss"]:
            best_validation_metric = float(metrics["macro_ctc_loss"])

        if action == "improved":
            stage_path = run_directory / f"stage_{observed_stage}_best.pt"
            stage_payload = build_checkpoint(
                student,
                optimizer,
                scheduler,
                scaler,
                signature,
                curriculum,
                optimizer_step,
                completed_epoch,
                0,
                best_validation_metric,
                best_metrics,
                initialization_audit,
            )
            stage_payload["validation_metrics"] = metrics
            stage_payload["stage_index"] = observed_stage
            atomic_torch_save(stage_payload, stage_path)
            del stage_payload
            print("Saved stage best:", stage_path)
        if metric_improvements["macro_ctc_loss"]:
            best_payload = build_checkpoint(
                student,
                optimizer,
                scheduler,
                scaler,
                signature,
                curriculum,
                optimizer_step,
                completed_epoch,
                0,
                best_validation_metric,
                best_metrics,
                initialization_audit,
            )
            best_payload["validation_metrics"] = metrics
            atomic_torch_save(best_payload, best_path)
            del best_payload
            print("Saved global best:", best_path)

        if action in {"advanced", "complete"} and restore_stage_best:
            stage_path = run_directory / f"stage_{observed_stage}_best.pt"
            if not stage_path.is_file():
                raise FileNotFoundError(
                    f"Cannot hand off stage {observed_stage}: missing {stage_path}"
                )
            stage_best = torch.load(stage_path, map_location="cpu", weights_only=False)
            student.load_state_dict(stage_best["student_state_dict"], strict=True)
            optimizer.load_state_dict(stage_best["optimizer_state_dict"])
            scheduler.load_state_dict(stage_best["scheduler_state_dict"])
            scaler.load_state_dict(stage_best.get("scaler_state_dict", {}))
            restored_step = int(stage_best["optimizer_step"])
            restored_metric = float(stage_best["validation_metrics"]["macro_ctc_loss"])
            del stage_best
            active_learning_rate = apply_current_stage_learning_rate()
            append_jsonl(
                run_directory / "stage_transition_history.jsonl",
                {
                    "unix_time": time.time(),
                    "action": action,
                    "completed_epoch": completed_epoch,
                    "current_optimizer_step": optimizer_step,
                    "source_stage_index": observed_stage,
                    "source_checkpoint": str(stage_path),
                    "source_optimizer_step": restored_step,
                    "source_macro_ctc_loss": restored_metric,
                    "next_stage_index": curriculum.stage_index,
                    "next_negative_positive_ratio": int(
                        ratios[curriculum.stage_index]
                    ),
                    "stage_lr_decay": stage_lr_decay,
                    "active_learning_rate": active_learning_rate,
                },
            )
            print(
                f"[stage-restore] source_stage={observed_stage} "
                f"source_step={restored_step:,} macro_ctc={restored_metric:.6f} "
                f"next_stage={curriculum.stage_index} lr={active_learning_rate:.3e}"
            )

        epoch = completed_epoch
        save_latest(epoch, 0)

    default_selection = f"stage_{len(ratios) - 1}_best.pt"
    final_selection_name = str(
        getattr(args, "final_selection_checkpoint", default_selection)
    )
    selected_checkpoint_path = run_directory / final_selection_name
    if not selected_checkpoint_path.is_file():
        raise FileNotFoundError(
            "Training converged without the configured final-selection checkpoint: "
            f"{selected_checkpoint_path}"
        )
    selected = torch.load(
        selected_checkpoint_path, map_location="cpu", weights_only=False
    )
    student.load_state_dict(selected["student_state_dict"], strict=True)
    selected_optimizer_step = int(selected.get("optimizer_step", optimizer_step))
    selected_completed_epochs = int(selected.get("completed_epochs", epoch))
    development_metrics = dict(selected.get("validation_metrics", {}))
    test_metrics = None
    if development_only:
        print("=" * 88)
        print("DEVELOPMENT-ONLY RUN: FINAL TEST INTENTIONALLY NOT EXECUTED")
        print("Selected development checkpoint:", selected_checkpoint_path)
        print("=" * 88)
    else:
        if test_loader is None:
            raise RuntimeError("Final test loader is unavailable")
        print("=" * 88)
        print("FINAL HELD-OUT TEST (NOT USED FOR CHECKPOINT SELECTION)")
        print("=" * 88)
        test_metrics, test_preview = evaluate(
            student,
            test_loader,
            device,
            str(getattr(args, "test_amp_dtype", args.amp_dtype)),
            args.validation_preview_samples,
            include_full_detection_metrics=True,
            detection_mode=getattr(args, "detection_mode", "greedy"),
            phrase_score_threshold=getattr(args, "phrase_score_threshold", None),
            collect_phrase_score_audit=False,
        )
        test_record = {
            "kind": "test",
            "unix_time": time.time(),
            "optimizer_step": optimizer_step,
            "completed_epochs": epoch,
            "selected_checkpoint": str(selected_checkpoint_path),
            **test_metrics,
        }
        artifacts.record_test(test_record, test_preview)
        print(
            f"[test] recall={100 * test_metrics['recall']:.2f}% "
            f"precision={100 * test_metrics['precision']:.2f}% "
            f"F1={100 * test_metrics['f1_score']:.2f}% "
            f"accuracy={100 * test_metrics['accuracy']:.2f}% "
            f"FRR={100 * test_metrics['false_reject_rate']:.2f}% "
            f"FAR={100 * test_metrics['false_accept_rate']:.2f}% "
            f"FA/h={test_metrics['false_accepts_per_hour']:.4f} "
            f"TP/TN/FP/FN={test_metrics['true_positives']}/"
            f"{test_metrics['true_negatives']}/{test_metrics['false_positives']}/"
            f"{test_metrics['false_negatives']}"
        )
    final_payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "trainer_variant": signature.get(
            "trainer_variant", "v12_reduced_streaming_pure_ctc"
        ),
        "optimizer_step": selected_optimizer_step,
        "completed_epochs": selected_completed_epochs,
        "training_completed_optimizer_step": optimizer_step,
        "training_completed_epochs": epoch,
        "selected_checkpoint": str(selected_checkpoint_path),
        "signature": signature,
        "vocabulary": list(REDUCED_VOCABULARY),
        "ctc_blank_id": CTC_BLANK_ID,
        "wake_sequence": list(WAKE_SEQUENCE),
        "initialization_audit": initialization_audit,
        "best_metrics": dict(best_metrics),
        "development_only": development_only,
        "development_metrics": development_metrics,
        "test_metrics": None if test_metrics is None else dict(test_metrics),
        "student_state_dict": student.state_dict(),
    }
    atomic_torch_save(final_payload, final_path)
    artifacts.finish()
    print("Training converged. Exported:", final_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ratios = validate_args(args)
    install_runtime_imports()
    args.project_root = args.project_root.expanduser().resolve()
    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be one safe path component")
    run_directory = args.project_root / "checkpoints" / args.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    tokenizer_path = locate_tokenizer(args)
    tokenizer, old_piece_ids, old_to_reduced = build_reduced_tokenizer(tokenizer_path)
    print("Original tokenizer :", tokenizer_path)
    print("Old piece IDs      :", dict(zip(REDUCED_PIECES, old_piece_ids)))
    print("Reduced IDs        :", dict(zip(REDUCED_PIECES, range(len(REDUCED_PIECES)))))
    print("Blank ID           :", CTC_BLANK_ID)

    if args.wakeword_data_root is None and args.wakeword_manifest is None:
        raise ValueError("Pass --wakeword-manifest or --wakeword-data-root")
    if args.wakeword_data_root is not None and args.wakeword_manifest is not None:
        raise ValueError("Pass only one of --wakeword-manifest and --wakeword-data-root")
    extensions = {
        value.strip().lower()
        if value.strip().startswith(".")
        else "." + value.strip().lower()
        for value in args.audio_extensions.split(",")
        if value.strip()
    }
    if args.wakeword_manifest is not None:
        new_samples, speaker_splits = load_prepared_wakeword_samples(
            args.wakeword_manifest,
            tokenizer,
            old_to_reduced,
            args,
        )
    else:
        new_samples, speaker_splits = load_new_speaker_samples(
            args.wakeword_data_root,
            extensions,
            tokenizer,
            old_to_reduced,
            args,
        )
    data_root = locate_data_root(args)
    infore2_samples = (
        []
        if data_root is None
        else load_infore2_samples(data_root, tokenizer, old_to_reduced, args)
    )
    samples = tuple(infore2_samples + new_samples)
    audit_samples(
        samples,
        speaker_splits,
        require_test=not bool(getattr(args, "development_only", False)),
    )
    manifest_path = run_directory / "unified_manifest.jsonl"
    manifest_sha1 = write_unified_manifest(
        samples,
        manifest_path,
        protect_existing=(run_directory / "latest.pt").is_file(),
    )
    atomic_write_text(
        run_directory / "speaker_splits.json",
        json.dumps(speaker_splits, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print("Unified manifest  :", manifest_path)
    print("Manifest SHA1     :", manifest_sha1)

    print("=" * 88)
    trainer_label = (
        "V12.2"
        if bool(getattr(args, "development_only", False))
        else "V12.1" if hasattr(args, "stage_lr_decay") else "V12"
    )
    print(f"{trainer_label} CURRICULUM BATCH AUDIT (EPOCH 1)")
    print("=" * 88)
    for stage_index, ratio in enumerate(ratios):
        plan = build_epoch_batch_plan(
            samples,
            ratio,
            args.max_utterances_per_batch,
            args.max_padded_audio_seconds,
            args.new_data_negative_share,
            args.hard_negative_share,
            args.seed,
            0,
            shard_aware_batching=bool(
                getattr(args, "shard_aware_batching", False)
            ),
            shard_row_group_size=int(getattr(args, "shard_row_group_size", 64)),
            shard_locality_block_size=int(
                getattr(args, "shard_locality_block_size", 64)
            ),
        )
        audit = batch_plan_audit(samples, plan)
        if bool(getattr(args, "shard_aware_batching", False)):
            audit.update(
                batch_storage_audit(
                    samples,
                    plan,
                    int(getattr(args, "shard_row_group_size", 64)),
                )
            )
        print(
            f"stage={stage_index} target={ratio}:1 batches={int(audit['batches']):,} "
            f"actual={audit['negative_positive_ratio']:.3f}:1 "
            f"positive={int(audit['positive']):,} negative={int(audit['negative']):,} "
            f"hard-neg={100 * audit['hard_negative_share']:.1f}% "
            f"new-neg={100 * audit['new_negative_share']:.1f}%"
            + (
                f" storage-groups/batch={audit['storage_groups_per_batch']:.2f}"
                if "storage_groups_per_batch" in audit
                else ""
            )
        )
    if args.audit_only:
        print("Audit-only complete; GPU model was not constructed")
        return 0

    signature = training_signature(
        args, ratios, manifest_sha1, tokenizer_path, old_piece_ids
    )
    train_v12(
        args,
        ratios,
        samples,
        signature,
        old_piece_ids,
        run_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
