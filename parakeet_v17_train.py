#!/usr/bin/env python3
"""V17 = phase 2 redone on the phase-3a encoder (ctctknk-docs/33).

Everything is V14.2 (``parakeet_v14_2_train``: 5-class wake head, pure CTC,
channel copies, long-audio selection under ``streaming_protocol``) except:

1. **Targets from transcript words, not SentencePiece pieces.**  V12 tokenised
   the transcript with Parakeet's SentencePiece model and kept the four exact
   pieces, so a word whose tokenisation happened not to contain one (``không``
   is a single piece) contributed nothing.  V17 maps words:

       word == "năng"                         -> ▁năng
       word starts with "kh"                  -> ▁kh
       word ends with "iếu" (not "yếu" alone) -> iếu
       word == "ơi"                           -> ▁ơi

   A word may give two symbols ("khiếu" -> ▁kh iếu); every other word gives none
   (blank between symbols, as in V12).  "năng khiếu ơi" -> (0, 1, 2, 3) exactly
   as before.  Installed as ``v12.transcript_target_ids`` so every loader
   (wake-word data, InfoRE2, V14 negatives, channel copies) uses it; the policy
   name is written into the training signature.
2. **Attention context ``[64, 7]`` while training** (deployment context; the 3a
   encoder was trained at it).  V12's student uses ``[-1, 7]``.
3. **Initialisation from the 3a encoder** and **the 3a corpora as negatives** are
   launcher settings (``train_v17.sh``: ``V14_INIT_CHECKPOINT``,
   ``V14_NEGATIVE_MANIFEST``, ``--v14-negative-source-shares``), not code.

    bash train_v17.sh   # see there for the environment
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import parakeet_v12_train as v12
import parakeet_v14_2_train as v14_2
import parakeet_v14_train as v14

TARGET_POLICY = "normalized_words_v17"
TARGET_POLICY_WHOLE = "whole_phrase_only_v17"   # ablation: no partial targets
WAKE_WORDS = ("năng", "khiếu", "ơi")
TRAINER_VARIANT = "v17_words_targets_ctx64_3a_encoder_on_v14.2"
TRAIN_LEFT_FRAMES = 64

_ORIGINAL_BUILD = v12.build_student_class
_ORIGINAL_TARGETS = v12.transcript_target_ids


def word_target_ids(text: str) -> tuple[int, ...]:
    """Wake symbols of a transcript, word by word (see the module docstring)."""
    ids: list[int] = []
    for word in v12.normalize_text(text).split():
        if word == "năng":
            ids.append(0)
        if word.startswith("kh"):
            ids.append(1)
        if word.endswith("iếu") and word != "yếu":
            ids.append(2)
        if word == "ơi":
            ids.append(3)
    return tuple(ids)


def whole_target_ids(text: str) -> tuple[int, ...]:
    """Full wake sequence only when the phrase occurs as consecutive words, else empty."""
    words = v12.normalize_text(text).split()
    for i in range(len(words) - len(WAKE_WORDS) + 1):
        if tuple(words[i:i + len(WAKE_WORDS)]) == WAKE_WORDS:
            return (0, 1, 2, 3)
    return ()


def transcript_target_ids(text, tokenizer, old_to_reduced):
    if v12.TARGET_PROJECTION == TARGET_POLICY:
        return word_target_ids(text)
    if v12.TARGET_PROJECTION == TARGET_POLICY_WHOLE:
        return whole_target_ids(text)
    return _ORIGINAL_TARGETS(text, tokenizer, old_to_reduced)


def build_student_class():
    base = _ORIGINAL_BUILD()

    class _Context64Student(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.encoder.att_context_size = [TRAIN_LEFT_FRAMES,
                                             v12.STREAMING_CHUNK_ENCODER_FRAMES - 1]
            self.encoder.setup_streaming_params()

    return _Context64Student


def self_check() -> None:
    cases = {"năng khiếu ơi": (0, 1, 2, 3), "năng khơi ơi": (0, 1, 3),
             "khiếu nại không hiểu ơi": (1, 2, 1, 3), "thiếu nhi yếu": (2,),
             "nắng nâng với": (), "Năng, KHIẾU ơi!": (0, 1, 2, 3)}
    for text, expected in cases.items():
        got = word_target_ids(text)
        if got != expected:
            raise RuntimeError(f"word targets of {text!r}: {got} != {expected}")


def apply_patches() -> None:
    self_check()
    # V17_TARGET_POLICY=pieces keeps V12's SentencePiece targets (ablation, ctctknk-docs/33
    # §5): everything else of V17 stays, so the word rule's effect can be isolated.
    policy = os.environ.get("V17_TARGET_POLICY", "words")
    if policy not in ("words", "pieces", "whole"):
        raise ValueError(f"V17_TARGET_POLICY must be 'words', 'pieces' or 'whole', got {policy!r}")
    if policy == "words":
        v12.TARGET_PROJECTION = TARGET_POLICY
        v12.transcript_target_ids = transcript_target_ids
    if policy == "whole":   # ablation (ICIIT revision): no partial-match targets
        v12.TARGET_PROJECTION = TARGET_POLICY_WHOLE
        v12.transcript_target_ids = transcript_target_ids
    # V14 wraps the student class it captured at import; replace that capture so
    # its frequency-mask subclass is built on the [64, 7] student.
    v14._V12_BUILD_STUDENT = build_student_class
    v14_2.TRAINER_VARIANT = TRAINER_VARIANT
    # Phase 2 v2 (ctctknk-docs/33 §3l): clips inside continuous audio, noise, rooms.
    if os.environ.get("V17_CONTEXT_AUG", "0") == "1":
        import parakeet_v17_context
        parakeet_v17_context.install()
        v14_2.TRAINER_VARIANT = TRAINER_VARIANT + "+context_noise_room_v2"


def main(argv: Sequence[str] | None = None) -> int:
    apply_patches()
    print(f"V17: targets={v12.TARGET_PROJECTION}, train att_context=[{TRAIN_LEFT_FRAMES}, "
          f"{v12.STREAMING_CHUNK_ENCODER_FRAMES - 1}]", flush=True)
    return v14_2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
