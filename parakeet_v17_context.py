"""V17 phase 2 v2: wake-word clips inside continuous audio (ctctknk-docs/33 §3l).

Phase-2 positives were isolated clips, so the detector could learn "speech after
silence" and missed the wake word inside continuous audio (§3h: in-context FRR
at 1 FA/h 0.16 vs 0.05 isolated).  Reverberation also multiplied FA/h by 4-18
(§3i).  For training samples only, this patch adds, on the fly:

* WakewordV12 items (positives AND near-miss phrases alike, so context is not a
  shortcut for "not the wake word"):
  - with ``V17_NOISE_PROB``: MUSAN noise over the clip at an SNR drawn from
    ``V17_NOISE_SNR_DB`` (active-speech power);
  - with ``V17_CONTEXT_PROB``: ``V17_CONTEXT_SECONDS`` of background audio before
    the clip, ``V17_CONTEXT_LEVEL_DB`` dB under its active speech.  Background
    comes from training negatives whose targets are empty, so the CTC targets
    of the item stay exactly those of the clip.
* every training original that is not already a channel copy: with
  ``V17_ROOM_PROB`` a measured room response (``parakeet_v14_2_channel``).

Validation and test samples are never touched.  ``install()`` patches both
``__getitem__`` and ``__getitems__`` (the DataLoader uses the batched path).

    V17_CONTEXT_AUG=1 bash train_v17.sh ...
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import incontext_protocol as incontext
import parakeet_v12_train as v12
import parakeet_v14_2_channel as channel
import parakeet_v14_train as v14

NOISE_SOURCE = "MUSAN"
SAMPLE_RATE = 16000
MAX_EMPTY_DRAWS = 50
BANK_SIZE = 256                          # decoded pool items kept per worker and pool
REFRESH_DRAWS = 2000                     # draws before a bank is replaced by a fresh one

_CONFIG: dict[str, object] = {}
_ORIGINAL_GETITEM = None
_ORIGINAL_GETITEMS = None
_RIRS: list[np.ndarray] = []
_RNG: dict[int, np.random.Generator] = {}
_BANKS: dict[tuple[int, int], dict] = {}


def _pair(name: str, default: str) -> tuple[float, float]:
    low, high = (float(x) for x in os.environ.get(name, default).split(","))
    if low > high:
        raise ValueError(f"{name}: lower bound {low} > upper bound {high}")
    return low, high


def _probability(name: str, default: str) -> float:
    value = float(os.environ.get(name, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}")
    return value


def read_config() -> dict[str, object]:
    return {"context_prob": _probability("V17_CONTEXT_PROB", "0.5"),
            "context_seconds": _pair("V17_CONTEXT_SECONDS", "2,6"),
            "context_level_db": _pair("V17_CONTEXT_LEVEL_DB", "5,30"),
            "noise_prob": _probability("V17_NOISE_PROB", "0.3"),
            "noise_snr_db": _pair("V17_NOISE_SNR_DB", "0,10"),
            "room_prob": _probability("V17_ROOM_PROB", "0.3"),
            "rir_root": os.environ.get("V17_RIR_ROOT", str(channel.DEFAULT_RIR_ROOT)),
            # "background" = any empty-target negative (speech included); "noise" = MUSAN only
            "context_pool": os.environ.get("V17_CONTEXT_POOL", "background"),
            "salt": int(os.environ.get("V17_CONTEXT_SALT", "0"))}


def _rng() -> np.random.Generator:
    pid = os.getpid()
    if pid not in _RNG:                        # one stream per DataLoader worker
        _RNG[pid] = np.random.default_rng([int(_CONFIG["salt"]), pid])
    return _RNG[pid]


def _to_numpy(audio) -> np.ndarray:
    return audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)


def _like(original, values: np.ndarray):
    if hasattr(original, "numpy"):
        import torch
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    return values.astype(np.float32)


def _pools(dataset) -> tuple[np.ndarray, np.ndarray]:
    """(background, noise) sample indices of this dataset, built once."""
    cached = getattr(dataset, "_v17_context_pools", None)
    if cached is not None:
        return cached
    background, noise = [], []
    for index, sample in enumerate(dataset.samples):
        if (sample.split != "train" or sample.is_positive
                or sample.source_dataset == v14.WAKEWORD_SOURCE
                or channel.is_channel_copy(sample.uid) or tuple(sample.reduced_token_ids)):
            continue
        background.append(index)
        if sample.source_dataset == NOISE_SOURCE:
            noise.append(index)
    pools = (np.array(background, dtype=np.int64), np.array(noise, dtype=np.int64))
    dataset._v17_context_pools = pools
    return pools


def _cached_waves(dataset, pool: np.ndarray, rng) -> list[np.ndarray]:
    """A per-worker bank of decoded pool items, refreshed every ``REFRESH_DRAWS``.

    Reading one pool item at a time decodes a whole Parquet row group for one
    utterance, which cut training from ~2.5 to ~0.4 steps/s; one batched read of
    ``BANK_SIZE`` items is amortised over many draws.
    """
    key = (os.getpid(), id(pool))
    bank = _BANKS.get(key)
    if bank is None or bank["draws"] >= REFRESH_DRAWS:
        chosen = rng.choice(pool, size=min(BANK_SIZE, len(pool)), replace=False)
        items = _ORIGINAL_GETITEMS(dataset, [int(i) for i in np.sort(chosen)])
        waves = [_to_numpy(item["audio"]).astype(np.float32) for item in items]
        bank = {"waves": [w for w in waves if w.size], "draws": 0}
        _BANKS[key] = bank
    bank["draws"] += 1
    return bank["waves"]


def _background(dataset, pool: np.ndarray, need: int, rng) -> np.ndarray:
    """Exactly ``need`` samples drawn from ``pool`` (items joined when too short)."""
    need = max(1, int(need))
    waves = _cached_waves(dataset, pool, rng)
    if not waves:
        raise RuntimeError("context augmentation: background pool yields empty audio")
    pieces, have, empty = [], 0, 0
    while have < need:
        piece = waves[int(rng.integers(len(waves)))].astype(np.float64)
        if piece.size == 0:
            empty += 1
            if empty > MAX_EMPTY_DRAWS:
                raise RuntimeError("context augmentation: background pool yields empty audio")
            continue
        pieces.append(piece)
        have += piece.size
    joined = np.concatenate(pieces)
    start = int(rng.integers(0, joined.size - need + 1))
    return joined[start:start + need]


def _scaled(reference: np.ndarray, other: np.ndarray, below_db: float) -> np.ndarray:
    target = incontext.active_power(reference) / 10 ** (below_db / 10)
    return other * np.sqrt(target / max(float(np.mean(other ** 2)), 1e-12))


def augment(dataset, item: dict) -> dict:
    sample = item["sample"]
    if getattr(sample, "split", None) != "train":
        return item
    rng = _rng()
    audio = _to_numpy(item["audio"]).astype(np.float64)
    changed = False
    background_pool, noise_pool = _pools(dataset)
    if sample.source_dataset == v14.WAKEWORD_SOURCE and audio.size:
        if len(noise_pool) and rng.random() < _CONFIG["noise_prob"]:
            noise = _background(dataset, noise_pool, audio.size, rng)
            audio = audio + _scaled(audio, noise, rng.uniform(*_CONFIG["noise_snr_db"]))
            changed = True
        if _CONFIG["context_pool"] == "noise":
            background_pool = noise_pool
        if len(background_pool) and rng.random() < _CONFIG["context_prob"]:
            seconds = rng.uniform(*_CONFIG["context_seconds"])
            context = _background(dataset, background_pool, int(seconds * SAMPLE_RATE), rng)
            context = _scaled(audio, context, rng.uniform(*_CONFIG["context_level_db"]))
            audio = np.concatenate([context, audio])
            changed = True
    if _RIRS and not channel.is_channel_copy(sample.uid) and rng.random() < _CONFIG["room_prob"]:
        audio = channel.apply_room(audio, _RIRS[int(rng.integers(len(_RIRS)))])
        changed = True
    if not changed:
        return item
    return {**item, "audio": _like(item["audio"], channel.limit_peak(audio))}


def install() -> dict[str, object]:
    global _ORIGINAL_GETITEM, _ORIGINAL_GETITEMS, _RIRS
    _CONFIG.clear()
    _CONFIG.update(read_config())
    if _CONFIG["context_pool"] not in ("background", "noise"):
        raise ValueError(f"V17_CONTEXT_POOL must be 'background' or 'noise', "
                         f"got {_CONFIG['context_pool']!r}")
    _RIRS = (channel.load_rirs(Path(str(_CONFIG["rir_root"])))
             if float(_CONFIG["room_prob"]) > 0 else [])
    if float(_CONFIG["room_prob"]) > 0 and not _RIRS:
        raise RuntimeError(f"V17_ROOM_PROB > 0 but no RIRs under {_CONFIG['rir_root']}")
    if _ORIGINAL_GETITEM is None:
        _ORIGINAL_GETITEM = v12.UnifiedAudioDataset.__getitem__
        _ORIGINAL_GETITEMS = v12.UnifiedAudioDataset.__getitems__

    def patched_getitem(self, index):
        return augment(self, _ORIGINAL_GETITEM(self, index))

    def patched_getitems(self, indices):
        return [augment(self, item) for item in _ORIGINAL_GETITEMS(self, indices)]

    v12.UnifiedAudioDataset.__getitem__ = patched_getitem
    v12.UnifiedAudioDataset.__getitems__ = patched_getitems
    summary = {**_CONFIG, "rirs": len(_RIRS)}
    print(f"V17 context augmentation: {summary}", flush=True)
    return summary
