#!/usr/bin/env python3
"""Channel augmentation: room reverberation and band limiting.

Measured on 2026-09-21 (`ctctknk-docs/18` section 10), with the best V14
checkpoint and the deployment detector:

| channel                | recall | false accepts/hour |
|------------------------|-------:|-------------------:|
| clean                  | 0.9480 |               0.00 |
| telephone 300-3400 Hz  | **0.2224** |           0.00 |

Re-calibrating the threshold only recovers 0.436, so a narrowband channel
destroys discrimination, not just calibration.  The room numbers of that
section were later retracted (one synthetic response shared by every sample;
trends read from 3-5 events), which is why the band transform now carries the
larger weight (``ctctknk-docs/20`` section 4).

**Both transforms are applied to positives and negatives alike.**  Reverberating
only the negatives would teach the model "reverberant means not the wake word";
band-limiting only the positives would teach it "narrowband means wake word",
and it would fire at everything over a phone.  A channel is a property of the
room and the microphone, never of what was said.  With precomputed shards this
becomes a numerical constraint -- the augmented share of negative draws must
equal that of positive draws -- checked by ``parakeet_v14_2_train``.

Two ways to use this module
---------------------------
* **Precomputed (default since V14.2 disk shards).**
  ``prepare_v14_2_channel_shards.py`` writes augmented copies next to the
  originals, using the plan helpers below.  The runtime probabilities stay 0.
* **Live (the original V14.2 path, kept for comparison).**  ``install(...)``
  patches the dataset so a hash of the uid picks a fixed transform per sample.
  A live transform *replaces* the clean sample for the whole run; a
  precomputed copy is *added* beside it.

Never both: ``parakeet_v14_2_train`` refuses a run that has precomputed copies
and a live probability above zero.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import parakeet_v12_train as v12

SAMPLE_RATE = 16_000
DEFAULT_RIR_ROOT = Path("/home/ml4u/datasets/rirs")

# Telephony and small speakers; the low edge matters as much as the high one.
BANDS = ((300.0, 3400.0), (200.0, 3800.0), (0.0, 4000.0), (100.0, 5000.0))

# Chosen in ctctknk-docs/20: band carries the measured recall collapse, room
# the weaker (and partly retracted) false-accept evidence.
ROOM_PROBABILITY = 0.15
BAND_PROBABILITY = 0.25
KINDS = ("room", "band", "room+band")
# Every precomputed copy carries this in its uid; the trainer counts copies by it.
CHANNEL_UID_TAG = "#ch"
RIR_SECONDS = 1.5

_RIRS: list[np.ndarray] = []
_CONFIG = {"room_probability": 0.0, "band_probability": 0.0, "salt": 0}
_ORIGINAL_GETITEM = None


# ------------------------------------------------------------------ loading

def is_impulse_response(path: Path) -> bool:
    """RIRS_NOISES mixes room responses with plain noise recordings.

    Only simulated responses and the ``*_rir_*`` files of the real set are
    impulse responses; convolving speech with a noise file would be nonsense.
    """
    text = str(path).lower()
    if "simulated_rirs" in text:
        return True
    return "real_rirs" in text and "_rir_" in path.name.lower()


def usable_rir_paths(root: Path) -> list[Path]:
    paths = sorted(p for p in Path(root).rglob("*.wav") if is_impulse_response(p))
    if not paths:
        raise FileNotFoundError(f"no RIR wav files under {root}")
    return paths


def read_rir(path: Path) -> np.ndarray | None:
    """One 16 kHz response, direct path first, unit L1 norm; None if unusable."""
    import soundfile as sf

    try:
        wave, rate = sf.read(str(path), dtype="float64")
    except Exception:
        return None
    if wave.ndim > 1:
        wave = wave[:, 0]
    if rate != SAMPLE_RATE or wave.size < 32:
        return None
    peak = np.abs(wave).max()
    if not np.isfinite(peak) or peak <= 0:
        return None
    # Trim the leading delay so the direct path stays aligned: a raw RIR would
    # otherwise shift the utterance by its own propagation delay, which is not
    # what a microphone does to the speech it records.
    start = int(np.argmax(np.abs(wave) > 0.05 * peak))
    wave = wave[start:start + int(RIR_SECONDS * SAMPLE_RATE)]
    return (wave / np.abs(wave).sum()).astype(np.float32)


def load_rirs(root: Path, limit: int = 2000, seed: int = 20260921) -> list[np.ndarray]:
    """Load impulse responses for the live path (bounded by ``limit`` for RAM)."""
    paths = usable_rir_paths(root)
    rng = np.random.default_rng(seed)
    if len(paths) > limit:
        paths = [paths[i] for i in sorted(rng.choice(len(paths), limit, replace=False))]
    responses = [rir for rir in (read_rir(p) for p in paths) if rir is not None]
    if not responses:
        raise RuntimeError(f"{root}: found wav files but none usable as 16 kHz RIRs")
    return responses


# --------------------------------------------------------------- transforms

def apply_room(wave: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve and restore the original loudness (a room changes colour, not gain).

    FFT convolution: a direct ``np.convolve`` of a 6 s clip with a 1.5 s
    response costs ~2e9 multiply-adds, which made offline generation of the
    V14.2 shards a ten-hour job.
    """
    from scipy.signal import fftconvolve

    before = float(np.sqrt(np.mean(np.square(wave))))
    wet = fftconvolve(wave, rir)[:len(wave)]
    after = float(np.sqrt(np.mean(np.square(wet))))
    if after <= 1e-9 or before <= 1e-9:
        return wave
    return wet * (before / after)


def apply_band(wave: np.ndarray, low: float, high: float) -> np.ndarray:
    spectrum = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(len(wave), 1.0 / SAMPLE_RATE)
    mask = np.ones(len(freqs))
    if low > 0:
        mask[freqs < low] = 0.0
    if high < SAMPLE_RATE / 2:
        mask[freqs > high] = 0.0
    return np.fft.irfft(spectrum * mask, n=len(wave))


def limit_peak(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio).max() if audio.size else 0.0
    if peak > 1.0:                      # keep the waveform inside [-1, 1]
        audio = audio / peak
    return audio.astype(np.float32)


def apply_kind(wave: np.ndarray, kind: str, rir: np.ndarray | None,
               band: tuple[float, float] | None) -> np.ndarray:
    """Apply one planned channel: ``room``, ``band`` or ``room+band`` (room first)."""
    if kind not in KINDS:
        raise ValueError(f"unknown channel kind {kind!r}")
    audio = np.asarray(wave, dtype=np.float64)
    if "room" in kind:
        if rir is None:
            raise ValueError(f"{kind} needs an impulse response")
        audio = apply_room(audio, rir)
    if "band" in kind:
        if band is None:
            raise ValueError(f"{kind} needs a band")
        audio = apply_band(audio, *band)
    return limit_peak(audio)


def draw(uid: str, salt: int) -> np.random.Generator:
    digest = hashlib.blake2b(f"{salt}:{uid}".encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def transform(wave: np.ndarray, uid: str) -> np.ndarray:
    """Live path: apply the channel this sample was assigned, if any."""
    if not _RIRS and _CONFIG["band_probability"] <= 0:
        return wave
    rng = draw(uid, _CONFIG["salt"])
    audio = np.asarray(wave, dtype=np.float64)
    if _RIRS and rng.random() < _CONFIG["room_probability"]:
        audio = apply_room(audio, _RIRS[int(rng.integers(len(_RIRS)))])
    if rng.random() < _CONFIG["band_probability"]:
        low, high = BANDS[int(rng.integers(len(BANDS)))]
        audio = apply_band(audio, low, high)
    return limit_peak(audio)


# --------------------------------------------------------------- the plan
#
# The V12.2 sampler fills a fixed number of slots per source and rotates
# speaker -> family -> variant (``family_balanced_sequence``).  So the share
# of draws that land on an augmented copy is decided by how copies are laid
# out, not by any probability at training time:
#
# * a family of ``n`` variants plus ``k`` copies in the SAME family gives
#   k / (n + k)  -- used for WakewordV12, whose families hold ~20 variants;
# * a one-row family (every ASR corpus, MUSAN, InfoRE2) must get its copy as
#   a SEPARATE family: joined to the original, the rotation would alternate
#   clean/augmented and the share would halve.  Adding ``q = p/(1-p)`` copies
#   per row on average then gives exactly ``p``.

def augmented_fraction(room: float = ROOM_PROBABILITY, band: float = BAND_PROBABILITY) -> float:
    """Share of draws that pass through any channel (transforms independent)."""
    return 1.0 - (1.0 - room) * (1.0 - band)


def kind_weights(room: float = ROOM_PROBABILITY, band: float = BAND_PROBABILITY) -> dict[str, float]:
    """Mix of channel kinds among augmented draws (sums to 1)."""
    raw = {"room": room * (1.0 - band), "band": (1.0 - room) * band, "room+band": room * band}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("room and band probabilities are both zero")
    return {kind: value / total for kind, value in raw.items()}


def copies_per_original(room: float = ROOM_PROBABILITY, band: float = BAND_PROBABILITY) -> float:
    p = augmented_fraction(room, band)
    if not 0.0 < p < 1.0:
        raise ValueError(f"augmented fraction must be in (0, 1), got {p}")
    return p / (1.0 - p)


def copies_for_family(variants: int, room: float = ROOM_PROBABILITY,
                      band: float = BAND_PROBABILITY) -> int:
    """Copies added to a multi-variant family so it lands near the target share."""
    return int(round(variants * copies_per_original(room, band)))


def split_kinds(count: int, weights: dict[str, float]) -> list[str]:
    """Largest-remainder split of ``count`` copies into kinds, in KINDS order."""
    raw = {kind: weights[kind] * count for kind in KINDS}
    counts = {kind: int(np.floor(value)) for kind, value in raw.items()}
    leftover = count - sum(counts.values())
    order = sorted(KINDS, key=lambda kind: (-(raw[kind] - counts[kind]), KINDS.index(kind)))
    for kind in order[:leftover]:
        counts[kind] += 1
    return [kind for kind in KINDS for _ in range(counts[kind])]


def hash_fraction(*parts: object) -> float:
    """Deterministic value in [0, 1) from the given parts."""
    text = "\x1f".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2 ** 64


def pick_index(count: int, *parts: object) -> int:
    return min(int(hash_fraction(*parts) * count), count - 1)


def copies_for_row(uid: str, salt: int, room: float = ROOM_PROBABILITY,
                   band: float = BAND_PROBABILITY) -> int:
    """Copies for a one-row family: floor(q) always, one more with prob. frac(q)."""
    q = copies_per_original(room, band)
    whole = int(np.floor(q))
    return whole + int(hash_fraction(salt, "copies", uid) < q - whole)


def pick_kind(uid: str, salt: int, weights: dict[str, float]) -> str:
    value = hash_fraction(salt, "kind", uid)
    cumulative = 0.0
    for kind in KINDS:
        cumulative += weights[kind]
        if value < cumulative:
            return kind
    return KINDS[-1]


def copy_uid(origin_uid: str, index: int, kind: str) -> str:
    return f"{origin_uid}{CHANNEL_UID_TAG}{index}:{kind}"


def is_channel_copy(uid: str) -> bool:
    return CHANNEL_UID_TAG in str(uid)


# ------------------------------------------------------------------ install

def install(rir_root: Path | str = DEFAULT_RIR_ROOT, room_probability: float = 0.0,
            band_probability: float = 0.0, salt: int = 0, rir_limit: int = 2000) -> dict:
    """Live path: patch the dataset so training samples pass through a channel.

    Both probabilities default to 0 -- the channel now lives in precomputed
    shards, and augmenting live on top of them would do it twice.
    """
    global _ORIGINAL_GETITEM, _RIRS

    if not 0.0 <= room_probability <= 1.0 or not 0.0 <= band_probability <= 1.0:
        raise ValueError("probabilities must be within [0, 1]")
    _CONFIG["room_probability"] = float(room_probability)
    _CONFIG["band_probability"] = float(band_probability)
    _CONFIG["salt"] = int(salt)
    _RIRS = load_rirs(Path(rir_root), limit=rir_limit) if room_probability > 0 else []

    if _ORIGINAL_GETITEM is None:
        _ORIGINAL_GETITEM = v12.UnifiedAudioDataset.__getitem__

    def patched(self, index):
        item = _ORIGINAL_GETITEM(self, index)
        sample = item["sample"]
        if getattr(sample, "split", None) != "train":
            return item                     # validation and test stay clean
        audio = item["audio"]
        as_numpy = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        changed = transform(as_numpy, getattr(sample, "uid", str(index)))
        if changed is as_numpy:
            return item
        if hasattr(audio, "numpy"):
            import torch

            item["audio"] = torch.from_numpy(np.ascontiguousarray(changed)).to(audio.dtype)
        else:
            item["audio"] = changed
        return item

    v12.UnifiedAudioDataset.__getitem__ = patched
    return {"rirs": len(_RIRS), "room_probability": _CONFIG["room_probability"],
            "band_probability": _CONFIG["band_probability"], "salt": _CONFIG["salt"]}


def uninstall() -> None:
    """Restore the dataset and switch the live path off (probabilities back to 0)."""
    global _RIRS
    if _ORIGINAL_GETITEM is not None:
        v12.UnifiedAudioDataset.__getitem__ = _ORIGINAL_GETITEM
    _CONFIG.update(room_probability=0.0, band_probability=0.0, salt=0)
    _RIRS = []


def live_active() -> bool:
    return _CONFIG["room_probability"] > 0 or _CONFIG["band_probability"] > 0


def signature(precomputed: dict | None = None) -> dict:
    """What to record in the checkpoint so a run can be reproduced."""
    return {"channel_augmentation": {
        "mode": "precomputed_shards" if precomputed else
                ("live" if live_active() else "off"),
        "precomputed": precomputed,
        "live": {
            "rir_count": len(_RIRS),
            "room_probability": _CONFIG["room_probability"],
            "band_probability": _CONFIG["band_probability"],
            "salt": _CONFIG["salt"],
        },
        "bands_hz": [list(band) for band in BANDS],
        "applies_to": "train split only, positives and negatives alike",
    }}
