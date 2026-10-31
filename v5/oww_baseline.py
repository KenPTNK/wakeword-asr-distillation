#!/usr/bin/env python3
"""openWakeWord baseline on the v5 Phase-2 data, scored with the v5 in-context protocol.

openWakeWord (dscripka/openWakeWord 0.6.0): frozen melspectrogram + Google speech-embedding ONNX
models (release v0.5.1) give a 96-d feature every 80 ms; a DNN (layer_dim 32, one block) classifies
16 consecutive features and is trained with the library's own ``Model.auto_train`` (3 sequences,
negative weight 1 -> 1500, target 0.2 FA/h, checkpoint averaging).  Kept to the library recipe:
  * positives: every train positive clip of the v5 manifest (phrase alone or with a command,
    channel copies included), windowed so the 2 s window ends where "năng khiếu ơi" ends
    (+ 0-0.2 s jitter, as ``create_fixed_size_clip``); command clips get the phrase end from
    forced alignment with the teacher (transcript known); 75 % get background audio at
    -10..15 dB SNR (``augment_clips`` AddBackgroundNoise), background = train negative audio;
  * adversarial negatives: the train near-miss clips, windowed and mixed the same way, plus two
    random windows each;
  * general negatives: up to three random 2 s windows of train negative clips drawn with the
    Phase-2 source shares (near-miss source excluded);
  * batches of 1024 general + 50 adversarial + 50 positive windows (library default config);
  * validation: selection-speaker clips (X_val) and the 4.87 h selection audio as
    false-positive data, with ``val_set_hrs`` = 4.87 instead of the library's hard-coded 11.3.
Scoring: the embedded test streams of dump_scores.py (salts 1/2), best score inside the clip window
+ 0.64 s; hk15 and sealed-set events with a 1 s refractory period, sealed masks as v5_longfa.py.
Writes logs/v5/scores/oww.npz and logs/v5/longfa/oww.npz in the format of v5_tables.py.

    v5/python_v5.sh -u v5/oww_baseline.py align | features | train | score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OWW_PKGS = "/media/ml4u/KINGSTON/ctctknk/oww_pkgs"
sys.path.insert(0, OWW_PKGS)
MODELS = Path("/media/ml4u/KINGSTON/ctctknk/oww_models")
WORK = Path("/media/ml4u/KINGSTON/ctctknk/oww_v5")
REF = Path("/home/ml4u/datasets/v5_ref/unified_manifest.jsonl")
POS = Path("logs/v5/positives_v5.jsonl")
HK15 = Path("multi_phone_threshold_analysis/hk15_24h/audio_16k_mono.wav")
SELECTION_AUDIO = [Path(f"multi_phone_threshold_analysis/{d}/audio_16k_mono.wav")
                   for d in ("2_ngay_1_dem_s3_ep60", "have_a_sip_256", "have_a_sip_52")]
RATE = 16000
WINDOW = 2 * RATE                       # 2 s of audio -> exactly 16 embeddings
HOP = 1280                              # one embedding every 80 ms
FIRST_END = 12400                       # sample at which embedding 0 ends (76 mel frames)
PHRASE_IDS = (435, 20, 558, 645)        # ▁năng ▁kh iếu ▁ơi in the teacher vocabulary
TEACHER_BLANK = 1024
SHARES = {"InfoRE2": 0.30, "VLSP": 0.15, "LSVSC": 0.12, "ViMD": 0.10, "MUSAN": 0.08}
REFRACTORY = 1.0
TAIL = 0.64
SEED = 20260928


# ----------------------------------------------------------------------------- data
def load_samples(split: str):
    import parakeet_v12_train as v12
    fields = set(v12.Sample.__dataclass_fields__)
    out = []
    for line in REF.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("split") != split:
            continue
        record["reduced_token_ids"] = tuple(record["reduced_token_ids"])
        out.append(v12.Sample(**{k: v for k, v in record.items() if k in fields}))
    return out


def read_audio(samples):
    """int16 waveforms of ``samples`` (read in storage order, returned in input order)."""
    import parakeet_v12_train as v12
    dataset = v12.UnifiedAudioDataset(list(samples))
    order = sorted(range(len(samples)), key=lambda i: (samples[i].audio_path, samples[i].audio_row or 0))
    waves = [None] * len(samples)
    for start in range(0, len(order), 512):
        chunk = order[start:start + 512]
        for i, item in zip(chunk, dataset.__getitems__(chunk)):
            audio = np.asarray(item["audio"], dtype=np.float32).reshape(-1)
            waves[i] = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    return waves


def speaker(sample) -> str:
    return str(sample.speaker_id).zfill(3)


# ----------------------------------------------------------------------------- align
def align(_args) -> int:
    """Phrase end (s) of every train / selection positive: teacher CTC forced alignment."""
    import sentencepiece as spm
    import torch
    import torchaudio
    import evaluate_baselines as base
    import incontext_protocol as ic
    import parakeet_v12_train as v12
    v12.install_runtime_imports()
    samples = [s for s in load_samples("train") if s.is_positive]
    samples += [s for s in load_samples("validation")
                if s.is_positive and speaker(s) in ic.SELECTION_SPEAKERS]
    print(f"align: {len(samples):,} positive clips", flush=True)
    sp = spm.SentencePieceProcessor(model_file="checkpoints/parakeet_teacher_tokenizer.model")
    model = base.TeacherForward().model
    raw_end, next_start, duration = {}, {}, {}
    for start in range(0, len(samples), 2048):
        group = samples[start:start + 2048]
        waves = read_audio(group)
        for b in range(0, len(group), 32):
            batch, wv = group[b:b + 32], waves[b:b + 32]
            lengths = [len(w) for w in wv]
            block = np.zeros((len(wv), max(lengths)), np.float32)
            for r, w in enumerate(wv):
                block[r, :len(w)] = w / 32767.0
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                audio = torch.from_numpy(block).cuda()
                feats, flen = model.preprocessor(input_signal=audio, length=torch.tensor(lengths).cuda())
                enc, elen = model.encoder(audio_signal=feats, length=flen)
                logits = model.decoder.decoder_layers(enc).transpose(1, 2).float()
            logp = torch.log_softmax(logits, -1).cpu()
            for r, sample in enumerate(batch):
                ids = sp.encode(v12.normalize_text(sample.transcript))
                k = next((i for i in range(len(ids) - 3) if tuple(ids[i:i + 4]) == PHRASE_IDS), None)
                frames = int(elen[r])
                duration[sample.uid] = lengths[r] / RATE
                if k is None or frames < len(ids):
                    continue
                labels, scores = torchaudio.functional.forced_align(
                    logp[r:r + 1, :frames], torch.tensor([ids], dtype=torch.int32), blank=TEACHER_BLANK)
                spans = torchaudio.functional.merge_tokens(labels[0], scores[0].exp(), blank=TEACHER_BLANK)
                raw_end[sample.uid] = spans[k + 3].end * 0.08
                next_start[sample.uid] = spans[k + 4].start * 0.08 if k + 4 < len(spans) else None
        print(f"aligned {min(start + 2048, len(samples)):,}", flush=True)
    alone = [s.uid for s in samples
             if v12.normalize_text(s.transcript) == "năng khiếu ơi" and s.uid in raw_end]
    delta = float(np.median([duration[u] - raw_end[u] for u in alone]))
    ends = {}
    for s in samples:
        if v12.normalize_text(s.transcript) == "năng khiếu ơi":
            ends[s.uid] = duration[s.uid]
        elif s.uid in raw_end:
            end = raw_end[s.uid] + delta
            if next_start[s.uid] is not None:
                end = min(end, max(raw_end[s.uid], next_start[s.uid]))
            ends[s.uid] = min(end, duration[s.uid])
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "align.json").write_text(json.dumps({"delta": delta, "ends": ends}), encoding="utf-8")
    print(f"align: delta {delta:.3f} s from {len(alone):,} phrase-only clips; "
          f"{len(ends):,}/{len(samples):,} ends ({len(samples) - len(ends)} unaligned, dropped)", flush=True)
    return 0


# ----------------------------------------------------------------------------- features
_FEAT = None


def _init_worker():
    global _FEAT
    from openwakeword.utils import AudioFeatures
    _FEAT = AudioFeatures(melspec_model_path=str(MODELS / "melspectrogram.onnx"),
                          embedding_model_path=str(MODELS / "embedding_model.onnx"),
                          inference_framework="onnx", ncpu=1)


def _embed_windows(block: np.ndarray) -> np.ndarray:
    return _FEAT.embed_clips(block, batch_size=128, ncpu=1)[:, -16:, :].astype(np.float16)


def end_window(wave: np.ndarray, end: int, rng, background) -> np.ndarray:
    """2 s window ending at ``end`` (+0-0.2 s jitter), left-padded, background at p=0.75."""
    end = min(len(wave), end) + int(rng.integers(0, int(0.2 * RATE) + 1))
    clip = wave[max(0, end - WINDOW):end].astype(np.float32)
    out = np.zeros(WINDOW, np.float32)
    out[WINDOW - len(clip):] = clip
    if background and rng.random() < 0.75:
        bg = background[int(rng.integers(len(background)))].astype(np.float32)
        snr = rng.uniform(-10.0, 15.0)
        p_fg = float(np.mean(clip ** 2)) + 1e-8
        p_bg = float(np.mean(bg ** 2)) + 1e-8
        out += bg * np.sqrt(p_fg / p_bg / 10 ** (snr / 10))
    return np.clip(out, -32768, 32767).astype(np.int16)


def random_windows(wave: np.ndarray, count: int, rng) -> list:
    if len(wave) <= WINDOW:
        out = np.zeros(WINDOW, np.int16)
        out[WINDOW - len(wave):] = wave
        return [out]
    starts = rng.integers(0, len(wave) - WINDOW + 1, count)
    return [wave[s:s + WINDOW] for s in starts]


def embed_all(pool, windows: list) -> np.ndarray:
    blocks = [np.stack(windows[i:i + 1024]) for i in range(0, len(windows), 1024)]
    return np.concatenate(list(pool.map(_embed_windows, blocks))) if blocks else np.zeros((0, 16, 96), np.float16)


def embed_chunked(pool, samples, make_windows, chunk: int = 8000) -> np.ndarray:
    """Read ``chunk`` clips at a time, window them, keep only the embeddings (RAM-bounded)."""
    out = [np.zeros((0, 16, 96), np.float16)]
    for start in range(0, len(samples), chunk):
        part = samples[start:start + chunk]
        out.append(embed_all(pool, make_windows(part, read_audio(part))))
        print(f"  embedded {min(start + chunk, len(samples)):,}/{len(samples):,} clips", flush=True)
    return np.concatenate(out)


def features(args) -> int:
    import soundfile as sf
    import incontext_protocol as ic
    import parakeet_v12_train as v12
    v12.install_runtime_imports()
    rng = np.random.default_rng(SEED)
    ends = json.loads((WORK / "align.json").read_text(encoding="utf-8"))["ends"]
    train, val = load_samples("train"), load_samples("validation")
    selection = [s for s in val if speaker(s) in ic.SELECTION_SPEAKERS]
    pool = ProcessPoolExecutor(args.workers, initializer=_init_worker)
    # general negatives with the Phase-2 source shares
    by_source: dict = {}
    for s in train:
        if not s.is_positive and s.source_dataset in SHARES:
            by_source.setdefault(s.source_dataset, []).append(s)
    total = sum(SHARES.values())
    chosen = []
    for src, share in SHARES.items():
        pool_src = by_source.get(src, [])
        n = min(len(pool_src), int(args.negative_clips * share / total))
        chosen += [pool_src[i] for i in rng.choice(len(pool_src), n, replace=False)]
        print(f"negatives {src}: {n:,} of {len(pool_src):,} clips", flush=True)
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    background = [w for wv in read_audio(chosen[:6000]) for w in random_windows(wv, 1, rng)]

    def negative_windows(_samples, waves):
        return [w for wv in waves for w in random_windows(wv, min(3, max(1, len(wv) // WINDOW)), rng)]

    def phrase_windows(samples, waves):
        return [end_window(w, int(ends[s.uid] * RATE), rng, background) for s, w in zip(samples, waves)]

    def near_miss_windows(_samples, waves):
        out = []
        for w in waves:
            out.append(end_window(w, len(w), rng, background))
            out += random_windows(w, 2, rng)
        return out

    def save(name, samples, make_windows):
        emb = embed_chunked(pool, samples, make_windows)
        np.save(WORK / f"{name}.npy", emb)
        print(f"{name}: {len(emb):,} windows from {len(samples):,} clips", flush=True)

    save("neg", chosen, negative_windows)
    save("pos", [s for s in train if s.is_positive and s.uid in ends], phrase_windows)
    save("adv", [s for s in train if not s.is_positive and s.source_dataset == "WakewordV12"], near_miss_windows)
    save("val_pos", [s for s in selection if s.is_positive and s.uid in ends], phrase_windows)
    save("val_neg", [s for s in selection if not s.is_positive and s.source_dataset == "WakewordV12"],
         near_miss_windows)
    fp = []
    for path in SELECTION_AUDIO:
        wave, _ = sf.read(str(path), dtype="int16")
        emb = np.concatenate(list(pool.map(_stream_embeddings, pieces(wave))))
        fp.append(np.lib.stride_tricks.sliding_window_view(emb, (16, 96))[:, 0])
    np.save(WORK / "fp_val.npy", np.concatenate(fp).astype(np.float16))
    print(f"fp validation windows {sum(len(f) for f in fp):,}", flush=True)
    pool.shutdown()
    return 0


# ----------------------------------------------------------------------------- streaming
PIECE = 60 * RATE
OVERLAP = 3 * RATE


def pieces(wave: np.ndarray) -> list:
    """Overlapping pieces; each keeps only outputs that end after its overlap."""
    out = []
    for start in range(0, len(wave), PIECE):
        lo = max(0, start - OVERLAP)
        out.append((lo, start, wave[lo:start + PIECE]))
    return out


def _stream_embeddings(piece) -> np.ndarray:
    lo, start, wave = piece
    emb = _FEAT._get_embeddings(wave)
    ends = lo + FIRST_END + HOP * np.arange(len(emb))
    keep = ends > start + FIRST_END if start > 0 else np.ones(len(emb), bool)
    return emb[keep].astype(np.float16)


_HEAD = None
HEAD_NAME = "oww"
HEAD_CONFIG = dict(n_classes=1, input_shape=(16, 96), model_type="dnn", layer_dim=32, n_blocks=1,
                   seconds_per_example=HOP * 16 / RATE)


def oww_model_class():
    """openwakeword.train.Model; its TTS / augmentation imports are not needed here."""
    stub = types.ModuleType("openwakeword.data")
    stub.generate_adversarial_texts = stub.augment_clips = stub.mmap_batch_generator = None
    sys.modules.setdefault("openwakeword.data", stub)
    from openwakeword.train import Model
    return Model


def _init_scorer(head_path: str):
    global _HEAD
    import torch
    _init_worker()
    torch.set_num_threads(1)
    model = oww_model_class()(**HEAD_CONFIG)
    model.model.load_state_dict(torch.load(head_path, map_location="cpu")["state_dict"])
    _HEAD = model.model.eval()


def _score_embeddings(emb: np.ndarray) -> np.ndarray:
    import torch
    if len(emb) < 16:
        return np.zeros(0, np.float32)
    win = np.lib.stride_tricks.sliding_window_view(emb, (16, 96))[:, 0].astype(np.float32)
    with torch.no_grad():
        return _HEAD(torch.from_numpy(np.ascontiguousarray(win))).numpy().reshape(-1)


def _score_stream(item) -> float:
    wave, (a, b) = item
    scores = _score_embeddings(_FEAT._get_embeddings(wave))
    ends = FIRST_END + HOP * np.arange(15, 15 + len(scores))
    inside = (ends >= a) & (ends <= b + TAIL * RATE)
    return float(scores[inside].max()) if inside.any() else 0.0


def _score_piece(piece) -> tuple:
    lo, start, wave = piece
    scores = _score_embeddings(_FEAT._get_embeddings(wave))
    ends = lo + FIRST_END + HOP * np.arange(15, 15 + len(scores))
    keep = ends > start + FIRST_END + 15 * HOP if start > 0 else np.ones(len(scores), bool)
    return scores[keep], ends[keep]


def long_scores(pool, wave) -> tuple:
    parts = list(pool.map(_score_piece, pieces(wave)))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]) / RATE


def events(scores, times, thresholds, cap) -> np.ndarray:
    """Detections with a 1 s refractory period; counting stops above ``cap`` (FA/h far above 2)."""
    out = np.zeros(len(thresholds), np.int64)
    for j, t in enumerate(thresholds):
        above = times[scores >= t]
        n, nxt = 0, 0
        while nxt < len(above) and n <= cap:
            n += 1
            nxt = int(np.searchsorted(above, above[nxt] + REFRACTORY, side="left"))
        out[j] = n
    return out


def score(args) -> int:
    import soundfile as sf
    import evaluate_baselines as base
    import evaluate_wake_suite as suite
    import incontext_protocol as ic
    import parakeet_v12_train as v12
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import v5_longfa as lf
    v12.install_runtime_imports()
    pool = ProcessPoolExecutor(args.workers, initializer=_init_scorer, initargs=(str(WORK / f"{HEAD_NAME}_head.pt"),))
    pos_waves, _ = suite.ev.load_dev_positives(POS)
    labels = suite.positive_paths(POS)
    keep = [i for i, lab in enumerate(labels) if ic.role_of(lab) == "test"]
    families = base.positive_augmentations(POS)
    neg_waves, neg_labels = suite.load_hard_negatives(Path("/home/ml4u/datasets/v5_ref") / "unified_manifest.jsonl")
    nkeep = [i for i, lab in enumerate(neg_labels) if ic.role_of(lab) == "test"]
    long_wave, rate = sf.read(str(HK15), dtype="int16")
    hours = len(long_wave) / rate / 3600
    ps, pw = base.test_streams([pos_waves[i] for i in keep], [long_wave], 1)
    ns, nw = base.test_streams([neg_waves[i] for i in nkeep], [long_wave], 2)
    pos_best = np.array(list(pool.map(_score_stream, zip(ps, pw), chunksize=16)))
    neg_best = np.array(list(pool.map(_score_stream, zip(ns, nw), chunksize=16)))
    print(f"test streams: {len(pos_best):,} positives, {len(neg_best):,} near misses", flush=True)
    thresholds = np.unique(np.concatenate([np.linspace(0, 1, 257), 1 - np.logspace(-7, -1, 193),
                                           np.quantile(pos_best, np.linspace(0, 1, 257))]))
    s, t = long_scores(pool, long_wave)
    hk_events = events(s, t, thresholds, cap=int(100 * hours))
    out = Path(f"logs/v5/scores/{HEAD_NAME}.npz")
    np.savez_compressed(out, pos_best=pos_best, pos_label=np.array([labels[i] for i in keep]),
                        pos_family=np.array([families[i] for i in keep]), neg_best=neg_best,
                        neg_label=np.array([neg_labels[i] for i in nkeep]), thresholds=thresholds,
                        events=hk_events, hours=hours)
    print(f"wrote {out}; hk15 events at 0.5: {hk_events[np.searchsorted(thresholds, 0.5)]}", flush=True)
    ev, hrs, names, cats = [], [], [], []
    for path, category, mask in lf.sealed_files():
        wave, _ = sf.read(str(path), dtype="int16")
        s, t = long_scores(pool, wave)
        inside = np.zeros(len(t), bool)
        for a, b in mask:
            inside |= (t >= a) & (t <= b)
        _, h = lf.segments(int(np.ceil(len(wave) / rate * 12.5)), mask, len(wave) / rate)
        ev.append(events(s[~inside], t[~inside], thresholds, cap=int(100 * max(h, 0.01))))
        hrs.append(h)
        names.append(path.name)
        cats.append(category)
        print(f"{path.name} {category}: {h:.2f} h", flush=True)
    np.savez_compressed(f"logs/v5/longfa/{HEAD_NAME}.npz", thresholds=thresholds, events=np.array(ev),
                        hours=np.array(hrs), files=np.array(names), categories=np.array(cats))
    pool.shutdown()
    return 0


# ----------------------------------------------------------------------------- train
def train(args) -> int:
    import logging
    import soundfile as sf
    import torch
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    class Model(oww_model_class()):
        """Library auto_train with the false-positive validation hours as a parameter."""

        def auto_train(self, X_train, X_val, false_positive_val_data, steps=50000,
                       max_negative_weight=1000, target_fp_per_hour=0.2, val_set_hrs=11.3):
            lr = 0.0001
            weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
            val_steps = np.linspace(steps - int(steps * 0.25), steps, 20).astype(np.int64)
            self.train_model(X=X_train, X_val=X_val, false_positive_val_data=false_positive_val_data,
                             max_steps=steps, negative_weight_schedule=weights, val_steps=val_steps,
                             warmup_steps=steps // 5, hold_steps=steps // 3, lr=lr, val_set_hrs=val_set_hrs)
            steps = steps / 10
            for _ in range(2):          # sequences 2 and 3: lr / 10 each, same (shorter) length
                lr = lr / 10
                if self.best_val_fp > target_fp_per_hour:
                    max_negative_weight = max_negative_weight * 2
                weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
                val_steps = np.linspace(1, steps, 20).astype(np.int16)
                self.train_model(X=X_train, X_val=X_val, false_positive_val_data=false_positive_val_data,
                                 max_steps=steps, negative_weight_schedule=weights, val_steps=val_steps,
                                 warmup_steps=steps // 5, hold_steps=steps // 3, lr=lr, val_set_hrs=val_set_hrs)
            acc = np.percentile(self.history["val_accuracy"], 90)
            rec = np.percentile(self.history["val_recall"], 90)
            fp = np.percentile(self.history["val_fp_per_hr"], 10)
            models = [m for m, sc in zip(self.best_models, self.best_model_scores)
                      if sc["val_accuracy"] >= acc and sc["val_recall"] >= rec and sc["val_fp_per_hr"] <= fp]
            print(f"merging {len(models)} of {len(self.best_models)} checkpoints", flush=True)
            return self.average_models(models=models) if models else self.model

    pos, adv, neg = (np.load(WORK / f"{n}.npy") for n in ("pos", "adv", "neg"))
    vpos, vneg, fpv = (np.load(WORK / f"{n}.npy") for n in ("val_pos", "val_neg", "fp_val"))
    print(f"train windows: {len(pos):,} pos, {len(adv):,} adversarial, {len(neg):,} general; "
          f"val {len(vpos):,}/{len(vneg):,}; fp val {len(fpv):,}", flush=True)
    rng = np.random.default_rng(SEED)

    def batches():
        y = torch.cat([torch.zeros(1024 + 50), torch.ones(50)])
        while True:
            x = np.concatenate([neg[rng.integers(0, len(neg), 1024)], adv[rng.integers(0, len(adv), 50)],
                                pos[rng.integers(0, len(pos), 50)]]).astype(np.float32)
            yield torch.from_numpy(x), y

    x_val = [(torch.from_numpy(np.concatenate([vpos, vneg]).astype(np.float32)),
              torch.cat([torch.ones(len(vpos)), torch.zeros(len(vneg))]).to(torch.int64))]
    fp_val = [(torch.from_numpy(fpv[i:i + 20000].astype(np.float32)), torch.zeros(len(fpv[i:i + 20000])))
              for i in range(0, len(fpv), 20000)]
    hours = sum(sf.info(str(p)).duration for p in SELECTION_AUDIO) / 3600
    torch.manual_seed(SEED)
    model = Model(**HEAD_CONFIG)
    start = time.time()
    best = model.auto_train(batches(), x_val, fp_val, steps=50000, max_negative_weight=1500,
                            target_fp_per_hour=0.2, val_set_hrs=hours)
    torch.save({"state_dict": best.to("cpu").state_dict(), "config": HEAD_CONFIG}, WORK / f"{HEAD_NAME}_head.pt")
    params = sum(p.numel() for p in best.parameters())
    print(f"trained in {time.time() - start:.0f} s; head parameters {params:,}; fp val hours {hours:.2f}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("align", "features", "train", "score"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--negative-clips", type=int, default=180000)
    parser.add_argument("--head", choices=("dnn", "rnn"), default="dnn",
                        help="library model_type; rnn writes oww_rnn.* (workers fork after this)")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.head != "dnn":
        global HEAD_NAME
        HEAD_CONFIG["model_type"] = args.head
        HEAD_NAME = f"oww_{args.head}"
    return {"align": align, "features": features, "train": train, "score": score}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
