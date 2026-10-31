#!/usr/bin/env python3
"""BC-ResNet baseline on the v5 Phase-2 data, scored with the v5 in-context protocol.

BC-ResNet-8 (Kim et al., Interspeech 2021; architecture of the authors' release: 5x5 conv head,
stages of 2/2/4/4 broadcasted residual blocks with sub-spectral norm, temporal dilation and
frequency stride, depthwise classifier) classifies 1.5 s windows of 40 log-mel bands (30 ms / 10 ms)
as wake phrase or not, trained from scratch with the paper's recipe: SGD (momentum 0.9, weight
decay 1e-3), learning rate 0.1 with warm-up and cosine decay, SpecAugment, background noise.
Same data as oww_baseline.py: positive windows end where the phrase ends (align.json), 75 % mixed
with background at -10..15 dB; near-miss windows (end + two random); general negatives from the
Phase-2 source shares.  Batches of 64 positive, 64 near-miss and 128 general windows.  Checkpoints
every 10k steps are chosen with the frozen selection rule (in-context FRR at 1 FA/h with near-miss
accepts <= 10 % on the selection speakers, embedded in and counted on the 4.87 h selection audio).
Writes logs/v5/scores/bcresnet.npz and logs/v5/longfa/bcresnet.npz (format of v5_tables.py).

    v5/python_v5.sh -u v5/bcresnet_baseline.py features | train | score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oww_baseline as ob  # noqa: E402

WORK = Path("/media/ml4u/KINGSTON/ctctknk/bcresnet_v5")
RATE = 16000
WINDOW = int(1.5 * RATE)
N_FFT, HOP_MEL, N_MELS = 480, 160, 40
FRAMES = (WINDOW - N_FFT) // HOP_MEL + 1          # 148
STEP = 8                                         # windows every 80 ms, as the other detectors
SCALE = 8
STEPS = 40000
SEED = 20260929


# ----------------------------------------------------------------------------- model
def build_model(scale: int = SCALE, classes: int = 2):
    import torch
    from torch import nn

    class SubSpectralNorm(nn.Module):
        def __init__(self, channels, sub=5):
            super().__init__()
            self.sub, self.bn = sub, nn.BatchNorm2d(channels * sub)

        def forward(self, x):
            n, c, f, t = x.shape
            return self.bn(x.reshape(n, c * self.sub, f // self.sub, t)).reshape(n, c, f, t)

    class Block(nn.Module):
        """Transition block (channel change / frequency stride) or normal block."""

        def __init__(self, cin, cout, stride, dilation):
            super().__init__()
            self.transition = cin != cout or stride != 1
            if self.transition:
                self.pre = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
            self.freq = nn.Conv2d(cout, cout, (3, 1), stride=(stride, 1), padding=(1, 0), groups=cout, bias=False)
            self.ssn = SubSpectralNorm(cout)
            self.temp = nn.Sequential(
                nn.Conv2d(cout, cout, (1, 3), padding=(0, dilation), dilation=(1, dilation), groups=cout, bias=False),
                nn.BatchNorm2d(cout), nn.SiLU(), nn.Conv2d(cout, cout, 1, bias=False), nn.Dropout2d(0.1))

        def forward(self, x):
            identity = x
            if self.transition:
                x = self.pre(x)
            aux = self.ssn(self.freq(x))
            out = self.temp(aux.mean(2, keepdim=True)) + aux
            if not self.transition:
                out = out + identity
            return torch.relu(out)

    class BCResNet(nn.Module):
        def __init__(self):
            super().__init__()
            base = 8 * scale
            c = [base * 2, base, int(base * 1.5), base * 2, int(base * 2.5), base * 4]
            self.head = nn.Sequential(nn.Conv2d(1, c[0], 5, stride=(2, 1), padding=2, bias=False),
                                      nn.BatchNorm2d(c[0]), nn.ReLU())
            blocks = []
            for idx, n in enumerate((2, 2, 4, 4)):
                dilation = 2 ** (idx - 1) if idx >= 1 else 1
                stride = 2 if idx in (1, 2) else 1
                blocks.append(Block(c[idx], c[idx + 1], stride, dilation))
                blocks += [Block(c[idx + 1], c[idx + 1], 1, dilation) for _ in range(n - 1)]
            self.blocks = nn.Sequential(*blocks)
            self.classifier = nn.Sequential(
                nn.Conv2d(c[-2], c[-2], (5, 5), padding=(0, 2), groups=c[-2], bias=False),
                nn.Conv2d(c[-2], c[-1], 1, bias=False), nn.BatchNorm2d(c[-1]), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(c[-1], classes, 1))

        def forward(self, x):                       # x: [B, 40, T] log-mel
            return self.classifier(self.blocks(self.head(x.unsqueeze(1)))).flatten(1)

    return BCResNet()


def melspec(device="cpu"):
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(sample_rate=RATE, n_fft=N_FFT, win_length=N_FFT,
                                                hop_length=HOP_MEL, n_mels=N_MELS, center=False).to(device)


# ----------------------------------------------------------------------------- features
_W = {}


def _init_reader(background):
    import torch
    import parakeet_v12_train as v12
    torch.set_num_threads(1)
    v12.install_runtime_imports()
    _W["mel"], _W["bg"] = melspec(), background


def _log_mel(windows) -> np.ndarray:
    import torch
    if not windows:
        return np.zeros((0, N_MELS, FRAMES), np.float16)
    x = torch.from_numpy(np.stack(windows).astype(np.float32) / 32767.0)
    with torch.no_grad():
        return torch.log(_W["mel"](x) + 1e-6).numpy().astype(np.float16)


def _chunk(job) -> np.ndarray:
    kind, samples, ends, seed = job
    rng = np.random.default_rng(seed)
    waves = ob.read_audio(samples)
    bg = _W["bg"]
    out = []
    for s, w in zip(samples, waves):
        if kind == "pos":
            out.append(ob.end_window(w, int(ends[s.uid] * RATE), rng, bg)[-WINDOW:])
        elif kind == "adv":
            out.append(ob.end_window(w, len(w), rng, bg)[-WINDOW:])
            out += [x[-WINDOW:] for x in ob.random_windows(w, 2, rng)]
        else:
            out += [x[-WINDOW:] for x in ob.random_windows(w, min(3, max(1, len(w) // (2 * RATE))), rng)]
    return _log_mel(out)


def features(args) -> int:
    import parakeet_v12_train as v12
    v12.install_runtime_imports()
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ends = json.loads((ob.WORK / "align.json").read_text(encoding="utf-8"))["ends"]
    train = ob.load_samples("train")
    by_source: dict = {}
    for s in train:
        if not s.is_positive and s.source_dataset in ob.SHARES:
            by_source.setdefault(s.source_dataset, []).append(s)
    total, chosen = sum(ob.SHARES.values()), []
    for src, share in ob.SHARES.items():
        pool_src = by_source.get(src, [])
        n = min(len(pool_src), int(args.negative_clips * share / total))
        chosen += [pool_src[i] for i in rng.choice(len(pool_src), n, replace=False)]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    background = [w for wv in ob.read_audio(chosen[:2000]) for w in ob.random_windows(wv, 1, rng)]
    sets = {
        "neg": ("neg", chosen),
        "pos": ("pos", [s for s in train if s.is_positive and s.uid in ends]),
        "adv": ("adv", [s for s in train if not s.is_positive and s.source_dataset == "WakewordV12"]),
    }
    with ProcessPoolExecutor(args.workers, initializer=_init_reader, initargs=(background,)) as pool:
        for name, (kind, samples) in sets.items():
            jobs = [(kind, samples[i:i + 1000], ends, SEED + i) for i in range(0, len(samples), 1000)]
            mel = np.concatenate(list(pool.map(_chunk, jobs)))
            np.save(WORK / f"{name}.npy", mel)
            print(f"{name}: {len(mel):,} windows from {len(samples):,} clips", flush=True)
    return 0


# ----------------------------------------------------------------------------- scoring
class Scorer:
    """Window scores (P(wake)) every 80 ms over a waveform, on the GPU."""

    def __init__(self, model, mean, std):
        import torch
        self.torch, self.model = torch, model.eval().cuda()
        self.mel, self.mean, self.std = melspec("cuda"), mean, std

    def __call__(self, wave: np.ndarray) -> tuple:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(wave.astype(np.float32) / 32767.0).cuda()
            logmel = (torch.log(self.mel(x) + 1e-6) - self.mean) / self.std        # [40, T]
            if logmel.shape[1] < FRAMES:
                return np.zeros(0, np.float32), np.zeros(0, np.int64)
            win = logmel.unfold(1, FRAMES, STEP).permute(1, 0, 2)                 # [W, 40, 148]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = torch.cat([torch.softmax(self.model(win[i:i + 1024]).float(), -1)[:, 1]
                                    for i in range(0, len(win), 1024)]).cpu().numpy()
        ends = (np.arange(len(scores)) * STEP + FRAMES - 1) * HOP_MEL + N_FFT
        return scores, ends

    def window_best(self, streams, windows) -> np.ndarray:
        best = np.zeros(len(streams))
        for k, (wave, (a, b)) in enumerate(zip(streams, windows)):
            s, e = self(wave)
            inside = (e >= a) & (e <= b + ob.TAIL * RATE)
            best[k] = float(s[inside].max()) if inside.any() else 0.0
        return best

    def long(self, wave: np.ndarray) -> tuple:
        scores, times = [], []
        piece, overlap = 600 * RATE, 2 * RATE
        for start in range(0, len(wave), piece):
            lo = max(0, start - overlap)
            s, e = self(wave[lo:start + piece])
            keep = e + lo > start if start > 0 else np.ones(len(s), bool)
            scores.append(s[keep])
            times.append((e[keep] + lo) / RATE)
        return np.concatenate(scores), np.concatenate(times)


def protocol_sets(role: str, contexts):
    import evaluate_baselines as base
    import evaluate_wake_suite as suite
    import incontext_protocol as ic
    pos_waves, _ = suite.ev.load_dev_positives(ob.POS)
    labels = suite.positive_paths(ob.POS)
    keep = [i for i, lab in enumerate(labels) if ic.role_of(lab) == role]
    neg_waves, neg_labels = suite.load_hard_negatives(ob.REF)
    nkeep = [i for i, lab in enumerate(neg_labels) if ic.role_of(lab) == role]
    ps, pw = base.test_streams([pos_waves[i] for i in keep], contexts, 1)
    ns, nw = base.test_streams([neg_waves[i] for i in nkeep], contexts, 2)
    families = base.positive_augmentations(ob.POS)
    return (ps, pw, ns, nw, [labels[i] for i in keep], [families[i] for i in keep],
            [neg_labels[i] for i in nkeep])


def grid() -> np.ndarray:
    return np.unique(np.concatenate([np.linspace(0, 1, 257), 1 - np.logspace(-7, -1, 193)]))


def frr_at(pos, neg, events, hours, thresholds, limit=1.0, cap=0.10):
    ok = [float((pos < t).mean()) for j, t in enumerate(thresholds)
          if events[j] / hours <= limit and (neg >= t).mean() <= cap]
    return min(ok) if ok else None


# ----------------------------------------------------------------------------- train
def train(args) -> int:
    import soundfile as sf
    import torch
    import parakeet_v12_train as v12
    v12.install_runtime_imports()
    pos, adv, neg = (np.load(WORK / f"{n}.npy") for n in ("pos", "adv", "neg"))
    sample = np.concatenate([pos[:5000], neg[:20000]]).astype(np.float32)
    mean, std = float(sample.mean()), float(sample.std())
    print(f"windows: {len(pos):,} pos, {len(adv):,} near-miss, {len(neg):,} general; "
          f"log-mel mean {mean:.3f} std {std:.3f}", flush=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = build_model().cuda()
    params = sum(p.numel() for p in model.parameters())
    print(f"BC-ResNet-{SCALE}: {params:,} parameters", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3)
    warm = 2000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / (STEPS - warm))))
    y = torch.cat([torch.ones(64), torch.zeros(64 + 128)]).long().cuda()
    sel_audio = [sf.read(str(p), dtype="int16")[0] for p in ob.SELECTION_AUDIO]
    sel_hours = sum(len(w) for w in sel_audio) / RATE / 3600
    ps, pw, ns, nw, *_ = protocol_sets("selection", sel_audio)
    print(f"selection: {len(ps):,} positive / {len(ns):,} near-miss streams, {sel_hours:.2f} h", flush=True)
    thresholds = grid()
    band = torch.arange(N_MELS, device="cuda")[None]
    frame = torch.arange(FRAMES, device="cuda")[None]
    choice, start = None, time.time()
    for step in range(1, STEPS + 1):
        x = np.concatenate([pos[rng.integers(0, len(pos), 64)], adv[rng.integers(0, len(adv), 64)],
                            neg[rng.integers(0, len(neg), 128)]]).astype(np.float32)
        x = (torch.from_numpy(x).cuda() - mean) / std
        for _ in range(2):                                             # SpecAugment
            f0 = torch.randint(0, N_MELS - 5, (len(x),), device="cuda")
            fw = torch.randint(0, 6, (len(x),), device="cuda")
            x = x.masked_fill(((band >= f0[:, None]) & (band < (f0 + fw)[:, None]))[:, :, None], 0.0)
            t0 = torch.randint(0, FRAMES - 10, (len(x),), device="cuda")
            tw = torch.randint(0, 11, (len(x),), device="cuda")
            x = x.masked_fill(((frame >= t0[:, None]) & (frame < (t0 + tw)[:, None]))[:, None, :], 0.0)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(x).float(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.4f} "
                  f"{time.time() - start:.0f} s", flush=True)
        if step % 10000 == 0:
            scorer = Scorer(model, mean, std)
            pb, nb = scorer.window_best(ps, pw), scorer.window_best(ns, nw)
            ev = np.zeros(len(thresholds), np.int64)
            for w in sel_audio:
                s, t = scorer.long(w)
                ev += ob.events(s, t, thresholds, cap=int(100 * sel_hours))
            frr = frr_at(pb, nb, ev, sel_hours, thresholds)
            print(f"selection step {step}: FRR at 1 FA/h (NM<=0.1) {frr}", flush=True)
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "step": step,
                        "selection_frr": frr}, WORK / f"ckpt_{step}.pt")
            if frr is not None and (choice is None or frr < choice[1]):
                choice = (step, frr)
    (WORK / "chosen.json").write_text(json.dumps({"step": choice[0] if choice else STEPS,
                                                  "selection_frr": choice[1] if choice else None,
                                                  "parameters": params}), encoding="utf-8")
    print(f"chosen {choice}", flush=True)
    return 0


def score(args) -> int:
    import soundfile as sf
    import torch
    import parakeet_v12_train as v12
    import v5_longfa as lf
    v12.install_runtime_imports()
    chosen = json.loads((WORK / "chosen.json").read_text(encoding="utf-8"))
    ck = torch.load(WORK / f"ckpt_{chosen['step']}.pt", map_location="cpu")
    model = build_model()
    model.load_state_dict(ck["state_dict"])
    scorer = Scorer(model, ck["mean"], ck["std"])
    long_wave, rate = sf.read(str(ob.HK15), dtype="int16")
    hours = len(long_wave) / rate / 3600
    ps, pw, ns, nw, pos_label, pos_family, neg_label = protocol_sets("test", [long_wave])
    pos_best, neg_best = scorer.window_best(ps, pw), scorer.window_best(ns, nw)
    thresholds = np.unique(np.concatenate([grid(), np.quantile(pos_best, np.linspace(0, 1, 257))]))
    s, t = scorer.long(long_wave)
    np.savez_compressed("logs/v5/scores/bcresnet.npz", pos_best=pos_best, pos_label=np.array(pos_label),
                        pos_family=np.array(pos_family), neg_best=neg_best, neg_label=np.array(neg_label),
                        thresholds=thresholds, events=ob.events(s, t, thresholds, cap=int(100 * hours)),
                        hours=hours)
    print(f"test: {len(pos_best):,} positives, {len(neg_best):,} near misses (step {chosen['step']})", flush=True)
    ev, hrs, names, cats = [], [], [], []
    for path, category, mask in lf.sealed_files():
        wave, _ = sf.read(str(path), dtype="int16")
        s, t = scorer.long(wave)
        inside = np.zeros(len(t), bool)
        for a, b in mask:
            inside |= (t >= a) & (t <= b)
        _, h = lf.segments(int(np.ceil(len(wave) / rate * 12.5)), mask, len(wave) / rate)
        ev.append(ob.events(s[~inside], t[~inside], thresholds, cap=int(100 * max(h, 0.01))))
        hrs.append(h)
        names.append(path.name)
        cats.append(category)
    np.savez_compressed("logs/v5/longfa/bcresnet.npz", thresholds=thresholds, events=np.array(ev),
                        hours=np.array(hrs), files=np.array(names), categories=np.array(cats))
    print(f"sealed: {sum(hrs):.2f} h", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("features", "train", "score"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--negative-clips", type=int, default=180000)
    args = parser.parse_args()
    return {"features": features, "train": train, "score": score}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
