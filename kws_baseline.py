#!/usr/bin/env python3
"""A conventional small-footprint KWS baseline on the same data and protocol (§3q).

Causal depthwise-separable CNN (DS-CNN family) on 64 log-mel bands: a strided
Conv2d stem brings 10 ms frames to 80 ms (the student's frame rate), then
causal dilated depthwise-separable Conv1d blocks, and a 5-class CTC head
(▁năng ▁kh iếu ▁ơi + blank) — so the same keyword automaton, the same
in-context protocol and the same frozen selection rule apply.  No distillation,
no ASR pre-training: this is the "train a KWS model directly" arm.

    python kws_baseline.py train --width 64 --name dscnn_w64 ...
    python kws_baseline.py evaluate --name dscnn_w64 ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

import parakeet_v12_train as v12

SAMPLE_RATE = 16000
CLASSES = 5
BLANK = 4
STEM_STRIDE = 8
STEM_FREQ_BINS = 8              # 64 mel bands after three stride-2 convs (padded above by 1)


def build_model(width: int, blocks: int, kernel: int):
    import torch
    import torchaudio
    from torch import nn

    class CausalDSBlock(nn.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            self.pad = (kernel - 1) * dilation
            self.depthwise = nn.Conv1d(channels, channels, kernel, dilation=dilation,
                                       groups=channels)
            self.pointwise = nn.Conv1d(channels, channels, 1)
            self.norm = nn.LayerNorm(channels)     # per frame: GroupNorm pooled over time

        def forward(self, x):
            y = self.pointwise(self.depthwise(nn.functional.pad(x, (self.pad, 0))))
            y = self.norm(y.transpose(1, 2)).transpose(1, 2)
            return x + torch.relu(y)

    class DSCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_fft=400, win_length=400, hop_length=160,
                n_mels=64, center=False)
            stem, channels_in = [], 1
            for _ in range(3):                       # time and frequency / 2 each
                stem += [nn.Conv2d(channels_in, width, 3, stride=2), nn.ReLU()]
                channels_in = width
            self.stem = nn.Sequential(*stem)
            self.project = nn.Linear(width * STEM_FREQ_BINS, width)
            self.blocks = nn.Sequential(*[CausalDSBlock(width, 2 ** (i % 3))
                                          for i in range(blocks)])
            self.head = nn.Conv1d(width, CLASSES, 1)

        def forward(self, audio, lengths):
            features = torch.log(self.mel(audio) + 1e-6)            # [B, 64, T]
            x = features.unsqueeze(1)                              # [B, 1, 64, T]
            for layer in self.stem:
                if isinstance(layer, nn.Conv2d):                   # causal in time
                    x = nn.functional.pad(x, (2, 0, 1, 0))
                x = layer(x)
            batch, channels, freq, frames = x.shape
            x = self.project(x.permute(0, 3, 1, 2).reshape(batch, frames, channels * freq))
            x = self.blocks(x.transpose(1, 2))
            logits = self.head(x).transpose(1, 2)                  # [B, T', 5]
            mel_frames = torch.clamp((lengths - 400) // 160 + 1, min=1)
            out_lengths = torch.clamp(torch.div(mel_frames + STEM_STRIDE - 1, STEM_STRIDE,
                                                rounding_mode="floor"), max=frames)
            return logits, out_lengths

    return DSCNN()


def load_samples(manifest: Path):
    fields = set(v12.Sample.__dataclass_fields__)
    positives, negatives = [], []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split") != "train":
                continue
            record["reduced_token_ids"] = tuple(record["reduced_token_ids"])
            sample = v12.Sample(**{k: v for k, v in record.items() if k in fields})
            (positives if sample.is_positive else negatives).append(sample)
    return positives, negatives


ROW_GROUP = 64                  # rows per Parquet row group in the wake-word / negative shards
GROUPS_PER_SIDE = 4             # storage groups per batch for positives and for negatives


def storage_groups(samples, offset: int):
    """Dataset indices grouped by (file, row group), so a batch decodes few row groups."""
    groups: dict = {}
    for index, sample in enumerate(samples):
        key = (sample.audio_path, (sample.audio_row or 0) // ROW_GROUP)
        groups.setdefault(key, []).append(offset + index)
    return [np.array(v) for v in groups.values()]


class GroupedBatches:
    """``steps`` batches: GROUPS_PER_SIDE positive + negative storage groups, half each."""

    def __init__(self, pos_groups, neg_groups, batch: int, steps: int, seed: int):
        self.pos, self.neg, self.batch, self.steps, self.seed = pos_groups, neg_groups, batch, steps, seed

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        per_group = max(1, self.batch // (2 * GROUPS_PER_SIDE))
        for _ in range(self.steps):
            chosen = []
            for groups in (self.pos, self.neg):
                for g in rng.integers(0, len(groups), GROUPS_PER_SIDE):
                    chosen.extend(rng.choice(groups[g], per_group).tolist())
            yield chosen


def train(args) -> int:
    import torch
    v12.install_runtime_imports()
    if args.context_aug:
        os.environ["V17_CONTEXT_AUG"] = "1"
        import parakeet_v17_context
        parakeet_v17_context.install()
    positives, negatives = load_samples(args.manifest)
    print(f"train: {len(positives):,} positives, {len(negatives):,} negatives", flush=True)
    dataset = v12.UnifiedAudioDataset(positives + negatives)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=GroupedBatches(storage_groups(positives, 0),
                                              storage_groups(negatives, len(positives)),
                                              args.batch, args.steps, args.seed),
        num_workers=args.workers, collate_fn=v12.collate_audio_items, prefetch_factor=4,
        persistent_workers=True)
    device = torch.device("cuda")
    model = build_model(args.width, args.blocks, args.kernel).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"model: width {args.width}, {args.blocks} blocks, {params:,} parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    out_dir = args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "parameters": params},
                                                    default=str, indent=1))
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        lr = args.lr * min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        audio = batch["audio"][:, :int(args.max_seconds * SAMPLE_RATE)].to(device)
        lengths = torch.clamp(batch["audio_lengths"], max=audio.shape[1]).to(device)
        logits, out_lengths = model(audio, lengths)
        loss = ctc(torch.log_softmax(logits.float(), -1).transpose(0, 1),
                   batch["ctc_targets"].to(device), out_lengths,
                   batch["ctc_target_lengths"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"[kws] step={step} loss={loss.item():.4f} lr={lr:.2e} "
                  f"{(time.time() - started) / step:.3f}s/step", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"step": step, "width": args.width, "blocks": args.blocks,
                        "kernel": args.kernel, "state_dict": model.state_dict()},
                       out_dir / f"ckpt_{step:06d}.pt")
    return 0


def load_forward(path: Path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(payload["width"], payload["blocks"], payload["kernel"])
    model.load_state_dict(payload["state_dict"])
    model = model.cuda().eval()

    def forward(block, lengths):
        audio = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).cuda()
        sizes = torch.tensor(list(lengths), device="cuda")
        with torch.inference_mode():
            return model(audio, sizes)

    return forward, payload["step"], sum(p.numel() for p in model.parameters())


def evaluate(args) -> int:
    import evaluate_baselines as base
    import evaluate_wake_suite as suite
    import incontext_protocol as ic
    import reselect_in_context as reselect
    import soundfile as sf
    import streaming_protocol as protocol
    v12.install_runtime_imports()
    out_dir = args.out / args.name
    pos_waves, _ = suite.ev.load_dev_positives(args.positive_manifest)
    labels = suite.positive_paths(args.positive_manifest)
    families = base.positive_augmentations(args.positive_manifest)
    neg_waves, neg_labels = suite.load_hard_negatives(args.manifest)
    pos = reselect.split_by_role(pos_waves, labels)
    neg = reselect.split_by_role(neg_waves, neg_labels)
    selection_audio = reselect.read_waves(reselect.SELECTION_AUDIO)
    hours = sum(len(w) for w in selection_audio) / SAMPLE_RATE / 3600
    rows = {}
    for path in sorted(out_dir.glob("ckpt_*.pt")):
        forward, step, params = load_forward(path)
        measured = ic.measure(forward, selection_audio, pos["selection"], neg["selection"],
                              reselect.SEED)
        finite = measured["positive_best"][np.isfinite(measured["positive_best"])]
        thresholds = np.linspace(float(finite.min()), float(finite.max()), args.thresholds)
        events = np.zeros(len(thresholds), dtype=np.int64)
        for wave in selection_audio:
            probabilities = protocol.stream_probabilities(forward, wave)
            events += protocol.batched_events(probabilities[None], thresholds)[0]
        rows[path.name] = {"step": step, **ic.summarize(measured, thresholds, events / hours)}
        print(f"{path.name}: rule-v2 score {rows[path.name]['ic_selection_score']:.4f}", flush=True)
    chosen = max(rows, key=lambda k: (rows[k]["ic_selection_score"],
                                      -rows[k]["ic_selection_near_miss_fa"]))
    forward, step, params = load_forward(out_dir / chosen)
    long_wave, rate = sf.read(str(args.audio), dtype="int16")
    test_hours = len(long_wave) / rate / 3600
    test_pos = [w for w, lab in zip(pos_waves, labels) if ic.role_of(lab) == "test"]
    test_fam = [f for f, lab in zip(families, labels) if ic.role_of(lab) == "test"]
    test_neg = [w for w, lab in zip(neg_waves, neg_labels) if ic.role_of(lab) == "test"]
    streams = (base.test_streams(test_pos, [long_wave], 1),
               base.test_streams(test_neg, [long_wave], 2))
    result = base.single_detector(args.name, forward, long_wave, test_hours, streams,
                                  args.thresholds)
    result[0]["by_augmentation"] = base.by_family(result[1], test_fam, result[0]["frr_at_fa"])
    report = {"name": args.name, "parameters": params, "chosen": chosen, "selection": rows,
              "test": result[0]}
    (args.out / f"kws_baseline_{args.name}.json").write_text(
        json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("TEST", json.dumps(result[0]["frr_at_fa"], default=float), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        p.add_argument("--manifest", type=Path, required=True,
                       help="unified_manifest.jsonl of a V17 run (word targets)")
        p.add_argument("--out", type=Path, required=True)
        if name == "train":
            p.add_argument("--width", type=int, default=64)
            p.add_argument("--blocks", type=int, default=6)
            p.add_argument("--kernel", type=int, default=9)
            p.add_argument("--steps", type=int, default=20000)
            p.add_argument("--batch", type=int, default=64)
            p.add_argument("--lr", type=float, default=2e-3)
            p.add_argument("--max-seconds", type=float, default=8.0)
            p.add_argument("--save-every", type=int, default=2000)
            p.add_argument("--context-aug", action="store_true")
            p.add_argument("--workers", type=int, default=6)
            p.add_argument("--seed", type=int, default=20260925)
        else:
            p.add_argument("--positive-manifest", type=Path, required=True)
            p.add_argument("--audio", type=Path, required=True)
            p.add_argument("--thresholds", type=int, default=256)
    args = parser.parse_args(argv)
    return train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
