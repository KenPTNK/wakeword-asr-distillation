#!/usr/bin/env python3
"""V16 = phase 1 redone (ctctknk-docs/26 §3a, dataset plan in 30).

V11 distilled Parakeet into the 5 M student on **InfoRE2 only**, with unlimited
left attention context.  Measured on 2026-09-22: that student transcribes
InfoRE2 word-for-word but reaches WER 0.77 on the other six corpora, so the
merged model of phase 3b would carry an ASR head that only works on one domain.
V16 redoes phase 1 on the full corpus with the deployment context:

* **Attention context ``[64, 7]`` while training** — 64 frames (5.12 s) of left
  context, look-ahead to the end of the 0.64 s chunk: what the device runs
  (``streaming_protocol``).  V11 trained at ``[-1, 7]`` and only ran at ``[64, 7]``.
* **All ten sources** of the fixed split (``splits_3a.jsonl``, role ``train``:
  1,470,048 utterances / 1,775.7 h), drawn per dataset with temperature
  ``--dataset-alpha`` so InfoRE2 and VSV-1100 do not swamp the rest.
* **Teacher targets from the compact top-64 logits on the NVMe** (``top64_stem``),
  which reproduce the KD loss exactly (max |Δp| 3.9e-6).
* Loss = ``--kd-weight`` · KL(top-64 + rest bucket, T) + ``--ctc-weight`` · CTC
  on the normalised transcript (numbers spoken, ``vi_number_norm``).
* SpecAugment as in Parakeet's own config: 2 frequency masks, 10 time masks.

Checkpoints go to ``--run-root`` (NVMe: the project disk has 13 GB left).
Validation every ``--validate-every`` steps: greedy WER on a fixed sample of the
``validation`` role; the best WER is kept as ``best_wer.pt``, and every
validation writes ``latest.pt`` (model + optimizer) so a run resumes exactly.

    python parakeet_v16_phase1_train.py --run-name v16_phase1_trial1 --steps 100000
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import parakeet_v12_train as v12
import vi_number_norm

ASR_CLASSES = 1025
BLANK = ASR_CLASSES - 1
KD_TOPK = 64
MAX_TEACHER_SHORTFALL = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("/home/ml4u/datasets/ctctknk_3a/splits_3a.jsonl"))
    parser.add_argument("--run-root", type=Path,
                        default=Path("/home/ml4u/datasets/ctctknk_3a/checkpoints"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path(
        "/media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk/checkpoints/parakeet_teacher_tokenizer.model"))
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="optional warm start (V11); default is from scratch")
    parser.add_argument("--att-left-frames", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-seconds", type=float, default=200.0,
                        help="padded audio per optimizer step (Parakeet used 200)")
    parser.add_argument("--max-utterances", type=int, default=64)
    parser.add_argument("--rowgroup-take", type=int, default=8,
                        help="utterances taken from one Parquet row group (read locality)")
    parser.add_argument("--dataset-alpha", type=float, default=0.5)
    # 1e-3 held constant diverged in all three runs of 2026-09-23 (gradient norm
    # 3 -> 88 between steps 13k and 16k): peak 5e-4 with cosine decay instead.
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollback-threshold", type=float, default=0.10,
                        help="validation loss this far above the stage best = divergence: "
                             "reload the stage best, scale the LR, stay in the stage")
    parser.add_argument("--rollback-lr-factor", type=float, default=0.5)
    parser.add_argument("--max-rollbacks", type=int, default=2,
                        help="per stage; after that a divergence ends the stage")
    parser.add_argument("--stop-at-step", type=int, default=0,
                        help="stop early without changing the LR schedule (tests)")
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--stages", default="0.9:0.1,0.7:0.3,0.5:0.5,0.3:0.7,0.1:0.9",
                        help="KD:CTC weights per stage; a stage ends when the validation "
                             "loss stops improving (run 2 = '1:0', run 3 = '0:1')")
    parser.add_argument("--stage-min-steps", type=int, default=6_000)
    parser.add_argument("--stage-max-steps", type=int, default=30_000)
    parser.add_argument("--patience", type=int, default=3,
                        help="validations without improvement that end a stage")
    parser.add_argument("--min-delta", type=float, default=0.005,
                        help="relative improvement that counts (0.005 = 0.5 %%)")
    parser.add_argument("--stage-lr-decay", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--freq-masks", type=int, default=2)
    parser.add_argument("--time-masks", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validate-every", type=int, default=2_000)
    parser.add_argument("--validate-utterances", type=int, default=1_500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="fp32",
                        help="fp32 by default: the 5 M model costs little, and bf16 was "
                             "active when the first runs diverged")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-free-gb", type=float, default=3.0)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="offline")
    parser.add_argument("--wandb-project", default="parakeet-vietnamese-distillation")
    parser.add_argument("--wandb-entity", default="khactrungkiennguyen-ai-tech-lab")
    parser.add_argument("--wandb-tags", default="v16,phase1,64x7,full-corpus")
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="stop after N steps (smoke test: no W&B, 64-utterance validation)")
    return parser.parse_args(argv)


# ============================================================
# Data
# ============================================================

def load_manifest(path: Path) -> tuple[dict, list]:
    """Training rows compactly (a dict per row costs GBs across DataLoader workers):
    ``{dataset: {audio_path: [(audio_row, top64_stem, offset, length, norm), ...]}}``
    with the repeated path / stem strings interned.  Validation keeps plain rows."""
    train: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    validation: list = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["role"] == "train":
                train[row["dataset"]][sys.intern(row["audio_path"])].append(
                    (int(row["audio_row"]), sys.intern(row["top64_stem"]),
                     int(row["teacher_offset"]), int(row["teacher_length"]), row["norm"]))
            elif row["role"] == "validation":
                validation.append(row)
    for dataset in train:
        for rows in train[dataset].values():
            rows.sort()
    return {d: dict(files) for d, files in train.items()}, validation


def train_size(train: dict) -> dict:
    return {dataset: sum(len(rows) for rows in files.values())
            for dataset, files in train.items()}


def dataset_probabilities(train: dict, alpha: float) -> dict:
    sizes = train_size(train)
    counts = np.array([sizes[d] for d in sorted(sizes)], dtype=np.float64) ** alpha
    return dict(zip(sorted(sizes), counts / counts.sum()))


class Utterances:
    """Reads one utterance: audio, transcript ids, teacher top-64 rows."""

    def __init__(self, tokenizer_path: Path):
        import sentencepiece as spm

        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.readers: collections.OrderedDict = collections.OrderedDict()
        self.arrays: collections.OrderedDict = collections.OrderedDict()

    def _reader(self, path: str):
        import pyarrow.parquet as pq

        reader = self.readers.pop(path, None)
        if reader is None:
            reader = pq.ParquetFile(path)
            ends, total = [], 0
            for group in range(reader.metadata.num_row_groups):
                total += reader.metadata.row_group(group).num_rows
                ends.append(total)
            reader.ctctknk_ends = tuple(ends)
        self.readers[path] = reader
        while len(self.readers) > 8:
            self.readers.popitem(last=False)
        return reader

    def _array(self, path: str):
        array = self.arrays.pop(path, None)
        if array is None:
            array = np.load(path, mmap_mode="r")
        self.arrays[path] = array
        while len(self.arrays) > 64:
            self.arrays.popitem(last=False)
        return array

    def row_group_of(self, path: str, audio_row: int) -> tuple[int, int, int]:
        """(row group index, first row of that group, rows in it)."""
        reader = self._reader(path)
        ends = np.asarray(reader.ctctknk_ends)
        group = int(np.searchsorted(ends, audio_row, side="right"))
        start = 0 if group == 0 else int(ends[group - 1])
        return group, start, int(ends[group]) - start

    def audio_batch(self, path: str, group: int, wanted: list[int]) -> dict[int, np.ndarray]:
        """Decode several rows of one row group in a single read."""
        reader = self._reader(path)
        values = reader.read_row_group(group, columns=["audio"]).column("audio")
        return {index: np.asarray(v12.decode_audio_object(values[index].as_py()),
                                  dtype=np.float32) for index in wanted}

    def teacher(self, stem: str, start: int, length: int, temperature: float):
        values = np.asarray(self._array(f"{stem}.top64v.npy")[start:start + length],
                            dtype=np.float32)
        indices = np.asarray(self._array(f"{stem}.top64i.npy")[start:start + length],
                             dtype=np.int64)
        lse = np.asarray(self._array(f"{stem}.lseT{temperature:g}.npy")[start:start + length],
                         dtype=np.float32)
        return values, indices, lse


class TrainStream(torch.utils.data.IterableDataset):
    """Endless batches: pick a dataset by temperature, a row group, then rows of it."""

    def __init__(self, train: dict, probabilities: dict, options):
        super().__init__()
        self.train = train
        self.datasets = sorted(train)
        self.probabilities = np.array([probabilities[d] for d in self.datasets])
        self.options = options
        # A batch comes from one Parquet row group, so one decoded group serves
        # ~rowgroup_take utterances.  Files are drawn in proportion to their rows.
        self.file_weights: dict[str, tuple[list[str], np.ndarray]] = {}
        for dataset, files in train.items():
            names = sorted(files)
            counts = np.array([len(files[p]) for p in names], dtype=np.float64)
            self.file_weights[dataset] = (names, counts / counts.sum())

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker = info.id if info is not None else 0
        rng = np.random.default_rng([self.options.seed, worker])
        reader = Utterances(self.options.tokenizer)
        batch, longest = [], 0.0
        while True:
            dataset = self.datasets[int(rng.choice(len(self.datasets), p=self.probabilities))]
            names, weights = self.file_weights[dataset]
            path = names[int(rng.choice(len(names), p=weights))]
            rows = self.train[dataset][path]
            anchor = rows[int(rng.integers(len(rows)))]
            group, first, size = reader.row_group_of(path, anchor[0])
            in_group = {row[0] - first: row for row in rows
                        if first <= row[0] < first + size}
            wanted = [int(o) for o in rng.permutation(size)[:self.options.rowgroup_take]
                      if int(o) in in_group]
            if not wanted:
                continue
            audio = reader.audio_batch(path, group, wanted)
            for offset in wanted:
                item = self.item(reader, dataset, in_group[offset], audio[offset])
                if item is None:
                    continue
                if batch and max(longest, item["seconds"]) * (len(batch) + 1) \
                        > self.options.batch_seconds:
                    yield collate(batch)
                    batch, longest = [], 0.0
                batch.append(item)
                longest = max(longest, item["seconds"])
                if len(batch) >= self.options.max_utterances:
                    yield collate(batch)
                    batch, longest = [], 0.0

    def item(self, reader: Utterances, dataset: str, row: tuple, wave: np.ndarray):
        _, stem, offset, length, norm = row
        values, indices, lse = reader.teacher(stem, offset, length, self.options.temperature)
        if len(values) == 0:
            return None
        return {"audio": wave, "seconds": len(wave) / v12.TARGET_SAMPLE_RATE,
                "targets": reader.tokenizer.encode(norm), "values": values,
                "indices": indices, "lse": lse, "dataset": dataset}


def collate(items: list[dict]) -> dict:
    lengths = [len(i["audio"]) for i in items]
    frames = [len(i["values"]) for i in items]
    audio = torch.zeros(len(items), max(lengths))
    values = torch.zeros(len(items), max(frames), KD_TOPK)
    indices = torch.zeros(len(items), max(frames), KD_TOPK, dtype=torch.long)
    lse = torch.zeros(len(items), max(frames))
    targets, target_lengths = [], []
    for row, item in enumerate(items):
        audio[row, :len(item["audio"])] = torch.from_numpy(item["audio"])
        values[row, :len(item["values"])] = torch.from_numpy(item["values"])
        indices[row, :len(item["indices"])] = torch.from_numpy(item["indices"])
        lse[row, :len(item["lse"])] = torch.from_numpy(item["lse"])
        targets.extend(item["targets"])
        target_lengths.append(len(item["targets"]))
    return {"audio": audio, "audio_lengths": torch.tensor(lengths),
            "teacher_values": values, "teacher_indices": indices, "teacher_lse": lse,
            "teacher_lengths": torch.tensor(frames),
            "targets": torch.tensor(targets, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths),
            "seconds": sum(i["seconds"] for i in items),
            "datasets": [i["dataset"] for i in items]}


# ============================================================
# Model and losses
# ============================================================

def build_student(att_left: int, freq_masks: int, time_masks: int):
    """V12's encoder (chunk-causal, causal convolutions) with Parakeet's 1,025-class head."""
    model = v12.build_student_class()()
    pieces = [f"<{i}>" for i in range(ASR_CLASSES - 1)]
    model.decoder = v12.ConvASRDecoder(feat_in=v12.STUDENT_D_MODEL, num_classes=len(pieces),
                                       vocabulary=pieces, add_blank=True)
    model.encoder.att_context_size = [int(att_left), v12.STREAMING_CHUNK_ENCODER_FRAMES - 1]
    model.encoder.setup_streaming_params()
    model.spec_augment = None
    if freq_masks or time_masks:
        from nemo.collections.asr.modules import SpectrogramAugmentation

        model.spec_augment = SpectrogramAugmentation(
            freq_masks=freq_masks, time_masks=time_masks, freq_width=27, time_width=0.05)
    return model


def forward(model, audio, audio_lengths, augment: bool):
    features, feature_lengths = model.preprocessor(input_signal=audio, length=audio_lengths)
    if augment and model.spec_augment is not None:
        features = model.spec_augment(input_spec=features, length=feature_lengths)
    encoded, encoded_lengths = model.encoder(audio_signal=features, length=feature_lengths)
    return model.decoder(encoder_output=encoded), encoded_lengths


def kd_loss(student_logits, batch: dict, student_lengths, temperature: float):
    """V8/V11's top-k KL with one bucket for the rest, read from the compact logits."""
    device = student_logits.device
    teacher_lengths = batch["teacher_lengths"].to(device)
    shortfall = student_lengths - teacher_lengths
    if torch.any(shortfall > MAX_TEACHER_SHORTFALL):
        raise RuntimeError(f"teacher shorter than student by {int(shortfall.max())} frames")
    max_time = min(student_logits.shape[1], batch["teacher_values"].shape[1])
    valid = torch.minimum(student_lengths, teacher_lengths).clamp(max=max_time)
    mask = torch.arange(max_time, device=device)[None, :] < valid[:, None]
    with torch.autocast(device_type=device.type, enabled=False):
        student = student_logits[:, :max_time].float() / temperature
        values = batch["teacher_values"][:, :max_time].to(device).float() / temperature
        indices = batch["teacher_indices"][:, :max_time].to(device)
        lse = batch["teacher_lse"][:, :max_time].to(device).float()
        teacher_top_logp = values - lse[:, :, None]
        student_top_logp = (torch.gather(student, -1, indices)
                            - torch.logsumexp(student, dim=-1, keepdim=True))
        teacher_top = teacher_top_logp.exp()
        teacher_other = (1.0 - teacher_top.sum(-1)).clamp_min(1e-8)
        student_other = (1.0 - student_top_logp.exp().sum(-1)).clamp_min(1e-8)
        frame = ((teacher_top * (teacher_top_logp - student_top_logp)).sum(-1)
                 + teacher_other * (teacher_other.log() - student_other.log()))
        frame = frame * temperature ** 2
        return (frame * mask).sum() / mask.sum().clamp_min(1)


def ctc_loss(student_logits, student_lengths, batch: dict):
    """CTC NLL per target token, averaged over the utterances CTC can align."""
    lengths = batch["target_lengths"]
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        log_probs = F.log_softmax(student_logits.float(), dim=-1).transpose(0, 1)
        losses = F.ctc_loss(log_probs, batch["targets"].to(student_logits.device),
                            student_lengths.cpu(), lengths, blank=BLANK,
                            reduction="none", zero_infinity=True)
        per_token = losses / lengths.to(losses.device).clamp_min(1).float()
        usable = torch.isfinite(per_token) & (lengths.to(losses.device) > 0) & (losses > 0)
        return per_token[usable].mean() if usable.any() else per_token.sum() * 0.0


# ============================================================
# Validation
# ============================================================

def greedy_text(best: np.ndarray, tokenizer) -> str:
    ids = [int(t) for i, t in enumerate(best) if t != BLANK and (i == 0 or t != best[i - 1])]
    return vi_number_norm.normalize_for_asr(tokenizer.decode(ids))


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    import editdistance

    return editdistance.eval(reference.split(), hypothesis.split()), len(reference.split())


def validate(model, rows, reader: Utterances, device, temperature: float) -> dict:
    """Greedy WER plus the KD and CTC losses on the fixed validation sample."""
    model.eval()
    errors = words = 0
    kd_sum = ctc_sum = batches = 0.0
    per_dataset: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    items: list[dict] = []
    started = time.time()
    try:
        with torch.inference_mode():
            for index, row in enumerate(rows):
                group, first, _ = reader.row_group_of(row["audio_path"], row["audio_row"])
                audio = reader.audio_batch(row["audio_path"], group,
                                           [row["audio_row"] - first])[row["audio_row"] - first]
                values, indices, lse = reader.teacher(
                    row["top64_stem"], row["teacher_offset"], row["teacher_length"], temperature)
                items.append({"audio": audio, "seconds": len(audio) / v12.TARGET_SAMPLE_RATE,
                              "targets": reader.tokenizer.encode(row["norm"]), "values": values,
                              "indices": indices, "lse": lse, "dataset": row["dataset"],
                              "norm": row["norm"]})
                if len(items) < 16 and index + 1 < len(rows):
                    continue
                batch = collate(items)
                logits, lengths = forward(model, batch["audio"].to(device),
                                          batch["audio_lengths"].to(device), False)
                kd_sum += float(kd_loss(logits, batch, lengths, temperature))
                ctc_sum += float(ctc_loss(logits, lengths, batch))
                batches += 1
                best = logits.argmax(-1).cpu().numpy()
                for position, item in enumerate(items):
                    hypothesis = greedy_text(best[position, :int(lengths[position])],
                                             reader.tokenizer)
                    wrong, count = word_errors(item["norm"], hypothesis)
                    errors += wrong
                    words += count
                    cell = per_dataset[item["dataset"]]
                    cell[0] += wrong
                    cell[1] += count
                items = []
    finally:
        model.train()
    summary = {"wer": errors / max(words, 1), "kd": kd_sum / max(batches, 1),
               "ctc": ctc_sum / max(batches, 1),
               "validation_seconds": round(time.time() - started, 1)}
    for dataset, (wrong, count) in sorted(per_dataset.items()):
        summary[f"wer_{dataset}"] = wrong / max(count, 1)
    return summary


# ============================================================
# Training
# ============================================================

def parse_stages(text: str) -> list[tuple[float, float]]:
    stages = []
    for entry in text.split(","):
        kd, _, ctc = entry.partition(":")
        stages.append((float(kd), float(ctc)))
    if not stages or any(kd + ctc <= 0 for kd, ctc in stages):
        raise ValueError(f"--stages {text!r}: every stage needs a positive weight")
    return stages


def learning_rate(step: int, stage_index: int, scale: float, options) -> float:
    """Warm-up, cosine decay over ``--steps``, times the per-stage decay and the
    rollback scale."""
    if step < options.warmup_steps:
        base = options.learning_rate * (step + 1) / options.warmup_steps
    else:
        progress = (step - options.warmup_steps) / max(options.steps - options.warmup_steps, 1)
        base = options.learning_rate * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    base *= options.stage_lr_decay ** stage_index * scale
    return max(base, options.min_learning_rate)


def make_optimizer(model, options):
    """A fresh AdamW: used at start, at every stage change and every rollback, so the
    moments of a diverging trajectory are not carried into the restored weights."""
    return torch.optim.AdamW(model.parameters(), lr=options.learning_rate,
                             betas=(0.9, 0.98), weight_decay=options.weight_decay)


def save(path: Path, payload: dict) -> None:
    temporary = path.parent / (path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def free_gigabytes(path: Path) -> float:
    usage = os.statvfs(path)
    return usage.f_bavail * usage.f_frsize / 1e9


def main(argv=None) -> int:
    options = parse_args(argv)
    v12.install_runtime_imports()
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    run_directory = options.run_root / options.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    if free_gigabytes(run_directory) < options.min_free_gb:
        raise SystemExit(f"{run_directory}: only {free_gigabytes(run_directory):.1f} GB free")

    train, validation_rows = load_manifest(options.manifest)
    probabilities = dataset_probabilities(train, options.dataset_alpha)
    validation_rows = random.Random(options.seed).sample(
        validation_rows, min(options.validate_utterances, len(validation_rows)))
    print(f"train {sum(train_size(train).values()):,} utterances in {len(train)} datasets; "
          f"validation sample {len(validation_rows):,}", flush=True)
    print("dataset draw shares: "
          + ", ".join(f"{d} {p:.3f}" for d, p in sorted(probabilities.items())), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_student(options.att_left_frames, options.freq_masks, options.time_masks)
    initialisation = {"mode": "random"}
    if options.init_checkpoint is not None:
        state, _ = v12.checkpoint_student_state(
            torch.load(options.init_checkpoint, map_location="cpu", weights_only=False))
        missing, unexpected = model.load_state_dict(state, strict=False)
        initialisation = {"mode": "checkpoint", "path": str(options.init_checkpoint),
                          "missing": len(missing), "unexpected": len(unexpected)}
    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"student {parameters:,} parameters, att_context {model.encoder.att_context_size}, "
          f"init {initialisation}", flush=True)

    optimizer = make_optimizer(model, options)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[options.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=options.amp == "fp16")
    stages = parse_stages(options.stages)
    state = {"step": 0, "stage": 0, "stage_start": 0, "best_stage_loss": float("inf"),
             "bad_validations": 0, "best_wer": float("inf"), "lr_scale": 1.0,
             "rollbacks": 0}
    latest = run_directory / "latest.pt"
    if latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        state.update(payload["state"])
        print(f"resumed from {latest} at step {state['step']:,}, stage {state['stage'] + 1}"
              f"/{len(stages)} (best WER {state['best_wer']:.4f})", flush=True)
    print("stages (KD:CTC): " + ", ".join(f"{kd:g}:{ctc:g}" for kd, ctc in stages)
          + f" | end a stage after {options.patience} validations without "
          f"{options.min_delta:.1%} improvement, {options.stage_min_steps:,}–"
          f"{options.stage_max_steps:,} steps", flush=True)

    run = None
    if options.wandb_mode != "disabled" and not options.smoke_steps:
        import wandb

        run = wandb.init(project=options.wandb_project, entity=options.wandb_entity,
                         name=options.run_name, mode=options.wandb_mode,
                         tags=options.wandb_tags.split(","),
                         config={**vars(options), "parameters": parameters,
                                 "initialisation": initialisation,
                                 "dataset_shares": probabilities})

    loader = torch.utils.data.DataLoader(
        TrainStream(train, probabilities, options), batch_size=None,
        num_workers=options.workers, persistent_workers=options.workers > 0,
        prefetch_factor=4 if options.workers else None)
    reader = Utterances(options.tokenizer)
    metrics = run_directory / "training_metrics.jsonl"
    window = collections.deque(maxlen=options.log_every)
    started, audio_seconds = time.time(), 0.0
    target_steps = options.smoke_steps or options.stop_at_step or options.steps
    model.train()
    finished = False
    for batch in loader:
        step = state["step"]
        if step >= target_steps or finished:
            break
        kd_weight, ctc_weight = stages[state["stage"]]
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, state["stage"], state["lr_scale"], options)
        audio = batch["audio"].to(device, non_blocking=True)
        audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            logits, lengths = forward(model, audio, audio_lengths, True)
            kd = (kd_loss(logits, batch, lengths, options.temperature) if kd_weight
                  else torch.zeros((), device=device))
            ctc = ctc_loss(logits, lengths, batch) if ctc_weight else torch.zeros((), device=device)
            loss = kd_weight * kd + ctc_weight * ctc
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        state["step"] = step = step + 1
        audio_seconds += batch["seconds"]
        window.append((float(kd.detach()), float(ctc.detach()), float(loss.detach())))
        if step % options.log_every == 0:
            kd_mean, ctc_mean, loss_mean = (sum(x) / len(window) for x in zip(*window))
            elapsed = time.time() - started
            print(f"[train] step={step:,} loss={loss_mean:.4f} kd={kd_mean:.4f} "
                  f"ctc={ctc_mean:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"grad={float(norm):.2f} audio={audio_seconds / 3600:.1f} h "
                  f"{step / max(elapsed, 1e-9):.2f} steps/s", flush=True)
            record = {"kind": "train", "step": step, "loss": loss_mean, "kd": kd_mean,
                      "ctc": ctc_mean, "lr": optimizer.param_groups[0]["lr"],
                      "audio_hours": audio_seconds / 3600}
            with metrics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if run is not None:
                run.log(record, step=step)
        if step % options.validate_every == 0 or step >= target_steps:
            rows = validation_rows[:64] if options.smoke_steps else validation_rows
            summary = validate(model, rows, reader, device, options.temperature)
            stage_loss = kd_weight * summary["kd"] + ctc_weight * summary["ctc"]
            summary["stage_loss"] = stage_loss
            summary["stage"] = state["stage"] + 1
            print(f"[validation] step={step:,} stage={state['stage'] + 1}/{len(stages)} "
                  f"({kd_weight:g}:{ctc_weight:g}) loss={stage_loss:.4f} kd={summary['kd']:.4f} "
                  f"ctc={summary['ctc']:.4f} WER={summary['wer']:.4f} "
                  + " ".join(f"{k[4:]}={v:.3f}" for k, v in summary.items()
                             if k.startswith("wer_")), flush=True)
            with metrics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "validation", "step": step, **summary}) + "\n")
            if run is not None:
                run.log({f"validation/{k}": v for k, v in summary.items()
                         if isinstance(v, (int, float))}, step=step)
            improved = stage_loss < state["best_stage_loss"] * (1 - options.min_delta)
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "state": dict(state), "validation": summary,
                       "options": {**vars(options), "run_name": options.run_name},
                       "initialisation": initialisation, "vocabulary_size": ASR_CLASSES,
                       "att_context": model.encoder.att_context_size}
            save(latest, payload)
            stage_best = run_directory / f"best_stage{state['stage'] + 1}.pt"
            if improved:
                state["best_stage_loss"] = stage_loss
                state["bad_validations"] = 0
                save(stage_best, payload)
            else:
                state["bad_validations"] += 1
            if summary["wer"] < state["best_wer"]:
                state["best_wer"] = summary["wer"]
                save(run_directory / "best_wer.pt", payload)
                print(f"saved best_wer.pt at step {step:,} (WER {state['best_wer']:.4f})",
                      flush=True)
            in_stage = step - state["stage_start"]
            diverged = (stage_best.is_file() and stage_loss
                        > state["best_stage_loss"] * (1 + options.rollback_threshold))
            if diverged and state["rollbacks"] < options.max_rollbacks:
                # Divergence is not convergence: restore the stage best and continue
                # the same stage at a lower LR with fresh optimizer moments.
                model.load_state_dict(torch.load(stage_best, map_location="cpu",
                                                 weights_only=False)["model"])
                model.to(device)
                optimizer = make_optimizer(model, options)
                state.update(lr_scale=state["lr_scale"] * options.rollback_lr_factor,
                             rollbacks=state["rollbacks"] + 1, bad_validations=0)
                print(f"[rollback] step {step:,}: validation loss {stage_loss:.4f} > "
                      f"{1 + options.rollback_threshold:.2f} x best {state['best_stage_loss']:.4f};"
                      f" restored {stage_best.name}, lr scale {state['lr_scale']:.3f} "
                      f"(rollback {state['rollbacks']}/{options.max_rollbacks})", flush=True)
                continue
            plateau = (state["bad_validations"] >= options.patience
                       and in_stage >= options.stage_min_steps)
            if plateau or diverged or in_stage >= options.stage_max_steps:
                reason = ("divergence after rollbacks" if diverged and not plateau
                          else "plateau" if plateau else "stage step cap")
                if stage_best.is_file():
                    model.load_state_dict(torch.load(stage_best, map_location="cpu",
                                                     weights_only=False)["model"])
                    model.to(device)
                optimizer = make_optimizer(model, options)
                if state["stage"] + 1 >= len(stages):
                    print(f"[stage] last stage ended ({reason}) at step {step:,}", flush=True)
                    finished = True
                else:
                    state.update(stage=state["stage"] + 1, stage_start=step,
                                 best_stage_loss=float("inf"), bad_validations=0,
                                 lr_scale=1.0, rollbacks=0)
                    print(f"[stage] {reason} at step {step:,}; continuing from "
                          f"{stage_best.name} at stage {state['stage'] + 1}/{len(stages)} "
                          f"({stages[state['stage']][0]:g}:{stages[state['stage']][1]:g}), "
                          f"lr x{options.stage_lr_decay ** state['stage']:.2f}, "
                          "fresh optimizer", flush=True)
            if free_gigabytes(run_directory) < options.min_free_gb:
                raise SystemExit(f"disk below {options.min_free_gb} GB; stopping cleanly")
    save(run_directory / "student_final_weights.pt",
         {"student_state_dict": model.state_dict(), "step": state["step"],
          "state": dict(state), "att_context": model.encoder.att_context_size,
          "options": vars(options)})
    print(f"done at step {state['step']:,}, stage {state['stage'] + 1}/{len(stages)}; "
          f"best WER {state['best_wer']:.4f}", flush=True)
    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
