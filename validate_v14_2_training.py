#!/usr/bin/env python3
"""CPU-only checks for V14.2, before any GPU time is spent.

Covers the three places V14.2 differs from V14, and the two mistakes that cost
this project a run each: a curriculum ratio above what a batch can hold (the
``ratios_steep`` run died 47 seconds in), and a channel transform leaking into
validation data (which would move the metric with the augmentation instead of
with the model).

    python validate_v14_2_training.py
"""

from __future__ import annotations

import collections
import json
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

import parakeet_v12_train as v12
import parakeet_v14_2_channel as channel
import parakeet_v14_2_train as v14_2
import streaming_protocol as protocol

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f" | {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def fake_rirs(directory: Path, count: int = 6) -> Path:
    """A few short impulse responses, shaped like the openSLR tree."""
    import soundfile as sf

    root = directory / "RIRS_NOISES" / "simulated_rirs" / "smallroom" / "Room001"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for index in range(count):
        length = int(0.3 * channel.SAMPLE_RATE)
        decay = np.exp(-np.arange(length) / (0.08 * channel.SAMPLE_RATE))
        wave = rng.normal(0, 1, length) * decay
        wave[0] += 4.0
        sf.write(str(root / f"Room001-{index:05d}.wav"),
                 (wave / np.abs(wave).max() * 0.9).astype(np.float32),
                 channel.SAMPLE_RATE)
    noise = directory / "RIRS_NOISES" / "pointsource_noises"
    noise.mkdir(parents=True, exist_ok=True)
    sf.write(str(noise / "noise-free-sound-0001.wav"),
             rng.normal(0, 0.1, channel.SAMPLE_RATE).astype(np.float32), channel.SAMPLE_RATE)
    return directory


def test_channel_transforms() -> None:
    print("[1] biến đổi kênh truyền")
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 0.1, 2 * channel.SAMPLE_RATE)

    narrow = channel.apply_band(wave, 300.0, 3400.0)
    spectrum = np.abs(np.fft.rfft(narrow))
    freqs = np.fft.rfftfreq(len(narrow), 1.0 / channel.SAMPLE_RATE)
    check("cắt băng bỏ hết ngoài dải",
          spectrum[freqs < 250].max() < 1e-6 and spectrum[freqs > 3600].max() < 1e-6,
          f"đỉnh ngoài dải {max(spectrum[freqs < 250].max(), spectrum[freqs > 3600].max()):.2e}")
    check("cắt băng giữ nguyên độ dài", len(narrow) == len(wave))

    with tempfile.TemporaryDirectory() as tmp:
        root = fake_rirs(Path(tmp))
        responses = channel.load_rirs(root)
        check("chỉ nạp impulse response, bỏ file nhiễu", len(responses) == 6,
              f"nạp {len(responses)}")
        wet = channel.apply_room(wave, responses[0])
        before = float(np.sqrt(np.mean(wave ** 2)))
        after = float(np.sqrt(np.mean(wet ** 2)))
        check("vang phòng giữ nguyên độ to", abs(before - after) < 1e-9,
              f"{before:.4f} -> {after:.4f}")
        check("vang phòng giữ nguyên độ dài", len(wet) == len(wave))


def test_channel_determinism() -> None:
    print("[2] tính xác định và phạm vi áp dụng")
    with tempfile.TemporaryDirectory() as tmp:
        root = fake_rirs(Path(tmp))
        report = channel.install(rir_root=root, room_probability=1.0,
                                 band_probability=1.0, salt=0)
        check("install báo đúng số RIR", report["rirs"] == 6, str(report))
        rng = np.random.default_rng(1)
        wave = rng.normal(0, 0.1, channel.SAMPLE_RATE).astype(np.float32)
        first = channel.transform(wave, "uid-a")
        again = channel.transform(wave, "uid-a")
        other = channel.transform(wave, "uid-b")
        check("cùng uid cho cùng kết quả", np.array_equal(first, again))
        check("uid khác cho kết quả khác", not np.array_equal(first, other))
        check("biên độ nằm trong [-1, 1]", float(np.abs(first).max()) <= 1.0,
              f"{float(np.abs(first).max()):.3f}")

        # The dataset patch must leave validation and test untouched.
        seen = {}

        def fake_getitem(self, index):
            return {"sample": seen["sample"], "audio": torch.from_numpy(wave.copy())}

        original = channel._ORIGINAL_GETITEM
        channel._ORIGINAL_GETITEM = fake_getitem
        try:
            patched = v12.UnifiedAudioDataset.__getitem__
            for split, should_change in (("train", True), ("validation", False),
                                         ("test", False)):
                seen["sample"] = types.SimpleNamespace(split=split, uid=f"uid-{split}")
                result = patched(None, 0)["audio"].numpy()
                changed = not np.array_equal(result, wave)
                check(f"split {split}: {'có' if should_change else 'không'} augment",
                      changed == should_change)
        finally:
            channel._ORIGINAL_GETITEM = original
            channel.uninstall()


def test_options_and_guards() -> None:
    print("[3] tuỳ chọn và guard")
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "clip.wav"
        sf.write(str(audio), np.zeros(channel.SAMPLE_RATE, dtype=np.float32),
                 channel.SAMPLE_RATE)
        options = v14_2.option_parser().parse_args(["--v14-2-select-audio", str(audio)])
        v14_2.configure(options)
        check("mặc định 2 stage 6k/14k", v14_2.STAGE_STEPS == (6000, 14000),
              str(v14_2.STAGE_STEPS))
        try:
            v14_2.configure(v14_2.option_parser().parse_args(
                ["--v14-2-select-audio", str(Path(tmp) / "khong-co.wav")]))
            check("báo lỗi khi thiếu file audio chọn", False)
        except FileNotFoundError:
            check("báo lỗi khi thiếu file audio chọn", True)

        args = types.SimpleNamespace(validation_every_steps=2000,
                                     max_utterances_per_batch=32,
                                     development_only=False,
                                     final_selection_checkpoint="best_f1.pt")
        positives = [types.SimpleNamespace(split="validation", is_positive=True,
                                           source_dataset="WakewordV12", uid="WakewordV12:v")]
        for ratios, every, reason in (([1, 8, 16], 2000, "số stage không khớp"),
                                      ([1, 8], 3000, "bước stage không chia hết"),
                                      ([1, 64], 2000, "tỉ lệ vượt trần batch")):
            args.validation_every_steps = every
            try:
                v14_2.train_v12(args, ratios, positives, {}, [], Path("."))
                check(f"chặn: {reason}", False)
            except ValueError:
                check(f"chặn: {reason}", True)

        args.validation_every_steps = 2000
        original = v14_2._V14_TRAIN
        seen = {}
        v14_2._V14_TRAIN = lambda *a, **k: seen.setdefault("called", True)
        try:
            v14_2.train_v12(args, [1, 8], positives, {}, [], Path("."))
            check("chuyển tiếp sang V14", seen.get("called") is True)
            check("xuất theo checkpoint audio dài",
                  args.final_selection_checkpoint == v14_2.BEST_FILE)
            check("giữ lại positive để chấm", len(v14_2._EVAL_POSITIVES) == 1)
        finally:
            v14_2._V14_TRAIN = original


def test_best_checkpoint() -> None:
    print("[4] lưu checkpoint tốt nhất theo audio dài")
    v14_2._LAST_STUDENT = torch.nn.Linear(4, 5)
    v14_2._SIGNATURE.clear()
    v14_2._SIGNATURE.update({"trainer_variant": v14_2.TRAINER_VARIANT})
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = types.SimpleNamespace(
            run_directory=Path(tmp), wandb_run=None,
            update_best_summary=lambda *a: calls.append(a))
        original = v14_2._V12_APPEND
        v14_2._V12_APPEND = lambda self, record, preview=None: None
        try:
            def entry(step, stream, zero_fa, cold2, cold0):
                return {"kind": "validation", "optimizer_step": step, "epoch": 1,
                        "stage_index_before_observation": 0,
                        "la_selection_recall": stream,
                        "la_selection_score": stream,
                        "la_recall_at_zero_fa": zero_fa,
                        protocol.cold_key(2.0): cold2,
                        protocol.cold_key(0.0): cold0}

            target = Path(tmp) / v14_2.BEST_FILE
            best = lambda: torch.load(target, map_location="cpu",
                                      weights_only=False)["optimizer_step"]
            v14_2.artifacts_append(artifacts, entry(4000, 0.9994, 0.997, 0.0, 0.21))
            check("lần đầu có lưu", target.is_file() and len(calls) == 1)
            v14_2.artifacts_append(artifacts, entry(6000, 0.9980, 0.999, 0.0, 0.01))
            check("recall streaming thấp hơn thì không chọn", best() == 4000)
            v14_2.artifacts_append(artifacts, entry(8000, 0.9994, 0.998, 0.0, 0.30))
            check("hoà recall thì chọn bên recall 0 FA cao hơn", best() == 8000)
            v14_2.artifacts_append(artifacts, entry(10000, 0.9994, 0.998, 0.0, 0.10))
            check("hoà tiếp thì chọn bên ít kêu nhầm khi khởi động lạnh", best() == 10000)
            v14_2.artifacts_append(artifacts, entry(12000, 0.9994, 0.998, 0.02, 0.0))
            check("pre-roll 2 s xét trước không pre-roll", best() == 10000)
            state = json.loads((Path(tmp) / v14_2.BEST_STATE).read_text())
            check("ghi lại trạng thái và quy tắc",
                  state["optimizer_step"] == 10000 and state["rule"] == v14_2.SELECTION_RULE,
                  str(state))
            (Path(tmp) / v14_2.BEST_STATE).write_text(
                json.dumps({"key": [9.0, 9.0, 0.0], "optimizer_step": 1}))
            v14_2.artifacts_append(artifacts, entry(24000, 0.5, 0.5, 5.0, 5.0))
            check("trạng thái của quy tắc cũ không chặn quy tắc mới", best() == 24000)
            snapshots = sorted(p.name for p in (Path(tmp) / v14_2.SNAPSHOT_DIR).iterdir())
            check("lưu snapshot ở mọi lần validation", snapshots == [
                "step_004000.pt", "step_006000.pt", "step_008000.pt", "step_010000.pt",
                "step_012000.pt", "step_024000.pt"], str(snapshots))
            payload = torch.load(target, map_location="cpu", weights_only=False)
            check("payload đủ khoá",
                  {"criterion", "signature", "student_state_dict", "vocabulary"}
                  <= set(payload))
        finally:
            v14_2._V12_APPEND = original
            v14_2._LAST_STUDENT = None


def test_protocol_summary() -> None:
    print("[4b] giao thức đo chung: tóm tắt chỉ số")
    thresholds = np.array([0.0, 1.0, 2.0, 3.0])
    positives = np.array([0.5, 1.5, 2.5, 3.0])
    measured = {"hours": 2.0, "positives": positives, "thresholds": thresholds,
                "events": np.array([10, 2, 1, 0]),
                "cold_events": {0.0: np.array([50, 20, 10, 0]), 2.0: np.array([5, 2, 0, 0])},
                "cold_starts": {0.0: 100, 2.0: 100}}
    summary = protocol.summarize(measured, 1.0)
    check("recall tại <= 1 FA/h lấy ngưỡng thấp nhất đạt", summary["la_recall_at_fa_1"] == 0.75,
          str(summary["la_recall_at_fa_1"]))
    check("ngưỡng vận hành đúng", summary["la_selection_threshold"] == 1.0)
    check("kêu nhầm mỗi lần khởi động lạnh tại ngưỡng vận hành",
          summary[protocol.cold_key(0.0)] == 0.2 and summary[protocol.cold_key(2.0)] == 0.02,
          f"{summary[protocol.cold_key(0.0)]} {summary[protocol.cold_key(2.0)]}")
    check("recall ở 0 FA", summary["la_recall_at_zero_fa"] == 0.25)
    check("đoạn phải thẳng lưới chunk", _raises(lambda: protocol._samples(120.0,
                                                                        protocol.CHUNK_SAMPLES)))
    check("khởi động lạnh đủ chỗ cho pre-roll", min(protocol.restart_points(
        16000 * 60, 12.0)) >= 12 * 16000)


def _raises(action) -> bool:
    try:
        action()
    except ValueError:
        return True
    return False


def test_schedule_and_patches() -> None:
    print("[5] lịch stage và patch")
    state = types.SimpleNamespace(complete=False, stage_index=0, epochs_in_stage=0,
                                  best_metric=None, bad_validations=0,
                                  validation_observations_in_stage=0,
                                  stage_start_optimizer_step=0, metric_ema=None,
                                  best_detection_metric=None,
                                  bad_detection_validations=0)
    decisions = [v14_2.staged_schedule_observe(state, 1.0, step, 2, 0, 0, 0, 0.0, 0.0)
                 for step in range(2000, 22000, 2000)]
    check("chuyển stage đúng bước 6.000", decisions[:3] == ["held", "held", "advanced"],
          str(decisions[:4]))
    check("kết thúc sau stage cuối", "complete" in decisions, str(decisions))

    import parakeet_v12_2_sharded_train as previous
    import parakeet_v14_train as v14
    before = (v12.evaluate, v12.curriculum_observe_steps, v12.RunArtifacts.append,
              v14.train_v12, v14.stratified_family_negative_sequence,
              v14.load_infore2_samples, v14.training_signature, v14.audit_samples,
              previous.reshuffle_speaker_splits, v14._V12_2_AUDIT_SAMPLES)
    saved_split = v14_2.SPLIT_FILE
    v14_2.SPLIT_FILE = {"001": "train"}
    v14_2.apply_patches()
    try:
        check("evaluate được patch", v12.evaluate is v14_2.evaluate_with_long_audio)
        check("curriculum được patch",
              v12.curriculum_observe_steps is v14_2.staged_schedule_observe)
        check("artifacts được patch", v12.RunArtifacts.append is v14_2.artifacts_append)
        check("KHÔNG cài sampler theo lớp của V14.1",
              v14.stratified_family_negative_sequence is before[4])
        check("nạp bản sao negative qua loader V14",
              v14.load_infore2_samples is v14_2.load_infore2_with_channel)
        check("có split file thì tắt reshuffle của V12.2",
              previous.reshuffle_speaker_splits is v14_2.keep_split_file)
        check("có split file thì dùng audit cho split tay",
              v14._V12_2_AUDIT_SAMPLES is v14_2.split_file_audit)
    finally:
        (v12.evaluate, v12.curriculum_observe_steps, v12.RunArtifacts.append,
         v14.train_v12, v14.stratified_family_negative_sequence,
         v14.load_infore2_samples, v14.training_signature, v14.audit_samples,
         previous.reshuffle_speaker_splits, v14._V12_2_AUDIT_SAMPLES) = before
        v14_2.SPLIT_FILE = saved_split


# ------------------------------------------------------------ precomputed copies

def sample(uid, source, speaker, transcript, index, positive=False, split="train",
           tokens=()):
    return v12.Sample(uid=uid, source_dataset=source, source_kind="parquet",
                      audio_path="x.parquet", audio_row=0, speaker_id=speaker,
                      transcript=transcript, utterance_index=str(index),
                      duration_seconds=1.0, reduced_token_ids=tuple(tokens),
                      is_positive=positive, split=split)


def test_plan_math() -> None:
    print("[6] quy đổi xác suất -> số bản sao")
    p = channel.augmented_fraction(0.15, 0.25)
    check("p bất kỳ = 1 - 0,85 x 0,75 = 0,3625", abs(p - 0.3625) < 1e-12, f"{p}")
    weights = channel.kind_weights(0.15, 0.25)
    check("tỉ trọng các loại cộng lại bằng 1", abs(sum(weights.values()) - 1) < 1e-12)
    check("gia đình 20 biến thể -> 11 bản sao", channel.copies_for_family(20) == 11,
          str(channel.copies_for_family(20)))
    kinds = channel.split_kinds(11, weights)
    check("11 bản = 3 vang + 7 băng + 1 cả hai",
          (kinds.count("room"), kinds.count("band"), kinds.count("room+band")) == (3, 7, 1),
          str(kinds))
    uids = [f"LSVSC:{i}" for i in range(20000)]
    copies = sum(channel.copies_for_row(uid, 0) for uid in uids)
    share = copies / (len(uids) + copies)
    check("nguồn một-dòng: tỉ lệ lượt rút vào bản sao ~ 0,3625", abs(share - p) < 0.01,
          f"{share:.4f}")
    mix = collections.Counter(channel.pick_kind(uid, 0, weights) for uid in uids)
    check("trộn loại theo tỉ trọng", abs(mix["band"] / len(uids) - weights["band"]) < 0.02,
          str(dict(mix)))
    uid = channel.copy_uid("InfoRE2:7", 0, "band")
    check("uid bản sao nhận ra được và truy về bản gốc",
          channel.is_channel_copy(uid) and v14_2.origin_uid(uid) == "InfoRE2:7"
          and not channel.is_channel_copy("InfoRE2:7"), uid)
    wave = np.random.default_rng(2).normal(0, 0.3, channel.SAMPLE_RATE)
    rir = np.zeros(400, dtype=np.float32)
    rir[0], rir[200] = 0.7, 0.3
    for kind in channel.KINDS:
        out = channel.apply_kind(wave, kind, rir, (300.0, 3400.0))
        check(f"{kind}: giữ độ dài, biên độ <= 1",
              len(out) == len(wave) and float(np.abs(out).max()) <= 1.0)
    fast = channel.apply_room(wave, rir)
    slow = np.convolve(wave, rir)[:len(wave)]
    slow = slow * np.sqrt(np.mean(wave ** 2)) / np.sqrt(np.mean(slow ** 2))
    check("fftconvolve khớp np.convolve", float(np.abs(fast - slow).max()) < 1e-6,
          f"{float(np.abs(fast - slow).max()):.1e}")


def channel_samples(copy_negatives=True, copy_positives=True):
    """Two train speakers of positives, one negative per source, one eval speaker."""
    samples = []
    for speaker in ("001", "002"):
        for utterance in range(3):
            text = "năng khiếu ơi"
            for variant in range(20):
                samples.append(sample(f"W:{speaker}:{utterance}:{variant}", "WakewordV12",
                                      speaker, text, utterance, positive=True, tokens=(0, 1, 2, 3)))
            if copy_positives:
                for j in range(11):
                    samples.append(sample(channel.copy_uid(f"W:{speaker}:{utterance}:{j}", j,
                                                           "band"), "WakewordV12", speaker,
                                          text, utterance, positive=True, tokens=(0, 1, 2, 3)))
    samples.append(sample("W:009:0:0", "WakewordV12", "009", "năng khiếu ơi", 0,
                          positive=True, split="validation", tokens=(0, 1, 2, 3)))
    for source in ("WakewordV12", "InfoRE2", "VLSP", "LSVSC", "ViMD", "MUSAN"):
        for row in range(2000):
            hard = row % 4 == 0 and source != "MUSAN"
            speaker = "001" if source == "WakewordV12" else None
            uid = f"{source}:{row}"
            if source == "WakewordV12":
                for variant in range(20):
                    samples.append(sample(f"{uid}:{variant}", source, speaker, f"câu {row}", row,
                                          tokens=(1,) if hard else ()))
                if copy_negatives:
                    for j in range(11):
                        samples.append(sample(channel.copy_uid(f"{uid}:{j}", j, "room"), source,
                                              speaker, f"câu {row}", row,
                                              tokens=(1,) if hard else ()))
                continue
            samples.append(sample(uid, source, speaker, f"câu {row}", row,
                                  tokens=(1,) if hard else ()))
            if copy_negatives:
                for j in range(channel.copies_for_row(uid, 0)):
                    samples.append(sample(channel.copy_uid(uid, j, "band"), source, speaker,
                                          f"câu {row}", f"{row}{channel.CHANNEL_UID_TAG}{j}",
                                          tokens=(1,) if hard else ()))
    return samples


def expect_error(name: str, action) -> None:
    try:
        action()
        check(name, False, "không báo lỗi")
    except RuntimeError as error:
        check(name, True, str(error)[:90])


def test_channel_audit() -> None:
    print("[7] audit bản sao kênh truyền")
    import parakeet_v14_train as v14

    saved = dict(v14.SOURCE_SHARES)
    v14.SOURCE_SHARES.clear()
    v14.SOURCE_SHARES.update(v14.parse_source_shares(v14_2.DEFAULT_SOURCE_SHARES))
    try:
        report = v14_2.audit_channel(channel_samples())
        check("tỉ lệ positive ~ negative ~ 0,36",
              abs(report["positive_draw_share"] - report["negative_draw_share"]) < 0.02
              and abs(report["positive_draw_share"] - 0.355) < 0.01, str(
                  (report["positive_draw_share"], report["negative_draw_share"])))
        check("không có bản sao thì không audit", v14_2.audit_channel(
            channel_samples(False, False)) == {})
        expect_error("chặn: bản sao chỉ có ở positive",
                     lambda: v14_2.audit_channel(channel_samples(copy_negatives=False)))
        leaked = channel_samples() + [sample(channel.copy_uid("W:009:0:0", 0, "band"),
                                             "WakewordV12", "009", "năng khiếu ơi", 0,
                                             positive=True, split="validation")]
        expect_error("chặn: bản sao nằm ngoài train", lambda: v14_2.audit_channel(leaked))
        orphan = channel_samples() + [sample(channel.copy_uid("VLSP:999999", 0, "band"),
                                             "VLSP", None, "câu", "x#ch0")]
        expect_error("chặn: bản sao không có bản gốc trong train",
                     lambda: v14_2.audit_channel(orphan))
        channel._CONFIG["band_probability"] = 0.25
        try:
            expect_error("chặn: augment hai lần (bản sao + lúc chạy)",
                         lambda: v14_2.audit_channel(channel_samples()))
        finally:
            channel._CONFIG["band_probability"] = 0.0
    finally:
        v14.SOURCE_SHARES.clear()
        v14.SOURCE_SHARES.update(saved)


def test_split_and_guards() -> None:
    print("[8] split theo file và chặn cấu hình")
    import parakeet_v12_2_sharded_train as previous

    samples = [sample("a", "WakewordV12", "001", "năng khiếu ơi", 0, positive=True),
               sample("b", "WakewordV12", "013", "năng khiếu ơi", 0, positive=True,
                      split="validation")]
    kept, splits = v14_2.keep_split_file(samples, 2026)
    check("không xáo lại split", splits == {"001": "train", "013": "validation"}
          and [s.split for s in kept] == ["train", "validation"], str(splits))
    original = previous._ORIGINAL_AUDIT_SAMPLES
    saved = v14_2.SPLIT_FILE
    previous._ORIGINAL_AUDIT_SAMPLES = lambda *a, **k: None
    try:
        v14_2.SPLIT_FILE = {"001": "train", "013": "validation"}
        v14_2.split_file_audit(samples, splits)
        check("audit split tay chấp nhận 0 người test", True)
        v14_2.SPLIT_FILE = {"001": "train", "013": "train"}
        expect_error("chặn: split lệch file", lambda: v14_2.split_file_audit(samples, splits))
    finally:
        previous._ORIGINAL_AUDIT_SAMPLES = original
        v14_2.SPLIT_FILE = saved

    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "clip.wav"
        sf.write(str(audio), np.zeros(channel.SAMPLE_RATE, dtype=np.float32),
                 channel.SAMPLE_RATE)
        manifest = Path(tmp) / "neg.jsonl"
        manifest.write_text("")
        options = v14_2.option_parser().parse_args(
            ["--v14-2-select-audio", str(audio), "--v14-2-channel-negative-manifest",
             str(manifest), "--v14-2-band-probability", "0.25"])
        try:
            v14_2.configure(options)
            check("chặn: manifest bản sao + xác suất lúc chạy > 0", False)
        except ValueError:
            check("chặn: manifest bản sao + xác suất lúc chạy > 0", True)
        options = v14_2.option_parser().parse_args(["--v14-2-select-audio", str(audio)])
        check("mặc định tắt augment lúc chạy",
              options.v14_2_room_probability == 0 and options.v14_2_band_probability == 0)
        v14_2.configure(options)


# ------------------------------------------------------------ prepare script, end to end

def write_parquet(path: Path, payloads, extra=None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {"audio": pa.array(payloads, type=pa.binary() if isinstance(payloads[0], bytes)
                                 else None)}
    columns.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, row_group_size=4)


def wav(subtype: str, seconds: float, seed: int) -> bytes:
    import io
    import soundfile as sf

    buffer = io.BytesIO()
    wave = np.random.default_rng(seed).normal(0, 0.1, int(seconds * channel.SAMPLE_RATE))
    sf.write(buffer, wave.astype(np.float32), channel.SAMPLE_RATE, subtype=subtype, format="WAV")
    return buffer.getvalue()


def fake_sources(root: Path) -> dict:
    """Wakeword (2 train + 1 eval speaker), prepared negatives, InfoRE2, RIRs."""
    import pyarrow as pa

    wake = root / "wake"
    records, payloads = [], []
    for speaker, split in (("001", "train"), ("002", "train"), ("013", "validation")):
        for utterance in range(2):
            for variant in range(20):
                positive = utterance == 0
                payloads.append(wav("FLOAT", 0.5, len(payloads)))
                records.append({
                    "uid": f"WakewordV12:{speaker}-{utterance}-{variant}",
                    "audio_path": "shards/all-00000.parquet", "audio_row": len(records),
                    "speaker_id": speaker, "split_group": speaker, "split": split,
                    "transcript": "năng khiếu ơi" if positive else "năng lực",
                    "utterance_index": str(utterance), "augmentation": f"v{variant}",
                    "is_augmented": variant > 0, "is_positive": positive,
                    "duration_seconds": 0.5, "source_kind": "parquet", "sample_rate": 16000,
                    "prepared_audio_subtype": "FLOAT", "shard_row_group_size": 64})
    write_parquet(wake / "shards" / "all-00000.parquet", payloads)
    wake_manifest = wake / "manifest.jsonl"
    wake_manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                             encoding="utf-8")

    negatives = root / "neg"
    rows = []
    for source in ("LSVSC", "VLSP", "ViMD", "MUSAN"):
        payloads = [wav("PCM_16", 0.4, i) for i in range(12)]
        write_parquet(negatives / "shards" / f"{source}-00000.parquet", payloads)
        rows += [{"uid": f"{source}:{i}", "source_dataset": source, "duration_seconds": 0.4,
                  "transcript": "" if source == "MUSAN" else f"câu số {i}",
                  "speaker_id": "musan:t" if source == "MUSAN" else None,
                  "utterance_index": str(i), "audio_relpath": f"shards/{source}-00000.parquet",
                  "audio_row": i} for i in range(12)]
    negative_manifest = negatives / "manifest.jsonl"
    negative_manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                 encoding="utf-8")

    data = root / "data"
    infore = [{"bytes": wav("PCM_16", 0.4, 100 + i), "path": None} for i in range(12)]
    write_parquet(data / "audio_shards" / "InfoRE2" / "0000.parquet", infore,
                  {"transcription": pa.array([f"sách nói {i}" for i in range(12)])})
    aligned = [{"dataset": "InfoRE2", "sample_id": i, "audio_relpath": "InfoRE2/0000.parquet",
                "audio_row": i, "teacher_length": 5} for i in range(12)]
    (data / "manifests").mkdir(parents=True)
    (data / "manifests" / "aligned_manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in aligned), encoding="utf-8")

    splits = root / "splits.json"
    splits.write_text(json.dumps({"001": "train", "002": "train", "013": "validation"}))
    return {"wake": wake_manifest, "neg": negative_manifest, "data": data, "splits": splits,
            "rirs": fake_rirs(root / "rirs")}


def run_prepare(paths: dict, output: Path, *extra: str) -> int:
    import prepare_v14_2_channel_shards as prepare

    return prepare.main(["--wakeword-manifest", str(paths["wake"]),
                         "--negative-manifest", str(paths["neg"]),
                         "--data-root", str(paths["data"]),
                         "--speaker-split-file", str(paths["splits"]),
                         "--rir-root", str(paths["rirs"]),
                         "--output-root", str(output), "--workers", "2", *extra])


def test_prepare_end_to_end() -> None:
    print("[9] script tạo shard, đầu-cuối trên dữ liệu tổng hợp")
    import contextlib
    import io

    import pyarrow.parquet as pq
    import soundfile as sf

    import prepare_v14_2_channel_shards as prepare

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = fake_sources(root)
        output = root / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            status = run_prepare(paths, output)
        check("chạy xong", status == 0)
        wake = [json.loads(l) for l in (output / prepare.WAKEWORD_MANIFEST_NAME).open(encoding="utf-8")]
        copies = [r for r in wake if channel.is_channel_copy(r["uid"])]
        originals = [r for r in wake if not channel.is_channel_copy(r["uid"])]
        check("manifest wakeword giữ đủ 120 dòng gốc", len(originals) == 120, str(len(originals)))
        check("4 gia đình train x 11 bản sao", len(copies) == 44, str(len(copies)))
        check("không có bản sao của người validation",
              all(r["speaker_id"] in ("001", "002") for r in copies))
        check("bản sao cùng speaker/split/transcript/utterance với gốc", all(
            r["split"] == "train" and r["uid"].startswith(f"WakewordV12:{r['speaker_id']}-"
                                                          f"{r['utterance_index']}-")
            for r in copies))
        check("đường dẫn audio tuyệt đối và tồn tại",
              all(Path(r["audio_path"]).is_absolute() and Path(r["audio_path"]).is_file()
                  for r in wake))
        row = copies[0]
        audio = pq.read_table(row["audio_path"]).column("audio")[row["audio_row"]].as_py()
        data, rate = sf.read(io.BytesIO(audio), dtype="float32")
        check("bản sao wakeword: float32 16 kHz, cùng độ dài",
              rate == 16000 and len(data) == 8000 and row["prepared_audio_subtype"] == "FLOAT")
        negatives = [json.loads(l) for l in (output / prepare.NEGATIVE_MANIFEST_NAME).open(encoding="utf-8")]
        sources = collections.Counter(r["source_dataset"] for r in negatives)
        check("có bản sao cho cả 5 nguồn negative", set(sources) ==
              {"InfoRE2", "LSVSC", "VLSP", "ViMD", "MUSAN"}, str(dict(sources)))
        check("bản sao nguồn một-dòng là gia đình riêng", all(
            channel.CHANNEL_UID_TAG in r["utterance_index"] for r in negatives))
        check("InfoRE2 lấy transcript từ parquet",
              all(r["transcript"].startswith("sách nói") for r in negatives
                  if r["source_dataset"] == "InfoRE2"))
        check("uid không trùng", len({r["uid"] for r in wake + negatives}) == len(wake + negatives))

        before = sorted(p.name for p in (output / "parts").iterdir())
        with contextlib.redirect_stdout(io.StringIO()):
            run_prepare(paths, output)
        check("chạy lại: không làm gì thêm, kết quả y hệt",
              sorted(p.name for p in (output / "parts").iterdir()) == before)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run_prepare(paths, output, "--band-probability", "0.30")
            check("chặn: resume với cấu hình khác", False)
        except RuntimeError:
            check("chặn: resume với cấu hình khác", True)


def main() -> int:
    print("=" * 74)
    print("KIỂM TRA V14.2 (chỉ CPU)")
    print("=" * 74)
    v12.install_runtime_imports()
    for test in (test_channel_transforms, test_channel_determinism,
                 test_options_and_guards, test_best_checkpoint, test_protocol_summary,
                 test_schedule_and_patches, test_plan_math, test_channel_audit,
                 test_split_and_guards, test_prepare_end_to_end):
        test()
    print("=" * 74)
    if FAILURES:
        print(f"HỎNG {len(FAILURES)}: " + ", ".join(FAILURES))
        return 1
    print("TẤT CẢ ĐỀU ĐẠT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
