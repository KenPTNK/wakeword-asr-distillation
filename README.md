# Efficient Wake-Word Detection for Low-Resource Languages through Speech Recognizer Distillation

Code for the ICIIT 2027 paper by Khac Trung Kien Nguyen\*, Quang Nam Khanh Nguyen\*, Ngoc Minh Pham,
Quang Phuoc Hua, The Hieu Pham and Duc Dung Nguyen (AI Tech Lab, HCMUT, VNU-HCM; \* equal contribution).

A 0.6B-parameter Vietnamese speech recognizer
([nvidia/parakeet-ctc-0.6b-vi](https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi)) is distilled into a
5M-parameter streaming Conformer (Phase 1), which is then fine-tuned on the wake phrase
"năng khiếu ơi" with sub-word CTC targets taken from transcripts (Phase 2). Test phrases are scored
inside continuous audio. At 0.1 false accepts per hour the detector misses 20.5 % of wake phrases;
the DS-CNN, BC-ResNet-8 and openWakeWord baselines miss 57–85 %.

This is the code that produced the paper's numbers, released as it ran. The layout is flat because the
scripts import each other by module name (`import parakeet_v12_train as v12`). Run everything from the
repository root.

## Layout

| Part | Files | What it does |
|---|---|---|
| Model core and Phase 1 | `parakeet_v12_train.py` (model, data loading, CTC/KD losses), `parakeet_v12_*`, `streaming_protocol.py`, `vi_number_norm.py`, `parakeet_v16_phase1_train.py`, `v16_lanes.sh`, `generate_teacher_logits.py`, `compact_teacher_logits.py`, `prepare_3a_sources.py`, `prepare_nvme_audio_3a.py`, `build_splits_3a.py`, `dedup_fingerprint.py`, `run_with_oom_retry.sh` | Streaming Conformer student (5 layers, d=192, chunk 0.64 s, left context 64 frames), teacher top-64 logits, multi-domain corpus split, Phase-1 distillation |
| Phase 2 and streaming evaluation | `parakeet_v14*`, `parakeet_v17_train.py`, `parakeet_v17_context.py`, `train_v14_2.sh`, `train_v17.sh`, `validate_v14_2_training.py`, `build_v17_negatives.py`, `build_oi_negatives.py`, `prepare_v14_*`, `prepare_v17_channel.py`, `incontext_protocol.py`, `evaluate_*.py`, `step0_score_separability.py`, `reselect_in_context.py`, `final_gate.py`, `measure_channel_robustness.py`, `v5/proto/`, `v5/launch_v5.py`, `v5/python_v5.sh` | Word-to-sub-word CTC targets, near-miss negatives, channel copies, checkpoint selection on long audio, in-context scoring, false accepts per hour, sealed deployment gate |
| Baselines | `kws_baseline.py`, `kws_baseline_run.sh`, `v5/v5b_dscnn.sh`, `v5/bcresnet_baseline.py`, `v5/bcresnet_queue.sh`, `v5/oww_baseline.py`, `v5/oww_queue.sh`, `evaluate_baselines.py`, `teacher_textmatch.py`, `v5/v5_teacher*.py` | DS-CNN, BC-ResNet-8, openWakeWord and the unadapted teacher, scored with the same protocol |
| Speaker split, experiment queues, paper tables | `v5/make_split.py`, `v5/*.json`, `augment_newspk.py`, `v5/v5_*manifest.py`, `v5/v5_channel_input.py`, `v5/v5_drop_infeasible.py`, `v5/v5_counts.py`, `v5/v5_queue.sh`, `v5/v5_eval_queue.sh`, `dump_scores.py`, `v5/v5_dump_scores.py`, `v5/v5_longfa*.{py,sh}`, `v5/v5_isolated.py`, `isolated_vs_context.py`, `v5/v5_cascade_text_hk15.py`, `paper/` | Seeded speaker split (10 test / 5 selection / 37 train), the queue that trains every system in the paper, per-clip scores, Tables 1–3, confidence intervals, Fig. 2 |

## Setup

Python 3.11 with a CUDA GPU (the paper used one RTX 4090, 24 GB).

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

## Reproducing the paper

1. **Paths.** The scripts still contain the absolute paths of our server (`/media/ml4u/...`,
   `/home/ml4u/...`). Search for them and point them at your data:
   `grep -rn "/media/ml4u\|/home/ml4u" .`
   Weights & Biases logging needs `WANDB_API_KEY` in the environment; set `WANDB_ENTITY` to your own.
2. **Phase 1 (distillation).** Build the corpus split (`prepare_3a_sources.py`, `build_splits_3a.py`,
   `dedup_fingerprint.py`), dump teacher logits (`generate_teacher_logits.py`, then
   `compact_teacher_logits.py`), then train: `bash v16_lanes.sh` (our encoder is run `trial6`).
3. **Speaker split.** `cd v5 && python make_split.py` (seed 20260927) writes `v5_speaker_splits.json`.
4. **Phase 2, ablations and baselines.** `bash v5/v5_queue.sh` trains every system in Tables 1–2 and
   writes per-clip scores; `bash v5/v5_eval_queue.sh` and `bash v5/v5_longfa_queue.sh` add
   robustness and false accepts on the long negative audio; the openWakeWord and BC-ResNet queues are
   `v5/oww_queue.sh` and `v5/bcresnet_queue.sh`.
5. **Tables and figure.** Copy `logs/v5/scores/` and `logs/v5/longfa/` into `paper/v5/`, then
   `cd paper && python v5_tables.py > v5/v5_stats.json && python v5_ci.py && python make_fig2.py`.
   Before training anything, `python validate_v14_2_training.py` checks the data path on CPU.

## Data

The wake-phrase recordings are not included. The public corpora used in Phase 1 and as negatives are
listed in the paper. The speaker split files hold anonymous speaker numbers only.

## License

MIT, see `LICENSE`. The teacher model keeps its own license (see its model card).

## Citation

```bibtex
@inproceedings{nguyen2027wakeword,
  title     = {Efficient Wake-Word Detection for Low-Resource Languages through Speech Recognizer Distillation},
  author    = {Nguyen, Khac Trung Kien and Nguyen, Quang Nam Khanh and Pham, Ngoc Minh and
               Hua, Quang Phuoc and Pham, The Hieu and Nguyen, Duc Dung},
  booktitle = {Proceedings of the International Conference on Intelligent Information Technology (ICIIT)},
  year      = {2027}
}
```
