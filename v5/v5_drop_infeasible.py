"""v5: drop rows whose audio is too short for their CTC target (the trainer's own audit rule,
parakeet_v14_train: _required_frames(reduced_ids) > _encoder_frames(duration)).  Rewrites the v5
manifest in place after saving the dropped rows."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "/media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk")
import parakeet_v12_train as v12  # noqa: E402

v12.install_runtime_imports()
import parakeet_v14_train as v14  # noqa: E402

M = Path("/home/ml4u/datasets/v5_wakeword_channel_manifest.jsonl")
D = Path(__file__).resolve().parent
SPLIT = json.load(open(D / "v5_speaker_splits.json"))
tokenizer, _, old_to_reduced = v12.build_reduced_tokenizer(
    Path("/media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk/checkpoints/parakeet_teacher_tokenizer.model"))
keep, drop = [], []
for line in open(M, encoding="utf-8"):
    r = json.loads(line)
    ids = v12.transcript_target_ids(v12.normalize_text(r["transcript"]), tokenizer, old_to_reduced)
    bad = bool(ids) and v14._required_frames(ids) > v14._encoder_frames(float(r["duration_seconds"]))
    (drop if bad else keep).append(r)
(D / "v5_dropped_infeasible.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in drop), encoding="utf-8")
M.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
by = collections.Counter((r["speaker_id"], SPLIT[r["speaker_id"]]) for r in drop)
print("dropped", len(drop), dict(by), "kept", len(keep))
print(sorted({r["original_path"].rsplit("_", 1)[0] for r in drop})[:10])
