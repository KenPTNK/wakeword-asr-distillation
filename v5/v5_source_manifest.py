"""v5: source manifest of the augmented 048-054 clips for prepare_wakeword_v12_16k_shards.py.
Fields follow the WakewordV12 sharded manifest; split = the speaker's v5 role."""
import glob
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, "/media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk")
import parakeet_v12_train as v12  # noqa: E402

AUG = Path("/home/ml4u/datasets/wakeword_newspk_048_054_aug_all")
SPLIT = json.load(open(Path(__file__).with_name("v5_speaker_splits.json")))
OUT = Path(sys.argv[1])
rows = []
for path in sorted(AUG.glob("*/*.wav")):
    parts = path.stem.split("_")
    speaker, text, index, aug = parts[0], parts[1], parts[2], "_".join(parts[3:])
    info = sf.info(str(path))
    assert info.samplerate == 16000 and info.channels == 1, path
    rows.append({
        "uid": f"WakewordV12:newspk/{speaker}/{path.name}",
        "audio_path": str(path), "speaker_id": speaker, "transcript": text,
        "normalized_transcript": v12.normalize_text(text), "utterance_index": index,
        "augmentation": aug, "is_augmented": aug != "orig",
        "is_positive": bool(v12.contains_word_sequence(v12.normalize_text(text))),
        "duration_seconds": info.frames / 16000, "split": SPLIT[speaker],
        "original_path": f"{speaker}/{path.name}", "source_dataset": "WakewordV12",
        "provenance_kind": "newspk_augmented_v5", "google_drive_object_id": None,
        "md5": hashlib.md5(path.read_bytes()).hexdigest(), "channels": 1, "sample_rate": 16000,
        "manifest_version": 1})
OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(len(rows), "rows;", sum(r["is_positive"] for r in rows), "positive")
