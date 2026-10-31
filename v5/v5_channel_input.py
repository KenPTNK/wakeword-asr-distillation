"""v5: inputs for prepare_v14_2_channel_shards.py -- the non-copy rows of the speakers that are
v5 train speakers but were not v14.2 train speakers (so have no channel copies yet)."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
OLD_SPLIT = json.load(open("/media/ml4u/Samsung_T52/ptnk_team/tknk/ctctknk/v14_2_speaker_splits.json"))
NEW_SPLIT = json.load(open(D / "v5_speaker_splits.json"))
NEED = sorted(s for s, r in NEW_SPLIT.items() if r == "train" and OLD_SPLIT.get(s) != "train")
print("speakers needing channel copies:", NEED)
rows = []
for manifest in ("/home/ml4u/datasets/wakeword_v12_16k_sharded/wakeword_v12_16k_sharded_manifest.jsonl",
                 "/home/ml4u/datasets/wakeword_newspk_v5_sharded/wakeword_v12_16k_sharded_manifest.jsonl"):
    root = Path(manifest).parent
    for line in open(manifest, encoding="utf-8"):
        r = json.loads(line)
        if r["speaker_id"] in NEED:
            if not Path(r["audio_path"]).is_absolute():
                r["audio_path"] = str(root / r["audio_path"])
            rows.append(r)
(D / "channel_input_manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
json.dump({s: "train" for s in NEED}, open(D / "channel_input_split.json", "w"), indent=1)
print(len(rows), "rows")
