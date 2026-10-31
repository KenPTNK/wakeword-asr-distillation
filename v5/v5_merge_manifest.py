"""v5 wake-word training manifest: v14.2 channel manifest (originals of 001-047 + copies of the v14.2
train speakers) minus copies of speakers that are not v5 train speakers, plus the 048-054 originals
(wakeword_newspk_v5_sharded) and the channel copies made for the new v5 train speakers
(v5_channel_16k).  Checks: unique uids/locations, copies only for v5 train speakers, every v5 train
speaker has copies, all 52 speakers present."""
import collections
import json
from pathlib import Path

D = Path(__file__).resolve().parent
SPLIT = json.load(open(D / "v5_speaker_splits.json"))
OUT = Path("/home/ml4u/datasets/v5_wakeword_channel_manifest.jsonl")
is_copy = lambda r: "ch:" in (r.get("augmentation") or "")


def rows(path):
    root = Path(path).parent
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not Path(r["audio_path"]).is_absolute():
            r["audio_path"] = str(root / r["audio_path"])
        yield r


out, dropped = [], 0
for r in rows("/home/ml4u/datasets/v14_2_channel_16k/v14_2_wakeword_channel_manifest.jsonl"):
    if is_copy(r) and SPLIT[r["speaker_id"]] != "train":
        dropped += 1
        continue
    out.append(r)
new = list(rows("/home/ml4u/datasets/wakeword_newspk_v5_sharded/wakeword_v12_16k_sharded_manifest.jsonl"))
copies = [r for r in rows("/home/ml4u/datasets/v5_channel_16k/v14_2_wakeword_channel_manifest.jsonl") if is_copy(r)]
out += new + copies
uids = collections.Counter(r["uid"] for r in out)
locs = collections.Counter((r["audio_path"], r.get("audio_row")) for r in out)
assert max(uids.values()) == 1 and max(locs.values()) == 1, "duplicate uid/location"
spk_copies = collections.Counter(r["speaker_id"] for r in out if is_copy(r))
assert all(SPLIT[s] == "train" for s in spk_copies), "copy of a held-out speaker"
assert all(spk_copies[s] > 0 for s, v in SPLIT.items() if v == "train"), "train speaker without copies"
assert set(r["speaker_id"] for r in out) == set(SPLIT)
OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
summary = {"rows": len(out), "dropped_copies_of_heldout": dropped, "new_originals": len(new),
           "new_copies": len(copies),
           "by_role": {role: {"speakers": sum(v == role for v in SPLIT.values()),
                              "rows": sum(SPLIT[r["speaker_id"]] == role for r in out),
                              "positives_non_copy": sum(SPLIT[r["speaker_id"]] == role and r["is_positive"]
                                                        and not is_copy(r) for r in out)}
                       for role in ("train", "validation", "test")}}
json.dump(summary, open(D / "v5_manifest_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
