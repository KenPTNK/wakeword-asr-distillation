import json, collections
S = json.load(open("v5/v5_speaker_splits.json"))
rec = collections.defaultdict(dict)
for line in open("/home/ml4u/datasets/v5_wakeword_channel_manifest.jsonl", encoding="utf-8"):
    r = json.loads(line); a = r["augmentation"] or ""
    if "ch:" in a:
        continue
    rec[(r["is_positive"], r["speaker_id"], r["transcript"], str(r["utterance_index"]))][a] = r["duration_seconds"]
out = {}
for role in ("train", "validation", "test"):
    for pos in (True, False):
        keys = [k for k in rec if k[0] == pos and S[k[1]] == role]
        d = [rec[k].get("orig", rec[k].get("vol_low", 0)) for k in keys]
        out[role + ("_pos" if pos else "_neg")] = {"recordings": len(keys), "speakers": len({k[1] for k in keys}), "hours": round(sum(d) / 3600, 3)}
print(json.dumps(out))
for run in ["v5_prop", "v5_prop_s2027", "v5_prop_s2028", "v5_v2"]:
    print(run, json.load(open(f"checkpoints/student_L5_H192_M64_{run}/best_longaudio_state.json"))["key"][:2])
cat = {}
for line in open("/home/ml4u/datasets/sealed_longaudio_v1/manifest.jsonl", encoding="utf-8"):
    row = json.loads(line); cat[row["audio"].split("/")[-1]] = row["category"]
for g in ["logs/v5/final_gate_v5_prop.json", "logs/v5/final_gate_v5_v2.json"]:
    d = json.load(open(g)); c = collections.defaultdict(lambda: [0, 0.0])
    for f in d["files"]:
        c[cat[f["file"]]][0] += f["events"]; c[cat[f["file"]]][1] += f["hours"]
    print(g, {k: (v[0], round(v[1], 2), round(v[0] / v[1], 2)) for k, v in sorted(c.items())})
