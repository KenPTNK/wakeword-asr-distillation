"""v5 speaker split (27/09/2026): 52 speakers -> 10 test / 5 validation (selection) / 37 train,
stratified by gender and accent (team labels, ctctknk-docs/18 §11-12 and doc 38: 048-054 Southern),
proportional allocation per stratum, seeded draw.  Decided before any v5 training; not tuned."""
import json
import numpy as np

SPEAKERS = [f"{i:03d}" for i in range(1, 55) if i not in (36, 44)]
FEMALE = {"001", "003", "013", "034", "049", "050", "051", "052", "053", "054"}
NON_SOUTHERN = {"001", "014", "015", "016", "017", "022", "026", "028", "029"}
STRATA = {
    "female": sorted(FEMALE),
    "male_non_southern": sorted(NON_SOUTHERN - FEMALE),
    "male_southern": sorted(set(SPEAKERS) - FEMALE - NON_SOUTHERN),
}
ALLOCATION = {"female": (2, 1), "male_non_southern": (2, 1), "male_southern": (6, 3)}  # (test, val)
SEED = 20260927

rng = np.random.default_rng(SEED)
split, detail = {}, {}
for name, members in STRATA.items():
    order = list(rng.permutation(members))
    n_test, n_val = ALLOCATION[name]
    for i, spk in enumerate(order):
        split[spk] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    detail[name] = {r: sorted(s for s in members if split[s] == r) for r in ("test", "validation", "train")}
assert len(split) == 52 and sum(v == "test" for v in split.values()) == 10
assert sum(v == "validation" for v in split.values()) == 5
json.dump(dict(sorted(split.items())), open("v5_speaker_splits.json", "w"), indent=1)
json.dump({"seed": SEED, "strata": STRATA, "allocation": ALLOCATION, "result": detail},
          open("v5_split_design.json", "w"), indent=1)
print(json.dumps(detail, indent=1))
