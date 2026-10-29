"""Per-step GPU time of recorded TM runs (read-only): ms per minibatch step and ns per (clause x literal) per step (1e6 * ms / (N*m)).

train_s (CoTM.fit wall time, incl. the dev pass every 2 epochs and the upload) / (epochs x steps per epoch); steps per
epoch = ceil(n_train / 128) with n_train from the decision_layer prepare logs: ABCD 29,190 (src/dl/abcd.py doc), STAR 54,913
(logs/star_prepare.out), MultiWOZ 56,581 (logs/e77_prepare.out). Rows without dataset/n_bits are skipped.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(f"{_REPO_ROOT}/studies/decision_layer")
NTR = {"abcd": 29190, "star": 54913, "multiwoz": 56581}
rows = []
for line in (ROOT / "results/explore/abcd.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("arm") != "tm" or r.get("dataset") not in NTR or not r.get("n_bits") or not r.get("train_s"):
        continue
    c = r["cfg"]
    steps = math.ceil(NTR[r["dataset"]] / c["batch_size"]) * c["epochs"]
    Nm = c["n_clauses"] * 2 * r["n_bits"]
    ms = 1000 * r["train_s"] / steps
    h = r.get("history", [])
    d2 = [b["t"] - a["t"] for a, b in zip(h, h[1:])]
    rows.append((r["dataset"], r["tag"], r["seed"], c["n_clauses"], r["n_bits"], c["s"], c.get("max_literals", 0), Nm, r["train_s"], ms,
                 1e6 * ms / Nm, statistics.median(d2) if d2 else None, r["time"]))
print("dataset\ttag\tseed\tclauses\tbits\ts\tmaxlit\tN*m(M)\ttrain_s\tms_per_step\tns_per_elem_step\tmedian_s_per_2ep\ttime")
for x in sorted(rows, key=lambda x: (x[0], x[3], x[5], x[12])):
    print("\t".join(str(round(v, 3)) if isinstance(v, float) else str(v) for v in
                    (x[0], x[1], x[2], x[3], x[4], x[5], x[6], round(x[7] / 1e6, 1), x[8], x[9], x[10], x[11], x[12])))
g = defaultdict(list)
for x in rows:
    g[(x[0], x[3], x[5])].append(x[10])
print("\n# median ns per (clause x literal) per step, by (dataset, clauses, s)")
for k in sorted(g):
    print("#", k, len(g[k]), round(statistics.median(g[k]), 1))
