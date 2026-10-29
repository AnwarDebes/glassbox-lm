"""Read-only summary of recorded TM run times (train_s) in the decision_layer results files.

Reads results/explore/abcd.jsonl (arm tm) and results/confirm/*/*/seed*.json (arm tm), prints one line per run:
tag/name, dataset, clauses, bits, epochs, train_s, seconds per epoch, best_epoch. Writes nothing outside stdout.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import glob
import json
import sys
from pathlib import Path

ROOT = Path(f"{_REPO_ROOT}/studies/decision_layer")
rows = []
for line in (ROOT / "results/explore/abcd.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("arm") != "tm":
        continue
    cfg = r.get("cfg", {})
    rows.append(("explore", r.get("tag"), r.get("dataset"), cfg.get("n_clauses"), r.get("n_bits"), cfg.get("epochs"),
                 cfg.get("batch_size"), cfg.get("max_literals"), r.get("train_s"), r.get("best_epoch"), r.get("time"),
                 r.get("history", [])))
for p in sorted(glob.glob(str(ROOT / "results/confirm/*/*/seed*.json"))):
    r = json.loads(Path(p).read_text())
    if r.get("arm") != "tm":
        continue
    cfg = r.get("cfg", {})
    rows.append(("confirm", f"{r.get('registration')}/{r.get('name')}/s{r.get('seed')}", r.get("dataset", "?"), cfg.get("n_clauses"),
                 r.get("n_bits"), cfg.get("epochs"), cfg.get("batch_size"), cfg.get("max_literals"), r.get("train_s"),
                 r.get("best_epoch"), r.get("time"), []))
print("src\ttag\tdataset\tclauses\tbits\tepochs\tbatch\tmaxlit\ttrain_s\ts_per_epoch\tbest_epoch\ttime")
tot = 0.0
for src, tag, ds, N, nb, ep, bs, ml, ts, be, tm, hist in rows:
    spe = (ts / ep) if (ts and ep) else None
    tot += ts or 0.0
    print(f"{src}\t{tag}\t{ds}\t{N}\t{nb}\t{ep}\t{bs}\t{ml}\t{ts}\t{spe and round(spe, 2)}\t{be}\t{tm}")
print(f"# rows {len(rows)}; sum train_s {tot:.0f} s = {tot / 3600:.2f} h", file=sys.stderr)
