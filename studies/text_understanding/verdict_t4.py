#!/usr/bin/env python
"""Verdict of REGISTRATION_T4.md (MASSIVE en-US test, span F1).

    verdict_t4.py [EMB_DIR]

Reads the registered TM rows (tags T4_*) of results_slot_tm.jsonl and the --test rows of results_slot_lr.jsonl
(the last row per parts), prints the bars, and the span F1 per slot type for the 10 most frequent test slot types
from the saved predictions (pred_slot_tm_TAG.npz, pred_slot_lr_PARTS.npz).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slot_eval as E  # noqa: E402
import slot_feats as SF  # noqa: E402

EMB = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "emb"
rows = [json.loads(x) for x in (HERE / "results_slot_tm.jsonl").read_text().splitlines() if x.strip()]
tm = sorted((r for r in rows if r["tag"].startswith("T4_") and "test_f1" in r), key=lambda r: r["seed"])
lr = {}
for x in (HERE / "results_slot_lr.jsonl").read_text().splitlines():
    if x.strip() and "test_f1" in json.loads(x):
        r = json.loads(x)
        lr[r["parts"]] = r
f = [100 * r["test_f1"] for r in tm]
twin, probe, words = (100 * lr[p]["test_f1"] if p in lr else float("nan") for p in ("W+S+P+N+G", "minilm3", "W"))
print(f"TM seeds {[r['seed'] for r in tm]}: test span F1 {[round(x, 2) for x in f]}, mean {np.mean(f):.2f} "
      f"(val {[r['val_f1'] for r in tm]}, best epochs {[r['best_epoch'] for r in tm]})")
print(f"LR twin W+S+P+N+G {twin:.2f}; frozen MiniLM probe minilm3 {probe:.2f}; LR words only W {words:.2f}")
print(f"H1 (TM > minilm3): {'MET' if np.mean(f) > probe else 'NOT MET'} ({np.mean(f) - probe:+.2f}); "
      f"strong (every seed > minilm3): {'MET' if min(f) > probe else 'NOT MET'}")
print(f"H2 (TM >= 0.90 x LR twin): {'MET' if np.mean(f) >= 0.9 * twin else 'NOT MET'} (ratio {np.mean(f) / twin:.3f})")

z = np.load(EMB / "massive_en__slots_K512.npz")
names = json.loads((EMB / "massive_en__slots_K512.meta.json").read_text())["labels"]
off = z["test__off"]
gold = SF.label_seqs(z["test__y"], off, names)
G = [E.spans(g) for g in gold]
count = {}
for s in G:
    for t, _, _ in s:
        count[t] = count.get(t, 0) + 1
top = sorted(count, key=lambda t: -count[t])[:10]
preds = {f"TM seed {r['seed']}": HERE / f"pred_slot_tm_{r['tag']}.npz" for r in tm}
preds.update({f"LR {p}": HERE / f"pred_slot_lr_{p}.npz" for p in ("W+S+P+N+G", "minilm3")})
print("\nper slot type (test span F1; gold spans):")
print("type".ljust(16) + "".join(k.rjust(14) for k in preds))
P = {k: [E.spans(p) for p in SF.label_seqs(np.load(v)["test"], off, names)] for k, v in preds.items() if v.exists()}
for t in top:
    line = f"{t} ({count[t]})".ljust(16)
    for k in preds:
        if k not in P:
            line += "-".rjust(14)
            continue
        tp = sum(len({s for s in g if s[0] == t} & {s for s in p if s[0] == t}) for g, p in zip(G, P[k]))
        npred = sum(sum(1 for s in p if s[0] == t) for p in P[k])
        prec, rec = (tp / npred if npred else 0.0), tp / count[t]
        line += f"{(2 * prec * rec / (prec + rec) if prec + rec else 0.0) * 100:14.1f}"
    print(line)
