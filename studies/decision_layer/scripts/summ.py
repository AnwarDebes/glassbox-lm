#!/usr/bin/env python
"""Print the exploratory TM runs (results/explore/tm.jsonl) as one line each; optional tag prefix filter."""
import json, sys
pref = sys.argv[1] if len(sys.argv) > 1 else ""
for l in open("results/explore/tm.jsonl"):
    r = json.loads(l)
    if not r["tag"].startswith(pref):
        continue
    s = r["stats"]; c = r["cfg"]
    print(f'{r["tag"]:26s} {r["dataset"][:5]} {r["encoder"]:9s} {r["view"][:30]:30s} bits={r["n_bits"]:5d} N={c["n_clauses"]} s={c["s"]} neg={int(c["negation"])} ml={c["max_literals"]} '
          f'val={r["val_acc"]:.4f} ep={r["best_epoch"]} t={r["train_s"]:6.1f} med={s["median_literals"]} mean={s["mean_literals"]:.1f} max={s["max_literals"]} '
          f'nopos={s.get("no_positive_literal")} oos={r.get("oos_auroc_val")} bitsLR={r.get("bits_logreg_val")}')
