#!/usr/bin/env python
"""Where the TM reads the C4 bits worse than LR on MultiWOZ dev (dev only; CPU; read only; no rule; the text-stack
map's P5). Added 2026-10-20.

    e77_lr_vs_tm.py

The TM is e77b's tuned C4 agent (e77_mwoz_Bc4_c16000_s8, seed 1): its dev decisions are rebuilt on the CPU (C4 view B
under dl.count_text.patch; bits and accuracy checked equal to the row's, as scripts/e77_breakdown.py does). LR is e77a's
dev run on the same bits (results/explore/dryrun/dry_e77mwoz_lr_mem_c4, C 0.001; its predictions are saved). Reported:
both right, only LR, only TM; accuracy of each by the gold label's train frequency (20-49, 50-199, 200-999, 1000+
examples) and by first against later actions; the labels where LR gains most. Writes
results/explore/e77_lr_vs_tm_mwoz.json (refuses to overwrite).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import count_text, cpu_tm_fast  # noqa: E402

TAG = "e77_mwoz_Bc4_c16000_s8"
OUT = ROOT / "results" / "explore" / "e77_lr_vs_tm_mwoz.json"
BANDS = [(20, 50), (50, 200), (200, 1000), (1000, 10**9)]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    row = [d for d in map(json.loads, (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines())
           if d.get("arm") == "tm" and d.get("tag") == TAG]
    assert len(row) == 1
    row = row[0]
    count_text.patch("c4words-S1000-v1")
    from dl.abcd import ABCDView, load_examples

    tr, blist = load_examples("train", "multiwoz")
    dv, _ = load_examples("dev", "multiwoz")
    E = np.load(ROOT / "data" / "emb" / "multiwoz__none.npz")
    emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=tuple(row["parts"]), lex_k=1000, seed=0).fit(tr, emb["train"], blist)
    X = view.transform(dv, emb["dev"])
    assert hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest() == row["bits_sha256"]["dev"], "dev bits differ"
    z = np.load(ROOT / "results" / "explore" / "abcd_models" / f"{TAG}_seed1.npz")
    tm = cpu_tm_fast.class_sums(cpu_tm_fast.pack(z["ta"], z["w"]), X).argmax(1)
    y = np.array([r["y"] for r in dv])
    assert float((tm == y).mean()) == row["acc"], "TM accuracy differs from the row"
    L = np.load(ROOT / "results" / "explore" / "dryrun" / "dry_e77mwoz_lr_mem_c4" / "seed0.npz")
    assert np.array_equal(L["y"], y), "LR rows are not the dev rows"
    lr = L["pred"]
    ntr = Counter(r["y"] for r in tr)
    first = np.array([r["k"] == 0 for r in dv])
    ok_t, ok_l = tm == y, lr == y
    out = {"what": "e77: TM against LR on the same C4 bits, MultiWOZ dev (no rule)", "tm": TAG, "lr": "dry_e77mwoz_lr_mem_c4 (C 0.001)",
           "acc_tm": float(ok_t.mean()), "acc_lr": float(ok_l.mean()), "both_right": float((ok_t & ok_l).mean()),
           "only_lr": float((ok_l & ~ok_t).mean()), "only_tm": float((ok_t & ~ok_l).mean()),
           "first": {"tm": float(ok_t[first].mean()), "lr": float(ok_l[first].mean())}, "later": {"tm": float(ok_t[~first].mean()), "lr": float(ok_l[~first].mean())},
           "by_train_frequency": [], "lr_gain_by_label": []}
    f = np.array([ntr[g] for g in y])
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out["by_train_frequency"].append({"train_examples": f"{lo}-{hi - 1 if hi < 10**9 else 'max'}", "n_dev": int(m.sum()),
                                          "n_labels": len({int(g) for g in y[m]}), "tm": float(ok_t[m].mean()) if m.any() else None,
                                          "lr": float(ok_l[m].mean()) if m.any() else None})
    gains = []
    for c in sorted(set(y.tolist())):
        m = y == c
        gains.append((int(m.sum()) * (ok_l[m].mean() - ok_t[m].mean()), blist[c], int(m.sum()), float(ok_t[m].mean()), float(ok_l[m].mean())))
    out["lr_gain_by_label"] = [{"label": b, "n": n, "tm": round(t, 3), "lr": round(l, 3), "turns_gained": round(g, 1)} for g, b, n, t, l in sorted(gains, reverse=True)[:15]]
    # how often each predicts a label never right / the share of predictions on the 20 most frequent labels
    top20 = {c for c, _ in ntr.most_common(20)}
    out["share_predictions_top20_labels"] = {"tm": float(np.isin(tm, list(top20)).mean()), "lr": float(np.isin(lr, list(top20)).mean()),
                                             "gold": float(np.isin(y, list(top20)).mean())}
    out["n_distinct_labels_predicted"] = {"tm": int(len(set(tm.tolist()))), "lr": int(len(set(lr.tolist()))), "gold": int(len(set(y.tolist())))}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "lr_gain_by_label"}, indent=1))
    for d in out["lr_gain_by_label"][:10]:
        print(d)


if __name__ == "__main__":
    main()
