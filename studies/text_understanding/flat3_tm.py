#!/usr/bin/env python
"""Test 3 flat rows (LOG.md): the decision-layer TM on flat bit views, for datasets outside test 2 (massive_en).

    flat3_tm.py --dataset massive_en --view mean|mean+existsT|existsT|minilm --seed 1 --emb-dir DIR --tag TAG [--test]

The views, features and TM settings of confirm_tm.py (imported from it, unchanged; features() refits the
concepts on this dataset's training tokens). minilm reads DIR/{dataset}__minilm.npz (embed_minilm.py).
The epoch is chosen on val; --test scores the test split once, at that epoch. Appends to results_flat3.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import confirm_tm as C  # noqa: E402
import exists_tm as X  # noqa: E402
from dl.boolean import Thermo  # noqa: E402
from dl.data import load  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ds = load(args.dataset)
    if args.view == "minilm":
        z = np.load(args.emb_dir / f"{args.dataset}__minilm.npz")
        th = Thermo(8).fit(z["train"])
        B = {sp: th.transform(z[sp]) for sp in ("train", "val", "test")}
        y = {sp: z[f"y_{sp}"] for sp in ("train", "val", "test")}
    else:
        F, meta = C.features(args.dataset, args.K, args.emb_dir, "cuda")
        for sp in C.ALL:
            if f"{sp}__exists" in F:
                F[f"{sp}__existsT"] = F[f"{sp}__exists"]
        view = X.View(args.view, meta).fit(F)
        B = {sp: view.transform(F, sp) for sp in ("train", "val", "test")}
        y = {sp: F[f"y_{sp}"] for sp in ("train", "val", "test")}
    for sp in ("train", "val", "test"):
        assert (y[sp] == ds.labels[sp]).all()
    cfg = TMConfig(n_clauses=4000, T=4000, s=8.0, epochs=40, batch_size=128, neg_sampling="hardest", negation=True,
                   max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0,
                   half=True, seed=args.seed, eval_every=2)
    tm = make_tm(len(ds.intents), B["train"].shape[1], cfg, "half")
    tm.fit(B["train"], y["train"], B["val"], y["val"])
    rec = {"tag": args.tag, "dataset": args.dataset, "view": args.view, "seed": args.seed, "n_bits": int(B["train"].shape[1]),
           "on_share": round(float(B["train"].mean()), 3), "cfg": asdict(cfg), "val_acc": round(tm.best_val, 4),
           "best_epoch": getattr(tm, "best_epoch", None), "curve": [[h["epoch"], round(h["val_acc"], 4)] for h in tm.history]}
    if args.test:
        pt = tm.predict(torch.as_tensor(B["test"], dtype=torch.uint8, device="cuda")).cpu().numpy()
        rec["test_acc"] = round(float((pt == y["test"]).mean()), 4)
        rec["n_test"] = int(len(pt))
    rec["train_s"], rec["total_s"] = round(tm.train_seconds, 1), round(time.time() - t0, 1)
    with open(HERE / "results_flat3.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "curve")}))


if __name__ == "__main__":
    main()
