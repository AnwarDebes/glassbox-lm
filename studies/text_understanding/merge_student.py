#!/usr/bin/env python
"""Logic-reader step 1b: join the cross-fitted student guesses (student_tm.py folds 0 and 1 for the train words,
fold all for val) into DIR/massive_en__slots_student_{TAG}.npz (train_c, val_c, K), the input of
headroom_lr.py --onehot student_{TAG}. Checks that every train word is guessed exactly once.

    merge_student.py --tag TAG [--emb-dir DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--emb-dir", type=Path, default=HERE / "emb")
args = ap.parse_args()
z = np.load(args.emb_dir / "massive_en__slots_K512.npz")
n_tr = int(z["train__off"][-1])
f = {k: np.load(args.emb_dir / f"student_{args.tag}_f{k}.npz") for k in ("0", "1", "all")}
K = int(f["all"]["K"])
assert int(f["0"]["K"]) == K == int(f["1"]["K"])
tr = np.full(n_tr, -1, np.int64)
for k in ("0", "1"):
    idx = f[k]["pred_idx"]
    assert (tr[idx] == -1).all(), "a train word is guessed twice"
    tr[idx] = f[k]["pred"]
assert (tr >= 0).all(), "a train word has no guess"
va = f["all"]["val_pred"].astype(np.int64)
out = args.emb_dir / f"massive_en__slots_student_{args.tag}.npz"
np.savez(out, train_c=tr[:, None].astype(np.int16), val_c=va[:, None].astype(np.int16), K=K, pq=1)
print(json.dumps({"file": out.name, "K": K, "train_words": int(n_tr), "val_words": int(len(va))}))
