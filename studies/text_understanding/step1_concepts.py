#!/usr/bin/env python
"""Logic-reader step 1a (EXPLORATORY, val only): cut the frozen teacher's word vectors to concept ids, the target a
Tsetlin machine student can learn as a class.

    step1_concepts.py --emb-dir DIR --teacher bert-base --K 256 [--pq M]

Spherical k-means (numpy, seed 0, 25 iterations, start = K random train rows) on the L2-normalized TRAIN word
vectors of massive_en__slots_{teacher}.npz (teacher_slots.py); every train and val word gets its nearest concept
(cosine). --pq M: product quantization instead, M codebooks of K concepts, one per block of dim / M coordinates
(each block L2-normalized on its own). Writes DIR/massive_en__slots_{teacher}_K{K}[_pq{M}].npz with train_c and
val_c (int16, [words, M]) and the centroids, and prints how static the concepts are: the share of train words whose
concept is the majority concept of their word type, and the val accuracy of that lookup (majority concept of the
word type in train; a type unseen in train gets the overall majority) per codebook. Never reads the test split.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir", type=Path, required=True)
ap.add_argument("--teacher", default="bert-base")
ap.add_argument("--K", type=int, default=256)
ap.add_argument("--pq", type=int, default=1)
ap.add_argument("--iters", type=int, default=25)
args = ap.parse_args()
t0 = time.time()
z = np.load(args.emb_dir / "massive_en__slots_K512.npz")
T = np.load(args.emb_dir / f"massive_en__slots_{args.teacher}.npz")
Vtr, Vva = T["train"].astype(np.float32), T["val"].astype(np.float32)
wtr, wva = z["train__wid"].astype(np.int64), z["val__wid"].astype(np.int64)
assert len(Vtr) == len(wtr) and len(Vva) == len(wva)
d = Vtr.shape[1]
assert d % args.pq == 0
blk = d // args.pq


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def kmeans(X, K, iters, rng):
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        a = np.argmax(X @ C.T, 1)
        S = np.zeros_like(C)
        np.add.at(S, a, X)
        cnt = np.bincount(a, minlength=K)
        empty = cnt == 0
        S[empty] = X[rng.choice(len(X), int(empty.sum()), replace=False)]
        C = unit(S)
    return C


rng = np.random.default_rng(0)
ctr, cva, cents, rep = [], [], [], []
for m in range(args.pq):
    Xtr, Xva = unit(Vtr[:, m * blk : (m + 1) * blk]), unit(Vva[:, m * blk : (m + 1) * blk])
    C = kmeans(Xtr, args.K, args.iters, rng)
    a_tr, a_va = np.argmax(Xtr @ C.T, 1), np.argmax(Xva @ C.T, 1)
    n_w = int(max(wtr.max(), wva.max())) + 1
    tab = np.zeros((n_w, args.K), np.int64)
    np.add.at(tab, (wtr, a_tr), 1)
    maj = tab.argmax(1)
    glob = int(np.bincount(a_tr, minlength=args.K).argmax())
    look_va = np.where((wva >= 0) & (tab.sum(1)[np.maximum(wva, 0)] > 0), maj[np.maximum(wva, 0)], glob)
    rep.append({"codebook": m, "train_share_type_majority": round(float((maj[wtr] == a_tr).mean()), 4),
                "val_lookup_acc": round(float((look_va == a_va).mean()), 4),
                "concepts_used_train": int((np.bincount(a_tr, minlength=args.K) > 0).sum())})
    ctr.append(a_tr)
    cva.append(a_va)
    cents.append(C.astype(np.float16))
name = f"{args.teacher}_K{args.K}" + (f"_pq{args.pq}" if args.pq > 1 else "")
out = args.emb_dir / f"massive_en__slots_{name}.npz"
np.savez(out, train_c=np.stack(ctr, 1).astype(np.int16), val_c=np.stack(cva, 1).astype(np.int16),
         centroids=np.stack(cents), K=args.K, pq=args.pq)
print(json.dumps({"file": out.name, "teacher": args.teacher, "K": args.K, "pq": args.pq, "codebooks": rep,
                  "seconds": round(time.time() - t0, 1)}))
