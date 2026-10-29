#!/usr/bin/env python
"""Logic-reader step 0 (EXPLORATORY, val only; no test path): does a transformer's contextual word vector add
anything to the per-word bits of test 4 for slot filling? If not, a Tsetlin machine taught to copy that teacher
word by word (idea 1, the logic reader) cannot gain on this task.

    headroom_lr.py --emb-dir DIR [--bits W+S+P+N+G|none] [--extra minilm3|minilm|none] [--teacher minilm|bge-base|bert-base]
                   [--max-train N]

The logistic-regression twin of slot_lr.py (same Adam, mini-batches of 2,048 words, 12 epochs, L2 weight decay
chosen on val span F1 from 0, 1e-5, 1e-4, seed 0) on the bits of slot_feats.py, with the frozen all-MiniLM-L6-v2
word vectors of minilm_slots.py (minilm3: also the previous and next word's vectors) appended as dense columns,
joined per mini-batch so the uint8 bits are never copied to float as a whole. --extra none gives the bits alone
in the same harness. Step 0b: --teacher bge-base|bert-base reads massive_en__slots_{teacher}.npz (teacher_slots.py,
train and val only) in place of the MiniLM file (--extra minilm3 / minilm then mean: that teacher with / without the
neighbour words); --bits none gives the teacher alone. Step 0c: --static replaces every word's teacher vector by the
mean of the teacher vectors of that word type over the TRAIN words (zeros for a word type unseen in train): the
static part of what the teacher knows, a lookup table. Step 1a: --onehot NAME uses the one-hot concept ids of
massive_en__slots_{NAME}.npz (step1_concepts.py; train_c / val_c) as the dense block, in place of the vectors;
--signbits cuts every teacher coordinate to one bit, [v > its train mean] (768 bits per word for bert-base);
with --pca D the bits are the signs of the top D principal components of the centred train vectors instead.
Appends one line per run to results_headroom.jsonl. Never reads the test split.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slot_eval as E  # noqa: E402
import slot_feats as SF  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir", type=Path, required=True)
ap.add_argument("--bits", default="W+S+P+N+G")
ap.add_argument("--extra", default="minilm3", choices=("minilm3", "minilm", "none"))
ap.add_argument("--wd", default="0,1e-5,1e-4")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--teacher", default="minilm", choices=("minilm", "bge-base", "bert-base"))
ap.add_argument("--static", action="store_true", help="step 0c: per word type means of the train teacher vectors")
ap.add_argument("--onehot", default=None, help="step 1a: one-hot concept ids from massive_en__slots_{NAME}.npz")
ap.add_argument("--signbits", action="store_true", help="step 1a: one bit per teacher coordinate, [v > train mean]")
ap.add_argument("--pca", type=int, default=0, help="with --signbits: signs of the top D principal components")
ap.add_argument("--max-train", type=int, default=0, help="smoke runs only: the first N training words")
args = ap.parse_args()
torch.set_num_threads(2)
t0 = time.time()
z = np.load(args.emb_dir / "massive_en__slots_K512.npz")
meta = json.loads((args.emb_dir / "massive_en__slots_K512.meta.json").read_text())
names = meta["labels"]
n_sym, n_w = SF.load(z)


def dense(sp):
    if args.extra == "none":
        return None
    if args.onehot:
        q = np.load(args.emb_dir / f"massive_en__slots_{args.onehot}.npz")
        c, K = q[f"{sp}_c"].astype(np.int64), int(q["K"])
        V = np.zeros((len(c), c.shape[1] * K), np.float32)
        V[np.repeat(np.arange(len(c)), c.shape[1]), (c + K * np.arange(c.shape[1])[None, :]).ravel()] = 1.0
    else:
        V = np.load(args.emb_dir / f"massive_en__slots_{args.teacher}.npz")[sp].astype(np.float32)
    if args.signbits and args.pca:
        V = (((V - TRAIN_MEAN[None, :]) @ PCS) > 0).astype(np.float32)
    elif args.signbits:
        V = (V > TRAIN_MEAN[None, :]).astype(np.float32)
    if args.static:
        w = z[f"{sp}__wid"].astype(np.int64)
        V = np.where((w >= 0)[:, None], TYPE_MEAN[np.maximum(w, 0)], 0.0).astype(np.float32)
    if args.extra == "minilm3":
        off = z[f"{sp}__off"]
        sent = np.repeat(np.arange(len(off) - 1), np.diff(off))
        pos = np.arange(len(V)) - off[sent]
        length = np.diff(off)[sent]
        prev, nxt = np.zeros_like(V), np.zeros_like(V)
        ok = pos > 0
        prev[ok] = V[np.flatnonzero(ok) - 1]
        ok = pos < length - 1
        nxt[ok] = V[np.flatnonzero(ok) + 1]
        V = np.concatenate([prev, V, nxt], 1)
    return V


def bits(sp):
    if args.bits == "none":
        off = z[f"{sp}__off"]
        return np.zeros((int(off[-1]), 0), np.uint8), z[f"{sp}__y"], off
    return SF.build(z, sp, args.bits, n_sym, n_w)


assert not (args.bits == "none" and args.extra == "none"), "nothing to learn from"
TYPE_MEAN = None
TRAIN_MEAN = (np.load(args.emb_dir / f"massive_en__slots_{args.teacher}.npz")["train"].astype(np.float64).mean(0)
              if args.signbits else None)
PCS = None
if args.signbits and args.pca:
    _V = np.load(args.emb_dir / f"massive_en__slots_{args.teacher}.npz")["train"].astype(np.float64) - TRAIN_MEAN[None, :]
    PCS = np.linalg.svd(_V, full_matrices=False)[2][: args.pca].T.astype(np.float32)  # [dim, D], train only
    TRAIN_MEAN = TRAIN_MEAN.astype(np.float32)
    del _V
if args.static:
    assert args.extra != "none"
    _V = np.load(args.emb_dir / f"massive_en__slots_{args.teacher}.npz")["train"].astype(np.float64)
    _w = z["train__wid"].astype(np.int64)
    assert (_w >= 0).all(), "every train word has a word id"
    _S, _c = np.zeros((n_w, _V.shape[1])), np.zeros(n_w)
    np.add.at(_S, _w, _V)
    np.add.at(_c, _w, 1.0)
    TYPE_MEAN = (_S / np.maximum(_c, 1.0)[:, None]).astype(np.float32)
    del _V, _S
Btr, ytr, _ = bits("train")
Dtr = dense("train")
if args.max_train:
    Btr, ytr = Btr[: args.max_train], ytr[: args.max_train]
    Dtr = None if Dtr is None else Dtr[: args.max_train]
Bva, yva, offva = bits("val")
Dva = dense("val")
C = int(meta["n_train_labels"])
gold_va = SF.label_seqs(yva, offva, names)
n_in = Btr.shape[1] + (0 if Dtr is None else Dtr.shape[1])


def rows(B, D, idx):
    xb = torch.as_tensor(B[idx]).float()
    return xb if D is None else torch.cat([xb, torch.as_tensor(D[idx])], 1)


def fit(wd):
    torch.manual_seed(0)
    lin = torch.nn.Linear(n_in, C)
    opt = torch.optim.Adam(lin.parameters(), lr=2e-3, weight_decay=wd)
    yt = torch.as_tensor(ytr)
    for ep in range(args.epochs):
        perm = torch.randperm(len(yt))
        for a in range(0, len(yt), 2048):
            idx = perm[a : a + 2048].numpy()
            loss = torch.nn.functional.cross_entropy(lin(rows(Btr, Dtr, idx)), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return lin


def predict(lin, B, D):
    with torch.no_grad():
        return np.concatenate([lin(rows(B, D, np.arange(a, min(a + 8192, len(B))))).argmax(1).numpy()
                               for a in range(0, len(B), 8192)])


best = None
for wd in [float(x) for x in args.wd.split(",")]:
    lin = fit(wd)
    pv = predict(lin, Bva, Dva)
    f = E.span_f1(gold_va, SF.label_seqs(pv, offva, names))
    print(f"wd {wd}: val span F1 {f['f1']:.4f} (P {f['precision']:.4f} R {f['recall']:.4f}), word acc {(pv == yva).mean():.4f}", flush=True)
    if best is None or f["f1"] > best[0]["f1"]:
        best = (f, wd)
rec = {"step": "logic reader step 0 (exploratory, val only)", "bits": args.bits, "extra": args.extra, "teacher": args.teacher,
       "static": args.static, "onehot": args.onehot, "signbits": args.signbits, "pca": args.pca,
       "n_features": int(n_in), "n_train": int(len(ytr)), "wd": best[1], "val_f1": round(best[0]["f1"], 4),
       "val_precision": round(best[0]["precision"], 4), "val_recall": round(best[0]["recall"], 4),
       "total_s": round(time.time() - t0, 1)}
if not args.max_train:
    with open(HERE / "results_headroom.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
print(json.dumps(rec))
