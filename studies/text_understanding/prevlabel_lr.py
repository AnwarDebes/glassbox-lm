#!/usr/bin/env python
"""Slot-filler context checks with the LR twin (EXPLORATORY, val only; no test path), two cheap questions:

    prevlabel_lr.py --emb-dir DIR [--bits W+S+P+N+G] [--prev]

1. --bits W+S+P+N+P2+N2+G: do wider static windows (words -2 and +2) add to the bits of test 4?
2. --prev: label feedback (the simplest recurrent reader, a maximum-entropy Markov model): the one-hot slot label of
   the previous word is appended; training uses the gold previous label, val is decoded greedily left to right
   with the model's own previous guess (the first word of an utterance gets no previous label).
The LR is the slot_lr.py twin (Adam, mini-batches of 2,048 words, 12 epochs, L2 weight decay chosen on val span
F1 from 0, 1e-5, 1e-4, seed 0). Appends one line per run to results_context_lr.jsonl. Never reads the test split.
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
ap.add_argument("--prev", action="store_true")
ap.add_argument("--wd", default="0,1e-5,1e-4")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--max-train", type=int, default=0, help="smoke runs only")
args = ap.parse_args()
torch.set_num_threads(2)
t0 = time.time()
z = np.load(args.emb_dir / "massive_en__slots_K512.npz")
meta = json.loads((args.emb_dir / "massive_en__slots_K512.meta.json").read_text())
names = meta["labels"]
n_sym, n_w = SF.load(z)
C = int(meta["n_train_labels"])
Btr, ytr, offtr = SF.build(z, "train", args.bits, n_sym, n_w)
Bva, yva, offva = SF.build(z, "val", args.bits, n_sym, n_w)


def prev_index(off, n):
    """Row of the previous word in the same utterance, -1 for an utterance's first word."""
    sent = np.repeat(np.arange(len(off) - 1), np.diff(off))
    pos = np.arange(n) - off[sent]
    return np.where(pos > 0, np.arange(n) - 1, -1), pos


ptr, _ = prev_index(offtr, len(ytr))
pva, posva = prev_index(offva, len(yva))
if args.max_train:
    Btr, ytr, ptr = Btr[: args.max_train], ytr[: args.max_train], ptr[: args.max_train]
gold_va = SF.label_seqs(yva, offva, names)
n_in = Btr.shape[1] + (C if args.prev else 0)


def rows(B, idx, prev_lab):
    xb = torch.as_tensor(B[idx]).float()
    if not args.prev:
        return xb
    oh = torch.zeros(len(idx), C)
    ok = prev_lab >= 0
    oh[torch.as_tensor(np.flatnonzero(ok)), torch.as_tensor(prev_lab[ok])] = 1.0
    return torch.cat([xb, oh], 1)


def fit(wd):
    torch.manual_seed(0)
    lin = torch.nn.Linear(n_in, C)
    opt = torch.optim.Adam(lin.parameters(), lr=2e-3, weight_decay=wd)
    yt = torch.as_tensor(ytr)
    for ep in range(args.epochs):
        perm = torch.randperm(len(yt))
        for a in range(0, len(yt), 2048):
            idx = perm[a : a + 2048].numpy()
            pl = np.where(ptr[idx] >= 0, ytr[np.maximum(ptr[idx], 0)], -1)  # gold previous label (teacher forcing)
            loss = torch.nn.functional.cross_entropy(lin(rows(Btr, idx, pl)), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return lin


def predict(lin):
    out = np.full(len(yva), -1, np.int64)
    with torch.no_grad():
        if not args.prev:
            return np.concatenate([lin(rows(Bva, np.arange(a, min(a + 8192, len(yva))), None)).argmax(1).numpy()
                                   for a in range(0, len(yva), 8192)])
        for p in range(int(posva.max()) + 1):  # greedy, left to right: position p needs the guesses at p - 1
            idx = np.flatnonzero(posva == p)
            pl = np.where(pva[idx] >= 0, out[np.maximum(pva[idx], 0)], -1)
            assert not ((pva[idx] >= 0) & (pl < 0)).any()
            out[idx] = lin(rows(Bva, idx, pl)).argmax(1).numpy()
    return out


best = None
for wd in [float(x) for x in args.wd.split(",")]:
    lin = fit(wd)
    pv = predict(lin)
    f = E.span_f1(gold_va, SF.label_seqs(pv, offva, names))
    print(f"wd {wd}: val span F1 {f['f1']:.4f} (P {f['precision']:.4f} R {f['recall']:.4f}), word acc {(pv == yva).mean():.4f}", flush=True)
    if best is None or f["f1"] > best[0]["f1"]:
        best = (f, wd)
rec = {"step": "slot context check (exploratory, val only)", "bits": args.bits, "prev_label": args.prev,
       "n_features": int(n_in), "n_train": int(len(ytr)), "wd": best[1], "val_f1": round(best[0]["f1"], 4),
       "val_precision": round(best[0]["precision"], 4), "val_recall": round(best[0]["recall"], 4),
       "total_s": round(time.time() - t0, 1)}
if not args.max_train:
    with open(HERE / "results_context_lr.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
print(json.dumps(rec))
