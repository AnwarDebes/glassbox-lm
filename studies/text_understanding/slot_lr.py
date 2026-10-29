#!/usr/bin/env python
"""Test 4 logistic-regression twin (exploratory, val only unless --test): multinomial LR on the same per-word bits
as the TM (slot_feats.py), CPU, 2 threads. Adam, mini-batches of 2,048 words, 12 epochs, L2 weight decay chosen on
val span F1 (slot_eval.py). Appends one line per run to results_slot_lr.jsonl; --test also saves the val and test
predictions (pred_slot_lr_PARTS.npz).

    slot_lr.py --parts S+P+N+W --emb-dir DIR [--test]
    slot_lr.py --parts minilm|minilm3 --emb-dir DIR   (frozen MiniLM word vectors, minilm_slots.py; minilm3 adds the
                                                       previous and next word's vectors)
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
ap.add_argument("--parts", required=True)
ap.add_argument("--emb-dir", type=Path, required=True)
ap.add_argument("--wd", default="0,1e-5,1e-4")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
torch.set_num_threads(2)
torch.manual_seed(0)
t0 = time.time()
z = np.load(args.emb_dir / "massive_en__slots_K512.npz")
meta = json.loads((args.emb_dir / "massive_en__slots_K512.meta.json").read_text())
names = meta["labels"]
n_sym, n_w = SF.load(z)
def minilm(sp):
    V = np.load(args.emb_dir / "massive_en__slots_minilm.npz")[sp]
    off = z[f"{sp}__off"]
    if args.parts == "minilm3":
        sent = np.repeat(np.arange(len(off) - 1), np.diff(off))
        pos = np.arange(len(V)) - off[sent]
        length = np.diff(off)[sent]
        prev, nxt = np.zeros_like(V), np.zeros_like(V)
        ok = pos > 0
        prev[ok] = V[np.flatnonzero(ok) - 1]
        ok = pos < length - 1
        nxt[ok] = V[np.flatnonzero(ok) + 1]
        V = np.concatenate([prev, V, nxt], 1)
    return V, z[f"{sp}__y"], off


def feats(sp):
    return minilm(sp) if args.parts.startswith("minilm") else SF.build(z, sp, args.parts, n_sym, n_w)


Xtr, ytr, _ = feats("train")
Xva, yva, offva = feats("val")
C = int(meta["n_train_labels"])
gold_va = SF.label_seqs(yva, offva, names)


def fit(wd):
    torch.manual_seed(0)
    lin = torch.nn.Linear(Xtr.shape[1], C)
    opt = torch.optim.Adam(lin.parameters(), lr=2e-3, weight_decay=wd)
    yt = torch.as_tensor(ytr)
    for ep in range(args.epochs):
        perm = torch.randperm(len(yt))
        for a in range(0, len(yt), 2048):
            idx = perm[a : a + 2048]
            xb = torch.as_tensor(Xtr[idx.numpy()]).float()
            loss = torch.nn.functional.cross_entropy(lin(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return lin


def predict(lin, X):
    with torch.no_grad():
        return np.concatenate([lin(torch.as_tensor(X[a : a + 8192]).float()).argmax(1).numpy() for a in range(0, len(X), 8192)])


best = None
for wd in [float(x) for x in args.wd.split(",")]:
    lin = fit(wd)
    pv = predict(lin, Xva)
    f = E.span_f1(gold_va, SF.label_seqs(pv, offva, names))
    print(f"wd {wd}: val span F1 {f['f1']:.4f} (P {f['precision']:.4f} R {f['recall']:.4f}), word acc {(pv == yva).mean():.4f}", flush=True)
    if best is None or f["f1"] > best[0]["f1"]:
        best = (f, wd, lin)
rec = {"parts": args.parts, "n_features": int(Xtr.shape[1]), "wd": best[1], "val_f1": round(best[0]["f1"], 4),
       "val_precision": round(best[0]["precision"], 4), "val_recall": round(best[0]["recall"], 4)}
if args.test:
    Xte, yte, offte = feats("test")
    pt = predict(best[2], Xte)
    ft = E.span_f1(SF.label_seqs(yte, offte, names), SF.label_seqs(pt, offte, names))
    np.savez(HERE / f"pred_slot_lr_{args.parts}.npz", val=predict(best[2], Xva).astype(np.int16), test=pt.astype(np.int16))
    rec.update({"test_f1": round(ft["f1"], 4), "test_precision": round(ft["precision"], 4), "test_recall": round(ft["recall"], 4)})
rec["total_s"] = round(time.time() - t0, 1)
with open(HERE / "results_slot_lr.jsonl", "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(json.dumps(rec))
