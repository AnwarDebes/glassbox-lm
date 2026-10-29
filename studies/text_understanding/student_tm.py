#!/usr/bin/env python
"""Logic-reader step 1b (EXPLORATORY, val only): a Tsetlin machine student that reads the per-word bits of test 4
(no transformer) and predicts the frozen teacher's concept id of each word (step1_concepts.py), so that BERT's
context can be recovered without BERT at run time.

    student_tm.py --target bert-base_K256 --fold 0|1|all --seed 1 --tag TAG [--model tm|lr] [--smoke]

Cross-fitting, so that the slot model downstream never sees a student's guess on words that student trained on:
the train utterances are split in two halves (numpy default_rng(12345) permutation of the utterance index, parity);
--fold 0 trains on the words of half 1 and predicts half 0, --fold 1 the reverse, --fold all trains on every train
word and predicts val. Every run picks its epoch on the val concept accuracy (val teacher concepts) and also
predicts val. --model tm: the decision-layer CoTM (dl/tm.py, read only; settings below), trained by its own batch
rule (CoTM._step) as slot_tm.py; --model lr: the multinomial logistic-regression twin of slot_lr.py (Adam,
mini-batches of 2,048, wd 1e-5, CPU) on the same bits, one fit of --lr-epochs epochs (no selection). Writes
DIR/student_{TAG}_f{FOLD}.npz (pred_idx, pred: the predicted words and their concepts; val_pred) and one line in
results_student.jsonl (val accuracy curve, the lookup bar of step1_concepts.py on the same words). Never reads the
test split.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import slot_feats as SF  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--codebook", type=int, default=0)
    ap.add_argument("--fold", required=True, choices=("0", "1", "all"))
    ap.add_argument("--model", default="tm", choices=("tm", "lr"))
    ap.add_argument("--parts", default="W+S+P+N+G")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr-epochs", type=int, default=12)
    ap.add_argument("--mem", type=float, default=0.15)
    ap.add_argument("--emb-dir", type=Path, default=HERE / "emb")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 4,000 training words, nothing written")
    args = ap.parse_args()
    t0 = time.time()
    z = np.load(args.emb_dir / "massive_en__slots_K512.npz")
    n_sym, n_w = SF.load(z)
    q = np.load(args.emb_dir / f"massive_en__slots_{args.target}.npz")
    K = int(q["K"])
    c_tr, c_va = q["train_c"][:, args.codebook].astype(np.int64), q["val_c"][:, args.codebook].astype(np.int64)
    Xtr_all, _, off_tr = SF.build(z, "train", args.parts, n_sym, n_w)
    Xva, _, _ = SF.build(z, "val", args.parts, n_sym, n_w)
    assert len(Xtr_all) == len(c_tr) and len(Xva) == len(c_va)
    n_utt = len(off_tr) - 1
    utt = np.repeat(np.arange(n_utt), np.diff(off_tr))
    half = (np.random.default_rng(12345).permutation(n_utt) % 2)[utt]
    if args.fold == "all":
        tr_idx, pr_idx = np.arange(len(c_tr)), np.zeros(0, np.int64)
    else:
        f = int(args.fold)
        tr_idx, pr_idx = np.flatnonzero(half != f), np.flatnonzero(half == f)
    if args.smoke:
        tr_idx = tr_idx[:4000]
    Xtr, ytr = Xtr_all[tr_idx], c_tr[tr_idx]
    # the lookup bar on the predicted words: majority train concept of the word type, from the training words only
    wid_tr, wid_va = z["train__wid"].astype(np.int64), z["val__wid"].astype(np.int64)
    tab = np.zeros((n_w, K), np.int64)
    np.add.at(tab, (wid_tr[tr_idx], ytr), 1)
    glob = int(np.bincount(ytr, minlength=K).argmax())

    def lookup(w):
        ok = (w >= 0) & (tab.sum(1)[np.maximum(w, 0)] > 0)
        return np.where(ok, tab.argmax(1)[np.maximum(w, 0)], glob)

    look_va = float((lookup(wid_va) == c_va).mean())
    look_pr = float((lookup(wid_tr[pr_idx]) == c_tr[pr_idx]).mean()) if len(pr_idx) else None
    curve, best_epoch = [], None
    t1 = time.time()
    if args.model == "tm":
        torch.cuda.set_per_process_memory_fraction(args.mem)
        cfg = TMConfig(n_clauses=args.clauses, T=args.T, s=args.s, epochs=args.epochs, batch_size=args.batch,
                       neg_sampling="hardest", negation=True, max_literals=0, budget_mode="exact",
                       budget_select="threshold", max_negated=0, clause_drop=0.0, half=True, seed=args.seed, eval_every=2)
        tm = make_tm(K, Xtr.shape[1], cfg, "half")

        def predict(X, bs=4096):
            return np.concatenate([tm.class_sums(torch.as_tensor(X[a : a + bs], device="cuda")).argmax(1).cpu().numpy()
                                   for a in range(0, len(X), bs)])

        gen = torch.Generator()
        gen.manual_seed(args.seed)
        epochs = 2 if args.smoke else args.epochs
        best, state, stale = -1.0, None, 0
        for ep in range(epochs):
            perm = torch.randperm(len(ytr), generator=gen).numpy()
            for a in range(0, len(ytr), args.batch):
                idx = perm[a : a + args.batch]
                tm._step(torch.as_tensor(Xtr[idx], device="cuda"), torch.as_tensor(ytr[idx], device="cuda"))
            if ep % 2 == 1 or ep == epochs - 1:
                acc = float((predict(Xva) == c_va).mean())
                curve.append([ep + 1, round(acc, 4)])
                print(f"epoch {ep + 1} val concept acc {acc:.4f} (lookup {look_va:.4f}) ({time.time() - t1:.0f}s)", flush=True)
                if acc > best:
                    best, state, stale = acc, (tm.ta.clone(), tm.w.clone(), ep + 1), 0
                else:
                    stale += 1
                if args.patience and stale >= args.patience:
                    break
        tm.ta, tm.w, best_epoch = state
        pv, pp = predict(Xva), (predict(Xtr_all[pr_idx]) if len(pr_idx) else np.zeros(0, np.int64))
        extra = {"cfg": {"clauses": args.clauses, "T": args.T, "s": args.s, "batch": args.batch, "seed": args.seed}}
    else:
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        lin = torch.nn.Linear(Xtr.shape[1], K)
        opt = torch.optim.Adam(lin.parameters(), lr=2e-3, weight_decay=1e-5)
        yt = torch.as_tensor(ytr)
        for ep in range(2 if args.smoke else args.lr_epochs):
            perm = torch.randperm(len(yt))
            for a in range(0, len(yt), 2048):
                idx = perm[a : a + 2048]
                loss = torch.nn.functional.cross_entropy(lin(torch.as_tensor(Xtr[idx.numpy()]).float()), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        def predict(X):
            with torch.no_grad():
                return np.concatenate([lin(torch.as_tensor(X[a : a + 8192]).float()).argmax(1).numpy()
                                       for a in range(0, len(X), 8192)])

        pv, pp = predict(Xva), (predict(Xtr_all[pr_idx]) if len(pr_idx) else np.zeros(0, np.int64))
        best_epoch = 2 if args.smoke else args.lr_epochs
        curve.append([best_epoch, round(float((pv == c_va).mean()), 4)])
        extra = {"cfg": {"lr": 2e-3, "wd": 1e-5, "epochs": best_epoch, "seed": args.seed}}
    rec = {"step": "logic reader step 1b (exploratory, val only)", "tag": args.tag, "model": args.model,
           "target": args.target, "codebook": args.codebook, "K": K, "fold": args.fold, "parts": args.parts,
           "n_train_words": int(len(ytr)), "n_pred_words": int(len(pr_idx)),
           "val_acc": round(float((pv == c_va).mean()), 4), "val_lookup_acc": round(look_va, 4),
           "pred_acc": round(float((pp == c_tr[pr_idx]).mean()), 4) if len(pr_idx) else None,
           "pred_lookup_acc": round(look_pr, 4) if look_pr is not None else None,
           "best_epoch": best_epoch, "curve": curve, **extra,
           "train_s": round(time.time() - t1, 1), "total_s": round(time.time() - t0, 1)}
    print(json.dumps(rec))
    if args.smoke:
        return
    np.savez(args.emb_dir / f"student_{args.tag}_f{args.fold}.npz", pred_idx=pr_idx, pred=pp.astype(np.int16),
             val_pred=pv.astype(np.int16), K=K)
    with open(HERE / "results_student.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
