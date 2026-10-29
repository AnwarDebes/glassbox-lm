#!/usr/bin/env python
"""Test 4 (LOG.md): a Tsetlin machine slot filler for MASSIVE en-US, one example per word.

    slot_tm.py --parts S+P+N+W --seed 1 --emb-dir DIR --tag TAG [--epochs 40 --patience 0 --batch 512] [--test] [--smoke]

Input: the per-word bits of slot_feats.py; output: the word's slot label (56 classes: "Other" and 55 slot types).
The machine is the decision-layer CoTM (dl/tm.py, read only; 4,000 clauses, T 4000, s 8, hardest negative,
negation, init half), trained by its own batch rule (CoTM._step) in this file's loop so that the epoch can be
chosen on val SPAN F1 (slot_eval.py, the MASSIVE convention) instead of word accuracy (77.5% of training words are
"Other"). The best-val state is restored and val scored again as a check; --test scores the test split once and saves the
val and test predictions (pred_slot_tm_TAG.npz).
The process is capped at --mem of the GPU (shared card). Appends one line to results_slot_tm.jsonl.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import slot_eval as E  # noqa: E402
import slot_feats as SF  # noqa: E402
from dl.explain import top_rules  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402


def predict(tm, X, bs=4096):
    out = []
    for a in range(0, len(X), bs):
        out.append(tm.class_sums(torch.as_tensor(X[a : a + bs], device="cuda")).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=0, help="evaluations without a new best val F1 before stopping (0: never)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--other-keep", type=float, default=1.0, help="share of the 'Other' words used in each epoch (fresh draw per epoch)")
    ap.add_argument("--mem", type=float, default=0.12)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on the first 4,000 training words, nothing written")
    ap.add_argument("--rules", type=int, default=0, help="write this many top rules per slot type to rules_slot.txt")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.mem)
    t0 = time.time()
    z = np.load(args.emb_dir / "massive_en__slots_K512.npz")
    meta = json.loads((args.emb_dir / "massive_en__slots_K512.meta.json").read_text())
    names = meta["labels"]
    n_sym, n_w = SF.load(z)
    Xtr, ytr, _ = SF.build(z, "train", args.parts, n_sym, n_w)
    Xva, yva, offva = SF.build(z, "val", args.parts, n_sym, n_w)
    if args.smoke:
        Xtr, ytr = Xtr[:4000], ytr[:4000].copy()
        ytr[-1] = meta["n_train_labels"] - 1
    gold_va = SF.label_seqs(yva, offva, names)
    C = int(meta["n_train_labels"])
    cfg = TMConfig(n_clauses=args.clauses, T=args.T, s=args.s, epochs=args.epochs, batch_size=args.batch,
                   neg_sampling="hardest", negation=True, max_literals=0, budget_mode="exact", budget_select="threshold",
                   max_negated=0, clause_drop=0.0, half=True, seed=args.seed, eval_every=2)
    tm = make_tm(C, Xtr.shape[1], cfg, "half")
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    epochs = 2 if args.smoke else args.epochs
    best, state, stale, curve, stopped = -1.0, None, 0, [], epochs
    t1 = time.time()
    for ep in range(epochs):
        if args.other_keep < 1.0:  # every slot word, and a fresh random share of the "Other" words
            keep = (ytr != 0) | (torch.rand(len(ytr), generator=gen).numpy() < args.other_keep)
            pool = np.flatnonzero(keep)
            perm = pool[torch.randperm(len(pool), generator=gen).numpy()]
        else:
            perm = torch.randperm(len(ytr), generator=gen).numpy()
        for a in range(0, len(ytr), args.batch):
            idx = perm[a : a + args.batch]
            tm._step(torch.as_tensor(Xtr[idx], device="cuda"), torch.as_tensor(ytr[idx], device="cuda"))
        if ep % 2 == 1 or ep == epochs - 1:
            pv = predict(tm, Xva)
            f = E.span_f1(gold_va, SF.label_seqs(pv, offva, names))
            curve.append([ep + 1, round(f["f1"], 4), round(float((pv == yva).mean()), 4)])
            print(f"epoch {ep + 1} val span F1 {f['f1']:.4f} word acc {(pv == yva).mean():.4f} ({time.time() - t1:.0f}s)", flush=True)
            if f["f1"] > best:
                best, state, stale = f["f1"], (tm.ta.clone(), tm.w.clone(), ep + 1), 0
            else:
                stale += 1
            if args.patience and stale >= args.patience:
                stopped = ep + 1
                break
    train_s = time.time() - t1
    tm.ta, tm.w, best_epoch = state
    pv = predict(tm, Xva)
    fv = E.span_f1(gold_va, SF.label_seqs(pv, offva, names))
    rec = {"tag": args.tag, "parts": args.parts, "seed": args.seed, "n_features": int(Xtr.shape[1]), "cfg": asdict(cfg),
           "val_f1": round(best, 4), "val_f1_check": round(fv["f1"], 4), "val_precision": round(fv["precision"], 4),
           "val_recall": round(fv["recall"], 4), "best_epoch": best_epoch, "patience": args.patience, "stopped_epoch": stopped,
           "other_keep": args.other_keep, "mem": args.mem,
           "curve": curve}
    if args.test:
        Xte, yte, offte = SF.build(z, "test", args.parts, n_sym, n_w)
        pt = predict(tm, Xte)
        ft = E.span_f1(SF.label_seqs(yte, offte, names), SF.label_seqs(pt, offte, names))
        np.savez(HERE / f"pred_slot_tm_{args.tag}.npz", val=pv.astype(np.int16), test=pt.astype(np.int16))
        rec.update({"test_f1": round(ft["f1"], 4), "test_precision": round(ft["precision"], 4),
                    "test_recall": round(ft["recall"], 4), "test_spans": ft["gold_spans"]})
    rec["train_s"], rec["total_s"] = round(train_s, 1), round(time.time() - t0, 1)
    if args.smoke:
        print(json.dumps({"smoke": True, **{k: rec[k] for k in ("parts", "n_features", "val_f1", "val_f1_check", "train_s")}}))
        return
    with open(HERE / "results_slot_tm.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if args.rules:
        g = np.load(args.emb_dir / "massive_en__graph_K512.npz")
        gm = json.loads((args.emb_dir / "massive_en__graph_K512.meta.json").read_text())
        bit_names = SF.names(args.parts, g, gm, meta["vocab"])
        assert len(bit_names) == Xtr.shape[1]
        show = [names.index(c) for c in ("time", "date", "place_name", "person", "alarm_type") if c in names]
        lines = top_rules(tm.ta.cpu().numpy(), tm.w.cpu().numpy(), bit_names, names, n_states=cfg.n_states,
                          per_class=args.rules, classes=show)
        with open(HERE / "rules_slot.txt", "a") as fh:
            fh.write(f"=== {args.tag} {args.parts} seed {args.seed} val F1 {best:.4f}\n" + "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "curve")}))


if __name__ == "__main__":
    main()
