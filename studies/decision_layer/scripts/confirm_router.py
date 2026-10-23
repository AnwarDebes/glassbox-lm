#!/usr/bin/env python
"""Registered router runs (R1, R2, R4): one (dataset, encoder, view, TM setting) over seeds, test once.

For every seed k (seeds of the registration): the TM (or an ensemble of M machines, seeds
100k..100k+M-1) is trained on train with the best-validation snapshot; its test predictions,
class sums and clause statistics are stored. Logistic regression on the raw embedding (C from the
val grid; deterministic) is fitted once. Calibration: one temperature per model fitted on val,
then ECE (15 bins) and NLL on test. CLINC150: out-of-scope AUROC on test (4,500 in-scope + 1,000
oos_test) with the top class sum (TM) and the max softmax probability (LR).

Writes results/confirm/<registration>/<run name>/{seed<k>.json, preds_seed<k>.npz, lr.json, lr_preds.npz}.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.boolean import make_view  # noqa: E402
from dl.calib import ece, fit_temperature, nll  # noqa: E402
from dl.data import load  # noqa: E402
from dl.linear import accuracy, auroc, fit_logreg, logits, recall_at_precision  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import INITS, make_tm  # noqa: E402

DEV = "cuda"
GRID_C = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def calib_block(S_val, y_val, S_test, y_test) -> dict:
    tau = fit_temperature(S_val, y_val)
    P = torch.softmax(S_test.double() / tau, 1)
    return {"tau": tau, "ece_test": ece(P, y_test), "nll_test": nll(P, y_test),
            "ece_test_raw_softmax": ece(torch.softmax(S_test.double(), 1), y_test)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--view", default="thermo:8")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-literals", type=int, default=0)
    ap.add_argument("--budget-mode", default="exact", choices=["exact", "tmu", "block_full"])
    ap.add_argument("--budget-select", default="threshold", choices=["threshold", "rank"], help="rank: exact selection, no overshoot on ties (src/dl/tm.py select_k)")
    ap.add_argument("--clause-drop", type=float, default=0.0)
    ap.add_argument("--init", default="half", choices=INITS, help="start state (src/dl/tm_init.py); half = the code of R2 to R7")
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--lr-only", action="store_true")
    ap.add_argument("--lr-input", default="raw", choices=["raw", "bits"], help="LR on the raw embedding or on the view's bits")
    ap.add_argument("--readout", action="store_true", help="also fit convex vote weights on the clause outputs (+ input bits) of member 0")
    args = ap.parse_args()

    out = ROOT / "results" / "confirm" / args.registration / args.name
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "data" / "emb" / f"{args.dataset}__{args.encoder}.npz")
    ds = load(args.dataset)
    C = len(ds.intents)
    ytr, yva, yte = d["y_train"], d["y_val"], d["y_test"]
    has_oos = "oos_test" in d.files

    view = make_view(args.view, seed=0)
    view.fit(d["train"], ytr, texts=ds.texts["train"], class_names=ds.intents)
    enc = lambda split: view.transform(d[split], ds.texts[split])  # noqa: E731
    Btr, Bva, Bte = enc("train"), enc("val"), enc("test")
    # ---- logistic regression on the raw embedding (once) ----
    if args.lr_input == "raw":
        Ltr_in, Lva_in, Lte_in = d["train"], d["val"], d["test"]
        Loo_in = d["oos_test"] if has_oos else None
    else:
        Ltr_in, Lva_in, Lte_in = Btr.astype(np.float32), Bva.astype(np.float32), Bte.astype(np.float32)
        Loo_in = enc("oos_test").astype(np.float32) if has_oos else None
    if not (out / "lr.json").exists():
        t0 = time.time()
        best = (-1.0, None)
        for Cc in GRID_C:
            W, b = fit_logreg(Ltr_in, ytr, C, Cc)
            a = accuracy(logits(Lva_in, W, b), yva)
            if a > best[0]:
                best = (a, (Cc, W, b))
        Cc, W, b = best[1]
        Lte, Lva = logits(Lte_in, W, b), logits(Lva_in, W, b)
        rec = {"input": args.lr_input, "C": Cc, "val_acc": best[0], "test_acc": accuracy(Lte, yte),
               "calib": calib_block(Lva, yva, Lte, yte), "seconds": round(time.time() - t0, 1)}
        save = {"pred": Lte.argmax(1).cpu().numpy(), "y": yte}
        if has_oos:
            Loo = logits(Loo_in, W, b)
            msp_in = torch.softmax(Lte, 1).max(1).values.cpu().numpy()
            msp_oo = torch.softmax(Loo, 1).max(1).values.cpu().numpy()
            is_oos = np.r_[np.zeros(len(msp_in)), np.ones(len(msp_oo))]
            rec["oos_auroc_test_msp"] = auroc(is_oos, -np.r_[msp_in, msp_oo])
            rec["oos_recall_at_95prec_msp"] = recall_at_precision(is_oos, -np.r_[msp_in, msp_oo], 0.95)
            save.update({"oos_score_in": -msp_in, "oos_score_oos": -msp_oo})
        json.dump(rec, open(out / "lr.json", "w"), indent=1)
        np.savez_compressed(out / "lr_preds.npz", **save)
        print("LR", json.dumps(rec))
    if args.lr_only:
        return

    Xva = torch.as_tensor(Bva, dtype=torch.uint8, device=DEV)
    Xte = torch.as_tensor(Bte, dtype=torch.uint8, device=DEV)
    Xoo = torch.as_tensor(enc("oos_test"), dtype=torch.uint8, device=DEV) if has_oos else None
    for seed in args.seeds:
        path = out / f"seed{seed}.json"
        if path.exists():
            print("exists", path)
            continue
        t0 = time.time()
        Sva = Ste = Soo = 0
        members = []
        for j in range(args.ensemble):
            cfg = TMConfig(n_clauses=args.clauses, T=args.T or args.clauses, s=args.s, epochs=args.epochs,
                           max_literals=args.max_literals, budget_mode=args.budget_mode, budget_select=args.budget_select, clause_drop=args.clause_drop,
                           seed=100 * seed + j)
            tm = make_tm(C, Btr.shape[1], cfg, args.init).fit(Btr, ytr, Bva, yva)
            sv, st = tm.class_sums(Xva), tm.class_sums(Xte)
            scale = sv.std()
            Sva, Ste = Sva + sv / scale, Ste + st / scale
            if has_oos:
                Soo = Soo + tm.class_sums(Xoo) / scale
            members.append({"cfg": asdict(cfg), "init": args.init, "best_val": tm.best_val, "best_epoch": tm.best_epoch,
                            "test_acc": float((st.argmax(1).cpu().numpy() == yte).mean()), "stats": tm.stats(),
                            "train_s": round(tm.train_seconds, 1)})
            if j == 0:
                np.savez_compressed(out / f"model_seed{seed}.npz", **tm.state())
                tm0 = tm
        pred = Ste.argmax(1).cpu().numpy()
        rec = {"seed": seed, "test_acc": float((pred == yte).mean()),
               "val_acc": float((Sva.argmax(1).cpu().numpy() == yva).mean()),
               "calib": calib_block(Sva, yva, Ste, yte), "members": members,
               "seconds": round(time.time() - t0, 1)}
        save = {"pred": pred, "y": yte, "S_test": Ste.half().cpu().numpy(), "S_val": Sva.half().cpu().numpy()}
        if has_oos:
            save["S_oos"] = Soo.half().cpu().numpy()
        if has_oos:
            top_in, top_oo = Ste.max(1).values.cpu().numpy(), Soo.max(1).values.cpu().numpy()
            s2_in, s2_oo = Ste.topk(2, 1).values.cpu().numpy(), Soo.topk(2, 1).values.cpu().numpy()
            is_oos = np.r_[np.zeros(len(top_in)), np.ones(len(top_oo))]
            rec["oos_auroc_test_maxsum"] = auroc(is_oos, -np.r_[top_in, top_oo])
            rec["oos_recall_at_95prec_maxsum"] = recall_at_precision(is_oos, -np.r_[top_in, top_oo], 0.95)
            rec["oos_auroc_test_margin"] = auroc(is_oos, -np.r_[s2_in[:, 0] - s2_in[:, 1], s2_oo[:, 0] - s2_oo[:, 1]])
            save.update({"oos_score_in": -top_in, "oos_score_oos": -top_oo})
        if args.readout:
            Xtr_t = torch.as_tensor(Btr, dtype=torch.uint8, device=DEV)
            cl = {sp: tm0.clause_outputs(X).float() for sp, X in (("train", Xtr_t), ("val", Xva), ("test", Xte))}
            if has_oos:
                cl["oos"] = tm0.clause_outputs(Xoo).float()
            bits = {"train": Xtr_t.float(), "val": Xva.float(), "test": Xte.float()}
            if has_oos:
                bits["oos"] = Xoo.float()
            rec["readout"] = {}
            for kind in ("clauses", "clauses+bits"):
                F = {sp: (cl[sp] if kind == "clauses" else torch.cat([cl[sp], bits[sp]], 1)) for sp in cl}
                best = (-1.0, None)
                for Cc in [0.3, 1.0, 3.0, 10.0, 30.0]:
                    W, b = fit_logreg(F["train"], ytr, C, Cc)
                    a = accuracy(logits(F["val"], W, b), yva)
                    if a > best[0]:
                        best = (a, (Cc, W, b))
                Cc, W, b = best[1]
                Lt = logits(F["test"], W, b)
                rr = {"C": Cc, "val_acc": best[0], "test_acc": accuracy(Lt, yte),
                      "calib": calib_block(logits(F["val"], W, b), yva, Lt, yte)}
                save[f"readout_{kind}_pred"] = Lt.argmax(1).cpu().numpy()
                if has_oos:
                    Lo = logits(F["oos"], W, b)
                    mi, mo = torch.softmax(Lt, 1).max(1).values.cpu().numpy(), torch.softmax(Lo, 1).max(1).values.cpu().numpy()
                    rr["oos_auroc_test_msp"] = auroc(np.r_[np.zeros(len(mi)), np.ones(len(mo))], -np.r_[mi, mo])
                rec["readout"][kind] = rr
        json.dump(rec, open(path, "w"), indent=1)
        np.savez_compressed(out / f"preds_seed{seed}.npz", **save)
        print("TM", json.dumps({k: v for k, v in rec.items() if k != "members"}))


if __name__ == "__main__":
    main()
