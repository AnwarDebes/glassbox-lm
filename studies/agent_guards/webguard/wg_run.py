#!/usr/bin/env python
"""WebGuard veto test, the fits (webguard/PLAN_wg_veto.md "Models"; CPU only, 2 threads; DEV makes every choice).

    python -I wg_run.py --speed      # one TM fit on 200 train rows for 2 epochs, timed; writes results/wg_speed.json
    python -I wg_run.py              # the dev screen, the final fits, the saved predictions

TM: dl CoTM on the CPU (torch.cuda.synchronize is replaced by a no-op here, so no CUDA context is ever made), 1,000
clauses, T 1,000, 40 epochs, batch 64, hardest negatives, negated literals, eval on dev every 2 epochs, the best dev
epoch kept. s from {2, 4, 8, 12, 16, 24} at seed 1 on dev by e74's rule (the best s, but s 8 unless the best is at
least 0.3 points above it; exact ties to the first listed; integer units of 0.01 points); then seeds 1-3 at that s.
LR: dl.linear multinomial logistic regression on the CPU, C in {1, 10, 100}, the best on dev.
Saves: results/wg_dev.jsonl (dev accuracy of every fit), results/models/tm_seed<k>.npz (ta, w, cfg), and
results/wg_preds.npz with the dev and test class sums / probabilities of the final models. Test predictions are
stored, never scored here (wg_eval.py scores them once). Restart-safe (2026-10-20, before any TM fit; needed for
stop-and-restart during 3a): a screen s already in wg_dev.jsonl is skipped, each final seed is kept in
results/preds/tm_seed<k>.npz and LR in results/preds/lr.npz, and wg_preds.npz is assembled from them at the end. The
models, settings and data do not change. wg_preds.npz is never overwritten.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

torch.cuda.synchronize = lambda *a, **k: None  # CPU only: CoTM.fit calls it at the end
HERE = Path(__file__).resolve().parent
R = HERE / "results"
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.linear import fit_logreg, logits  # noqa: E402
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

S_LIST = [2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
N_CLAUSES, EPOCHS, BATCH = 1000, 40, 64


def cfg_for(s: float, seed: int, epochs: int = EPOCHS) -> TMConfig:
    return TMConfig(n_clauses=N_CLAUSES, T=N_CLAUSES, s=s, epochs=epochs, batch_size=BATCH, neg_sampling="hardest", negation=True,
                    max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0, clause_drop=0.0, half=False,
                    seed=seed, eval_every=2)


def fit_tm(Btr, ytr, Bdv, ydv, s, seed, epochs=EPOCHS):
    tm = make_tm(3, Btr.shape[1], cfg_for(s, seed, epochs), "half", device="cpu")
    tm.fit(Btr, ytr, Bdv, ydv)
    return tm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    z = np.load(HERE / "feats" / "wg_bits.npz")
    B, y, part = z["B"], z["y"], z["part"]
    tr, dv, te = (np.flatnonzero(part == k) for k in range(3))
    R.mkdir(exist_ok=True)
    if args.speed:
        out = R / "wg_speed.json"
        assert not out.exists()
        sub = tr[:200]
        t0 = time.time()
        fit_tm(B[sub], y[sub], B[dv], y[dv], 8.0, 1, epochs=2)
        sec = time.time() - t0
        per = sec / (2 * len(sub))  # seconds per training row-epoch (the dev evals included)
        proj_h = per * EPOCHS * len(tr) * (len(S_LIST) + 3) / 3600
        res = {"seconds_200x2": round(sec, 1), "per_row_epoch_s": per, "projected_hours": round(proj_h, 2), "threads": torch.get_num_threads()}
        out.write_text(json.dumps(res))
        print(json.dumps(res))
        return
    assert not (R / "wg_preds.npz").exists(), "wg_preds.npz exists"
    (R / "models").mkdir(exist_ok=True)
    (R / "preds").mkdir(exist_ok=True)
    dev_rows = [json.loads(l) for l in open(R / "wg_dev.jsonl")] if (R / "wg_dev.jsonl").exists() else []

    def log(rec):
        dev_rows.append(rec)
        with open(R / "wg_dev.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    acc = {float(r["s"]): round(r["dev_acc"] * 10000) for r in dev_rows if r.get("phase") == "screen" and r["model"] == "tm"}
    for s in S_LIST:
        if s in acc:
            continue
        tm = fit_tm(B[tr], y[tr], B[dv], y[dv], s, 1)
        acc[s] = round(tm.best_val * 10000)
        log({"model": "tm", "phase": "screen", "s": s, "seed": 1, "dev_acc": round(tm.best_val, 4), "best_epoch": tm.best_epoch,
             "train_s": round(tm.train_seconds, 1)})
    best = S_LIST[0]
    for s in S_LIST[1:]:
        if acc[s] > acc[best]:
            best = s
    pick = best if acc[best] - acc[8.0] >= 30 else 8.0
    if not any(r.get("phase") == "pick" and r["model"] == "tm" for r in dev_rows):
        log({"model": "tm", "phase": "pick", "pick": pick, "best": best, "screen": {str(s): acc[s] / 100 for s in S_LIST}})
    preds = {}
    for seed in (1, 2, 3):
        f = R / "preds" / f"tm_seed{seed}.npz"
        if not f.exists():
            tm = fit_tm(B[tr], y[tr], B[dv], y[dv], pick, seed)
            st = tm.state()
            np.savez(R / "models" / f"tm_seed{seed}.npz", ta=st["ta"], w=st["w"], cfg=json.dumps(asdict(tm.cfg)))
            sums = {name: tm.class_sums(torch.as_tensor(B[idx], dtype=torch.uint8)).cpu().numpy() for name, idx in (("dev", dv), ("test", te))}
            np.savez(f, dev=sums["dev"], test=sums["test"], s=pick)
            log({"model": "tm", "phase": "final", "s": pick, "seed": seed, "dev_acc": round(tm.best_val, 4), "best_epoch": tm.best_epoch,
                 "train_s": round(tm.train_seconds, 1)})
        zz = np.load(f)
        assert float(zz["s"]) == pick
        preds[f"tm{seed}_dev_sums"], preds[f"tm{seed}_test_sums"] = zz["dev"], zz["test"]
    f = R / "preds" / "lr.npz"
    if not f.exists():
        best_lr = (-1.0, None)
        for C in (1.0, 10.0, 100.0):
            W, b = fit_logreg(B[tr].astype(np.float32), y[tr], 3, C, device="cpu")
            a = float((logits(B[dv].astype(np.float32), W, b, device="cpu").argmax(1).numpy() == y[dv]).mean())
            log({"model": "lr", "phase": "screen", "C": C, "dev_acc": round(a, 4)})
            if a > best_lr[0]:
                best_lr = (a, C, W, b)
        _, C, W, b = best_lr
        log({"model": "lr", "phase": "pick", "C": C})
        np.savez(f, dev=torch.softmax(logits(B[dv].astype(np.float32), W, b, device="cpu"), 1).numpy(),
                 test=torch.softmax(logits(B[te].astype(np.float32), W, b, device="cpu"), 1).numpy(), C=C)
    zz = np.load(f)
    preds["lr_dev_prob"], preds["lr_test_prob"], C = zz["dev"], zz["test"], float(zz["C"])
    np.savez(R / "wg_preds.npz", **preds, dev_idx=dv, test_idx=te, tm_s=pick, lr_C=C)
    print(json.dumps({"saved": str(R / "wg_preds.npz"), "tm_s": pick, "lr_C": C}))


if __name__ == "__main__":
    main()
