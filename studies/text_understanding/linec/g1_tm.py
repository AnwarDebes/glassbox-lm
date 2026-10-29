#!/usr/bin/env python
"""Line C G1, the TM arbiter (EXPLORATORY, validation only; runs only if G0 PASSED and its own registration line is
in LOG.md; plan LINE_C_NOTE.md). One run = one TM setting and seed on G0's rows; GPU.

    g1_tm.py --T 2000 --s 5 --clauses 4000 --seed 1 --stage grid [--rows linec/g0]

The rows, labels and floor numbers are G0's (rows_arbiter / rows_calibration / rows_validation .npz). The TM: dl
CoTM, two classes (the candidate is / is not the true next token), negation, hardest negative, batch 128, init half,
EPOCHS epochs, no best-state restore (N8's run loop). Score = class sum of "true" minus class sum of "not true";
isotonic regression fitted on the calibration rows maps it to a trust q; b in 0, 0.05, ..., 1 by calibration bits;
then the validation bits per token with G0's mixing formula. One row per run in linec/results_g1.jsonl, with the
per-document validation bits for the paired bootstrap.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

K = 8
EPOCHS = 10


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_rows(d: Path, name: str):
    z = np.load(d / f"rows_{name}.npz")
    X = np.unpackbits(z["bits"], axis=1)[:, : int(z["n_facts"])]
    return X, z["y"].reshape(-1).astype(np.int64), z


def mix_bits(q, pfC, pft, pfsum, in_C, true_rank, b) -> np.ndarray:  # G0's formula
    qq = np.maximum(np.asarray(q, dtype=np.float64), 1e-4)
    share = qq / qq.sum(1, keepdims=True)
    tr = np.where(in_C, true_rank, 0)
    st = share[np.arange(q.shape[0]), tr]
    pin = pfsum * ((1 - b) * pft / pfsum + b * st)
    return -np.log2(np.where(in_C, pin, pft))


def main() -> None:
    from sklearn.isotonic import IsotonicRegression

    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--rows", type=Path, default=HERE / "g0")
    ap.add_argument("--epochs", type=int, default=EPOCHS)  # smoke tests only; the registered runs use EPOCHS
    ap.add_argument("--out", type=Path, default=HERE / "results_g1.jsonl")
    args = ap.parse_args()
    t0 = time.time()
    torch.cuda.set_per_process_memory_fraction(0.25)
    XA, yA, _ = load_rows(args.rows, "arbiter")
    XC, yC, zc = load_rows(args.rows, "calibration")
    XV, yV, zv = load_rows(args.rows, "validation")
    cfg = TMConfig(n_clauses=args.clauses, T=args.T, s=args.s, epochs=args.epochs, batch_size=128, neg_sampling="hardest",
                   negation=True, max_literals=0, budget_mode="exact", budget_select="threshold", max_negated=0,
                   clause_drop=0.0, half=True, seed=args.seed, eval_every=1)
    tm = make_tm(2, XA.shape[1], cfg, "half")
    dev = tm.dev
    Xt = torch.as_tensor(np.ascontiguousarray(XA), dtype=torch.uint8, device=dev)
    yt = torch.as_tensor(yA, dtype=torch.long, device=dev)
    n = Xt.shape[0]
    t1 = time.time()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=tm.gen, device=dev)
        for start in range(0, n, cfg.batch_size):
            bidx = perm[start : start + cfg.batch_size]
            tm._step(Xt[bidx], yt[bidx])
    torch.cuda.synchronize()
    train_s = time.time() - t1
    del Xt, yt

    def score(X):
        out = []
        for i in range(0, X.shape[0], 262144):
            S = tm.class_sums(torch.as_tensor(np.ascontiguousarray(X[i : i + 262144]), dtype=torch.uint8, device=dev))
            out.append((S[:, 1] - S[:, 0]).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    sc, sv = score(XC), score(XV)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(sc, yC)
    qc, qv = iso.predict(sc).reshape(-1, K), iso.predict(sv).reshape(-1, K)
    RC = (zc["pfC"], zc["pft"], zc["pfsum"], zc["in_C"], zc["true_rank"])
    RV = (zv["pfC"], zv["pft"], zv["pfsum"], zv["in_C"], zv["true_rank"])
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    cal = [float(mix_bits(qc, *RC, b).mean()) for b in grid]
    b = float(grid[int(np.argmin(cal))])
    vb = mix_bits(qv, *RV, b)
    vbits = np.load(args.rows / "val_bits_per_position.npz")
    doc_of = vbits["doc_of"]
    rec = {"study": "Line C G1 (exploratory, validation)", "stage": args.stage, "seed": args.seed,
           "setting": {"clauses": args.clauses, "T": args.T, "s": args.s, "epochs": args.epochs, "batch": 128},
           "b": b, "calibration_bits_per_token": min(cal), "validation_bits_per_token": float(vb.mean()),
           "validation_doc_bits": np.bincount(doc_of, weights=vb).tolist(),
           "floor_validation_bits_per_token": float(vbits["floor"].mean()),
           "train_s": round(train_s, 1), "total_s": round(time.time() - t0, 1), "cfg": asdict(cfg),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "sha256": {"g1_tm.py": sha16(Path(__file__).resolve()), "rows_arbiter.npz": sha16(args.rows / "rows_arbiter.npz")}}
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("validation_doc_bits", "cfg")}), flush=True)


if __name__ == "__main__":
    main()
