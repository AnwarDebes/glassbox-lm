#!/usr/bin/env python
"""Step 3b, the twins of the TM student (tu/PLAN_3b.md, "Students"): a linear student and a counting student, on the
same input bits, the same training rows and the same code bits as the TM student.

    python -I -B b3_twins.py --train T.npz --held H.npz --out-dir DIR --tag NAME [--device cpu]
    python -I -B b3_twins.py predict --model DIR/NAME.linear.npz --x X.npz --out P.npz

Linear student: one ridge regression per output bit, least squares on +1 / -1 targets with an intercept, in closed form
from streamed sums (X'X, X'Y over the packed rows, float64): W = (X'X + lam I)^-1 X'Y on centred columns; a predicted
bit is X W + b > 0. lam = f x n with f from {1e-4, 1e-3, 1e-2, 1e-1}, chosen by the mean balanced bit accuracy on the
held-out file (C4 held-out only, never an intent split).
Counting student: m_ij = the mean of code bit j over the training rows in which input bit i is on (0.5 when bit i is
never on); score_j(x) = the mean of m_ij over the on bits of x (0.5 for a row with no bit on); bit j is on when
score_j > c_j, with c_j the (1 - p_j) quantile of score_j over the training rows, p_j the training on-rate of bit j.
Counts only, no fitted weights.
Data files: as b3_student.py ("X" packed input bits with "n_features"; "Y" packed code bits with "n_outputs").
Writes DIR/NAME.linear.npz, DIR/NAME.counting.npz and DIR/NAME.twins.json (held-out metrics of both, the lam grid);
never overwrites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b3_student import load, metrics, pack, unpack

GRID = (1e-4, 1e-3, 1e-2, 1e-1)


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def rows(Xp: np.ndarray, F: int, dev, bs: int = 8192):
    for i in range(0, Xp.shape[0], bs):
        yield i, unpack(torch.as_tensor(Xp[i : i + bs], device=dev), F).double()


def fit_linear(tr: dict, held: dict, dev, grid=GRID) -> tuple[dict, dict]:
    F, J, n = tr["F"], tr["J"], tr["X"].shape[0]
    sx, sy = torch.zeros(F, dtype=torch.float64, device=dev), torch.zeros(J, dtype=torch.float64, device=dev)
    xx = torch.zeros((F, F), dtype=torch.float64, device=dev)
    xy = torch.zeros((F, J), dtype=torch.float64, device=dev)
    for (i, x), (_, y) in zip(rows(tr["X"], F, dev), rows(tr["Y"], J, dev)):
        y = 2 * y - 1
        sx += x.sum(0)
        sy += y.sum(0)
        xx += x.T @ x
        xy += x.T @ y
    mx, my = sx / n, sy / n
    cxx = xx - n * torch.outer(mx, mx)
    cxy = xy - n * torch.outer(mx, my)
    Yh = unpack(torch.as_tensor(held["Y"]), J).numpy()
    best, scores = None, {}
    eye = torch.eye(F, dtype=torch.float64, device=dev)
    for f in grid:
        W = torch.linalg.solve(cxx + f * n * eye, cxy)
        b = my - mx @ W
        m = {"W": W.float().cpu().numpy(), "b": b.float().cpu().numpy(), "lam_factor": f}
        pred = predict_linear(m, held["X"], F, dev)
        scores[str(f)] = metrics(pred, Yh)
        if best is None or scores[str(f)]["balanced_acc"] > scores[str(best["lam_factor"])]["balanced_acc"]:
            best = m
    return best, scores


def predict_linear(m: dict, Xp: np.ndarray, F: int, dev) -> np.ndarray:
    W, b = torch.as_tensor(m["W"], device=dev).double(), torch.as_tensor(m["b"], device=dev).double()
    return np.concatenate([((x @ W + b) > 0).to(torch.uint8).cpu().numpy() for _, x in rows(Xp, F, dev)])


def fit_counting(tr: dict, dev) -> dict:
    F, J, n = tr["F"], tr["J"], tr["X"].shape[0]
    on = torch.zeros(F, dtype=torch.float64, device=dev)
    hit = torch.zeros((F, J), dtype=torch.float64, device=dev)
    py = torch.zeros(J, dtype=torch.float64, device=dev)
    for (_, x), (_, y) in zip(rows(tr["X"], F, dev), rows(tr["Y"], J, dev)):
        on += x.sum(0)
        hit += x.T @ y
        py += y.sum(0)
    M = torch.where(on[:, None] > 0, hit / on.clamp(min=1)[:, None], torch.full_like(hit, 0.5))
    m = {"M": M.float().cpu().numpy(), "p": (py / n).float().cpu().numpy()}
    S = scores_counting(m, tr["X"], F, dev)
    m["c"] = np.array([np.quantile(S[:, j], 1 - m["p"][j]) for j in range(J)], np.float32)
    return m


def scores_counting(m: dict, Xp: np.ndarray, F: int, dev) -> np.ndarray:
    M = torch.as_tensor(m["M"], device=dev).double()
    out = []
    for _, x in rows(Xp, F, dev):
        k = x.sum(1, keepdim=True)
        out.append(torch.where(k > 0, (x @ M) / k.clamp(min=1), torch.full((x.shape[0], M.shape[1]), 0.5, dtype=torch.float64,
                                                                             device=dev)).cpu().numpy())
    return np.concatenate(out)


def predict_counting(m: dict, Xp: np.ndarray, F: int, dev) -> np.ndarray:
    return (scores_counting(m, Xp, F, dev) > m["c"][None, :]).astype(np.uint8)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--model", type=Path, required=True)
        ap.add_argument("--x", type=Path, required=True)
        ap.add_argument("--out", type=Path, required=True)
        ap.add_argument("--device", default="cpu")
        a = ap.parse_args()
        if a.out.exists():
            print(f"refusing to overwrite {a.out}", file=sys.stderr)
            return 2
        torch.set_num_threads(2)
        m = dict(np.load(a.model))
        x = load(a.x)
        dev = torch.device(a.device)
        bits = predict_linear(m, x["X"], x["F"], dev) if "W" in m else predict_counting(m, x["X"], x["F"], dev)
        np.savez(a.out, X=pack(bits), n_features=bits.shape[1], model_sha256=sha16(a.model), x_sha256=sha16(a.x))
        print(json.dumps({"out": str(a.out), "rows": int(bits.shape[0]), "on_share": round(float(bits.mean()), 4)}))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--held", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    outs = [a.out_dir / f"{a.tag}.{k}" for k in ("linear.npz", "counting.npz", "twins.json")]
    for p in outs:
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    t0 = time.time()
    tr, held = load(a.train), load(a.held)
    lin, lin_scores = fit_linear(tr, held, dev)
    t_lin = time.time() - t0
    cnt = fit_counting(tr, dev)
    Yh = unpack(torch.as_tensor(held["Y"]), held["J"]).numpy()
    m_cnt = metrics(predict_counting(cnt, held["X"], held["F"], dev), Yh)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(outs[0], "xb") as fo:
        np.savez(fo, **lin)
    with open(outs[1], "xb") as fo:
        np.savez(fo, **cnt)
    rec = {"tag": a.tag, "n_train": int(tr["X"].shape[0]), "n_held": int(held["X"].shape[0]), "device": a.device,
           "linear": {"lam_factor": lin["lam_factor"], "held_by_lam": lin_scores, "held": lin_scores[str(lin["lam_factor"])],
                      "seconds": round(t_lin, 1), "sha16": sha16(outs[0])},
           "counting": {"held": m_cnt, "seconds": round(time.time() - t0 - t_lin, 1), "sha16": sha16(outs[1])},
           "inputs_sha256": {str(a.train): sha16(a.train), str(a.held): sha16(a.held)},
           "b3_twins.py": sha16(Path(__file__).resolve()), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(outs[2], "x") as fo:
        fo.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"linear": rec["linear"]["held"], "lam": lin["lam_factor"], "counting": m_cnt}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
