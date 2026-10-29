#!/usr/bin/env python
"""Step 3a, the fit (tu/PLAN_3a.md "The fit"): the C4 word table refined toward the teacher by least squares under
mean pooling (potion's recipe, Model2Vec / Tokenlearn, in closed form), on our C4 word list.

    python -I a3_fit.py --selftest   # synthetic test on the CPU (no files)
    python -I a3_fit.py              # the GPU job: tu/a3/tables/a3A_d768.npz, a3B_d768.npz (the registered arm),
                                     # a3B_lam{lambda}.npz for every lambda, tu/a3/fit_meta.json; refuses to overwrite

Student pooling: sentence vector = mean of the rows of its known words (live rows of ppw_S1000, repeats counted):
A[n, w] = count of w in n / known words in n. Targets: the teacher vectors (tu/a3/T_*.f16.npy) minus their mean over
the training sentences (the held-out set uses the same mean); 768 dimensions, no PCA.
- a3A (descriptive): M = argmin ||A W0 M - T||^2 + mu ||M||^2 (500 x 768, mu = 1e-6 trace(G) / 500, G = (A W0)^T A W0);
  W_A = W0 M.
- a3B (registered): W = argmin ||A W - T||^2 + lambda ||W - W_A||^2, i.e. (A^T A + lambda I) W = A^T T + lambda W_A, by
  conjugate gradients with the Jacobi preconditioner, for lambda 10, 1, 0.1, 0.01 in that order: the first started at
  W_A, each next one at the previous solution (the same solution at convergence; fewer steps). Stop at a relative
  residual (Frobenius) of at most 1e-4, or after 400 steps. A word that never occurs keeps its W_A row; a word dead in
  ppw_S1000 (zero row) stays dead. lambda* = the highest mean held-out cosine between A_held W and the held-out targets;
  ties to the larger lambda. The intent sets play no part.
Tables: unit rows, float16, in ppw_S1000's format (words, codes, counts, vocab_ids copied from it).
Input sentences: tu/a3/c4_sent_{held,train}_f.txt (after the MultiWOZ filter, a3_mwoz_filter.py), aligned with T_*.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
A3 = HERE / "a3"
TABLE = EMB / "w1" / "tables" / "ppw_S1000_d500_e0.5.npz"
LAMBDAS = [10.0, 1.0, 0.1, 0.01]
CHUNK = 250_000
TOL, MAXIT = 1e-4, 400


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def sparse_rows(rows: list[np.ndarray], V: int, device: str):
    """CSR chunks (A_c, A_c^T) for sentences given as arrays of row ids (repeats kept), with mean-pooling weights."""
    out = []
    for st in range(0, len(rows), CHUNK):
        part = rows[st : st + CHUNK]
        r_idx, c_idx, vals = [], [], []
        for i, x in enumerate(part):
            u, c = np.unique(x, return_counts=True)
            r_idx.append(np.full(len(u), i, dtype=np.int64))
            c_idx.append(u.astype(np.int64))
            vals.append((c / len(x)).astype(np.float32))
        r = torch.from_numpy(np.concatenate(r_idx))
        c = torch.from_numpy(np.concatenate(c_idx))
        v = torch.from_numpy(np.concatenate(vals))
        A = torch.sparse_coo_tensor(torch.stack([r, c]), v, (len(part), V)).coalesce().to_sparse_csr().to(device)
        At = torch.sparse_coo_tensor(torch.stack([c, r]), v, (V, len(part))).coalesce().to_sparse_csr().to(device)
        out.append((A, At))
    return out


def targets(T: np.ndarray, st: int, n: int, mean: torch.Tensor, device: str) -> torch.Tensor:
    return torch.from_numpy(T[st : st + n].astype(np.float32)).to(device) - mean


def fit(rows_tr, T_tr, rows_ho, T_ho, W0: np.ndarray, device: str, lambdas=LAMBDAS, tol=TOL, maxit=MAXIT, log=print):
    V, d0 = W0.shape
    D = T_tr.shape[1]
    t0 = time.time()
    A_tr = sparse_rows(rows_tr, V, device)
    A_ho = sparse_rows(rows_ho, V, device)
    mean = torch.zeros(D, dtype=torch.float64, device=device)
    for st in range(0, len(T_tr), CHUNK):
        mean += torch.from_numpy(T_tr[st : st + CHUNK].astype(np.float32)).to(device).double().sum(0)
    mean = (mean / len(T_tr)).float()
    W0t = torch.from_numpy(W0.astype(np.float32)).to(device)
    G = torch.zeros((d0, d0), dtype=torch.float64, device=device)
    R0 = torch.zeros((d0, D), dtype=torch.float64, device=device)
    AT = torch.zeros((V, D), dtype=torch.float32, device=device)
    diag = torch.zeros(V, dtype=torch.float32, device=device)
    for k, (A, At) in enumerate(A_tr):
        Tc = targets(T_tr, k * CHUNK, A.shape[0], mean, device)
        P = torch.sparse.mm(A, W0t)
        G += (P.T @ P).double()
        R0 += (P.T @ Tc).double()
        AT += torch.sparse.mm(At, Tc)
        diag.index_add_(0, A.col_indices(), A.values() ** 2)
    mu = 1e-6 * float(torch.trace(G)) / d0
    M = torch.linalg.solve(G + mu * torch.eye(d0, dtype=torch.float64, device=device), R0).float()
    WA = W0t @ M
    log(f"step A done ({time.time() - t0:.0f}s)")

    def op(X, lam):
        Y = lam * X
        for A, At in A_tr:
            Y += torch.sparse.mm(At, torch.sparse.mm(A, X))
        return Y

    def heldout_cos(W):
        num, n = 0.0, 0
        for k, (A, _) in enumerate(A_ho):
            Tc = targets(T_ho, k * CHUNK, A.shape[0], mean, device)
            P = torch.sparse.mm(A, W)
            num += float(torch.nn.functional.cosine_similarity(P, Tc, dim=1).double().sum())
            n += A.shape[0]
        return num / n

    res = {"heldout_cos": {"W_A": round(heldout_cos(WA), 5)}, "cg": {}, "mu": mu,
           "words_seen": int((diag > 0).sum()), "words": V}
    sols = {}
    X = WA.clone()
    for lam in lambdas:
        B = AT + lam * WA
        dinv = 1.0 / (diag + lam)
        R = B - op(X, lam)
        Z = R * dinv[:, None]
        P = Z.clone()
        rz = (R * Z).sum(0)
        bn = float(B.norm())
        rel, it = float(R.norm()) / bn, 0
        while rel > tol and it < maxit:
            it += 1
            Q = op(P, lam)
            alpha = rz / (P * Q).sum(0)
            X = X + P * alpha
            R = R - Q * alpha
            rel = float(R.norm()) / bn
            Z = R * dinv[:, None]
            rz_new = (R * Z).sum(0)
            P = Z + P * (rz_new / rz)
            rz = rz_new
        sols[lam] = X.clone()
        res["cg"][str(lam)] = {"steps": it, "rel_residual": rel}
        res["heldout_cos"][str(lam)] = round(heldout_cos(X), 5)
        log(f"lambda {lam}: {it} steps, residual {rel:.2e}, held-out cos {res['heldout_cos'][str(lam)]} ({time.time() - t0:.0f}s)")
    best = max(lambdas, key=lambda l: (res["heldout_cos"][str(l)], l))
    res["lambda_star"] = best
    res["seconds"] = round(time.time() - t0, 1)
    return WA.cpu().numpy(), {l: s.cpu().numpy() for l, s in sols.items()}, M.cpu().numpy(), res


def unit_rows(W: np.ndarray, dead: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(W, axis=1, keepdims=True)
    out = np.where(n > 0, W / np.maximum(n, 1e-30), 0.0)
    out[dead] = 0.0
    return out.astype(np.float16)


def selftest() -> None:
    rng = np.random.default_rng(0)
    V, d0, D, N, NH = 300, 20, 16, 20000, 2000
    dead = np.zeros(V, dtype=bool)
    dead[-10:] = True
    live = np.flatnonzero(~dead)
    W_true = rng.normal(size=(V, D)).astype(np.float32)
    W0 = rng.normal(size=(V, d0)).astype(np.float32)
    W0[dead] = 0
    p = 1.0 / np.arange(1, len(live) + 1)
    p /= p.sum()

    def sents(n):
        return [rng.choice(live, size=rng.integers(3, 16), p=p) for _ in range(n)]

    rows_tr, rows_ho = sents(N), sents(NH)
    rows_tr.append(live[-1:].repeat(3))  # make sure the rarest live word occurs
    pool = lambda rows: np.stack([W_true[r].mean(0) for r in rows])  # noqa: E731
    T_tr, T_ho = pool(rows_tr).astype(np.float16), pool(rows_ho).astype(np.float16)
    WA, sols, M, res = fit(rows_tr, T_tr, rows_ho, T_ho, W0, "cpu", lambdas=[1e-6], tol=1e-9, maxit=3000, log=lambda s: None)
    m = T_tr.astype(np.float32).mean(0)
    target = W_true - m
    seen = np.zeros(V, dtype=bool)
    for r in rows_tr:
        seen[r] = True
    err = np.linalg.norm(sols[1e-6][seen] - target[seen]) / np.linalg.norm(target[seen])
    # step A against numpy on the dense problem
    A = np.zeros((len(rows_tr), V), dtype=np.float64)
    for i, r in enumerate(rows_tr):
        u, c = np.unique(r, return_counts=True)
        A[i, u] = c / len(r)
    P = A @ W0
    Tc = T_tr.astype(np.float64) - m
    mu = 1e-6 * np.trace(P.T @ P) / d0
    M_np = np.linalg.solve(P.T @ P + mu * np.eye(d0), P.T @ Tc)
    errA = np.abs(M - M_np).max() / np.abs(M_np).max()
    # held-out cosine against a direct loop
    W = sols[1e-6]
    direct = float(np.mean([np.dot(W[r].mean(0), T_ho[i].astype(np.float32) - m) /
                            (np.linalg.norm(W[r].mean(0)) * np.linalg.norm(T_ho[i].astype(np.float32) - m))
                            for i, r in enumerate(rows_ho)]))
    errC = abs(direct - res["heldout_cos"]["1e-06"])
    dead_ok = bool((np.abs(sols[1e-6][dead]).max() == 0) and (np.abs(WA[dead]).max() == 0))
    print(json.dumps({"recover_rel_err": float(err), "stepA_rel_err": float(errA), "heldout_cos_err": float(errC),
                      "dead_rows_stay_zero": dead_ok, "cg": res["cg"]}))
    assert err < 1e-3 and errA < 1e-4 and errC < 1e-4 and dead_ok, "SELFTEST FAILED"
    print("SELFTEST PASSED")


def main() -> None:
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        return
    assert sys.argv[1:] == [], sys.argv
    sys.path.insert(0, str(EMB))
    import glove_feats as GF

    t0 = time.time()
    out = A3 / "tables"
    out.mkdir(exist_ok=True)
    meta_p = A3 / "fit_meta.json"
    names = ["a3A_d768.npz", "a3B_d768.npz"] + [f"a3B_lam{l:g}.npz" for l in LAMBDAS]
    for f in [meta_p, *(out / n for n in names)]:
        assert not f.exists(), f"{f} exists"
    z = np.load(TABLE)
    W0 = z["codes"].astype(np.float32)
    dead = np.abs(W0).sum(1) == 0
    row = {str(w): i for i, w in enumerate(z["words"]) if not dead[i]}

    def rows_of(sp):
        texts = (A3 / f"c4_sent_{sp}_f.txt").read_text(encoding="utf-8").split("\n")[:-1]
        rows = [np.array([row[t] for t in GF.glove_tokens(s) if t in row], dtype=np.int64) for s in texts]
        assert min(len(r) for r in rows) >= 3
        return rows

    rows_tr, rows_ho = rows_of("train"), rows_of("held")
    T_tr = np.load(A3 / "T_train.f16.npy")
    T_ho = np.load(A3 / "T_held.f16.npy")
    assert len(T_tr) == len(rows_tr) and len(T_ho) == len(rows_ho)
    print(f"{len(rows_tr):,} training and {len(rows_ho):,} held-out sentences ({time.time() - t0:.0f}s)", flush=True)
    WA, sols, M, res = fit(rows_tr, T_tr, rows_ho, T_ho, W0, "cuda", log=lambda s: print(s, flush=True))
    keep = {k: z[k] for k in ("words", "counts", "vocab_ids")}
    np.savez(out / "a3A_d768.npz", codes=unit_rows(WA, dead), **keep)
    for l, W in sols.items():
        np.savez(out / f"a3B_lam{l:g}.npz", codes=unit_rows(W, dead), **keep)
    np.savez(out / "a3B_d768.npz", codes=unit_rows(sols[res["lambda_star"]], dead), **keep)
    res.update({"dead_words": int(dead.sum()), "train": len(rows_tr), "held": len(rows_ho),
                "sha256": {n: sha16(out / n) for n in names} | {"a3_fit.py": sha16(Path(__file__).resolve()),
                                                                 "table": sha16(TABLE)},
                "total_seconds": round(time.time() - t0, 1), "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    meta_p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
