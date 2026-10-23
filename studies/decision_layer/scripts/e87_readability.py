#!/usr/bin/env python
"""e87 A (plan logs/plan_e87_receipts.md): how readable the C4 agent's dev decisions are, against LR and L1-LR on the
same bits. CPU only. Reads results/explore/e87/<dataset>_c4.npz (scripts/e87_bits.py) and the registered C4 models
(model_seed<k>.npz; from seed<k>.json only dev_acc and bits_sha256). No test file and no test prediction is read.
No rule: everything is reported per dataset, per seed and as the mean over seeds.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python scripts/e87_readability.py --dataset abcd

Definitions (the plan's):
- A clause fires on x when all its included literals hold (plain literal j: x_j = 1; negated: x_j = 0). An empty
  clause does not fire (as CoTM.clause_outputs at inference). Class sums S_y = sum of w[k, y] over the firing k.
- The decision c = argmax S (lowest index on ties); the runner-up r = argmax over the other classes.
- TM receipt: the firing clauses with d_k = w[k, c] - w[k, r] > 0, ranked by d_k (descending; ties: lower clause
  index first). The receipt is the shortest prefix (at most 20 clauses) whose sums alone (the sum of w over the
  prefix) give argmax c. Its size: clauses, total literals (plain + negated), and its longest clause.
- LR receipt (the L2 LR of A0, and L1-LR; each with its own c and r): the active bits with d_j = W[j, c] - W[j, r] > 0,
  ranked the same way; the shortest prefix (0 to 20 bits) whose scores alone, intercept included (b plus the sum of
  the prefix's W rows), give argmax c. Its size: bits (= literals).
- Coverage at k: the share of dev decisions with a receipt of at most k literals (k = 1 to 10). S1's point: at most 3
  clauses of at most 3 literals each (TM); at most 9 bits (LR).
- Clause precision: for a clause firing on at least 10 dev rows, the share of those rows whose label is the clause's
  strongest class (argmax over w[k], lowest index on ties).
Writes results/explore/e87/readability_<dataset>.json (refuses to overwrite) and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"multiwoz": "results/confirm/R24mwoz/tm_mem_c4_nrand", "star": "results/confirm/R22star/tm_mem_c4_s16",
          "abcd": "results/confirm/R23abcd/tm_mem_c4_s16"}
KMAX, KS, BS = 20, tuple(range(1, 11)), 256
L1_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)


def fires(X: np.ndarray, incl: np.ndarray, bs: int = BS) -> np.ndarray:
    """X uint8 [n, F], incl bool [K, 2F] -> bool [n, K]: every included literal holds and the clause is not empty."""
    F = X.shape[1]
    P, N = incl[:, :F].T.astype(np.float32), incl[:, F:].T.astype(np.float32)
    has = incl.any(1)
    out = np.empty((X.shape[0], incl.shape[0]), dtype=bool)
    for i in range(0, X.shape[0], bs):
        x = X[i : i + bs].astype(np.float32)
        out[i : i + bs] = (((1.0 - x) @ P + x @ N) == 0) & has
    return out


def decide(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = S.argmax(1)
    T = S.astype(np.float64)
    T[np.arange(len(c)), c] = -np.inf
    return c, T.argmax(1)


def receipts(D: np.ndarray, V: np.ndarray, base: np.ndarray | None, c: np.ndarray, sizes: np.ndarray, kmax: int = KMAX) -> dict:
    """D [n, M]: each item's d (an item is a candidate only if d > 0); V [M, C] the items' score vectors; base [C]
    (LR's intercept) or None (TM: the prefix may not be empty). Returns per decision: n_items (kmax + 1 = no receipt
    within kmax), literals and the longest item of the receipt."""
    n = D.shape[0]
    order = np.argsort(-D, axis=1, kind="stable")[:, :kmax]  # stable: ties keep the lower index first
    d_top = np.take_along_axis(D, order, 1)
    valid = d_top > 0
    cum = np.cumsum(V[order] * valid[:, :, None], axis=1)
    if base is not None:
        cum = cum + base[None, None, :]
    match = (cum.argmax(2) == c[:, None]) & valid
    j = np.where(match.any(1), match.argmax(1) + 1, kmax + 1)
    if base is not None:
        j = np.where(base.argmax() == c, 0, j)
    sz = sizes[order] * valid
    csz, msz = np.cumsum(sz, 1), np.maximum.accumulate(sz, 1)
    at = np.clip(j - 1, 0, order.shape[1] - 1)[:, None]  # fewer items than kmax: the prefix ends at the last item
    lits = np.where(j == 0, 0, np.take_along_axis(csz, at, 1)[:, 0])
    longest = np.where(j == 0, 0, np.take_along_axis(msz, at, 1)[:, 0])
    found = j <= kmax
    return {"n_items": j, "literals": np.where(found, lits, -1), "longest": np.where(found, longest, -1), "found": found}


def summary(r: dict, tm: bool) -> dict:
    f, j, lit, lg = r["found"], r["n_items"], r["literals"], r["longest"]
    out = {"coverage_at_k_literals": {str(k): float((f & (lit <= k)).mean()) for k in KS},
           "share_no_receipt_within_20": float((~f).mean()),
           "median_items": float(np.median(np.where(f, j, KMAX + 1))),
           "median_literals_found": float(np.median(lit[f])) if f.any() else None}
    out["s1_point"] = float((f & (j <= 3) & (lg <= 3)).mean()) if tm else float((f & (j <= 9)).mean())
    return out


def tm_part(X: np.ndarray, y: np.ndarray, ta: np.ndarray, w: np.ndarray, n_states: int) -> dict:
    incl = ta > n_states
    F = X.shape[1]
    plain, neg = incl[:, :F].sum(1), incl[:, F:].sum(1)
    lens = plain + neg
    fi = fires(X, incl)
    w64 = w.astype(np.int64)
    S = fi.astype(np.int64) @ w64
    c, r = decide(S)
    rec = {"n_items": [], "literals": [], "longest": [], "found": []}
    for i in range(0, X.shape[0], BS):
        sl = slice(i, i + BS)
        D = (w64[:, c[sl]].T - w64[:, r[sl]].T) * fi[sl]
        part = receipts(D, w64, None, c[sl], lens)
        for k in rec:
            rec[k].append(part[k])
    rec = {k: np.concatenate(v) for k, v in rec.items()}
    cnt = fi.sum(0)
    cls = w64.argmax(1)
    onehot = np.zeros((len(y), w.shape[1]), dtype=np.float32)
    onehot[np.arange(len(y)), y] = 1
    hits = (fi.T.astype(np.float32) @ onehot)[np.arange(len(cls)), cls]
    ok = cnt >= 10
    prec = hits[ok] / cnt[ok]
    fl = np.repeat(lens, cnt)  # literals of every (decision, firing clause) pair
    short = {}
    for b in (3, 9):
        m = (lens <= b) & (lens > 0)
        short[str(b)] = float(((fi[:, m].astype(np.int64) @ w64[m]).argmax(1) == y).mean())
    has = lens > 0
    return {"dev_acc": float((c == y).mean()), "pred": c,
            "clauses": {"non_empty": int(has.sum()), "median_literals": float(np.median(lens[has])),
                        "median_plain": float(np.median(plain[has])), "median_negated": float(np.median(neg[has])),
                        "share_negated_only": float(((plain == 0) & (neg > 0)).sum() / max(has.sum(), 1)),
                        "share_le3_literals": float(((lens <= 3) & has).sum() / max(has.sum(), 1))},
            "firing": {"median_per_decision": float(np.median(fi.sum(1))), "median_literals_of_firing": float(np.median(fl)) if fl.size else None,
                       "p90_literals_of_firing": float(np.percentile(fl, 90)) if fl.size else None},
            "precision": {"n_clauses_ge10": int(ok.sum()), "median": float(np.median(prec)) if prec.size else None,
                          "q25": float(np.percentile(prec, 25)) if prec.size else None, "q75": float(np.percentile(prec, 75)) if prec.size else None},
            "accuracy_short_clauses_only": short, "receipts": summary(rec, tm=True)}


def lr_part(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> dict:
    W64, b64 = W.astype(np.float64), b.astype(np.float64)
    c, r = decide(X.astype(np.float64) @ W64 + b64)
    rec = {"n_items": [], "literals": [], "longest": [], "found": []}
    ones = np.ones(W.shape[0], dtype=np.int64)
    for i in range(0, X.shape[0], BS):
        sl = slice(i, i + BS)
        D = (W64[:, c[sl]].T - W64[:, r[sl]].T) * X[sl]
        part = receipts(D, W64, b64, c[sl], ones)
        for k in rec:
            rec[k].append(part[k])
    rec = {k: np.concatenate(v) for k, v in rec.items()}
    return {"dev_acc": float((c == y).mean()), "pred": c, "nonzero_weights": int((W64 != 0).sum()),
            "bits_with_a_weight": int((W64 != 0).any(1).sum()), "receipts": summary(rec, tm=False)}


def l1_fit(Xtr: np.ndarray, ytr: np.ndarray, n_classes: int, C: float) -> tuple[np.ndarray, np.ndarray, dict]:
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    m = LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=200, tol=1e-3, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(Xtr.astype(np.float32), ytr)
    W = np.zeros((Xtr.shape[1], n_classes))
    b = np.full(n_classes, -1e9)  # a label absent from train is never predicted
    W[:, m.classes_] = m.coef_.T
    b[m.classes_] = m.intercept_
    return W, b, {"C": C, "seconds": round(time.time() - t0, 1), "n_iter": int(np.max(m.n_iter_)),
                  "converged": not any(issubclass(x.category, ConvergenceWarning) for x in caught)}


L1_EXT = (0.003, 0.001, 0.0003, 0.0001)


def l1_at_budget(l1: dict, pruned_acc: float, fit=None) -> dict:
    """Part B's kill rule (the plan's, with its fixed extension). l1 maps C to {"dev_acc", "cov9"} (dev
    accuracy, coverage at 9 bits) for the fitted grid; fit(C) returns the same for a new C; pruned_acc is the pruned
    agent's mean dev accuracy over its seeds. The comparator is the most accurate C with cov9 >= 0.90 (ties: smaller
    C). If the grid has none, C 0.003, 0.001, 0.0003, 0.0001 are added in that order, stopping at the first that
    reaches 0.90 or once dev accuracy falls below pruned_acc - 5 points. No comparator: "not testable" (never "not
    killed"). Killed: the comparator's dev accuracy is at least pruned_acc - 0.5 points."""
    ok = [C for C, r in l1.items() if r["cov9"] >= 0.90]
    added = []
    if not ok and fit is not None:
        for C in L1_EXT:
            l1[C] = fit(C)
            added.append(C)
            if l1[C]["cov9"] >= 0.90:
                ok = [C]
                break
            if l1[C]["dev_acc"] < pruned_acc - 0.05:
                break
    if not ok:
        return {"C": None, "added": added, "verdict": "not testable", "why": "L1-LR does not reach the budget"}
    C = max(ok, key=lambda c: (l1[c]["dev_acc"], -c))
    killed = l1[C]["dev_acc"] >= pruned_acc - 0.005 - 1e-12
    return {"C": C, "added": added, "verdict": "killed" if killed else "not killed", "dev_acc": l1[C]["dev_acc"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--l1-grid", type=float, nargs="+", default=list(L1_GRID))
    a = ap.parse_args()
    out = ROOT / "results" / "explore" / "e87" / f"readability_{a.dataset}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    z = np.load(ROOT / "results" / "explore" / "e87" / f"{a.dataset}_c4.npz")
    Xtr, ytr, Xdv, ydv = z["X_train"], z["y_train"], z["X_dev"], z["y_dev"]
    hashes = json.loads(str(z["bits_sha256"]))
    n_classes = len(z["buttons"])
    res = {"dataset": a.dataset, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_dev": int(len(ydv)),
           "n_bits": int(Xdv.shape[1]), "models": MODELS[a.dataset], "tm": {}, "lr": {}, "l1": {}}
    for sd in a.seeds:
        js = json.loads((ROOT / MODELS[a.dataset] / f"seed{sd}.json").read_text())
        if js["bits_sha256"]["dev"] != hashes["dev"] or js["bits_sha256"]["train"] != hashes["train"]:
            raise SystemExit(f"seed {sd}: the model's bits are not A0's")
        m = np.load(ROOT / MODELS[a.dataset] / f"model_seed{sd}.npz")
        cfg = json.loads(str(m["cfg"]))
        t = tm_part(Xdv, ydv, m["ta"], m["w"], cfg["n_states"])
        if t["dev_acc"] != js["dev_acc"]:
            raise SystemExit(f"seed {sd}: recomputed dev accuracy {t['dev_acc']!r} is not the run's {js['dev_acc']!r}")
        t.pop("pred")
        res["tm"][str(sd)] = t
        print(f"tm seed {sd}: dev {t['dev_acc']:.4f} (equals the run's); receipts at <= 9 literals "
              f"{t['receipts']['coverage_at_k_literals']['9']:.3f}; S1 point {t['receipts']['s1_point']:.3f}", flush=True)
    lr = lr_part(Xdv, ydv, z["lr_W"], z["lr_b"])
    lr["agrees_with_A0_gpu_predictions"] = float((lr.pop("pred") == z["lr_pred_dev"]).mean())
    lr["C"] = float(z["lr_C"])
    res["lr"] = lr
    print(f"lr (C {lr['C']:g}): dev {lr['dev_acc']:.4f}; at <= 9 bits {lr['receipts']['coverage_at_k_literals']['9']:.3f}", flush=True)
    for C in a.l1_grid:
        W, b, info = l1_fit(Xtr, ytr, n_classes, C)
        part = lr_part(Xdv, ydv, W, b)
        part.pop("pred")
        res["l1"][f"{C:g}"] = {**info, **part}
        print(f"l1 C {C:g}: dev {part['dev_acc']:.4f}; bits with a weight {part['bits_with_a_weight']}; at <= 9 bits "
              f"{part['receipts']['coverage_at_k_literals']['9']:.3f}; {info['seconds']} s, converged {info['converged']}", flush=True)
    best = max(res["l1"], key=lambda k: (res["l1"][k]["dev_acc"], -float(k)))
    res["l1_chosen_on_dev"] = best
    seeds = list(res["tm"].values())
    res["tm_mean"] = {"dev_acc": float(np.mean([s["dev_acc"] for s in seeds])),
                      "coverage_at_k_literals": {str(k): float(np.mean([s["receipts"]["coverage_at_k_literals"][str(k)] for s in seeds])) for k in KS},
                      "s1_point": float(np.mean([s["receipts"]["s1_point"] for s in seeds]))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("coverage at k literals (k = 1..10), mean over seeds for the TM:")
    print("  TM    " + " ".join(f"{res['tm_mean']['coverage_at_k_literals'][str(k)]:.3f}" for k in KS))
    print("  LR    " + " ".join(f"{lr['receipts']['coverage_at_k_literals'][str(k)]:.3f}" for k in KS))
    l1 = res["l1"][best]
    print(f"  L1-LR " + " ".join(f"{l1['receipts']['coverage_at_k_literals'][str(k)]:.3f}" for k in KS) + f"  (C {best}, chosen on dev)")
    print(f"dev accuracy: TM {res['tm_mean']['dev_acc']:.4f} (mean of {len(seeds)} seeds), LR {lr['dev_acc']:.4f}, L1-LR {l1['dev_acc']:.4f}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
