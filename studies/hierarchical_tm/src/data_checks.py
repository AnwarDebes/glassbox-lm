"""Check 1b (gates.md, "no shortcuts") and the first baselines, for each pair version, at F=1024.

Everything is fitted and tuned on train and evaluated on dev. 95% CIs are Wilson intervals (pairs treated as
independent, which makes the CI narrower, so the "CI includes 0.50" rules are harder to pass).
Check 1b, per version:
  z-test        two-proportion z-test of single-paragraph word frequency, positives vs negatives, on
                paragraph A of every train pair (A is a random side, so pairs stay independent).
                Gate: at most about 3 words with |z| > 3 (we count strictly: <= 3). For reference we also
                give the chance level: 1024 * P(|Z| > 3) = 2.76, and the 95th percentile of a label-permutation null.
  one-paragraph logistic regression (LR) on paragraph A only. Gate: dev accuracy CI includes 0.50.
  partner-shuffle  B is replaced by the B of a random other pair of the same split (labels kept), in
                train and in dev; LR on [A, B, A AND B]. Gate: CI includes 0.50.
  ns only       overlap is 0 for every pair; length-only classifiers on (len A, len B, |len A - len B|,
                min, max), an LR and a gradient-boosted tree model. Gate: both CIs include 0.50.
Baselines (dev accuracy, AUC, always-no accuracy): Jaccard, tf-idf cosine and shared-word count with a
threshold tuned on train; LR on A AND B; LR on [A, B, A AND B]; PPMI-SVD cosine (100 dims, word
co-occurrence within train-split paragraphs; paragraph vector = mean of its word vectors).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairdata import DATA, ids_to_csr, load_pairs, load_vocab  # noqa: E402

F = 1024
Z_CHANCE = 1024 * 0.0026998


def wilson(acc, n, z=1.96):
    d = 1 + z * z / n
    c, h = (acc + z * z / (2 * n)) / d, z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def report(pred, score, y):
    acc = float((pred == y).mean())
    ci = wilson(acc, len(y))
    return {"acc": round(acc, 4), "ci": ci, "ci_has_0.50": ci[0] <= 0.5 <= ci[1],
            "auc": round(float(roc_auc_score(y, score)), 4), "always_no": round(float(1 - y.mean()), 4),
            "pos_share": round(float(pred.mean()), 4), "n": len(y)}


def tune_threshold(s, y):
    """Cut-off t (predict 1 when s >= t) with the best train accuracy."""
    u = np.unique(s)
    sp, sn = np.sort(s[y == 1]), np.sort(s[y == 0])
    acc = (len(sp) - np.searchsorted(sp, u) + np.searchsorted(sn, u)) / len(s)
    return float(u[np.argmax(acc)])


def threshold_baseline(s_tr, y_tr, s_dev, y_dev):
    t = tune_threshold(s_tr, y_tr)
    return {**report((s_dev >= t).astype(int), s_dev, y_dev), "threshold": round(t, 5)}


def lr_baseline(X_tr, y_tr, X_dev, y_dev, grid=(0.01, 0.03, 0.1, 0.3, 1.0), seed=0):
    """LR with C chosen on a random 80/20 inner split of train, then refitted on all of train."""
    idx = np.random.default_rng(seed).permutation(len(y_tr))
    a, b = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
    inner = {C: LogisticRegression(C=C, solver="liblinear").fit(X_tr[a], y_tr[a]).score(X_tr[b], y_tr[b]) for C in grid}
    C = max(grid, key=lambda c: inner[c])
    m = LogisticRegression(C=C, solver="liblinear").fit(X_tr, y_tr)
    return {**report(m.predict(X_dev), m.decision_function(X_dev), y_dev), "C": C}


def feats(p, kind):
    A, B = p["A"].astype(np.float64), p["B"].astype(np.float64)
    if kind == "A":
        return A.tocsr()
    if kind == "AND":
        return A.multiply(B).tocsr()
    return sparse.hstack([A, B, A.multiply(B)]).tocsr()


def ztest(A, y):
    n1, n0 = y.sum(), len(y) - y.sum()
    c1, c0 = np.asarray(A[y == 1].sum(0)).ravel(), np.asarray(A[y == 0].sum(0)).ravel()
    p = (c1 + c0) / (n1 + n0)
    se = np.sqrt(np.maximum(p * (1 - p), 1e-12) * (1 / n1 + 1 / n0))
    return (c1 / n1 - c0 / n0) / se


def train_paragraphs():
    z = np.load(os.path.join(DATA, "paragraphs_F3072.npz"))
    train = json.load(open(os.path.join(DATA, "splits.json")))["train"]
    keep = np.isin(np.repeat(np.arange(len(z["npar"])), z["npar"]), train)
    return ids_to_csr(z["ids"], z["off"], F)[keep]


def ppmi_vectors(P, dim=100):
    """Word vectors from PPMI of paragraph-level co-occurrence (P: paragraphs x F, 0/1)."""
    C = (P.T.astype(np.float64) @ P.astype(np.float64)).toarray()
    df, N = np.diag(C).copy(), P.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(C * N / np.outer(df, df))
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)
    U, S, _ = np.linalg.svd(ppmi)
    return U[:, :dim] * np.sqrt(S[:dim])


def cosine_rows(VA, VB):
    na, nb = np.linalg.norm(VA, axis=1), np.linalg.norm(VB, axis=1)
    return np.where((na > 0) & (nb > 0), (VA * VB).sum(1) / np.maximum(na * nb, 1e-12), 0.0)


def scores(p, idf2, W):
    A, B = p["A"], p["B"]
    inter = np.asarray(A.multiply(B).sum(1)).ravel().astype(float)
    la, lb = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    jac = inter / np.maximum(la + lb - inter, 1)
    wa, wb = A @ idf2, B @ idf2
    tfidf = (A.multiply(B) @ idf2) / np.maximum(np.sqrt(wa * wb), 1e-12)
    VA, VB = (A @ W) / np.maximum(la, 1)[:, None], (B @ W) / np.maximum(lb, 1)[:, None]
    return {"jaccard": jac, "tfidf_cosine": tfidf, "shared_count": inter, "ppmi_svd_cosine": cosine_rows(VA, VB)}


def length_feats(p):
    # The counts are cast to signed integers before the difference (fix of 2026-10-25, HTM-B2): scipy sums the uint8
    # 0/1 matrices of pairdata.ids_to_csr in uint64, where la - lb wrapped around to about 1.8e19 whenever lb > la.
    la, lb = (np.asarray(p[k].sum(1)).ravel().astype(np.int64) for k in ("A", "B"))
    return np.c_[la, lb, np.abs(la - lb), np.minimum(la, lb), np.maximum(la, lb)].astype(float)


def check_version(v, idf2, W, vocab, rng):
    tr, dev = load_pairs(v, "train", F), load_pairs(v, "dev", F)
    for split, p in (("train", tr), ("dev", dev)):
        assert 2 * int(p["y"].sum()) == len(p["y"]), f"{v}/{split}: labels not balanced"
    out = {"n_train": len(tr["y"]), "n_dev": len(dev["y"]), "n_test": len(load_pairs(v, "test", F)["y"])}
    npar = np.load(os.path.join(DATA, "paragraphs_F3072.npz"))["npar"]
    out["position_A_train"] = {c: {"first_paragraph_share": round(float((tr["par_a"][tr["y"] == y] == 0).mean()), 4),
                                   "last_paragraph_share": round(float((tr["par_a"][tr["y"] == y] ==
                                                                        npar[tr["art_a"][tr["y"] == y]] - 1).mean()), 4)}
                               for c, y in (("pos", 1), ("neg", 0))}
    # z-test on paragraph A of the train pairs, with a label-permutation null for reference
    z = ztest(tr["A"], tr["y"])
    null = [int((np.abs(ztest(tr["A"], rng.permutation(tr["y"]))) > 3).sum()) for _ in range(200)]
    top = np.argsort(-np.abs(z))[:10]
    out["ztest"] = {"n_words_abs_z_over_3": int((np.abs(z) > 3).sum()), "chance_expected": round(Z_CHANCE, 2),
                    "perm_null_mean": round(float(np.mean(null)), 2), "perm_null_p95": float(np.percentile(null, 95)),
                    "top_words": [[vocab[i], round(float(z[i]), 2)] for i in top]}
    out["ztest"]["pass"] = out["ztest"]["n_words_abs_z_over_3"] <= 3
    out["one_paragraph_lr"] = lr_baseline(feats(tr, "A"), tr["y"], feats(dev, "A"), dev["y"])
    sh = {s: {**p, "B": p["B"][rng.permutation(p["B"].shape[0])]} for s, p in (("train", tr), ("dev", dev))}
    out["partner_shuffle_lr"] = lr_baseline(feats(sh["train"], "ALL"), tr["y"], feats(sh["dev"], "ALL"), dev["y"])
    gate = [out["ztest"]["pass"], out["one_paragraph_lr"]["ci_has_0.50"], out["partner_shuffle_lr"]["ci_has_0.50"]]
    if v == "ns":
        out["overlap"] = {}
        for split in ("train", "dev", "test"):
            for FF in (F, 3072):
                p = load_pairs(v, split, FF, allow_unchecked=True)
                out["overlap"][f"{split}_F{FF}_pairs_with_overlap"] = int((p["A"].multiply(p["B"]).sum(1) > 0).sum())
        assert sum(out["overlap"].values()) == 0, "ns overlap is not 0"
        out["k_hist_before_removal"] = {}
        for split in ("train", "dev", "test"):  # recompute k from the stored article / paragraph ids
            p, z = load_pairs(v, split, F), np.load(os.path.join(DATA, "paragraphs_F3072.npz"))
            M, st = ids_to_csr(z["ids"], z["off"], F), np.concatenate([[0], np.cumsum(z["npar"])])
            k = np.asarray(M[st[p["art_a"]] + p["par_a"]].multiply(M[st[p["art_b"]] + p["par_b"]]).sum(1)).ravel()
            h = {c: np.bincount(np.minimum(k[p["y"] == y], 10), minlength=11).tolist() for c, y in (("pos", 1), ("neg", 0))}
            assert h["pos"] == h["neg"], f"ns {split}: k histograms differ"
            out["k_hist_before_removal"][split] = h["pos"]
        Ltr, Ldev = length_feats(tr), length_feats(dev)
        sc = StandardScaler().fit(Ltr)
        m = LogisticRegression().fit(sc.transform(Ltr), tr["y"])
        out["length_lr"] = report(m.predict(sc.transform(Ldev)), m.decision_function(sc.transform(Ldev)), dev["y"])
        g = HistGradientBoostingClassifier(random_state=0).fit(Ltr, tr["y"])
        out["length_gbt"] = report(g.predict(Ldev), g.predict_proba(Ldev)[:, 1], dev["y"])
        out["mean_len_after_removal"] = {c: [round(float(Ldev[dev["y"] == y][:, :2].mean()), 2)]
                                         for c, y in (("pos", 1), ("neg", 0))}
        gate += [True, out["length_lr"]["ci_has_0.50"], out["length_gbt"]["ci_has_0.50"]]
    out["check1b_pass"] = bool(all(gate))
    s_tr, s_dev = scores(tr, idf2, W), scores(dev, idf2, W)
    out["baselines"] = {k: threshold_baseline(s_tr[k], tr["y"], s_dev[k], dev["y"]) for k in s_tr}
    out["baselines"]["lr_A_AND_B"] = lr_baseline(feats(tr, "AND"), tr["y"], feats(dev, "AND"), dev["y"])
    out["baselines"]["lr_A_B_AandB"] = lr_baseline(feats(tr, "ALL"), tr["y"], feats(dev, "ALL"), dev["y"])
    return out


def tables(res):
    L = ["| version | words with abs(z)>3 (perm. null p95) | one-paragraph LR acc [95% CI] | partner-shuffle acc [95% CI] |"
         " length-only LR / GBT acc [95% CI] | Check 1b |", "|---|---|---|---|---|---|"]
    for v, r in res.items():
        f = lambda d: f"{d['acc']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
        ln = f"{f(r['length_lr'])} / {f(r['length_gbt'])}" if "length_lr" in r else "-"
        L.append(f"| {v} | {r['ztest']['n_words_abs_z_over_3']} ({r['ztest']['perm_null_p95']:.0f}) | "
                 f"{f(r['one_paragraph_lr'])} | {f(r['partner_shuffle_lr'])} | {ln} | "
                 f"{'pass' if r['check1b_pass'] else 'FAIL'} |")
    L += ["", "| baseline (dev) | " + " | ".join(f"{v} acc / AUC" for v in res) + " |",
          "|---|" + "---|" * len(res)]
    for b in next(iter(res.values()))["baselines"]:
        L.append(f"| {b} | " + " | ".join(f"{r['baselines'][b]['acc']:.3f} / {r['baselines'][b]['auc']:.3f}"
                                           for r in res.values()) + " |")
    L.append("| always no | " + " | ".join(f"{r['baselines']['jaccard']['always_no']:.3f}" for r in res.values()) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["sa", "np", "ns"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0, run_dir = time.time(), os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        for f in ("data_checks.py", "pairdata.py"):
            shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), run_dir)
    P = train_paragraphs()
    df = np.asarray(P.sum(0)).ravel()
    idf2 = np.log(P.shape[0] / np.maximum(df, 1)) ** 2
    W = ppmi_vectors(P)
    vocab = load_vocab(F)
    res = {}
    for v in a.versions:
        res[v] = check_version(v, idf2, W, vocab, np.random.default_rng([a.seed, len(res)]))
        print(v, json.dumps(res[v]), f"({time.time() - t0:.0f}s)", flush=True)
    md = tables(res)
    print(md)
    out = {"run_dir": run_dir, "F": F, "results": res, "wall_s": round(time.time() - t0, 1)}
    if run_dir:
        json.dump(out, open(os.path.join(run_dir, "checks.json"), "w"), indent=1)
        open(os.path.join(run_dir, "checks.md"), "w").write(md + "\n")
        m = {}
        for v, r in res.items():
            m[f"{v}_check1b_pass"] = {"value": r["check1b_pass"], "label": "derived"}
            m[f"{v}_z_over_3"] = {"value": r["ztest"]["n_words_abs_z_over_3"], "label": "measured"}
            for k in ("one_paragraph_lr", "partner_shuffle_lr", "length_lr", "length_gbt"):
                if k in r:
                    m[f"{v}_{k}_acc"] = {"value": r[k]["acc"], "label": "measured", "ci95": r[k]["ci"]}
            for b, d in r["baselines"].items():
                m[f"{v}_{b}_acc"] = {"value": d["acc"], "label": "measured", "auc": d["auc"]}
        json.dump(m, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
