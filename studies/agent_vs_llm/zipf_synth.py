"""PLAN_zipf_synth.md: per-pattern recall of a Tsetlin Machine on Zipf-distributed planted conjunctions (CPU).

usage (tm_language_model's .venv-tm, OMP_NUM_THREADS=2, nice 19):
  python -I zipf_synth.py --s 8 --clauses 80 --seed 1 --out runs/zipf/s8_c80_seed1.json [--epochs 30] [--n 20000]
  python -I zipf_synth.py --lr --seed 1 --out runs/zipf/lr_seed1.json
"""
import argparse
import json
import os
import time

import numpy as np

D, K, P_BG, ARITY = 300, 60, 0.05, 3


def generate(seed_patterns, seed_rows, n, d=D, k=K, p_bg=P_BG):
    """X (n, d) uint32, y (n,), pid (n,) = planted pattern of positives (-1 for negatives), the patterns, Zipf weights.
    The patterns come from seed_patterns, the rows from seed_rows (train and test: same patterns, other rows)."""
    feats = np.random.default_rng(seed_patterns).permutation(d)[: k * ARITY].reshape(k, ARITY)  # disjoint patterns
    rng = np.random.default_rng(seed_rows)
    zipf = 1.0 / np.arange(1, k + 1)
    zipf /= zipf.sum()
    X = (rng.random((n, d)) < p_bg).astype(np.uint32)
    y = np.zeros(n, np.uint32)
    y[: n // 2] = 1
    rng.shuffle(y)
    pid = np.full(n, -1)
    pos = np.where(y == 1)[0]
    pid[pos] = rng.choice(k, size=len(pos), p=zipf)
    X[pos[:, None], feats[pid[pos]]] = 1
    # break every full pattern in negatives (and stray extra full patterns in positives) by clearing one random bit
    for j in range(k):
        full = np.all(X[:, feats[j]] == 1, axis=1) & (pid != j)
        rows = np.where(full)[0]
        if len(rows):
            X[rows, feats[j][rng.integers(0, ARITY, len(rows))]] = 0
    return X, y, pid, feats, zipf


def per_pattern(pred, y, pid, k=K):
    rec = [float(pred[pid == j].mean()) if np.any(pid == j) else None for j in range(k)]
    return rec, float(pred[y == 0].mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float)
    ap.add_argument("--clauses", type=int)
    ap.add_argument("--lr", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Xtr, ytr, ptr, feats, zipf = generate(1000 + a.seed, 2000 + a.seed, a.n)
    Xte, yte, pte, feats2, _ = generate(1000 + a.seed, 3000 + a.seed, a.n)
    assert np.array_equal(feats, feats2)
    f_train = np.array([np.mean(ptr[ytr == 1] == j) for j in range(K)])
    t0 = time.time()
    res = {"plan": "PLAN_zipf_synth.md", "seed": a.seed, "n": a.n, "d": D, "k": K, "p_bg": P_BG,
           "f_train": f_train.tolist(), "zipf": zipf.tolist()}
    if a.lr:
        from sklearn.linear_model import LogisticRegression
        nv = a.n // 10
        best = None
        for c in (0.1, 1.0, 10.0):
            m = LogisticRegression(C=c, max_iter=2000).fit(Xtr[nv:], ytr[nv:])
            acc = (m.predict(Xtr[:nv]) == ytr[:nv]).mean()
            if best is None or acc > best[0]:
                best = (acc, c)
        m = LogisticRegression(C=best[1], max_iter=2000).fit(Xtr, ytr)
        pred = m.predict(Xte)
        res.update({"learner": "lr", "C": best[1]})
    else:
        from tmu.models.classification.vanilla_classifier import TMClassifier
        tm = TMClassifier(a.clauses, a.clauses, a.s, platform="CPU", seed=a.seed)
        for _ in range(a.epochs):
            tm.fit(Xtr, ytr)
        pred = tm.predict(Xte)
        res.update({"learner": "tm", "s": a.s, "clauses": a.clauses, "T": a.clauses, "epochs": a.epochs})
    rec, fpr = per_pattern(pred, yte, pte)
    res.update({"test_acc": float((pred == yte).mean()), "recall_by_pattern": rec, "fpr_negatives": fpr,
                "seconds": round(time.time() - t0, 1), "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as h:
        json.dump(res, h)
    print(json.dumps({k: res[k] for k in ("learner", "seed", "test_acc", "fpr_negatives", "seconds")}))


if __name__ == "__main__":
    main()
