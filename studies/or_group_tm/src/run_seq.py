"""The OR-group Tsetlin machine on the synonym-sequence task of the upstream package's SequenceDemo.py (commit
f40b2a3): a sequence of L positions holds a permutation of L symbols, each symbol written with one of n_syn
interchangeable signs; bits = position x symbol x sign; label = parity of the permutation (inversions modulo 2).
The data is generated here with the demo's construction and a seeded generator (the demo itself draws from the
unseeded global generator).

Usage (from the study folder):
  .venv/bin/python -I src/run_seq.py --data-seed 1 --name e6_a_s1 [--syn-skew 1.0] [--syn-drift] [--perm-skew 1.0]
      [--perm-drift] [--save-data DIR]
Writes runs/<UTC>_<name>/{config.json, metrics.json, clauses.txt, state.npz} and appends a row to logs/RUNS_seq.md.
--save-data also writes the train and test sets in the text format the demo reads (one row of bits and the label).
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.stats import zipfian

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from orgroup_tm import OrGroupTM


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def inversions(p):
    return sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))


def make(n, L, n_syn, rng, perm_skew, syn_skew, perm_drift, syn_drift):
    """Order index [n], sign index [n, L], label [n], bit index inside the position block [n, L]."""
    perms = np.array(list(itertools.permutations(range(L))))
    P = len(perms)
    yu = np.array([inversions(p) % 2 for p in perms])
    idx = rng.integers(P, size=n) if perm_skew == 0 else zipfian.rvs(a=perm_skew, n=P, loc=-1, size=n, random_state=rng)
    syn = zipfian.rvs(a=syn_skew, n=n_syn, loc=-1, size=(n, L), random_state=rng)
    if perm_drift:
        idx = P - 1 - idx
    if syn_drift:
        syn = n_syn - 1 - syn
    bits = perms[idx] * n_syn + syn
    return idx, syn, yu[idx], bits.astype(np.int32)


def pack(bits):
    n, L = bits.shape
    bounds = np.arange(n * L + 1, dtype=np.int64)[: n * L + 1].reshape(-1)[: n * L + 1]
    b = np.zeros((n, L + 1), np.int64)
    b[:, 0] = np.arange(n) * L
    for s in range(L):
        b[:, s + 1] = b[:, 0] + s + 1
    del bounds
    return np.ascontiguousarray(bits.reshape(-1)), b


def dense(bits, L, n_syn):
    n = bits.shape[0]
    X = np.zeros((n, L * L * n_syn), np.uint8)
    for j in range(L):
        X[np.arange(n), j * L * n_syn + bits[:, j]] = 1
    return X


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-seed", type=int, required=True)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--n-syn", type=int, default=3)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--perm-skew", type=float, default=0.0)
    ap.add_argument("--syn-skew", type=float, default=0.0)
    ap.add_argument("--perm-drift", action="store_true")
    ap.add_argument("--syn-drift", action="store_true")
    ap.add_argument("--clauses", type=int, default=1000)
    ap.add_argument("--T", type=float, default=1)
    ap.add_argument("--mode", choices=["near", "all", "atleast"], default="atleast")
    ap.add_argument("--min-true", type=int, default=2)
    ap.add_argument("--reinforce", choices=["all", "critical", "share"], default="critical")
    ap.add_argument("--reject", choices=["random", "all", "weakest"], default="weakest")
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--init-state", type=int, default=-1)
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1, help="model seed")
    ap.add_argument("--save-data", default="")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = get_args(argv)
    t0 = time.time()
    rng = np.random.default_rng([20261112, a.data_seed])
    tr = make(a.n, a.L, a.n_syn, rng, a.perm_skew, a.syn_skew, False, False)
    te = make(a.n, a.L, a.n_syn, rng, a.perm_skew, a.syn_skew, a.perm_drift, a.syn_drift)
    F = a.L * a.n_syn
    ids_tr, b_tr = pack(tr[3])
    ids_te, b_te = pack(te[3])
    y_tr, y_te = tr[2].astype(np.int64), te[2].astype(np.int64)
    tm = OrGroupTM(a.clauses, F, sides=a.L, K=1, T=a.T, mode=a.mode, min_true=a.min_true, init_state=a.init_state, r=a.r,
                   max_seed=1, reject=a.reject, reinforce=a.reinforce, seed=a.seed)
    hist = []
    for p in range(1, a.passes + 1):
        t1 = time.time()
        tm.fit_pass(ids_tr, b_tr, y_tr)
        v = tm.votes(ids_te, b_te)
        hist.append({"pass": p, "fit_s": round(time.time() - t1, 3), "test_acc": float(((v >= 1).astype(int) == y_te).mean()),
                     "clauses_used": int((tm.nmem.sum(1) > 0).sum())})
        print("PASS", json.dumps(hist[-1]), flush=True)
    assert tm.nmem_consistent()
    v, fire = tm.votes(ids_te, b_te, return_fire=True)
    pred = (v >= 1).astype(int)
    key_tr = {r.tobytes() for r in tr[3]}
    unseen = np.array([r.tobytes() not in key_tr for r in te[3]])
    fired = fire.sum(0) > 0
    prec = (fire[:, fired] & (y_te == 1)[:, None]).sum(0) / np.maximum(fire[:, fired].sum(0), 1)
    rec = (fire[:, fired] & (y_te == 1)[:, None]).sum(0) / max(int((y_te == 1).sum()), 1)
    label_of = {}
    for r, yy in zip(tr[3], y_tr):
        label_of[r.tobytes()] = int(yy)
    memo = np.array([label_of.get(r.tobytes(), 0) for r in te[3]])
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000).fit(dense(tr[3], a.L, a.n_syn), y_tr)
    lr_acc = float((lr.predict(dense(te[3], a.L, a.n_syn)) == y_te).mean())
    used = tm.nmem.sum(1) > 0
    sizes = tm.nmem[used]
    final = {"test_acc": float((pred == y_te).mean()), "train_acc": float(((tm.votes(ids_tr, b_tr) >= 1).astype(int) == y_tr).mean()),
             "covered_share_of_label1": float(pred[y_te == 1].mean()), "fires_on_label0": float(pred[y_te == 0].mean()),
             "unseen_patterns_share": float(unseen.mean()),
             "test_acc_on_unseen_patterns": float((pred[unseen] == y_te[unseen]).mean()) if unseen.any() else None,
             "clauses_used": int(used.sum()), "clauses_that_fire_on_test": int(fired.sum()),
             "mean_clause_precision": float(prec.mean()) if fired.any() else None,
             "mean_clause_recall": float(rec.mean()) if fired.any() else None,
             "mean_signs_per_group": float(sizes.mean()) if used.any() else None,
             "share_of_groups_with_all_signs": float((sizes == a.n_syn).mean()) if used.any() else None,
             "memoriser_acc": float((memo == y_te).mean()), "logistic_regression_acc": lr_acc,
             "events": tm.event_counts(), "fit_s_per_pass": round(float(np.mean([h["fit_s"] for h in hist[1:] or hist])), 3),
             "wall_s": round(time.time() - t0, 1)}
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(a.out_root, f"{stamp}_{a.name}")
    os.makedirs(run_dir, exist_ok=False)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(a), "orgroup_tm_sha256": sha(os.path.join(SRC, "orgroup_tm.py")), "run_utc": stamp}, f, indent=1)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"final": final, "history": hist}, f, indent=1)
    np.savez_compressed(os.path.join(run_dir, "state.npz"), state=tm.state, pol=tm.pol)
    with open(os.path.join(run_dir, "clauses.txt"), "w") as f:
        for c in np.flatnonzero(used)[:40]:
            parts = []
            for j in range(a.L):
                m = np.flatnonzero(tm.state[c, j] >= 0)
                parts.append("(" + " | ".join(f"{chr(65 + i // a.n_syn)}{i % a.n_syn + 1}" for i in m) + ")")
            f.write(f"clause {c}: " + " AND ".join(parts) + "\n")
    if a.save_data:
        os.makedirs(a.save_data, exist_ok=True)
        np.savetxt(os.path.join(a.save_data, "SequenceTrainingData.txt"), np.c_[dense(tr[3], a.L, a.n_syn), y_tr], fmt="%d")
        np.savetxt(os.path.join(a.save_data, "SequenceTestingData.txt"), np.c_[dense(te[3], a.L, a.n_syn), y_te], fmt="%d")
    print("FINAL", json.dumps(final), flush=True)
    if not a.no_log:
        log = os.path.join(ROOT, "logs", "RUNS_seq.md")
        new = not os.path.exists(log)
        with open(log, "a") as f:
            if new:
                f.write("# Synonym-sequence runs (E6, exploratory; every run, newest last)\n\n| run folder | data seed, sign skew, order skew, drift | "
                        "clauses, T, mode, min true, passes | test acc | label-1 covered / label-0 fired | acc on unseen patterns (share) | "
                        "clause precision / recall | clauses used, all-sign groups | memoriser / logistic regression |\n|---|---|---|---|---|---|---|---|---|\n")
            drift = "+".join(x for x, on in (("sign drift", a.syn_drift), ("order drift", a.perm_drift)) if on) or "none"
            f.write(f"| {os.path.basename(run_dir)} | {a.data_seed}, {a.syn_skew:g}, {a.perm_skew:g}, {drift} | {a.clauses}, {a.T:g}, {a.mode}, "
                    f"{a.min_true}, {a.passes} | {final['test_acc']:.4f} | {final['covered_share_of_label1']:.4f} / {final['fires_on_label0']:.4f} | "
                    f"{final['test_acc_on_unseen_patterns']:.4f} ({final['unseen_patterns_share']:.3f}) | {final['mean_clause_precision']:.4f} / "
                    f"{final['mean_clause_recall']:.5f} | {final['clauses_used']}, {final['share_of_groups_with_all_signs']:.3f} | "
                    f"{final['memoriser_acc']:.4f} / {final['logistic_regression_acc']:.4f} |\n")
    return run_dir, final


if __name__ == "__main__":
    main()
