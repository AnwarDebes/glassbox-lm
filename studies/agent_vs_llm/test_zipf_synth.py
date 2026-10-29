"""Checks of zipf_synth.py: the generator's guarantees and a tiny end-to-end run (CPU).

usage: OMP_NUM_THREADS=2 .../.venv-tm/bin/python -I test_zipf_synth.py
"""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zipf_synth as Z
fails = 0
def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name); fails += 0 if cond else 1
X, y, pid, feats, zipf = Z.generate(1, 2, 4000)
full = np.stack([np.all(X[:, f] == 1, axis=1) for f in feats], 1)
check("negatives hold no full pattern", not full[y == 0].any())
check("each positive holds exactly its own pattern", np.all(full[y == 1].sum(1) == 1) and np.all(full[y == 1].argmax(1) == pid[y == 1]))
check("half the rows are positive", abs(y.mean() - 0.5) < 1e-9)
check("patterns are disjoint", len(set(feats.ravel())) == feats.size)
f = np.array([np.mean(pid[y == 1] == j) for j in range(Z.K)])
check("pattern 1 is the most frequent, close to its Zipf weight", f.argmax() == 0 and abs(f[0] - zipf[0]) < 0.03)
X2, _, _, feats2, _ = Z.generate(1, 3, 4000)
check("same patterns, other rows for another row seed", np.array_equal(feats, feats2) and not np.array_equal(X, X2))
with tempfile.TemporaryDirectory() as t:
    Z.main(["--s", "8", "--clauses", "20", "--seed", "1", "--epochs", "2", "--n", "2000", "--out", os.path.join(t, "o.json")])
    r = json.load(open(os.path.join(t, "o.json")))
    check("TM run writes per-pattern recall", r["learner"] == "tm" and len(r["recall_by_pattern"]) == Z.K and 0 <= r["test_acc"] <= 1)
    Z.main(["--lr", "--seed", "1", "--n", "2000", "--out", os.path.join(t, "l.json")])
    r = json.load(open(os.path.join(t, "l.json")))
    check("LR run writes per-pattern recall", r["learner"] == "lr" and len(r["recall_by_pattern"]) == Z.K)
print("FAILED" if fails else "PASSED", f"({fails} failed)")
sys.exit(1 if fails else 0)
