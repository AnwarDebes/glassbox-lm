"""Correctness tests of the A2 code on small synthetic streams (no PTB data).

    python scripts/a2_selftest.py

- exchange: the objective never falls, the last value equals the objective recomputed from the
  assignment, no single move of a word improves it at the end, and the tree levels are nested.
- random codes: the same class-size distribution as the base code at every level.
- class member: the full vector sums to 1 and is above 0, the fast path equals the vector entry,
  and a one-class-per-word code gives the word Kneser-Ney model itself.
- codeeval: n-gram keys and the unseen-bigram mask against plain Python sets.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tally.floor import codeeval, codes  # noqa: E402
from tally.floor import exchange as ex  # noqa: E402
from tally.floor.classlm import ClassMember  # noqa: E402
from tally.floor.kn import KN  # noqa: E402

OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(("ok   " if cond else "FAIL ") + name, flush=True)


def synthetic(V=300, eos=7, n=30000, seed=1):
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 10, V)
    T = rng.dirichlet(np.ones(10) * 0.3, 10)
    x, g = [], 0
    for _ in range(n):
        g = rng.choice(10, p=T[g])
        ws = np.flatnonzero(grp == g)
        w = ws[min(int(rng.zipf(1.5)) - 1, len(ws) - 1)]
        x.append(w if rng.random() > 0.05 else eos)
    return np.array(x), V, eos


def main():
    x, V, eos = synthetic()
    tr, va = x[:25000], x[25000:]
    n = V + 1
    l, r, c = ex.bigrams(tr, eos, n, V)
    freq = np.bincount(tr, minlength=n).astype(float)
    mov = np.ones(n, bool)
    mov[[eos, V]] = False
    fix = np.zeros(n, int)
    fix[eos], fix[V] = 12, 13
    cls, trace, moves = ex.exchange(l, r, c, n, 12, mov, freq, fix)
    check("exchange objective monotone", np.all(np.diff(trace) >= -1e-6))
    check("exchange final objective equals recomputed", np.isclose(trace[-1], ex.objective(cls, l, r, c)))
    if moves[-1] == 0:
        f0, better = ex.objective(cls, l, r, c), 0
        for w in np.flatnonzero(mov)[:80]:
            if (cls[mov] == cls[w]).sum() <= 1:
                continue
            for b in range(12):
                if b != cls[w]:
                    c2 = cls.copy()
                    c2[w] = b
                    better += ex.objective(c2, l, r, c) > f0 + 1e-6
        check("exchange: no improving single move at convergence", better == 0)
    C, info = codes.exchange_tree(tr, V, eos, levels=(4, 12, 40))
    nested = all(len(np.unique(C[C[:, i + 1] == k, i])) == 1 for i in range(2) for k in np.unique(C[:, i + 1]))
    check("tree nested (each finer class inside one coarser class)", nested)
    check("eos alone at every level", all((C[:, i] == C[eos, i]).sum() == 1 for i in range(3)))
    R, _ = codes.random_code(C, eos, 3)
    same = all(sorted(np.bincount(R[:, i])) == sorted(np.bincount(C[:, i])) for i in range(3))
    check("random code keeps the class-size distribution", same)
    check("random code nested", all(len(np.unique(R[R[:, i + 1] == k, i])) == 1
                                    for i in range(2) for k in np.unique(R[:, i + 1])))
    for lev in range(3):
        cm = ClassMember(tr, C[:, lev], eos, order=4)
        pos = np.arange(0, len(va), 37)
        D = cm.dists(va, pos)
        fast = cm.logprob(va)
        check(f"class member level {lev}: sum 1, above 0, fast = vector",
              np.abs(D.sum(1) - 1).max() < 1e-9 and D.min() > 0
              and np.abs(np.log(D[np.arange(len(pos)), va[pos]]) - fast[pos]).max() < 1e-9)
    ident = np.arange(V)
    cm = ClassMember(tr, ident, eos, order=4)
    kn = KN(tr, V, eos, order=4)
    seen = np.bincount(tr, minlength=V) > 0
    m = seen[va]
    check("one class per word equals word KN (seen words)",
          np.abs(cm.logprob(va)[m] - kn.logprob(va)[m]).max() < 1e-9)
    # n-gram keys and unseen bigram mask against sets
    ok = True
    for nn in (1, 2, 3):
        keys = codeeval._ngram_keys(tr, eos, nn, V + 1)
        ref = set()
        sent = []
        for w in tr:
            sent.append(int(w))
            if w == eos:
                for i in range(len(sent) - nn + 1):
                    ref.add(tuple(sent[i:i + nn]))
                sent = []
        for i in range(len(sent) - nn + 1):
            ref.add(tuple(sent[i:i + nn]))
        ok &= len(np.unique(keys)) == len(ref)
    check("distinct n-gram counts equal set counts", ok)
    um = codeeval.unseen_bigram_mask(tr, va, eos, V)
    trs = set()
    p = V
    for w in tr:
        trs.add((p, int(w)))
        p = V if w == eos else int(w)
    ref, p = [], V
    for w in va:
        ref.append((p, int(w)) not in trs)
        p = V if w == eos else int(w)
    check("unseen bigram mask equals set reference", np.array_equal(um, np.array(ref)))
    print("ALL OK" if all(OK) else "SOME FAILED")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
