"""Planted concept pairs (pcp): the artificial tests L1 to L3 of the ladder (design decision E2; gates.md Check 2a).

The world: V words in G hidden groups of m = V/G words; word i is in group i % G (at L1 these are the even
and odd words of Granmo's MultiConceptDemo). A paragraph is w distinct words of one group. A pair is the input
X = [paragraph A bits | paragraph B bits] (2V columns), with Y = 1 iff A and B come from the same group.
  L1  V=16,  G=2, w=1  Granmo's MultiConcept demo as paragraph pairs (one word per side; same parity = 1)
  L2  V=64,  G=8, w=1  more topics
  L3  V=256, G=8, w=2  A and B use DIFFERENT words of their group, so word overlap is 0 in every pair, and
                       25% of the same-group word pairs are held out of training
At L3 the 'cross pairs' of a pair are the (a, b) with a in A and b in B. The train, dev and test sets use
seen cross pairs only. The 'unseen' set uses held-out cross pairs only, so only a model that generalises
from words to groups can score it.

Sampling, so that one paragraph alone says nothing about Y:
  positive  group t uniform; A and B are w words of t each (rejection sampling for the L3 rules);
  negative  groups t1 != t2 uniform; A is one side of a fresh positive of group t1, B one side of a fresh
            positive of group t2. So every paragraph of either class is a side of a positive of the same set.
Labels are exactly balanced, the side order is random, and every set is fixed by (rung, seed, set name).

Self-checks (python pcp_world.py, written to data/pcp/): labels balanced; the construction rules; a logistic
regression on one paragraph, fitted on 20,000 and tested on 20,000 other fresh pairs per rung and seed (its
95% Wilson CI must include 0.50); at L3 word overlap is 0 in every pair. worlds.json keeps the hidden groups
and the held-out pairs; recovery() scores learned OR-groups against them (definition in gates.md).
"""
import argparse, json, os, sys
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
RUNGS = {"L1": dict(V=16, G=2, w=1, distinct=False, holdout=0.0),
         "L2": dict(V=64, G=8, w=1, distinct=False, holdout=0.0),
         "L3": dict(V=256, G=8, w=2, distinct=True, holdout=0.25)}
SETS = {"train": 1, "dev": 2, "test": 3, "unseen": 4,  # random stream of each pair set
        "check_fit": 5, "check_eval": 6, "check_fit_unseen": 7, "check_eval_unseen": 8}


class World:
    def __init__(self, rung, seed):
        self.rung, self.seed = rung, int(seed)
        self.__dict__.update(RUNGS[rung])
        self.m = self.V // self.G
        self.groups = np.arange(self.V).reshape(self.m, self.G).T  # [G, m]: group g = words g, g+G, g+2G, ...
        self.vocab = [f"g{i % self.G}w{i}" for i in range(self.V)]  # the name shows the hidden group, for reading rules
        self.held = np.zeros((self.V, self.V), bool)  # held-out same-group word pairs (symmetric)
        if self.holdout:
            iu = np.triu_indices(self.m, 1)
            pairs = np.concatenate([np.c_[g[iu[0]], g[iu[1]]] for g in self.groups])
            pick = pairs[self.rng(0).choice(len(pairs), round(self.holdout * len(pairs)), replace=False)]
            self.held[pick[:, 0], pick[:, 1]] = self.held[pick[:, 1], pick[:, 0]] = True

    def rng(self, stream):
        return np.random.default_rng([20261001, int(self.rung[1:]), self.seed, stream])

    def positives(self, rng, t, unseen):
        """Word ids A, B [n, w] of positive pairs of groups t. With distinct, A and B share no word. With held-out
        pairs, every cross pair is seen (unseen=False) or every cross pair is held out (unseen=True)."""
        n, w = len(t), self.w
        A, B, todo = np.empty((n, w), np.int64), np.empty((n, w), np.int64), np.arange(n)
        while todo.size:
            k = rng.random((todo.size, self.m)).argsort(1)  # a random order of the group's words
            pa = k[:, :w]
            pb = k[:, w:2 * w] if self.distinct else rng.random((todo.size, self.m)).argsort(1)[:, :w]
            g = self.groups[t[todo]]
            a, b = np.take_along_axis(g, pa, 1), np.take_along_axis(g, pb, 1)
            cross = self.held[a[:, :, None], b[:, None, :]].reshape(todo.size, -1)
            ok = cross.all(1) if unseen else ~cross.any(1)
            A[todo[ok]], B[todo[ok]], todo = a[ok], b[ok], todo[~ok]
        return A, B

    def sample(self, n, name):
        """Pair set `name` (a key of SETS), n pairs (n even): X uint32 [n, 2V], y uint32 [n], word ids A, B [n, w]."""
        unseen = name.endswith("unseen")
        assert self.holdout or not unseen, "only rungs with held-out pairs have an unseen set"
        rng, h = self.rng(SETS[name]), n // 2
        A1, B1 = self.positives(rng, rng.integers(self.G, size=h), unseen)
        t1 = rng.integers(self.G, size=h)
        t2 = (t1 + rng.integers(1, self.G, size=h)) % self.G  # uniform over the groups other than t1
        A = np.concatenate([A1, self.positives(rng, t1, unseen)[0]])
        B = np.concatenate([B1, self.positives(rng, t2, unseen)[1]])
        y = (np.arange(2 * h) < h).astype(np.uint32)
        swap = rng.random(2 * h) < 0.5  # random side order
        A[swap], B[swap] = B[swap], A[swap]
        o = rng.permutation(2 * h)
        A, B, y = A[o], B[o], y[o]
        X = np.zeros((2 * h, 2 * self.V), np.uint32)
        r = np.arange(2 * h)[:, None]
        X[r, A], X[r, self.V + B] = 1, 1
        return X, y, A, B


def recovery(inc, groups, A, K, thr=0.8):
    """Group recovery (gates.md): the share of hidden groups matched, with Jaccard >= thr, by a counted OR-group.
    inc: include bits [clauses, 2*K*A, V] of the class-1 machine (leaf a + A*(k + K*p)). Counted OR-groups come
    from its positive (even) clauses; a group with an empty option is ALWAYS-TRUE and not counted; repeated
    words are removed; at least 2 distinct words. recovery_ignoring_empty also counts ALWAYS-TRUE groups
    (a diagnostic, not the gate)."""
    counted, lenient, n_all, n_always = [], [], 0, 0
    for c in range(0, inc.shape[0], 2):
        for j in range(2 * K):  # j = k + K*p: OR-group k of paragraph p
            opts = inc[c, j * A:(j + 1) * A]
            words, always = set(np.flatnonzero(opts.any(0)).tolist()), not opts.any(1).all()
            n_all, n_always = n_all + 1, n_always + always
            if len(words) >= 2:
                lenient.append(words)
                if not always:
                    counted.append(words)
    share = lambda found: float(np.mean([any(len(g & s) / len(g | s) >= thr for s in found)
                                         for g in map(set, np.asarray(groups).tolist())]))
    return {"recovery": share(counted), "recovery_ignoring_empty": share(lenient),
            "n_counted_groups": len(counted), "always_true_group_share": n_always / max(n_all, 1)}


def check(w, n):
    """Self-checks of one world (see the module docstring). Returns a dict with 'passed'."""
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, SRC)
    from data_checks import wilson
    out, ok = {}, True
    for mode in ("seen", "unseen") if w.holdout else ("seen",):
        sfx = "_unseen" if mode == "unseen" else ""
        (Xf, yf, Af, Bf), (Xe, ye, Ae, Be) = w.sample(n, "check_fit" + sfx), w.sample(n, "check_eval" + sfx)
        r = {}
        for X, y, A, B in ((Xf, yf, Af, Bf), (Xe, ye, Ae, Be)):
            ga, gb = A % w.G, B % w.G
            cross = w.held[A[:, :, None], B[:, None, :]].reshape(len(y), -1)[y == 1]
            overlap = (X[:, :w.V] & X[:, w.V:]).sum(1) > 0
            rules = {"balanced": 2 * int(y.sum()) == len(y),
                     "one_group_per_side": bool((ga == ga[:, :1]).all() and (gb == gb[:, :1]).all()),
                     "label_is_same_group": bool((y == (ga[:, 0] == gb[:, 0])).all()),
                     "w_distinct_words_per_side": bool((X[:, :w.V].sum(1) == w.w).all() and (X[:, w.V:].sum(1) == w.w).all()),
                     "overlap_zero_every_pair": bool(not overlap.any()) if w.distinct else None,
                     "positives_cross_pairs_ok": bool(cross.all() if mode == "unseen" else not cross.any())}
            ok &= all(v for v in rules.values() if v is not None)
            r.setdefault("rules", []).append(rules)
            r.setdefault("overlap_share_pos_neg", []).append([float(overlap[y == 1].mean()), float(overlap[y == 0].mean())])
        lr = LogisticRegression(max_iter=2000).fit(Xf[:, :w.V], yf)
        acc = float((lr.predict(Xe[:, :w.V]) == ye).mean())
        ci = wilson(acc, len(ye))
        r.update(one_paragraph_acc=round(acc, 4), one_paragraph_ci=ci, one_paragraph_ci_has_050=ci[0] <= 0.5 <= ci[1],
                 n_fit=len(yf), n_eval=len(ye))
        ok &= r["one_paragraph_ci_has_050"]
        out[mode] = r
    out["passed"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser(description="pcp self-checks; writes worlds.json and checks.json")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--n", type=int, default=20000, help="fresh pairs to fit, and again to test, the one-paragraph LR")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(SRC), "data", "pcp"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    worlds, checks = {}, {}
    for rung in RUNGS:
        for seed in map(int, a.seeds.split(",")):
            w, key = World(rung, seed), f"{rung}_seed{seed}"
            worlds[key] = {**RUNGS[rung], "groups": w.groups.tolist(), "vocab": w.vocab,
                           "held_out_pairs": np.argwhere(np.triu(w.held)).tolist()}
            checks[key] = check(w, a.n)
            print(key, json.dumps(checks[key]), flush=True)
    passed = all(c["passed"] for c in checks.values())
    json.dump(worlds, open(os.path.join(a.out, "worlds.json"), "w"))
    json.dump({"passed": passed, **checks}, open(os.path.join(a.out, "checks.json"), "w"), indent=1)
    if os.environ.get("TMLM_RUN_DIR"):
        m = {"all_checks_passed": passed, **{f"{k}_{mode}_one_paragraph_acc": c[mode]["one_paragraph_acc"]
                                             for k, c in checks.items() for mode in ("seen", "unseen") if mode in c}}
        json.dump({k: {"value": v, "label": "measured"} for k, v in m.items()},
                  open(os.path.join(os.environ["TMLM_RUN_DIR"], "metrics.json"), "w"), indent=1)
    print("ALL CHECKS PASSED" if passed else "SOME CHECK FAILED", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
