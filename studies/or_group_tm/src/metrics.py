"""Scores shared by the runners of this line. auc and tuned_acc follow studies/hierarchical_tm/src/htm_driver.py."""
import numpy as np
from scipy.stats import rankdata


def auc(score, y):
    y = np.asarray(y)
    r, n1 = rankdata(score), int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def tuned_acc(score, y):
    """Accuracy at the best cut-off 'predict 1 iff score >= c', chosen on these same pairs."""
    score, y = np.asarray(score), np.asarray(y)
    o = np.argsort(-score, kind="stable")
    s, yy = score[o], y[o].astype(np.int64)
    tp, fp, n0 = np.cumsum(yy), np.cumsum(1 - yy), len(y) - int(yy.sum())
    end = np.r_[s[1:] != s[:-1], True]
    return float(max(((tp[end] + n0 - fp[end]) / len(y)).max(), n0 / len(y)))


def plain_acc(votes, y, cut=1):
    """Accuracy at the natural cut-off: predict 1 iff the vote sum is at least `cut`."""
    return float(((np.asarray(votes) >= cut).astype(np.int64) == np.asarray(y).astype(np.int64)).mean())


def group_stats(groups, n_hidden, m):
    """Purity and coverage as in pcp_purity.py: a word's hidden group is word % n_hidden; groups of >= 2 words."""
    groups = [np.asarray(w) for w in groups if len(w) >= 2]
    if not groups:
        return {"n": 0, "purity": None, "coverage": None, "mean_words": None}
    pur, cov = [], []
    for w in groups:
        cnt = np.bincount(w % n_hidden, minlength=n_hidden)
        pur.append(cnt.max() / len(w))
        cov.append(cnt.max() / m)
    return {"n": len(groups), "purity": round(float(np.mean(pur)), 4), "coverage": round(float(np.mean(cov)), 4),
            "mean_words": round(float(np.mean([len(w) for w in groups])), 2)}


def groups_to_include_bits(groups_per_clause, n_groups, V):
    """Include bits in the layout that pcp_world.recovery reads: [2 * clauses, n_groups * A, V], the clauses at the
    even indices, one member word per option, A = the largest group size. A group with fewer members repeats its
    first member in the spare options, so that no option is empty; a group without members keeps all its options
    empty and is therefore not counted by recovery()."""
    C = len(groups_per_clause)
    A = max([1] + [len(w) for gs in groups_per_clause for w in gs])
    inc = np.zeros((2 * C, n_groups * A, V), bool)
    for c, gs in enumerate(groups_per_clause):
        for j, words in enumerate(gs):
            if len(words) == 0:
                continue
            for a in range(A):
                inc[2 * c, j * A + a, words[a] if a < len(words) else words[0]] = True
    return inc, A
