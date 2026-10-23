"""B1 search arms at equal evaluations: greedy (best added literal) and random clauses.

Both search one member at a time, round robin, on the same team and counts as the Tsetlin
arms. A step for member i: draw B fresh samples; member i is silent during the step; every
candidate clause is evaluated on every sample of the batch and counted; the score of a
candidate is the bits the team (without i) saves on the batch when the candidate speaks with
the evidence counted on the batch:
    score_c = sum over batch samples where c fires of L(z_-i) - L(z_-i + w_c)
The best candidate is kept (ties: the current clause). The current clause and the empty clause
(silent, score 0) are always candidates.
    greedy: current clause plus one literal, for every literal that is not in it and whose
            negation is not in it (forward greedy; "empty" allows a restart)
    random: P_RAND clauses with a length uniform in 1..8 on distinct random variables and signs
Evaluations per step: B * (M - 1) for the other members plus B * (number of candidates).
The sample axis of these arms is evaluations / M ("evaluation-equivalent samples"), so they
spend exactly M clause evaluations per sample, as a Tsetlin team does.
"""
import numpy as np
from numba import njit

from tally.b1.team import (P_ALPHA, WORDS, clause_hash, fires_mask, logit, loss_bits, team_eval,
                           update_dup)

B_SEARCH = 512
P_RAND = 64
RAND_MAXLEN = 8


@njit(cache=True)
def greedy_cands(pw, nw, cp, cn):
    """Fill cp, cn with [current, empty, current + literal ...]; return count."""
    for k in range(WORDS):
        cp[0, k] = pw[k]
        cn[0, k] = nw[k]
        cp[1, k] = 0
        cn[1, k] = 0
    c = 2
    for v in range(256):
        b = np.uint64(1) << np.uint64(v & 63)
        wv = v >> 6
        if (pw[wv] & b) != 0 or (nw[wv] & b) != 0:
            continue
        for sign in range(2):
            for k in range(WORDS):
                cp[c, k] = pw[k]
                cn[c, k] = nw[k]
            if sign == 1:
                cp[c, wv] |= b
            else:
                cn[c, wv] |= b
            c += 1
    return c


@njit(cache=True)
def search_step(X, Y, i, cp, cn, C, prm, pos, neg, chash, has_dup, n1, n, bg, stats):
    """One search step for member i on batch X, Y with candidates cp[:C], cn[:C]."""
    M = pos.shape[0]
    alpha = prm[P_ALPHA]
    B = X.shape[0]
    # silence member i
    old_h = chash[i]
    keep_n1, keep_n = n1[i], n[i]
    for k in range(WORDS):
        pos[i, k] = 0
        neg[i, k] = 0
    chash[i] = 0
    update_dup(chash, has_dup, i, old_h)
    fire = np.zeros(M, dtype=np.bool_)
    active = np.zeros(M, dtype=np.bool_)
    alt = np.zeros(M, dtype=np.int64)
    dsil = np.zeros(M, dtype=np.bool_)
    w = np.zeros(M)
    zm = np.zeros(B)
    cf = np.zeros((C, B), dtype=np.bool_)
    c1 = np.zeros(C)
    cnt = np.zeros(C)
    cand_live = np.zeros(C, dtype=np.bool_)
    for c in range(C):
        for k in range(WORDS):
            if cp[c, k] != 0 or cn[c, k] != 0:
                cand_live[c] = True
    for t in range(B):
        x = X[t]
        y = Y[t]
        z, z_bg, any_active = team_eval(x, pos, neg, chash, has_dup, n1, n, bg, alpha,
                                        fire, active, alt, dsil, w)
        zm[t] = z
        for c in range(C):
            if c == 1 or not cand_live[c]:
                continue            # empty clauses are silent and never scored
            if fires_mask(x, cp[c], cn[c]):
                cf[c, t] = True
                c1[c] += y
                cnt[c] += 1.0
        for j in range(M):
            if fire[j]:
                n1[j] += y
                n[j] += 1.0
        if not any_active:
            bg[0] += y
            bg[1] += 1.0
    stats[0] += B
    stats[1] += B * (M - 1) + B * C
    p_bg = (bg[0] + 0.5 * alpha) / (bg[1] + alpha)
    z_bg = logit(p_bg)
    best = 0
    best_score = -1e300
    for c in range(C):
        sc = 0.0
        if cand_live[c] and cnt[c] > 0:
            wc = max(0.0, logit((c1[c] + alpha * p_bg) / (cnt[c] + alpha)) - z_bg)
            for t in range(B):
                if cf[c, t]:
                    sc += loss_bits(zm[t], Y[t]) - loss_bits(zm[t] + wc, Y[t])
        if sc > best_score + 1e-12:
            best_score = sc
            best = c
    for k in range(WORDS):
        pos[i, k] = cp[best, k]
        neg[i, k] = cn[best, k]
    chash[i] = clause_hash(pos[i], neg[i])
    update_dup(chash, has_dup, i, np.uint64(0))
    if best == 0:
        n1[i] = keep_n1 + c1[0]
        n[i] = keep_n + cnt[0]
    else:
        n1[i] = c1[best]
        n[i] = cnt[best]
        stats[2] += 1
    return best


def random_cands(pw, nw, rng):
    C = P_RAND + 2
    cp = np.zeros((C, WORDS), dtype=np.uint64)
    cn = np.zeros((C, WORDS), dtype=np.uint64)
    cp[0], cn[0] = pw, nw
    for c in range(2, C):
        L = int(rng.integers(1, RAND_MAXLEN + 1))
        for v in rng.choice(256, L, replace=False):
            m = cp if rng.integers(0, 2) else cn
            m[c, v // 64] |= np.uint64(1) << np.uint64(v % 64)
    return cp, cn, C


class Search:
    def __init__(self, team, kind, seed):
        self.T = team
        self.kind = kind
        self.rng = np.random.default_rng([seed, team.M, 11])
        self.next = 0
        self.cp = np.zeros((2 + 512, WORDS), dtype=np.uint64)
        self.cn = np.zeros((2 + 512, WORDS), dtype=np.uint64)

    def step(self, X, Y):
        T = self.T
        i = self.next
        self.next = (self.next + 1) % T.M
        if self.kind == "greedy":
            C = greedy_cands(T.pos[i], T.neg[i], self.cp, self.cn)
            cp, cn = self.cp, self.cn
        else:
            cp, cn, C = random_cands(T.pos[i], T.neg[i], self.rng)
        search_step(X, Y, i, cp, cn, C, T.prm, T.pos, T.neg, T.chash, T.has_dup, T.n1, T.n,
                    T.bg, T.stats)
