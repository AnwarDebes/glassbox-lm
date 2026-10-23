"""B1 team of one-clause Tsetlin members (numba inner loops).

Member i holds one clause over 2n = 512 literals (j < 256: x_j, j >= 256: not x_(j-256)).
Literal inclusion is chosen by two-action Tsetlin automata with N = 64 states each
(states 0..63, include iff state >= 32).

Team prediction (counted log-odds sum):
    p_bg = (bg1 + alpha/2) / (bgn + alpha)          counted on samples where no member speaks
    p_i  = (n1_i + alpha p_bg) / (n_i + alpha)      counted on samples where member i fired
    w_i  = logit(p_i) - logit(p_bg)                 a member with no counts has w_i = 0
    z    = logit(p_bg) + sum of w_i over speaking members
Members are positive clauses: member i speaks when its clause is non-empty, fires, and w_i > 0.
Identical clauses speak once: of the eligible holders the one with the most counts speaks
(ties: lowest index), the others are silent duplicates. Counts are prequential (a sample is
predicted, then counted). When a clause changes, its counts restart at zero, so a new clause is
silent until its own counts give it positive evidence.

Rewards, in bits; L = log-loss of a team logit on the label; z_-i = team logit without member i
(for a speaking member: z - w_i, plus the evidence of the next holder of the same clause if any);
probe evidence v_i = max(w_i, W_PROBE) for a member that does not speak:
    shared   r_i = L(z_bg) - L(z)                                same for every member
    selfish  r_i = L(z_bg) - L(z_bg + w_i)    speaking member
             r_i = L(z_bg) - L(z_bg + v_i)    otherwise
    marginal r_i = L(z_-i) - L(z)             speaking member (leave-one-out, one subtraction)
             r_i = 0                          silent duplicate of a clause that speaks
             r_i = L(z) - L(z + v_i)          otherwise (bits saved had it spoken)
For marginal and selfish the sign of r_i is the label's (y = 1: +, y = 0: -), as for a positive
clause of a standard Tsetlin machine; the magnitude differs. The shared sign is the team's outcome.
Reward to feedback (identical in every arm), g = min(1, |r_i| / RHO):
    clause fires (empty clauses fire during training), r_i > 0: with prob g Type Ia
    clause fires, r_i < 0: with prob g Type II
    clause does not fire, r_i > 0: with prob g Type Ib
    clause does not fire, r_i <= 0: nothing
    Type Ia: true literal +1 with prob (s-1)/s, false literal -1 with prob 1/s
    Type Ib: every literal -1 with prob 1/s
    Type II: every excluded false literal +1

Change modes: 0 direct (a changed clause replaces the old one at once), 1 dual-state trial
(Wald SPRT over three hypotheses: old, new, both), 2 fixed-sample A/B test (same trial, decided
after a fixed number of informative samples).
"""
import numpy as np
from numba import njit

NLIT = 512
TH = 32          # include iff state >= TH
SMAX = 63
WORDS = 4

# parameter vector indices
P_S, P_RHO, P_ALPHA, P_KAPPA, P_SPRT_A, P_TMAX, P_TMAX_SAMPLES, P_NAB, P_EMA, P_WPROBE = range(10)
PARAM_NAMES = ["s", "rho", "alpha", "kappa", "sprt_bound_bits", "tmax_inf", "tmax_samples",
               "n_ab", "ema", "w_probe"]


def default_params():
    p = np.zeros(10)
    p[P_S] = 3.9
    p[P_RHO] = 0.25
    p[P_ALPHA] = 2.0
    p[P_KAPPA] = 0.0                     # counts kept on a clause change (0: restart)
    p[P_SPRT_A] = np.log2(0.95 / 0.05)   # Wald bound for alpha = beta = 0.05, in bits
    p[P_TMAX] = 200.0                    # truncation: informative samples
    p[P_TMAX_SAMPLES] = 100_000.0        # truncation: samples
    p[P_NAB] = 0.0                       # fixed A/B length (informative samples)
    p[P_EMA] = 1e-3
    p[P_WPROBE] = 1.0                    # nats
    return p


# ---------------------------------------------------------------- small helpers

@njit(cache=True, inline="always")
def rnd(st):
    x = st[0]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    st[0] = x
    return (x * np.uint64(2685821657736338717)) >> np.uint64(11)   # 53 random bits


@njit(cache=True, inline="always")
def softplus(a):
    if a > 30.0:
        return a
    if a < -30.0:
        return np.exp(a)
    return np.log1p(np.exp(a))


@njit(cache=True, inline="always")
def loss_bits(z, y):
    # -log2 sigma(z) for y = 1, -log2 (1 - sigma(z)) for y = 0
    return softplus(-z if y == 1 else z) / np.log(2.0)


@njit(cache=True, inline="always")
def logit(p):
    return np.log(p) - np.log1p(-p)


@njit(cache=True, inline="always")
def fires(x, pos, neg, i):
    for w in range(WORDS):
        if (x[w] & pos[i, w]) != pos[i, w]:
            return False
        if ((~x[w]) & neg[i, w]) != neg[i, w]:
            return False
    return True


@njit(cache=True, inline="always")
def fires_mask(x, pw, nw):
    for w in range(WORDS):
        if (x[w] & pw[w]) != pw[w]:
            return False
        if ((~x[w]) & nw[w]) != nw[w]:
            return False
    return True


@njit(cache=True)
def mix64(h):
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xff51afd7ed558ccd)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xc4ceb9fe1a85ec53)
    h ^= h >> np.uint64(33)
    return h


@njit(cache=True)
def clause_hash(pw, nw):
    h = np.uint64(0x9e3779b97f4a7c15)
    empty = True
    for w in range(WORDS):
        if pw[w] != 0 or nw[w] != 0:
            empty = False
        h = mix64(h ^ pw[w]) + np.uint64(w)
        h = mix64(h ^ nw[w]) + np.uint64(w + 7)
    if empty:
        return np.uint64(0)
    if h == 0:
        h = np.uint64(1)
    return h


@njit(cache=True)
def masks_from_ta(ta, i, pw, nw):
    for w in range(WORDS):
        pw[w] = 0
        nw[w] = 0
    for j in range(NLIT):
        if ta[i, j] >= TH:
            if j < 256:
                pw[j >> 6] |= np.uint64(1) << np.uint64(j & 63)
            else:
                jj = j - 256
                nw[jj >> 6] |= np.uint64(1) << np.uint64(jj & 63)


@njit(cache=True, inline="always")
def lit(x, j):
    if j < 256:
        return (x[j >> 6] >> np.uint64(j & 63)) & np.uint64(1)
    jj = j - 256
    return np.uint64(1) - ((x[jj >> 6] >> np.uint64(jj & 63)) & np.uint64(1))


# ---------------------------------------------------------------- team state

@njit(cache=True)
def recompute_dup(chash, has_dup):
    M = len(chash)
    for i in range(M):
        has_dup[i] = False
    for i in range(M):
        if chash[i] == 0:
            continue
        for j in range(i + 1, M):
            if chash[j] == chash[i]:
                has_dup[i] = True
                has_dup[j] = True


@njit(cache=True)
def update_dup(chash, has_dup, i, old_h):
    """Member i changed hash from old_h to chash[i]; fix has_dup for both groups."""
    M = len(chash)
    new_h = chash[i]
    c_old = 0
    last = -1
    c_new = 0
    for j in range(M):
        if old_h != 0 and chash[j] == old_h:
            c_old += 1
            last = j
        if new_h != 0 and chash[j] == new_h:
            c_new += 1
    if c_old == 1:
        has_dup[last] = False
    for j in range(M):
        if new_h != 0 and chash[j] == new_h:
            has_dup[j] = c_new > 1
    if new_h == 0:
        has_dup[i] = False


@njit(cache=True)
def team_eval(x, pos, neg, chash, has_dup, n1, n, bg, alpha, fire, active, alt, dsil, w):
    """Fill fire, active (speaks), alt (next holder of the same clause when an active member
    leaves, or -1), dsil (silent duplicate of a clause that speaks) and w (evidence of every
    member). Returns (z, z_bg, any_active)."""
    M = pos.shape[0]
    p_bg = (bg[0] + 0.5 * alpha) / (bg[1] + alpha)
    z_bg = logit(p_bg)
    z = z_bg
    ndup = 0
    any_active = False
    for i in range(M):
        f = fires(x, pos, neg, i)
        fire[i] = f
        active[i] = False
        dsil[i] = False
        alt[i] = -1
        pi = (n1[i] + alpha * p_bg) / (n[i] + alpha)
        w[i] = logit(pi) - z_bg
        if f and chash[i] != 0 and w[i] > 0.0:
            if has_dup[i]:
                ndup += 1
            else:
                active[i] = True
    if ndup > 0:
        # identical eligible clauses: the holder with the most counts speaks (ties: lowest index)
        for i in range(M):
            if not (fire[i] and chash[i] != 0 and has_dup[i] and w[i] > 0.0):
                continue
            lowest = True
            for j in range(i):
                if fire[j] and chash[j] == chash[i] and w[j] > 0.0:
                    lowest = False
                    break
            if not lowest:
                continue        # group handled from its lowest index
            best = i
            second = -1
            for j in range(i + 1, M):
                if fire[j] and chash[j] == chash[i] and w[j] > 0.0:
                    if n[j] > n[best]:
                        second = best
                        best = j
                    elif second == -1 or n[j] > n[second]:
                        second = j
            active[best] = True
            alt[best] = second
            for j in range(i, M):
                if j != best and fire[j] and chash[j] == chash[i] and w[j] > 0.0:
                    dsil[j] = True
    for i in range(M):
        if active[i]:
            z += w[i]
            any_active = True
    return z, z_bg, any_active


@njit(cache=True)
def team_predict(X, pos, neg, chash, has_dup, n1, n, bg, alpha):
    """Team probability of y = 1 on every row of X (no learning)."""
    M = pos.shape[0]
    fire = np.zeros(M, dtype=np.bool_)
    active = np.zeros(M, dtype=np.bool_)
    alt = np.zeros(M, dtype=np.int64)
    dsil = np.zeros(M, dtype=np.bool_)
    w = np.zeros(M)
    q = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        z, zb, _ = team_eval(X[t], pos, neg, chash, has_dup, n1, n, bg, alpha, fire, active, alt,
                             dsil, w)
        q[t] = 1.0 / (1.0 + np.exp(-z))
    return q


@njit(cache=True)
def rewards(mode, y, z, z_bg, active, alt, dsil, w, wprobe, r):
    """mode 0 shared, 1 selfish, 2 marginal. Fills r (bits); returns L(z)."""
    M = len(w)
    Lz = loss_bits(z, y)
    Lbg = loss_bits(z_bg, y)
    for i in range(M):
        if mode == 0:
            r[i] = Lbg - Lz
            continue
        v = w[i] if active[i] else max(w[i], wprobe)
        if mode == 1:
            r[i] = Lbg - loss_bits(z_bg + v, y)
        elif active[i]:
            zi = z - w[i]
            if alt[i] >= 0:
                zi += w[alt[i]]
            r[i] = loss_bits(zi, y) - Lz
        elif dsil[i]:
            r[i] = 0.0
        else:
            r[i] = Lz - loss_bits(z + v, y)
    return Lz


@njit(cache=True)
def feedback(ta, i, x, kind, s, st):
    """kind 1 Ia, 2 Ib, 3 II. Returns True when some automaton crossed the include boundary."""
    changed = False
    p_inc = np.uint64(((s - 1.0) / s) * 9007199254740992.0)
    p_dec = np.uint64((1.0 / s) * 9007199254740992.0)
    if kind == 1:
        for j in range(NLIT):
            a = ta[i, j]
            if lit(x, j) == 1:
                if a < SMAX and rnd(st) < p_inc:
                    ta[i, j] = a + 1
                    if a + 1 == TH:
                        changed = True
            else:
                if a > 0 and rnd(st) < p_dec:
                    ta[i, j] = a - 1
                    if a == TH:
                        changed = True
    elif kind == 2:
        for j in range(NLIT):
            a = ta[i, j]
            if a > 0 and rnd(st) < p_dec:
                ta[i, j] = a - 1
                if a == TH:
                    changed = True
    else:
        for j in range(NLIT):
            a = ta[i, j]
            if a < TH and lit(x, j) == 0:
                ta[i, j] = a + 1
                if a + 1 == TH:
                    changed = True
    return changed


@njit(cache=True)
def shrink(n1, n, i, kappa):
    if n[i] > kappa:
        n1[i] *= kappa / n[i]
        n[i] = kappa


# trial state columns (float64 array tr[M, 8])
T_ON, T_LOLD, T_LNEW, T_LBOTH, T_NINF, T_NSAMP, T_N1, T_N = range(8)


@njit(cache=True)
def train_chunk(X, Y, rmode, cmode, prm, ta, pos, neg, chash, has_dup, n1, n, bg,
                spos, sneg, tr, ema, st, stats):
    """Learn on the samples X, Y in order. stats (int64[12]):
    0 samples, 1 evaluations, 2 clause changes proposed, 3 accepted new, 4 kept old,
    5 kept both, 6 truncated, 7 informative samples in finished trials, 8 samples in finished
    trials, 9 finished trials, 10 feedback events, 11 unused."""
    M = pos.shape[0]
    s = prm[P_S]
    rho = prm[P_RHO]
    alpha = prm[P_ALPHA]
    kappa = prm[P_KAPPA]
    A = prm[P_SPRT_A]
    tmax = prm[P_TMAX]
    tmax_s = prm[P_TMAX_SAMPLES]
    nab = prm[P_NAB]
    ema_r = prm[P_EMA]
    fire = np.zeros(M, dtype=np.bool_)
    active = np.zeros(M, dtype=np.bool_)
    alt = np.zeros(M, dtype=np.int64)
    dsil = np.zeros(M, dtype=np.bool_)
    w = np.zeros(M)
    r = np.zeros(M)
    wprobe = prm[P_WPROBE]
    pw = np.zeros(WORDS, dtype=np.uint64)
    nw = np.zeros(WORDS, dtype=np.uint64)
    for t in range(X.shape[0]):
        x = X[t]
        y = Y[t]
        z, z_bg, any_active = team_eval(x, pos, neg, chash, has_dup, n1, n, bg, alpha,
                                        fire, active, alt, dsil, w)
        stats[0] += 1
        stats[1] += M
        Lz = rewards(rmode, y, z, z_bg, active, alt, dsil, w, wprobe, r)
        for i in range(M):
            ema[i] += ema_r * (r[i] - ema[i])
        # trials: score old, new and both on this sample before counting it
        if cmode != 0:
            p_bg = (bg[0] + 0.5 * alpha) / (bg[1] + alpha)
            for i in range(M):
                if tr[i, T_ON] == 0.0:
                    continue
                stats[1] += 1
                f_new = fires_mask(x, spos[i], sneg[i])
                tr[i, T_NSAMP] += 1.0
                if fire[i] or f_new:
                    contrib = 0.0
                    if active[i]:
                        contrib = w[i]
                        if alt[i] >= 0:
                            contrib -= w[alt[i]]
                    w_new = 0.0
                    s_empty = True
                    for k in range(WORDS):
                        if spos[i, k] != 0 or sneg[i, k] != 0:
                            s_empty = False
                    if f_new and not s_empty:
                        w_new = max(0.0, logit((tr[i, T_N1] + alpha * p_bg)
                                               / (tr[i, T_N] + alpha)) - z_bg)
                    tr[i, T_LOLD] += Lz
                    tr[i, T_LNEW] += loss_bits(z - contrib + w_new, y)
                    tr[i, T_LBOTH] += loss_bits(z + w_new, y)
                    tr[i, T_NINF] += 1.0
                if f_new:
                    tr[i, T_N1] += y
                    tr[i, T_N] += 1.0
        # counting (prequential: after the prediction)
        for i in range(M):
            if fire[i]:
                n1[i] += y
                n[i] += 1.0
        if not any_active:
            bg[0] += y
            bg[1] += 1.0
        # trial decisions
        if cmode != 0:
            for i in range(M):
                if tr[i, T_ON] == 0.0:
                    continue
                lo, ln, lb = tr[i, T_LOLD], tr[i, T_LNEW], tr[i, T_LBOTH]
                verdict = -1     # 0 old, 1 new, 2 both
                if cmode == 1:
                    if lo - ln >= A and lb - ln >= A:
                        verdict = 1
                    elif ln - lo >= A and lb - lo >= A:
                        verdict = 0
                    elif lo - lb >= A and ln - lb >= A:
                        verdict = 2
                    elif tr[i, T_NINF] >= tmax or tr[i, T_NSAMP] >= tmax_s:
                        verdict = 1
                        stats[6] += 1
                else:
                    if tr[i, T_NINF] >= nab or tr[i, T_NSAMP] >= tmax_s:
                        if tr[i, T_NSAMP] >= tmax_s and tr[i, T_NINF] < nab:
                            stats[6] += 1
                        verdict = 1
                        if lo < ln and lo <= lb:
                            verdict = 0
                        elif lb < ln and lb < lo:
                            verdict = 2
                if verdict < 0:
                    continue
                stats[7] += np.int64(tr[i, T_NINF])
                stats[8] += np.int64(tr[i, T_NSAMP])
                stats[9] += 1
                tr[i, T_ON] = 0.0
                if verdict == 1:
                    stats[3] += 1
                    old_h = chash[i]
                    for k in range(WORDS):
                        pos[i, k] = spos[i, k]
                        neg[i, k] = sneg[i, k]
                    chash[i] = clause_hash(pos[i], neg[i])
                    update_dup(chash, has_dup, i, old_h)
                    n1[i] = tr[i, T_N1]
                    n[i] = tr[i, T_N]
                else:
                    if verdict == 2:
                        stats[5] += 1
                        # the new clause takes the least useful other member's slot:
                        # an empty member if there is one, else the lowest running reward
                        jbest = -1
                        for j in range(M):
                            if j == i or tr[j, T_ON] != 0.0:
                                continue
                            if jbest < 0:
                                jbest = j
                            elif chash[j] == 0 and chash[jbest] != 0:
                                jbest = j
                            elif (chash[j] == 0) == (chash[jbest] == 0) and ema[j] < ema[jbest]:
                                jbest = j
                        if jbest >= 0:
                            for jj in range(NLIT):
                                ta[jbest, jj] = ta[i, jj]
                            old_h = chash[jbest]
                            for k in range(WORDS):
                                pos[jbest, k] = spos[i, k]
                                neg[jbest, k] = sneg[i, k]
                            chash[jbest] = clause_hash(pos[jbest], neg[jbest])
                            update_dup(chash, has_dup, jbest, old_h)
                            n1[jbest] = tr[i, T_N1]
                            n[jbest] = tr[i, T_N]
                            ema[jbest] = ema[i]
                    else:
                        stats[4] += 1
                    # member i returns to its old clause: automata put back at the boundary
                    for j in range(NLIT):
                        inc_old = False
                        if j < 256:
                            inc_old = ((pos[i, j >> 6] >> np.uint64(j & 63)) & np.uint64(1)) == 1
                        else:
                            jj = j - 256
                            inc_old = ((neg[i, jj >> 6] >> np.uint64(jj & 63)) & np.uint64(1)) == 1
                        if inc_old and ta[i, j] < TH:
                            ta[i, j] = TH
                        elif (not inc_old) and ta[i, j] >= TH:
                            ta[i, j] = TH - 1
        # automaton feedback (members in a trial are frozen)
        for i in range(M):
            if cmode != 0 and tr[i, T_ON] != 0.0:
                continue
            ri = r[i]
            if ri == 0.0:
                continue
            g = abs(ri) / rho
            if g < 1.0 and rnd(st) >= np.uint64(g * 9007199254740992.0):
                continue
            if fire[i]:
                kind = 1 if ri > 0 else 3
            elif ri > 0:
                kind = 2
            else:
                continue
            stats[10] += 1
            if not feedback(ta, i, x, kind, s, st):
                continue
            masks_from_ta(ta, i, pw, nw)
            same = True
            for k in range(WORDS):
                if pw[k] != pos[i, k] or nw[k] != neg[i, k]:
                    same = False
            if same:
                continue
            stats[2] += 1
            if cmode == 0:
                old_h = chash[i]
                for k in range(WORDS):
                    pos[i, k] = pw[k]
                    neg[i, k] = nw[k]
                chash[i] = clause_hash(pw, nw)
                update_dup(chash, has_dup, i, old_h)
                shrink(n1, n, i, kappa)
            else:
                for k in range(WORDS):
                    spos[i, k] = pw[k]
                    sneg[i, k] = nw[k]
                tr[i, T_ON] = 1.0
                tr[i, T_LOLD] = 0.0
                tr[i, T_LNEW] = 0.0
                tr[i, T_LBOTH] = 0.0
                tr[i, T_NINF] = 0.0
                tr[i, T_NSAMP] = 0.0
                k1 = n1[i]
                kn = n[i]
                if kn > kappa:
                    k1 *= kappa / kn
                    kn = kappa
                tr[i, T_N1] = k1
                tr[i, T_N] = kn


class Team:
    """State of a team of M one-clause members, with python-side helpers."""

    def __init__(self, M, seed, prm=None, init_lits=1, init_depth=0):
        self.M = M
        self.prm = default_params() if prm is None else prm
        rng = np.random.default_rng([seed, M, 7])
        self.ta = np.full((M, NLIT), init_depth, dtype=np.uint8)
        for i in range(M):
            # init_lits random literals on distinct variables, at the weakest include state
            vs = rng.choice(256, size=init_lits, replace=False)
            for v in vs:
                self.ta[i, v + 256 * int(rng.integers(0, 2))] = TH
        self.pos = np.zeros((M, WORDS), dtype=np.uint64)
        self.neg = np.zeros((M, WORDS), dtype=np.uint64)
        self.chash = np.zeros(M, dtype=np.uint64)
        for i in range(M):
            masks_from_ta(self.ta, i, self.pos[i], self.neg[i])
            self.chash[i] = clause_hash(self.pos[i], self.neg[i])
        self.has_dup = np.zeros(M, dtype=np.bool_)
        recompute_dup(self.chash, self.has_dup)
        self.n1 = np.zeros(M)
        self.n = np.zeros(M)
        self.bg = np.zeros(2)
        self.spos = np.zeros((M, WORDS), dtype=np.uint64)
        self.sneg = np.zeros((M, WORDS), dtype=np.uint64)
        self.tr = np.zeros((M, 8))
        self.ema = np.zeros(M)
        self.st = np.array([int(rng.integers(1, 2**63))], dtype=np.uint64)
        self.stats = np.zeros(12, dtype=np.int64)

    def train(self, X, Y, rmode, cmode):
        train_chunk(X, Y, rmode, cmode, self.prm, self.ta, self.pos, self.neg, self.chash,
                    self.has_dup, self.n1, self.n, self.bg, self.spos, self.sneg, self.tr,
                    self.ema, self.st, self.stats)

    def predict(self, X):
        return team_predict(X, self.pos, self.neg, self.chash, self.has_dup, self.n1, self.n,
                            self.bg, self.prm[P_ALPHA])

    def set_clauses(self, pos, neg, n1=None, n=None):
        """Put given clauses into the first rows (used by the oracle team and the checks)."""
        k = len(pos)
        self.pos[:k] = pos
        self.neg[:k] = neg
        for i in range(k):
            self.chash[i] = clause_hash(self.pos[i], self.neg[i])
            if n1 is not None:
                self.n1[i], self.n[i] = n1[i], n[i]
        recompute_dup(self.chash, self.has_dup)

    def clause_sizes(self):
        return np.array([sum(bin(int(a)).count("1") for a in np.concatenate([p, q]))
                         for p, q in zip(self.pos, self.neg)])
