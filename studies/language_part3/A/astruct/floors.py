"""The conditional counting floors (PLAN_part3.md section 3, floor row; 10 configurations) and their summaries.

Notation (per position): S = the act-slot set, A = the domain-act set, R = the due placeholders (multiset),
w-1, w-2 = the previous tokens (BOS at the reply start, PAD before it). KN-n = tally floor/kn.py, sentence mode,
one reply = one sentence ending with the end token, counted on the counting replies.
- F1 n {4, 6}: KN chain (S, w-n+1..w-1) -> ... -> (S) -> the unconditional KN-n given w-n+1..w-1.
- F2 n {3, 5}: per-condition KN, the KN chain (A, w-n+1..w-1) -> ... -> (A) -> uniform; mixed with the
  unconditional KN-n by one EM weight per frequency bucket of A (its count of counting replies), fitted on tune-cal.
- F3 a/b: absolute discounting, one level (R, w-1) [a] or (R, A, w-1) [b], backing off to F1 n 6.
- F4 eps {1e-2, 1e-4}: the tune-cal-better F3, then placeholders with no due copy and the end token while a
  placeholder is due are multiplied by eps; renormalised.
- F5: absolute discounting chain (R, S, w-2, w-1) -> (R, A, w-1) -> (R, w-1) -> the tune-cal-best F2.
- F6: the tune-cal-best F2 and F4 mixed by one EM weight per frequency bucket of S, fitted on tune-cal.
Frequency buckets: counts {0}, {1}, {2-4}, {5-19}, {20-99}, {100-999}, {>= 1000}. EM: tally mix.em_bucketed with
tau 20 pseudo-tokens toward the unbucketed EM weights (tally mix.em).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .chain import Packed, apply_chain, fit_chain, lookup_all
from .data import Corpus
from .paths import add_study_paths
from .positions import Positions

add_study_paths(tally=True)
from tally.floor import kn as knmod
from tally.floor import mix as tmix

CONFIGS = ("F1n4", "F1n6", "F2n3", "F2n5", "F3a", "F3b", "F4e2", "F4e4", "F5", "F6")
EPS = {"F4e2": 1e-2, "F4e4": 1e-4}
FREQ_EDGES = (1, 2, 5, 20, 100, 1000)
N_FREQ = len(FREQ_EDGES) + 1
TAU = 20.0
K_TOP = 8


def freq_bucket(cnt: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(FREQ_EDGES), np.asarray(cnt), side="right").astype(np.int64)


class KNVec:
    """Full KN-n vectors over V for positions of one stream (tally's own kernel, one preparation per stream)."""

    def __init__(self, model, stream: np.ndarray):
        self.m = model
        self.s, self.target, self.ids = model._prep(np.asarray(stream, np.int64))

    def dists(self, pos: np.ndarray) -> np.ndarray:
        m = self.m
        return knmod._dists(self.s, self.target[np.asarray(pos, np.int64)], self.ids, m.N, m.B, m.P1, m.keys, m.koff,
                            m.num, m.cstart, m.coff, m.gam)


def _kn_levels(P: Positions, cond: np.ndarray | None, n: int):
    """KN chain levels for order n with an optional condition column: k = n-1 .. 0 history words."""
    Ts, ex, raw = [], [], []
    for k in range(n - 1, -1, -1):
        cols = ([cond[:, None]] if cond is not None else []) + [P.H[:, :k]]
        T = np.concatenate(cols, 1) if cols else np.zeros((P.n, 0), np.int64)
        if T.shape[1] == 0:
            T = np.zeros((P.n, 1), np.int64)  # the empty context
        Ts.append(T)
        ex.append(P.t + 1 >= k)
        raw.append((P.t + 1 == k) & (k >= 1))
    return Ts, ex, raw


def _abs_levels(P: Positions, name: str):
    one = np.ones(P.n, bool)
    if name == "F3a":
        Ts = [np.stack([P.R, P.H[:, 0]], 1)]
    elif name == "F3b":
        Ts = [np.stack([P.R, P.A, P.H[:, 0]], 1)]
    elif name == "F5":
        Ts = [np.stack([P.R, P.S, P.H[:, 1], P.H[:, 0]], 1), np.stack([P.R, P.A, P.H[:, 0]], 1), np.stack([P.R, P.H[:, 0]], 1)]
    else:
        raise ValueError(name)
    return Ts, [one] * len(Ts), None


def chain_inputs(name: str, P: Positions):
    if name.startswith("F1n"):
        return _kn_levels(P, P.S, int(name[3:]))
    if name.startswith("F2c"):
        return _kn_levels(P, P.A, int(name[3:]))
    if name.startswith("KNc"):  # plain KN chain with no condition (self-test against tally KN)
        return _kn_levels(P, None, int(name[3:]))
    return _abs_levels(P, name)


def chain_mode(name: str) -> str:
    return "kn" if name[:3] in ("F1n", "F2c", "KNc") else "abs"


class View:
    """The scored positions of one set of replies, with per-chain context lookups computed once."""

    def __init__(self, P: Positions):
        self.P = P
        self.kn = {}
        self.cidx = {}


class FloorSet:
    """Every floor component counted on one set of counting replies; picks and EM weights are set later."""

    def __init__(self, corpus: Corpus, count_replies: np.ndarray, picks: dict | None = None,
                 need: tuple = CONFIGS, log=None):
        self.c = corpus
        self.V = corpus.V
        self.count_replies = np.asarray(count_replies, np.int64)
        self.picks = dict(picks or {})
        self.lam = {}
        self.em_info = {}
        P = Positions(corpus, self.count_replies)
        self.n_count_positions = P.n
        comps = self._components(need)
        self.kn = {}
        for n in sorted({int(c[2:]) for c in comps if c.startswith("KN") and c[2:].isdigit()}):
            self.kn[n] = knmod.KN(P.stream, self.V, corpus.END, order=n, mode="sent", discount_fallback=knmod.FALLBACK)
            if log:
                log(f"KN-{n} counted:", self.kn[n].summary())
        self.chains, self.packed = {}, {}
        for c in comps:
            if c[:3] in ("F1n", "F2c", "KNc") or c in ("F3a", "F3b", "F5"):
                Ts, ex, raw = chain_inputs(c, P)
                self.chains[c] = fit_chain(Ts, ex, raw, P.x, self.V, chain_mode(c))
                self.packed[c] = Packed(self.chains[c])
                if log:
                    log(f"{c} levels:", [(lv.n_ctx, len(lv.w), [round(float(x), 4) for x in lv.D[1:]], lv.fallback)
                                         for lv in self.chains[c]])
        # condition frequencies in the counting replies (for the EM buckets)
        self.A_cnt = _counter(corpus.a["A_hash"][self.count_replies])
        self.S_cnt = _counter(corpus.a["S_hash"][self.count_replies])

    def _components(self, need) -> set:
        out = set()

        def add(name):
            if name in out:
                return
            out.add(name)
            for d in self._deps(name):
                add(d)
        for n in need:
            add(n)
        return out

    def _deps(self, name: str) -> list:
        if name.startswith("F1n"):
            return ["KN" + name[3:]]
        if name.startswith("F2n"):
            return ["F2c" + name[3:], "KN" + name[3:]]
        if name in ("F3a", "F3b"):
            return ["F1n6"]
        if name in EPS:
            return [self.picks.get("F3", "F3a"), self.picks.get("F3", "F3b")]
        if name == "F5":
            return [self.picks.get("F2", "F2n3"), self.picks.get("F2", "F2n5")]
        if name == "F6":
            return [self.picks.get("F2", "F2n3"), self.picks.get("F2", "F2n5"),
                    self.picks.get("F4", "F4e2"), self.picks.get("F4", "F4e4")]
        return []

    # ---- vectors -----------------------------------------------------------------------------------------
    def view(self, P: Positions) -> View:
        return View(P)

    def _cidx(self, view: View, name: str) -> np.ndarray:
        if name not in view.cidx:
            Ts, ex, _ = chain_inputs(name, view.P)
            view.cidx[name] = lookup_all(self.chains[name], Ts, ex)
        return view.cidx[name]

    def _bucket(self, view: View, name: str, idx: np.ndarray) -> np.ndarray:
        if name.startswith("F2n"):
            return freq_bucket(_lookup_count(self.A_cnt, view.P.A[idx]))
        return freq_bucket(_lookup_count(self.S_cnt, view.P.S[idx]))

    def vectors(self, view: View, idx: np.ndarray, names, memo: dict | None = None) -> dict:
        memo = {} if memo is None else memo
        for n in names:
            self._vec(view, idx, n, memo)
        return memo

    def _vec(self, view: View, idx: np.ndarray, name: str, memo: dict) -> np.ndarray:
        if name in memo:
            return memo[name]
        V = self.V
        if name.startswith("KN") and name[2:].isdigit():
            n = int(name[2:])
            if n not in view.kn:
                view.kn[n] = KNVec(self.kn[n], view.P.stream)
            out = view.kn[n].dists(idx)
        elif name.startswith("F1n"):
            out = apply_chain(self._vec(view, idx, "KN" + name[3:], memo).copy(), self._cidx(view, name)[:, idx],
                              self.packed[name])
        elif name.startswith(("F2c", "KNc")):
            out = apply_chain(np.full((len(idx), V), 1.0 / V), self._cidx(view, name)[:, idx], self.packed[name])
        elif name.startswith("F2n") or name == "F6":
            a_name, b_name = (("F2c" + name[3:], "KN" + name[3:]) if name.startswith("F2n")
                              else (self.picks["F2"], self.picks["F4"]))
            if name not in self.lam:
                raise RuntimeError(f"{name}: EM weights not fitted (fit_em on tune-cal first)")
            lam = self.lam[name][self._bucket(view, name, idx)]
            out = lam[:, :1] * self._vec(view, idx, a_name, memo) + lam[:, 1:] * self._vec(view, idx, b_name, memo)
        elif name in ("F3a", "F3b"):
            out = apply_chain(self._vec(view, idx, "F1n6", memo).copy(), self._cidx(view, name)[:, idx], self.packed[name])
        elif name in EPS:
            out = eps_rule(self._vec(view, idx, self.picks["F3"], memo), view.P, idx, self.c, EPS[name])
        elif name == "F5":
            out = apply_chain(self._vec(view, idx, self.picks["F2"], memo).copy(), self._cidx(view, name)[:, idx],
                              self.packed[name])
        else:
            raise ValueError(name)
        memo[name] = out
        return out

    def fit_em(self, view: View, name: str, chunk: int = 2048) -> dict:
        """One EM weight pair per frequency bucket on the given (tune-cal) positions."""
        a_name, b_name = (("F2c" + name[3:], "KN" + name[3:]) if name.startswith("F2n")
                          else (self.picks["F2"], self.picks["F4"]))
        P = view.P
        pa, pb, bk = [], [], []
        for s in range(0, P.n, chunk):
            idx = np.arange(s, min(P.n, s + chunk))
            memo = self.vectors(view, idx, [a_name, b_name])
            r = np.arange(len(idx))
            pa.append(memo[a_name][r, P.x[idx]])
            pb.append(memo[b_name][r, P.x[idx]])
            bk.append(self._bucket(view, name, idx))
        M = np.stack([np.concatenate(pa), np.concatenate(pb)], 1)
        b = np.concatenate(bk)
        prior, _ = tmix.em(M)
        lam, tr = tmix.em_bucketed(M, b, N_FREQ, prior, tau=TAU)
        self.lam[name] = lam
        self.em_info[name] = {"prior": prior.tolist(), "lam": lam.tolist(), "bucket_n": np.bincount(b, minlength=N_FREQ).tolist(),
                              "em_steps": len(tr), "monotone": bool(np.all(np.diff(tr) >= -1e-9 * len(b)))}
        return self.em_info[name]


def _counter(h: np.ndarray) -> tuple:
    u, c = np.unique(np.asarray(h, np.int64), return_counts=True)
    return u, c


def _lookup_count(tab: tuple, q: np.ndarray) -> np.ndarray:
    u, c = tab
    if len(u) == 0:
        return np.zeros(len(q), np.int64)
    i = np.minimum(np.searchsorted(u, q), len(u) - 1)
    return np.where(u[i] == q, c[i], 0)


def eps_rule(P3: np.ndarray, P: Positions, idx: np.ndarray, c: Corpus, eps: float) -> np.ndarray:
    """F4: placeholders with no due copy, and the end token while a placeholder is due, times eps; renormalise."""
    out = P3.copy()
    f = np.ones(P3.shape[1])
    f[c.ph_ids] = eps
    out *= f[None, :]
    for r, k in enumerate(idx):
        d = P.due_types(int(k))
        if len(d):
            out[r, d] = P3[r, d]
        if P.due_n[k] > 0:
            out[r, c.END] *= eps
    return out / out.sum(1, keepdims=True)


@njit(cache=True)
def topk(P, k):
    """Top k per row by value, ties to the smaller id; returns (ids, values)."""
    n, V = P.shape
    idx = np.full((n, k), -1, np.int64)
    val = np.full((n, k), -1.0)
    for r in range(n):
        m = 0
        for v in range(V):
            p = P[r, v]
            if m == k and p <= val[r, k - 1]:
                continue
            j = m if m < k else k - 1
            while j > 0 and val[r, j - 1] < p:
                val[r, j] = val[r, j - 1]
                idx[r, j] = idx[r, j - 1]
                j -= 1
            val[r, j] = p
            idx[r, j] = v
            if m < k:
                m += 1
    return idx, val


def summarize(fs: FloorSet, view: View, name: str, chunk: int = 2048, keep_full: np.ndarray | None = None) -> dict:
    """Per position: the true token's floor probability, top-1, entropy, and the candidate set with its floor
    probabilities: the top 8 (ties: smaller id), then the due placeholders not among them (by probability, ties:
    smaller id), then the end token if not yet in."""
    P = view.P
    p_true = np.empty(P.n)
    top1 = np.empty(P.n)
    ent = np.empty(P.n)
    cands, cps, crank, coff = [], [], [], [0]
    full = {}
    END = fs.c.END
    for s in range(0, P.n, chunk):
        idx = np.arange(s, min(P.n, s + chunk))
        Pv = fs.vectors(view, idx, [name])[name]
        r = np.arange(len(idx))
        p_true[idx] = Pv[r, P.x[idx]]
        ids, vals = topk(Pv, K_TOP)
        top1[idx] = vals[:, 0]
        ent[idx] = -(Pv * np.log2(Pv)).sum(1)
        for i, k in enumerate(idx):
            top = list(ids[i])
            ex = [int(d) for d in P.due_types(int(k)) if d not in top]
            ex.sort(key=lambda d: (-Pv[i, d], d))
            if END not in top:
                ex.append(END)
            cl = top + ex
            cands.extend(cl)
            cps.extend(Pv[i, cl].tolist())
            crank.extend(list(range(K_TOP)) + [K_TOP] * len(ex))
            coff.append(coff[-1] + len(cl))
        if keep_full is not None:
            for i in np.flatnonzero(keep_full[idx]):
                full[int(idx[i])] = Pv[i].copy()
    return {"p_true": p_true, "top1": top1, "ent": ent, "cand_off": np.array(coff, np.int64),
            "cand": np.array(cands, np.int64), "cand_p": np.array(cps), "cand_rank": np.array(crank, np.int8), "full": full}


def bits_true(fs: FloorSet, view: View, names, chunk: int = 2048) -> dict:
    """-log2 P(true token) per position for several configurations (shared components computed once)."""
    P = view.P
    out = {n: np.empty(P.n) for n in names}
    for s in range(0, P.n, chunk):
        idx = np.arange(s, min(P.n, s + chunk))
        memo = fs.vectors(view, idx, names)
        r = np.arange(len(idx))
        for n in names:
            out[n][idx] = -np.log2(memo[n][r, P.x[idx]])
    return out
