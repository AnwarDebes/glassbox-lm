"""Floor F of C-core (PLAN_part3.md section 4, "Floor F"): KN5 (tally floor/kn.py, sentence mode)
plus a decayed unigram cache (tally floor/cache.py: half-life 100, reset per 100-sentence block,
<eos> not added; the setting A1 chose on valid), mixed with bucketed MAP-EM weights fitted on calib
(tally floor/mix.py: longest KN5 match x count bucket, 40 buckets, tau 20, prior = the global EM
weights on calib). Where the cache is empty, the cache member is the KN5 member (tally Cache rule).

F(w | h) over the full vocabulary, its class marginals F(c | h) = sum of F(w | h) over the words of c,
and the true-word probability (the fast path) are all computed here."""
import numpy as np

from . import paths

paths.setup()
from tally.floor import mix  # noqa: E402
from tally.floor.cache import Cache  # noqa: E402
from tally.floor.kn import KN  # noqa: E402

ORDER = 5
HALF_LIFE = 100.0
RESET = "block"
TAU = 20.0
N_BUCKETS = mix.n_buckets(ORDER)          # 40
CHUNK = 1024
# A1's own valid numbers (tally results/20261015T072513Z_a1_baseline_check/metrics.json, valid_table),
# made with an earlier kn.py (sha256 e40909db...) that matched KenLM within 3.8e-5 in relative perplexity;
# the current kn.py must reproduce them exactly, so that match carries over. Known before R1.
A1_VALID_BPW = {"KN5": 7.255057512770932, "KN3": 7.34256972151113, "KN5 + cache (global weights)": 7.054522637538526}
A1_TOL = 1e-9


class Floor:
    def __init__(self, ids, blocks, V, eos, fit_split="calib"):
        self.V, self.eos = V, eos
        self.kn = KN(ids["train"], V, eos, order=ORDER, mode="sent")
        self.cache = Cache(V, HALF_LIFE, RESET, skip=(eos,))
        P = self.members(ids[fit_split], blocks[fit_split])
        self.lam_global, self.trace_global = mix.em(P)
        b, _, _ = self.buckets(ids[fit_split])
        self.lam, self.trace = mix.em_bucketed(P, b, N_BUCKETS, self.lam_global, tau=TAU)
        assert np.allclose(self.lam.sum(1), 1.0) and (self.lam > 0).all()

    def members(self, x, bl):
        """(n, 2): true-word probability under KN5 and under the cache (KN5 where the cache is empty)."""
        p_kn = np.exp(self.kn.logprob(x))
        p_c, _ = self.cache.prob(x, bl, p_kn)
        return np.stack([p_kn, p_c], 1)

    def buckets(self, x):
        L, cnt = self.kn.match(x)
        return mix.buckets(L, cnt), L, cnt

    def prob(self, x, bl):
        """True-word probability of F at every position (fast path) and the bucket ids."""
        P = self.members(x, bl)
        b, L, cnt = self.buckets(x)
        return (P * self.lam[b]).sum(1), b, L

    def dists(self, x, bl, pos, b=None):
        """Full vectors F(. | h) over V at positions pos, shape (len(pos), V)."""
        pos = np.asarray(pos, dtype=np.int64)
        if b is None:
            b = self.buckets(x)[0]
        Dk = self.kn.dists(x, pos)
        Dc = self.cache.dists(x, bl, pos, Dk)
        lb = self.lam[b[pos]]
        return lb[:, :1] * Dk + lb[:, 1:] * Dc

    def class_marginals(self, x, bl, cls, K, b=None, chunk=CHUNK):
        """F(c | h) at every position (n, K), with the sum and minimum of every full vector checked."""
        n = len(x)
        if b is None:
            b = self.buckets(x)[0]
        M = np.zeros((self.V, K))
        M[np.arange(self.V), cls] = 1.0
        Fc = np.empty((n, K))
        max_sum_err, min_p = 0.0, np.inf
        for a in range(0, n, chunk):
            pos = np.arange(a, min(a + chunk, n))
            D = self.dists(x, bl, pos, b)
            max_sum_err = max(max_sum_err, float(np.abs(D.sum(1) - 1.0).max()))
            min_p = min(min_p, float(D.min()))
            Fc[pos] = D @ M
        return Fc, {"positions": int(n), "max_sum_err": max_sum_err, "min_p": min_p,
                    "pass": bool(max_sum_err <= 1e-6 and min_p > 0)}

    def summary(self):
        return {"kn5": self.kn.summary(), "cache": self.cache.config(), "lam_global": self.lam_global.tolist(),
                "em_steps_global": int(len(self.trace_global)), "em_steps_bucketed": int(len(self.trace)),
                "buckets": N_BUCKETS, "tau": TAU,
                "em_monotone": bool(np.all(np.diff(self.trace) >= -1e-6))}


def a1_check(fl, ids, bl, V, eos, compare=True):
    """KN5, KN3 and KN5 + cache (global calib weights) on valid must equal A1's own numbers within 1e-9.
    compare=False (dry runs on pseudo splits) runs the same code and reports without comparing."""
    LN2 = np.log(2.0)
    Pm = fl.members(ids["valid"], bl["valid"])
    got = {"KN5": float(-np.log(Pm[:, 0]).mean() / LN2),
           "KN3": float(-KN(ids["train"], V, eos, order=3, mode="sent").logprob(ids["valid"]).mean() / LN2),
           "KN5 + cache (global weights)": float(-np.log(Pm @ fl.lam_global).mean() / LN2)}
    out = {k: {"A1": A1_VALID_BPW[k], "now": got[k], "diff": got[k] - A1_VALID_BPW[k]} for k in got}
    if not compare:
        return {k: {"now": v["now"], "compared": False} for k, v in out.items()}
    if any(abs(v["diff"]) > A1_TOL for v in out.values()):
        raise SystemExit(f"the floor code does not reproduce A1's valid numbers: {out}")
    return out
