"""Calibration of class sums into probabilities (docs/03 "Probabilities and perplexity"; docs/04
T2.5; docs/05 CP; docs/06 "Confidence"): two scalars fitted on validation NLL, split-conformal
prediction sets as the ConfTM stand-in, and ECE.

    logit_c = a * score_c + b * class_bias_c / 256          (nats)

with `a = a_q16 / 2^16`, `b = b_q16 / 2^16` fixed-point scalars and `class_bias` the int32
log-unigram of bank.py (round(256 ln p)). The logit numerator

    num_c = a_q16 * score_c * 256 + b_q16 * class_bias_c            (exact int64, logit = num / 2^24)

is integer, so the greedy (argmax) decision is exact integer arithmetic that the standalone
checker re-derives; only the softmax is floating point. Every row of `logprobs` is a normalised
distribution over the full vocabulary (logsumexp 0 within 1e-6, asserted), and no entry is -inf.

The fit minimises the mean validation NLL over (a >= 0, b), a convex function of (a, b)
(log-sum-exp of linear forms minus a linear form), by deterministic coordinate-wise
golden-section search; the returned scalars are the Q16-quantised optimum and the reported NLL
is evaluated at the quantised values. Nothing here uses the test split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Q16 = 1 << 16
LOGIT_DEN = float(1 << 24)


def logit_numerators(scores: np.ndarray, class_bias: np.ndarray, a_q16: int, b_q16: int) -> np.ndarray:
    """int64 [n, V]: a_q16 * score * 256 + b_q16 * class_bias. Exact."""
    s = np.asarray(scores, dtype=np.int64)
    b = np.asarray(class_bias, dtype=np.int64)
    return int(a_q16) * s * 256 + int(b_q16) * b[None, :]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


@dataclass(frozen=True)
class Calibration:
    a_q16: int
    b_q16: int

    @staticmethod
    def identity() -> "Calibration":
        """a = 1, b = 0: logits are the raw class sums (an uncalibrated placeholder)."""
        return Calibration(Q16, 0)

    @property
    def a(self) -> float:
        return self.a_q16 / Q16

    @property
    def b(self) -> float:
        return self.b_q16 / Q16

    def logits(self, scores: np.ndarray, class_bias: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        num = logit_numerators(scores, class_bias, self.a_q16, self.b_q16)
        return num.astype(np.float64) / (LOGIT_DEN * float(temperature))

    def logprobs(self, scores: np.ndarray, class_bias: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """float64 [n, V] natural-log probabilities, each row summing to one (asserted)."""
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = scores[None, :]
        lp = log_softmax(self.logits(scores, class_bias, temperature))
        lse = np.log(np.exp(lp).sum(axis=1))
        assert np.all(np.abs(lse) < 1e-6), "logprobs must normalise over the vocabulary"
        assert np.all(np.isfinite(lp)), "no -inf entries"
        return lp

    def argmax(self, scores: np.ndarray, class_bias: np.ndarray) -> np.ndarray:
        """Greedy decision on the exact integer numerators, ties toward the lower class id."""
        num = logit_numerators(np.atleast_2d(scores), class_bias, self.a_q16, self.b_q16)
        return num.argmax(axis=1)

    def to_dict(self) -> dict:
        return {"a_q16": int(self.a_q16), "b_q16": int(self.b_q16), "a": self.a, "b": self.b}


def mean_nll(scores: np.ndarray, gold: np.ndarray, class_bias: np.ndarray, cal: Calibration) -> float:
    lp = cal.logprobs(scores, class_bias)
    return float(-lp[np.arange(lp.shape[0]), np.asarray(gold, dtype=np.int64)].mean())


def _nll_float(scores: np.ndarray, gold: np.ndarray, bias: np.ndarray, a: float, b: float) -> float:
    logits = a * scores + b * bias[None, :]
    lp = log_softmax(logits)
    return float(-lp[np.arange(lp.shape[0]), gold].mean())


def _golden(f, lo: float, hi: float, iters: int = 60) -> float:
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = f(d)
    return (lo + hi) / 2.0


def fit_two_scalar(scores: np.ndarray, gold: np.ndarray, class_bias: np.ndarray,
                   a_range: tuple[float, float] = (0.0, 16.0), b_range: tuple[float, float] = (-16.0, 16.0),
                   rounds: int = 12) -> tuple[Calibration, dict]:
    """Fit (a, b) on VALIDATION rows (`scores` int32 [n, V] from the engine, `gold` [n] class ids).
    Returns the Q16-quantised calibration and a report (float optimum, quantised NLL, identity NLL).
    Deterministic: no random numbers, fixed iteration counts."""
    s = np.asarray(scores, dtype=np.float64)
    g = np.asarray(gold, dtype=np.int64)
    bias = np.asarray(class_bias, dtype=np.float64) / 256.0
    if s.ndim != 2 or g.shape != (s.shape[0],):
        raise ValueError("scores must be [n, V] and gold [n]")
    a, b = 1.0, 1.0
    for _ in range(rounds):
        a = _golden(lambda x: _nll_float(s, g, bias, x, b), a_range[0], a_range[1])
        b = _golden(lambda x: _nll_float(s, g, bias, a, x), b_range[0], b_range[1])
    cal = Calibration(int(round(a * Q16)), int(round(b * Q16)))
    report = {"a_float": a, "b_float": b, "nll_float_optimum": _nll_float(s, g, bias, a, b),
              "nll_quantised": mean_nll(scores, g, class_bias, cal),
              "nll_identity": mean_nll(scores, g, class_bias, Calibration.identity()),
              "nll_unigram_only": _nll_float(s, g, bias, 0.0, 1.0), "n_rows": int(s.shape[0])}
    return cal, report


def ece(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15) -> dict:
    """Expected calibration error of the top-1 prediction with equal-width confidence bins
    (docs/06: target below 5% after calibration). Returns ece, accuracy, mean confidence, bins."""
    p = np.asarray(probs, dtype=np.float64)
    g = np.asarray(gold, dtype=np.int64)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    hit = (pred == g).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    total = 0.0
    bins = []
    for k in range(n_bins):
        m = idx == k
        n = int(m.sum())
        if n == 0:
            bins.append({"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": 0})
            continue
        acc_k, conf_k = float(hit[m].mean()), float(conf[m].mean())
        total += n / p.shape[0] * abs(acc_k - conf_k)
        bins.append({"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": n, "accuracy": acc_k, "confidence": conf_k})
    return {"ece": float(total), "accuracy": float(hit.mean()), "mean_confidence": float(conf.mean()), "n_bins": n_bins, "bins": bins}


def conformal_qhat(probs: np.ndarray, gold: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal threshold on the score 1 - p(gold) from a CALIBRATION split (the ConfTM
    stand-in of docs/03): the ceil((n + 1)(1 - alpha)) / n empirical quantile."""
    p = np.asarray(probs, dtype=np.float64)
    g = np.asarray(gold, dtype=np.int64)
    n = p.shape[0]
    s = 1.0 - p[np.arange(n), g]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(s)[k - 1])


def conformal_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """bool [n, V]: the prediction set {c : 1 - p_c <= qhat}, never empty (the top-1 is always in)."""
    p = np.asarray(probs, dtype=np.float64)
    sets = (1.0 - p) <= qhat
    sets[np.arange(p.shape[0]), p.argmax(axis=1)] = True
    return sets


def conformal_report(probs: np.ndarray, gold: np.ndarray, qhat: float) -> dict:
    sets = conformal_sets(probs, qhat)
    g = np.asarray(gold, dtype=np.int64)
    return {"qhat": float(qhat), "coverage": float(sets[np.arange(sets.shape[0]), g].mean()),
            "mean_set_size": float(sets.sum(axis=1).mean()), "n": int(sets.shape[0])}
