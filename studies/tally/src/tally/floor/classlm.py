"""Class n-gram member: P(w | h) = P(c(w) | classes of h) * P(w | c(w)).

The class part is the interpolated modified Kneser-Ney model of floor/kn.py run on the class
stream (sentence mode; <eos> has a class of its own, so contexts never cross a sentence end).
Small class sets can leave an order without valid count-of-counts; that order then uses the
fallback discounts 0.5 / 1 / 1.5 (KenLM's --discount_fallback).

The word part is P(w | c) = (n(w) + delta) / (n(c) + delta * |c|), with n the train counts, so
that every word of the fixed vocabulary is above 0 and each class sums to 1. With both parts
summing to 1, the full vector over V sums to 1.
"""
import numpy as np

from .kn import FALLBACK, KN


class ClassMember:
    def __init__(self, train, cls, eos, order=5, delta=0.5):
        cls = np.asarray(cls, dtype=np.int64)
        self.cls, self.eos, self.order, self.delta = cls, eos, order, delta
        self.K = int(cls.max()) + 1
        assert (cls == cls[eos]).sum() == 1
        self.kn = KN(cls[np.asarray(train)], self.K, int(cls[eos]), order=order, mode="sent",
                     discount_fallback=FALLBACK)
        n = np.bincount(np.asarray(train), minlength=len(cls)).astype(np.float64)
        size = np.bincount(cls, minlength=self.K).astype(np.float64)
        ncls = np.bincount(cls, weights=n, minlength=self.K)
        self.emis = (n + delta) / (ncls[cls] + delta * size[cls])
        self.log_emis = np.log(self.emis)

    def logprob(self, stream):
        s = np.asarray(stream)
        return self.kn.logprob(self.cls[s]) + self.log_emis[s]

    def dists(self, stream, positions):
        Dc = self.kn.dists(self.cls[np.asarray(stream)], positions)
        return Dc[:, self.cls] * self.emis

    def summary(self):
        return {"classes": self.K, "order": self.order, "delta": self.delta,
                "n_grams": [int(x) for x in self.kn.n_grams],
                "fallback_orders": self.kn.fallback_orders}
