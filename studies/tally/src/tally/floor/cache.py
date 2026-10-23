"""Decayed unigram cache over the running token stream (member C of the floor).

P_cache(w | history) = c(w) / C, with c(w) = sum of decay^(i - 1 - j) over the earlier tokens
j of the current segment with w_j = w, and C = sum of c over V. decay = 0.5 ** (1 / half_life);
half_life = inf gives the plain unigram cache.

Reset rule (PTB has no document marks): "none" keeps one segment per split stream; "block"
starts a new segment at every 100-sentence block, so the score of a token depends only on its
own block and on train (this is what the block bootstrap and cross-fitting by block assume).
Words in skip (for example <eos>) are not added to the cache; they still advance the clock.

When the cache is empty (first token of a segment, or only skipped words so far) the member is
not defined; the caller then uses the fallback member (the Kneser-Ney member), as section 4 of
the design says for members that propose only a few words. Note that the cache alone gives 0 to
every word it has not seen; only the mixture with a Kneser-Ney weight above 0 is above 0.
"""
import numpy as np
from numba import njit

RESETS = ("none", "block")


def decay_of(half_life):
    return 1.0 if not np.isfinite(half_life) else 0.5 ** (1.0 / half_life)


def segment_starts(blocks, reset):
    """Boolean array: True where a new cache segment starts."""
    st = np.zeros(len(blocks), dtype=np.bool_)
    st[0] = True
    if reset == "block":
        st[1:] = blocks[1:] != blocks[:-1]
    elif reset != "none":
        raise ValueError(reset)
    return st


@njit(cache=True)
def _run(stream, starts, decay, skip, V, pos, want_true):
    """One pass over the stream. Returns the true-word probability at every token (nan if the
    cache is empty) and the full vector at the sorted positions pos (rows of nan if empty)."""
    n = len(stream)
    val = np.zeros(V)
    last = np.zeros(V, dtype=np.int64)
    C = 0.0
    p_true = np.full(n if want_true else 0, np.nan)
    out = np.full((len(pos), V), np.nan)
    q = 0
    for i in range(n):
        if starts[i]:
            val[:] = 0.0
            C = 0.0
        w = stream[i]
        if C > 0:
            if want_true:
                p_true[i] = val[w] * decay ** (i - 1 - last[w]) / C
            while q < len(pos) and pos[q] == i:
                for v in range(V):
                    out[q, v] = val[v] * decay ** (i - 1 - last[v]) / C
                q += 1
        else:
            while q < len(pos) and pos[q] == i:
                q += 1
        # add token i
        if skip[w]:
            C = C * decay
        else:
            val[w] = val[w] * decay ** (i - last[w]) + 1.0
            last[w] = i
            C = C * decay + 1.0
    return p_true, out


class Cache:
    def __init__(self, V, half_life=np.inf, reset="block", skip=()):
        assert reset in RESETS
        self.V, self.half_life, self.reset = V, half_life, reset
        self.decay = decay_of(half_life)
        self.skip = np.zeros(V, dtype=np.bool_)
        self.skip[list(skip)] = True

    def config(self):
        return {"half_life": float(self.half_life), "reset": self.reset,
                "skip": [int(x) for x in np.flatnonzero(self.skip)]}

    def prob(self, stream, blocks, fallback):
        """P_cache of the true word at every token; fallback (array of the same length, the
        Kneser-Ney member's probabilities) where the cache is empty."""
        p, _ = _run(np.asarray(stream, dtype=np.int64), segment_starts(blocks, self.reset),
                    self.decay, self.skip, self.V, np.zeros(0, dtype=np.int64), True)
        empty = np.isnan(p)
        p[empty] = fallback[empty]
        return p, empty

    def dists(self, stream, blocks, positions, fallback_rows):
        """Full vectors over V at positions; fallback_rows (len(positions), V) where empty."""
        positions = np.asarray(positions, dtype=np.int64)
        order = np.argsort(positions, kind="stable")
        _, out = _run(np.asarray(stream, dtype=np.int64), segment_starts(blocks, self.reset),
                      self.decay, self.skip, self.V, positions[order], False)
        res = np.empty_like(out)
        res[order] = out
        empty = np.isnan(res[:, 0])
        res[empty] = fallback_rows[empty]
        return res
