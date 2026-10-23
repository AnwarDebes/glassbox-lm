"""Shared measurement utilities for line A and later steps.

- report: bits per token and perplexity together, always with split, token count and vocabulary.
- block_bootstrap: paired per-token differences, resampling the 100-sentence blocks.
- duplicate_mask / duplicate_report: tokens that lie inside a span of at least 8 tokens that
  also occurs in train.
- contract: the sum-to-one, above-zero and fast-path test of section 4.
"""
import numpy as np
from numba import njit

VOCAB = "PTB Mikolov 10k (9,999 words incl. <unk>, plus <eos>), closed"
LN2 = np.log(2.0)


def report(logp, split, vocab=VOCAB, eos_counted=True):
    """logp: natural log probabilities of the true tokens of one split."""
    logp = np.asarray(logp, dtype=np.float64)
    bits = -logp.mean() / LN2
    return {"split": split, "tokens": int(len(logp)), "vocab": vocab, "eos_counted": eos_counted,
            "bits_per_token": float(bits), "ppl": float(2.0 ** bits)}


def block_bootstrap(d, blocks, n_boot=1000, seed=0, level=0.95):
    """Mean of paired per-token differences d (e.g. bits of arm A minus bits of arm B) with a
    percentile interval from resampling whole blocks (ratio estimator sum d / tokens)."""
    d = np.asarray(d, dtype=np.float64)
    ub, inv = np.unique(blocks, return_inverse=True)
    S = np.bincount(inv, weights=d)
    n = np.bincount(inv).astype(np.float64)
    nb = len(ub)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot).astype(np.float64)
    stat = (W @ S) / (W @ n)
    a = (1 - level) / 2
    lo, hi = np.quantile(stat, [a, 1 - a])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "level": level,
            "n_boot": n_boot, "n_blocks": int(nb), "tokens": int(len(d)), "seed": seed,
            "boot_sd": float(stat.std(ddof=1))}


def bits_diff(logp_a, logp_b):
    """Per-token bits of A minus bits of B (negative = A better)."""
    return (np.asarray(logp_b) - np.asarray(logp_a)) / LN2


@njit(cache=True)
def _window_hashes(x, L, ok):
    n = len(x) - L + 1
    h = np.zeros(max(n, 0), dtype=np.uint64)
    for a in range(n):
        v = np.uint64(14695981039346656037)
        for t in range(L):
            v = (v ^ np.uint64(x[a + t] + 1)) * np.uint64(1099511628211)
        h[a] = v if ok[a] else np.uint64(0)
    return h


@njit(cache=True)
def _matches(tr, th, torder, ev, eh, L):
    """For each eval window, True if the same L tokens occur in train (hash, then exact check)."""
    n = len(eh)
    out = np.zeros(n, dtype=np.bool_)
    sh = th[torder]
    for a in range(n):
        lo = np.searchsorted(sh, eh[a])
        j = lo
        while j < len(sh) and sh[j] == eh[a]:
            p = torder[j]
            same = True
            for t in range(L):
                if tr[p + t] != ev[a + t]:
                    same = False
                    break
            if same:
                out[a] = True
                break
            j += 1
    return out


def duplicate_mask(train, train_blocks, ev, min_len=8):
    """True for every eval token inside a span of >= min_len tokens that occurs in train.

    A span of length >= min_len occurs in train exactly when every min_len-window inside it does,
    so the mask is the union of the matching windows. Train windows that cross a join of two
    blocks that are not neighbours in the original file are excluded (not real text)."""
    tr = np.asarray(train, dtype=np.int64)
    ev = np.asarray(ev, dtype=np.int64)
    L = min_len
    tb = np.asarray(train_blocks)
    ok = np.ones(max(len(tr) - L + 1, 0), dtype=np.bool_)
    ok &= (tb[L - 1:] - tb[:len(ok)]) <= 1
    th = _window_hashes(tr, L, ok)
    idx = np.flatnonzero(ok)
    torder = idx[np.argsort(th[idx], kind="stable")]
    eh = _window_hashes(ev, L, np.ones(max(len(ev) - L + 1, 0), dtype=np.bool_))
    hit = _matches(tr, th, torder, ev, eh, L)
    cover = np.zeros(len(ev) + 1, dtype=np.int64)
    a = np.flatnonzero(hit)
    np.add.at(cover, a, 1)
    np.add.at(cover, a + L, -1)
    return np.cumsum(cover)[:-1] > 0


def duplicate_report(train, train_blocks, ev, split, min_len=8):
    m = duplicate_mask(train, train_blocks, ev, min_len)
    return {"split": split, "min_len": min_len, "tokens": int(len(ev)),
            "dup_tokens": int(m.sum()), "dup_share": float(m.mean())}, m


def contract(dists_fn, logp_fast, stream_len, n=1000, seed=0, true_ids=None):
    """Section 4 contract on n random positions: sum to 1 within 1e-6, minimum above 0, and the
    fast path equal to the vector entry of the true word (|log difference| <= 1e-9)."""
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(stream_len, size=min(n, stream_len), replace=False))
    D = dists_fn(pos)
    sum_err = float(np.abs(D.sum(1) - 1.0).max())
    pmin = float(D.min())
    fast_err = float(np.abs(np.log(D[np.arange(len(pos)), true_ids[pos]]) - logp_fast[pos]).max())
    return {"positions": int(len(pos)), "seed": seed, "max_sum_err": sum_err, "min_p": pmin,
            "max_fast_vs_vector_log_err": fast_err,
            "pass": bool(sum_err <= 1e-6 and pmin > 0 and fast_err <= 1e-9)}
