"""Pilot P2 of docs/13 A5 (src/tmlm/cvtm/templates.py): exact template keys and counts against a
brute-force dictionary count, the hash table, the random partition, the mixture fit against the source
script's fit_mix, the residue 2a/2b split, a planted PPMI-SVD map, and the verdict rules."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tmlm.cvtm import templates as T

ROOT = Path(__file__).resolve().parents[1]


def _stream(seed=0, n_docs=12, vocab=40):
    rng = np.random.default_rng(seed)
    lens = rng.integers(1, 60, size=n_docs)
    ds = np.zeros(n_docs + 1, np.int64)
    ds[1:] = np.cumsum(lens)
    tok = rng.integers(0, vocab, size=int(ds[-1])).astype(np.uint16)
    return tok, ds


def _cmap(seed, vocab, kc, kf):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, kc, size=vocab)
    f = rng.integers(0, kf, size=vocab)
    c[:kc] = np.arange(kc)
    f[:kf] = np.arange(kf)
    return T.ClassMap("toy", T.contiguous(c), T.contiguous(f))


def _brute_ctx(name, cm, tok, lo, p, vocab):
    out = []
    for kind, off in T.TEMPLATES[name]:
        q = p - off
        if q < lo:
            out.append(("EDGE", kind))
        else:
            t = int(tok[q])
            out.append(t if kind == T.TOK else (int(cm.coarse[t]) if kind == T.COARSE else int(cm.fine[t])) + 10 ** 6 * kind)
    return tuple(out)


def test_template_names_are_the_a5_list():
    assert T.TEMPLATE_NAMES == ("c2_1", "c2_21", "t1_c2_2", "t1_c3_2", "c3_321", "t21_c2_3", "t32_skip", "t321_c2_4")
    assert not T.uses_map("t32_skip")
    assert all(T.uses_map(n) for n in T.TEMPLATE_NAMES if n != "t32_skip")


def test_counts_equal_brute_force():
    vocab = 40
    cm = _cmap(3, vocab, 5, 11)
    tok, ds = _stream(0, vocab=vocab)
    etok, eds = _stream(1, n_docs=5, vocab=vocab)
    keep = np.ones(ds.shape[0] - 1, bool)
    keep[[2, 5]] = False
    rows = np.arange(0, etok.shape[0], 2)
    for name in T.TEMPLATE_NAMES:
        tc = T.count_template(name, cm, vocab, etok, eds, rows, tok, ds, keep)
        ctxc, fullc = Counter(), Counter()
        for d in range(ds.shape[0] - 1):
            if not keep[d]:
                continue
            for p in range(ds[d], ds[d + 1]):
                c = _brute_ctx(name, cm, tok, ds[d], p, vocab)
                ctxc[c] += 1
                fullc[(c, int(tok[p]))] += 1
        dof = np.repeat(np.arange(eds.shape[0] - 1), np.diff(eds))
        for i, p in enumerate(rows):
            c = _brute_ctx(name, cm, etok, eds[dof[p]], p, vocab)
            assert tc.n[i] == ctxc[c], name
            assert tc.c[i] == fullc[(c, int(etok[p]))], name
        assert tc.counting_positions == int(np.diff(ds)[keep].sum())
        assert tc.n.sum() > 0


def test_keys_are_injective_and_in_range():
    vocab = 30
    cm = _cmap(5, vocab, 4, 9)
    tok, ds = _stream(2, vocab=vocab)
    for name in T.TEMPLATE_NAMES:
        kinds, offs, maps, bases, edges = T.template_arrays(name, cm, vocab)
        keys = T.stream_keys(tok.astype(np.int64), ds, kinds, offs, maps, bases, edges)
        top = T.check_key_range(kinds, bases, vocab)
        assert keys.min() >= 0 and keys.max() * vocab + vocab - 1 < top
        dof = np.repeat(np.arange(ds.shape[0] - 1), np.diff(ds))
        tuples = [_brute_ctx(name, cm, tok, ds[dof[p]], p, vocab) for p in range(tok.shape[0])]
        m = {}
        for k, t in zip(keys.tolist(), tuples):
            assert m.setdefault(k, t) == t, name       # one key, one tuple
        assert len(set(m.values())) == len(m)


def test_full_scale_key_range_fits_int64():
    for name in T.TEMPLATE_NAMES:
        kinds = np.array([k for k, _ in T.TEMPLATES[name]], np.int64)
        for kc, kf in ((243, 2497), (256, 2048)):
            T.check_key_range(kinds, np.array([8001, kc + 1, kf + 1], np.int64), 8000)


def test_hash_table_lookup():
    rng = np.random.default_rng(7)
    keys = np.unique(rng.integers(0, 2 ** 62, size=5000))
    tk, ti, mask = T.build_table(keys)
    for i in range(0, keys.shape[0], 37):
        assert T.lookup(tk, ti, mask, keys[i]) == i
    absent = np.setdiff1d(rng.integers(0, 2 ** 62, size=500), keys)
    assert all(T.lookup(tk, ti, mask, k) == -1 for k in absent)


def test_random_like_preserves_histogram_and_is_seeded():
    cm = _cmap(9, 200, 13, 57)
    r1 = T.random_like(cm, (100, 1))
    r2 = T.random_like(cm, (100, 1))
    r3 = T.random_like(cm, (101, 1))
    assert np.array_equal(r1.coarse, r2.coarse) and np.array_equal(r1.fine, r2.fine)
    assert not np.array_equal(r1.coarse, r3.coarse)
    assert np.array_equal(np.sort(np.bincount(r1.coarse)), np.sort(np.bincount(cm.coarse)))
    assert np.array_equal(np.sort(np.bincount(r1.fine)), np.sort(np.bincount(cm.fine)))
    assert not np.array_equal(r1.coarse, cm.coarse)


def test_fit_mix_equals_source_script():
    sys.path.insert(0, str(ROOT / "experiments"))
    import tm_levers_feasibility_causal as R1

    rng = np.random.default_rng(11)
    n = 3000
    pb = rng.uniform(0.001, 0.5, size=n)
    rows = np.concatenate([rng.choice(n, 2000, replace=False), rng.choice(n, 1500, replace=False)])
    b = np.concatenate([rng.integers(0, 6, 2000), 6 + rng.integers(0, 6, 1500)])
    p = rng.uniform(0, 1, size=rows.shape[0])
    fit = np.zeros(n, bool)
    fit[::2] = True
    th, res = T.fit_mix(pb, rows, b, p, 12, fit, iters=200)
    th2, res2 = R1.fit_mix(pb, rows, b, p, 12, fit, iters=200)
    assert np.array_equal(th, th2)
    assert np.array_equal(T.eval_mix(pb, rows, b, p, th), R1.eval_mix(pb, rows, b, p, th2))


def test_mixture_recovers_a_useful_component():
    rng = np.random.default_rng(4)
    n = 20000
    pb = np.full(n, 0.01)
    good = rng.uniform(size=n) < 0.5
    counts = [T.TemplateCounts("x", n=np.where(good, 8, 0).astype(np.int64),
                               c=np.where(good, 6, 0).astype(np.int64), n_contexts_eval=1, counting_positions=1)]
    rows, b, p, nb = T.components(counts, np.zeros(n, np.int64))
    fit = np.arange(n) % 2 == 0
    th, _ = T.fit_mix(pb, rows, b, p, nb, fit)
    bits = T.eval_mix(pb, rows, b, p, th)
    base = -np.log2(pb)
    assert bits[~fit & good].mean() < base[~fit & good].mean() - 3
    assert np.allclose(bits[~good], base[~good])


def test_split_2ab():
    di = np.array([50, 3, 7, 9, 12, 40, 41, 2])
    res = np.array([2, 2, 1, 2, 0, 2, 2, 2])
    a, b = T.split_2ab(di, res)
    # residue-2 docs by ID: 2(i7), 3(i1), 9(i3), 40(i5), 41(i6), 50(i0) -> 2a = i7, i3, i6; 2b = i1, i5, i0
    assert np.flatnonzero(a).tolist() == [3, 6, 7]
    assert np.flatnonzero(b).tolist() == [0, 1, 5]
    assert not (a & b).any() and not (a | b)[res != 2].any()


def test_ppmi_svd_recovers_planted_classes():
    # two word classes: class-0 words are followed by class-1 words and vice versa
    rng = np.random.default_rng(1)
    vocab, n_docs, L = 20, 60, 200
    cls = np.array([0] * 10 + [1] * 10)
    toks = []
    for _ in range(n_docs):
        c = 0
        seq = []
        for _ in range(L):
            seq.append(int(rng.integers(0, 10)) + 10 * c)
            c = 1 - c
        toks.extend(seq)
    tok = np.array(toks, np.uint16)
    ds = np.arange(0, n_docs * L + 1, L, dtype=np.int64)
    cm, info = T.ppmi_svd_map(tok, ds, np.ones(n_docs, bool), vocab, sizes=(2, 4), dims=4, seed=100)
    assert cm.k_coarse == 2
    assert len(set(cm.coarse[cls == 0])) == 1 and len(set(cm.coarse[cls == 1])) == 1
    assert cm.coarse[0] != cm.coarse[10]
    cm2, _ = T.ppmi_svd_map(tok, ds, np.ones(n_docs, bool), vocab, sizes=(2, 4), dims=4, seed=100)
    assert np.array_equal(cm.fine, cm2.fine)


@pytest.mark.parametrize("g,ci,r,want", [
    (0.003, (0.001, 0.005), 0.003, "NO-GO"),
    (0.02, (0.01, 0.03), 0.001, "NO-GO"),        # overlap: NO-GO precedes
    (0.012, (0.004, 0.02), 0.006, "GO"),
    (0.012, (-0.001, 0.02), 0.006, "IN_BETWEEN"),
    (0.007, (0.003, 0.011), 0.004, "IN_BETWEEN"),
])
def test_verdict(g, ci, r, want):
    v = T.verdict(g, ci, r)
    assert v["verdict"] == want
    assert v["overlap"] == (want == "NO-GO" and g >= 0.010 and ci[0] > 0)
