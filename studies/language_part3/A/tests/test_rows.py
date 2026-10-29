"""Rows and bits: the frozen bit list, labels, type flags, LR2 pairs, shuffled labels, and the leak test (review 2)."""

import numpy as np
import pytest
from helpers import synth_corpus

from astruct import build, rows
from astruct.data import CAL, FIT, HOLD, Corpus
from astruct.positions import Positions
from astruct.rates import Rates


def test_bit_list_sizes_at_real_list_lengths():
    """35 domain-acts, 64 pairs, 32 top words, 40 cue words: 364 bits, 44 count-derived, 320 for A-nf."""
    vocab = ["<end>", "<unk>", "[x-y]"] + [f"w{i}" for i in range(80)]
    meta = {"vocab": vocab, "act_names": [f"D-A{i}" for i in range(35)], "pair_names": [[f"D-A{i}", "s"] for i in range(70)],
            "dialogue_ids": [], "lists": {"top32": vocab[3:35], "cue40": vocab[3:43], "pairs64": [[f"D-A{i}", "s"] for i in range(64)]}}
    z = np.zeros(1, np.int64)
    a = {"r_dlg": z, "tok_off": np.zeros(2, np.int64), "tok": z, "concept32": -np.ones(len(vocab), np.int16),
         "concept64": -np.ones(len(vocab), np.int16)}
    names, cd, n_pos = rows.bit_names(Corpus(a, meta))
    assert len(names) == 364 and int(cd.sum()) == 44 and int((~cd).sum()) == 320 and n_pos == 226
    assert len(set(names)) == 364
    derived = {n.split(">=")[0].split("=")[0] for n, d in zip(names, cd) if d}
    assert derived == {"floor_top1", "floor_entropy", "rank", "log2p", "p/top1", "rate_da", "rate_set"}


@pytest.fixture(scope="module")
def built():
    c = synth_corpus("rows", n=240, seed=3)
    fit = c.replies(FIT)
    s1 = {"floor": "F1n4", "picks": {}}
    P_cal = Positions(c, c.replies(CAL))
    fs = build.make_floor(c, fit, s1, P_cal)
    rates = Rates(c, fit)
    hold, _ = build.rows_for(c, fs, rates, c.replies(HOLD), "F1n4", bits=True)
    return c, fs, rates, hold, s1, P_cal


def test_labels_and_flags(built):
    c, _fs, _rates, R, _, _ = built
    off = R["pos_off"]
    names, _cd, n_pos = rows.bit_names(c)
    assert R["bits"].shape == (len(R["y"]), len(names)) and set(np.unique(R["bits"])) <= {0, 1}
    npos = np.bincount(R["pos"], weights=R["y"], minlength=R["n_pos"])
    assert npos.max() <= 1
    # the true token is a candidate whenever it is in the floor's top 8, a due placeholder or the end token
    for k in range(R["n_pos"]):
        cl = R["cand"][off[k]:off[k + 1]]
        x = R["pos_x"][k]
        if x == c.END or x in cl[:8]:
            assert npos[k] == 1
    col = {n: i for i, n in enumerate(names)}
    B = R["bits"]
    is_end = R["cand"] == c.END
    assert (B[:, col["type=end"]] == is_end).all()
    assert (B[:, col["type=due_placeholder"]] <= c.is_ph[R["cand"]]).all()
    assert (B[:, col["rank=extra"]] == (R["rank"] == 8)).all()
    # a position bit is the same on every row of a position
    for j in range(n_pos):
        v = B[:, j]
        assert (np.maximum.reduceat(v, off[:-1]) == np.minimum.reduceat(v, off[:-1])).all()


def test_pair_features_match_brute_force(built):
    c, _, _, R, _, _ = built
    names, cd, n_pos = rows.bit_names(c)
    # the pair partners are exactly the 8 type flags and the candidate identity bits (len(top32) of them)
    q_names = [n for n in names if n.startswith(("type=", "cand="))]
    assert len(q_names) == 8 + len(c.top32) and not any(n.startswith("cand_concept") for n in q_names)
    for keep in (np.ones(len(names), bool), ~cd):
        X = rows.pair_features(R["bits"][:200], names, n_pos, keep).toarray()
        kept = np.flatnonzero(keep)
        pcols = [i for i in kept if i < n_pos]
        qcols = [i for i in kept if names[i] in q_names]
        plain = R["bits"][:200][:, kept]
        ref = np.concatenate([plain] + [R["bits"][:200][:, [p]] * R["bits"][:200][:, qcols] for p in pcols], 1)
        assert X.shape == ref.shape and np.array_equal(X, ref)
        # built in chunks: the same matrix
        Xc = rows.pair_features(R["bits"][:200], names, n_pos, keep, chunk=37).toarray()
        assert np.array_equal(Xc, X)


def test_shuffled_labels(built):
    _, _, _, R, _, _ = built
    ys = rows.shuffled_labels(R, seed=1)
    _t, inC = rows.true_index(R)
    per = np.bincount(R["pos"], weights=ys, minlength=R["n_pos"])
    assert (per == inC).all()
    assert (ys != R["y"]).any()  # it really moves the positives


def test_leak_test_review2(built):
    c, _, _, _, s1, P_cal = built
    fit = c.replies(FIT)
    fold = build.folds_of(c, fit, build.FOLD_SEED)
    cf = build.crossfit(c, fit, fold, s1, P_cal, c.a["r_train"][fit] | True, bits=True)
    lk = cf["leak"]
    assert lk["rows"] > 0 and lk["rows"] == lk["equal"]
    # negative control: without cross-fitting, every true word of a singleton act-slot set has rate 1
    assert lk["true_rows"] > 0 and lk["true_rows_leak_without_crossfit"] == lk["true_rows"]
    # amendment 1: the fold floors never saw those act-slot sets (every S-keyed context unseen, S count 0) ...
    assert lk["S_chains"] == ["F1n4"] and lk["S_lookups"] > 0 and lk["S_lookups"] == lk["S_lookups_unseen"]
    assert lk["S_positions"] > 0 and lk["S_positions"] == lk["S_positions_count_zero"]
    # ... and the floor counted on all of fit (no cross-fitting) has: the check can fail
    fs_full = build.make_floor(c, fit, s1, P_cal)
    P = Positions(c, fit[c.a["r_train"][fit] | True])
    uS, nS = np.unique(c.a["S_hash"][fit], return_counts=True)
    sl = build.s_lookups(fs_full, P, np.isin(P.S, uS[nS == 1]))
    assert sl["positions"] == lk["S_positions"] and sl["lookups"] == lk["S_lookups"]
    assert sl["unseen"] < sl["lookups"] and sl["count_zero"] == 0
    assert build.s_keyed_levels("F1n6", 6) == list(range(6)) and build.s_keyed_levels("F5", 3) == [0]
    assert build.s_keyed_levels("F2c5", 5) == [] and build.s_keyed_levels("F3b", 1) == []
    # the twin rows cover every fit position once
    assert cf["twin"]["n_pos"] == sum(len(c.tokens(r)) for r in fit)


def test_rates_backoff_chain(built):
    c, _, rates, _, _, _ = built
    w = np.arange(c.V)
    unseen_S = np.full(c.V, 12345, np.int64)
    A = np.full(c.V, c.a["A_hash"][c.replies(FIT)[0]])
    assert np.array_equal(rates.rate_set(unseen_S, A, w), rates.backoff_of_set(A, w))
    unseen_A = np.full(c.V, 999, np.int64)
    assert np.allclose(rates.rate_set(unseen_S, unseen_A, w), rates.global_rate(w))
    assert (rates.global_rate(np.array([c.END])) == 1.0).all()  # the end token is in every reply


def test_nofuture_passes_and_catches_an_off_by_one(built, monkeypatch):
    import s2a_rows
    from astruct.positions import Positions as Pos
    c, fs, rates, R, _, _ = built
    hold = c.replies(HOLD)
    res, _ = s2a_rows.nofuture(c, fs, rates, hold, R, "F1n4", log=lambda *a: None)
    assert res["pass"] and res["swap_targets_changed"] == res["positions"]
    orig = Pos.said_before

    def leaky(self, pos, tok):  # "said before" that also sees the current token: a label leak
        pos = np.asarray(pos, np.int64)
        q = self.local[pos] * self._W + np.asarray(tok, np.int64)
        i = np.minimum(np.searchsorted(self._fk, q), len(self._fk) - 1)
        return (self._fk[i] == q) & (self._ft[i] <= self.t[pos])
    monkeypatch.setattr(Pos, "said_before", leaky)
    R2, _ = build.rows_for(c, fs, rates, hold, "F1n4", bits=True)
    res2, _ = s2a_rows.nofuture(c, fs, rates, hold, R2, "F1n4", log=lambda *a: None)
    assert not res2["pass"] and ("said" in res2["mismatched_fields"]["swap"] or "bits" in res2["mismatched_fields"]["swap"])
    monkeypatch.setattr(Pos, "said_before", orig)
