"""Self-tests of C-core in the counting environment (.venv-tm: numpy, numba, scipy, sklearn).

    scripts/run_stage.sh selftest_tm        # python -I -B, nice 19, 2 threads, caches in C/tmp

Tiny or synthetic data only; the PTB parts use train blocks only (no calib, valid or dev-test number
is computed). Prints PASS or FAIL per test and exits 1 on any failure. Under 2 minutes."""
import glob
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import env  # noqa: E402

env.guard()
import numpy as np  # noqa: E402

from ccore import chains, codes, data, gates, hashing, layout, paths, step0, systems  # noqa: E402
from ccore.floor import Floor  # noqa: E402

RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


# ---------------------------------------------------------------- tree E: merge losses and tree
@test
def merge_losses_equal_brute_force():
    rng = np.random.default_rng(0)
    K, f = 8, 2
    M = rng.integers(0, 20, size=(K + f, K + f)).astype(float)
    M[K, :] = 0          # an <eos>-like item never on the left
    M[:, K + 1] = 0      # a BOS-like item never on the right
    loss = codes.merge_losses(M, list(range(K)))
    base = codes.objective(M)
    for u in range(K):
        for v in range(u + 1, K):
            W = M.copy()
            W[u, :] += W[v, :]
            W[:, u] += W[:, v]
            W = np.delete(np.delete(W, v, 0), v, 1)
            assert abs((base - codes.objective(W)) - loss[u, v]) < 1e-8, (u, v)
    assert np.isinf(loss[np.tril_indices(K)]).all()


@test
def balanced_tree_is_a_valid_code():
    rng = np.random.default_rng(1)
    K, depth = 16, 4
    M = rng.integers(0, 50, size=(K + 2, K + 2)).astype(float)
    M[K, :] = 0
    M[:, K + 1] = 0
    p, levels = codes.balanced_tree(M, K, depth)
    assert sorted(p.tolist()) == list(range(K)), "paths are not a bijection onto 0..K-1"
    assert [len(lv) for lv in levels] == [8, 4, 2, 1]
    for li, lv in enumerate(levels):          # level li merges clusters of size 2**li
        for c0, c1, _ in lv:
            assert len(c0) == len(c1) == 2 ** li and min(c0) < min(c1)
            # all leaves of a cluster share the prefix above this level; the children differ at bit li
            pre = {int(p[x]) >> (li + 1) for x in c0 + c1}
            assert len(pre) == 1
            assert {int(p[x]) >> li & 1 for x in c0} == {0} and {int(p[x]) >> li & 1 for x in c1} == {1}
    p2, _ = codes.balanced_tree(M, K, depth)
    assert np.array_equal(p, p2), "not deterministic"


@test
def greedy_merge_is_least_loss():
    rng = np.random.default_rng(2)
    K = 8
    M = rng.integers(0, 30, size=(K + 2, K + 2)).astype(float)
    loss = codes.merge_losses(M, list(range(K)))
    i, j = divmod(int(np.argmin(loss)), K)
    _, levels = codes.balanced_tree(M, K, 3)
    first = levels[0]
    assert any(set(c0 + c1) == {i, j} for c0, c1, _ in first), "the least-loss pair was not merged"


@test
def tree_E_on_tiny_train():
    P = data.load_ptb()
    Q = data.dry_splits(P, sizes={"train": 12, "calib": 2, "valid": 2, "devtest": 2}, seed=11)
    tr, V, eos = Q["ids"]["train"], Q["V"], Q["eos"]
    cls, path6, leaf, meta = codes.tree_E(tr, V, eos)
    words = np.arange(V) != eos
    assert cls[eos] == 64 and set(np.unique(cls[words]).tolist()) == set(range(64))
    assert sorted(leaf.tolist()) == list(range(64)) and np.array_equal(path6[words], leaf[cls[words]])
    obj = meta["exchange"][0]["objective"]
    assert all(b >= a - 1e-6 for a, b in zip(obj, obj[1:])), "exchange objective decreased"
    assert all(x >= -1e-9 for lv in meta["merge_losses_per_level"] for x in lv), "a merge raised the objective"
    assert [len(lv) for lv in meta["merge_losses_per_level"]] == [32, 16, 8, 4, 2, 1]


# ---------------------------------------------------------------- tree B
@test
def tree_B_splits_at_weighted_median_and_keeps_ties():
    rng = np.random.default_rng(3)
    V, dim = 400, 12
    X = rng.standard_normal((V, dim))
    X[100:130] = X[100]                       # 30 identical vectors
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    w = rng.integers(0, 50, size=V).astype(float)
    w[200:260] = 0                            # words never seen in train
    eos = 7
    path, info = codes.tree_B(X, w, eos)
    assert path.min() >= 0 and path.max() < 256
    assert len(set(path[100:130].tolist())) == 1, "identical vectors were split"
    # root split: left weight closest to half among distinct-projection boundaries
    node = np.flatnonzero(np.arange(V) != eos)
    left, right, note = codes._split(X, w, node)
    assert note == "split" and len(left) + len(right) == len(node)
    assert set(np.flatnonzero((path >> 7) == 0).tolist()) - {eos} == set(left.tolist())
    p, ww = codes._project(X, w, node)
    best = None
    for thr in np.unique(p)[:-1]:             # every cut between distinct projections
        lw = ww[p <= thr].sum()
        if best is None or abs(lw - ww.sum() / 2) < abs(best - ww.sum() / 2):
            best = lw
    assert abs(w[left].sum() - best) < 1e-9, "the root cut is not the one closest to half the weight"
    assert np.all(p[np.isin(node, left)].max() < p[np.isin(node, right)].min())
    p2, _ = codes.tree_B(X, w, eos)
    assert np.array_equal(path, p2)


@test
def c4_mapping_rule():
    P = data.load_ptb()
    counts = np.bincount(P["ids"]["train"], minlength=P["V"]).astype(np.float64)
    X, kind, cov = codes.c4_vectors(P["vocab"], counts, P["eos"])
    nz = np.arange(P["V"]) != P["eos"]
    assert np.allclose(np.linalg.norm(X[nz], axis=1), 1.0)
    assert kind[P["vocab"].index("N")] == "digits" and cov["unk_is_fallback"]
    fb = np.flatnonzero(kind == "fallback")
    assert len(fb) > 100 and np.allclose(X[fb], X[fb[0]])
    shares = sum(cov[k]["train_token_share"] for k in ("row", "digits", "fallback", "eos"))
    assert abs(shares - 1.0) < 1e-12
    words = (paths.C4DIR / "words.txt").read_text(encoding="utf-8").splitlines()
    vec = np.load(paths.C4DIR / "vectors.npy")
    w_the = P["w2i"]["the"]
    ref = vec[words.index("the")].astype(np.float64)
    assert np.allclose(X[w_the], ref / np.linalg.norm(ref))
    print(f"    coverage: {json.dumps(cov)}")


# ---------------------------------------------------------------- layout
def tiny_stream(rng, V=30, eos=0, n_sent=40):
    out = []
    for _ in range(n_sent):
        out += list(rng.integers(1, V, size=rng.integers(1, 9))) + [eos]
    return np.array(out, dtype=np.int64)


@test
def features_decode_roundtrip_and_pads():
    rng = np.random.default_rng(4)
    V, eos = 30, 0
    x = tiny_stream(rng, V, eos)
    p6 = rng.integers(0, 64, size=V)
    p8 = rng.integers(0, 256, size=V)
    X = layout.features(x, eos, p6, p8)
    assert X.shape == (len(x), 60) and X.dtype == np.uint8
    E6, B8, pad = layout.decode(X)
    prev = layout.prev_words(x, eos)
    for t in range(len(x)):
        for j in range(4):
            w = prev[t, j]
            if w < 0:
                assert pad[t, j] and E6[t, j] == 64 and B8[t, j] == 256
            else:
                assert not pad[t, j] and E6[t, j] == p6[w] and B8[t, j] == p8[w]
                assert x[t - 1 - j] == w
    starts = np.flatnonzero(np.concatenate([[True], x[:-1] == eos]))
    assert pad[starts].all(), "a sentence start without pads"


@test
def future_change_inputs():
    rng = np.random.default_rng(5)
    V, eos = 30, 0
    x = tiny_stream(rng, V, eos)
    p6 = rng.integers(0, 64, size=V)
    p8 = rng.integers(0, 256, size=V)
    X0 = layout.features(x, eos, p6, p8)
    for t in rng.choice(np.arange(1, len(x) - 1), size=20, replace=False):
        x1 = x.copy()
        x1[t:] = rng.integers(0, V, size=len(x) - t)
        x1[-1] = eos
        X1 = layout.features(x1, eos, p6, p8)
        assert np.array_equal(X0[:t + 1], X1[:t + 1]), f"inputs at or before {t} changed"
        c0 = chains.class_contexts(x, eos, 4)
        c1 = chains.class_contexts(x1, eos, 4)
        assert np.array_equal(c0[:t + 1], c1[:t + 1])


# ---------------------------------------------------------------- chains
def ad_reference(keys_train, y, keys_eval, K):
    """Dictionary implementation of interpolated modified absolute discounting (levels finest first)."""
    from collections import Counter

    def disc(counts):
        from tally.floor.kn import FALLBACK, discounts
        return discounts(np.asarray(counts, dtype=np.int64), FALLBACK)
    n0 = np.bincount(y, minlength=K)
    D0 = disc(n0)
    d0 = D0[np.minimum(n0, 3)]
    P0 = np.maximum(n0 - d0, 0) / n0.sum() + d0.sum() / n0.sum() / K
    tables = []
    for kk in keys_train[::-1]:
        c = Counter(zip(kk.tolist(), y.tolist()))
        D = disc(list(c.values()))
        ctx = {}
        for (h, cl), n in c.items():
            ctx.setdefault(h, {})[cl] = n
        tables.append((ctx, D))
    out = np.zeros((len(keys_eval[0]), K))
    for i in range(len(keys_eval[0])):
        p = P0.copy()
        for (ctx, D), kk in zip(tables, keys_eval[::-1]):
            h = int(kk[i])
            if h not in ctx:
                continue
            cnt = ctx[h]
            N = sum(cnt.values())
            g = sum(D[min(n, 3)] for n in cnt.values()) / N
            q = g * p
            for cl, n in cnt.items():
                q[cl] += max(n - D[min(n, 3)], 0) / N
            p = q
        out[i] = p
    return out


@test
def ad_chain_equals_reference_and_sums_to_one():
    rng = np.random.default_rng(6)
    K = chains.K
    n = 3000
    comp = {"E6": rng.integers(0, 65, size=(n, 4)), "B8": rng.integers(0, 257, size=(n, 4))}
    comp["B6"] = np.where(comp["B8"] == 256, 64, comp["B8"] >> 2)
    comp["B4"] = np.where(comp["B8"] == 256, 16, comp["B8"] >> 4)
    # coarse inputs so that contexts repeat
    for k in ("E6", "B8", "B6", "B4"):
        comp[k] = comp[k] % 5
    y = rng.integers(0, K, size=n)
    for name, spec in chains.SPECS.items():
        kt = [chains.level_key({k: v[:2000] for k, v in comp.items()}, lv) for lv in spec]
        ke = [chains.level_key({k: v[2000:] for k, v in comp.items()}, lv) for lv in spec]
        m = chains.ADChain(name, kt, y[:2000])
        P = m.dist(ke)
        R = ad_reference(kt, y[:2000], ke, K)
        assert np.abs(P - R).max() < 1e-12, (name, np.abs(P - R).max())
        assert np.abs(P.sum(1) - 1).max() < 1e-12 and P.min() > 0


@test
def pair_kn_equals_tally_kn():
    from tally.floor.kn import FALLBACK, KN
    rng = np.random.default_rng(7)
    K = chains.K
    eos = K - 1
    def stream(n_sent):
        out = []
        for _ in range(n_sent):
            out += list(rng.integers(0, 12, size=rng.integers(1, 7))) + [eos]
        return np.array(out, dtype=np.int64)
    tr, ev = stream(800), stream(200)
    ev[5:9] = [40, 41, 42, 43]                 # contexts never seen in train
    for order in (2, 3, 4, 5):
        kn = KN(tr, K, eos, order=order, mode="sent", discount_fallback=FALLBACK)
        ref = kn.dists(ev, np.arange(len(ev)))
        pk = chains.PairKN(chains.class_contexts(tr, eos, order - 1), tr, order)
        got = pk.dist(chains.class_contexts(ev, eos, order - 1))
        err = np.abs(got - ref).max()
        assert err < 1e-12, (order, err)
        assert np.abs(got.sum(1) - 1).max() < 1e-12


@test
def pair_kn_runs_on_permuted_targets():
    rng = np.random.default_rng(8)
    K = chains.K
    eos = K - 1
    out = []
    for _ in range(500):
        out += list(rng.integers(0, 10, size=rng.integers(1, 6))) + [eos]
    tr = np.array(out)
    ctx = chains.class_contexts(tr, eos, 4)
    y = tr[np.random.default_rng(1).permutation(len(tr))]
    P = chains.PairKN(ctx, y, 5).dist(ctx[:300])
    assert np.abs(P.sum(1) - 1).max() < 1e-12 and P.min() > 0


@test
def every_code_bit_is_in_a_counted_chain():
    tall5 = [chains.te_spec(5)] + [chains.SPECS[n] for n in ("TB", "TEB", "TB4", "TEB2")]
    cov = chains.covered_bits(tall5)
    assert cov.all(), f"T-ALL at T-E order 5 leaves bits {np.flatnonzero(~cov).tolist()} uncovered"
    # every input bit changes the key of a covering level: flip it in the decoded components
    rng = np.random.default_rng(9)
    n = 200
    E6 = rng.integers(0, 64, size=(n, 4))
    B8 = rng.integers(0, 256, size=(n, 4))
    pad = np.zeros((n, 4), dtype=bool)

    def comps(E6, B8, pad):
        E = np.where(pad, 64, E6)
        B = np.where(pad, 256, B8)
        return {"E6": E, "B8": B, "B6": np.where(pad, 64, B >> 2), "B4": np.where(pad, 16, B >> 4)}
    base = comps(E6, B8, pad)
    specs = [chains.te_spec(5)] + [chains.SPECS[k] for k in ("TB", "TEB", "TB4", "TEB2")]
    for k in range(4):
        for bit in range(6):
            E1 = E6.copy()
            E1[:, k] ^= 1 << bit
            c = comps(E1, B8, pad)
            assert any((chains.level_key(c, lv) != chains.level_key(base, lv)).all() for s in specs for lv in s)
        for bit in range(8):
            B1 = B8.copy()
            B1[:, k] ^= 1 << bit
            c = comps(E6, B1, pad)
            hit = [lv for s in specs for lv in s if (chains.level_key(c, lv) != chains.level_key(base, lv)).all()]
            assert hit and any(("B8", k + 1) in lv for lv in hit), (k, bit)
        p1 = pad.copy()
        p1[:, k] = True
        c = comps(E6, B8, p1)
        assert any((chains.level_key(c, lv) != chains.level_key(base, lv)).all() for s in specs for lv in s)
    # which bits each of the 8 registered trials leaves out (reported, not a rule)
    from ccore import twin
    for name, members in twin.TRIALS:
        sp = [chains.te_spec(m[1]) if isinstance(m, tuple) else chains.SPECS[m] for m in members]
        unc = np.flatnonzero(~chains.covered_bits(sp)).tolist()
        print(f"    {name}: {len(unc)} of 60 bits not in any chain")


# ---------------------------------------------------------------- floor, contract, future change
def tiny_ptb():
    P = data.load_ptb()
    return data.dry_splits(P, sizes={"train": 12, "calib": 2, "valid": 2, "devtest": 2}, seed=11)


@test
def floor_contract_and_marginals():
    Q = tiny_ptb()
    ids, bl, V, eos = Q["ids"], Q["blocks"], Q["V"], Q["eos"]
    fl = Floor(ids, bl, V, eos)
    rng = np.random.default_rng(10)
    cls = np.where(np.arange(V) == eos, 64, rng.integers(0, 64, size=V))
    for s in ("calib", "valid"):
        pF, b, L = fl.prob(ids[s], bl[s])
        pos = np.sort(rng.choice(len(ids[s]), size=200, replace=False))
        D = fl.dists(ids[s], bl[s], pos, b)
        assert np.abs(D.sum(1) - 1).max() < 1e-9 and D.min() > 0
        assert np.abs(np.log(D[np.arange(len(pos)), ids[s][pos]]) - np.log(pF[pos])).max() < 1e-9
        Fc, chk = fl.class_marginals(ids[s], bl[s], cls, 65, b, chunk=97)
        assert chk["pass"] and np.allclose(Fc[pos], D @ np.eye(65)[cls])


@test
def future_change_floor_and_class_models():
    Q = tiny_ptb()
    ids, bl, V, eos = Q["ids"], Q["blocks"], Q["V"], Q["eos"]
    fl = Floor(ids, bl, V, eos)
    rng = np.random.default_rng(12)
    cls = np.where(np.arange(V) == eos, 64, rng.integers(0, 64, size=V))
    te = chains.TEModel(cls[ids["train"]], 5)
    x = ids["calib"]
    b0 = fl.buckets(x)[0]
    for t in rng.choice(np.arange(5, len(x) - 5), size=6, replace=False):
        x1 = x.copy()
        x1[t:] = rng.integers(0, V, size=len(x) - t)
        x1[-1] = eos
        b1 = fl.buckets(x1)[0]
        assert np.array_equal(b0[:t + 1], b1[:t + 1])
        pos = np.array([t - 3, t - 1, t])
        assert np.array_equal(fl.dists(x, bl["calib"], pos, b0), fl.dists(x1, bl["calib"], pos, b1))
        assert np.array_equal(te.dist(cls[x])[pos], te.dist(cls[x1])[pos])


# ---------------------------------------------------------------- systems
def synthetic_split(rng, n, V, K, cls):
    F = rng.dirichlet(np.ones(V) * 0.3, size=n)
    F = np.maximum(F, 1e-9)
    F /= F.sum(1, keepdims=True)
    w = np.array([rng.choice(V, p=F[i]) for i in range(n)])
    M = np.zeros((V, K))
    M[np.arange(V), cls] = 1
    Fc = F @ M
    e = rng.random(V) + 0.1
    for c in range(K):
        e[cls == c] /= e[cls == c].sum()
    b = rng.integers(0, 40, size=n)
    sp = systems.Split(w, cls[w], F[np.arange(n), w], Fc, b, b // 8, np.arange(n) // 50, e)
    return sp, F, e


@test
def system_contract_and_em():
    rng = np.random.default_rng(13)
    V, K = 200, 65
    cls = np.concatenate([np.arange(K), rng.integers(0, K, size=V - K)])
    sp, F, e = synthetic_split(rng, 600, V, K, cls)
    PX = rng.dirichlet(np.ones(K), size=600)
    lam_g, lamb, em = systems.fit_system(PX, sp)
    assert em["em_monotone"] and np.allclose(lamb.sum(1), 1)
    pos = np.arange(0, 600, 3)
    r = systems.contract(PX, sp, lamb, pos, F[pos], cls, e)
    assert r["pass"], r
    fixed = np.tile([0.5, 0.3, 0.2], (40, 1))   # weights that give the class members real mass
    assert systems.contract(PX, sp, fixed, pos, F[pos], cls, e)["pass"]
    bad = PX.copy()
    bad[:, 0] += 0.01                         # rows no longer sum to 1: the contract must fail
    assert not systems.contract(bad, sp, fixed, pos, F[pos], cls, e)["pass"]
    assert systems.all_positions(PX, lamb, {"pass": True})["pass"]
    assert not systems.all_positions(bad, lamb, {"pass": True})["pass"]
    assert systems.emission_check(e, cls)["pass"]


# ---------------------------------------------------------------- gates on made-up numbers
@test
def gate_arithmetic():
    F = 7.0400
    assert gates.k0(F, F - 0.0439)["pass"] and gates.k0(F, F - 0.0500)["pass"]
    assert not gates.k0(F, F - 0.0438)["pass"] and not gates.k0(F, F + 0.01)["pass"]
    assert gates.k0(7.0, 6.9561)["pass"], "the 3% boundary must pass despite float representation"
    tw = 6.9000
    assert gates.k0b(tw, tw + 0.0072, tw + 0.5)["pass"] and gates.k0b(tw, tw + 0.5, tw - 0.01)["pass"]
    assert not gates.k0b(tw, tw + 0.0073, tw + 0.0080)["pass"]
    assert gates.k0b(tw, 6.95, 6.91)["best_learner"] == "LightGBM"
    assert gates.aa(tw, tw + 0.0036)["pass"] and gates.aa(tw, tw - 0.0036)["pass"]
    assert not gates.aa(tw, tw + 0.0037)["pass"] and not gates.aa(tw, tw - 0.0040)["pass"]
    assert gates.shuffled(F, F - 0.0010)["pass"] and gates.shuffled(F, F + 0.02)["pass"]
    assert not gates.shuffled(F, F - 0.0011)["pass"]
    ok = {"pass": True}
    no = {"pass": False}
    assert gates.step6_decision(ok, ok, ok, {"twin": ok}).startswith("PASS")
    assert "K0 (sanity)" in gates.step6_decision(no, ok, ok, {"twin": ok})
    assert "K0b" in gates.step6_decision(ok, no, ok, {"twin": ok})
    assert "A/A" in gates.step6_decision(ok, ok, no, {"twin": ok})
    assert "shuffled" in gates.step6_decision(ok, ok, ok, {"twin": ok, "lr": no})
    assert gates.k1(6.92, 6.90)["pass"] and not gates.k1(6.93, 6.90)["pass"]
    lv = gates.level(0.0050, [0.001, 0.0071], -0.010, -0.008, -0.009, -0.0073, -0.0001, [False, True], -0.030)
    assert lv["LEVEL"]
    assert not gates.level(0.0050, [0.001, 0.0072], -0.010, -0.008, -0.009, -0.0073, -0.0001, [True], -0.03)["P1"]
    assert not gates.level(0.0050, [0.001], -0.010, -0.008, -0.009, -0.0073, -0.0001, [False, False], -0.03)["P2"]
    assert not gates.level(0.0050, [0.001], -0.010, -0.008, -0.009, -0.0073, 0.0, [True], -0.03)["P2"]
    assert not gates.level(0.0050, [0.001], -0.010, -0.008, -0.009, -0.0073, -0.001, [True], -0.0291)["P3"]
    assert gates.beats(lv, -0.0146, 6.80, 6.81) and not gates.beats(lv, -0.0145, 6.80, 6.81)
    assert not gates.beats(lv, -0.02, 6.82, 6.81)
    assert gates.adds(-0.0073, -0.0037, -0.0037, -0.0037) and not gates.adds(-0.0073, -0.0036, -0.004, -0.004)
    assert gates.non_vacuous(0.01, 0.01) and not gates.non_vacuous(0.0099, 0.5) and not gates.non_vacuous(0.5, 0.0099)


# ---------------------------------------------------------------- Step 0 inputs
@test
def step0_inputs_match_t0_logs():
    P = data.load_ptb()
    T0, chk = step0.load_t0(P, log=lambda *a: None)
    assert chk["runs"] == 30 and chk["max_mean_err"] <= step0.MEAN_TOL
    rng = np.random.default_rng(14)
    n = 3000
    pF = rng.random(n) * 0.2 + 1e-4
    b = rng.integers(0, 40, size=n)
    Fd = {s: (pF, b) for s in step0.SPLITS}
    bits_t0 = {s: -np.log2(rng.random(n) * 0.1 + 1e-5) for s in step0.SPLITS}
    out, info = step0.mix_with_floor(bits_t0, Fd)
    assert 0 < info["mean_t0_weight"]["valid"] < 1 and set(out) == set(step0.SCORED)


# ---------------------------------------------------------------- writes, test file, frozen record
LISTED_CMD = "for d in " + " ".join(f'"{d}"' for d in paths.LISTED_DIRS) + \
    '; do echo "== $d"; ls -la --time-style=full-iso "$d" 2>&1; done'


def listing():
    return subprocess.run(["bash", "-c", LISTED_CMD], capture_output=True, text=True, check=True).stdout


@test
def no_writes_outside_C():
    before = listing()
    numba_before = set(glob.glob(str(paths.TMP / "numba" / "**" / "*"), recursive=True))
    from tally.floor import cache, exchange, kn, stats  # noqa: F401 (numba cache=True functions)
    Q = tiny_ptb()
    m = kn.KN(Q["ids"]["train"], Q["V"], Q["eos"], order=3)
    m.logprob(Q["ids"]["calib"])
    m.dists(Q["ids"]["calib"], [0, 5])
    m.match(Q["ids"]["calib"])
    c = cache.Cache(Q["V"], 100.0, "block", (Q["eos"],))
    c.prob(Q["ids"]["calib"], Q["blocks"]["calib"], np.full(len(Q["ids"]["calib"]), 1e-3))
    stats.duplicate_mask(Q["ids"]["train"], Q["blocks"]["train"], Q["ids"]["calib"], 8)
    from sklearn.tree import DecisionTreeClassifier
    DecisionTreeClassifier(min_samples_leaf=5, random_state=0).fit(np.random.default_rng(0).integers(0, 2, (300, 6)),
                                                                     np.arange(300) % 3)
    after = listing()
    assert before == after, "a listed folder changed"
    numba_after = set(glob.glob(str(paths.TMP / "numba" / "**" / "*"), recursive=True))
    assert any(p.endswith((".nbi", ".nbc")) for p in numba_after), "numba did not cache into C/tmp/numba"
    print(f"    numba cache files in C/tmp: {len([p for p in numba_after if p.endswith('.nbi')])} "
          f"(new this test: {len(numba_after - numba_before)})")
    # positive control: a change inside a listed-style folder is seen by the same listing command
    scratch = paths.TMP / "tmp" / "listing_control"
    os.makedirs(scratch, exist_ok=True)
    cmd = f'ls -la --time-style=full-iso "{scratch}"'
    a = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    time.sleep(0.01)
    (scratch / "x").write_text("x")
    b = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    (scratch / "x").unlink()
    assert a != b, "the listing did not see a new file"


@test
def no_code_reads_the_test_file():
    hits = []
    for f in hashing.code_files():
        if "/studies/language_part3/C/" not in f or f.endswith("selftest_tm.py"):
            continue
        s = open(f).read()
        if "PTB_TEST_FILE" in s and not f.endswith("paths.py"):
            hits.append(f)
        if "test/text.utf8" in s or '"test", "text.utf8"' in s:
            hits.append(f)
    assert not hits, hits
    from tally import ptb
    assert "test" not in ptb.FILES.values() and "test" not in ptb.FILES


@test
def frozen_record_detects_changes():
    snap = hashing.snapshot()
    assert any(k.endswith("ccore/chains.py") for k in snap) and any(k.endswith("bits.npz") for k in snap)
    assert len([k for k in snap if k.endswith("bits.npz")]) == 30
    rec = {"files": dict(snap)}
    k = next(iter(rec["files"]))
    rec["files"][k] = "0" * 64
    changed = [x for x, v in rec["files"].items() if snap.get(x) != v]
    assert changed == [k]


@test
def two_level_bootstrap():
    from ccore import boot
    from tally.floor import stats
    rng = np.random.default_rng(15)
    n, blocks = 4000, np.repeat(np.arange(40), 100)
    base = rng.standard_normal(n)
    a = base + 0.05 + 0.3 * rng.standard_normal((10, n))       # 10 seeds with seed noise
    b = base + 0.02 * rng.standard_normal(n)
    r = boot.two_level(a, b, blocks)
    ref = stats.block_bootstrap(a.mean(0) - b, blocks)
    assert abs(r["mean"] - ref["mean"]) < 1e-12 and r["lo"] < r["mean"] < r["hi"]
    r1 = boot.two_level(a[:1], b, blocks)
    ref1 = stats.block_bootstrap(a[0] - b, blocks)
    assert abs(r1["mean"] - ref1["mean"]) < 1e-12 and abs(r1["boot_sd"] - ref1["boot_sd"]) < 0.5 * ref1["boot_sd"]
    seed_shift = a + 0.2 * rng.standard_normal((10, 1))       # a per-seed offset: the seed level must widen it
    wide = boot.two_level(seed_shift, b, blocks)
    narrow = stats.block_bootstrap(seed_shift.mean(0) - b, blocks)
    assert wide["boot_sd"] > 1.5 * narrow["boot_sd"], (wide, narrow)


@test
def no_neural_network_on_the_cpu_path():
    import importlib
    for m in ("ccore.chains", "ccore.codes", "ccore.floor", "ccore.layout", "ccore.step0", "ccore.systems",
              "ccore.twin", "ccore.gates", "ccore.boot", "ccore.runlog", "ccore.hashing"):
        importlib.import_module(m)
    import sklearn.tree  # noqa: F401 (the tree twin of step 10)
    assert env.neural_modules_loaded() == [], env.neural_modules_loaded()


@test
def run_once_and_input_hash_guards():
    import shutil

    from ccore import runlog
    fake = paths.TMP / "tmp" / "fake_runs"
    shutil.rmtree(fake, ignore_errors=True)
    os.makedirs(fake)
    saved = paths.RUNS
    try:
        paths.RUNS = fake
        runlog.refuse_second_real_run("twin", False)            # no finished run yet: allowed
        d = fake / "20261021T000000Z_twin"
        os.makedirs(d)
        out = paths.TMP / "tmp" / "fake_output.npz"
        out.write_bytes(b"abc")
        hashing.write_json(d / "run.json", {"state": "finished"})
        hashing.write_json(d / "outputs.json", hashing.manifest([out]))
        try:
            runlog.refuse_second_real_run("twin", False)
            raise AssertionError("a second real run was allowed")
        except SystemExit:
            pass
        runlog.refuse_second_real_run("twin", True)             # dry runs may repeat
        assert runlog.check_inputs(["twin"], False) == {"twin": str(d)}
        out.write_bytes(b"abd")                                 # the output changed after its run
        try:
            runlog.check_inputs(["twin"], False)
            raise AssertionError("a changed input was accepted")
        except SystemExit:
            pass
        os.makedirs(fake / "20261021T000001Z_twin")
        hashing.write_json(fake / "20261021T000001Z_twin" / "run.json", {"state": "finished"})
        try:
            runlog.check_inputs(["twin"], False)
            raise AssertionError("two finished real runs were accepted")
        except SystemExit:
            pass
    finally:
        paths.RUNS = saved
        shutil.rmtree(fake, ignore_errors=True)
        (paths.TMP / "tmp" / "fake_output.npz").unlink(missing_ok=True)


def main():
    t0 = time.time()
    fails = 0
    for fn in RESULTS:
        t = time.time()
        try:
            fn()
            print(f"PASS {fn.__name__} ({time.time() - t:.1f} s)", flush=True)
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__} ({time.time() - t:.1f} s)\n{traceback.format_exc()}", flush=True)
    print(f"selftest_tm: {len(RESULTS) - fails} of {len(RESULTS)} passed in {time.time() - t0:.1f} s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
