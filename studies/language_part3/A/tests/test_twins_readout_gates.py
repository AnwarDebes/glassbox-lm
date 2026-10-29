"""Count twin and tree twin arithmetic, the read-out (oracle, constant scorer, sums), gates on made-up numbers."""

import numpy as np
import pytest

from astruct import gates, readout, twins
from astruct.paths import add_study_paths

add_study_paths(tally=True)
from tally.floor import stats as tstats


def _rows(n_pos=40, k=5, seed=0):
    """A calibrated synthetic floor: the true candidate is drawn from the floor's own distribution inside C."""
    rng = np.random.default_rng(seed)
    off = np.arange(0, n_pos * k + 1, k)
    pos = np.repeat(np.arange(n_pos), k)
    inC = rng.random(n_pos) < 0.8
    pin = rng.dirichlet(np.full(k, 0.3), n_pos)
    true = np.array([rng.choice(k, p=pin[i]) for i in range(n_pos)])
    y = np.zeros(n_pos * k, bool)
    y[off[:-1][inC] + true[inC]] = True
    p = (np.maximum(pin, 1e-9) * 0.7).reshape(-1)  # the candidates hold 70% of the floor's mass
    pt_true = np.where(inC, p.reshape(n_pos, k)[np.arange(n_pos), true], 0.01)
    R = {"pos_off": off, "pos": pos, "n_pos": n_pos, "y": y, "cand_p": p, "cand": np.tile(np.arange(k), n_pos),
         "rank": np.tile(np.arange(k), n_pos), "pb": rng.integers(0, 8, n_pos * k), "type5": rng.integers(0, 5, n_pos * k),
         "due3": rng.integers(0, 3, n_pos * k), "said": rng.random(n_pos * k) < 0.3, "ident": rng.integers(0, 33, n_pos * k),
         "dset": rng.integers(0, 4, n_pos * k)}
    return R, pt_true


def test_count_twin_by_hand():
    R, _ = _rows(n_pos=60, seed=1)
    y = R["y"].astype(float)
    tabs = twins.twin_tables(R, y, ["J2"])
    tw = twins.CountTwin("J2", 3.0, tabs)
    keys = twins.key_columns(R, "J2")
    i = 7
    m2 = (keys == keys[i]).all(1)
    k1 = twins.key_columns(R, "J1")
    m1 = (k1 == k1[i]).all(1)
    kt = twins.key_columns(R, "RBT")
    mt = (kt == kt[i]).all(1)
    kb = twins.key_columns(R, "RB")
    mb = (kb == kb[i]).all(1)
    p0 = y.mean()
    pb = (y[mb].sum() + 3 * p0) / (mb.sum() + 3)
    pt = (y[mt].sum() + 3 * pb) / (mt.sum() + 3)
    p1 = (y[m1].sum() + 3 * pt) / (m1.sum() + 3)
    p2 = (y[m2].sum() + 3 * p1) / (m2.sum() + 3)
    assert abs(tw.predict({k: v[i:i + 1] for k, v in R.items() if isinstance(v, np.ndarray) and len(v) == len(y)})[0] - p2) < 1e-12
    # an unseen J2 cell takes its J1 parent's value
    q = {k: (v[i:i + 1].copy() if isinstance(v, np.ndarray) and len(v) == len(y) else v) for k, v in R.items()}
    q["said"] = ~q["said"]
    k2q = twins.key_columns(q, "J2")
    if not (keys == k2q[0]).all(1).any():
        assert abs(tw.predict(q)[0] - p1) < 1e-12
    assert twins.chain_of("J5") == ["J5", "J3", "J2", "J1", "RBT", "RB"]
    assert twins.chain_of("J4") == ["J4", "J2", "J1", "RBT", "RB"]


def test_tree_twin_path_smoothing():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, (400, 6)).astype(np.uint8)
    y = ((X[:, 0] & X[:, 1]) | (rng.random(400) < 0.1)).astype(int)
    tt = twins.TreeTwin(20, 10.0).fit(X, y)
    tr = tt.tree.tree_
    leaf = tt.tree.apply(X)
    for node in range(tr.node_count):  # counts: every node holds exactly the rows routed through it
        through = tt.tree.decision_path(X)[:, node].toarray().ravel().astype(bool)
        assert tt.n_node[node] == through.sum() and tt.t_node[node] == y[through].sum()
    v = np.empty(tr.node_count)
    for node in range(tr.node_count):
        par = tt.parent[node]
        v[node] = tt.t_node[node] / tt.n_node[node] if par < 0 else (tt.t_node[node] + 10 * v[par]) / (tt.n_node[node] + 10)
    assert np.allclose(tt.predict(X), v[leaf])
    tt2 = twins.TreeTwin(20, 1.0).fit(X, y, tree=tt.tree)  # alpha only changes the read-out
    assert tt2.tree is tt.tree and not np.allclose(tt2.val, tt.val)


def test_readout_oracle_constant_and_guard():
    R, pf_true = _rows(n_pos=3000, seed=2)
    pt = readout.PosTable(R, pf_true)
    floor_bits = -np.log2(pf_true)
    oracle_bits = -np.log2(np.where(pt.in_C, pt.pfC, pt.pf_true))
    q_or = np.where(R["y"], 1.0, readout.Q_FLOOR)
    b, _curve = readout.choose_b(q_or, pt)
    assert b == 0.95  # the cap of the grid
    assert np.allclose(readout.mix_bits(q_or, pt, 1.0), oracle_bits, atol=1e-4)  # b = 1 reproduces the oracle bound
    q_const = np.full(len(R["y"]), 0.3)
    b0, _ = readout.choose_b(q_const, pt)
    assert b0 == 0.0 and np.array_equal(readout.mix_bits(q_const, pt, b0), floor_bits)
    assert np.array_equal(readout.mix_bits(np.zeros(len(R["y"])), pt, 0.5), floor_bits)  # q sums to 0: the floor


def test_contract_passes_and_catches_a_broken_mix():
    R, pf_true = _rows(n_pos=50, k=5, seed=3)
    V = 12
    rng = np.random.default_rng(4)
    full, x = {}, np.zeros(R["n_pos"], np.int64)
    for k in range(R["n_pos"]):  # floor vectors whose first 5 entries are the candidates
        v = rng.random(V) + 0.01
        v[:5] = R["cand_p"][5 * k:5 * k + 5]
        v[5:] = v[5:] / v[5:].sum() * (1 - v[:5].sum())
        full[k] = v
        t = np.flatnonzero(R["y"][5 * k:5 * k + 5])
        x[k] = t[0] if len(t) else 7
    pf_true = np.array([full[k][x[k]] for k in range(R["n_pos"])])
    pt = readout.PosTable(R, pf_true)
    q = rng.random(len(R["y"])) + 1e-6
    fast = readout.mix_bits(q, pt, 0.4)
    good = readout.contract(q, R, full, pt, 0.4, x, fast)
    assert good["pass"] and good["max_sum_err"] < 1e-12
    bad = readout.contract(q, R, full, pt, 0.4, x, fast + 1e-6)  # a fast path that disagrees with the vector
    assert not bad["pass"]


def test_gate_arithmetic_on_made_up_numbers():
    assert gates.k1(9.9e-4)["pass"] and not gates.k1(-1.2e-3)["pass"]
    g = gates.k3(d_trees=-0.050, d_lr2=-0.010, d_lr1=-0.030)
    assert g["i_pass"] and g["ii_pass"] and g["pass"]
    g = gates.k3(d_trees=-0.040, d_lr2=-0.043, d_lr1=-0.035)
    assert not g["i_pass"] and not g["pass"]  # best of trees and LR2 is -0.043 > -0.0439
    g = gates.k3(d_trees=-0.060, d_lr2=-0.010, d_lr1=-0.0231)  # G1's LR margin: stops at (ii)
    assert g["i_pass"] and not g["ii_pass"] and not g["pass"]
    assert gates.k3c(d_pilot=-0.0081 / 0.0640 * 0.06, d_trees=-0.06)["pass"] is False  # G1's 12.7% share
    assert gates.k3c(d_pilot=-0.021, d_trees=-0.06)["pass"]
    # amendment 1: K3 (i) passed through LR2 alone while the trees mix is worse than B; a pilot worse than B used to
    # pass ("a third of a negative gain"); now it stops, and a pilot better than B still passes
    g = gates.k3(d_trees=+0.03, d_lr2=-0.05, d_lr1=-0.03)
    assert g["pass"]
    edge = gates.k3c(d_pilot=+0.010, d_trees=+0.030)
    assert edge["third_of_trees_gain"] and not edge["pilot_beats_B"] and not edge["pass"]
    assert gates.k3c(d_pilot=-0.001, d_trees=+0.030)["pass"]
    assert not gates.k3c(d_pilot=0.0, d_trees=+0.030)["pass"]  # level with B is not better than B
    assert gates.k8({"mean": 0.002, "lo": -0.001, "hi": 0.004})["pass"]
    assert not gates.k8({"mean": 0.004, "lo": 0.001, "hi": 0.006})["pass"]
    assert not gates.k8({"mean": 0.001, "lo": 0.0005, "hi": 0.002})["pass"]  # interval misses 0
    assert gates.k9(0.0, -0.5)["pass"] and gates.k9(0.3, -0.0009)["pass"] and not gates.k9(0.3, -0.0011)["pass"]
    assert gates.k4b(-0.0146)["pass"] and not gates.k4b(-0.014)["pass"]
    boot = lambda m, hi: {"mean": m, "lo": m - 0.01, "hi": hi}
    ok = gates.pass_lines(boot(-0.03, -0.01), [-0.02] * 10, boot(-0.02, -0.001), boot(-0.016, -0.002), boot(-0.01, -0.001),
                          {"uniform": boot(-0.02, -0.008), "anchored": boot(-0.03, -0.0075)}, {"uniform": 0.0, "anchored": 0.2})
    assert ok["pass"]
    vac = gates.pass_lines(boot(-0.03, -0.01), [-0.02] * 10, boot(-0.02, -0.001), boot(-0.016, -0.002), boot(-0.01, -0.001),
                           {"uniform": boot(-0.02, -0.008), "anchored": boot(-0.03, -0.0075)}, {"uniform": 0.0, "anchored": 0.0})
    assert not vac["d"] and not vac["pass"]  # both banks vacuous: no learning credit
    one_seed = gates.pass_lines(boot(-0.03, -0.01), [-0.02] * 9 + [0.001], boot(-0.02, -0.001), boot(-0.016, -0.002),
                                boot(-0.01, -0.001), {"u": boot(-0.02, -0.008)}, {"u": 0.1})
    assert not one_seed["a"]  # every seed must be below B


def test_bootstrap_equals_tally_block_bootstrap():
    rng = np.random.default_rng(5)
    d = rng.normal(-0.01, 0.2, 3000)
    g = np.sort(rng.integers(0, 120, 3000))
    mine = gates.paired_bootstrap(d, g, n_boot=2000, seed=0)
    ref = tstats.block_bootstrap(d, g, n_boot=2000, seed=0)
    assert abs(mine["lo"] - ref["lo"]) < 1e-12 and abs(mine["hi"] - ref["hi"]) < 1e-12
    w = gates.paired_bootstrap(d, g, level=0.975)
    assert w["lo"] <= mine["lo"] + 0.01 and w["level"] == 0.975


@pytest.mark.parametrize("seed", [0, 1])
def test_isotonic_q_bounded_and_monotone(seed):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=2000)
    y = rng.random(2000) < 1 / (1 + np.exp(-3 * s))
    iso = readout.Isotonic().fit(s, y)
    q = iso(np.concatenate([s, [-1e9, 1e9]]))
    assert q.min() >= readout.Q_FLOOR and q.max() <= 1.0
    order = np.argsort(s)
    assert (np.diff(iso(s[order])) >= -1e-15).all()


@pytest.mark.parametrize("level", twins.TWIN_KEYS)
@pytest.mark.parametrize("w", twins.TWIN_W)
def test_count_twin_matches_cellwise_reference(level, w):
    R, _ = _rows(n_pos=400, k=6, seed=8)
    Q, _ = _rows(n_pos=150, k=6, seed=9)
    Q["dset"] = Q["dset"] + np.where(np.arange(len(Q["dset"])) % 5 == 0, 100, 0)  # some sets never counted
    y = R["y"].astype(float)
    tw = twins.CountTwin(level, w, twins.twin_tables(R, y, [level]))
    chain = twins.chain_of(level)
    stats = {}
    for k in chain:
        kc = twins.key_columns(R, k)
        d = {}
        for row, yy in zip(map(tuple, kc), y):
            n, t = d.get(row, (0.0, 0.0))
            d[row] = (n + 1, t + yy)
        stats[k] = d

    def value(k, cols_row):
        """cell-wise: (t + w P(parent)) / (n + w); an unseen cell is its parent's value."""
        par = twins.PARENT[k]
        if par is None:
            prior = y.mean()
        else:
            mine, pc = twins.COLS[k], twins.COLS[par]
            prior = value(par, tuple(cols_row[mine.index(f)] for f in pc))
        n, t = stats[k].get(tuple(cols_row), (0.0, 0.0))
        return (t + w * prior) / (n + w)
    kq = twins.key_columns(Q, level)
    ref = np.array([value(level, tuple(r)) for r in kq])
    assert np.allclose(tw.predict(Q), ref, rtol=0, atol=1e-12)
