"""Tests of experiments/ssl_r1_controls.py (the exploratory R1 control study: young ladder, literal-frozen twins,
Q_pred). CPU engine and oracle only, tiny corpora; the one CUDA test skips without a device and is run under
scripts/gpu_run.sh shared.

(a) twin freezing: with p_ins = 0, p_neg = 0, s = 2^62 every clause alive at two checkpoints with the same birth
    step has identical literal slots, all included, states never lower; seeding, recycling and votes still move;
    the default arm under the same tiny config does change literals (the check is not vacuous); oracle == CPU engine.
(b) the ladder checkpoints do not change the trajectory (state hashes equal R1's train_arm and a plain loop).
(c) features of arbitrary checkpoint names round-trip through the AnyStore subclass.
(d) Q_pred on a tiny synthetic case equals a brute-force count and a brute-force T1 score.
(e) config: twin thresholds, ladder rounding, shared params, the reading rule rows."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _q in (ROOT / "experiments", ROOT / "src"):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import ssl_r1_controls as SC  # noqa: E402
import ssl_r1_pretext as R  # noqa: E402
from tmlm.data.synthetic import toy_corpus  # noqa: E402
from tmlm.engine.api import CorpusArrays  # noqa: E402
from tmlm.engine.checkpoint import load_checkpoint  # noqa: E402
from tmlm.engine.config import INCLUDE_THRESHOLD, LIT_EMPTY  # noqa: E402
from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: E402
from tmlm.engine.rng import inv_q32  # noqa: E402
from tmlm.ssl import controls as K  # noqa: E402
from tmlm.ssl import features as F  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402

CJ = SC.load_config(ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_controls.json")
TINY = dict(vocab=40, n_clauses=64, batch=16, sweep_period=8, n_bag_k=2, dead_sweeps=2)


def _params(**over) -> dict:
    p = copy.deepcopy(CJ["params"])
    p["young_ladder"] = {**p["young_ladder"], "abs_batches": [2, 3], "percent_of_epoch1": [25, 50]}
    p["young_fraction"] = 0.3
    p.update(over)
    return p


def _train(arm: str, corpus, tmp: Path, epochs: int = 3, p: dict | None = None, seed: int = 1):
    p = p or _params()
    base, twin = SC.CONTROL_ARMS[arm]
    ad = P.arm_data(base, corpus, vocab=40)
    cfg = SC.arm_config(arm, seed, p, **TINY)
    rec = R.Rec(smoke=True)
    tmp.mkdir(parents=True, exist_ok=True)
    h = SC.train_ladder(ad, cfg, epochs, tmp, "cpu", rec, p, twin, yield_gpu=False)
    rec.m("control.arm", arm)
    rec.m("control.seed", seed)
    rec.m("final.state_hash", h["epochs"][epochs])
    (tmp / "metrics.json").write_text(json.dumps(rec.metrics, default=str))  # what run_logged would hold
    return ad, cfg, rec, h


def _states(run: Path) -> dict[str, dict]:
    return {n: load_checkpoint(run / f"checkpoint_{n}.npz")[1] for n in SC.checkpoint_names(run)}


# ------------------------------------------------------------------ (a) twin freezing
def test_twin_config_zeroes_every_literal_threshold_and_keeps_the_rest():
    p = CJ["params"]
    for arm, base in (("A0T", "A0"), ("A3T", "A3")):
        cfg = SC.arm_config(arm, 100, p)
        ref = P.engine_config(base, 100)
        t = SC.twin_thresholds(cfg)
        assert (t["p_ins_q32"], t["p_neg_q32"], t["inv_s_q32"], t["s_num"], t["s_den"]) == (0, 0, 0, 1 << 62, 1)
        assert SC.twin_thresholds_ok(cfg, ref)
        # only p_ins, p_neg and s differ from the R1 arm
        diff = {k for k, v in cfg.to_dict().items() if ref.to_dict()[k] != v}
        assert diff == {"p_ins", "p_neg", "s"}
    for arm in ("A0", "A3"):
        assert SC.arm_config(arm, 101, p) == P.engine_config(arm, 101)
    # s = 2^30 would NOT stop the Type Ia decrements (and settle's n_q16 is 0 only below owed = s)
    assert inv_q32(2 ** 30) == 4 and inv_q32(2 ** 62) == 0


def test_twin_freezes_literals_while_seeding_recycling_and_votes_move(tmp_path):
    c = toy_corpus(vocab=40, n_docs=4, doc_len=80, seed=5)
    _, cfg, rec, h = _train("A0T", c, tmp_path / "twin")
    assert rec.failed == [], rec.failed
    t = h["totals"]
    assert t.get("n_inserted", 0) == 0 and t.get("n_neg_inserted", 0) == 0
    assert t["n_seeded"] > 0 and t["n_recycled"] > 0 and t["n_tgt_gate_pass"] > 0
    assert h["sum_G"] < SC.twin_thresholds(cfg)["s_num"]
    st = _states(tmp_path / "twin")
    names = list(st)
    assert names[0].startswith("young_b") and names[-1] == "epoch3" and "c0y" in names
    votes_moved = states_rose = False
    n_common = 0
    for i, a in enumerate(names):
        A = st[a]
        alive = A["alive"].astype(bool)
        tracked = (A["lit_id"] != LIT_EMPTY) & alive[:, None]
        assert not np.any(tracked & (A["lit_state"] < INCLUDE_THRESHOLD)), a        # every literal included
        assert not np.any(tracked & ((A["lit_id"] & 1) == 1)), a                     # no negated literal
        assert not np.any(tracked & ((A["lit_id"] >> 17) >= 16)), a                  # no bag literal
        for b in names[i + 1:]:
            B = st[b]
            both = alive & B["alive"].astype(bool) & (A["birth_step"] == B["birth_step"])
            assert np.array_equal(A["lit_id"][both], B["lit_id"][both]), (a, b)   # identical literal slots
            assert np.all(B["lit_state"][both] >= A["lit_state"][both]), (a, b)   # states only rise (Type Ia)
            votes_moved |= not np.array_equal(A["vote_w"][both], B["vote_w"][both])
            states_rose |= bool(np.any(B["lit_state"][both] > A["lit_state"][both]))
            n_common += int(both.sum())
    assert votes_moved and states_rose and n_common > 0   # the checks compare real incarnations
    # the same checks see literal learning in the default arm (not vacuous)
    _, _, rec0, h0 = _train("A0", c, tmp_path / "bank")
    assert h0["totals"]["n_inserted"] > 0
    lit = [v["value"] for k, v in rec0.metrics.items() if k.startswith("literals.")]
    assert any(not SC.twin_frozen(s) for s in lit)


def test_twin_config_oracle_equals_cpu_engine(tmp_path):
    from tmlm.engine.oracle.oracle import OracleEngine
    c = toy_corpus(vocab=40, n_docs=3, doc_len=60, seed=9)
    cfg = SC.arm_config("A0T", 3, _params(), **TINY)
    a, b = OracleEngine(cfg, corpus=c), CpuEngine(cfg, corpus=c)
    for e in range(2):
        for k in range(a.n_batches()):
            da, db = a.train_batch(e, k), b.train_batch(e, k)
            assert da == db
    assert a.state_hash() == b.state_hash()
    assert a.check_invariants() == []


# ------------------------------------------------------------------ (b) the ladder does not change the trajectory
@pytest.mark.parametrize("arm", ["A0", "A3", "A3T"])
def test_ladder_checkpoints_do_not_change_the_state_hashes(tmp_path, monkeypatch, arm):
    monkeypatch.setattr(P, "F2_M", 5)  # tiny cap: the F2 example sets differ per epoch (engine rebuilt per epoch)
    c = toy_corpus(vocab=40, n_docs=3, doc_len=60, seed=11)
    ad, cfg, rec, h = _train(arm, c, tmp_path / "ladder")
    if P.ARMS[SC.CONTROL_ARMS[arm][0]].f2:
        assert not np.array_equal(ad.targets(0), ad.targets(1))
    # R1's own loop (C0y checkpoint only) gives the same c0y and epoch hashes
    ref = R.train_arm(ad, cfg, 3, tmp_path / "r1", "cpu", R.Rec(smoke=True), young_fraction=0.3, yield_gpu=False)
    assert ref["c0y"] == h["c0y"] and ref["epochs"] == h["epochs"]
    # and so does a loop without any checkpoint
    eng = None
    for e in range(3):
        if eng is None or ad.targets_vary_by_epoch:
            st = None if eng is None else eng.state()
            eng = CpuEngine(cfg, corpus=ad.corpus(e))
            if st is not None:
                eng.load_state(st)
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
    assert eng.state_hash() == h["epochs"][3]
    # every ladder rung is the state after exactly its batch count of epoch 1
    nb = (ad.targets(0).shape[0] + cfg.batch - 1) // cfg.batch
    pts = SC.ladder_points(nb, _params())
    assert set(h["young"]) == set(pts.values()) and len(pts) >= 2
    for b, name in pts.items():
        _, st, meta = load_checkpoint(tmp_path / "ladder" / f"checkpoint_{name}.npz")
        assert int(st["step"][0]) == b and meta["extra"]["batches"] == b
        assert meta["extra"]["examples"] == min(ad.targets(0).shape[0], b * cfg.batch)
        assert rec.metrics[f"young.{name}.state_hash"]["value"] == meta["state_hash"]
    assert rec.metrics["c0y.batches"]["value"] == K.c0y_batches(ad.targets(0).shape[0], cfg.batch, 0.3)


# ------------------------------------------------------------------ (c) features of any checkpoint name round-trip
def _tokens_npz(d: Path, docs: F.TokenDocs, n_dev: int, arm: str = "A3T", seed: int = 101) -> tuple[dict, dict]:
    n = docs.n_doc
    layout = {"agnews": {"dev": [0, n_dev], "lab": [n_dev, n], "lab_pool_rows": list(range(n - n_dev))}}
    labels = {"agnews": {"pool_y": [i % 2 for i in range(n - n_dev)], "dev_y": [i % 2 for i in range(n_dev)]}}
    SC.save_tokens(d, docs, layout, labels, arm, seed)
    return layout, labels


def test_any_checkpoint_name_round_trips_through_the_store(tmp_path):
    rng = np.random.default_rng(0)
    docs = F.TokenDocs(tok=rng.integers(0, 40, 90).astype(np.uint16), doc_start=np.arange(0, 91, 10, dtype=np.uint32), end_id=40)
    _tokens_npz(tmp_path, docs, 3)
    sds = {}
    for name in ("young_b77", "c0y", "epoch4", "young_b1953"):
        ev_d = rng.integers(0, docs.n_doc, 200)
        ev_j = rng.integers(0, 50, 200)
        sd = F.aggregate_doc_counts(ev_d, ev_j, docs.n_doc, 64)
        R.save_sparse(tmp_path / f"features_{name}.npz", sd, state_hash="x")
        np.save(tmp_path / f"dim_{name}.npy", np.array([1, 3, 5, 7], np.int64))
        sds[name] = sd
    st = SC.AnyStore(tmp_path, None, CJ["params"])
    assert (st.arm, st.seed) == ("A3T", 101)
    assert st.feature_names() == sorted(sds)
    assert SC.parse_sets("auto", st) == [f"{n}.{c}" for n in sorted(sds) for c in ("all", "dim")]
    with pytest.raises(SystemExit):
        SC.parse_sets("young_b5.all", st)
    with pytest.raises(SystemExit):
        SC.parse_sets("epoch4.bank", st)
    lab = st.rows("agnews", np.array([0, 2, 5]))
    dev = st.dev_rows("agnews")
    for name, sd in sds.items():
        got = st.sparse(name)
        assert np.array_equal(got.indptr, sd.indptr) and np.array_equal(got.ids, sd.ids) and np.array_equal(got.cnt, sd.cnt)
        Xl, Xd, dense = st.lr_matrices(f"{name}.dim", lab, dev)
        assert not dense and Xl.shape == (3, 4) and Xd.shape == (3, 4)
        Xl, _, _ = st.lr_matrices(f"{name}.all", lab, dev)
        assert Xl.shape == (3, 64)
        assert np.allclose(Xl.toarray(), sd.rows(lab).to_csr("log1p").toarray())
    assert st.file_sha256("c0y") == st.file_sha256("c0y")


def test_featurize_named_checkpoints_of_a_train_run(tmp_path):
    c = toy_corpus(vocab=40, n_docs=4, doc_len=60, seed=2)
    nb = (int(c.tok.shape[0]) + 15) // 16                       # batches of epoch 1 (A0: every position)
    b_c0y = K.c0y_batches(int(c.tok.shape[0]), 16, 0.3)
    p = _params(forward_chunk=64, dim_matched=16)
    p["young_ladder"] = {**p["young_ladder"], "abs_batches": [b_c0y], "percent_of_epoch1": [20]}
    b_pct = max(1, (nb * 20 + 50) // 100)
    assert b_pct < b_c0y
    run = tmp_path / "train"
    ad, cfg, rec, h = _train("A0", c, run, epochs=2, p=p)
    assert rec.metrics["c0y.batches"]["value"] == b_c0y and h["young"][f"young_b{b_c0y}"] == h["c0y"]  # same batch, same state
    names = SC.resolve_names("c0y,ladder,epoch2", run)
    assert names == ["c0y", f"young_b{b_pct}", f"young_b{b_c0y}", "epoch2"]
    assert SC.resolve_names("all", run) == [f"young_b{b_pct}", f"young_b{b_c0y}", "c0y", "epoch1", "epoch2"]
    docs_c = toy_corpus(vocab=40, n_docs=5, doc_len=12, seed=3)
    docs = F.TokenDocs(tok=np.asarray(docs_c.tok), doc_start=np.asarray(docs_c.doc_start), end_id=40)
    out = tmp_path / "feat"
    out.mkdir()
    _tokens_npz(out, docs, 2, arm="A0", seed=1)
    frec = R.Rec(smoke=True)
    done = SC.featurize_checkpoints(names, run, "A0", 1, docs, F.corpus_arrays(docs, c), c, int(c.tok.shape[0]), p, "cpu",
                                    out, frec, **TINY)
    assert frec.failed == [], frec.failed
    assert frec.metrics[f"features.young_b{b_c0y}.duplicate_of"]["value"] == "c0y"
    store = SC.AnyStore(out, None, p)
    assert store.feature_names() == sorted(names)
    for name in names:
        _, st, meta = load_checkpoint(run / f"checkpoint_{name}.npz")
        assert done[name] == meta["state_hash"]
        eng = CpuEngine(cfg, corpus=F.corpus_arrays(docs, c))
        eng.load_state(st)
        ref = R.bank_codes(eng, cfg, docs, 64, R.Rec(smoke=True), name)
        got = store.sparse(name)
        assert np.array_equal(got.ids, ref.ids) and np.array_equal(got.cnt, ref.cnt) and np.array_equal(got.indptr, ref.indptr)
        e2 = CpuEngine(cfg, corpus=c)
        e2.load_state(st)
        fc = R.fire_counts(e2, cfg, int(c.tok.shape[0]), 64)
        assert np.array_equal(store.dim_cols(name), K.top_columns_by(fc.astype(np.int64), 16))
    # a checkpoint whose config is not the arm's is refused
    with pytest.raises(SystemExit):
        SC.load_named_checkpoint(run, "epoch2", "A0T", 1, p, R.Rec(smoke=True), **TINY)


# ------------------------------------------------------------------ (d) Q_pred against brute force
def _split_corpus(c: CorpusArrays, n_train_docs: int) -> tuple[CorpusArrays, CorpusArrays]:
    ds = np.asarray(c.doc_start, np.int64)
    cut = int(ds[n_train_docs])

    def part(lo, hi, d0, d1):
        return CorpusArrays(tok=np.ascontiguousarray(c.tok[lo:hi]), doc_start=np.ascontiguousarray((ds[d0:d1 + 1] - lo).astype(np.uint32)),
                            kind_map=c.kind_map, stop=c.stop, sym_freq=c.sym_freq, targets=np.arange(hi - lo, dtype=np.uint32))
    return part(0, cut, 0, n_train_docs), part(cut, int(ds[-1]), n_train_docs, ds.shape[0] - 1)


def _qpred_case(tmp_path: Path, engine_kind: str):
    """A tiny bank, a tiny 'prefix' (fit = documents 0 and 1) and 'validation' split; qpred_checkpoint on it."""
    c = toy_corpus(vocab=40, n_docs=12, doc_len=70, seed=21)
    prefix, val = _split_corpus(c, 9)
    p = _params()
    cfg = SC.arm_config("A0", 1, p, **TINY)
    eng = CpuEngine(cfg, corpus=prefix)
    for e in range(3):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
    st = eng.state()
    targets = np.arange(1, val.tok.shape[0], 4, dtype=np.int64)
    hidden = np.zeros(val.tok.shape[0], np.uint8)
    hidden[targets] = 1
    ds = np.asarray(prefix.doc_start, np.int64)
    fit_pos = np.arange(ds[0], ds[2], dtype=np.int64)
    cnt_pos = np.arange(ds[2], ds[-1], dtype=np.int64)
    qp = copy.deepcopy(p["qpred"])
    qp["checks"] = {"cross_engine_count_rows": 20, "cross_engine_fit_rows": 10, "validation_api_rows": 20,
                    "validation_bruteforce_rows": 3, "signed_predict_scores_rows": 30}
    rec = R.Rec(smoke=True)
    row = SC.qpred_checkpoint("epoch3", cfg, st, prefix, val, targets, hidden, fit_pos, cnt_pos, qp, engine_kind, 64, rec, tmp_path)
    return eng, cfg, st, prefix, val, targets, hidden, cnt_pos, rec, row


def _brute_force_check(tmp_path, eng, cfg, st, prefix, val, targets, hidden, cnt_pos, row) -> None:
    V, C = 40, 64
    tok = np.asarray(prefix.tok).astype(np.int64)
    zero = np.zeros(prefix.tok.shape[0], np.uint8)
    # brute-force O_jc through the public API (a different call path than the pipeline)
    O = np.zeros((C, V), np.int64)
    for q in cnt_pos.tolist():
        for j in eng.fired_clauses(0, q, zero).tolist():
            O[j, tok[q]] += 1
    counts, n_ev, _ = SC.count_table(eng, "cpu", cnt_pos, tok[cnt_pos], C, V, 64, zero)
    assert np.array_equal(counts, O) and n_ev == int(O.sum()) and O.sum() > 0
    # brute-force T1 score with the fitted alpha, beta0 (docs/11 11.5)
    rd = json.loads((tmp_path / "qpred_epoch3_readout.json").read_text())
    alpha = np.zeros(13 * 24)
    for g, a in rd["alpha_by_group"].items():
        alpha[int(g)] = a
    beta0 = rd["beta0"]
    N, Dn = O.sum(axis=1), (O > 0).sum(axis=1)
    uni = np.bincount(tok[cnt_pos], minlength=V).astype(np.float64)
    pu = (uni + 1.0) / (uni.sum() + V)
    alive = st["alive"].astype(bool)
    n_inc = (((st["lit_state"] >= 128) & (st["lit_id"] != LIT_EMPTY)) & alive[:, None]).sum(axis=1)
    grp = np.minimum(n_inc, 12) * 24 + np.minimum(np.floor(np.log2(np.maximum(N, 1))).astype(np.int64), 23)
    ev = CpuEngine(cfg, corpus=val)
    ev.load_state(st)
    gold = np.asarray(val.tok)[targets].astype(np.int64)
    hits, hits5, voted = 0, 0, 0
    z = np.load(tmp_path / "qpred_epoch3.npz")
    for r, t in enumerate(targets.tolist()):
        s = beta0 * np.log(pu)
        for j in ev.fired_clauses(0, t, hidden).tolist():
            if N[j] == 0:
                continue
            voted += 1
            lam = N[j] / (N[j] + Dn[j])
            for cc in np.flatnonzero(O[j]).tolist():
                s[cc] += alpha[grp[j]] * math.log(1.0 + lam * O[j, cc] / (N[j] * (1.0 - lam) * pu[cc]))
        pr = np.exp(s - s.max())
        pr /= pr.sum()
        g = gold[r]
        rank = int((pr > pr[g]).sum() + ((pr == pr[g]) & (np.arange(V) < g)).sum())
        hits += rank == 0
        hits5 += rank < 5
        assert abs(pr[g] - z["loglin_gold_prob"][r]) < 1e-12
        assert rank == int(z["loglin_rank"][r])
    assert voted > 0 and np.any(alpha != 0)
    assert row["loglin_top1"] == hits / targets.shape[0] and row["loglin_top5"] == hits5 / targets.shape[0]
    # signed row = argmax of the engine's own class sums (ties toward the lowest id)
    ps = ev.predict_scores(0, targets.astype(np.uint32), hidden)
    assert row["signed_top1"] == float((ps.argmax(axis=1) == gold).mean())
    assert 0.0 < row["loglin_top1"] <= 1.0


def test_qpred_equals_brute_force_count_and_score(tmp_path):
    eng, cfg, st, prefix, val, targets, hidden, cnt_pos, rec, row = _qpred_case(tmp_path, "cpu")
    assert rec.failed == [], rec.failed
    _brute_force_check(tmp_path, eng, cfg, st, prefix, val, targets, hidden, cnt_pos, row)


def test_qpred_cuda_path_equals_brute_force(tmp_path):
    """The production path: CUDA counting and fit rows, the H8.1 GPU fit, the GPU validation cross-check."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")
    eng, cfg, st, prefix, val, targets, hidden, cnt_pos, rec, row = _qpred_case(tmp_path, "cuda")
    assert rec.failed == [], rec.failed
    for k in ("cuda_fired_lists_equal_cpu_engine", "fit_objective_gpu_equals_cpu", "validation_gpu_equals_cpu"):
        assert rec.metrics[f"check.qpred.epoch3.{k}"]["value"] is True
    _brute_force_check(tmp_path, eng, cfg, st, prefix, val, targets, hidden, cnt_pos, row)


def test_qpred_split_uses_the_fit_document_rule():
    c = toy_corpus(vocab=40, n_docs=250, doc_len=3, seed=1)
    fit_doc, fit_pos, cnt_pos = SC.qpred_split(c)
    from tmlm.engine.rng import mix64
    exp = np.array([mix64(i ^ 0xC0FFEE) % 100 == 0 for i in range(250)])
    assert np.array_equal(fit_doc, exp) and exp.any()
    assert fit_pos.shape[0] == 3 * int(exp.sum()) and fit_pos.shape[0] + cnt_pos.shape[0] == 750
    assert not set(fit_pos.tolist()) & set(cnt_pos.tolist())


def test_count_table_cuda_equals_cpu():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    c = toy_corpus(vocab=40, n_docs=6, doc_len=80, seed=4)
    cfg = SC.arm_config("A0", 2, _params(), **TINY)
    eng = CpuEngine(cfg, corpus=c)
    for e in range(2):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
    st = eng.state()
    g = CudaEngine(cfg, corpus=c)
    g.load_state(st)
    pos = np.arange(0, c.tok.shape[0], dtype=np.int64)
    tok = np.asarray(c.tok).astype(np.int64)
    zero = np.zeros(c.tok.shape[0], np.uint8)
    chk = np.array([0, 5, 100, 479])
    a, na, ga = SC.count_table(eng, "cpu", pos, tok, 64, 40, 64, zero, chk)
    b, nb_, gb = SC.count_table(g, "cuda", pos, tok, 64, 40, 64, zero, chk)
    assert np.array_equal(a, b) and na == nb_
    assert all(np.array_equal(ga[r], gb[r]) for r in chk.tolist())
    p1, i1 = SC.fit_rows_fired(eng, "cpu", pos[:50], zero, 64)
    p2, i2 = SC.fit_rows_fired(g, "cuda", pos[:50], zero, 16)
    assert np.array_equal(p1, p2) and np.array_equal(i1, i2)


# ------------------------------------------------------------------ (e) config and reading rule
def test_ladder_rounding_matches_the_config_note():
    p = CJ["params"]
    assert SC.ladder_points(7639, p) == {77: "young_b77", 196: "young_b196", 255: "young_b255", 382: "young_b382", 764: "young_b764"}
    assert SC.ladder_points(19531, p) == {77: "young_b77", 196: "young_b196", 255: "young_b255", 977: "young_b977", 1953: "young_b1953"}
    assert (7822117 + 1023) // 1024 == 7639 and (19998748 + 1023) // 1024 == 19531       # logged epoch-1 targets
    assert K.c0y_batches(7822117, 1024) == 77 and K.c0y_batches(19998748, 1024) == 196
    assert SC.ladder_points(100, p) == {5: "young_b5", 10: "young_b10", 77: "young_b77"}  # rungs beyond nb skipped
    assert SC.ladder_points(10, {"young_ladder": {"abs_batches": [], "percent_of_epoch1": [5]}}) == {1: "young_b1"}


def test_config_shares_r1_params_and_states_the_rule():
    SC.check_shared_params(CJ)
    rr = CJ["reading_rule"]
    assert rr["codes"] == ["all", "dim"] and rr["seeds"] == [100, 101] and len(rr["rows"]) == 4
    assert CJ["params"]["seeds"] == [100, 101] and CJ["params"]["probe_budgets"] == [8, 32, 128]
    assert set(CJ["params"]["probe_budgets"]) <= set(CJ["params"]["budgets"])
    h81 = json.loads((ROOT / "experiments" / "cvtm" / "configs" / "H8_1_masked.json").read_text())["fit"]["loglin"]
    assert CJ["params"]["qpred"]["fit"] == h81
    for arm, (base, twin) in SC.CONTROL_ARMS.items():
        assert base in P.ARMS and twin == arm.endswith("T")
    for k in ("A3", "A0"):
        assert (ROOT / CJ["replay_reference"][k]["100"] / "metrics.json").exists()


def _comp(dl, dt, qb=0.5, qt=0.4):
    return {"d_ladder": dl, "d_twin": dt, "q_pred_bank": qb, "q_pred_twin": qt}


@pytest.mark.parametrize("dl,dt,qb,qt,expect", [
    (1.5, 1.2, 0.5, 0.4, "1 LITERAL LEARNING CREDITED"),
    (1.5, 1.2, 0.3, 0.4, "4 MIXED"),                     # Q_pred of the bank below the twin: not credited
    (1.5, 0.4, 0.5, 0.4, "2 GAIN FROM CONTINUED SEEDING, SELECTION OR VOTES, NOT FROM LOGIC"),
    (1.5, -0.9, 0.5, 0.4, "2 GAIN FROM CONTINUED SEEDING, SELECTION OR VOTES, NOT FROM LOGIC"),
    (0.5, -1.0, 0.5, 0.4, "4 MIXED"),
    (-0.1, 0.2, 0.5, 0.4, "3 TRAINING DOES NOT ADD VALUE"),
    (0.5, 0.5, 0.5, 0.4, "4 MIXED"),
    (None, 0.5, 0.5, 0.4, "INCOMPLETE"),
])
def test_reading_rule_rows(dl, dt, qb, qt, expect):
    comp = {(s, c): _comp(dl, dt, qb, qt) for s in (100, 101) for c in ("all", "dim")}
    assert SC.reading(comp, [100, 101]) == expect
    if expect.startswith("1"):  # one cell below the threshold on one seed breaks the credit
        comp[(101, "dim")] = _comp(0.9, dt, qb, qt)
        assert SC.reading(comp, [100, 101]) != expect


def test_report_components_from_probe_rows():
    rows = []
    for arm, q0 in (("A3", 0.60), ("A3T", 0.58)):
        for name, dq in (("epoch4", 0.0), ("c0y", -0.03), ("young_b382", -0.005)):
            if arm == "A3T" and name != "epoch4":
                continue
            for d in range(5):
                for task in ("agnews", "sst2", "trec"):
                    rows.append({"arm": arm, "seed": 100, "set": f"{name}.all", "clf": "lr", "task": task, "budget": 32,
                                 "draw": d, "acc": q0 + dq + 0.001 * d})
    q = R.summarise(rows, ["agnews", "sst2", "trec"])
    comp = SC.components(q, {("A3", 100, "epoch4"): 0.2, ("A3T", 100, "epoch4"): 0.19}, "A3", 100, "all")
    assert comp["max_rung"] == "young_b382.all"
    assert abs(comp["d_ladder"] - 0.5) < 1e-9 and abs(comp["d_twin"] - 2.0) < 1e-9
    assert comp["q_pred_bank"] == 0.2 and comp["q_pred_twin"] == 0.19


def test_report_stage_end_to_end_through_main(tmp_path, monkeypatch):
    """--stage report over synthetic probe cells and qpred metrics: the reading of the config, per arm pair."""
    monkeypatch.setenv("SSL_SCRATCH", str(tmp_path / "scratch"))
    probe_runs, qpred_runs = [], []
    acc = {("A3", "epoch4"): 0.62, ("A3", "c0y"): 0.60, ("A3", "young_b382"): 0.605, ("A3T", "epoch4"): 0.615,
           ("A0", "epoch4"): 0.55, ("A0", "c0y"): 0.57, ("A0T", "epoch4"): 0.55}
    for s in (100, 101):
        for arm in ("A3", "A0", "A3T", "A0T"):
            d = tmp_path / f"probe_{arm}_{s}"
            d.mkdir()
            with open(d / "cells.jsonl", "w") as fh:
                for (a, name), q0 in acc.items():
                    if a != arm:
                        continue
                    for code in ("all", "dim"):
                        for dr in range(5):
                            for task in ("agnews", "sst2", "trec"):
                                fh.write(json.dumps({"arm": arm, "seed": s, "set": f"{name}.{code}", "clf": "lr", "task": task,
                                                     "budget": 32, "draw": dr, "acc": q0 + 0.001 * dr}) + "\n")
            probe_runs.append(str(d))
            q = tmp_path / f"qpred_{arm}_{s}"
            q.mkdir()
            (q / "metrics.json").write_text(json.dumps({"control.arm": {"value": arm, "label": "measured"},
                                                        "control.seed": {"value": s, "label": "measured"},
                                                        "qpred.epoch4.loglin_top1": {"value": 0.3 if not arm.endswith("T") else 0.29, "label": "measured"}}))
            qpred_runs.append(str(q))
    rc = SC.main(["--stage", "report", "--probe-runs", ",".join(probe_runs), "--qpred-runs", ",".join(qpred_runs)])
    assert rc == 0
    out = next((tmp_path / "scratch").glob("ssl_r1c_report_*"))
    reading = json.loads((out / "reading.json").read_text())
    assert reading["A3"] == "2 GAIN FROM CONTINUED SEEDING, SELECTION OR VOTES, NOT FROM LOGIC"   # d_ladder 1.5, d_twin 0.5
    assert reading["A0"] == "3 TRAINING DOES NOT ADD VALUE"                                         # bank below c0y
    with pytest.raises(SystemExit):
        SC.main(["--stage", "train", "--arm", "A3", "--seed", "7"])                                # not a seed of the study
