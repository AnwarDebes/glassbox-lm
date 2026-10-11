"""Tests of the exact Q_pred speedups (src/tmlm/ssl/qpred_fast.py; experiments/ssl_r1_confirm.py features_core,
qpred_core, speedup_used; compute-reduction items O4, O10, O12; docs/13 A8 item 7).

(a) O4 source: the objective kernel is the registered cvtm_t1_readout.T1_SRC (sha256 pinned) with exactly the listed
    replacements (the log term loaded instead of recomputed); the one-off kernel holds the registered expression.
(b) O4 on the GPU (skipped without a device): every output of T1GpuG.run is bit-identical to T1Gpu.run on synthetic
    tables at several (alpha, beta0), with and without gradients, for two chunkings and two grids.
(c) O10: prefix_pass equals ssl_r1_pretext.fire_counts, ssl_r1_controls.count_table and fit_rows_fired (CPU engine;
    the CUDA engine against the CPU engine and for two chunk sizes when a device is visible); the table file round-trips.
(d) the stages on toy data: features_core writes the registered features files plus the tables; qpred_core with the
    tables loaded and the state-hash dedupe gives the registered stage_qpred's metrics and files (timing excluded), the
    duplicate's metrics are copies labelled derived, the registered computation of the duplicate equals the copy
    (CPU engine; the CUDA engine with T1GpuG when a device is visible).
(e) the loader refuses tables computed from other inputs or with another file hash; the speedup gate uses an item only
    when it is enabled and its verification runs are clean.
All data are synthetic and seeded; no corpus, no results/ access."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _q in (ROOT / "experiments", ROOT / "src"):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))

import cvtm_t1_readout as T1  # noqa: E402
import ssl_r1_confirm as S  # noqa: E402
import ssl_r1_controls as C  # noqa: E402
import ssl_r1_pretext as R  # noqa: E402
from ckm_posthoc_probe import k_dense_to_csr, k_gval  # noqa: E402
from tmlm.data.synthetic import toy_corpus  # noqa: E402
from tmlm.engine.cpu.cpu_engine import CpuEngine  # noqa: E402
from tmlm.ssl import pretext as P  # noqa: E402
from tmlm.ssl import qpred_fast as QF  # noqa: E402

T1_SRC_SHA256 = "0a16974f13f031d11540d57cb796dcf371e162699ad6d56aedb201ae3bdcbd2c"
EXPR = "log(1.0 + l * (double)val[e] * inn / (oml * pu[c]))"
CTL = C.load_config(ROOT / "experiments" / "cvtm" / "configs" / "ssl_r1_controls.json")
TINY = dict(vocab=40, n_clauses=64, batch=16, sweep_period=8, n_bag_k=2, dead_sweeps=2)


def _gpu_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no CUDA device: {exc}")
    return cp


# ------------------------------------------------------------------ (a) the kernel source
def test_objective_source_is_the_registered_kernel_with_the_log_term_loaded():
    assert hashlib.sha256(T1.T1_SRC.encode()).hexdigest() == T1_SRC_SHA256, "the registered kernel changed: re-derive and re-verify"
    assert QF.derive_objective_source(T1.T1_SRC) == QF.T1G_OBJECTIVE_SRC
    assert T1.T1_SRC.count(EXPR) == 2 and QF.T1G_GVAL_SRC.count(EXPR) == 1 and EXPR not in QF.T1G_OBJECTIVE_SRC
    assert "const double l = lam[k], inn = inv_n[k], oml = 1.0 - l;" in QF.T1G_GVAL_SRC
    assert QF.T1G_OPTIONS == ("-std=c++14",)
    for use in ("llrint(cf * g * scale)", "eg += P[c] * gv;", "if (c == g) gy += gv;"):   # the uses of g, unchanged
        assert T1.T1_SRC.count(use) == QF.T1G_OBJECTIVE_SRC.count(use) == 1
    with pytest.raises(ValueError):
        QF.derive_objective_source(T1.T1_SRC.replace(EXPR, "0.0", 1))


# ------------------------------------------------------------------ (b) bit identity on the GPU
def _synthetic(C_: int, V: int, n_rows: int, seed: int, max_fired: int) -> dict:
    rng = np.random.default_rng(seed)
    counts = np.zeros((C_, V), np.int32)
    for j in range(1, C_):
        k = int(rng.integers(1, min(V, 200)))
        cls = rng.choice(V, size=k, replace=False)
        counts[j, cls] = rng.integers(1, 5000, size=k)
    indptr, idx, val = k_dense_to_csr(counts)
    N = T1.k_row_sums(indptr, val)
    lam, inv_n = T1.witten_bell(N, np.diff(indptr))
    p_uni = T1.add_one_unigram(rng.integers(0, V, 5000), V)
    grp = T1.clause_groups(rng.integers(0, 15, C_), N)
    ptr, ids = [0], []
    for _ in range(n_rows):
        f = np.sort(rng.choice(C_, size=int(rng.integers(0, max_fired + 1)), replace=False)).astype(np.uint32)
        ids.append(f)
        ptr.append(ptr[-1] + f.shape[0])
    rp, rowid, comp = T1.csr_select(np.array(ptr, np.int64), np.concatenate(ids), N > 0)
    rows = dict(n=n_rows, rp=rp, rowid=rowid, comp=comp, bkt=grp[comp].astype(np.int64), gold=rng.integers(0, V, n_rows).astype(np.int64))
    return dict(indptr=indptr, idx=idx, val=val, lam=lam, inv_n=inv_n, p_uni=p_uni, rows=rows,
                gval=k_gval(indptr, idx, val, lam, inv_n, p_uni))


@pytest.mark.parametrize("V,grid", [(300, 64), (8000, 1024)])
def test_fast_kernel_outputs_are_bit_identical_to_the_registered_kernel(V, grid):
    _gpu_or_skip()
    s = _synthetic(70, V, 400, 11, 16)
    r = s["rows"]
    used = np.unique(r["bkt"])
    args = (V, s["indptr"], s["idx"], s["val"], s["lam"], s["inv_n"], s["p_uni"])
    reg, fast = T1.T1Gpu(*args, grid=grid), QF.T1GpuG(*args, grid=grid)
    assert np.abs(fast.gval() - s["gval"]).max() < 1e-12            # the stored log terms (CPU reference to rounding)
    rng = np.random.default_rng(3)
    half = np.zeros(T1.N_BUCKETS)
    half[used] = rng.normal(0.3, 0.2, used.shape[0])
    half[used[1::2]] = 0.0
    pts = [(rng.normal(0.4, 0.3, T1.N_BUCKETS) * np.isin(np.arange(T1.N_BUCKETS), used), 0.9), (half, 0.5),
           (3.0 * np.isin(np.arange(T1.N_BUCKETS), used), -2.0)]
    n_pairs = int(r["comp"].shape[0])
    for rows_per_chunk in (37, 16384):
        ch_r = reg.prepare(r["rp"], r["comp"], r["bkt"], r["gold"], rows_per_chunk=rows_per_chunk)
        ch_f = fast.prepare(r["rp"], r["comp"], r["bkt"], r["gold"], rows_per_chunk=rows_per_chunk)
        for alpha, beta0 in pts:
            for wg in (True, False):
                eq = QF.outputs_bit_identical(reg.run(ch_r, r["n"], n_pairs, alpha, beta0, wg), fast.run(ch_f, r["n"], n_pairs, alpha, beta0, wg))
                assert all(eq.values()), eq
                assert set(eq) == ({"lpg", "pg", "rank", "top", "sum", "gpair", "gbeta"} if wg else {"lpg", "pg", "rank", "top", "sum"})


def test_outputs_bit_identical_sees_signed_zero_and_dtype():
    a = {"x": np.array([0.0, 1.0]), "r": np.array([1, 2], np.int64)}
    assert all(QF.outputs_bit_identical(a, copy.deepcopy(a)).values())
    assert not QF.outputs_bit_identical(a, {"x": np.array([-0.0, 1.0]), "r": a["r"]})["x"]
    assert not QF.outputs_bit_identical(a, {"x": a["x"], "r": a["r"].astype(np.int32)})["r"]
    assert not QF.outputs_bit_identical(a, {"x": a["x"]})["r"]


# ------------------------------------------------------------------ (c) the one prefix pass
def _params(**over) -> dict:
    p = copy.deepcopy(CTL["params"])
    p["young_ladder"] = {**p["young_ladder"], "abs_batches": [2, 3], "percent_of_epoch1": [25, 50]}
    p["young_fraction"] = 0.3
    p.update(over)
    return p


def _bank(seed: int = 4, n_docs: int = 8):
    c = toy_corpus(vocab=40, n_docs=n_docs, doc_len=70, seed=seed)
    cfg = C.arm_config("A0", 2, _params(), **TINY)
    eng = CpuEngine(cfg, corpus=c)
    for e in range(2):
        for b in range(eng.n_batches()):
            eng.train_batch(e, b)
    ds = np.asarray(c.doc_start, np.int64)
    fit_pos = np.concatenate([np.arange(ds[0], ds[1]), np.arange(ds[3], ds[4])]).astype(np.int64)
    cnt_pos = np.setdiff1d(np.arange(c.tok.shape[0]), fit_pos).astype(np.int64)
    check_rows = np.array([0, 5, 17, cnt_pos.shape[0] - 1], np.int64)
    return c, cfg, eng, fit_pos, cnt_pos, check_rows


def _rows(eng, kind: str, n: int, chunk: int):
    """The registered fired-list function with Q_pred's all-zero bitmap (what the features stage binds)."""
    return lambda pos: C.fit_rows_fired(eng, kind, pos, np.zeros(n, np.uint8), chunk)


def _assert_pass_equals_registered(pp: QF.PrefixPass, eng, engine_kind: str, cfg, c, fit_pos, cnt_pos, check_rows, chunk: int):
    n = int(c.tok.shape[0])
    tok = np.asarray(c.tok).astype(np.int64)
    zero = np.zeros(n, np.uint8)
    fc = R.fire_counts(eng, cfg, n, chunk)
    assert pp.fire_counts.dtype == fc.dtype and np.array_equal(pp.fire_counts, fc)
    counts, n_ev, got = C.count_table(eng, engine_kind, cnt_pos, tok[cnt_pos], 64, 40, chunk, zero, check_rows)
    dense = pp.counts_dense()
    assert dense.dtype == counts.dtype and np.array_equal(dense, counts) and counts.sum() > 0
    assert pp.n_events == n_ev
    g = pp.got()
    assert sorted(g) == sorted(got) and all(np.array_equal(g[r], got[r]) and g[r].dtype == got[r].dtype for r in got)
    fptr, fids = C.fit_rows_fired(eng, engine_kind, fit_pos, zero, chunk)
    assert np.array_equal(pp.fit_ptr, fptr) and pp.fit_ids.dtype == fids.dtype and np.array_equal(pp.fit_ids, fids)


def test_prefix_pass_equals_fire_counts_count_table_and_fit_rows_cpu():
    c, cfg, eng, fit_pos, cnt_pos, check_rows = _bank()
    n = int(c.tok.shape[0])
    pp = QF.prefix_pass(eng, 64, 40, n, 64, R.hidden_for(cfg, n), c.tok, cnt_pos, fit_pos, check_rows, _rows(eng, "cpu", n, 64))
    _assert_pass_equals_registered(pp, eng, "cpu", cfg, c, fit_pos, cnt_pos, check_rows, 64)


def test_prefix_pass_cuda_equals_registered_and_cpu_for_any_chunk():
    _gpu_or_skip()
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    c, cfg, eng, fit_pos, cnt_pos, check_rows = _bank(seed=6)
    n = int(c.tok.shape[0])
    g = CudaEngine(cfg, corpus=c)
    g.load_state(eng.state())
    ref = QF.prefix_pass(eng, 64, 40, n, 64, R.hidden_for(cfg, n), c.tok, cnt_pos, fit_pos, check_rows, _rows(eng, "cpu", n, 64))
    for chunk in (64, 7):
        pg = QF.prefix_pass(g, 64, 40, n, chunk, R.hidden_for(cfg, n), c.tok, cnt_pos, fit_pos, check_rows, _rows(g, "cuda", n, chunk))
        _assert_pass_equals_registered(pg, g, "cuda", cfg, c, fit_pos, cnt_pos, check_rows, chunk)
        for f in ("fire_counts", "indptr", "idx", "val", "fit_ptr", "fit_ids", "check_rows", "check_ptr", "check_ids"):
            assert np.array_equal(getattr(pg, f), getattr(ref, f)) and getattr(pg, f).dtype == getattr(ref, f).dtype, f
        assert pg.n_events == ref.n_events


def test_prefix_counts_file_round_trips(tmp_path):
    c, cfg, eng, fit_pos, cnt_pos, check_rows = _bank(seed=7, n_docs=5)
    n = int(c.tok.shape[0])
    pp = QF.prefix_pass(eng, 64, 40, n, 64, R.hidden_for(cfg, n), c.tok, cnt_pos, fit_pos, check_rows, _rows(eng, "cpu", n, 64))
    meta = QF.pass_meta(eng.state_hash(), c, c, fit_pos, cnt_pos, check_rows, cfg, "cpu")
    sha = QF.save_prefix_counts(tmp_path / "t.npz", pp, meta)
    assert len(sha) == 64
    back, m2 = QF.load_prefix_counts(tmp_path / "t.npz")
    assert m2 == meta
    assert np.array_equal(back.counts_dense(), pp.counts_dense()) and back.n_events == pp.n_events
    assert all(np.array_equal(getattr(back, f), getattr(pp, f)) for f in ("fit_ptr", "fit_ids", "check_rows", "check_ptr", "check_ids"))


# ------------------------------------------------------------------ (d) the stages on toy data
def _clean_run(d: Path) -> str:
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"exit_status": 0}))
    (d / "metrics.json").write_text(json.dumps({"failed_checks": {"value": []}}))
    return str(d)


class _World:
    """A fresh run directory per stage (C.out_dir returns it)."""

    def __init__(self, tmp: Path, monkeypatch) -> None:
        self.tmp, self.n, self.cur = tmp, 0, None
        monkeypatch.setattr(C, "out_dir", lambda stage: self.cur)


def _stage(w: _World, name: str, fn, args, cj, train_run, **kw) -> tuple[Path, dict, list]:
    from tmlm.common import runlog
    w.n += 1
    w.cur = w.tmp / f"{w.n:02d}_{name}"
    w.cur.mkdir()
    rec = R.Rec(False)
    fn(args, cj, rec, train_run, **kw)
    rec.m("failed_checks", rec.failed)
    m = runlog.validate_metrics(rec.metrics)
    (w.cur / "metrics.json").write_text(json.dumps(m))
    (w.cur / "run.json").write_text(json.dumps({"exit_status": 1 if rec.failed else 0}))
    return w.cur, m, list(rec.failed)


def _canon(e: dict, fit: bool = False) -> str:
    e = {k: v for k, v in e.items() if k not in ("exploratory", "confirmatory", "smoke")}
    if fit:
        e["value"] = {k: v for k, v in e["value"].items() if k not in ("wall_s", "s_per_evaluation")}
    return json.dumps(e, sort_keys=True)


def _timing(k: str) -> bool:
    last = k.split(".")[-1]
    return last.endswith("wall_s") or last.endswith("_per_s") or k.startswith("memory.")


def _toy_world(tmp_path, monkeypatch, engine: str):
    import tmlm.data.corpus_store as CS
    import tmlm.data.evalsets as EV

    monkeypatch.setenv("TMLM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(C, "CONTROL_ARMS", dict(C.CONTROL_ARMS))
    monkeypatch.setattr(P, "F2_M", 60)
    cj = copy.deepcopy(CTL)
    p = cj["params"]
    p.update(vocab=40, n_clauses=64, epochs=2, forward_chunk=64, dim_matched=16, young_fraction=0.3)
    p["young_ladder"] = {**p["young_ladder"], "abs_batches": [1, 2], "percent_of_epoch1": [50]}
    p["qpred"]["checks"] = {"cross_engine_count_rows": 20, "cross_engine_fit_rows": 10, "validation_api_rows": 20,
                            "validation_bruteforce_rows": 3, "signed_predict_scores_rows": 30}
    cj["doc_set_reference"] = {}
    cj["replay_reference"] = {}
    full = toy_corpus(vocab=40, n_docs=30, doc_len=120, seed=5)
    prefix = P.slice_prefix(full, 24 * 120)
    docs_c = toy_corpus(vocab=40, n_docs=6, doc_len=15, seed=3)
    from tmlm.ssl import features as F
    docs = F.TokenDocs(tok=np.asarray(docs_c.tok), doc_start=np.asarray(docs_c.doc_start), end_id=39)
    monkeypatch.setattr(R, "load_prefix", lambda p_, rec: prefix)
    monkeypatch.setattr(R, "doc_set", lambda p_, rec, full_: (docs, {"toy": {}}, {"toy": {}}))
    monkeypatch.setattr(R, "dataset_hashes", lambda d: None)
    val = toy_corpus(vocab=40, n_docs=4, doc_len=60, seed=9)
    targets = np.arange(3, val.tok.shape[0], 5, dtype=np.int64)
    hidden = np.zeros(val.tok.shape[0], np.uint8)
    hidden[targets] = 1
    vdir = tmp_path / "vmeta"
    vdir.mkdir()
    (vdir / "meta.json").write_text(json.dumps({"sha256": {"tok.npy": "x"}}))
    monkeypatch.setattr(CS, "load_split", lambda corpus, vocab, split, **kw: val if split == "validation" else full)
    monkeypatch.setattr(CS, "split_dir", lambda corpus, vocab, split, root=None: vdir)
    monkeypatch.setattr(EV, "load_evalset", lambda corpus, vocab, split, variant, **kw: {"targets": targets, "hidden": hidden, "meta": {"sha256_combined": "e"}})
    w = _World(tmp_path, monkeypatch)
    # a train run of A3 (seed 1) with its ladder, through the registered stage
    w.n += 1
    w.cur = tmp_path / f"{w.n:02d}_train"
    w.cur.mkdir()
    rec = R.Rec(False)
    C.stage_train(argparse.Namespace(arm="A3", seed=1, engine=engine), cj, rec)
    from tmlm.common import runlog
    tm = runlog.validate_metrics(rec.metrics)
    tr = w.cur
    # a young rung with the state of c0y (as A3's young_b77 in the real runs): a copy of the checkpoint, its logged hash
    shutil.copyfile(tr / "checkpoint_c0y.npz", tr / "checkpoint_young_b9999.npz")
    tm["young.young_b9999.state_hash"] = dict(tm["c0y.state_hash"])
    (tr / "metrics.json").write_text(json.dumps(tm))
    (tr / "run.json").write_text(json.dumps({"exit_status": 0}))
    return w, cj, tr


def _check_stages(tmp_path, monkeypatch, engine: str):
    w, cj, tr = _toy_world(tmp_path, monkeypatch, engine)
    names = "c0y,young_b9999,epoch2"
    ns = argparse.Namespace(checkpoints=names, engine=engine)
    # ---- features: registered vs the one pass
    fa, ma, fla = _stage(w, "features_registered", C.stage_features, ns, cj, tr)
    fb, mb, flb = _stage(w, "features_onepass", S.features_core, argparse.Namespace(checkpoints=names, engine=engine), cj, tr)
    assert fla == [] and flb == [], (fla, flb)
    for n in names.split(","):
        for pat in ("features_{}.npz", "fires_{}.npy", "dim_{}.npy"):
            assert (fa / pat.format(n)).read_bytes() == (fb / pat.format(n)).read_bytes(), pat.format(n)
    assert {q.name for q in fb.glob("prefix_counts_*.npz")} == {"prefix_counts_c0y.npz", "prefix_counts_epoch2.npz"}
    assert mb["prefix_counts.young_b9999.duplicate_of"]["value"] == "c0y"
    for k, e in ma.items():
        if not _timing(k) and k not in ("wall_total_s",):
            assert k in mb and _canon(e) == _canon(mb[k]), k
    # ---- qpred: registered vs O10 + O12 (+ O4 on CUDA)
    qa, qma, qfa = _stage(w, "qpred_registered", C.stage_qpred, argparse.Namespace(checkpoints=names, engine=engine), cj, tr)
    qb, qmb, qfb = _stage(w, "qpred_fast", S.qpred_core, argparse.Namespace(checkpoints=names, engine=engine), cj, tr,
                          features_run=fb, fast_kernel=True, load_counts=True, dedupe=True)
    assert qfa == [] and qfb == [], (qfa, qfb)
    assert qmb["qpred.c0y.count_source"]["value"] == "loaded" and qmb["qpred.epoch2.count_source"]["value"] == "loaded"
    assert qmb["qpred.young_b9999.duplicate_of"]["value"] == "c0y"
    n_cmp = 0
    for k, e in qma.items():
        if _timing(k) or k in ("wall_total_s", "failed_checks"):
            continue
        if ".young_b9999." in k and k.startswith("check.qpred."):
            assert k not in qmb                     # the copy runs no checks of its own
            continue
        assert k in qmb, k
        if k.startswith("qpred.young_b9999."):     # registered computation == the copy (O12), labelled derived
            assert qmb[k]["label"] == "derived" and qmb[k]["copied_from"] == "c0y", k
            assert _canon({"value": e["value"]}, fit=k.endswith(".fit")) == _canon({"value": qmb[k]["value"]}, fit=k.endswith(".fit")), k
        elif k == "qpred.summary":
            assert json.dumps(e["value"], sort_keys=True) == json.dumps(qmb[k]["value"], sort_keys=True)
        else:
            assert _canon(e, fit=k.endswith(".fit")) == _canon(qmb[k], fit=k.endswith(".fit")), k
        n_cmp += 1
    assert n_cmp > 30
    for n in names.split(","):
        with np.load(qa / f"qpred_{n}.npz") as x, np.load(qb / f"qpred_{n}.npz") as y:
            assert sorted(x.files) == sorted(y.files) and all(x[f].tobytes() == y[f].tobytes() and x[f].dtype == y[f].dtype for f in x.files)
        assert (qa / f"qpred_{n}_readout.json").read_bytes() == (qb / f"qpred_{n}_readout.json").read_bytes()
    return w, cj, tr, fb


def test_stages_on_toy_data_cpu(tmp_path, monkeypatch):
    _check_stages(tmp_path, monkeypatch, "cpu")


def test_stages_on_toy_data_cuda(tmp_path, monkeypatch):
    _gpu_or_skip()
    w, cj, tr, fb = _check_stages(tmp_path, monkeypatch, "cuda")
    qm = json.loads((w.tmp / f"{w.n:02d}_qpred_fast" / "metrics.json").read_text())
    assert "T1GpuG" in qm["qpred.c0y.t1_setup_wall_s"]["kernel"]


# ------------------------------------------------------------------ (e) loader refusals and the speedup gate
def test_loader_refuses_other_inputs_and_another_file_hash(tmp_path):
    c, cfg, eng, fit_pos, cnt_pos, check_rows = _bank(seed=8, n_docs=5)
    n = int(c.tok.shape[0])
    pp = QF.prefix_pass(eng, 64, 40, n, 64, R.hidden_for(cfg, n), c.tok, cnt_pos, fit_pos, check_rows, _rows(eng, "cpu", n, 64))
    h = eng.state_hash()
    meta = QF.pass_meta(h, c, c, fit_pos, cnt_pos, check_rows, cfg, "cpu")
    f = tmp_path / "prefix_counts_x.npz"
    sha = QF.save_prefix_counts(f, pp, meta)
    idx = {h: (f, sha)}
    got, why = S.load_matching_counts(idx, None, meta)
    assert got is not None and why is None
    assert S.load_matching_counts(idx, None, {**meta, "state_hash": "other"})[0] is None
    for k in ("prefix_tok_sha256", "kind_map_sha256", "check_rows_sha256", "engine", "fit_pos_sha256"):
        got, why = S.load_matching_counts(idx, None, {**meta, k: "x"})
        assert got is None and k in why
    got, why = S.load_matching_counts({h: (f, "0" * 64)}, None, meta)
    assert got is None and "sha256" in why


def test_prefix_count_index_refuses_another_train_run_or_an_unclean_run(tmp_path):
    fr = tmp_path / "f"
    fr.mkdir()
    m = {"control.arm": {"value": "A3"}, "control.seed": {"value": 1}, "features.train_run": {"value": str(tmp_path / "tr")},
         "prefix_counts.c0y.file": {"value": "prefix_counts_c0y.npz"}, "prefix_counts.c0y.state_hash": {"value": "h"},
         "prefix_counts.c0y.file_sha256": {"value": "s"}, "failed_checks": {"value": []}}
    (fr / "metrics.json").write_text(json.dumps(m))
    (fr / "run.json").write_text(json.dumps({"exit_status": 0}))
    idx, why = S.prefix_count_index(fr, tmp_path / "tr", "A3", 1)
    assert why is None and idx == {"h": (fr / "prefix_counts_c0y.npz", "s")}
    assert S.prefix_count_index(fr, tmp_path / "other", "A3", 1)[0] == {}
    assert S.prefix_count_index(fr, tmp_path / "tr", "A2", 1)[0] == {}
    assert S.prefix_count_index(None, tmp_path / "tr", "A3", 1) == ({}, "no features run given")
    (fr / "run.json").write_text(json.dumps({"exit_status": 1}))
    assert S.prefix_count_index(fr, tmp_path / "tr", "A3", 1)[0] == {}


def test_speedup_gate_needs_enabled_and_clean_verification_runs(tmp_path, monkeypatch):
    good, bad = _clean_run(tmp_path / "good"), str(tmp_path / "bad")
    Path(bad).mkdir()
    (Path(bad) / "run.json").write_text(json.dumps({"exit_status": 1}))
    rec = R.Rec(False)
    assert not S.speedup_used({}, "O4_t1_gval", rec) and rec.metrics["speedup.O4_t1_gval"]["value"]["used"] is False
    assert not S.speedup_used({"speedups": {"O4_t1_gval": {"enabled": True, "verified_by": None}}}, "O4_t1_gval")
    assert not S.speedup_used({"speedups": {"O4_t1_gval": {"enabled": False, "verified_by": [good]}}}, "O4_t1_gval")
    assert not S.speedup_used({"speedups": {"O4_t1_gval": {"enabled": True, "verified_by": [good, bad]}}}, "O4_t1_gval")
    assert S.speedup_used({"speedups": {"O4_t1_gval": {"enabled": True, "verified_by": [good]}}}, "O4_t1_gval")
    # no item used: the registered stage runs
    called = []
    monkeypatch.setattr(C, "stage_qpred", lambda args, cj, rec, tr: called.append("registered"))
    monkeypatch.setattr(S, "qpred_core", lambda *a, **k: called.append(("fast", k)))
    S.stage_qpred(argparse.Namespace(), {}, R.Rec(False), tmp_path, None)
    cj = {"speedups": {k: {"enabled": True, "verified_by": [good]} for k in S.SPEEDUPS}}
    S.stage_qpred(argparse.Namespace(), cj, R.Rec(False), tmp_path, None)
    S.stage_qpred(argparse.Namespace(), cj, R.Rec(False), tmp_path, tmp_path)
    assert called[0] == "registered"
    assert called[1][1] == {"fast_kernel": True, "load_counts": False, "dedupe": True}      # no features run: recompute
    assert called[2][1] == {"fast_kernel": True, "load_counts": True, "dedupe": True}


def test_config_speedups_block_names_every_item():
    cj = S.load_config(S.DEFAULT_CONFIG)
    assert set(cj["speedups"]) - {"rule"} == set(S.SPEEDUPS)
    for k in S.SPEEDUPS:
        spec = cj["speedups"][k]
        assert isinstance(spec["enabled"], bool)
        if spec["enabled"]:                         # an enabled item names its verification runs
            assert spec["verified_by"] and all(str(r).startswith("results/") for r in spec["verified_by"])
