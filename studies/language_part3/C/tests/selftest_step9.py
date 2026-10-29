"""Unit tests of the step-9 code of C-core (CoTM loading, loop, read-out, epoch and grid picks, C-K1 / C-K2 / C-K3,
subsets, seed averaging, systems, the TM run record, and issue 7: freeze.py --amend and check_frozen). Synthetic
data only, on the CPU with the GPU hidden; no number of C.

    bash tests/run_step9_tests.sh            # from C/: runs this file, the end-to-end test and the two older self-tests

or alone, with the environment the runners set (python -I -B, nice 19, 2 threads, caches in C/tmp, GPU hidden):
    cd studies/language_part3/C && env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 RAYON_NUM_THREADS=2 NUMBA_CACHE_DIR=$PWD/tmp/numba \
      TMPDIR=$PWD/tmp/tmp JOBLIB_TEMP_FOLDER=$PWD/tmp/joblib MPLCONFIGDIR=$PWD/tmp/mpl XDG_CACHE_HOME=$PWD/tmp/xdg \
      CUDA_VISIBLE_DEVICES= CUDA_CACHE_DISABLE=1 PYTHONHASHSEED=0 C_GPU_RUNNER=1 nice -n 19 \
      ../../decision_layer/.venv-dl/bin/python -I -B tests/selftest_step9.py

Prints PASS or FAIL per test and exits 1 on any failure."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import env  # noqa: E402

env.guard()
import numpy as np  # noqa: E402

from ccore import gates, hashing, learners, paths, runlog, systems, tmcore, tmtorch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step9_synth  # noqa: E402

RESULTS = []
SCRATCH = paths.TMP / "step9_build" / "unit"


def test(fn):
    RESULTS.append(fn)
    return fn


def torch():
    return tmtorch.setup_torch("cpu")


def tiny_xy(rng, n, f=60, rule=True):
    X = rng.integers(0, 2, size=(n, f)).astype(np.uint8)
    y = (2 * X[:, 0] + X[:, 1]).astype(np.int64) if rule else rng.integers(0, 65, size=n)
    noise = rng.random(n) < 0.1
    y = np.where(noise, rng.integers(0, 65, size=n), y)
    return X, y


# ---------------------------------------------------------------- the CoTM: loaded by path, hash checked
@test
def cotm_loaded_by_path_with_its_full_hash():
    mod = tmtorch.cotm()
    assert hashlib.sha256(tmtorch.TM_FILE.read_bytes()).hexdigest() == tmtorch.TM_SHA256
    assert tmtorch.TM_SHA256 == "f649043f5801605b35174c526b290a0cfbadbdb66ade5bf04dafc122289d3424"
    assert mod.__file__ == str(tmtorch.TM_FILE) and hasattr(mod, "CoTM") and hasattr(mod, "TMConfig")
    assert "dl" not in sys.modules and "dl.tm" not in sys.modules, "the dl package was imported"
    fast = tmtorch.fast()
    assert hashlib.sha256(tmtorch.FAST_FILE.read_bytes()).hexdigest() == tmtorch.FAST_SHA256 and hasattr(fast, "pack")
    try:
        tmtorch.load_by_path(tmtorch.TM_FILE, "0" * 64, "c_selftest_bad_hash")
        raise AssertionError("a wrong hash was accepted")
    except SystemExit as e:
        assert "not the registered" in str(e)
    assert "c_selftest_bad_hash" not in sys.modules


@test
def registered_config_and_settings():
    cfg = tmtorch.make_config(8000, 10, 1, 8000, 20)
    from dataclasses import asdict
    d = asdict(cfg)
    assert (d["n_clauses"], d["T"], d["s"], d["epochs"], d["batch_size"], d["neg_sampling"], d["seed"]) == \
        (8000, 8000, 10.0, 20, 128, "random", 1)
    assert all(d[k] == v for k, v in tmtorch.DEFAULTS.items()) and d["negation"] is True
    R = tmcore.REGISTERED
    assert R.clauses == 8000 and R.max_epochs == 20 and R.seeds == tuple(range(1, 11)) and R.c0_seeds == R.seeds
    assert R.grid == ((8000, 3.0), (8000, 10.0), (8000, 30.0), (8000, 90.0),
                      (16000, 3.0), (16000, 10.0), (16000, 30.0), (16000, 90.0))
    assert R.smoke == {"share": 0.10, "subset_seed": 1, "epochs": 2, "T": 8000, "s": 10.0, "seed": 1}
    assert R.c0 == {"share": 0.01, "subset_seed": 1} and R.device == "cuda"
    assert tmcore.N_CLASSES == 65 and tmcore.N_FEATURES == 60 and tmcore.K3_LIMIT_MIN == 230.0
    assert tmcore.SHUFFLE_PERM_SEED == 1 and tmcore.GRID_SEED == 1
    assert len(tmcore.TEST.grid) == 8 and tmcore.TEST.device == "cpu" and tmcore.DRY.clauses == 8000


# ---------------------------------------------------------------- the training loop
def reference_loop(X, y, Xc, yc, T, s, seed, clauses, epochs):
    """The loop written out by hand: CoTM, permutation from its own generator, _step per batch of 128."""
    import torch as t
    mod = tmtorch.cotm()
    tm = mod.CoTM(65, X.shape[1], tmtorch.make_config(T, s, seed, clauses, epochs), device="cpu")
    Xt, yt = t.as_tensor(X), t.as_tensor(y)
    states, losses = [], []
    for _ in range(epochs):
        perm = t.randperm(len(X), generator=tm.gen)
        for a in range(0, len(X), 128):
            tm._step(Xt[perm[a:a + 128]], yt[perm[a:a + 128]])
        S = tm.class_sums(t.as_tensor(Xc)).round().long().numpy()
        losses.append(tmcore.fit_readout(S, yc)["calib_class_bits"])
        states.append((tm.ta.numpy().astype(np.int16).copy(), tm.w.numpy().copy()))
    return states, losses


@test
def loop_equals_the_reference_loop_and_keeps_the_rule_epochs():
    torch()
    rng = np.random.default_rng(1)
    X, y = tiny_xy(rng, 700)
    Xc, yc = tiny_xy(rng, 300)
    res = tmtorch.train(X, y, Xc, yc, 20, 5.0, 3, 30, 4, "cpu", log=lambda *a: None)
    states, losses = reference_loop(X, y, Xc, yc, 20, 5.0, 3, 30, 4)
    assert [h["calib_class_bits"] for h in res["history"]] == losses
    e = res["best"]["epoch"]
    assert e == tmcore.pick_epoch(losses)
    assert np.array_equal(res["best"]["ta"], states[e - 1][0]) and np.array_equal(res["best"]["w"], states[e - 1][1])
    assert res["epoch1"]["epoch"] == 1 and np.array_equal(res["epoch1"]["ta"], states[0][0])
    assert np.array_equal(res["epoch1"]["w"], states[0][1])
    assert res["batches"] == 4 * math.ceil(700 / 128) and res["n_train"] == 700
    res2 = tmtorch.train(X, y, Xc, yc, 20, 5.0, 3, 30, 4, "cpu", log=lambda *a: None)
    assert np.array_equal(res2["best"]["ta"], res["best"]["ta"]) and res2["history"][-1]["calib_class_bits"] == \
        res["history"][-1]["calib_class_bits"], "not deterministic for a seed"
    res3 = tmtorch.train(X, y, Xc, yc, 20, 5.0, 4, 30, 4, "cpu", log=lambda *a: None)
    assert not np.array_equal(res3["epoch1"]["ta"], res["epoch1"]["ta"]), "another seed gave the same machine"


@test
def the_cotm_learns_a_rule_on_the_cpu():
    torch()
    rng = np.random.default_rng(2)
    X, y = tiny_xy(rng, 3000)
    Xc, yc = tiny_xy(rng, 1000)
    res = tmtorch.train(X, y, Xc, yc, 30, 3.0, 1, 60, 5, "cpu", log=lambda *a: None)
    uni = tmcore.class_logloss_bits(systems.null_class_model(y, len(yc)), yc)
    assert res["best"]["calib_class_bits"] < uni - 0.5, (res["best"]["calib_class_bits"], uni)


@test
def stored_state_gives_the_same_sums_and_packed_cpu_sums_agree():
    torch()
    rng = np.random.default_rng(3)
    X, y = tiny_xy(rng, 900)
    Xc, yc = tiny_xy(rng, 400)
    res = tmtorch.train(X, y, Xc, yc, 20, 5.0, 1, 50, 3, "cpu", log=lambda *a: None)
    ck = res["best"]
    S = tmtorch.checkpoint_sums(res["cfg"], ck, Xc, "cpu")
    assert S.dtype == np.int64 and S.shape == (400, 65)
    assert tmcore.class_logloss_bits(tmcore.readout_proba(S, ck["readout"]), yc) == ck["calib_class_bits"]
    pk = tmtorch.packed_equal(ck, Xc, S)
    assert pk["equal"] and pk["rows"] == 400
    bad = S.copy()
    bad[0, 0] += 1
    assert not tmtorch.packed_equal(ck, Xc, bad)["equal"]
    assert tmcore.clause_stats(ck["ta"]) == ck["stats"], "stats of the stored state differ from CoTM.stats()"
    h = tmcore.length_histogram(ck["ta"])
    assert sum(h["counts"]) == 50 and h["edges"][-1] == 121


# ---------------------------------------------------------------- the read-out
def loss(S, y, a, b):
    L = S * a + b
    L = L - L.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(L).sum(1)) - L[np.arange(len(y)), y]))


@test
def readout_is_the_calib_logloss_optimum():
    rng = np.random.default_rng(4)
    n, K = 3000, 65
    y = rng.integers(0, K, size=n)
    S = rng.normal(0, 300, size=(n, K))
    S[np.arange(n), y] += 400
    S = np.round(S).astype(np.int64)
    ro = tmcore.fit_readout(S, y)
    assert ro["converged"] and ro["a_scaled"] > 0 and abs(ro["tau"] - ro["sigma"] / ro["a_scaled"]) < 1e-12
    PX = tmcore.readout_proba(S, ro)
    assert np.allclose(PX.sum(1), 1, atol=1e-12) and PX.min() > 0
    a, b = ro["a_scaled"] / ro["sigma"], np.asarray(ro["b"])
    f0 = loss(S.astype(float), y, a, b)
    assert abs(f0 / math.log(2) - ro["calib_class_bits"]) < 1e-9
    for _ in range(50):                                 # no nearby point is better
        da, db = rng.normal(0, 0.01 * a), rng.normal(0, 0.01, size=K)
        assert loss(S.astype(float), y, a + da, b + db) >= f0 - 1e-9
    from scipy.optimize import minimize               # an independent fit (unbounded, log a, numerical gradient)
    Sf = S.astype(float) / ro["sigma"]
    r = minimize(lambda x: loss(Sf, y, math.exp(x[0]), x[1:]), np.concatenate([[0.0], np.zeros(K)]), method="BFGS",
                 options={"gtol": 1e-7, "maxiter": 2000})
    assert abs(r.fun - f0) < 1e-6, (r.fun, f0)
    ro3 = tmcore.fit_readout(3 * S, y)                 # the fit does not depend on the scale of S
    assert abs(ro3["calib_class_bits"] - ro["calib_class_bits"]) < 1e-9


@test
def readout_without_signal_gives_the_calib_class_frequencies():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 65, size=5000)
    y[:65] = np.arange(65)
    ro = tmcore.fit_readout(np.zeros((5000, 65), dtype=np.int64), y)
    PX = tmcore.readout_proba(np.zeros((3, 65)), ro)
    freq = np.bincount(y, minlength=65) / 5000
    assert np.abs(PX[0] - freq).max() < 1e-6
    assert ro["a_scaled"] == 0.0 and ro["tau"] == float("inf")
    y2 = y.copy()
    y2[y2 == 7] = 8                                    # a class absent from calib keeps a probability above 0
    ro2 = tmcore.fit_readout(rng.normal(0, 10, size=(5000, 65)), y2)
    P2 = tmcore.readout_proba(rng.normal(0, 10, size=(10, 65)), ro2)
    assert P2.min() > 0 and P2[:, 7].max() < 1e-8


@test
def epoch_and_trial_picks_tie_to_the_earlier():
    assert tmcore.pick_epoch([3.0, 2.0, 2.0, 5.0]) == 2 and tmcore.pick_epoch([1.0]) == 1
    assert tmcore.pick_epoch([2.0, 1.0, 0.5, 0.5, 0.7]) == 3
    assert tmcore.pick_trial([5.0, 4.0, 4.0, 6.0]) == 1 and tmcore.pick_trial([3.0]) == 0
    assert tmcore.pick_trial([6.9, 6.8, 6.85, 6.8, 6.9, 6.95, 7.0, 6.81]) == 1


# ---------------------------------------------------------------- C-K1, C-K2, C-K3
@test
def k1_rule():
    assert gates.k1(6.9 + 0.0218, 6.9)["pass"] and not gates.k1(6.9 + 0.0219, 6.9)["pass"]
    assert gates.k1(6.80, 6.90)["pass"] and gates.K1_MAX_GAP == 0.0218


@test
def k2_rules_at_their_boundaries():
    base = {"n_clauses": 8000, "empty": 100, "median_literals": 12.0, "no_positive_literal": 200}
    assert tmcore.k2(base, 4.0, 5.0)["pass"]
    for key, ok, bad, name in (("empty", 4000, 4001, "empty_share"), ("median_literals", 30.0, 30.5, "median_literals"),
                               ("no_positive_literal", 4000, 4001, "no_positive_share")):
        assert tmcore.k2({**base, key: ok}, 4.0, 5.0)["pass"], key
        r = tmcore.k2({**base, key: bad}, 4.0, 5.0)
        assert not r["pass"] and r["stops"] == [name], (key, r["stops"])
    assert tmcore.k2(base, 5.0, 5.0)["pass"], "equal class bits are not worse"
    r = tmcore.k2(base, 5.0 + 1e-9, 5.0)
    assert not r["pass"] and r["stops"] == ["class_bits_vs_unigram"]
    r = tmcore.k2({**base, "empty": 5000, "no_positive_literal": 5000}, 6.0, 5.0)
    assert r["stops"] == ["empty_share", "no_positive_share", "class_bits_vs_unigram"]
    assert abs(r["reported_no_rule"]["empty_or_no_positive_share"] - 1.25) < 1e-12


@test
def k3_plan_projection_and_line():
    R = tmcore.REGISTERED
    plan = tmcore.remaining_plan("tm_smoke", R, 836253, 8363)
    assert plan == [("tm_grid", 8, 20, 836253, 9), ("tm_seeds", 9, 20, 836253, 18), ("tm_c0", 10, 20, 8363, 10)]
    assert tmcore.remaining_plan("tm_grid", R, 836253, 8363) == plan[1:]
    assert tmcore.remaining_plan("tm_c0", R, 836253, 8363) == []
    assert tmcore.remaining_plan(None, R, 836253, 8363) == plan
    assert tmcore.batches_per_epoch(836253) == 6534 and tmcore.batches_per_epoch(128) == 1
    m = tmcore.project_minutes(plan, 0.0015, 1.0, 5.0, 30.0)
    want = (8 * 20 * (6534 * 0.0015 + 1.0) + 9 * 5.0 + 30 + 9 * 20 * (6534 * 0.0015 + 1.0) + 18 * 5.0 + 30
            + 10 * 20 * (66 * 0.0015 + 1.0) + 10 * 5.0 + 30) / 60
    assert abs(m - want) < 1e-9
    assert tmcore.k3(100.0, 130.0)["pass"] and not tmcore.k3(100.0, 130.001)["pass"]
    assert tmcore.k3(0.0, 0.0)["limit_min"] == 230.0


# ---------------------------------------------------------------- subsets, targets, seed means
@test
def subsets_and_shuffled_targets():
    blocks = np.repeat(np.arange(379), 50)
    idx = tmcore.smoke_subset(blocks, tmcore.REGISTERED)
    assert np.array_equal(idx, np.flatnonzero(learners.subset_blocks(blocks, 0.10, 1)))
    assert len(np.unique(blocks[idx])) == 38 and np.array_equal(idx, tmcore.smoke_subset(blocks, tmcore.REGISTERED))
    assert not np.array_equal(idx, np.flatnonzero(learners.subset_blocks(blocks, 0.10, 0))), "seed 1 = seed 0?"
    c0 = tmcore.c0_subset(836253, tmcore.REGISTERED)
    assert len(c0) == 8363 and np.all(np.diff(c0) > 0) and c0.max() < 836253
    assert np.array_equal(c0, tmcore.c0_subset(836253, tmcore.REGISTERED))
    y = np.arange(500) % 65
    ysh = tmcore.shuffled_targets(y)
    assert sorted(ysh.tolist()) == sorted(y.tolist()) and not np.array_equal(ysh, y)
    assert np.array_equal(ysh, learners.permuted(y, seed=1))


@test
def seed_means_average_bits_not_probabilities():
    rng = np.random.default_rng(6)
    lps = [np.log(rng.uniform(0.01, 0.9, size=200)) for _ in range(3)]
    mean, B, per = tmcore.seed_mean_bits(lps)
    assert np.allclose(mean, np.mean([-lp / math.log(2) for lp in lps], axis=0), atol=1e-15)
    assert B.shape == (3, 200) and np.allclose(per, [float(-lp.mean() / math.log(2)) for lp in lps])
    prob_mean = -np.log2(np.mean([np.exp(lp) for lp in lps], axis=0))
    assert not np.allclose(prob_mean, mean), "bits and probabilities gave the same average"
    assert abs(float(mean.mean()) - float(np.mean(per))) < 1e-12


# ---------------------------------------------------------------- systems with a TM class model
@test
def system_score_runs_the_contract_and_stops_on_a_bad_model():
    Z = step9_synth.make_data(seed=7, n_train=3000, n_calib=1200, n_valid=1200, n_pos=200)
    e = Z["emission"]
    sp = {s: tmcore.split(Z, s, e) for s in ("calib", "valid")}
    rng = np.random.default_rng(7)
    PX = {s: rng.dirichlet(np.ones(65), size=len(Z[f"y_{s}"])) for s in ("calib", "valid")}
    lp, lamb, info = tmcore.system_score(PX, sp, Z, e, "synthetic")
    assert lp.shape == (len(Z["y_valid"]),) and len(Z["y_valid"]) >= 1200
    assert lamb.shape == (40, 3) and np.allclose(lamb.sum(1), 1)
    assert info["checks"]["valid"]["contract"]["pass"] and abs(info["valid_bpw"] - systems.bpw(lp)) < 1e-15
    bad = {s: P.copy() for s, P in PX.items()}
    bad["valid"][:, 0] += 0.05
    try:
        tmcore.system_score(bad, sp, Z, e, "broken")
        raise AssertionError("a class model whose rows do not sum to 1 passed")
    except SystemExit as ex:
        assert "contract or all-position check failed" in str(ex)


# ---------------------------------------------------------------- run record, mode, issue 7
def redirected(root):
    saved = ({k: getattr(paths, k) for k in ("DATA", "DATA_DRY", "RUNS", "LISTINGS", "REG", "FROZEN")}, hashing.LOGMD)
    tmcore.redirect_to_test_root(root)
    return saved


def restore(saved):
    for k, v in saved[0].items():
        setattr(paths, k, v)
    hashing.LOGMD = saved[1]


@test
def tm_run_record_allows_only_the_cotm_torch():
    torch()
    root = SCRATCH / "runrec"
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root / "runs")
    saved = redirected(root)
    try:
        assert "torch" in env.neural_modules_loaded()
        r = tmcore.TMRun("tm_selftest", {"x": 1}, dry=True)
        r.finish({"decision": "PASS: self-test", "device": "cpu"})
        meta = json.loads((root / "runs" / r.id / "run.json").read_text())
        assert meta["state"] == "finished" and "torch" in meta["neural_modules_loaded"] and meta["torch_role"]
        lines = (root / "runs" / "gpu_time.tsv").read_text().splitlines()
        assert lines[0].split("\t") == list(tmcore.GPU_TIME_COLS) and lines[1].split("\t")[3] == "stage wall time"
        sys.modules["transformers"] = type(sys)("transformers")   # a stand-in: any other library fails the run
        try:
            r2 = tmcore.TMRun("tm_selftest2", {}, dry=True)
            try:
                r2.finish({"device": "cpu"})
                raise AssertionError("a run with transformers loaded finished")
            except SystemExit as ex:
                assert "transformers" in str(ex)
        finally:
            del sys.modules["transformers"]
        r3 = runlog.Run("cpu_selftest", {}, dry=True)            # the CPU stages keep their no-torch rule
        try:
            r3.finish({})
            raise AssertionError("runlog.Run finished with torch loaded")
        except SystemExit as ex:
            assert "torch" in str(ex)
    finally:
        restore(saved)


@test
def decisions_come_from_the_verified_decision_file():
    root = SCRATCH / "decisions"
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root / "runs" / "20261023T000000Z_tm_smoke")
    os.makedirs(root / "data" / "tm")
    saved = redirected(root)
    try:
        rd = root / "runs" / "20261023T000000Z_tm_smoke"
        dec = root / "data" / "tm" / "smoke_decision.json"
        dec.write_text(json.dumps({"decision": "PASS: x", "K3": {}}))
        (rd / "run.json").write_text(json.dumps({"state": "finished", "wall_s": 3.0}))
        (rd / "metrics.json").write_text(json.dumps({"decision": "PASS: x"}))
        (rd / "outputs.json").write_text(json.dumps(hashing.manifest([dec])))
        d, rec, folder = tmcore.decision_of("tm_smoke", False)
        assert d == "PASS: x" and rec["K3"] == {} and folder == str(rd)
        assert tmcore.require_pass("tm_smoke", False)[0] == "PASS: x"
        assert tmcore.measured_minutes(["tm_smoke"], False)[0] == 3.0 / 60
        for name, meta in (("20261023T000001Z_tm_smoke", {"state": "crashed", "wall_s": 6.0}),
                           ("20261023T000002Z_tm_smoke", {"state": "running"})):
            os.makedirs(root / "runs" / name)
            (root / "runs" / name / "run.json").write_text(json.dumps(meta))
        m, rows = tmcore.measured_minutes(["tm_smoke"], False)
        assert m == 9.0 / 60 and rows["20261023T000002Z_tm_smoke"].startswith("no wall time"), rows
        for name in ("20261023T000001Z_tm_smoke", "20261023T000002Z_tm_smoke"):
            shutil.rmtree(root / "runs" / name)
        (rd / "metrics.json").write_text(json.dumps({"decision": "STOP: y"}))
        try:
            tmcore.decision_of("tm_smoke", False)
            raise AssertionError("a metrics.json that disagrees with the decision file was accepted")
        except SystemExit as ex:
            assert "disagree" in str(ex)
        dec.write_text(json.dumps({"decision": "STOP: y"}))
        try:
            tmcore.decision_of("tm_smoke", False)
            raise AssertionError("a changed decision file was accepted")
        except SystemExit as ex:
            assert "differs from what run" in str(ex)
        (rd / "outputs.json").write_text(json.dumps(hashing.manifest([dec])))
        try:
            tmcore.require_pass("tm_smoke", False)
            raise AssertionError("a STOP was passed on")
        except SystemExit as ex:
            assert "decided 'STOP: y'" in str(ex)
    finally:
        restore(saved)


@test
def test_root_guard_and_stage_rules():
    os.makedirs(SCRATCH, exist_ok=True)
    good = SCRATCH / "root_ok"
    shutil.rmtree(good, ignore_errors=True)
    os.makedirs(good)
    os.environ[tmcore.TEST_ENV] = str(good)
    try:
        try:
            tmcore.test_root_from_env()
            raise AssertionError("a root without its marker was accepted")
        except SystemExit as ex:
            assert "marker" in str(ex)
        (good / tmcore.TEST_MARKER).write_text("x")
        assert tmcore.test_root_from_env() == Path(os.path.realpath(good))
        os.environ[tmcore.TEST_ENV] = str(paths.ROOT / "glassbox-lm")
        try:
            tmcore.test_root_from_env()
            raise AssertionError("a root outside C/tmp was accepted")
        except SystemExit as ex:
            assert "not a folder under" in str(ex)
    finally:
        del os.environ[tmcore.TEST_ENV]
    assert tmcore.test_root_from_env() is None
    ok = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUDA_CACHE_DISABLE="1", C_GPU_RUNNER="1")
    assert tmcore.stage_problems(True, ok) == []
    p = tmcore.stage_problems(False, ok)
    assert len(p) == 1 and "needs CUDA_VISIBLE_DEVICES set to the GPU" in p[0]
    assert tmcore.stage_problems(False, dict(ok, CUDA_VISIBLE_DEVICES="0")) == []
    p = tmcore.stage_problems(True, dict(ok, C_GPU_RUNNER="", CUDA_CACHE_DISABLE="0", CUDA_VISIBLE_DEVICES="0"))
    assert len(p) == 3, p
    saved = redirected(good)
    try:
        assert paths.DATA == good / "data" and paths.RUNS == good / "runs" and paths.FROZEN.parent == good
        assert hashing.LOGMD == good / "LOG.md" and paths.C == Path(f"{_REPO_ROOT}/studies/language_part3/C")
    finally:
        restore(saved)
    assert paths.FROZEN == paths.C / "R1_C-core.frozen.json" and hashing.LOGMD == paths.TMLANG / "LOG.md"
    assert tmcore.exit_code("STOP: x") == 3 and tmcore.exit_code("HOLD at C-K3") == 3 and tmcore.exit_code("PASS") == 0


@test
def issue7_freeze_amend_and_frozen_record_checks():
    import importlib.util
    root = SCRATCH / "freeze"
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root / "runs" / "20261023T000000Z_prep")
    (root / "runs" / "20261023T000000Z_prep" / "run.json").write_text(json.dumps({"state": "finished"}))
    os.makedirs(root / "runs" / "20261023T000001Z_prep_dry")
    (root / "runs" / "20261023T000001Z_prep_dry" / "run.json").write_text(json.dumps({"state": "finished"}))
    (root / "R1_C-core.md").write_text("stand-in registration\n")
    saved = redirected(root)
    try:
        spec = importlib.util.spec_from_file_location("c_freeze_unit", paths.C / "scripts" / "freeze.py")
        fr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fr)
        try:
            fr.main(["--amend", "x"])
            raise AssertionError("an amendment without a record was accepted")
        except SystemExit as ex:
            assert "nothing to amend" in str(ex)
        fr.main([])
        rec1 = json.loads(paths.FROZEN.read_text())
        h1 = hashing.sha256(paths.FROZEN)
        assert "amendment" not in rec1 and rec1["files"] == hashing.snapshot()
        assert any(k.endswith("src/dl/tm.py") for k in rec1["files"]) and \
            not any(k.endswith("src/dl/tm.py") for k in rec1["reference_files_not_enforced"])
        try:
            fr.main([])
            raise AssertionError("the record was overwritten without --amend")
        except SystemExit as ex:
            assert "already frozen" in str(ex)
        try:
            fr.main(["--amend", "  "])
            raise AssertionError("an empty reason was accepted")
        except SystemExit as ex:
            assert "needs a reason" in str(ex)
        # a real stage needs the LOG.md line with the record's hash
        try:
            hashing.check_frozen(require=True)
            raise AssertionError("a record without its LOG.md line was accepted")
        except SystemExit as ex:
            assert "has no line with the frozen record's SHA-256" in str(ex)
        hashing.LOGMD.write_text(f"- R1 [C-core] stand-in {h1}\n")
        rec, new = hashing.check_frozen(require=True)
        assert new == [] and rec["frozen_utc"] == rec1["frozen_utc"]
        assert hashing.check_frozen(require=False)[1] == []
        # a tracked file that is not in the record: refused by a real stage, listed by a dry one
        drop = next(k for k in rec1["files"] if k.endswith("ccore/tmcore.py"))
        rec_cut = dict(rec1, files={k: v for k, v in rec1["files"].items() if k != drop})
        paths.FROZEN.write_text(json.dumps(rec_cut, indent=1, sort_keys=True))
        hashing.LOGMD.write_text(f"- stand-in {hashing.sha256(paths.FROZEN)}\n")
        try:
            hashing.check_frozen(require=True)
            raise AssertionError("a file outside the record was accepted")
        except SystemExit as ex:
            assert "not in the frozen record" in str(ex) and drop in str(ex)
        assert hashing.check_frozen(require=False)[1] == [drop]
        # a changed tracked file stops real and dry stages alike (unchanged rule)
        rec_bad = dict(rec1, files=dict(rec1["files"], **{drop: "0" * 64}))
        paths.FROZEN.write_text(json.dumps(rec_bad, indent=1, sort_keys=True))
        for req in (True, False):
            try:
                hashing.check_frozen(require=req)
                raise AssertionError("a changed file was accepted")
            except SystemExit as ex:
                assert "frozen files changed" in str(ex)
        # --amend keeps the old record byte for byte and writes a new one
        paths.FROZEN.write_text(json.dumps(rec1, indent=1, sort_keys=True))
        old_bytes = paths.FROZEN.read_bytes()
        h_old = hashing.sha256(paths.FROZEN)
        fr.main(["--amend", "self-test amendment"])
        rec2 = json.loads(paths.FROZEN.read_text())
        copy = root / "history" / "R1_C-core.frozen.v1.json"
        assert copy.read_bytes() == old_bytes
        assert rec2["amendment"] == "self-test amendment" and rec2["amendment_number"] == 1
        assert rec2["previous_record"]["sha256"] == h_old and rec2["previous_record"]["copy"] == hashing.rel(copy)
        assert rec2["real_runs_at_freeze"] == ["20261023T000000Z_prep"] and rec2["files"] == rec1["files"]
        h2 = hashing.sha256(paths.FROZEN)
        hashing.LOGMD.write_text(f"- only the old record {h_old}\n")
        try:
            hashing.check_frozen(require=True)
            raise AssertionError("the amended record ran on the old LOG.md line")
        except SystemExit:
            pass
        hashing.LOGMD.write_text(f"- old {h_old}\n- AMENDMENT 1 {h2}\n")
        assert hashing.check_frozen(require=True)[1] == []
        fr.main(["--amend", "second"])
        assert hashing.sha256(root / "history" / "R1_C-core.frozen.v2.json") == h2
        assert json.loads(paths.FROZEN.read_text())["amendment_number"] == 2
    finally:
        restore(saved)
    assert paths.FROZEN == paths.C / "R1_C-core.frozen.json"


@test
def no_test_file_reading_and_no_registered_number_in_the_new_code():
    new = ["ccore/tmcore.py", "ccore/tmtorch.py", "scripts/tm_smoke.py", "scripts/tm_grid.py", "scripts/tm_seeds.py",
           "scripts/tm_c0.py", "scripts/run_gpu_stage.sh", "tests/step9_synth.py", "tests/selftest_step9_e2e.py",
           "tests/run_step9_tests.sh"]
    for f in new:
        s = (paths.C / f).read_text()
        for bad in ("test" + "/text.utf8", '"test", ' + '"text.utf8"', "PTB_" + "TEST_FILE"):   # split: the older
            assert bad not in s, (f, bad)                                                     # self-test greps code
    s = (paths.C / "scripts" / "run_gpu_stage.sh").read_text()
    assert "touch $GO" not in s and ".gpu_go" in s and "> \"$GO\"" not in s


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
    print(f"selftest_step9: {len(RESULTS) - fails} of {len(RESULTS)} passed in {time.time() - t0:.1f} s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
