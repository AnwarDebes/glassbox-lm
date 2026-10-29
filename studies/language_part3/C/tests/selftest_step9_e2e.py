"""End-to-end smoke of C-core's step-9 code on tiny synthetic data, on the CPU with the GPU hidden, through the GPU
entry point scripts/run_gpu_stage.sh in self-test mode: test roots under C/tmp/step9_build/e2e_<UTC>/ hold a fake
go file, a fake GPU process list, a stand-in registration, a frozen record made by scripts/freeze.py and a LOG.md line
with its hash. Nothing under C/data, C/data_dry or C/runs is read or written (tests/run_step9_tests.sh checks it).

    bash tests/run_step9_tests.sh            # from C/: runs this file with the unit tests and the older self-tests

Covers: the refusals of the runner (no go file, a GPU compute process, an unreadable process list, C's lock busy,
usage) and of the stages (no runner mark, no LOG.md line, wrong stage order, a second real run); the four stages in
order (tm_smoke, tm_grid, tm_seeds, tm_c0) with their decisions, records, GPU-time rows and outputs; an independent
recompute of the pick's system from its stored state through dl/cpu_tm_fast; a K1 STOP that blocks tm_seeds; and the
four stages with --dry. Prints PASS or FAIL per test and exits 1 on any failure."""
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccore import env  # noqa: E402

env.guard()
import numpy as np  # noqa: E402

from ccore import gates, hashing, learners, paths, systems, tmcore, tmtorch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step9_synth  # noqa: E402

RESULTS = []
STAMP = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
BASE = paths.TMP / "step9_build" / f"e2e_{STAMP}"
RUNNER = paths.C / "scripts" / "run_gpu_stage.sh"
A = BASE / "pass"          # twin = F: C-K1 passes
B = BASE / "k1stop"        # a twin far better than any TM: C-K1 stops
STATE = {}


def test(fn):
    RESULTS.append(fn)
    return fn


def run(root, stage, *args, direct=False, env_extra=None):
    e = dict(os.environ, C_STEP9_TEST_ROOT=str(root))
    e.pop("C_GPU_RUNNER", None)
    if env_extra:
        e.update(env_extra)
    if direct:
        cmd = [sys.executable, "-I", "-B", str(paths.C / "scripts" / f"{stage}.py"), *args]
    else:
        cmd = ["bash", str(RUNNER), stage, *args]
    p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def runs_of(root, name):
    return sorted(d.name for d in (root / "runs").glob(f"*_{name}") if d.is_dir())


def state_of(root, run_name):
    return json.loads((root / "runs" / run_name / "run.json").read_text())["state"]


def expect(rc, out, want_rc, *phrases):
    assert rc == want_rc, f"exit {rc}, wanted {want_rc}:\n{out[-3000:]}"
    for ph in phrases:
        assert ph in out, f"{ph!r} not in the output:\n{out[-3000:]}"


# ---------------------------------------------------------------- setup
@test
def build_the_synthetic_roots_and_freeze_them():
    os.makedirs(BASE)
    STATE["Z"] = step9_synth.build_tree(A, twin_gain_nats=0.0)
    step9_synth.build_tree(B, twin_gain_nats=3.0)
    for r in (A, B):
        h = step9_synth.freeze_into(r)
        rec = json.loads((r / "R1_C-core.frozen.json").read_text())
        assert hashing.sha256(r / "R1_C-core.frozen.json") == h and h in (r / "LOG.md").read_text()
        assert rec["files"] == hashing.snapshot(), "the test root's record is not the current code"
    assert not (paths.C / ".gpu_go").exists(), "a real go file exists; this test never creates one"


# ---------------------------------------------------------------- refusals before any stage ran
@test
def refusals_of_the_runner():
    rc, out = run(A, "bogus")
    expect(rc, out, 2, "unknown GPU stage")
    rc, out = run(A, "tm_smoke", "--bogus")
    expect(rc, out, 2, "unknown argument")
    (A / ".gpu_go").rename(A / "gpu_go.off")
    try:
        rc, out = run(A, "tm_smoke")
        expect(rc, out, 2, "no go file")
    finally:
        (A / "gpu_go.off").rename(A / ".gpu_go")
    (A / "gpu_busy").write_text("")
    try:
        rc, out = run(A, "tm_smoke")
        expect(rc, out, 75, "GPU compute processes present", "424242")
    finally:
        (A / "gpu_busy").unlink()
    (A / "gpu_broken").write_text("")
    try:
        rc, out = run(A, "tm_smoke")
        expect(rc, out, 2, "cannot read the GPU compute-process list")
    finally:
        (A / "gpu_broken").unlink()
    with open(A / "tmp" / "c_job.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc, out = run(A, "tm_smoke")
        expect(rc, out, 75, "another C job holds")
    assert runs_of(A, "tm_smoke") == [], "a refused start left a run folder"
    rc, out = run(BASE / "nowhere", "tm_smoke")
    expect(rc, out, 2, "no test root")


@test
def refusals_of_the_stages():
    rc, out = run(A, "tm_smoke", direct=True)
    expect(rc, out, 1, "not started by scripts/run_gpu_stage.sh")
    (A / "LOG.md").rename(A / "LOG.off")
    try:
        rc, out = run(A, "tm_smoke")
        expect(rc, out, 1, "has no line with the frozen record's SHA-256")
    finally:
        (A / "LOG.off").rename(A / "LOG.md")
    assert runs_of(A, "tm_smoke") == []
    rc, out = run(A, "tm_grid")
    expect(rc, out, 1, "expected exactly one finished tm_smoke run")
    g = runs_of(A, "tm_grid")
    assert len(g) == 1 and state_of(A, g[0]) == "crashed", g
    shutil.rmtree(A / "runs" / g[0])


# ---------------------------------------------------------------- the four stages, real mode in the test root
def stage_record(root, stage):
    rs = [r for r in runs_of(root, stage) if state_of(root, r) == "finished"]
    assert len(rs) == 1, rs
    d = root / "runs" / rs[0]
    return d, json.loads((d / "metrics.json").read_text()), json.loads((d / "outputs.json").read_text())


@test
def stage_tm_smoke():
    rc, out = run(A, "tm_smoke")
    expect(rc, out, 0, "DECISION: PASS", "listing before/after: no change", "timed CPU epoch")
    d, m, o = stage_record(A, "tm_smoke")
    assert m["decision"].startswith("PASS") and m["K2"]["pass"] and m["shuffled"]["pass"] and m["K3"]["pass"]
    assert set(m["K2"]["checks"]) == {"empty_share", "median_literals", "no_positive_share", "class_bits_vs_unigram"}
    sm = tmcore.TEST.smoke
    Z = STATE["Z"]
    idx = np.flatnonzero(learners.subset_blocks(Z["blocks_train"], sm["share"], sm["subset_seed"]))
    z = np.load(A / "data" / "tm" / "smoke.npz")
    assert np.array_equal(z["idx"], idx) and m["smoke"]["positions"] == len(idx)
    assert np.array_equal(np.sort(z["ysh"]), np.sort(Z["y_train"][idx])) and not np.array_equal(z["ysh"],
                                                                                                 Z["y_train"][idx])
    hist = [h["calib_class_bits"] for h in m["smoke"]["history"]]
    assert len(hist) == sm["epochs"] and m["smoke"]["epoch_kept"] == tmcore.pick_epoch(hist)
    assert m["smoke"]["valid_class_bits"] < m["smoke"]["class_unigram_valid_class_bits"]
    assert m["smoke"]["cfg"]["n_clauses"] == tmcore.TEST.clauses and m["smoke"]["cfg"]["T"] == sm["T"]
    assert m["cpu_epoch"]["batches"] == math.ceil(len(idx) / 128) and m["cpu_epoch"]["torch_threads"] == 2
    assert m["shuffled"]["diff_vs_F_bpw"] >= gates.SHUFFLED_MIN_DIFF
    assert {Path(k).name for k in o} == {"smoke.npz", "smoke_decision.json"}
    for k, h in o.items():
        assert hashing.sha256(paths.ROOT / k) == h
    meta = json.loads((d / "run.json").read_text())
    assert "torch" in meta["neural_modules_loaded"] and meta["dry"] is False
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["mode"]["profile"] == "test" and cfg["mode"]["device"] == "cpu" and cfg["new_files_since_freeze"] == []


@test
def stage_tm_grid():
    rc, out = run(A, "tm_grid")
    expect(rc, out, 0, "DECISION: PASS: C-K1 passed", "listing before/after: no change")
    d, m, o = stage_record(A, "tm_grid")
    trials = m["trials"]
    assert [(t["T"], t["s"]) for t in trials] == [tuple(g) for g in tmcore.TEST.grid]
    bpws = [t["valid_bpw"] for t in trials]
    assert m["pick"] == min(range(8), key=lambda i: (bpws[i], i))
    k1 = gates.k1(bpws[m["pick"]], m["twin_bpw"])
    assert m["K1"] == k1 and k1["pass"]
    for i, t in enumerate(trials):
        z = np.load(A / "data" / "tm" / f"grid_t{i}.npz")
        hist = list(z["calib_class_bits_per_epoch"])
        assert len(hist) == tmcore.TEST.max_epochs and int(z["epoch_best"]) == tmcore.pick_epoch(hist)
        assert z["ta_best"].shape == (tmcore.TEST.clauses, 120) and z["w_best"].shape == (tmcore.TEST.clauses, 65)
        assert ("lp_valid_ep1" in z.files) == (i == m["pick"]), (i, z.files)
        assert abs(systems.bpw(z["lp_valid_best"]) - t["valid_bpw"]) < 1e-12
        assert all(v["equal"] for v in t["packed_checks"].values())
    assert m["epoch1_of_pick"] is not None and m["epoch1_of_pick"]["epoch"] == 1
    assert len(o) == 9


@test
def independent_recompute_of_the_pick_from_its_stored_state():
    """The pick's valid system bpw again, from its stored automata and weights through dl/cpu_tm_fast (integer class
    sums on the CPU, not the CoTM), its read-out refitted on calib, and a fresh system fit."""
    _, m, _ = stage_record(A, "tm_grid")
    Z = STATE["Z"]
    z = np.load(A / "data" / "tm" / f"grid_t{m['pick']}.npz")
    fast = tmtorch.fast()
    model = fast.pack(z["ta_best"], z["w_best"], n_states=256, negation=True)
    S = {s: fast.class_sums(model, Z[f"X_{s}"]) for s in ("calib", "valid")}
    ro = tmcore.fit_readout(S["calib"], Z["y_calib"])
    assert abs(ro["a_scaled"] - float(z["ro_best_a_scaled"])) < 1e-9 and np.allclose(ro["b"], z["ro_best_b"],
                                                                                      atol=1e-9)
    PX = {s: tmcore.readout_proba(S[s], ro) for s in S}
    e = Z["emission"]
    sp = {s: tmcore.split(Z, s, e) for s in S}
    _, lamb, _ = systems.fit_system(PX["calib"], sp["calib"])
    lp = systems.logp(PX["valid"], sp["valid"], lamb)
    rec = m["trials"][m["pick"]]["valid_bpw"]
    assert abs(systems.bpw(lp) - rec) < 1e-9, (systems.bpw(lp), rec)
    assert np.allclose(lp, z["lp_valid_best"], atol=1e-9)


@test
def stage_tm_seeds():
    rc, out = run(A, "tm_seeds")
    expect(rc, out, 0, "DECISION: PASS: seeds 2-10 trained", "listing before/after: no change")
    _, m, o = stage_record(A, "tm_seeds")
    _, g, _ = stage_record(A, "tm_grid")
    assert [r["seed"] for r in m["seeds"]] == [2, 3] and m["pick_setting"] == g["pick_setting"]
    for r in m["seeds"]:
        z = np.load(A / "data" / "tm" / f"seed{r['seed']}.npz")
        assert int(z["seed"]) == r["seed"] and "lp_valid_ep1" in z.files
        assert abs(systems.bpw(z["lp_valid_best"]) - r["valid_bpw"]) < 1e-12
        assert abs(systems.bpw(z["lp_valid_ep1"]) - r["epoch1"]["valid_bpw"]) < 1e-12
        assert int(z["T"]) == g["pick_setting"]["T"] and float(z["s"]) == g["pick_setting"]["s"]
    assert len(o) == 3


@test
def stage_tm_c0_and_the_seed_means():
    rc, out = run(A, "tm_c0")
    expect(rc, out, 0, "DECISION: PASS: step 9 complete", "listing before/after: no change")
    _, m, o = stage_record(A, "tm_c0")
    _, g, _ = stage_record(A, "tm_grid")
    Z = STATE["Z"]
    assert m["c0_subset_positions"] == int(round(0.01 * len(Z["y_train"])))
    assert [r["seed"] for r in m["c0"]] == list(tmcore.TEST.c0_seeds)
    tm_lp = [np.load(A / "data" / "tm" / f"grid_t{g['pick']}.npz")["lp_valid_best"]] + \
        [np.load(A / "data" / "tm" / f"seed{k}.npz")["lp_valid_best"] for k in (2, 3)]
    want = float(np.mean([-lp / math.log(2) for lp in tm_lp], axis=0).mean())
    assert abs(m["arms"]["TM"]["seed_mean_bpw"] - want) < 1e-12
    assert abs(m["arms"]["TM"]["seed_mean_bpw"] - float(np.mean(m["arms"]["TM"]["per_seed_bpw"]))) < 1e-12
    v = np.load(A / "data" / "tm" / "step9_valid.npz")
    assert v["tm_bits"].shape == (3, len(Z["y_valid"])) and v["c0_bits"].shape == (2, len(Z["y_valid"]))
    assert abs(float(v["tm_bits"].mean(0).mean()) - want) < 1e-12
    assert abs(m["differences_bpw"]["TM - twin"] - (want - systems.bpw(np.log(Z["pF_valid"])))) < 1e-12
    for k in ("TM - twin", "TM - C0", "TM - epoch1", "TM - F"):
        iv = m["seed_inclusive_intervals_bits_no_rule"][k]
        assert iv["lo"] <= iv["hi"] and iv["n_boot"] == 1000
    assert len(o) == 4
    rows = (A / "runs" / "gpu_time.tsv").read_text().splitlines()
    assert rows[0].split("\t") == list(tmcore.GPU_TIME_COLS)
    what = [r.split("\t")[3] for r in rows[1:]]
    assert what.count("stage wall time") == 4 and what.count("smoke TM") == 1 and what.count("timed CPU epoch") == 1
    assert sum(w.startswith("grid trial") for w in what) == 8 and what.count("seed 2") == 1
    assert sum(w.startswith("C0 seed") for w in what) == 2 and len(what) == 19, what
    jobs = (A / "runs" / "jobs_log.tsv").read_text().splitlines()
    done = [j.split("\t") for j in jobs if j.split("\t")[4] == "0"]
    assert [j[2] for j in done] == ["tm_smoke", "tm_grid", "tm_seeds", "tm_c0"], jobs


@test
def a_second_real_run_is_refused():
    rc, out = run(A, "tm_smoke")
    expect(rc, out, 1, "a finished real tm_smoke run exists")


# ---------------------------------------------------------------- C-K1 STOP blocks the seeds
@test
def k1_stop_blocks_the_seeds():
    rc, out = run(B, "tm_smoke")
    expect(rc, out, 0, "DECISION: PASS")
    rc, out = run(B, "tm_grid")
    expect(rc, out, 3, "DECISION: STOP: FAIL at K1 (exploratory)", "no test read")
    _, m, _ = stage_record(B, "tm_grid")
    assert not m["K1"]["pass"] and m["epoch1_of_pick"] is None
    assert "lp_valid_ep1" not in np.load(B / "data" / "tm" / f"grid_t{m['pick']}.npz").files
    rc, out = run(B, "tm_seeds")
    expect(rc, out, 1, "tm_grid decided 'STOP: FAIL at K1")
    rc, out = run(B, "tm_c0")
    expect(rc, out, 1, "tm_grid decided 'STOP: FAIL at K1")


# ---------------------------------------------------------------- the dry path
@test
def the_four_stages_with_dry():
    for st in tmcore.STAGES:
        rc, out = run(A, st, "--dry")
        assert rc in (0, 3), f"{st} --dry exit {rc}:\n{out[-3000:]}"
        rs = [r for r in runs_of(A, f"{st}_dry") if state_of(A, r) == "finished"]
        assert len(rs) == 1, (st, rs)
        cfg = json.loads((A / "runs" / rs[0] / "config.json").read_text())
        assert cfg["dry"] is True and cfg["mode"]["profile"] == "test"
    names = sorted(p.name for p in (A / "data_dry" / "tm").iterdir())
    assert "c0_decision.json" in names and "step9_valid.npz" in names and "grid_t7.npz" in names, names


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
            if fn is build_the_synthetic_roots_and_freeze_them:
                break
    print(f"test roots: {BASE}")
    print(f"selftest_step9_e2e: {len(RESULTS) - fails} of {len(RESULTS)} passed in {time.time() - t0:.1f} s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
