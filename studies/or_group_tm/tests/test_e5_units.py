"""Unit tests of the E5 guards and helpers: src/e5_common.py (record check, run arguments, pinned files, stock
folders, environment, run log), the refusals of src/run_wiki.py and src/wiki_report.py (before any pair is read), the
files that a run opens, the intervals of src/wiki_report.py on small made-up data, and the checks, rules, sentences,
failure paths and helpers of src/e5_report.py on small made-up run folders, including its registered-only branches
(plain script, no pytest). Every check has a case in which it must fail.

Run from the study folder:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -I tests/test_e5_units.py
Prints one line per test and 'ALL n TESTS PASSED', or stops with exit status 1 at the first failure. The tests that
need the Wikipedia data of the hierarchical_tm study (not distributed) are skipped when it is missing. Scratch files
go to a temporary folder that is removed at the end; nothing in the study folder is written, and no test pair is
read (the files of the test split are hashed, never loaded; the record of the registration is never handed to a
runner).
"""
import argparse
import contextlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True

import e5_common as e5
import e5_report
import run_wiki
import wiki_report
from metrics import auc

HAVE_DATA = os.path.isfile(os.path.join(ROOT, e5.H, "data", "sa", "train.npz"))
TMP = tempfile.mkdtemp(prefix="e5_units_")
PASSED, SKIPPED = [], []
OPENED = None  # a list while the audit hook below records the files that are opened


def _audit(event, args):
    if OPENED is not None and event == "open" and isinstance(args[0], str):
        OPENED.append(os.path.abspath(args[0]))


sys.addaudithook(_audit)


def test(fn):
    fn()
    PASSED.append(fn.__name__)
    print("ok  ", fn.__name__, flush=True)
    return fn


def needs_data(fn):
    if HAVE_DATA:
        return test(fn)
    SKIPPED.append(fn.__name__)
    print("skip", fn.__name__, "(the Wikipedia pairs are not distributed)", flush=True)
    return fn


def refused(fn, argv):
    """The message of the refusal, or '' when the call was not refused."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            fn(argv)
    except SystemExit as e:
        return err.getvalue() if e.code not in (0, None) else ""
    return ""


class Scratch:
    """A small copy of the layout that e5_common checks: src/, tests/, the plan, the pinned files and a record."""

    def __init__(self, dry=False):
        self.dry = dry
        self.root = tempfile.mkdtemp(prefix="study_", dir=TMP)
        self.saved = (e5.ROOT, e5.REQUIRED_EXTRAS)
        for rel, text in (("src/a.py", "A = 1\n"), ("tests/t.py", "T = 1\n"), (e5.PLAN[dry], "the plan\n"),
                          ("pinned/data.txt", "data\n"), ("pinned/job.sh", "job\n")):
            self.write(rel, text)
        e5.ROOT = self.root
        e5.REQUIRED_EXTRAS = {False: ("pinned/data.txt", "pinned/job.sh"), True: ("pinned/data.txt", "pinned/job.sh")}
        self.record = e5.record_path(dry)
        self.rec = {"plan": e5.PLAN[dry], "plan_sha256": e5.sha(os.path.join(self.root, e5.PLAN[dry])),
                    "frozen_utc": "2026-11-12T00:00:00Z", "sources_sha256": e5.source_hashes(),
                    "extras_sha256": {k: e5.sha(os.path.join(self.root, k)) for k in e5.REQUIRED_EXTRAS[dry]}}
        self.save(self.rec)

    def write(self, rel, text):
        p = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)

    def save(self, rec, path=None):
        with open(path or self.record, "w") as f:
            json.dump(rec, f)

    def problems(self, path=None):
        return e5.preflight(path or self.record, self.dry)[1]

    def close(self):
        e5.ROOT, e5.REQUIRED_EXTRAS = self.saved


def args(dry=False, **kw):
    seed = e5.SEEDS[dry][0]
    a = argparse.Namespace(**e5.SETTING, data="sa", seed=seed, name=e5.run_name("sa", seed, dry), split=e5.SPLIT[dry],
                           no_log=False, out_root=os.path.join(e5.ROOT, e5.RUNS_ROOT[dry]))
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def good_env():
    return {"python": platform.python_version(), "executable": e5.EXECUTABLE, "packages": e5.requirements()}


@test
def preflight_passes_and_fails_as_it_should():
    s = Scratch()
    try:
        assert s.problems() == [], s.problems()
        assert e5.preflight(os.path.join(s.root, "nothing.json"))[0] is None
        assert any("no record file" in p for p in e5.preflight("")[1])
        s.write("bad.json", "not json")
        assert any("not readable" in p for p in e5.preflight(os.path.join(s.root, "bad.json"))[1])
        other = os.path.join(s.root, "registrations", "copy.frozen.json")
        s.save(s.rec, other)
        assert any("the record is not at" in p for p in s.problems(other)), "a copy at another place is not the record"
        s.save({**s.rec, "plan": "registrations/E4_wikipedia_design.md"})
        assert any("the record is of" in p for p in s.problems()), "the record of another plan"
        s.save(s.rec)
        s.write(e5.PLAN[False], "the plan, edited\n")
        assert "the plan text differs from the record" in s.problems()
        s.write(e5.PLAN[False], "the plan\n")
        assert s.problems() == []
        s.write("src/a.py", "A = 2\n")
        assert "source differs from the record: src/a.py" in s.problems()
        s.write("src/a.py", "A = 1\n")
        s.write("src/new.py", "N = 1\n")
        assert "source differs from the record: src/new.py" in s.problems(), "a new source file"
        os.remove(os.path.join(s.root, "src", "new.py"))
        os.remove(os.path.join(s.root, "tests", "t.py"))
        assert "source differs from the record: tests/t.py" in s.problems(), "a removed source file"
        s.write("tests/t.py", "T = 1\n")
        assert s.problems() == []
        s.save({**s.rec, "extras_sha256": {"pinned/data.txt": s.rec["extras_sha256"]["pinned/data.txt"]}})
        assert "required file not pinned in the record: pinned/job.sh" in s.problems()
        s.save({**s.rec, "extras_sha256": {}})
        assert len([p for p in s.problems() if "not pinned" in p]) == 2, "an empty list of pinned files must not pass"
        s.save(s.rec)
        s.write("pinned/data.txt", "other data\n")
        assert "pinned file differs from the record: pinned/data.txt" in s.problems()
        s.write("pinned/data.txt", "data\n")
        os.remove(os.path.join(s.root, "pinned", "job.sh"))
        assert "pinned file differs from the record: pinned/job.sh" in s.problems(), "a pinned file that is gone"
        s.write("pinned/job.sh", "job\n")
        s.write("more.txt", "x\n")
        s.save({**s.rec, "extras_sha256": {**s.rec["extras_sha256"], "more.txt": e5.sha(os.path.join(s.root, "more.txt"))}})
        assert s.problems() == []
        s.write("more.txt", "y\n")
        assert "pinned file differs from the record: more.txt" in s.problems(), "a further pinned file is checked too"
    finally:
        s.close()


@test
def dry_and_registered_records_are_not_interchangeable():
    s = Scratch(dry=True)
    try:
        assert s.problems() == [] and e5.is_dry_record(s.record)
        assert e5.preflight(s.record, dry=False)[1], "the dry record is not a record of the registration"
        assert not e5.is_dry_record(e5.record_path(False))
    finally:
        s.close()


@test
def run_arguments_are_checked():
    s = Scratch()
    try:
        assert e5.check_run_args(args()) == []
        assert e5.check_run_args(args(data="ns_all", seed=15, name="e5_ns_all_s15")) == []
        wrong = {"F": 3072, "clauses": 4000, "K": 2, "T": 100.0, "mode": "near", "init_state": -1, "seed_state": 1,
                 "r": 4, "p_rec": 0.5, "p_rep": 0.5, "q_seed": 0.25, "max_seed": 0, "reject": "random",
                 "reinforce": "critical", "p_decay": 0.1, "seed_all": True, "tied": False, "wmax": 0, "neg_frac": 0.5,
                 "smin": -16, "smax": 15, "passes": 30, "eval_every": 1, "n_train": 300, "n_dev": 200, "top": 10}
        assert set(wrong) == set(e5.SETTING), "a wrong value for every registered setting"
        for k, v in wrong.items():
            assert any(f"setting {k} " in p for p in e5.check_run_args(args(**{k: v}))), k
        assert any("setting tied " in p for p in e5.check_run_args(args(tied=1))), "1 is not True"
        assert any("setting K " in p for p in e5.check_run_args(args(K=True))), "True is not 1"
        assert e5.check_run_args(args(T=40)) == [], "40 and 40.0 are the same T"
        assert any("not a registered data version" in p for p in e5.check_run_args(args(data="ns", name="e5_ns_s11")))
        for seed in (1, 10, 16):
            assert any("not a registered seed" in p for p in e5.check_run_args(args(seed=seed, name=f"e5_sa_s{seed}"))), seed
        assert any("name must be e5_sa_s11" in p for p in e5.check_run_args(args(name="e5_sa_s12")))
        assert any("name must be" in p for p in e5.check_run_args(args(data="np"))), "the name holds the data version"
        assert any("split must be test" in p for p in e5.check_run_args(args(split="dev")))
        assert e5.check_run_args(args(out_root=os.path.join(s.root, "elsewhere")))
        assert e5.check_run_args(args(no_log=True))
        os.makedirs(os.path.join(s.root, "runs", "20261112T000000Z_e5_sa_s11"))
        assert any("exists already" in p for p in e5.check_run_args(args()))
        assert e5.check_run_args(args(seed=12, name="e5_sa_s12")) == []
        assert e5.check_run_args(args(data="np", name="e5_np_s11")) == [], "another data version is another run"
        os.rename(os.path.join(s.root, "runs", "20261112T000000Z_e5_sa_s11"),
                  os.path.join(s.root, "runs", "20261112T000000Z_e5_sa_s11_CRASHED"))
        assert e5.check_run_args(args()) == [], "a crashed folder that was renamed does not block the rerun"
        dry = args(dry=True, out_root=os.path.join(s.root, "runs_dry"))
        assert dry.name == "e5dry_sa_s1" and dry.split == "dev" and e5.check_run_args(dry, dry=True) == []
        assert e5.check_run_args(dry, dry=False) and e5.check_run_args(args(), dry=True)
        assert any("not a registered seed" in p for p in e5.check_run_args(args(dry=True, seed=2, name="e5dry_sa_s2"), dry=True))
    finally:
        s.close()


@test
def the_lists_of_pinned_files_and_the_job():
    for dry in (True, False):
        x = e5.REQUIRED_EXTRAS[dry]
        assert len(x) == len(set(x)) and "registrations/E5_job.sh" in x and "env/requirements.txt" in x
        assert all(f"{e5.H}/data/{d}/train.npz" in x and f"{e5.H}/data/{d}/{e5.SPLIT[dry]}.npz" in x for d in e5.DATA)
        assert len([p for p in x if p.endswith(f"scores_{e5.SPLIT[dry]}.npz")]) == 15
        assert len([p for p in x if p.endswith("include_bits.npz")]) == 15
        assert {e5.DUPLICATES, e5.CPU_LOAD, e5.RUNLOG_PREFIX[dry], *e5.E3_SOURCES} <= set(x)
        other = e5.SPLIT[not dry]
        assert not [p for p in x if p.endswith((f"/{other}.npz", f"scores_{other}.npz"))], (dry, "other split")
    assert "logs/e5_dry_report.json" in e5.REQUIRED_EXTRAS[False] and "logs/e5_dry_report.json" not in e5.REQUIRED_EXTRAS[True]
    assert e5.FINAL_TEST_REPORT in e5.REQUIRED_EXTRAS[False] and e5.DEV_CHECK_REPORT in e5.REQUIRED_EXTRAS[True]
    for dry, flag in ((True, ["--dry"]), (False, [])):
        r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "src", "e5_common.py"), "--extras", *flag],
                           capture_output=True, text=True, check=True)
        assert tuple(r.stdout.split("\n")[:-1]) == e5.REQUIRED_EXTRAS[dry], "the job pins what the check requires"
    assert e5.SETTING["eval_every"] == e5.SETTING["passes"], "the pairs of the split are scored after the last pass only"
    with open(os.path.join(ROOT, "registrations", "E5_job.sh")) as f:
        job = f.read()
    for k, v in e5.SETTING.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            assert (f" {flag} " in job) == v, k
        else:
            assert f" {flag} {v:g} " in job if isinstance(v, float) else f" {flag} {v} " in job, k
    assert "E5_WAIT" not in job and "E5_PYTHON" not in job and "PY=.venv/bin/python" in job, "no setting from the environment"
    assert f"CPU_LOAD={e5.CPU_LOAD}" in job and "wait_cpu" in job, "the CPU rule is in the job"
    assert 'for f in config.json metrics.json state.npz groups.txt "scores_$SPLIT.npz"' in job, "skip test = report test"
    assert e5.RUN_FILES == ("config.json", "metrics.json", "state.npz", "groups.txt")
    assert e5.RULES == {"R1-sa": ("sa", "stock"), "R1-np": ("np", "stock"), "R1-ns_all": ("ns_all", "stock")}
    assert e5.PRIMARY_READING == "unit" and e5.RULE_INTERVALS == ("ci95", "ci95_seeds_and_articles")


@needs_data
def pinned_files_exist_and_are_the_study_files():
    made_by_job = {"logs/e5_dry_report.json", *e5.RUNLOG_PREFIX.values()}
    for dry in (True, False):
        missing = [p for p in e5.REQUIRED_EXTRAS[dry] if p not in made_by_job and not os.path.isfile(os.path.join(ROOT, p))]
        assert not missing, missing
    with open(os.path.join(ROOT, e5.FINAL_TEST_REPORT)) as f:
        rep = json.load(f)
    inputs = rep["inputs_sha256"]
    for d in e5.DATA:
        for sp in ("train", "test"):
            assert e5.sha(os.path.join(ROOT, e5.H, "data", d, f"{sp}.npz")) == inputs[f"data/{d}/{sp}.npz"], (d, sp)
        g = rep["groups"][d]
        assert [s["run"] for s in g["seeds"]] == list(e5.STOCK_TEST[d]), d
        assert [s["source_run"] for s in g["seeds"]] == list(e5.STOCK_SOURCE[d]), d
        for n in e5.STOCK_TEST[d]:  # hashed, never loaded
            assert e5.sha(os.path.join(ROOT, e5.H, "results", n, "scores_test.npz")) == g["scores_sha256"][n], n
        for n, src in zip(e5.STOCK_DEV[d], e5.STOCK_SOURCE[d]):
            with open(os.path.join(ROOT, e5.H, "results", n, "metrics.json")) as f:
                assert json.load(f)["run"]["value"] == src, n
    for k in ("paragraphs_F3072.npz", "splits.json"):
        assert e5.sha(os.path.join(ROOT, e5.H, "data", k)) == inputs[f"data/{k}"], k
    with open(os.path.join(ROOT, e5.DEV_CHECK_REPORT)) as f:
        dev = json.load(f)
    assert dev["best_baseline"] == wiki_report.BEST_ON_DEV, "the best-on-dev baselines are the study's"
    with open(os.path.join(ROOT, "registrations", "E3_slide10_text.frozen.json")) as f:
        e3 = json.load(f)["sources_sha256"]
    for p in e5.E3_SOURCES:
        assert e5.sha(os.path.join(ROOT, p)) == e3[f"src/{os.path.basename(p)}"], "the copies are the versions E3 pins"
    import e5_duplicates
    with open(os.path.join(ROOT, e5.DUPLICATES)) as f:
        dup = json.load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        again = e5_duplicates.main([])
    assert dup == json.loads(json.dumps(again)), "the pinned article lists are what the counting script gives now"
    assert dup["inputs_sha256"] == {k: inputs[f"data/{k}"] for k in ("paragraphs_F3072.npz", "splits.json")}


@test
def runner_refuses_the_test_pairs_before_any_pair_is_read():
    calls = []
    real = run_wiki.load_split
    run_wiki.load_split = lambda *x, **k: calls.append(x) or real(*x, **k)
    try:
        base = ["--data", "sa", "--seed", "11", "--tied", "--name", "e5_sa_s11", "--split", "test"]
        assert "needs --registered" in refused(run_wiki.main, base)
        fake = os.path.join(TMP, "any_file.frozen.json")
        with open(fake, "w") as f:
            json.dump({"plan": "registrations/E4_wikipedia_design.md"}, f)
        msg = refused(run_wiki.main, base + ["--registered", fake])
        assert "refused before any pair was read" in msg and "the record is not at" in msg, msg
        assert "no record file" in refused(run_wiki.main, base + ["--registered", os.path.join(TMP, "missing.json")])
        assert "does not open the test pairs" in refused(run_wiki.main, base + ["--registered", e5.record_path(True)])
        if not os.path.exists(e5.record_path(False)):  # once the record exists it is never handed to a runner here
            assert "no record file" in refused(run_wiki.main, base + ["--registered", e5.record_path(False)])
        dev = ["--data", "sa", "--seed", "1", "--name", "x", "--registered", fake]
        assert "refused before any pair was read" in refused(run_wiki.main, dev), "a record is checked on dev runs too"
        assert calls == [], "no pair was read by a refused call"
    finally:
        run_wiki.load_split = real
    ok = run_wiki.get_args(["--data", "sa", "--seed", "1", "--name", "x"])
    assert ok.split == "dev" and ok.registered == "", "a design run on the dev pairs needs no record"


@test
def report_tool_refuses_the_test_pairs_before_any_pair_is_read():
    calls = []
    real = wiki_report.load_pairs
    wiki_report.load_pairs = lambda *x, **k: calls.append(x) or real(*x, **k)
    try:
        base = ["--data", "sa", "--split", "test", "--runs", "runs/x"]
        assert "needs --registered" in refused(wiki_report.main, base)
        assert "does not open the test pairs" in refused(wiki_report.main, base + ["--registered", e5.record_path(True)])
        fake = os.path.join(TMP, "any_file.frozen.json")
        with open(fake, "w") as f:
            json.dump({"plan": e5.PLAN[False]}, f)
        msg = refused(wiki_report.main, base + ["--registered", fake])
        assert "test pairs refused before any pair was read" in msg and "the record is not at" in msg, msg
        assert "needs --subsets" in refused(wiki_report.main, ["--data", "sa", "--runs", "runs/x", "--subsets-only"])
        assert calls == [], "no pair was read by a refused call"
    finally:
        wiki_report.load_pairs = real
    a = wiki_report.get_args(["--data", "np", "--runs", "runs/x", "--reading", "ratio"])
    assert a.reading == "ratio" and a.split == "dev" and set(wiki_report.READINGS) == {"unit", "ratio", "weight", "logodds"}
    rec = os.path.relpath(e5.record_path(False), e5.ROOT)
    good = {"registered": rec, "seed": 12, "name": "e5_np_s12", "data": "np"}
    assert wiki_report.registered_run_problems(good, "np", "runs/x") == []
    for bad in ({**good, "registered": "registrations/dry/E5_dry.frozen.json"}, {**good, "seed": 1, "name": "e5_np_s1"},
                {**good, "name": "e5dry_np_s12"}, {**good, "data": "sa"}, {**good, "registered": ""}):
        assert wiki_report.registered_run_problems(bad, "np", "runs/x"), bad
    stock = [f"{e5.H}/results/{n}" for n in e5.STOCK_TEST["sa"]]
    assert wiki_report.stock_problems(stock, "sa") == []
    assert wiki_report.stock_problems(stock[::-1], "sa") and wiki_report.stock_problems(stock[:4], "sa")
    assert wiki_report.stock_problems([f"{e5.H}/results/{n}" for n in e5.STOCK_DEV["sa"]], "sa")
    assert wiki_report.stock_problems(stock, "np"), "the folders of another data version"


@test
def the_intervals_of_the_report_tool():
    rng = np.random.default_rng(3)
    n = 400
    y = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    art_a = rng.integers(0, 60, n)
    art_b = np.where(y == 1, art_a, rng.integers(0, 60, n))
    for score in (rng.random(n), np.round(rng.random(n), 1), rng.integers(0, 3, n).astype(float)):  # with ties
        W = wiki_report.WeightedAUC(score, y)
        assert abs(W(np.ones(n)) - auc(score, y)) < 1e-12
        for i in wiki_report.resamples(art_a):
            w = np.bincount(i, minlength=n).astype(float)
            assert abs(W(w) - auc(score[i], y[i])) < 1e-12, "a weighted AUC is the AUC of the resample"
            break
    c = np.array([2, 0, 1, 3])
    ia, ib = np.array([0, 1, 2, 3]), np.array([0, 2, 3, 1])
    same = ia == ib
    assert wiki_report.two_way_weights(c, ia, ib, same).tolist() == [2.0, 0.0, 3.0, 0.0], "copies, or the product of copies"
    m = wiki_report.keep_mask(np.array([1, 2, 3, 4]), np.array([5, 6, 2, 8]), [2, 8])
    assert m.tolist() == [True, False, False, False]
    items = {k: rng.random(n) + (0.4 if k.startswith("ours") else 0.2) * y for k in wiki_report.BASELINES}
    items.update({f"ours_seed{s}": rng.random(n) + 0.5 * y for s in (1, 2, 3)})
    items.update({f"stock_{s}": rng.random(n) + 0.1 * y for s in (1, 2)})
    ours, stock = ["ours_seed1", "ours_seed2", "ours_seed3"], ["stock_1", "stock_2"]
    saved = (wiki_report.SEED_RESAMPLES, wiki_report.TWO_WAY_RESAMPLES)
    wiki_report.SEED_RESAMPLES, wiki_report.TWO_WAY_RESAMPLES = 300, 200
    try:
        _, _, comps = wiki_report.compare(items, ours, stock, y, art_a, art_b, "lr_A_B_AandB", "sa")
        _, _, comps2 = wiki_report.compare(items, ours, stock, y, art_a, art_b, "lr_A_B_AandB", "sa")
    finally:
        wiki_report.SEED_RESAMPLES, wiki_report.TWO_WAY_RESAMPLES = saved
    assert comps == comps2, "the intervals are the same on every call"
    S = np.stack([[auc(items[k][i], y[i]) for k in ours] for i in wiki_report.resamples(art_a)]).mean(1)
    J = np.array([auc(items["jaccard"][i], y[i]) for i in wiki_report.resamples(art_a)])
    assert comps["word_overlap"]["ci95"] == wiki_report.ci(S - J), "the study's interval, unchanged"
    for k in ("word_overlap", "tfidf_cosine", "ppmi_svd_cosine", "best_on_dev", "stock"):
        c = comps[k]
        assert set(c) >= {"ci95", "ci95_seeds_and_articles", "ci95_two_way", "mean_diff"}, k
        assert all(lo <= c["mean_diff"] + 0.2 and lo < hi for lo, hi in (c["ci95"], c["ci95_seeds_and_articles"], c["ci95_two_way"]))
    _, _, comps3 = wiki_report.compare(items, ours, stock, y, art_a, art_b, "ppmi_svd_cosine", "ns_all", two_way=False)
    assert "ci95" not in comps3["word_overlap"] and "ci95_two_way" not in comps3["stock"]


@needs_data
def a_run_opens_only_pinned_data_and_writes_the_registered_fields():
    global OPENED
    out_root = os.path.join(TMP, "small_runs")
    OPENED = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run_dir, final = run_wiki.main(["--data", "sa", "--seed", "990", "--clauses", "30", "--passes", "3",
                                            "--eval-every", "3", "--n-train", "300", "--n-dev", "200", "--tied", "--wmax", "64",
                                            "--name", "unit_wiki", "--out-root", out_root, "--no-log"])
        opened = sorted({p for p in OPENED if p.startswith(os.path.normpath(os.path.join(ROOT, e5.H)) + os.sep) and os.path.isfile(p)})
    finally:
        OPENED = None
    want = sorted(os.path.normpath(os.path.join(ROOT, e5.H, p)) for p in ("data/sa/train.npz", "data/sa/dev.npz", "data/vocab_F1024.json"))
    assert opened == want, opened
    pinned = {os.path.normpath(os.path.join(ROOT, p)) for p in e5.REQUIRED_EXTRAS[True]}
    assert set(opened) <= pinned
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        hist = json.load(f)["history"]
    assert cfg["registered"] == "" and not os.path.isabs(cfg["out_root"]) and cfg["run_start_utc"] <= cfg["run_utc"]
    assert set(cfg["data_sha256"]) == {"sa/train.npz", "sa/dev.npz", "vocab_F1024.json"}
    assert cfg["orgroup_tm_sha256"] == e5.sha(os.path.join(ROOT, "src", "orgroup_tm.py"))
    assert set(cfg["versions"]) == {"python", "executable", "packages"} and cfg["versions"]["packages"] == e5.requirements()
    assert cfg["versions"]["executable"] == e5.EXECUTABLE, "the runs use the study's interpreter"
    assert [h["pass"] for h in hist if "dev_auc" in h] == [3], "with eval_every = passes the split is scored once"
    z = np.load(os.path.join(run_dir, "scores_dev.npz"))
    assert set(z.files) == {"votes", "y", "ratio", "weight", "logodds"} and auc(z["ratio"], z["y"]) == final["dev_auc_ratio"]


def fake_run(root, data, seed, dry, cfg_change=None, final_change=None, frozen_utc="2026-11-12T00:00:00Z", when="120000"):
    """A small run folder that passes every check of e5_report.seed_row, with the changes to its config.json and to
    the 'final' numbers of its metrics.json; also the record entries that the checks compare and the labels."""
    split = e5.SPLIT[dry]
    rng = np.random.default_rng(seed)
    n = e5.N_EVAL[dry]
    y = np.r_[np.ones(n // 2, np.int64), np.zeros(n // 2, np.int64)]
    sc = {k: rng.random(n) + 0.3 * y for k in ("votes", "ratio", "weight", "logodds")}
    name = e5.run_name(data, seed, dry)
    d = os.path.join(root, e5.RUNS_ROOT[dry], f"20261112T{when}Z_{name}")
    os.makedirs(d)
    os.makedirs(os.path.join(root, "env"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "env", "requirements.txt"), os.path.join(root, "env", "requirements.txt"))
    cfg = {**e5.SETTING, "data": data, "seed": seed, "name": name, "split": split, "no_log": False,
           "out_root": e5.RUNS_ROOT[dry], "registered": os.path.relpath(e5.record_path(dry), e5.ROOT),
           "orgroup_tm_sha256": "e" * 64, "data_sha256": {f"{data}/train.npz": "t" * 64, f"{data}/{split}.npz": "s" * 64,
                                                          "vocab_F1024.json": "v" * 64},
           "versions": good_env(), "run_start_utc": "20261112T110000Z"}
    final = {f"{split}_auc": auc(sc["votes"], y), f"{split}_auc_ratio": auc(sc["ratio"], y),
             f"{split}_auc_weight": auc(sc["weight"], y), f"{split}_auc_logodds": auc(sc["logodds"], y),
             f"{split}_tuned_acc": 0.7, f"{split}_tuned_acc_ratio": 0.7, "partner_shuffle_auc": 0.5, "train_auc": 0.9,
             "members": 10, "fit_s_per_pass": 1.0, "wall_s": 10.0, "n_train": e5.N_TRAIN, "n_eval": n}
    hist = [{"pass": p, "fit_s": 1.0} for p in range(1, e5.SETTING["passes"] + 1)]
    hist[-1][f"{split}_auc"] = final[f"{split}_auc"]
    cfg.update(cfg_change or {})
    final.update(final_change or {})
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump({"final": final, "history": hist}, f)
    np.savez(os.path.join(d, "state.npz"), state=np.zeros(1))
    np.savez(os.path.join(d, f"scores_{split}.npz"), y=y, **sc)
    with open(os.path.join(d, "groups.txt"), "w") as f:
        f.write("groups\n")
    rec = {"frozen_utc": frozen_utc, "sources_sha256": {"src/orgroup_tm.py": "e" * 64},
           "extras_sha256": {f"{e5.H}/data/{data}/train.npz": "t" * 64, f"{e5.H}/data/{data}/{split}.npz": "s" * 64,
                             f"{e5.H}/data/vocab_F1024.json": "v" * 64}}
    return d, rec, y


@test
def the_checks_of_one_run():
    saved, env = e5.ROOT, good_env()
    try:
        for dry in (True, False):
            s = e5.SEEDS[dry][0]
            sp = e5.SPLIT[dry]
            cases = [(None, {}, {}), ("setting", {"wmax": 0}, {}), ("setting", {"tied": 1}), ("setting", {"eval_every": 1}, {}),
                     ("data_seed_name_split", {"split": "eval"}, {}), ("data_seed_name_split", {"seed": s + 1}, {}),
                     ("data_seed_name_split", {"data": "sa"}, {}), ("logged_in_default_folder", {"no_log": True}, {}),
                     ("logged_in_default_folder", {"out_root": "runs_other"}, {}),
                     ("names_this_record", {"registered": "registrations/E3_slide10_text.frozen.json"}, {}),
                     ("engine_of_record", {"orgroup_tm_sha256": "0" * 64}, {}), ("data_of_record", {"data_sha256": {}}, {}),
                     ("environment_of_the_run", {"versions": {**env, "packages": {**env["packages"], "numpy": "0"}}}, {}),
                     ("environment_of_the_run", {"versions": {**env, "packages": {**env["packages"], "regex": "0"}}}, {}),
                     ("environment_of_the_run", {"versions": {**env, "executable": "/usr/bin/python3"}}, {}),
                     ("environment_of_the_run", {"versions": {**env, "python": "2.7.18"}}, {}),
                     ("environment_of_the_run", {"versions": {"numpy": env["packages"]["numpy"]}}, {}),
                     ("started_after_freeze", {"run_start_utc": "20261111T235959Z"}, {}),
                     ("started_after_freeze", {"run_start_utc": "x"}, {}), ("pair_counts", {}, {"n_eval": 9999}),
                     ("pair_counts", {}, {"n_train": 300}), ("partner_shuffle_in_bounds", {}, {"partner_shuffle_auc": 0.531}),
                     ("partner_shuffle_in_bounds", {}, {"partner_shuffle_auc": 0.469}),
                     ("score_file_of_this_run", {}, {f"{sp}_auc_ratio": 0.5}), ("score_file_of_this_run", {}, {f"{sp}_auc": 0.5})]
            for check, cfg_change, final_change in [c if len(c) == 3 else (c[0], c[1], {}) for c in cases]:
                e5.ROOT = tempfile.mkdtemp(prefix="checks_", dir=TMP)
                d, rec, y = fake_run(e5.ROOT, "np", s, dry, cfg_change, final_change)
                log = f"| {os.path.basename(d)} | np |"
                c, row, folder = e5_report.seed_row(e5.ROOT, rec, e5.record_path(dry), dry, "np", s, log, y)
                failed = sorted(k for k, v in c.items() if not v)
                assert failed == ([] if check is None else [check]), (dry, check, cfg_change, final_change, failed)
                assert row is not None and folder == d and row["checks_failed"] == failed, "the numbers are kept either way"
            for bound in e5.SHUFFLE_BOUNDS:  # the bounds belong to the allowed range
                e5.ROOT = tempfile.mkdtemp(prefix="checks_", dir=TMP)
                d, rec, y = fake_run(e5.ROOT, "np", s, dry, final_change={"partner_shuffle_auc": bound})
                c = e5_report.seed_row(e5.ROOT, rec, e5.record_path(dry), dry, "np", s, f"| {os.path.basename(d)} |", y)[0]
                assert all(c.values())
            e5.ROOT = tempfile.mkdtemp(prefix="checks_", dir=TMP)
            d, rec, y = fake_run(e5.ROOT, "np", s, dry)
            assert not e5_report.seed_row(e5.ROOT, rec, e5.record_path(dry), dry, "np", s, "", y)[0]["logged_in_default_folder"]
            assert not e5_report.seed_row(e5.ROOT, rec, e5.record_path(dry), dry, "np", s, f"| {os.path.basename(d)} |",
                                          y[::-1].copy())[0]["score_file_of_this_run"], "labels of another order"
            with open(os.path.join(d, "metrics.json")) as f:
                met = json.load(f)
            met["history"][4][f"{e5.SPLIT[dry]}_auc"] = 0.6
            with open(os.path.join(d, "metrics.json"), "w") as f:
                json.dump(met, f)
            c, _, _ = e5_report.seed_row(e5.ROOT, rec, e5.record_path(dry), dry, "np", s, f"| {os.path.basename(d)} |", y)
            assert [k for k, v in c.items() if not v] == ["scored_once_after_the_last_pass"]
            os.remove(os.path.join(e5.ROOT, "env", "requirements.txt"))
            c, _, _ = e5_report.seed_row(e5.ROOT, rec, e5.record_path(dry), dry, "np", s, f"| {os.path.basename(d)} |", y)
            assert not c["environment_of_the_run"], "no package list, no pass"
            os.remove(os.path.join(d, "groups.txt"))
            c, row, folder = e5_report.seed_row(e5.ROOT, rec, e5.record_path(dry), dry, "np", s, f"| {os.path.basename(d)} |", y)
            assert c == {"one_complete_run": False} and row is None and folder is None
            assert e5_report.seed_row(e5.ROOT, rec, e5.record_path(dry), dry, "sa", s, "", y)[0] == {"one_complete_run": False}
    finally:
        e5.ROOT = saved


@test
def environment_and_run_log():
    env = e5.environment()
    assert set(env) == {"python", "executable", "packages"} and set(env["packages"]) == set(e5.requirements())
    assert env["packages"] == e5.requirements(), "the study's environment holds the listed versions"
    assert e5_report.environment_ok(good_env())
    assert not e5_report.environment_ok({**good_env(), "executable": "python3"})
    assert not e5_report.environment_ok({**good_env(), "packages": {}}) and not e5_report.environment_ok(None)
    s = Scratch(dry=True)
    try:
        s.write(e5.RUNLOG, "# runs\n| a | row |\n")
        os.makedirs(os.path.dirname(os.path.join(s.root, e5.RUNLOG_PREFIX[True])), exist_ok=True)
        with open(os.path.join(s.root, e5.RUNLOG_PREFIX[True]), "w") as f:
            json.dump(e5.runlog_prefix(), f)
        assert e5.runlog_prefix_unchanged(True)
        with open(os.path.join(s.root, e5.RUNLOG), "a") as f:
            f.write("| a later row |\n")
        assert e5.runlog_prefix_unchanged(True), "rows appended after the freeze are allowed"
        s.write(e5.RUNLOG, "# runs\n| a | ROW |\n| a later row |\n")
        assert not e5.runlog_prefix_unchanged(True), "a row before the freeze was edited"
        s.write(e5.RUNLOG, "# runs\n")
        assert not e5.runlog_prefix_unchanged(True), "a shorter log"
        os.remove(os.path.join(s.root, e5.RUNLOG_PREFIX[True]))
        assert not e5.runlog_prefix_unchanged(True), "no stored prefix"
    finally:
        s.close()


@test
def report_helpers():
    ok = e5_report.allowed_test
    assert ok("runs/20261112T180000Z_e5_sa_s11", False) and ok("runs/20261112T180000Z_e5_ns_all_s15", False)
    assert ok("runs/20261112T180000Z_e5_np_s12_CRASHED", False)
    assert not ok("runs/20261112T180000Z_e5_sa_s16", False) and not ok("runs/20261112T180000Z_e5_sa_s1", False)
    assert not ok("runs/20261112T180000Z_e5_ns_s11", False) and not ok("runs/20261112T180000Z_e5dev_sa_TB_s11", False)
    assert not ok("runs_dry/20261112T180000Z_e5_sa_s11", False) and not ok("runs/sub/20261112T180000Z_e5_sa_s11", False)
    assert not ok("runs/20261112T180000Z_e5_sa_s11", True) and not ok("runs/20261112T180000Z_e5_sa_s11_x", False)
    root = tempfile.mkdtemp(prefix="traces_", dir=TMP)
    for rel, cfg in (("runs/a", {"split": "dev"}), ("runs/b", {"split": "test"}), ("runs_dry/c", {"split": "dev"}),
                     ("logs/tests/deep/d", {"split": "test"}), ("runs/e", {"split": "eval"}), ("runs/h", ["a list"])):
        os.makedirs(os.path.join(root, rel))
        with open(os.path.join(root, rel, "config.json"), "w") as f:
            json.dump(cfg, f)
    np.savez(os.path.join(root, "runs_dry", "c", "scores_test.npz"), y=np.zeros(2))
    os.makedirs(os.path.join(root, "runs", "f"))
    np.savez(os.path.join(root, "runs", "f", "scores_test.npz"), y=np.zeros(2))
    os.makedirs(os.path.join(root, "runs", "g"))
    with open(os.path.join(root, "runs", "g", "config.json"), "w") as f:
        f.write("not json")
    assert e5_report.test_traces(root) == ["logs/tests/deep/d", "runs/b", "runs/f", "runs/g", "runs/h", "runs_dry/c"]
    assert e5_report.same("a", "a") and not e5_report.same(None, None) and not e5_report.same("a", "b")
    assert e5_report.word({"ci95": [0.001, 0.2]}) == "interval above 0" and e5_report.word({"ci95": [-0.2, -0.001]}) == "interval below 0"
    assert e5_report.word({"ci95": [0.0, 0.2]}) == "interval includes 0"

    def res_with(lo, lo2):
        return {d: {e5.PRIMARY_READING: {"comparisons": {
            "stock": {"vs": "the stock model", "mean_diff": 0.1, "ci95": [lo, 0.2], "ci95_seeds_and_articles": [lo2, 0.2]}}}}
            for d in e5.DATA}
    rl = e5_report.rules(res_with(0.01, 0.02))
    assert set(rl) == set(e5.RULES) and all(r["met"] and r["reading"] == "unit" for r in rl.values())
    for lo, lo2 in ((0.0, 0.02), (0.01, 0.0), (0.01, -0.01), (-0.01, 0.01)):
        assert not any(r["met"] for r in e5_report.rules(res_with(lo, lo2)).values()), "both intervals above 0, or not met"


def made_up_reading(d, reading, n, dry, seeds):
    """A made-up output of wiki_report.main with the keys that e5_report reads."""
    def c(md, vs):
        return {"vs": vs, "mean_diff": md, "ci95": [md - 0.01, md + 0.01], "ci95_seeds_and_articles": [md - 0.012, md + 0.012],
                "ci95_two_way": [md - 0.013, md + 0.013], "verdict": "v"}
    comps = {"stock": c(0.09, "the stock model"), "ppmi_svd_cosine": c(-0.02, "ppmi_svd_cosine")}
    if d == "ns_all":
        comps.update(word_overlap={"vs": "jaccard", "value": "n"}, tfidf_cosine={"vs": "tfidf_cosine", "value": "n"},
                     best_on_dev=c(-0.02, "ppmi_svd_cosine"))
    else:
        comps.update(word_overlap=c(0.005, "jaccard"), tfidf_cosine=c(0.002, "tfidf_cosine"), best_on_dev=c(-0.04, "lr_A_B_AandB"))
    return {"data": d, "split": e5.SPLIT[dry], "reading": reading, "n_pairs": n, "ours_mean_auc": 0.83,
            "stock_mean_auc": 0.74, "stock": {f"stock_{i + 1}": {"auc": 0.74} for i in range(5)},
            "baselines": {k: {"auc": 0.8} for k in wiki_report.BASELINES}, "comparisons": comps,
            "ours": {f"ours_seed{s}": {"auc": 0.83} for s in seeds}}


@test
def sentences_and_the_registered_numbers_of_the_study():
    res = {d: {r: made_up_reading(d, r, e5.N_EVAL[False], False, e5.SEEDS[False]) for r in ("unit", "ratio")} for d in e5.DATA}
    rl = e5_report.rules(res)
    sub = {d: {"whole_copy_articles": {"articles_listed": 36, "n_pairs": 9800, "n_pairs_dropped": 200,
                                       "comparisons": res[d]["unit"]["comparisons"]}} for d in e5.DATA}
    text = e5_report.sentences(res, rl, False, (), sub)
    first = text[0]
    assert first.startswith("Same-article pairs (sa), 10,000 held-back test pairs: the mean AUC of 5 new seeds of the OR-group rule in the vote sum")
    assert "(paired article bootstrap, 95%) and [+0.078, +0.102] with the seeds resampled, so R1-sa (above the stock model) is met; predicted from DEV: met." in first
    assert "the ratio reading minus word overlap (AUC 0.800) is +0.005 [-0.005, +0.015] [-0.007, +0.017], and the vote sum minus word overlap is" in first
    assert e5.CAVEAT in text and not any("above word overlap" in t and "R2" in t for t in text)
    assert any(t.startswith("Readings, np (test), the ratio reading") and t.endswith("(no rule).") for t in text)
    assert any("Without the 200 test pairs of ns_all that touch one of the 36 articles that are whole copies" in t for t in text)
    assert not any("beats" in t or " above word overlap" in t for t in text), "no verdict words for the readings"
    void = e5_report.sentences(res, rl, False, ("np_seed12_setting",))
    assert void[0].startswith("registered run void: np_seed12_setting.")
    dry = e5_report.sentences({d: {"unit": made_up_reading(d, "unit", 20000, True, (1,))} for d in e5.DATA}, rl, True)
    assert "20,000 DEV pairs (dry run): the mean AUC of 1 dry seed of" in dry[0] and "In the ratio reading" not in dry[0]
    with open(os.path.join(ROOT, e5.FINAL_TEST_REPORT)) as f:
        rep = json.load(f)
    reg = {}
    for d in e5.DATA:
        g = rep["groups"][d]
        r = made_up_reading(d, "unit", 10000, False, e5.SEEDS[False])
        r["baselines"] = {k: {"auc": v["auc"]} for k, v in g["baselines"].items()}
        r["stock"] = {f"stock_{i + 1}": {"auc": s["auc"]} for i, s in enumerate(g["seeds"])}
        reg[d] = {"unit": r}
    sn = e5_report.study_numbers(ROOT, reg, False)
    assert all(v["equal_to_the_printed_digits"] and len(v["compared"]) == 11 for v in sn.values()), sn
    reg["np"]["unit"]["stock"]["stock_3"]["auc"] += 1e-6
    assert not e5_report.study_numbers(ROOT, reg, False)["np"]["equal_to_the_printed_digits"]


class FakeStudy:
    """A scratch study folder whose record passes, with one run per data version and seed, made-up data files and a
    made-up study report, for the paths of e5_report.main. wiki_report.main and score_groups.main are replaced by
    made-up numbers, so no pair is read; the environment of this process is taken as the study's."""

    def __init__(self, dry, broken=False):
        import hashlib

        import score_groups
        self.dry, self.calls = dry, []
        self.s = Scratch(dry)
        self.saved = (e5.environment, wiki_report.main, score_groups.main)
        e5.environment = good_env
        root, extras = self.s.root, dict(self.s.rec["extras_sha256"])
        self.s.write("src/orgroup_tm.py", "ENGINE = 1\n")
        engine = e5.sha(os.path.join(root, "src", "orgroup_tm.py"))
        hdata = os.path.join(root, e5.H, "data")
        os.makedirs(hdata, exist_ok=True)
        with open(os.path.join(hdata, "vocab_F1024.json"), "w") as f:
            json.dump(["a"], f)
        log, split, n = ["# runs"], e5.SPLIT[dry], e5.N_EVAL[dry]
        y = np.r_[np.ones(n // 2, np.int64), np.zeros(n // 2, np.int64)]
        for d in e5.DATA:
            os.makedirs(os.path.join(hdata, d), exist_ok=True)
            np.savez(os.path.join(hdata, d, "train.npz"), y=y[:10])
            np.savez(os.path.join(hdata, d, f"{split}.npz"), y=y)
            shas = {k: e5.sha(os.path.join(hdata, p)) for k, p in (
                (f"{d}/train.npz", f"{d}/train.npz"), (f"{d}/{split}.npz", f"{d}/{split}.npz"), ("vocab_F1024.json", "vocab_F1024.json"))}
            extras.update({f"{e5.H}/data/{k}": v for k, v in shas.items()})
            for i, seed in enumerate(e5.SEEDS[dry]):
                change = {"partner_shuffle_auc": 0.6} if (broken and d == "np" and i == 0) else {}
                run, _, _ = fake_run(root, d, seed, dry, {"orgroup_tm_sha256": engine, "data_sha256": shas}, change, when=f"12{i:02d}00")
                log.append(f"| {os.path.basename(run)} | {d} |")
        self.s.write(e5.RUNLOG, "\n".join(log) + "\n")
        text = b"# runs\n"
        self.s.write(e5.RUNLOG_PREFIX[dry], json.dumps({"file": e5.RUNLOG, "length": len(text), "sha256": hashlib.sha256(text).hexdigest()}))
        study = {"dev_auc": {d: {k: 0.8 for k in wiki_report.BASELINES} for d in e5.DATA},
                 "tm_dev_auc": {d: {f"seed{i}": 0.74 for i in range(1, 6)} for d in e5.DATA}} if dry else {
            "groups": {d: {"baselines": {k: {"auc": 0.8} for k in wiki_report.BASELINES}, "seeds": [{"auc": 0.74}] * 5} for d in e5.DATA}}
        self.s.write(e5.DEV_CHECK_REPORT if dry else e5.FINAL_TEST_REPORT, json.dumps(study))
        self.s.rec = {**self.s.rec, "sources_sha256": e5.source_hashes(), "extras_sha256": extras}
        self.s.save(self.s.rec)

        def fake_wiki(argv):
            d, reading = argv[argv.index("--data") + 1], argv[argv.index("--reading") + 1]
            self.calls.append((d, reading, "--subsets-only" in argv))
            out = made_up_reading(d, reading, n, dry, e5.SEEDS[dry])
            if "--subsets-only" in argv:
                return {"subsets": {"whole_copy_articles": {"articles_listed": 3, "n_pairs": 90, "n_pairs_dropped": 10,
                                                            "comparisons": out["comparisons"]}}}
            return out
        wiki_report.main = fake_wiki
        score_groups.main = lambda argv: {"data": argv[1]}

    def run(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            status = e5_report.main(["--frozen", self.s.record, *(["--dry"] if self.dry else []), *extra])
        return status, (json.loads(out.getvalue()) if out.getvalue() else None)

    def close(self):
        import score_groups
        e5.environment, wiki_report.main, score_groups.main = self.saved
        self.s.close()


@test
def the_report_paths_after_a_freeze():
    import score_groups
    for dry in (True, False):
        fs = FakeStudy(dry)
        try:
            out_file = os.path.join(fs.s.root, "logs", "x_report.json")
            status, res = fs.run("--out", out_file)
            assert status == 0 and res["checks_all_passed"] and all(res["checks"].values()), [k for k, v in res["checks"].items() if not v]
            assert set(res["rules"]) == set(e5.RULES) and all(r["met"] for r in res["rules"].values())
            assert "readings_failed" not in res and "registered_run_void" not in res, res.get("readings_failed")
            assert all(v["equal_to_the_printed_digits"] for v in res["study_numbers"].values())
            assert sorted({c[1] for c in fs.calls}) == ["ratio", "unit"] and sum(c[2] for c in fs.calls) == 3
            left = sorted(os.listdir(os.path.join(fs.s.root, "logs")))
            assert "x_report.json" in left and not [f for f in left if ".part_" in f], left
            with open(out_file) as f:
                assert json.load(f) == res, "the file holds what was printed"
            assert fs.run("--out", out_file)[0] == 2, "a passed report is never overwritten"

            def boom(argv):
                raise RuntimeError("no groups")
            score_groups.main = boom
            status, res = fs.run()
            assert status == 0 and set(res["rules"]) == set(e5.RULES) and set(res["readings_failed"]) == {"groups"}, res.get("readings_failed")
            assert "readings_without_duplicates" in res and "study_numbers" in res, "the steps after a failed one still run"
            status, res = fs.run("--steps", "rules")
            assert status == 0 and "rules" in res and "groups" not in res and res["steps"] == "rules"
            status, res = fs.run("--checks-only")
            assert status == 0 and "rules" not in res and "numbers" not in res
        finally:
            fs.close()
        fs = FakeStudy(dry, broken=True)
        try:
            out_file = os.path.join(fs.s.root, "logs", "x_report.json")
            status, res = fs.run("--out", out_file)
            s0 = e5.SEEDS[dry][0]
            assert status == 1 and res["registered_run_void"] == [f"registered run void: np_seed{s0}_partner_shuffle_in_bounds"], res.get("registered_run_void")
            assert set(res["rules"]) == set(e5.RULES) and res["sentences"][0].startswith("registered run void: ")
            assert len(res["rows"]) == 3 * len(e5.SEEDS[dry]) and "groups" in res, "every number is in the void report"
            left = [f for f in os.listdir(os.path.join(fs.s.root, "logs")) if f.startswith("x_report.json")]
            assert len(left) == 1 and ".FAILED_" in left[0] and not os.path.exists(out_file), left
        finally:
            fs.close()
    fs = FakeStudy(True)
    try:
        fs.s.write("src/a.py", "A = 3\n")
        status, res = fs.run()
        assert status == 1 and not res["checks"]["record_plan_sources_and_pinned_files"]
        assert res["numbers"].startswith("not computed: the record did not pass") and fs.calls == [], "no record, no pair"
    finally:
        fs.close()


@test
def report_refuses_a_record_that_is_not_e5s():
    fake = os.path.join(TMP, "fake.frozen.json")
    with open(fake, "w") as f:
        json.dump({"plan": e5.PLAN[False], "plan_sha256": "0", "frozen_utc": "2026-11-12T00:00:00Z",
                   "sources_sha256": {}, "extras_sha256": {}}, f)
    for argv in (["--frozen", fake], ["--frozen", fake, "--dry"], ["--frozen", fake, "--checks-only"]):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = e5_report.main(argv)
        res = json.loads(out.getvalue())
        assert status == 1 and res["checks_all_passed"] is False and "rules" not in res and "sentences" not in res, argv
        assert res["checks"]["record_plan_sources_and_pinned_files"] is False


if __name__ == "__main__":
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"ALL {len(PASSED)} TESTS PASSED" + (f" ({len(SKIPPED)} skipped: no data)" if SKIPPED else ""))
