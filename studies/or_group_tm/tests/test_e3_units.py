"""Unit tests of the E3 guards and helpers: src/e3_common.py (record check, run arguments), the export of a model to
the scorer's layout and the refusals of src/run_slide10.py, the texts and the guard of src/probe_slide10.py, and
the helpers of src/e3_report.py (plain script, no pytest). Every check has a case in which it must fail.

Run from the study folder:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -I tests/test_e3_units.py
Prints one line per test and 'ALL n TESTS PASSED', or stops with exit status 1 at the first failure. The tests that
need the slide-10 data of the hierarchical_tm study (not distributed) are skipped when it is missing. Scratch files
go to a temporary folder that is removed at the end; nothing in the study folder is written.
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True

import e3_common as e3
import e3_report
import probe_slide10
import run_slide10
from orgroup_tm import OrGroupTM, pack_pairs

HAVE_DATA = os.path.isfile(os.path.join(ROOT, e3.REQUIRED_EXTRAS[0]))
TMP = tempfile.mkdtemp(prefix="e3_units_")
PASSED, SKIPPED = [], []


def test(fn):
    fn()
    PASSED.append(fn.__name__)
    print("ok  ", fn.__name__, flush=True)
    return fn


def needs_data(fn):
    if HAVE_DATA:
        return test(fn)
    SKIPPED.append(fn.__name__)
    print("skip", fn.__name__, "(the slide-10 data are not distributed)", flush=True)
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
    """A small copy of the layout that e3_common checks: src/, tests/, the plan, the pinned files and a record."""

    def __init__(self, dry=False):
        self.dry = dry
        self.root = tempfile.mkdtemp(prefix="study_", dir=TMP)
        self.saved = (e3.ROOT, e3.REQUIRED_EXTRAS)
        for rel, text in (("src/a.py", "A = 1\n"), ("tests/t.py", "T = 1\n"), (e3.PLAN[dry], "the plan\n"),
                          ("pinned/data.txt", "data\n"), ("pinned/job.sh", "job\n")):
            self.write(rel, text)
        e3.ROOT, e3.REQUIRED_EXTRAS = self.root, ("pinned/data.txt", "pinned/job.sh")
        self.record = e3.record_path(dry)
        self.rec = {"plan": e3.PLAN[dry], "plan_sha256": e3.sha(os.path.join(self.root, e3.PLAN[dry])),
                    "frozen_utc": "2026-11-12T00:00:00Z", "sources_sha256": e3.source_hashes(),
                    "extras_sha256": {k: e3.sha(os.path.join(self.root, k)) for k in e3.REQUIRED_EXTRAS}}
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
        return e3.preflight(path or self.record, self.dry)[1]

    def close(self):
        e3.ROOT, e3.REQUIRED_EXTRAS = self.saved


def args(**kw):
    a = argparse.Namespace(**e3.SETTING, seed=11, name="e3_text_s11", split="eval", no_log=False,
                           out_root=os.path.join(e3.ROOT, "runs"))
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@test
def preflight_passes_and_fails_as_it_should():
    s = Scratch()
    try:
        assert s.problems() == [], s.problems()
        assert e3.preflight(os.path.join(s.root, "nothing.json"))[0] is None
        assert any("no record file" in p for p in e3.preflight("")[1])
        other = os.path.join(s.root, "registrations", "copy.frozen.json")
        s.save(s.rec, other)
        assert any("the record is not at" in p for p in s.problems(other)), "a copy at another place is not the record"
        s.save({**s.rec, "plan": "registrations/E2_slide10_design.md"})
        assert any("the record is of" in p for p in s.problems()), "the record of another plan"
        s.save(s.rec)
        s.write(e3.PLAN[False], "the plan, edited\n")
        assert "the plan text differs from the record" in s.problems()
        s.write(e3.PLAN[False], "the plan\n")
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
        s.write("more.txt", "x\n")
        s.save({**s.rec, "extras_sha256": {**s.rec["extras_sha256"], "more.txt": e3.sha(os.path.join(s.root, "more.txt"))}})
        assert s.problems() == []
        s.write("more.txt", "y\n")
        assert "pinned file differs from the record: more.txt" in s.problems(), "a further pinned file is checked too"
    finally:
        s.close()


@test
def dry_and_registered_records_are_not_interchangeable():
    s = Scratch(dry=True)
    try:
        assert s.problems() == [] and e3.is_dry_record(s.record)
        assert e3.preflight(s.record, dry=False)[1], "the dry record is not a record of the registration"
        assert not e3.is_dry_record(e3.record_path(False))
    finally:
        s.close()


@test
def run_arguments_are_checked():
    s = Scratch()
    try:
        assert e3.check_run_args(args()) == []
        assert e3.check_run_args(args(seed=35, name="e3_text_s35")) == [], "a further seed is a registered seed"
        for k, v in (("K", 16), ("clauses", 40), ("T", 4.0), ("passes", 50), ("seed_all", False), ("p_decay", 0.1),
                     ("reinforce", "all"), ("reject", "random"), ("n_train", 400), ("tied", True), ("version", "pairs")):
            assert any(f"setting {k} " in p for p in e3.check_run_args(args(**{k: v}))), k
        assert any("not a registered seed" in p for p in e3.check_run_args(args(seed=1, name="e3_text_s1")))
        assert any("not a registered seed" in p for p in e3.check_run_args(args(seed=36, name="e3_text_s36")))
        assert any("name must be" in p for p in e3.check_run_args(args(name="e3_text_s12")))
        assert any("split must be" in p for p in e3.check_run_args(args(split="dev")))
        assert e3.check_run_args(args(out_root=os.path.join(s.root, "elsewhere")))
        assert e3.check_run_args(args(no_log=True))
        os.makedirs(os.path.join(s.root, "runs", "20261112T000000Z_e3_text_s11"))
        assert any("exists already" in p for p in e3.check_run_args(args()))
        assert e3.check_run_args(args(seed=12, name="e3_text_s12")) == []
        os.rename(os.path.join(s.root, "runs", "20261112T000000Z_e3_text_s11"),
                  os.path.join(s.root, "runs", "20261112T000000Z_e3_text_s11_CRASHED"))
        assert e3.check_run_args(args()) == [], "a crashed folder that was renamed does not block the rerun"
        dry = args(seed=1, name="e3dry_text_s1", split="dev", out_root=os.path.join(s.root, "runs_dry"))
        assert e3.check_run_args(dry, dry=True) == []
        assert e3.check_run_args(dry, dry=False) and e3.check_run_args(args(), dry=True)
    finally:
        s.close()


@test
def export_gives_the_scorer_the_models_own_firing():
    F, K = 12, 3
    rng = np.random.default_rng(5)
    tm = OrGroupTM(4, F, sides=1, K=K, seed=3)
    tm.state[:] = -1
    members = {(0, 0): [1], (0, 1): [2, 3, 4], (0, 2): [5, 6],        # one member, three members, two members
               (1, 0): [0, 7], (1, 1): [], (1, 2): [8],               # a group without members: the clause never fires
               (2, 0): [9], (2, 1): [9], (2, 2): [9, 10],             # the same word in every group
               (3, 0): [], (3, 1): [], (3, 2): []}                    # an unused clause
    for (c, g), words in members.items():
        tm.state[c, g, words] = 0
        tm.nmem[c, g] = len(words)
    assert tm.nmem_consistent()
    X = (rng.random((300, F)) < 0.35).astype(np.uint8)
    X[0] = 0
    X[1] = 1
    X[1, 11] = 0  # every word but one that is in no group
    inc, A = run_slide10.include_bits_of(tm, F)
    assert A == 3 and inc.shape == (8, K * A, F) and not inc[1::2].any(), "clauses at the even rows, odd rows empty"
    assert inc[2, A:2 * A].all(), "a group without members becomes options that hold every word"
    assert run_slide10.export_equals_model(tm, inc, A, X)
    full = np.ones((1, F), np.uint8)  # the one text for which the export is not the model: it holds every word
    assert not run_slide10.export_equals_model(tm, inc, A, full), "the run-time check would stop such a data set"
    ids, bounds = pack_pairs(X, np.arange(len(X))[:, None])
    _, fire = tm.votes(ids, bounds, return_fire=True)
    assert fire[:, 0].any() and not fire[:, 1].any() and not fire[:, 3].any() and fire[1, 0] and fire[1, 2]
    wrong = inc.copy()
    wrong[2, A:2 * A] = False  # empty options: the scorer of the study would read the group as always true
    assert not run_slide10.export_equals_model(tm, wrong, A, X), "the check must notice a wrong export"


@test
def eval_sentences_are_refused_without_the_record_of_e3():
    base = ["--version", "text", "--seed", "11", "--seed-all", "--name", "e3_text_s11", "--split", "eval"]
    assert "needs --registered" in refused(run_slide10.get_args, base)
    fake = os.path.join(TMP, "any_file.frozen.json")
    with open(fake, "w") as f:
        json.dump({"plan": "registrations/E2_slide10_design.md"}, f)
    msg = refused(run_slide10.get_args, base + ["--registered", fake])
    assert "refused before any model was trained or scored" in msg and "the record is not at" in msg, msg
    assert "no record file" in refused(run_slide10.get_args, base + ["--registered", os.path.join(TMP, "missing.json")])
    assert "does not open the EVAL sentences" in refused(run_slide10.get_args, base + ["--registered", e3.record_path(True)])
    ok = run_slide10.get_args(["--version", "text", "--seed", "1", "--name", "x"])
    assert ok.split == "dev" and ok.registered == "", "a design run on the DEV sentences needs no record"


@test
def probe_reading_and_guard():
    r = probe_slide10.reading
    assert r(1.0, 0.0) == "template-like" and r(0.9, 0.05) == "template-like"
    assert r(0.89, 0.0) == "reject-all (degenerate)" and r(1.0, 0.051) == "neither (between)"
    assert r(1.0, 0.5) == "neither (between)" and r(1.0, 0.51).startswith("blacklist-like")
    run = os.path.join(TMP, "runs", "20261112T000000Z_e2_text_x_s1")
    os.makedirs(run)
    cfg = {"version": "text", "split": "dev", "registered": "", "seed": 1, "name": "e2_text_x_s1"}
    with open(os.path.join(run, "config.json"), "w") as f:
        json.dump(cfg, f)
    assert "the run was not scored on the EVAL sentences" in probe_slide10.eval_problems(run, cfg)
    assert "the EVAL sentences are refused" in refused(probe_slide10.main, [run, "--split", "eval"])
    cfg2 = {**cfg, "split": "eval", "registered": os.path.relpath(e3.record_path(True), e3.ROOT)}
    assert any("names no record" in p for p in probe_slide10.eval_problems(run, cfg2)), "the dry record opens nothing"
    cfg3 = {**cfg, "split": "eval", "registered": "registrations/none.frozen.json", "seed": 11, "name": "e3_text_s11"}
    p3 = probe_slide10.eval_problems(run, cfg3)
    assert any("no record file" in p for p in p3) and "not the folder of a registered run" in p3


@needs_data
def probe_texts_are_built_from_the_dev_family_sentences():
    X, intact, which = probe_slide10.deletion_texts(1)
    assert intact.shape == (101, 221) and X.shape == (606, 221) and np.bincount(which).tolist() == [101] * 6
    assert (X <= np.repeat(intact, 6, axis=0)).all() and (X.sum(1) < np.repeat(intact.sum(1), 6)).all()
    X1, which1 = probe_slide10.one_word_texts(1)
    assert len(X1) == 583 and {int(k) + 1: int((which1 == k).sum()) for k in np.unique(which1)} == {3: 74, 5: 303, 6: 206}
    seen = probe_slide10.is_training_negative(X)
    assert seen.dtype == bool and 0 < seen.sum() < len(X)
    assert not probe_slide10.is_training_negative(intact).any(), "an intact family sentence is no training negative"


@needs_data
def probe_default_is_dev_and_agrees_with_the_scorer():
    out_root = os.path.join(TMP, "small_runs")
    with contextlib.redirect_stdout(io.StringIO()):
        run_dir, final = run_slide10.main(["--version", "text", "--seed", "990", "--clauses", "6", "--K", "8", "--passes", "3",
                                           "--seed-all", "--name", "unit_text", "--out-root", out_root, "--no-log"])
        pr = probe_slide10.main([run_dir])
    assert pr["split"] == "dev" and os.path.isfile(os.path.join(run_dir, "probe_dev.json"))
    assert not os.path.exists(os.path.join(run_dir, "probe_eval.json")) and not os.path.exists(os.path.join(run_dir, "check2b_eval.txt"))
    chk = final["check2b"]
    assert pr["best_rule_accepts_intact"] == chk["coverage_dev_288"] and pr["n_intact"] == 101 and pr["n_deletion_texts"] == 606
    assert final["export_equals_model_on_train_and_dev"] is True
    tied = final["tied_best_rules"]
    assert tied[0]["rule"] == chk["best_rule"] and tied[0]["coverage_216"] == chk["coverage_dev_216"]
    assert tied[0]["fires_max"] == chk["fires_dev_max"] and tied[0]["brackets_216"] == chk["brackets_recovered_216"]
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["registered"] == "" and not os.path.isabs(cfg["out_root"]) and len(cfg["orgroup_tm_sha256"]) == 64
    assert set(cfg["data"]) == set(e3.RUN_HASHES) and cfg["run_start_utc"] <= cfg["run_utc"]
    assert "the EVAL sentences are refused" in refused(probe_slide10.main, [run_dir, "--split", "eval"])


@needs_data
def data_facts_quoted_by_the_registration():
    import slide10_facts
    with contextlib.redirect_stdout(io.StringIO()):
        f = slide10_facts.main([])
    assert {k: f["sentences_per_kind"][k]["eval"] for k in e3.N_SPLIT[False]} == e3.N_SPLIT[False]
    assert {k: f["sentences_per_kind"][k]["dev"] for k in e3.N_SPLIT[True]} == e3.N_SPLIT[True]
    assert f["family_sentences_in_the_216_view"] == {"train": 66, "dev": e3.N_IN216[True], "eval": e3.N_IN216[False]}
    assert f["equal_bags_between_splits"] == {"train_dev": 0, "train_eval": 0, "dev_eval": 0}
    assert f["text_train_npz"]["splits_of_the_rows"] == [0] and f["text_dev_npz"]["splits_of_the_rows"] == [1]
    assert f["template_clause_allowed_as_it_is"] is False and f["brackets_whose_full_group_is_not_allowed"] == [5]
    assert f["ceiling_of_the_bracket_jaccard_per_bracket"]["288"] == [1.0, 1.0, 1.0, 1.0, 0.667, 1.0]
    assert f["deletion_texts_dev"]["n"] == 606 and f["one_word_texts_dev"]["n"] == 583


@test
def report_helpers():
    ci = e3_report.exact_interval
    assert ci(3, 5) == [0.147, 0.947] and ci(4, 5) == [0.284, 0.995] and ci(5, 5) == [0.478, 1.0] and ci(0, 5) == [0.0, 0.522]
    assert e3_report.same("a", "a") and not e3_report.same(None, None) and not e3_report.same("a", "b")
    ok = e3_report.allowed_eval
    assert ok("runs/20261112T180000Z_e3_text_s11", False) and ok("runs/20261112T180000Z_e3_text_s35", False)
    assert ok("runs/20261112T180000Z_e3_text_s12_CRASHED", False)
    assert not ok("runs/20261112T180000Z_e3_text_s36", False) and not ok("runs/20261112T180000Z_e3_text_s1", False)
    assert not ok("runs/20261112T180000Z_e2_text_A6_s11", False) and not ok("runs_dry/20261112T180000Z_e3_text_s11", False)
    assert not ok("runs/sub/20261112T180000Z_e3_text_s11", False) and not ok("runs/20261112T180000Z_e3_text_s11", True)
    root = tempfile.mkdtemp(prefix="traces_", dir=TMP)
    for rel, cfg in (("runs/a", {"split": "dev"}), ("runs/b", {"split": "eval"}), ("runs_dry/c", {"split": "dev"}),
                     ("logs/tests/deep/d", {"split": "eval"}), ("runs/e", {"split": "test"})):
        os.makedirs(os.path.join(root, rel))
        with open(os.path.join(root, rel, "config.json"), "w") as f:
            json.dump(cfg, f)
    with open(os.path.join(root, "runs_dry", "c", "probe_eval.json"), "w") as f:
        f.write("{}")
    os.makedirs(os.path.join(root, "runs", "g"))
    with open(os.path.join(root, "runs", "g", "config.json"), "w") as f:
        f.write("not json")
    assert e3_report.eval_traces(root) == ["logs/tests/deep/d", "runs/b", "runs/g", "runs_dry/c"]
    rows = [{"both_flags": True, "reading_288": True, "flag_288": True, "flag_216": True, "cover_and_fire": True,
             "best_rule_reading": "template-like", "model_reading": "neither (between)"},
            {"both_flags": False, "reading_288": True, "flag_288": True, "flag_216": False, "cover_and_fire": True,
             "best_rule_reading": "template-like", "model_reading": "neither (between)"}]
    cnt = e3_report.counts(rows)
    assert cnt["both_flags"]["seeds"] == 1 and cnt["reading_288"]["seeds"] == 2 and cnt["model_template_like"]["seeds"] == 0
    assert cnt["reading_288"]["exact_95_interval_of_the_share"] == [0.158, 1.0]


@test
def report_refuses_a_record_that_is_not_e3s():
    fake = os.path.join(TMP, "fake.frozen.json")
    with open(fake, "w") as f:
        json.dump({"plan": e3.PLAN[False], "plan_sha256": "0", "frozen_utc": "2026-11-12T00:00:00Z",
                   "sources_sha256": {}, "extras_sha256": {}}, f)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = e3_report.main(["--frozen", fake])
    res = json.loads(out.getvalue())
    assert status == 1 and res["checks_all_passed"] is False and "rows" not in res and "sentences" not in res
    assert res["checks"]["record_plan_sources_and_pinned_files"] is False


if __name__ == "__main__":
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"ALL {len(PASSED)} TESTS PASSED" + (f" ({len(SKIPPED)} skipped: no data)" if SKIPPED else ""))
