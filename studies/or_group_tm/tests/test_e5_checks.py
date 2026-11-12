"""Check test of E5 on the outputs of the dry job (registrations/E5_job.sh dry): its run folders (runs_dry/*_e5dry_*),
its record and the run log are copied into a scratch copy of the study folder; there the report (src/e5_report.py)
must pass and give the numbers of logs/e5_dry_report.json while every file it opens is recorded, and then one thing
at a time is broken and the named check must fail. The guards of src/run_wiki.py and src/wiki_report.py are tried
with the dry record in place. No model is trained, nothing in the study folder is written, and no test pair is read.

Run from the study folder, after the dry job (about ten minutes on one core):
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I tests/test_e5_checks.py
Prints one line per case and 'CHECK TEST PASSED (n cases)', or stops with exit status 1 at the first failure.
Skipped when the Wikipedia data of the hierarchical_tm study (not distributed) or the outputs of the dry job are
missing.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True
import e5_common as e5

HSTUDY = os.path.normpath(os.path.join(ROOT, e5.H))
LMSTUDY = os.path.normpath(os.path.join(ROOT, "..", "tm_language_model"))
DLSTUDY = os.path.normpath(os.path.join(ROOT, "..", "decision_layer"))
RECORD = "registrations/dry/E5_dry.frozen.json"
CASES = []
TRACER = r"""
import os, runpy, sys
log, script = sys.argv[1], sys.argv[2]
seen = set()
def hook(event, args):
    if event == "open" and isinstance(args[0], str):
        seen.add(os.path.abspath(args[0]))
sys.addaudithook(hook)
sys.argv = [script] + sys.argv[3:]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    names = sorted(seen)
    with open(log, "w") as f:
        f.write("\n".join(names) + "\n")
"""


def ok(name):
    CASES.append(name)
    print("ok  ", name, flush=True)


def dry_runs():
    """{data: the run folder of the dry job} (paths relative to the study folder), or None when one is missing."""
    out = {}
    for d in e5.DATA:
        found = sorted(glob.glob(os.path.join(ROOT, e5.RUNS_ROOT[True], f"*_{e5.run_name(d, 1, True)}")))
        if len(found) != 1:
            return None
        out[d] = os.path.relpath(found[0], ROOT)
    return out


def main():
    runs = dry_runs()
    if not os.path.isfile(os.path.join(HSTUDY, "data", "sa", "train.npz")):
        print("CHECK TEST SKIPPED (the Wikipedia pairs are not distributed)")
        return
    if runs is None or not os.path.isfile(os.path.join(ROOT, RECORD)) or not os.path.isfile(os.path.join(ROOT, "logs", "e5_dry_report.json")):
        print("CHECK TEST SKIPPED (no complete output of registrations/E5_job.sh dry)")
        return
    tmp = tempfile.mkdtemp(prefix="e5_checks_")
    try:
        run_all(tmp, os.path.join(tmp, "or_group_tm"), runs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"CHECK TEST PASSED ({len(CASES)} cases)")


def run_all(tmp, study, runs):
    os.symlink(HSTUDY, os.path.join(tmp, "hierarchical_tm"))
    os.symlink(LMSTUDY, os.path.join(tmp, "tm_language_model"))
    os.symlink(DLSTUDY, os.path.join(tmp, "decision_layer"))
    for folder in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, folder), os.path.join(study, folder), ignore=shutil.ignore_patterns("__pycache__"))
    os.symlink(os.path.join(ROOT, ".venv"), os.path.join(study, ".venv"))  # the study's interpreter, by its place in the study
    PY = os.path.join(study, e5.EXECUTABLE)
    for rel in ("registrations/E5_job.sh", "registrations/dry/E5_dry.md", RECORD, "env/requirements.txt", "logs/RUNS_wiki.md",
                e5.DUPLICATES, *e5.E3_SOURCES, e5.RUNLOG_PREFIX[True], *runs.values()):
        src, dst = os.path.join(ROOT, rel), os.path.join(study, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def sh(*cmd):
        return subprocess.run(cmd, check=False, cwd=study, env=env, capture_output=True, text=True)

    def script(name, *args):
        return sh("nice", "-n", "19", PY, "-I", os.path.join("src", name), *args)

    def report(*extra):
        r = script("e5_report.py", "--dry", "--checks-only", "--frozen", RECORD, *extra)
        return r.returncode, json.loads(r.stdout)

    def read(rel):
        with open(os.path.join(study, rel)) as f:
            return f.read()

    def write(rel, text):
        p = os.path.join(study, rel)
        mode = os.stat(p).st_mode if os.path.exists(p) else None
        if mode is not None:
            os.chmod(p, mode | 0o200)
        with open(p, "w") as f:
            f.write(text)
        if mode is not None:
            os.chmod(p, mode)

    def broken(name, check, rel, change):
        """Change one file, expect the named check (and no other) to fail, put the file back."""
        before = read(rel)
        write(rel, change(before))
        status, res = report()
        write(rel, before)
        assert status == 1 and res["checks_all_passed"] is False, (name, status)
        assert [k for k, v in res["checks"].items() if not v] == [check], (name, [k for k, v in res["checks"].items() if not v])
        assert "rules" not in res and "sentences" not in res, name
        ok(name)

    def json_change(**kw):
        return lambda text: json.dumps({**json.loads(text), **kw})

    def final_change(**kw):
        def change(text):
            m = json.loads(text)
            m["final"].update(kw)
            return json.dumps(m)
        return change

    # ---- the report of the dry job, made again in the scratch copy; every file it opens is recorded
    os.makedirs(os.path.join(tmp, "trace"))
    trace = os.path.join(tmp, "trace", "opened.txt")
    r = sh("nice", "-n", "19", PY, "-I", "-c", TRACER, trace, os.path.join("src", "e5_report.py"), "--dry", "--frozen", RECORD)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads(r.stdout)
    with open(os.path.join(ROOT, "logs", "e5_dry_report.json")) as f:
        job = json.load(f)
    assert res["checks_all_passed"] and all(res["checks"].values()) and res["dry"] and res["split"] == "dev"
    assert len(res["checks"]) == 4 + 3 * 13, len(res["checks"])
    assert {k: v for k, v in res.items() if k != "report_utc"} == {k: v for k, v in job.items() if k != "report_utc"}, \
        "the report in the scratch copy equals the report of the dry job"
    ok("the report of the dry job passes again in a scratch copy and gives the same numbers")
    assert set(res["rules"]) == set(e5.RULES) and res["primary_reading"] == e5.PRIMARY_READING == "unit"
    for name, r in res["rules"].items():
        c = res["readings"][r["data"]]["unit"]["comparisons"]["stock"]
        assert r["met"] == (c["ci95"][0] > 0 and c["ci95_seeds_and_articles"][0] > 0), name
    for d in e5.DATA:
        for reading in (e5.PRIMARY_READING, e5.OTHER_READING):
            for k, v in res["readings"][d][reading]["comparisons"].items():
                assert "ci95" not in v or {"ci95_seeds_and_articles", "ci95_two_way"} <= set(v), (d, reading, k)
        assert res["study_numbers"][d]["equal_to_the_printed_digits"], (d, res["study_numbers"][d])
        assert set(res["readings_without_duplicates"][d]) == {"whole_copy_articles", "articles_with_a_duplicated_paragraph"}
    assert "readings_failed" not in res and len(res["groups"]) == 3
    ok("the rules need both intervals; every step ran; the baselines and stock seeds are the study's DEV numbers")
    for d, rel in runs.items():
        dev = sorted(glob.glob(os.path.join(ROOT, "runs", f"*_e5dev_{d}_TB_s1")))
        if len(dev) != 1:
            continue  # the DEV rows of the arm are not distributed
        with open(os.path.join(dev[0], "metrics.json")) as f:
            want = json.load(f)["final"]
        row = next(x for x in res["rows"] if x["data"] == d)
        for k, key in (("auc_unit", "dev_auc"), ("auc_ratio", "dev_auc_ratio"), ("auc_weight", "dev_auc_weight"),
                       ("auc_logodds", "dev_auc_logodds"), ("tuned_acc_unit", "dev_tuned_acc"),
                       ("partner_shuffle_auc", "partner_shuffle_auc"), ("train_auc", "train_auc"), ("members", "members")):
            assert row[k] == want[key], (d, k, row[k], want[key])
        a, b = np.load(os.path.join(study, rel, "state.npz")), np.load(os.path.join(dev[0], "state.npz"))
        assert all(np.array_equal(a[k], b[k]) for k in b.files), d
    ok("the dry runs are the DEV rows of seed 1 (numbers and model arrays equal)")
    with open(trace) as f:  # the hook also sees attempts on files that do not exist (compiled caches); those are left out
        opened = [p for p in f.read().split("\n") if p.startswith(tmp + os.sep) and p != trace and os.path.isfile(p)]
    pinned = {os.path.normpath(os.path.join(study, k)) for k in json.loads(read(RECORD))["extras_sha256"]}
    outside = [p for p in opened if not p.startswith(study + os.sep)]
    assert outside and set(outside) <= pinned, sorted(set(outside) - pinned)
    inside = {os.path.relpath(p, study) for p in opened if p.startswith(study + os.sep)}
    allowed = ("src/", "tests/", "registrations/", "env/", "runs_dry/", "verification/", "logs/RUNS_wiki.md", "logs/e5_dry/", ".venv/")
    assert all(p.startswith(allowed) for p in inside), sorted(p for p in inside if not p.startswith(allowed))
    assert os.path.join(tmp, "tm_language_model", "data", "concept_eval", "nltk_data", "corpora", "wordnet.zip") in opened
    assert not [p for p in opened if p.endswith(("/test.npz", "scores_test.npz"))], "no file of the test split"
    ok(f"every file the report opens outside the scratch study is pinned ({len(set(outside))} files)")

    # ---- the checker: one broken thing at a time
    status, res0 = report()
    assert status == 0 and res0["checks_all_passed"] and "rules" not in res0, "checks only: no rule is printed"
    cfg = {d: f"{rel}/config.json" for d, rel in runs.items()}
    met = {d: f"{rel}/metrics.json" for d, rel in runs.items()}
    broken("a setting other than the registered one", "sa_seed1_setting", cfg["sa"], json_change(clauses=4000))
    broken("1 in place of True", "np_seed1_setting", cfg["np"], json_change(tied=1))
    broken("the split scored at every pass", "ns_all_seed1_setting", cfg["ns_all"], json_change(eval_every=1))
    broken("a run of another data version", "sa_seed1_data_seed_name_split", cfg["sa"], json_change(data="np"))
    broken("a run under another seed", "np_seed1_data_seed_name_split", cfg["np"], json_change(seed=2))
    broken("a run under another name", "ns_all_seed1_data_seed_name_split", cfg["ns_all"], json_change(name="e5dev_ns_all_TB_s1"))
    broken("a run that was not written to the run log", "sa_seed1_logged_in_default_folder", cfg["sa"], json_change(no_log=True))
    broken("a run written to another folder", "np_seed1_logged_in_default_folder", cfg["np"], json_change(out_root="runs"))
    broken("a run whose row is missing from the run log", "ns_all_seed1_logged_in_default_folder", "logs/RUNS_wiki.md",
           lambda t: t.replace(f"| {os.path.basename(runs['ns_all'])} |", "| a row of another run |"))
    broken("a run that names another record", "sa_seed1_names_this_record", cfg["sa"],
           json_change(registered="registrations/E5_wikipedia_test.frozen.json"))
    broken("a run made with another engine", "np_seed1_engine_of_record", cfg["np"], json_change(orgroup_tm_sha256="0" * 64))
    broken("a run made with other data", "ns_all_seed1_data_of_record", cfg["ns_all"],
           lambda t: json.dumps({**json.loads(t), "data_sha256": {**json.loads(t)["data_sha256"], "ns_all/train.npz": "0" * 64}}))
    broken("a run made with another word list", "sa_seed1_data_of_record", cfg["sa"],
           lambda t: json.dumps({**json.loads(t), "data_sha256": {**json.loads(t)["data_sha256"], "vocab_F1024.json": "0" * 64}}))
    broken("a run made with other packages", "np_seed1_environment_of_the_run", cfg["np"],
           lambda t: json.dumps({**json.loads(t), "versions": {**json.loads(t)["versions"], "packages": {
               **json.loads(t)["versions"]["packages"], "scikit-learn": "0.1"}}}))
    broken("a run made with another interpreter", "sa_seed1_environment_of_the_run", cfg["sa"],
           lambda t: json.dumps({**json.loads(t), "versions": {**json.loads(t)["versions"], "executable": "../other/bin/python"}}))
    broken("an edited row of the run log from before the freeze", "run_log_prefix_unchanged", "logs/RUNS_wiki.md",
           lambda t: t.replace("| 20261112T174840Z_e5dev_sa_TB_s1 |", "| 20261112T174840Z_e5dev_sa_TB_s9 |"))
    broken("a run that started before the freeze", "ns_all_seed1_started_after_freeze", cfg["ns_all"],
           json_change(run_start_utc="20200101T000000Z"))
    broken("another number of scored pairs", "sa_seed1_pair_counts", met["sa"], final_change(n_eval=10000))
    broken("another number of training pairs", "np_seed1_pair_counts", met["np"], final_change(n_train=39999))

    def scored_twice(text):
        m = json.loads(text)
        m["history"][4]["dev_auc"] = 0.8
        return json.dumps(m)
    broken("the split scored before the last pass", "ns_all_seed1_scored_once_after_the_last_pass", met["ns_all"], scored_twice)
    broken("a partner-shuffle AUC above 0.53", "sa_seed1_partner_shuffle_in_bounds", met["sa"], final_change(partner_shuffle_auc=0.5301))
    broken("a partner-shuffle AUC below 0.47", "np_seed1_partner_shuffle_in_bounds", met["np"], final_change(partner_shuffle_auc=0.4699))
    broken("a logged AUC that the score file does not give", "ns_all_seed1_score_file_of_this_run", met["ns_all"],
           final_change(dev_auc_ratio=0.9))

    sc = os.path.join(study, runs["sa"], "scores_dev.npz")
    z = dict(np.load(sc))
    os.rename(sc, sc + ".away")
    for name, arrays in (("a score file with other labels", {**z, "y": z["y"][::-1].copy()}),
                         ("a score file with other scores", {**z, "ratio": z["ratio"][::-1].copy()})):
        with open(sc, "wb") as f:
            np.savez(f, **arrays)
        status, res = report()
        assert status == 1 and [k for k, v in res["checks"].items() if not v] == ["sa_seed1_score_file_of_this_run"], name
        ok(name)
    os.remove(sc)
    os.rename(sc + ".away", sc)

    broken("a changed source file", "record_plan_sources_and_pinned_files", "src/metrics.py", lambda t: t + "\n")
    broken("a changed test file", "record_plan_sources_and_pinned_files", "tests/test_e5_units.py", lambda t: t + "\n")
    broken("a changed plan text", "record_plan_sources_and_pinned_files", "registrations/dry/E5_dry.md", lambda t: t + "\n")
    broken("a changed job script", "record_plan_sources_and_pinned_files", "registrations/E5_job.sh", lambda t: t + "\n")
    broken("a changed package list", "record_plan_sources_and_pinned_files", "env/requirements.txt", lambda t: t + "\n")
    broken("a pinned file that is missing from the record", "record_plan_sources_and_pinned_files", RECORD,
           lambda t: json.dumps({**json.loads(t), "extras_sha256": {k: v for k, v in json.loads(t)["extras_sha256"].items()
                                                                    if not k.endswith("wordnet.zip")}}))
    first_stock = f"{e5.H}/results/{e5.STOCK_DEV['np'][0]}/scores_dev.npz"
    broken("a stock score file that is not the pinned one", "record_plan_sources_and_pinned_files", RECORD,
           lambda t: json.dumps({**json.loads(t), "extras_sha256": {**json.loads(t)["extras_sha256"], first_stock: "0" * 64}}))
    broken("a record of another plan", "record_plan_sources_and_pinned_files", RECORD,
           json_change(plan="registrations/E4_wikipedia_design.md"))

    os.makedirs(os.path.join(study, "src", "extra"))
    with open(os.path.join(study, "src", "extra", "new.py"), "w") as f:
        f.write("X = 1\n")
    status, res = report()
    shutil.rmtree(os.path.join(study, "src", "extra"))
    assert status == 1 and [k for k, v in res["checks"].items() if not v] == ["record_plan_sources_and_pinned_files"]
    ok("a new source file")
    twin = os.path.join(study, "runs_dry", "20200101T000000Z_" + e5.run_name("np", 1, True))
    shutil.copytree(os.path.join(study, runs["np"]), twin)
    status, res = report()
    shutil.rmtree(twin)
    assert status == 1 and [k for k, v in res["checks"].items() if not v] == ["np_seed1_one_complete_run"]
    ok("two run folders for one data version and seed")
    for f in ("state.npz", "scores_dev.npz", "groups.txt"):
        p = os.path.join(study, runs["ns_all"], f)
        os.rename(p, p + ".away")
        status, res = report()
        os.rename(p + ".away", p)
        assert status == 1 and [k for k, v in res["checks"].items() if not v] == ["ns_all_seed1_one_complete_run"], f
    ok("an incomplete run folder (no model, no scores, no groups)")
    for rel, how in (("logs/tests/x/20200101T000000Z_any", "config"), ("runs_dry/20200101T000000Z_other", "scores"),
                     ("runs/20261112T000000Z_e5_sa_s11", "config")):
        p = os.path.join(study, rel)
        os.makedirs(p)
        if how == "config":
            with open(os.path.join(p, "config.json"), "w") as f:
                json.dump({"split": "test"}, f)
        else:
            np.savez(os.path.join(p, "scores_test.npz"), y=np.zeros(2))
        status, res = report()
        shutil.rmtree(p)
        assert status == 1 and [k for k, v in res["checks"].items() if not v] == ["no_test_scoring_outside_the_registered_runs"], rel
        assert res["test_scoring_outside_the_registered_runs"] == [rel], res["test_scoring_outside_the_registered_runs"]
    for d in ("logs/tests/x", "runs"):
        shutil.rmtree(os.path.join(study, d), ignore_errors=True)
    ok("a scoring on the test pairs outside the registered runs (also in the place of a registered run, in dry mode)")
    status = script("e5_report.py", "--checks-only", "--frozen", RECORD).returncode
    assert status == 1, "the dry record is not accepted as the record of the registration"
    ok("the dry record given as the record of the registration")
    before = read(cfg["sa"])
    write(cfg["sa"], json_change(K=2)(before))
    full = script("e5_report.py", "--dry", "--steps", "rules", "--frozen", RECORD, "--out", os.path.join("logs", "x_report.json"))
    write(cfg["sa"], before)
    res = json.loads(full.stdout)
    left = [f for f in os.listdir(os.path.join(study, "logs")) if f.startswith("x_report.json")]
    assert full.returncode == 1 and len(left) == 1 and ".FAILED_" in left[0], left
    assert res["registered_run_void"] == ["registered run void: sa_seed1_setting"] and set(res["rules"]) == set(e5.RULES)
    assert res["sentences"][0].startswith("registered run void: sa_seed1_setting.") and len(res["rows"]) == 3
    ok("a report with a failed check still gives every number, under 'registered run void', in a file of its own")
    status, res = report()
    assert status == 0 and res["checks_all_passed"], "everything was put back"

    # ---- the guards of the runner and of the report tool, with a valid dry record in place
    def run_argv(data, seed, **change):
        a = {"data": data, "seed": seed, **e5.SETTING, "split": "dev", "registered": RECORD,
             "name": e5.run_name(data, seed, True), "out_root": e5.RUNS_ROOT[True], **change}
        out = []
        for k, v in a.items():
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                out += [flag] if v else []
            elif v is not None:
                out += [flag, str(v)]
        return out
    for name, argv, message in (
            ("a second run of a seed is refused", run_argv("sa", 1), "exists already"),
            ("a seed that is not registered is refused", run_argv("np", 2), "not a registered seed"),
            ("another setting is refused", run_argv("np", 2, K=2), "setting K"),
            ("another reading of the split is refused", run_argv("np", 2, eval_every=1), "setting eval_every"),
            ("another name is refused", run_argv("np", 2, name="anything"), "name must be"),
            ("the test pairs are refused with the dry record", run_argv("np", 2, split="test"), "does not open the test pairs")):
        r = script("run_wiki.py", *argv)
        assert r.returncode != 0 and message in r.stderr, (name, r.stderr[-800:])
        ok(name)
    r = script("run_wiki.py", *run_argv("sa", 11, split="test", registered="registrations/E5_wikipedia_test.frozen.json",
                                        name="e5_sa_s11", out_root="runs"))
    assert r.returncode != 0 and "no record file" in r.stderr, r.stderr[-800:]
    ok("the test pairs are refused while the registration has no record")
    for argv, message in ((["--registered", RECORD], "does not open the test pairs"), ([], "needs --registered")):
        r = script("wiki_report.py", "--data", "sa", "--split", "test", "--runs", runs["sa"], *argv)
        assert r.returncode != 0 and message in r.stderr, r.stderr[-800:]
    ok("the report tool refuses the test pairs with the dry record and without a record")
    write("src/metrics.py", read("src/metrics.py") + "\n")
    r = script("run_wiki.py", *run_argv("np", 1))
    assert r.returncode != 0 and "source differs from the record: src/metrics.py" in r.stderr, r.stderr[-800:]
    ok("a registered run is refused after a source file changed")
    assert sorted(os.listdir(os.path.join(study, "runs_dry"))) == sorted(os.path.basename(p) for p in runs.values())
    assert not os.path.exists(os.path.join(study, "runs"))
    for p in runs.values():
        assert not [f for f in os.listdir(os.path.join(study, p)) if "test" in f], p
    ok("no refused call left a run folder or a file of the test split")


if __name__ == "__main__":
    main()
