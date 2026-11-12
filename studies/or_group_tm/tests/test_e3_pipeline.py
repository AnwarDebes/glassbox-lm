"""End-to-end test of the E3 pipeline (registrations/E3_job.sh, src/run_slide10.py, src/probe_slide10.py,
src/e3_report.py) in a scratch copy of the study folder: the dry job is run there on the DEV sentences, the report
must pass, and then one thing at a time is broken and the checker must say so. Nothing in the study folder itself
is written, and no EVAL sentence is scored.

Run from the study folder (about three minutes on one core):
    CUDA_VISIBLE_DEVICES= .venv/bin/python -I tests/test_e3_pipeline.py
Prints one line per case and 'PIPELINE TEST PASSED (n cases)', or stops with exit status 1 at the first failure.
Skipped when the slide-10 data of the hierarchical_tm study (not distributed) are missing.

Regression values: seeds 1 to 5 of the dry job are the seeds of the design rows of arm A6 (logs/RUNS_slide10.md,
2026-11-12), so the dry report must show the numbers of those rows; the deletion and one-word shares were computed
once more by separate code when the registration was checked.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HSTUDY = os.path.normpath(os.path.join(ROOT, "..", "hierarchical_tm"))
PY = sys.executable
CASES = []

A6_DEV = {"coverage_288": [1.0, 1.0, 0.9604, 0.9406, 1.0], "coverage_216": [1.0, 1.0, 1.0, 0.9143, 1.0],
          "fires_max": [0.0] * 5, "brackets_288": [5, 4, 5, 4, 5], "brackets_216": [4, 3, 4, 3, 4],
          "best_rule_accepts_deletions": [0.0] * 5, "model_accepts_deletions": [0.2277, 0.1386, 0.1469, 0.2838, 0.2574],
          "best_rule_accepts_one_word": [0.918, 0.918, 0.875, 0.863, 0.918], "template_like_clauses": [3, 3, 5, 1, 3],
          "tied_best_rules": [3, 3, 4, 1, 3]}


def ok(name):
    CASES.append(name)
    print("ok  ", name, flush=True)


def main():
    if not os.path.isfile(os.path.join(HSTUDY, "data", "slide10", "pairs_slide10.npz")):
        print("PIPELINE TEST SKIPPED (the slide-10 data are not distributed)")
        return
    tmp = tempfile.mkdtemp(prefix="e3_pipeline_")
    study = os.path.join(tmp, "or_group_tm")
    try:
        run_all(tmp, study)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"PIPELINE TEST PASSED ({len(CASES)} cases)")


def run_all(tmp, study):
    os.symlink(HSTUDY, os.path.join(tmp, "hierarchical_tm"))
    for folder in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, folder), os.path.join(study, folder), ignore=shutil.ignore_patterns("__pycache__"))
    for rel in ("registrations/E3_job.sh", "registrations/dry/E3_dry.md", "env/requirements.txt"):
        os.makedirs(os.path.dirname(os.path.join(study, rel)), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), os.path.join(study, rel))
    env = {**os.environ, "E3_PYTHON": PY, "CUDA_VISIBLE_DEVICES": ""}
    record = "registrations/dry/E3_dry.frozen.json"

    def sh(*cmd):
        return subprocess.run(cmd, check=False, cwd=study, env=env, capture_output=True, text=True)

    def script(name, *args):
        return sh("nice", "-n", "19", PY, "-I", os.path.join("src", name), *args)

    def report():
        r = script("e3_report.py", "--dry", "--frozen", record)
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

    def broken(name, check, rel, change, rows_gone=True):
        """Change one file, expect the named check to fail, put the file back."""
        before = read(rel)
        write(rel, change(before))
        status, res = report()
        write(rel, before)
        assert status == 1 and res["checks"][check] is False and res["checks_all_passed"] is False, (name, status, res["checks"])
        assert [k for k, v in res["checks"].items() if not v] == [check], (name, res["checks"])
        assert ("rows" not in res and "sentences" not in res) or not rows_gone, name
        ok(name)

    def cfg_change(**kw):
        return lambda text: json.dumps({**json.loads(text), **kw})

    # ---- the dry job, from the tests to the report
    job = sh("bash", "registrations/E3_job.sh", "dry")
    assert job.returncode == 0, job.stdout[-3000:] + job.stderr[-3000:]
    assert "finished with exit code 0" in job.stdout
    res = json.loads(read("logs/e3_dry_report.json"))
    assert res["checks_all_passed"] and res["dry"] and res["split"] == "dev" and all(res["checks"].values())
    assert len(res["checks"]) == 2 + 5 * 14, len(res["checks"])
    assert [r["seed"] for r in res["rows"]] == [1, 2, 3, 4, 5] and [r["seed"] for r in res["extra_rows"]] == [6, 7]
    ok("the dry job runs from the tests to a passing report")
    for key, want in A6_DEV.items():
        got = [r[key] for r in res["rows"]]
        assert all(abs(g - w) < 6e-4 for g, w in zip(got, want)), (key, got, want)
    cnt = res["counts_registered_seeds"]
    assert cnt["both_flags"]["seeds"] == 3 and cnt["reading_288"]["seeds"] == 5 and cnt["flag_288"]["seeds"] == 5
    assert cnt["flag_216"]["seeds"] == 3 and cnt["model_template_like"]["seeds"] == 0 and cnt["best_rule_template_like"]["seeds"] == 5
    assert res["R1_both_flags_met"] is True and res["extra_seeds_checks_passed"] is True
    assert res["counts_all_seeds"]["n_seeds"] == 7 and len(res["sentences"]) == 4
    ok("the dry report shows the design rows of arm A6 and the recomputed probe shares")
    names = sorted(os.listdir(os.path.join(study, "runs_dry")))
    assert len(names) == 7 and not os.path.exists(os.path.join(study, "runs"))
    run = {s: f"runs_dry/{n}" for n in names for s in range(1, 8) if n.endswith(f"_e3dry_text_s{s}")}
    log = read("logs/RUNS_slide10.md")
    assert all(n in log for n in names), "every run of the job is in the run log"
    for f in os.listdir(os.path.join(study, "runs_dry", names[0])):
        assert "eval" not in f, f
    sums = read("logs/e3_dry/SHA256SUMS")
    assert "logs/e3_dry_report.json" in sums and "run_s7.out" in sums and "probe_s1.out" in sums
    ok("seven run folders, the run log and the hashes of the outputs are written")
    again = sh("bash", "registrations/E3_job.sh", "dry")
    assert again.returncode != 0 and "run folder exists, not repeated" in again.stdout, "a second job repeats no run"
    assert sorted(os.listdir(os.path.join(study, "runs_dry"))) == names
    assert read("logs/e3_dry_report.json") == json.dumps(res, indent=1) + "\n", "the first report is not overwritten"
    ok("a second job repeats no run and does not overwrite the report")

    # ---- the checker: one broken thing at a time
    status, res0 = report()
    assert status == 0 and res0["checks_all_passed"]
    broken("a setting other than the registered one", "seed1_setting", run[1] + "/config.json", cfg_change(K=16))
    broken("a run that was not written to the run log", "seed2_logged_in_default_folder", run[2] + "/config.json", cfg_change(no_log=True))
    broken("a run under another seed", "seed3_seed_name_split", run[3] + "/config.json", cfg_change(seed=4))
    broken("a run that started before the freeze", "seed4_started_after_freeze", run[4] + "/config.json",
           cfg_change(run_start_utc="20200101T000000Z"))
    broken("a run that names another record", "seed5_names_this_record", run[5] + "/config.json",
           cfg_change(registered="registrations/E3_slide10_text.frozen.json"))
    broken("a run made with another engine", "seed1_engine_of_record", run[1] + "/config.json", cfg_change(orgroup_tm_sha256="0" * 64))
    broken("a run made with other data", "seed2_data_of_record", run[2] + "/config.json",
           lambda t: json.dumps({**json.loads(t), "data": {**json.loads(t)["data"], "vocab_slide10.json": "0" * 64}}))
    broken("a run scored with another scorer", "seed3_scorer_of_record", run[3] + "/config.json", cfg_change(htm_driver_py_sha256="0" * 64))

    def metrics_change(fn):
        def change(text):
            m = json.loads(text)
            fn(m["final"])
            return json.dumps(m)
        return change
    broken("a scorer that saw another number of sentences", "seed4_sentence_counts", run[4] + "/metrics.json",
           metrics_change(lambda f: f["check2b"].update(n_dev_broken=972)))
    broken("an export that was not checked against the model", "seed5_export_equals_model", run[5] + "/metrics.json",
           metrics_change(lambda f: f.update(export_equals_model_on_train_and_dev=None)))
    broken("a probe file of another run", "seed1_probe_of_this_run", run[1] + "/probe_dev.json", cfg_change(run="another"))
    broken("a probe that scored another clause", "seed2_probe_intact_equals_coverage", run[2] + "/probe_dev.json",
           cfg_change(best_rule_accepts_intact=0.5))
    broken("a changed source file", "record_plan_sources_and_pinned_files", "src/metrics.py", lambda t: t + "\n")
    broken("a changed plan text", "record_plan_sources_and_pinned_files", "registrations/dry/E3_dry.md", lambda t: t + "\n")
    broken("a changed job script", "record_plan_sources_and_pinned_files", "registrations/E3_job.sh", lambda t: t + "\n")
    broken("a pinned file that is missing from the record", "record_plan_sources_and_pinned_files", record,
           lambda t: json.dumps({**json.loads(t), "extras_sha256": {k: v for k, v in json.loads(t)["extras_sha256"].items()
                                                                    if k != "env/requirements.txt"}}))
    broken("a record of another plan", "record_plan_sources_and_pinned_files", record,
           lambda t: json.dumps({**json.loads(t), "plan": "registrations/E2_slide10_design.md"}))

    probe = os.path.join(study, run[3], "probe_dev.json")
    os.rename(probe, probe + ".away")
    status, res = report()
    os.rename(probe + ".away", probe)
    assert status == 1 and res["checks"]["seed3_probe_file"] is False and "rows" not in res
    ok("a missing probe file")
    twin = os.path.join(study, "runs_dry", "20200101T000000Z_e3dry_text_s2")
    shutil.copytree(os.path.join(study, run[2]), twin)
    status, res = report()
    shutil.rmtree(twin)
    assert status == 1 and res["checks"]["seed2_one_complete_run"] is False and "rows" not in res
    ok("two run folders for one seed")
    state = os.path.join(study, run[4], "state.npz")
    os.rename(state, state + ".away")
    status, res = report()
    os.rename(state + ".away", state)
    assert status == 1 and res["checks"]["seed4_one_complete_run"] is False
    ok("an incomplete run folder")
    stray = os.path.join(study, "logs", "tests", "x", "20200101T000000Z_any")
    os.makedirs(stray)
    with open(os.path.join(stray, "config.json"), "w") as f:
        json.dump({"split": "eval"}, f)
    status, res = report()
    shutil.rmtree(os.path.join(study, "logs", "tests", "x"))
    assert status == 1 and res["checks"]["no_eval_scoring_outside_the_registered_runs"] is False and "rows" not in res
    assert res["eval_scoring_outside_the_registered_runs"] == ["logs/tests/x/20200101T000000Z_any"]
    ok("a scoring on the EVAL sentences outside the registered runs")
    status, res = script("e3_report.py", "--frozen", record).returncode, None
    assert status == 1, "the dry record is not accepted as the record of the registration"
    ok("the dry record given as the record of the registration")
    failed_out = os.path.join(study, "logs", "x_report.json")
    before = read(run[1] + "/config.json")
    write(run[1] + "/config.json", cfg_change(K=16)(before))
    r = script("e3_report.py", "--dry", "--frozen", record, "--out", failed_out)
    write(run[1] + "/config.json", before)
    left = [f for f in os.listdir(os.path.join(study, "logs")) if f.startswith("x_report.json")]
    assert r.returncode == 1 and len(left) == 1 and ".FAILED_" in left[0], left
    ok("a failed report is written under a name of its own")

    # a further seed that fails a check: the registered seeds keep their rule, the further seeds are left out
    before = read(run[6] + "/config.json")
    write(run[6] + "/config.json", cfg_change(K=16)(before))
    status, res = report()
    write(run[6] + "/config.json", before)
    assert status == 0 and res["checks_all_passed"] and res["extra_seeds_checks_passed"] is False
    assert res["extra_seeds_void"] == ["seed6_setting"] and "counts_all_seeds" not in res and "extra_rows" not in res
    ok("a further seed that fails a check is left out and named")
    status, res = report()
    assert status == 0 and res["checks_all_passed"] and res["rows"] == res0["rows"], "everything was put back"

    # ---- the guards of the runner and of the probe, with a valid dry record in place
    base = ["--version", "text", "--seed-all", "--registered", record, "--out-root", "runs_dry"]
    for name, extra, message in (
            ("a second run of a seed is refused", ["--seed", "1", "--name", "e3dry_text_s1"], "exists already"),
            ("a seed that is not registered is refused", ["--seed", "8", "--name", "e3dry_text_s8"], "not a registered seed"),
            ("another setting is refused", ["--seed", "8", "--name", "e3dry_text_s8", "--K", "16"], "setting K"),
            ("another name is refused", ["--seed", "6", "--name", "anything"], "name must be"),
            ("the EVAL sentences are refused with the dry record", ["--seed", "11", "--name", "e3_text_s11", "--split", "eval"],
             "does not open the EVAL sentences")):
        r = script("run_slide10.py", *base, *extra)
        assert r.returncode != 0 and message in r.stderr, (name, r.stderr[-500:])
        ok(name)
    r = script("run_slide10.py", "--version", "text", "--seed-all", "--seed", "11", "--name", "e3_text_s11", "--split", "eval",
               "--registered", "registrations/E3_slide10_text.frozen.json")
    assert r.returncode != 0 and "no record file" in r.stderr
    ok("the EVAL sentences are refused while the registration has no record")
    r = script("probe_slide10.py", run[1], "--split", "eval")
    assert r.returncode != 0 and "the EVAL sentences are refused" in r.stderr
    ok("the probe refuses the EVAL sentences for a run of the dry job")
    write("src/metrics.py", read("src/metrics.py") + "\n")
    r = script("run_slide10.py", *base, "--seed", "8", "--name", "e3dry_text_s8")
    assert r.returncode != 0 and "source differs from the record: src/metrics.py" in r.stderr
    ok("a registered run is refused after a source file changed")
    assert sorted(os.listdir(os.path.join(study, "runs_dry"))) == names and not os.path.exists(os.path.join(study, "runs"))
    for d in names:
        for f in os.listdir(os.path.join(study, "runs_dry", d)):
            assert "eval" not in f, (d, f)
    ok("no refused call left a run folder or a file of the EVAL split")


if __name__ == "__main__":
    main()
