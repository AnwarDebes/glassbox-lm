"""The guards and checks of the A2 dev-test look in scripts/a2_run.py, tested without ever starting a look.

Run: python tests/test_a2_args.py

The script is imported as a module (its main() does not run on import). No case starts
scripts/a2_run.py as a program, and no case can read data: before main() is called, the run
logger (`Run`) and the data loader (`ptb.load`) of the imported module are replaced by sentinels,
so a broken guard shows as a failed check, never as a look. Every case passes before and after the
freeze of registrations/A2.md.
1. final_guard() on a fake study (a temporary folder laid out like the real one, with small
   stand-in files, a copy of the real lock file of .venv-tm and a freeze record of their hashes):
   the registered command is allowed; each departure is refused: dev-test without --final, --final
   on valid, --smoke, --name, another or no claim code, another or no registration, a missing
   freeze record, a changed src file, a src file the record lacks, a changed or unpinned code file,
   a changed data file, a changed script, a changed registration, an earlier look's folder, a
   thread setting other than the development runs', NLTK_DATA set, an installed version that
   differs from the lock file.
2. main() on the real study with dev-test arguments that no state of the study allows (dev-test
   without --final, --final without a claim code and registration, --final on valid) is refused
   before `Run` and `ptb.load`.
3. main() on the fake study with the registered command passes the guard and reaches `Run`
   first (the sentinel), before any data access.
4. get_code() in the look refuses to build a code whose saved copy does not match its source.
5. The look's equality checks: the cache grid and choice, every arm's weights and EM steps of the
   second development run (results/20261025T123159Z_a2_valid_dev/metrics.json) against the
   pinned first one (EXPECT_METRICS) give no difference, also after the rounding the look applies
   to numpy weights; a perturbed grid value, cache choice, weight or step count, or a missing arm,
   is refused.
6. A stop after the run folder exists: main() on the fake study with a stand-in run logger and a
   data loader that stops (SystemExit) records the state "stopped", the reason and that the eval
   split was not scored, and stops; a crash records eval_scored and propagates.
Also checks that no run folder appears in the real results/.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
SCRIPT = os.path.join(STUDY, "scripts", "a2_run.py")
DEV2 = os.path.join(STUDY, "results", "20261025T123159Z_a2_valid_dev", "metrics.json")
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def load_module():
    spec = importlib.util.spec_from_file_location("a2_run_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Reached(Exception):
    pass


def sentinels(m):
    def run(*a, **k):
        raise Reached("Run")

    def load(*a, **k):
        raise Reached("ptb.load")
    m.Run = run
    m.ptb.load = load


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def fake_study(m, root):
    """A study folder at root/tally with every file the guard checks, and its freeze record."""
    study = os.path.join(root, "tally")
    write(os.path.join(study, m.REGISTRATION), "fake registration\n")
    for p in ("src/tally/__init__.py", "src/tally/ptb.py", "src/tally/floor/kn.py", "src/tally/b1/README.md"):
        write(os.path.join(study, p), f"stand-in for {p}\n")
    extras = [os.path.join("scripts", "a2_run.py"), m.EXPECT_METRICS] + m.code_files() + m.data_files_of()
    for p in extras:
        write(os.path.normpath(os.path.join(study, p)), f"stand-in for {p}\n")
    write(os.path.join(study, m.EXPECT_METRICS), json.dumps(      # the look reads this one as JSON
        {"cache": {"chosen": {}, "grid": []}, "arms": {"none": {"lam": [1.0], "em_steps": 1}}}))
    lock = os.path.normpath(os.path.join(study, m.LOCK_FILE))
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    shutil.copyfile(os.path.normpath(os.path.join(STUDY, m.LOCK_FILE)), lock)
    rec = {"frozen_utc": "2026-10-25T00:00:00Z",
           "registration": {m.REGISTRATION: sha(os.path.join(study, m.REGISTRATION))},
           "src": m.src_hashes(study),
           "extra": {p: sha(os.path.normpath(os.path.join(study, p))) for p in extras}}
    write(os.path.join(study, "registrations", "A2.frozen.json"), json.dumps(rec))
    return study, extras


def ns(**kw):
    d = dict(split="devtest", final=True, smoke=False, name=None, claim_arm="exchange", registration=None)
    d.update(kw)
    return argparse.Namespace(**d)


class FakeRun:
    """Stands in for runlog.Run: keeps meta and printed lines in memory, writes nothing."""
    last = None

    def __init__(self, name, cfg, data_files=(), registration=None):
        self.meta, self.lines, self.dumps, self.t0 = {"state": "running"}, [], [], 0.0
        FakeRun.last = self

    def print(self, *a):
        self.lines.append(" ".join(str(x) for x in a))

    def _dump(self, name, obj):
        self.dumps.append((name, dict(obj)))


def main():
    results = os.path.join(STUDY, "results")
    before = set(os.listdir(results)) if os.path.isdir(results) else set()
    m = load_module()
    sentinels(m)
    for k in m.THREAD_VARS:                       # the development runs' setting
        os.environ[k] = m.LOOK_THREADS
    os.environ.pop("NLTK_DATA", None)
    with tempfile.TemporaryDirectory() as root:
        study, extras = fake_study(m, root)
        reg = os.path.join(study, m.REGISTRATION)
        ok = m.final_guard(ns(registration=reg), study)
        check("the registered command on an unchanged frozen study is allowed", ok == [], f"{ok}")
        cases = [
            ("dev-test without --final", ns(final=False, registration=reg), "only with --final"),
            ("--final on valid", ns(split="valid", registration=reg), "only for --split devtest"),
            ("--smoke", ns(smoke=True, registration=reg), "no smoke"),
            ("--name", ns(name="a2_other", registration=reg), "no --name"),
            ("another claim code", ns(claim_arm="kind_topic", registration=reg), "--claim-arm exchange"),
            ("no claim code", ns(claim_arm=None, registration=reg), "--claim-arm exchange"),
            ("another registration", ns(registration=os.path.join(study, "registrations", "T0.md")),
             "--registration"),
            ("no registration", ns(), "--registration"),
        ]
        for name, a, msg in cases:
            e = m.final_guard(a, study)
            check(f"refused: {name}", any(msg in x for x in e), f"{e}")

        def mutate(path, text, name, msg):
            old = open(path).read() if os.path.exists(path) else None
            write(path, text)
            e = m.final_guard(ns(registration=reg), study)
            check(f"refused: {name}", any(msg in x for x in e), f"{e}")
            if old is None:
                os.remove(path)
            else:
                write(path, old)
        mutate(os.path.join(study, "src/tally/floor/kn.py"), "changed\n", "a changed src file",
               "src differs from the freeze record: src/tally/floor/kn.py")
        mutate(os.path.join(study, "src/tally/floor/new.py"), "new\n", "a src file the record lacks",
               "src differs from the freeze record: src/tally/floor/new.py")
        mutate(os.path.join(study, "data/a2_codes/exchange.npy"), "changed\n", "a changed code file",
               "changed after the freeze: data/a2_codes/exchange.npy")
        mutate(os.path.normpath(os.path.join(study, m.data_files_of()[0])), "changed\n", "a changed data file",
               "changed after the freeze")
        mutate(os.path.join(study, "scripts", "a2_run.py"), "changed\n", "a changed script",
               "changed after the freeze: scripts/a2_run.py")
        mutate(reg, "changed\n", "a changed registration", "changed after the freeze: registrations/A2.md")
        lock = os.path.normpath(os.path.join(study, m.LOCK_FILE))
        lock_text = open(lock).read()
        mutate(lock, lock_text.replace("numpy==", "numpy==0.0.0-not-"), "an installed version that differs from the lock",
               "numpy")
        os.environ["OMP_NUM_THREADS"] = "4"
        e = m.final_guard(ns(registration=reg), study)
        check("refused: a thread setting other than the development runs'", any("OMP_NUM_THREADS" in x for x in e),
              f"{e}")
        os.environ["OMP_NUM_THREADS"] = m.LOOK_THREADS
        os.environ["NLTK_DATA"] = "/somewhere"
        e = m.final_guard(ns(registration=reg), study)
        check("refused: NLTK_DATA set", any("NLTK_DATA" in x for x in e), f"{e}")
        os.environ.pop("NLTK_DATA")
        os.makedirs(os.path.join(study, "results", "20261026T000000Z_a2_devtest_final"))
        e = m.final_guard(ns(registration=reg), study)
        check("refused: an earlier look's folder exists", any("exists already" in x for x in e), f"{e}")
        os.rmdir(os.path.join(study, "results", "20261026T000000Z_a2_devtest_final"))
        frozen = os.path.join(study, "registrations", "A2.frozen.json")
        rec = json.load(open(frozen))
        del rec["extra"]["data/a2_codes/topic.json"]
        write(frozen, json.dumps(rec))
        e = m.final_guard(ns(registration=reg), study)
        check("refused: a code file not pinned", any("not pinned as an extra file: data/a2_codes/topic.json" in x
                                                    for x in e), f"{e}")
        os.remove(frozen)
        e = m.final_guard(ns(registration=reg), study)
        check("refused: no freeze record", any("not frozen" in x for x in e), f"{e}")

        # 3. main() on a fake, valid frozen study: guard passes, Run (sentinel) is reached first
        study2, _ = fake_study(m, os.path.join(root, "second"))
        m.STUDY_DIR = study2
        argv = ["a2_run.py", "--split", "devtest", "--final", "--claim-arm", "exchange", "--registration",
                os.path.join(study2, m.REGISTRATION)]
        sys.argv = argv
        try:
            m.main()
            got = "returned"
        except Reached as r:
            got = str(r)
        except SystemExit as e:
            got = f"exit {e.code}"
        check("main() with the registered command on a valid frozen study reaches Run first, before any data",
              got == "Run", got)

        # 6. a stop or a crash after the run folder exists, with a stand-in run logger
        m.Run = FakeRun

        def stop(*a, **k):
            raise SystemExit("simulated stop before the first score")
        m.ptb.load = stop
        sys.argv = argv
        try:
            m.main()
            got = "returned"
        except SystemExit as e:
            got = f"exit {e.code}"
        meta = FakeRun.last.meta if FakeRun.last else {}
        check("a stop is recorded: state stopped, the reason, eval split not scored, the stop propagates",
              got.startswith("exit simulated") and meta.get("state") == "stopped" and meta.get("eval_scored") is False
              and "simulated stop" in meta.get("stop_reason", "") and any("STOPPED before" in x for x in
                                                                         FakeRun.last.lines)
              and any(n == "run.json" and d.get("state") == "stopped" for n, d in FakeRun.last.dumps), f"{got} {meta}")

        def crash(*a, **k):
            raise RuntimeError("simulated crash")
        m.ptb.load = crash
        try:
            m.main()
            got = "returned"
        except RuntimeError as e:
            got = f"raised {e}"
        meta = FakeRun.last.meta
        check("a crash records eval_scored and propagates", got == "raised simulated crash"
              and meta.get("eval_scored") is False and meta.get("state") == "running", f"{got} {meta}")
        sentinels(m)
        m.STUDY_DIR = STUDY

    # 2. main() on the real study with dev-test arguments no state allows: refused before Run and ptb.load
    for args in (["--split", "devtest"], ["--split", "devtest", "--final"], ["--split", "valid", "--final"]):
        sys.argv = ["a2_run.py"] + args
        try:
            m.main()
            got = "returned"
        except Reached as r:
            got = f"reached {r}"
        except SystemExit as e:
            got = f"exit {str(e.code)[:60]}"
        check(f"real study, {' '.join(args)}: refused before any run folder or data", got.startswith("exit refused"),
              got)

    # 4. get_code in the look: no build
    called = []
    load_saved = m.codes.load
    m.codes.load = lambda name, source: (None, None)
    try:
        m.get_code("exchange", {}, lambda: called.append(1), print, True)
        got = "returned"
    except SystemExit as e:
        got = str(e.code)
    m.codes.load = load_saved
    check("the look builds no code when the saved one does not match", "builds no code" in got and not called, got)

    # 5. the look's equality checks, on the two development runs
    expect = m.expected_dev(os.path.join(STUDY, m.EXPECT_METRICS))
    with open(DEV2) as f:
        d2 = json.load(f)
    lams = {k: np.array(v["lam"]).round(5).tolist() for k, v in d2["arms"].items()}
    steps = {k: v["em_steps"] for k, v in d2["arms"].items()}
    check("second development run against the pinned first: cache grid and choice equal",
          m.cache_differences(d2["cache"], expect["cache"]) == [], f"{m.cache_differences(d2['cache'], expect['cache'])}")
    check("second development run against the pinned first: every weight and EM step equal",
          m.weight_differences(lams, steps, expect) == [], f"{m.weight_differences(lams, steps, expect)[:3]}")
    g = json.loads(json.dumps(d2["cache"]))
    g["grid"][5]["valid_bits"] += 1e-15
    check("refused: one cache-grid value perturbed by 1e-15", m.cache_differences(g, expect["cache"]) != [])
    g = json.loads(json.dumps(d2["cache"]))
    g["chosen"]["half_life"] = 50.0
    check("refused: another cache choice", m.cache_differences(g, expect["cache"]) != [])
    l2 = json.loads(json.dumps(lams))
    l2["exchange"][2] = round(l2["exchange"][2] + 1e-5, 5)
    check("refused: one weight perturbed by 1e-5", any("weights of exchange" in x
                                                       for x in m.weight_differences(l2, steps, expect)))
    s2 = dict(steps, none=steps["none"] + 1)
    check("refused: one EM step count changed", any("EM steps of none" in x
                                                    for x in m.weight_differences(lams, s2, expect)))
    l3 = {k: v for k, v in lams.items() if k != "orgroups"}
    check("refused: an arm missing", any("orgroups" in x for x in m.weight_differences(l3, steps, expect)))

    after = set(os.listdir(results)) if os.path.isdir(results) else set()
    check("no run folder was written in results/", after == before, f"new: {sorted(after - before)}")
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
