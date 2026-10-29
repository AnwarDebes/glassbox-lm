"""Self-tests of chain/part3_cpu_job.sh and chain/part3_status.py (plain asserts, no pytest).

    cd studies/language_part3 && \\
      studies/tm_language_model/.venv-tm/bin/python -I -B chain/tests/test_part3_cpu_job.py

Every test file goes to chain/tmp/selftest/ (the wrapper's test mode: PART3_CHAIN_TEST_DIR); the real lock and log of
the chain are never touched. One C self-test (C's selftest_dl, about 10 s, no CPU gate) runs through the real C runner
to check the pass-through; A's runner runs only in its own test mode. Takes about 1 minute.
"""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CH = Path(__file__).resolve().parents[1]
TM = CH.parent
WRAP = CH / "part3_cpu_job.sh"
T = CH / "tmp" / "selftest"
sys.path.insert(0, str(CH))
import part3_status  # noqa: E402

PASSED = []


def check(name, cond, info=""):
    if not cond:
        raise AssertionError(f"FAIL {name} {info}")
    PASSED.append(name)
    print("ok", name, flush=True)


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PART3_", "A_TEST", "A_JOBLOG", "A_RUNNER"))}
    env.update(PART3_CHAIN_TEST_DIR=str(T), **extra)
    return env


def wrap(*args, env=None, timeout=300):
    return subprocess.run(["bash", str(WRAP), *map(str, args)], env=env or clean_env(), capture_output=True, text=True,
                          timeout=timeout, check=False)


def log_lines():
    p = T / "part3_cpu_jobs.log"
    return p.read_text().splitlines() if p.exists() else []


def test_classifiers():
    parent = ("3c3\n< drwxr-sr-x 12 coder coder 12 2026-10-21 02:23:45.000000000 +0000 ..\n---\n"
              "> drwxr-sr-x 13 coder coder 13 2026-10-21 07:00:00.000000000 +0000 ..\n")
    other = parent + "5a6\n> -rw-r--r-- 1 coder coder 9 2026-10-21 07:00:01.000000000 +0000 kn.cpython-311.pyc\n"
    check("diff none", part3_status.diff_class("") == "none")
    check("diff parent-only", part3_status.diff_class(parent) == "parent-only")
    check("diff changed", part3_status.diff_class(other) == "changed")
    check("diff changed alone", part3_status.diff_class("1d0\n< -rw-r--r-- 1 c c 1 2026-10-21 x.py\n") == "changed")
    d = T / "decision.json"
    d.write_text(json.dumps({"decision": "PASS: K0 and K0b passed; the GPU TM step may be scheduled"}))
    check("decision pass", part3_status.c_decision(str(d), time.time() - 60)[0] == 0)
    d.write_text(json.dumps({"decision": "STOP at K0b: LR and LightGBM both more than 0.0072 bpw behind the twin system"}))
    check("decision stop", part3_status.c_decision(str(d), time.time() - 60)[0] == 3)
    check("decision stale", part3_status.c_decision(str(d), time.time() + 60)[0] == 1)
    check("decision missing", part3_status.c_decision(str(T / "absent.json"), 0)[0] == 1)
    d.write_text(json.dumps({"decision": "maybe"}))
    check("decision unknown", part3_status.c_decision(str(d), 0)[0] == 1)


def test_refusals():
    check("usage", wrap("A").returncode == 2)
    check("unknown arm", wrap("X", "y").returncode == 2)
    env = clean_env()
    env.pop("PART3_CHAIN_TEST_DIR")
    r = wrap("T", "true", env=env)
    check("arm T only in test mode", r.returncode == 2 and "self-tests only" in r.stdout)
    r = wrap("T", "true", env=clean_env(PART3_CPU_LOCK=str(T / "part3_cpu.lock")))
    check("deadlock guard", r.returncode == 2 and "must not be the chain lock" in r.stdout)
    r = wrap("T", "true", env=clean_env(PART3_CHAIN_LOCK=str(T / "part3_cpu.lock")))
    check("nested call refused", r.returncode == 2 and "nested call refused" in r.stdout)


def test_serialised():
    """Two jobs started together never overlap; the second waits for the lock (wait, start, end in the log)."""
    f = T / "intervals.txt"
    f.write_text("")
    job = f"echo start $(date +%s.%N) >> {f}; sleep 1.5; echo end $(date +%s.%N) >> {f}"
    n0 = len(log_lines())
    ps = [subprocess.Popen(["bash", str(WRAP), "T", "bash", "-c", job], env=clean_env(), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) for _ in range(2)]
    rcs = [p.wait(timeout=60) for p in ps]
    ev = [ln.split() for ln in f.read_text().splitlines()]
    check("serialised: both done", rcs == [0, 0] and len(ev) == 4, (rcs, ev))
    kinds = [k for k, _ in ev]
    check("serialised: no overlap", kinds == ["start", "end", "start", "end"] and float(ev[1][1]) <= float(ev[2][1]), ev)
    new = log_lines()[n0:]
    check("serialised: log lines", sum(" wait T " in x for x in new) == 2 and sum(" start T " in x for x in new) == 2
          and sum(" end T " in x and "-> 0" in x for x in new) == 2, new)


def test_a_passthrough():
    """A's runner in its own test mode under the wrapper: the job sees PART3_CHAIN_LOCK and A_RUNNER=test; A's exit
    codes pass through unchanged."""
    gl = T / "gate_lines.txt"
    gl.write_text("2026-10-21T03:00:00Z cores 1.00 over 10.0 s; throttled 0 of 100 periods\n")
    env = clean_env(A_TEST_GATE_LINES=str(gl), A_TEST_GATE_SLEEP="0", A_JOBLOG=str(T / "a_jobs.log"),
                    PART3_CPU_LOCK=str(T / "a.lock"))
    r = wrap("A", "tm", "tests/print_runner_env.py", env=env)
    got = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
    check("A pass-through 0", r.returncode == 0 and got and got[0]["A_RUNNER"] == "test"
          and got[0]["PART3_CHAIN_LOCK"] == str(T / "part3_cpu.lock") and got[0]["nice"] == 19, r.stdout[-2000:])
    gl.write_text("2026-10-21T03:00:00Z cores 1.00 over 10.0 s; throttled 0 of 100 periods\n")
    r = wrap("A", "tm", "listing.py", env=env)  # listing.py without arguments: its usage, exit 2
    check("A pass-through 2", r.returncode == 2 and "-> exit 2" in r.stdout, r.stdout[-1500:])


def test_c_mapping_with_a_stand_in_runner():
    """The C-side mapping on a stand-in runner (test mode only): '..'-only listing change -> 0 with a note; another
    change -> 4; lock busy -> 75; the real gates stage's decision file -> 0 or 3; --dry gates -> no decision read."""
    fake = T / "fake_run_stage.sh"
    fake.write_text("#!/bin/bash\n"
                    "stage=$1; shift\n"
                    "case \"$FAKE_MODE\" in\n"
                    "  busy) echo \"another C job holds /x/c_job.lock\" >&2; exit 3 ;;\n"
                    "  listing) echo \"run_stage: $stage $* exit 4; listing before/after: CHANGED ($FAKE_BEFORE)\"; exit 4 ;;\n"
                    "  gates) [ -n \"$FAKE_DECISION\" ] && cp \"$FAKE_DECISION\" \"$PART3_CHAIN_TEST_C_DECISION\"; "
                    "echo \"run_stage: $stage $* exit 0; listing before/after: no change (x)\"; exit 0 ;;\n"
                    "  *) exit 7 ;;\n"
                    "esac\n")
    lst = T / "listings"
    lst.mkdir(exist_ok=True)
    parent = ("3c3\n< drwxr-sr-x 12 coder coder 12 2026-10-21 02:23:45.000000000 +0000 ..\n---\n"
              "> drwxr-sr-x 13 coder coder 13 2026-10-21 07:00:00.000000000 +0000 ..\n")
    (lst / "s1_learners_diff.txt").write_text(parent)
    (lst / "s2_learners_diff.txt").write_text(parent + "> -rw-r--r-- 1 coder coder 9 2026-10-21 x.pyc\n")
    dec = T / "c_decision.json"
    if dec.exists():
        dec.unlink()
    base = dict(PART3_CHAIN_TEST_C_RUNNER=str(fake), PART3_CHAIN_TEST_C_DECISION=str(dec))
    r = wrap("C", "learners", env=clean_env(**base, FAKE_MODE="listing", FAKE_BEFORE=str(lst / "s1_learners_before.txt")))
    check("C '..' only -> 0", r.returncode == 0 and any(" note C learners" in x for x in log_lines()), r.stdout[-800:])
    r = wrap("C", "learners", env=clean_env(**base, FAKE_MODE="listing", FAKE_BEFORE=str(lst / "s2_learners_before.txt")))
    check("C real listing change -> 4", r.returncode == 4 and "(changed)" in r.stdout, r.stdout[-800:])
    r = wrap("C", "twin", env=clean_env(**base, FAKE_MODE="busy"))
    check("C lock busy -> 75", r.returncode == 75, r.stdout[-800:])
    r = wrap("C", "twin", env=clean_env(**base, FAKE_MODE="other"))
    check("C other exit passes through", r.returncode == 7, r.stdout[-800:])
    src = T / "decision_src.json"
    src.write_text(json.dumps({"decision": "STOP: A/A difference outside +-0.0036 (margins inside the noise)"}))
    r = wrap("C", "gates", env=clean_env(**base, FAKE_MODE="gates", FAKE_DECISION=str(src)))
    check("C gates STOP -> 3", r.returncode == 3 and "C gates STOP" in r.stdout, r.stdout[-800:])
    src.write_text(json.dumps({"decision": "PASS: K0 and K0b passed; the GPU TM step may be scheduled"}))
    r = wrap("C", "gates", env=clean_env(**base, FAKE_MODE="gates", FAKE_DECISION=str(src)))
    check("C gates PASS -> 0", r.returncode == 0 and "C gates: PASS" in r.stdout, r.stdout[-800:])
    os.utime(dec, (time.time() - 3600, time.time() - 3600))  # a decision file older than the job is not read
    r = wrap("C", "gates", env=clean_env(**base, FAKE_MODE="gates", FAKE_DECISION=""))
    check("C gates stale decision -> 1", r.returncode == 1 and "no fresh decision" in r.stdout, r.stdout[-800:])
    r = wrap("C", "gates", "--dry", env=clean_env(**base, FAKE_MODE="gates", FAKE_DECISION=""))
    check("C gates --dry -> 0 (no decision read)", r.returncode == 0, r.stdout[-800:])


def test_c_real_runner_selftest():
    """The real C runner through the wrapper: C's selftest_dl (no CPU gate) passes through with exit 0; with C's own
    lock held by someone else, the wrapper reports 75 (not run)."""
    r = wrap("C", "selftest_dl", timeout=200)
    check("C selftest_dl via wrapper -> 0", r.returncode == 0, r.stdout[-1500:])
    lockf = TM / "C" / "tmp" / "c_job.lock"
    fd = os.open(str(lockf), os.O_RDWR | os.O_CREAT)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("skip: C's lock is held by a running C job")
            return
        r = wrap("C", "selftest_dl", timeout=200)
        check("C runner lock busy -> 75", r.returncode == 75 and "not run" in r.stdout, r.stdout[-1500:])
    finally:
        os.close(fd)


def main():
    if T.exists():
        shutil.rmtree(T)
    T.mkdir(parents=True)
    t = time.time()
    for fn in (test_classifiers, test_refusals, test_serialised, test_a_passthrough, test_c_mapping_with_a_stand_in_runner,
               test_c_real_runner_selftest):
        fn()
    print(f"CHAIN WRAPPER TESTS: {len(PASSED)} passed in {time.time() - t:.1f} s")


if __name__ == "__main__":
    main()
