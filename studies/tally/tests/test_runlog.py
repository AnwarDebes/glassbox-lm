"""Run records of src/tally/runlog.py: data hashes keyed by path, and runs ended by a signal.

Run: PYTHONPATH=src python tests/test_runlog.py

Every run folder of this test is written into a temporary folder (runlog.RESULTS is replaced in
the test process and in each child process); the study's results/ folder is not touched.
Checks:
1. data_hashes.json keeps one entry per file when two data files have the same name in different
   folders (the PTB train and validation texts are both named text.utf8), keyed by the path
   relative to the study folder, each with the sha256 of its file.
2. A run that receives SIGTERM or SIGHUP is recorded as "killed" with the signal's name and an
   end time, and the process still ends by that signal.
3. A signal that was ignored before the run started (SIGHUP under nohup) stays ignored.
4. An exception still marks the run "crashed" (the excepthook is kept).
5. A finished run stays "finished" when the process gets SIGTERM afterwards, and hands the
   excepthook and the signal handlers back (a later exception is not its crash).
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

from tally import runlog

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_json(results, name):
    hits = [d for d in sorted(os.listdir(results)) if d.endswith("_" + name)]
    if len(hits) != 1:
        return None
    with open(os.path.join(results, hits[0], "run.json")) as f:
        return json.load(f)


CHILD = r"""
import os, signal, sys, time
sys.path.insert(0, {src!r})
from tally import runlog
runlog.RESULTS = {results!r}
if {ignore_hup}:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
run = runlog.Run({name!r}, {{"test": True}})
if {mode!r} == "crash":
    raise RuntimeError("planned crash")
if {mode!r} == "finish":
    run.finish({{"ok": 1}})
print("READY", flush=True)
time.sleep(60)
"""


def child(results, name, mode="wait", ignore_hup=False):
    code = CHILD.format(src=SRC, results=results, name=name, mode=mode, ignore_hup=ignore_hup)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    if mode == "crash":
        p.wait(timeout=60)
        return p
    seen = []
    for line in p.stdout:              # the run prints "run <id>" first, then READY
        seen.append(line)
        if line.strip() == "READY":
            return p
    raise AssertionError((seen, p.stderr.read()))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        results = os.path.join(tmp, "results")
        os.makedirs(results)
        runlog.RESULTS = results
        # 1. data hashes: two files named text.utf8 in different folders
        a = os.path.join(tmp, "data", "train", "text.utf8")
        b = os.path.join(tmp, "data", "validation", "text.utf8")
        for p, txt in ((a, "train text\n"), (b, "validation text\n")):
            os.makedirs(os.path.dirname(p))
            with open(p, "w") as f:
                f.write(txt)
        r = runlog.Run("test_runlog_data", {"test": True}, data_files=[a, b])
        r.finish({})
        with open(os.path.join(r.dir, "data_hashes.json")) as f:
            dh = json.load(f)
        want = {os.path.relpath(a, runlog.STUDY): sha(a), os.path.relpath(b, runlog.STUDY): sha(b)}
        check("data_hashes.json holds both files named text.utf8", len(dh) == 2, f"got {dh}")
        check("data_hashes.json keyed by path relative to the study folder, sha256 of each file",
              dh == want, f"got {dh}, want {want}")
        check("after finish the run's excepthook and signal handlers are handed back",
              sys.excepthook is sys.__excepthook__
              and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL, f"{sys.excepthook}")

        # 2. SIGTERM and SIGHUP mark the run killed; the process still ends by the signal
        for sig in (signal.SIGTERM, signal.SIGHUP):
            name = f"test_runlog_{sig.name.lower()}"
            p = child(results, name)
            p.send_signal(sig)
            rc = p.wait(timeout=60)
            rj = run_json(results, name) or {}
            check(f"{sig.name}: run.json state killed", rj.get("state") == "killed", f"state {rj.get('state')}")
            check(f"{sig.name}: run.json names the signal and has an end time",
                  rj.get("signal") == sig.name and "end_utc" in rj and "wall_s" in rj, f"{rj}")
            check(f"{sig.name}: process ended by the signal", rc == -sig, f"exit {rc}")
            time.sleep(1.1)   # run ids have one-second resolution

        # 3. SIGHUP ignored before the run started (nohup) stays ignored
        name = "test_runlog_nohup"
        p = child(results, name, ignore_hup=True)
        p.send_signal(signal.SIGHUP)
        time.sleep(1.0)
        alive = p.poll() is None
        rj = run_json(results, name) or {}
        check("ignored SIGHUP: process keeps running and the run stays running",
              alive and rj.get("state") == "running", f"alive {alive}, state {rj.get('state')}")
        p.send_signal(signal.SIGTERM)
        rc = p.wait(timeout=60)
        rj = run_json(results, name) or {}
        check("ignored SIGHUP: a later SIGTERM still marks the run killed",
              rj.get("state") == "killed" and rc == -signal.SIGTERM, f"state {rj.get('state')}, exit {rc}")

        # 4. exception: crashed
        name = "test_runlog_crash"
        p = child(results, name, mode="crash")
        rj = run_json(results, name) or {}
        check("exception: run.json state crashed", rj.get("state") == "crashed" and p.returncode == 1,
              f"state {rj.get('state')}, exit {p.returncode}")

        # 5. finished run, then SIGTERM: stays finished
        name = "test_runlog_finished"
        p = child(results, name, mode="finish")
        p.send_signal(signal.SIGTERM)
        rc = p.wait(timeout=60)
        rj = run_json(results, name) or {}
        check("finished run stays finished after a later SIGTERM",
              rj.get("state") == "finished" and rc == -signal.SIGTERM, f"state {rj.get('state')}, exit {rc}")
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
