"""Self-test of spin_check.py on the CPU, with this test's OWN children only (killed at the end).

Cases (window 3 s after 1 s of warm-up; thresholds fixed here before the first run):
  busy      python child spinning in its main thread         main thread 0.80-1.05 cores; process total within 0.05
  sleeping  python child in time.sleep                       main thread and all threads <= 0.05 cores
  split     main thread blocked in join(), one worker spins  main <= 0.05; all threads 0.80-1.05; busiest thread not main
  subtree   runner bash -> subshell bash -> python (sleep)   --descendant-of finds that python (a grandchild), <= 0.05
  none      --descendant-of a runner with no python inside   exit 2, nothing measured
  several   --descendant-of a runner with two python inside  exit 2, nothing measured
The busy cases need one free core: run at cpu_load <= 4.5 (the gate reading goes into the JSON via --gate).

    CUDA_VISIBLE_DEVICES= ... nice -n 19 <decision_layer>/.venv-dl/bin/python -I -B test_spin_check.py \
        --out OUT.json [--gate "..."]
Exit 0 only if every case passes.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../.."))

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = f"{_REPO_ROOT}/studies/decision_layer/.venv-dl/bin/python"
SPIN = HERE / "spin_check.py"
BUSY = "while True:\n    pass\n"
SLEEP = "import time\ntime.sleep(60)\n"
SPLIT = ("import threading\ndef spin():\n    while True:\n        pass\n"
         "t = threading.Thread(target=spin, daemon=True)\nt.start()\nt.join()\n")


def start(argv: list[str]) -> subprocess.Popen:
    # own session, so the whole subtree of THIS test's child can be killed by its group id at the end
    return subprocess.Popen(["nice", "-n", "19", *argv], start_new_session=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def spin_check(args: list[str]) -> tuple[int, dict]:
    p = subprocess.run([PY, "-I", "-B", str(SPIN), *args], capture_output=True, text=True, timeout=60, check=False)
    try:
        return p.returncode, json.loads(p.stdout)
    except json.JSONDecodeError:
        return p.returncode, {"raw": p.stdout[-500:], "stderr": p.stderr[-500:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gate", default="")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refused: {a.out} exists")
    utc0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    procs: list[subprocess.Popen] = []
    cases = []
    try:
        def single(name: str, code: str, check) -> None:
            p = start([PY, "-I", "-B", "-c", code])
            procs.append(p)
            time.sleep(1.0)
            rc, r = spin_check([str(p.pid), "--seconds", "3"])
            ok = rc == 0 and check(r)
            cases.append({"case": name, "pass": bool(ok), "exit": rc, "main_thread_cores": r.get("main_thread_cores"),
                          "all_threads_cores": r.get("all_threads_cores"), "process_cores": r.get("process_cores"),
                          "n_threads": r.get("n_threads_end"),
                          "busiest_tid_is_main": bool(r.get("threads")) and r["threads"][0]["tid"] == p.pid})
            print(("PASS" if ok else "FAIL"), cases[-1], flush=True)
            p.kill()
            p.wait()

        single("busy", BUSY, lambda r: 0.80 <= r["main_thread_cores"] <= 1.05
               and abs(r["process_cores"] - r["all_threads_cores"]) <= 0.05)
        single("sleeping", SLEEP, lambda r: r["main_thread_cores"] <= 0.05 and r["all_threads_cores"] <= 0.05)
        single("split", SPLIT, lambda r: r["main_thread_cores"] <= 0.05 and 0.80 <= r["all_threads_cores"] <= 1.05
               and r["threads"][0]["tid"] != r["pid"])
        # subtree: runner -> subshell -> python; the trailing ': ' commands keep bash from exec-ing its last command
        runner = start(["bash", "-c", f"( '{PY}' -I -B -c 'import time; time.sleep(60)'; : ); :"])
        procs.append(runner)
        time.sleep(1.0)
        rc, r = spin_check(["--descendant-of", str(runner.pid), "--seconds", "2"])
        tree = (r.get("found_via") or {}).get("subtree", [])
        depth_ok = len(tree) == 2 and tree[0][1] == runner.pid and tree[1][1] == tree[0][0] and tree[1][2].startswith("python")
        ok = rc == 0 and depth_ok and r.get("pid") == (tree[1][0] if len(tree) == 2 else None) and r["main_thread_cores"] <= 0.05
        cases.append({"case": "subtree", "pass": bool(ok), "exit": rc, "subtree": tree, "pid": r.get("pid"),
                      "main_thread_cores": r.get("main_thread_cores")})
        print(("PASS" if ok else "FAIL"), cases[-1], flush=True)
        os.killpg(runner.pid, signal.SIGKILL)
        runner.wait()
        # none: a runner whose subtree holds no python
        runner = start(["bash", "-c", "( sleep 60; : ); :"])
        procs.append(runner)
        time.sleep(0.5)
        rc, r = spin_check(["--descendant-of", str(runner.pid), "--seconds", "1"])
        cases.append({"case": "none (MUST refuse)", "pass": rc == 2 and "error" in r, "exit": rc, "error": r.get("error")})
        print(("PASS" if cases[-1]["pass"] else "FAIL"), cases[-1], flush=True)
        os.killpg(runner.pid, signal.SIGKILL)
        runner.wait()
        # several: two python processes under one runner
        sleeper = f"'{PY}' -I -B -c 'import time; time.sleep(60)'"
        runner = start(["bash", "-c", f"{sleeper} & {sleeper} & wait"])
        procs.append(runner)
        time.sleep(1.0)
        rc, r = spin_check(["--descendant-of", str(runner.pid), "--seconds", "1"])
        cases.append({"case": "several (MUST refuse)", "pass": rc == 2 and "error" in r, "exit": rc, "error": r.get("error")})
        print(("PASS" if cases[-1]["pass"] else "FAIL"), cases[-1], flush=True)
        os.killpg(runner.pid, signal.SIGKILL)
        runner.wait()
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
    res = {"utc_start": utc0, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gate": a.gate,
           "spin_check_sha256": hashlib.sha256(SPIN.read_bytes()).hexdigest(),
           "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": cases,
           "all_pass": len(cases) == 6 and all(c["pass"] for c in cases),
           "children_left": [p.pid for p in procs if p.poll() is None]}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'ALL PASS' if res['all_pass'] else 'FAIL'}: {sum(c['pass'] for c in cases)} of {len(cases)} -> {a.out}", flush=True)
    raise SystemExit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
