"""Launch logic of scripts/b1_launch.sh and src/tally/b6/launch.py, with stubs (no B1 or B6 job).

Run: PYTHONPATH=src python tests/test_launchers.py

Machine rule checked: every job runs under nice 19 with every thread pool capped at 1 (at most 2
allowed), and no job starts before the CPU gate reads at most 4.5 cores.
- b1_launch.sh: the dry run (B1_DRY_RUN=1) prints the job lines; then the launcher runs a small
  grid (2 sizes, 3 seeds) in a temporary folder with a stub interpreter (records each call, its
  thread variables and its start and end time), a stub slot script and a stub gate whose first
  reading fails (exit 3; since 2026-10-26 it counts as closed), whose second is closed (5.20
  cores) and every later one open (1.00). The real interpreter, slot script and gate are never
  called. Without a freeze record it must stop with exit 2.
- b6/launch.py: the dry run prints the job lines; then main() runs the jobs of one seed and one
  dataset with the job command replaced by a stub process, the gate reading by the same
  closed-then-open sequence, and the results folder by a temporary one.
- The parser of the real gate line, and one real reading of the gate (10 s).
Checks of the stub runs: one gate reading that is open before every start, the closed reading
first, at most JOBS (--parallel) jobs at once, phase and dependency order, thread caps.
"""
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
B1 = os.path.join(STUDY, "scripts", "b1_launch.sh")
FAILS = []
CAPS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS")


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def max_parallel(intervals):
    ev = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals], key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


STUB_PY = r"""#!/usr/bin/env bash
# stub interpreter: record the call, its thread variables, start and end
rec="${STUB_REC:?}"
start=$(date +%s.%N)
if [ "$2" = "tally.b1.run" ]; then sleep 0.3; fi
end=$(date +%s.%N)
echo "${start} ${end} OMP=${OMP_NUM_THREADS:-} MKL=${MKL_NUM_THREADS:-} OPENBLAS=${OPENBLAS_NUM_THREADS:-} NUMBA=${NUMBA_NUM_THREADS:-} RAYON=${RAYON_NUM_THREADS:-} | $*" >> "${rec}"
"""
STUB_SLOT = r"""#!/usr/bin/env bash
# stub cpu_slot.sh N -- cmd...
shift
[ "$1" = "--" ] && shift
exec "$@"
"""
STUB_GATE = r"""#!/usr/bin/env bash
# stub gate: the first reading fails (exit 3), the second is closed, then open
n="${STUB_GATE_N:?}"
c=$(cat "${n}" 2>/dev/null || echo 0)
echo $((c + 1)) > "${n}"
if [ "${c}" -eq 0 ]; then
    echo "stub gate: cannot read the cgroup" >&2
    exit 3
elif [ "${c}" -eq 1 ]; then
    echo "2026-10-25T00:00:00Z cores 5.20 over 10.0 s; throttled 0 of 100 periods"
else
    echo "2026-10-25T00:00:00Z cores 1.00 over 10.0 s; throttled 0 of 100 periods"
fi
"""


def b1_tests():
    src = open(B1).read()
    check("b1_launch.sh: no job line at nice 10", "nice -n 10" not in src)
    hooks = all(h in src for h in ("B1_PY", "B1_SLOT", "CPU_GATE_CMD", "B1_DRY_RUN", "B1_LOGS"))
    check("b1_launch.sh: has the dry run and the stub hooks", hooks)
    if not hooks:
        print("     (stub runs of b1_launch.sh not started: this version would start real jobs)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        regdir = os.path.join(tmp, "registrations")
        os.makedirs(regdir)
        reg = os.path.join(regdir, "B1_test.md")
        open(reg, "w").write("test registration\n")
        stubs = {}
        for name, body in (("py", STUB_PY), ("slot", STUB_SLOT), ("gate", STUB_GATE)):
            stubs[name] = os.path.join(tmp, f"stub_{name}.sh")
            open(stubs[name], "w").write(body)
            os.chmod(stubs[name], 0o755)
        env = dict(os.environ, B1_PY=stubs["py"], B1_SLOT=stubs["slot"], CPU_GATE_CMD=stubs["gate"],
                   CPU_GATE_RETRY="0.2", B1_SETTLE="0", B1_SIZES="16 8", B1_SEEDS="0 1 2",
                   B1_LOGS=os.path.join(tmp, "logs"), STUB_REC=os.path.join(tmp, "rec.txt"),
                   STUB_GATE_N=os.path.join(tmp, "gate_n.txt"))
        # dry run (no freeze record needed)
        r = subprocess.run(["bash", B1, reg, "2"], env=dict(env, B1_DRY_RUN="1"), capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.startswith(("phase1:", "phase2:"))]
        check("b1 dry run: exit 0, 56 job lines (2 sizes x (1 tuning + 7 arms x 3 seeds) + 2 x 2 x 3)",
              r.returncode == 0 and len(lines) == 56, f"exit {r.returncode}, {len(lines)} lines, {r.stderr[-300:]}")
        check("b1 dry run: every job line starts with nice -n 19 and runs cpu_slot 1",
              lines and all(re.match(r"phase[12]: nice -n 19 \S+ 1 -- ", l) for l in lines), lines[:1])
        check("b1 dry run: thread caps of 1 printed", "OMP_NUM_THREADS=1" in r.stdout and "NUMBA_NUM_THREADS=1" in r.stdout)
        check("b1 dry run: starts no job", not os.path.exists(env["STUB_REC"]))
        # no freeze record: stop
        r = subprocess.run(["bash", B1, reg, "2"], env=env, capture_output=True, text=True)
        check("b1 without a freeze record: exit 2, nothing started",
              r.returncode == 2 and "not frozen" in r.stderr and not os.path.exists(env["STUB_REC"]),
              f"exit {r.returncode} {r.stderr[-200:]}")
        # stub run
        open(os.path.join(regdir, "B1_test.frozen.json"), "w").write("{}\n")
        t0 = time.time()
        r = subprocess.run(["bash", B1, reg, "2"], env=env, capture_output=True, text=True, timeout=600)
        check("b1 stub run: exit 0", r.returncode == 0, f"exit {r.returncode} {r.stderr[-300:]}")
        recs = []
        for l in open(env["STUB_REC"]):
            head, cmd = l.rstrip("\n").split(" | ", 1)
            a, b, *caps = head.split()
            recs.append((float(a), float(b), dict(c.split("=", 1) for c in caps), cmd))
        runs = [x for x in recs if "-m tally.b1.run" in x[3]]
        check("b1 stub run: 56 runs, then the self-checks first and the report last",
              len(runs) == 56 and "tally.b1.selfcheck" in recs[0][3] and "tally.b1.report" in recs[-1][3],
              f"{len(runs)} runs, first {recs[0][3][:60]}, last {recs[-1][3][:60]}")
        check("b1 stub run: every run is registered (--label reg --registration <file>)",
              all("--label reg" in x[3] and f"--registration {reg}" in x[3] for x in runs))
        check("b1 stub run: at most 2 runs at once", max_parallel([(a, b) for a, b, _, _ in runs]) <= 2,
              f"max {max_parallel([(a, b) for a, b, _, _ in runs])}")
        check("b1 stub run: every job saw thread caps of 1",
              all(all(c[k] == "1" for k in ("OMP", "MKL", "OPENBLAS", "NUMBA", "RAYON")) for _, _, c, _ in recs))
        p1 = [x for x in runs if x[3].split("--arm ")[1].split()[0] not in ("flat", "ab")]
        p2 = [x for x in runs if x[3].split("--arm ")[1].split()[0] in ("flat", "ab")]
        check("b1 stub run: phase 2 (flat, ab) starts after every phase 1 run ended",
              len(p2) == 12 and min(a for a, _, _, _ in p2) >= max(b for _, b, _, _ in p1))
        gate = open(os.path.join(env["B1_LOGS"], "gate.log")).read().splitlines()
        opens = [g for g in gate if " open: " in g]
        check("b1 stub run: a failed gate reading (exit 3) counts as closed and the launcher went on",
              len(gate) > 1 and " closed (limit 4.5): gate command failed (exit 3)" in gate[0], gate[:1])
        check("b1 stub run: the next reading (5.20 cores) is closed and the launcher waited",
              len(gate) > 1 and " closed (limit 4.5): " in gate[1] and "cores 5.20" in gate[1], gate[1:2])
        check("b1 stub run: one open gate reading before every start (56 runs + self-checks + report)",
              len(opens) == 58, f"{len(opens)} open readings")
        starts = [l for l in open(os.path.join(env["B1_LOGS"], "starts.log")) if " start " in l]
        check("b1 stub run: starts.log lists 56 starts", len(starts) == 56, f"{len(starts)}")
        print(f"     b1 stub run took {time.time() - t0:.1f} s")


def b6_tests():
    from tally.b6 import launch
    buf = io.StringIO()
    sys.argv = ["launch", "--dry-run", "--seeds", "1-1", "--datasets", "fmnist"]
    with redirect_stdout(buf):
        launch.main()
    lines = [l for l in buf.getvalue().splitlines() if "tally.b6.run" in l]
    check("b6 dry run: 14 jobs for one seed of fmnist", len(lines) == 14, f"{len(lines)}")
    check("b6 dry run: every job line starts with nice -n 19 and runs cpu_slot 1",
          lines and all(re.match(r"nice -n 19 \S+ 1 -- ", l) for l in lines), lines[:1])
    hooks = all(hasattr(launch, h) for h in ("gate_reading", "wait_gate", "GATE_MAX", "THREAD_CAPS"))
    check("b6/launch.py: has the CPU gate and the thread caps", hooks)
    with tempfile.TemporaryDirectory() as tmp:
        rec = os.path.join(tmp, "rec.jsonl")
        stub = ("import json, os, sys, time; a = time.time(); time.sleep(0.3); "
                "open(sys.argv[1], 'a').write(json.dumps({'job': sys.argv[2], 'start': a, 'end': time.time(), "
                f"'caps': {{k: os.environ.get(k) for k in {CAPS!r}}}}}) + '\\n')")
        readings = iter([(5.2, "2026-10-25T00:00:00Z cores 5.20 over 10.0 s")])
        launch.RESULTS = tmp
        launch.finished = lambda *a, **k: None
        launch.command = lambda job, label, reg: [sys.executable, "-c", stub, rec, "_".join(str(x) for x in job if x)]
        launch.gate_reading = lambda: next(readings, (1.0, "2026-10-25T00:00:00Z cores 1.00 over 10.0 s"))
        if hasattr(launch, "POLL_S"):
            launch.POLL_S = 0.05
        if hasattr(launch, "GATE_RETRY_S"):
            launch.GATE_RETRY_S = 0.05
        sys.argv = ["launch", "--seeds", "1-1", "--datasets", "fmnist", "--parallel", "2", "--label", "b6test"]
        t0 = time.time()
        launch.main()
        jobs = [json.loads(l) for l in open(rec)]
        names = {j["job"]: j for j in jobs}
        check("b6 stub run: all 14 jobs ran", len(jobs) == 14, f"{len(jobs)}")
        check("b6 stub run: at most 2 jobs at once", max_parallel([(j["start"], j["end"]) for j in jobs]) <= 2,
              f"max {max_parallel([(j['start'], j['end']) for j in jobs])}")
        check("b6 stub run: every job saw thread caps of 1",
              all(all(v == "1" for v in j["caps"].values()) for j in jobs), jobs[0]["caps"])
        ok = len(jobs) == 14 and all(names[f"nocomm_fmnist_1_{m}"]["start"] >= names["ruleT_fmnist_1"]["end"]
                                     for m in ("thresh10", "thermo3", "thermo4", "hog"))
        ok &= len(jobs) == 14 and all(names[f"{arm}_fmnist_1"]["start"] >= max(names[f"member_fmnist_1_{m}"]["end"]
                                      for m in ("thresh10", "thermo3", "thermo4", "hog")) for arm in ("fusion", "gate"))
        check("b6 stub run: nocomm after ruleT, fusion and gate after the blind members", ok)
        log = open(os.path.join(tmp, "b6_launch_b6test.log")).read().splitlines()
        gl = [l for l in log if " gate " in l]
        check("b6 stub run: the first gate reading (5.20 cores) is closed",
              gl and "gate closed (limit 4.5)" in gl[0] and "cores 5.20" in gl[0], gl[:1])
        opens = [i for i, l in enumerate(log) if " gate open: " in l]
        starts = [i for i, l in enumerate(log) if " started " in l]
        check("b6 stub run: an open gate reading right before each of the 14 starts",
              len(starts) == 14 and len(opens) == 14 and all(o == s - 1 for o, s in zip(opens, starts)),
              f"{len(opens)} open, {len(starts)} starts")
        print(f"     b6 stub run took {time.time() - t0:.1f} s")


def gate_parse_test():
    line = "2026-10-25T11:20:25Z cores 2.30 over 10.0 s; throttled 0 of 100 periods"
    m = re.search(r" cores ([0-9.]+) ", line)
    sed = subprocess.run(["sed", "-n", r"s/.* cores \([0-9.][0-9.]*\) .*/\1/p"], input=line, capture_output=True, text=True)
    check("gate line parsed by the b6 regex and the b1 sed expression", m and m.group(1) == "2.30"
          and sed.stdout.strip() == "2.30", f"{m and m.group(1)} {sed.stdout!r}")
    from tally.b6 import launch
    if hasattr(launch, "GATE_SCRIPT") and os.path.exists(launch.GATE_SCRIPT):
        import importlib
        importlib.reload(launch)      # the stub test replaced gate_reading
        cores, line = launch.gate_reading()
        check("one real reading of the CPU gate (10 s) gives a number", cores is not None and cores >= 0, line)
        print(f"     real gate reading: {line}")
    else:
        check("the CPU gate script exists", False)


def main():
    b1_tests()
    b6_tests()
    gate_parse_test()
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
