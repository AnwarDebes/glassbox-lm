"""Run the B6 grid: every (arm, dataset, seed) job through cpu_slot.sh 1, dependencies respected.

    PYTHONPATH=src python -m tally.b6.launch [--label b6] [--seeds 1-10] [--datasets fmnist,agnews]
        [--parallel 5] [--registration registrations/<file>.md] [--dry-run]

Jobs per (dataset, seed): one per blind member, single, ruleT, coupled, seqboost; after ruleT,
one nocomm job per member; after all blind members, fusion and gate. Each job is
`nice -n 19 cpu_slot.sh 1 -- <python> -m tally.b6.run ...` with every thread pool capped at 1.
At most --parallel jobs run at once. Machine rule: before every start the launcher waits until
the CPU gate, scripts/cpu_load.py of the decision_layer study over 10 s, reports at most
GATE_MAX = 4.5 cores in use for the whole cgroup, reading it again every GATE_RETRY_S = 60 s
while it does not; jobs are started one after the other, so two never start on one reading.
A job whose finished run folder already exists is skipped, so the launcher can be restarted.
A failed job blocks only its dependents. The long jobs (ruleT, seqboost) are started first.
Log: results/b6_launch_<label>.log (the gate readings too).
Changed 2026-10-25: nice 19 instead of 10, the thread caps and the CPU gate.
"""
import argparse
import os
import re
import subprocess
import time

from tally.b6 import config
from tally.b6.report import finished
from tally.runlog import RESULTS, STUDY

TMLM = os.path.join(os.path.dirname(STUDY), "tm_language_model")
CPU_SLOT = os.path.join(TMLM, "scripts", "cpu_slot.sh")
PYTHON = os.path.join(TMLM, ".venv-tm", "bin", "python")     # the env with tmu at 5605ff0
GATE_SCRIPT = os.path.join(os.path.dirname(STUDY), "decision_layer", "scripts", "cpu_load.py")
GATE_MAX = 4.5
GATE_RETRY_S = 60
POLL_S = 5
THREAD_CAPS = {v: "1" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS")}
PRIORITY = {"ruleT": 0, "seqboost": 1, "coupled": 2, "single": 3, "member": 4, "nocomm": 5, "fusion": 6, "gate": 7}


def jobs(label, datasets, seeds):
    out = []
    for seed in seeds:
        for ds in datasets:
            names = [m["name"] for m in config.MEMBERS[ds]]
            blind = [("member", ds, seed, n) for n in names]
            for j in blind:
                out.append((j, []))
            for arm in ("single", "ruleT", "coupled", "seqboost"):
                out.append(((arm, ds, seed, None), []))
            for n in names:
                out.append((("nocomm", ds, seed, n), [("ruleT", ds, seed, None)]))
            out.append((("fusion", ds, seed, None), blind))
            out.append((("gate", ds, seed, None), blind))
    return out


def command(job, label, registration):
    arm, ds, seed, member = job
    cmd = ["nice", "-n", "19", CPU_SLOT, "1", "--", PYTHON, "-m", "tally.b6.run",
           "--arm", arm, "--dataset", ds, "--seed", str(seed), "--label", label]
    if member:
        cmd += ["--member", member]
    if registration:
        cmd += ["--registration", registration]
    return cmd


def gate_reading():
    """One reading of the CPU gate: (cores in use or None, the printed line)."""
    line = subprocess.run([PYTHON, "-I", GATE_SCRIPT, "10"], capture_output=True, text=True, check=True,
                          stdin=subprocess.DEVNULL).stdout.strip()
    m = re.search(r" cores ([0-9.]+) ", line)
    return (float(m.group(1)) if m else None), line


def wait_gate(log, read, max_cores, retry_s):
    """Return when the gate reads at most max_cores; log every reading."""
    while True:
        cores, line = read()
        if cores is not None and cores <= max_cores:
            log(f"gate open: {line}")
            return cores
        log(f"gate closed (limit {max_cores}): {line}")
        time.sleep(retry_s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="b6")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--datasets", default="fmnist,agnews")
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--registration", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    todo = jobs(a.label, a.datasets.split(","), range(lo, hi + 1))
    todo.sort(key=lambda jd: (PRIORITY[jd[0][0]], jd[0][2]))
    if a.dry_run:
        print("env:", " ".join(f"{k}={v}" for k, v in THREAD_CAPS.items()), f"PYTHONPATH={os.path.join(STUDY, 'src')}")
        print(f"gate before every start: {PYTHON} -I {GATE_SCRIPT} 10, at most {GATE_MAX} cores")
        for j, deps in todo:
            print(" ".join(command(j, a.label, a.registration)), "| after", deps or "-")
        print(f"{len(todo)} jobs")
        return
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, f"b6_launch_{a.label}.log"), "a", buffering=1)

    def logw(msg):
        log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")

    env = dict(os.environ, PYTHONPATH=os.path.join(STUDY, "src"), **THREAD_CAPS)
    done = {j for j, _ in todo if finished(a.label, *j)}
    failed, running = set(), {}
    pending = [(j, d) for j, d in todo if j not in done]
    logw(f"start {len(pending)} jobs, {len(done)} already finished")
    while pending or running:
        for j, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[j]
                (done if rc == 0 else failed).add(j)
                logw(f"{'done' if rc == 0 else f'FAILED {rc}'} {j}")
        for jd in list(pending):
            j, deps = jd
            if any(d in failed for d in deps):
                pending.remove(jd)
                failed.add(j)
                logw(f"skipped (dependency failed) {j}")
            elif len(running) < a.parallel and all(d in done for d in deps):
                wait_gate(logw, gate_reading, GATE_MAX, GATE_RETRY_S)
                pending.remove(jd)
                out = open(os.path.join(RESULTS, f"b6_launch_{a.label}_{'_'.join(str(x) for x in j if x)}.out"), "a")
                running[j] = subprocess.Popen(command(j, a.label, a.registration), env=env, cwd=STUDY,
                                              stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                logw(f"started {j}")
        time.sleep(POLL_S)
    logw(f"end: {len(done)} done, {len(failed)} failed")


if __name__ == "__main__":
    main()
