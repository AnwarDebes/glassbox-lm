"""Run the B6 grid: every (arm, dataset, seed) job through cpu_slot.sh 1, dependencies respected.

    PYTHONPATH=src python -m tally.b6.launch [--label b6] [--seeds 1-10] [--datasets fmnist,agnews]
        [--parallel 5] [--registration registrations/<file>.md] [--dry-run]

Jobs per (dataset, seed): one per blind member, single, ruleT, coupled, seqboost; after ruleT,
one nocomm job per member; after all blind members, fusion and gate. Each job is
`nice -n 10 cpu_slot.sh 1 -- <python> -m tally.b6.run ...`. At most --parallel jobs are
started at once (they then wait for a slot). A job whose finished run folder already exists
is skipped, so the launcher can be restarted. A failed job blocks only its dependents.
The long jobs (ruleT, seqboost) are started first. Log: results/b6_launch_<label>.log.
"""
import argparse
import os
import subprocess
import time

from tally.b6 import config
from tally.b6.report import finished
from tally.runlog import RESULTS, STUDY

TMLM = os.path.join(os.path.dirname(STUDY), "tm_language_model")
CPU_SLOT = os.path.join(TMLM, "scripts", "cpu_slot.sh")
PYTHON = os.path.join(TMLM, ".venv-tm", "bin", "python")     # the env with tmu at 5605ff0
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
    cmd = ["nice", "-n", "10", CPU_SLOT, "1", "--", PYTHON, "-m", "tally.b6.run",
           "--arm", arm, "--dataset", ds, "--seed", str(seed), "--label", label]
    if member:
        cmd += ["--member", member]
    if registration:
        cmd += ["--registration", registration]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="b6")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--datasets", default="fmnist,agnews")
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--registration", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    todo = jobs(a.label, a.datasets.split(","), range(lo, hi + 1))
    todo.sort(key=lambda jd: (PRIORITY[jd[0][0]], jd[0][2]))
    if a.dry_run:
        for j, deps in todo:
            print(" ".join(command(j, a.label, a.registration)), "| after", deps or "-")
        print(f"{len(todo)} jobs")
        return
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, f"b6_launch_{a.label}.log"), "a", buffering=1)
    env = dict(os.environ, PYTHONPATH=os.path.join(STUDY, "src"))
    done = {j for j, _ in todo if finished(a.label, *j)}
    failed, running = set(), {}
    pending = [(j, d) for j, d in todo if j not in done]
    log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} start {len(pending)} jobs, "
              f"{len(done)} already finished\n")
    while pending or running:
        for j, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[j]
                (done if rc == 0 else failed).add(j)
                log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {'done' if rc == 0 else f'FAILED {rc}'} {j}\n")
        for jd in list(pending):
            j, deps = jd
            if any(d in failed for d in deps):
                pending.remove(jd)
                failed.add(j)
                log.write(f"skipped (dependency failed) {j}\n")
            elif len(running) < a.parallel and all(d in done for d in deps):
                pending.remove(jd)
                out = open(os.path.join(RESULTS, f"b6_launch_{a.label}_{'_'.join(str(x) for x in j if x)}.out"), "a")
                running[j] = subprocess.Popen(command(j, a.label, a.registration), env=env, cwd=STUDY,
                                              stdout=out, stderr=subprocess.STDOUT)
                log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} started {j}\n")
        time.sleep(5)
    log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} end: {len(done)} done, {len(failed)} failed\n")


if __name__ == "__main__":
    main()
