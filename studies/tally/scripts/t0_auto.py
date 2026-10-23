"""Unattended T0 step: wait for the tuning runs of one machine, apply the registered selection
rule, write and freeze a dated amendment, then start the 10 final seeds.

    python scripts/t0_auto.py coalesced|flat

Rule (registrations/T0.md): among the seed-1 tuning runs (s = 5, 10, 20), the lowest
best_learned_valid_bpw; its best_epoch is the epoch count. A tuning run that crashed is listed
and not used. Ties go to the lower s.
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(HERE, "..", "tm_language_model", "scripts")
PY = os.path.join(HERE, "..", "tm_language_model", ".venv-tm", "bin", "python")
TORCH_PY = os.path.join(HERE, "..", "clause_lm", ".venv-cdllm-gpu", "bin", "python")
REG = os.path.join(HERE, "registrations", "T0.md")
machine = sys.argv[1]
tag, clauses = ("coal", 2000) if machine == "coalesced" else ("flat", 100)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


while True:
    runs = {}
    for s in (5, 10, 20):
        f = sorted(glob.glob(os.path.join(HERE, "results", f"*_t0_tune_{tag}_c{clauses}_s{s}_seed1")))
        if f:
            st = json.load(open(os.path.join(f[-1], "run.json")))["state"]
            if st != "running":
                runs[s] = (f[-1], st)
    if len(runs) == 3:
        break
    time.sleep(300)

cands, lines = [], []
for s, (f, st) in sorted(runs.items()):
    if st == "finished":
        m = json.load(open(os.path.join(f, "metrics.json")))
        cands.append((m["best_learned_valid_bpw"], s, m["best_epoch"], os.path.basename(f)))
        lines.append(f"- s = {s}: run `{os.path.basename(f)}`, best learned valid {m['best_learned_valid_bpw']:.6f} bits per word at epoch {m['best_epoch']}")
    else:
        lines.append(f"- s = {s}: run `{os.path.basename(f)}` {st}, not used")
if not cands:
    sys.exit("no finished tuning run")
bpw, s_best, e_best, rid = min(cands)
amend = os.path.join(HERE, "registrations", f"T0_amendment_{machine}_settings.md")
with open(amend, "w") as fh:
    fh.write(f"# T0 amendment, {utc()[:10]}: settings of the {machine} machine\n\n"
             f"Written at {utc()} by `scripts/t0_auto.py` from the tuning runs, by the rule of `T0.md` "
             f"(lowest learned-arm bits per word on valid, seed 1; ties to the lower s), before any final run of this machine.\n\n"
             + "\n".join(lines) +
             f"\n\nChosen: s = {s_best}, epochs = {e_best} (run `{rid}`).\n\n"
             "Code change after the freeze of T0.md, log text only: run.py printed 418 dropped positions per epoch "
             "also for the flat machine, whose trainer drops none; it now prints 0 for flat.\n")
subprocess.run([PY, os.path.join(HERE, "scripts", "freeze.py"), amend], check=True)

os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
procs = []
for seed in range(1, 11):
    out = os.path.join(HERE, "logs", f"t0_final_{tag}_seed{seed}.out")
    cmd = (f"nice -n 10 {SCRIPTS}/cpu_slot.sh 1 -- {PY} -m tally.t0.run --mode final --machine {machine} "
           f"--clauses {clauses} --seed {seed} --s {s_best} --epochs {e_best} --registration {REG} > {out} 2>&1")
    if machine == "coalesced":
        cmd += (f"; rid=$(sed -n 's/^run //p' {out} | head -n 1); {SCRIPTS}/gpu_run.sh exclusive -- "
                f"env PYTHONPATH=. {TORCH_PY} -m tally.t0.logreg {HERE}/results/$rid "
                f"> {HERE}/logs/t0_final_logreg_seed{seed}.out 2>&1")
    procs.append(subprocess.Popen(["bash", "-c", cmd], cwd=os.path.join(HERE, "src")))
    time.sleep(2)
for p in procs:
    p.wait()
print(utc(), "all final runs of", machine, "ended")
