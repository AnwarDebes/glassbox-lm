"""Run Granmo's MultiConceptDemo file unchanged, after seeding the global numpy RNG (Check 1a).

usage: run_demo_seeded.py --seed S --demo PATH [demo args...]

The demo has no seed argument: it draws its data and one machine seed per run from the
global numpy RNG, so np.random.seed(S) fixes the whole run. The demo writes its statistics
file ("run epoch accuracy" per line) to the working directory, so we chdir to the run dir
first. Afterwards we summarize that file into metrics.json.
"""
import argparse, glob, hashlib, json, os, runpy, shutil, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--demo", required=True)
a, rest = ap.parse_known_args()
demo = os.path.abspath(a.demo)
run_dir = os.environ.get("TMLM_RUN_DIR", os.getcwd())
for f in (__file__, demo):
    shutil.copy(f, run_dir)
os.chdir(run_dir)
sha = hashlib.sha256(open(demo, "rb").read()).hexdigest()
print("DEMO", json.dumps({"demo": demo, "sha256": sha, "seed": a.seed, "demo_args": rest}), flush=True)

np.random.seed(a.seed)
sys.argv = [demo] + rest
runpy.run_path(demo, run_name="__main__")

# summary of the demo's own statistics file: final-epoch test accuracy of each run
stats = sorted(glob.glob("multi_concept_statistics_*.txt"))
rows = np.loadtxt(stats[-1], ndmin=2)
runs = sorted(set(rows[:, 0].astype(int)))
last_epoch = rows[:, 1].max()
final = [float(rows[(rows[:, 0] == r) & (rows[:, 1] == last_epoch), 2][0]) / 100 for r in runs]
best = [float(rows[rows[:, 0] == r, 2].max()) / 100 for r in runs]
summary = dict(seed=a.seed, demo_sha256=sha, runs=len(runs), epochs=int(last_epoch) + 1,
               final_acc_per_run=final, mean_final_acc=float(np.mean(final)), min_final_acc=float(np.min(final)),
               mean_best_acc=float(np.mean(best)))
print("SUMMARY", json.dumps(summary), flush=True)
metrics = {k: {"value": v, "label": "measured"} for k, v in summary.items()}
json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)
