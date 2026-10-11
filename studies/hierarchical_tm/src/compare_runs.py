"""Compare two runs that must train identically (same seed, settings and train rows).
  compare_runs.py RUN_A RUN_B                 per-pass train_acc and empty-leaf share for all passes, and final include bits
  compare_runs.py RUN_A RUN_B --dev --upto N  per-pass dev_auc, dev_acc and pred_pos for passes 1..N (same dev set needed),
                                              e.g. the learning-curve run vs the earlier 100-pass run at the same s and seed
Prints COMPARE {json}: identical or not, the first differing pass and the largest difference."""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--dev", action="store_true"); ap.add_argument("--upto", type=int, default=0)
x = ap.parse_args()

def hist(d):
    return {h["pass"]: h for h in (json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(d, "log.txt")) if l.startswith("EPOCH "))}

ha, hb = hist(x.a), hist(x.b)
keys = ("dev_auc", "dev_acc", "pred_pos") if x.dev else ("train_acc", "empty_leaf_share")
passes = sorted(set(ha) & set(hb))
if x.upto:
    passes = [p for p in passes if p <= x.upto]
diffs = [(p, max(abs(ha[p][k] - hb[p][k]) for k in keys)) for p in passes]
first = next((p for p, d in diffs if d > 0), None)
out = {"a": os.path.basename(x.a), "b": os.path.basename(x.b), "fields": keys, "passes_compared": len(passes),
       "identical": first is None, "first_differing_pass": first, "max_abs_difference": max((d for _, d in diffs), default=0.0)}
if not x.dev:
    ia, ib = np.load(os.path.join(x.a, "include_bits.npz")), np.load(os.path.join(x.b, "include_bits.npz"))
    out["final_include_bits_identical"] = all(np.array_equal(ia[k], ib[k]) for k in ("class1", "class0"))
print("COMPARE", json.dumps(out))
