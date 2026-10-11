"""Compare every array of a rebuilt data folder with the study's data/ folder (exact equality)."""
import json, os, sys, numpy as np
new, ref = sys.argv[1], os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
bad = 0
for f in ["paragraphs_F3072.npz"] + [f"{v}/{s}.npz" for v in ("sa", "np", "ns") for s in ("train", "dev", "test")]:
    a, b = np.load(f"{new}/{f}"), np.load(f"{ref}/{f}")
    same = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    bad += not same
    print(f, "identical" if same else "DIFFERENT")
for f in ("vocab_F1024.json", "vocab_F3072.json", "splits.json"):
    same = json.load(open(f"{new}/{f}")) == json.load(open(f"{ref}/{f}"))
    bad += not same
    print(f, "identical" if same else "DIFFERENT")
print("ALL IDENTICAL" if bad == 0 else f"{bad} FILES DIFFER")
sys.exit(1 if bad else 0)
