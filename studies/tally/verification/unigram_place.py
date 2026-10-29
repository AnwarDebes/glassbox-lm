"""Placement of the T0 arms against an add-one unigram (no rule; dev-test not scored).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 ../tm_language_model/.venv-tm/bin/python -I verification/unigram_place.py
    (from the study folder)

Prints the bits per word of an add-one unigram counted on train, on calib and on valid, and the mean over seeds 1 to 10
of each arm's valid and calib bits per word, read from metrics.json of the 30 final runs of T0.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from tally import ptb  # noqa: E402

vocab, w2i, ids, blocks = ptb.load()
c = np.bincount(ids["train"], minlength=len(vocab)).astype(np.float64) + 1.0
p = c / c.sum()
out = {"unigram": {s: float(-np.log2(p[ids[s]]).mean()) for s in ("calib", "valid")}}
fams = {"coalesced": "*_t0_final_coal_c2000_s20_seed*", "flat": "*_t0_final_flat_c100_s5_seed*",
        "logreg": "*_t0_logreg_of_final_coal_c2000_s20_seed*"}
for fam, pat in fams.items():
    ms = [json.load(open(os.path.join(d, "metrics.json"))) for d in sorted(glob.glob(os.path.join("results", pat)))]
    assert len(ms) == 10, (fam, len(ms))
    for sp in ("calib", "valid"):
        keys = sorted(k for k in ms[0] if k.endswith(f"_{sp}_bpw") and not k.startswith("untrained"))
        out[f"{fam}_{sp}"] = {k: float(np.mean([m[k] for m in ms])) for k in keys}
        if fam == "coalesced":
            out[f"{fam}_{sp}"]["untrained_mean"] = float(np.mean([np.mean([m[k] for k in m if k.startswith("untrained") and k.endswith(f"_{sp}_bpw")]) for m in ms]))
print(json.dumps(out, indent=1))
