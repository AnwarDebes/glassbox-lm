"""Diagnostic for E0: how many clauses hold members of each hidden group (word % G), from the state.npz of runs.
Usage (from the study folder): .venv/bin/python -I src/diag_allocation.py runs/<run> [runs/<run> ...]"""
import json
import os
import sys

import numpy as np

for d in sys.argv[1:]:
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    st = np.load(os.path.join(d, "state.npz"))["state"]
    G = cfg["hidden_groups"]
    per = np.zeros(G, int)
    blank = 0
    mixed = 0
    for c in range(st.shape[0]):
        words = np.flatnonzero((st[c] >= 0).any(0))
        if len(words) == 0:
            blank += 1
            continue
        hg = np.unique(words % G)
        mixed += len(hg) > 1
        for h in hg:
            per[h] += 1
    print(os.path.basename(d), "clauses per hidden group", per.tolist(), "blank", blank, "mixed", mixed,
          "groups without a clause", int((per == 0).sum()))
