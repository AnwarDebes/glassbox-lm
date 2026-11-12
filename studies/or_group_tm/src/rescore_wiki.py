"""Readings of a finished Wikipedia run recomputed from its saved state (no training): AUC on the dev pairs of the
unit vote sum and of the ratio reading (clauses matched by both paragraphs divided by clauses matched by either).
Usage (from the study folder): .venv/bin/python -I src/rescore_wiki.py runs/<run> [runs/<run> ...]
Refuses runs that were scored on the test pairs. Writes rescore_dev.json into each run folder."""
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from metrics import auc, tuned_acc
from orgroup_tm import OrGroupTM
from run_wiki import load_split


def main():
    for d in sys.argv[1:]:
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["split"] == "dev", "dev runs only"
        z = np.load(os.path.join(d, "state.npz"))
        st = z["state"]
        tm = OrGroupTM(st.shape[0], st.shape[2], sides=2, K=cfg["K"], tied=cfg.get("tied", False), wmax=cfg.get("wmax", 0))
        tm.state[:] = st
        tm.nmem[:] = (st >= 0).sum(-1)
        tm.pol[:] = z["pol"]
        ids, bounds, y, _ = load_split(cfg["data"], "dev", cfg["F"], cfg.get("n_dev", 0))
        sc = tm.scores(ids, bounds)
        stored = np.load(os.path.join(d, "scores_dev.npz"))["votes"]
        assert np.array_equal(sc["unit"], stored.astype(float)), "the rebuilt model does not reproduce the stored votes"
        out = {"run": os.path.basename(os.path.normpath(d)), "auc_unit": auc(sc["unit"], y), "auc_ratio": auc(sc["ratio"], y),
               "tuned_acc_unit": tuned_acc(sc["unit"], y), "tuned_acc_ratio": tuned_acc(sc["ratio"], y)}
        with open(os.path.join(d, "rescore_dev.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
