"""End-to-end smoke test of src/run_pcp.py on tiny data: worlds of seed 990 generated on the spot (never the
study's files, never a seed used for a result), a few hundred pairs, 16 clauses, 2 passes.

Run from the study folder:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv/bin/python -I tests/test_smoke_pcp.py
Writes its run folders under logs/tests/smoke_runs/ and does not touch logs/RUNS.md.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_pcp

OUT = os.path.join(ROOT, "logs", "tests", "smoke_runs")
LOG = os.path.join(ROOT, "logs", "RUNS.md")


def log_state():
    return (os.path.getsize(LOG), os.path.getmtime(LOG)) if os.path.exists(LOG) else None


def one(rung, mode):
    before = log_state()
    run_dir, final = run_pcp.main(["--rung", rung, "--seed", "990", "--fresh", "--mode", mode, "--clauses", "16",
                                   "--passes", "2", "--n-train", "600", "--n-dev", "400", "--name", f"smoke_{rung}_{mode}",
                                   "--out-root", OUT, "--no-log"])
    assert log_state() == before, "--no-log must leave logs/RUNS.md alone"
    for f in ("config.json", "metrics.json", "groups.txt", "state.npz"):
        assert os.path.getsize(os.path.join(run_dir, f)) > 0, f
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    assert len(m["history"]) == 2 and m["final"]["n_train"] == 600 and m["final"]["n_dev"] == 400
    for k in ("dev_plain_acc", "dev_tuned_acc", "dev_auc", "recovery", "train_plain_acc"):
        assert 0.0 <= m["final"][k] <= 1.0, (k, m["final"][k])
    assert ("unseen_plain_acc" in m["final"]) == (rung == "L3")
    st = np.load(os.path.join(run_dir, "state.npz"))["state"]
    assert st.shape == (16, 2, {"L2": 64, "L3": 256}[rung])
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["fresh"] is True and cfg["seed"] == 990 and len(cfg["orgroup_tm_sha256"]) == 64
    assert final == m["final"]
    print("ok  ", rung, mode, os.path.basename(run_dir), flush=True)


if __name__ == "__main__":
    one("L2", "near")
    one("L3", "side")
    print("SMOKE TEST PASSED (2 runs)")
