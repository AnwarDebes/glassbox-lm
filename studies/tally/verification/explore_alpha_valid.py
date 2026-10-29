"""T0, post hoc and exploratory (valid only, no dev-test, no rule): is the counted read-out limited by its alpha grid?

    OMP_NUM_THREADS=2 nice -n 19 ../tm_language_model/.venv-tm/bin/python -I verification/explore_alpha_valid.py \
        --seeds 1 2 3 --out <json>        # from the study folder

All 120 counted fits of the T0 final runs chose alpha = 0.01, the smallest value of the registered grid
{0.01, 0.1, 0.5, 1, 2, 5}. This script re-fits the counted read-out of the trained coalesced banks with alpha down to
1e-5, using the clause outputs each final run saved (fired_train.npz for the counts N, fired_calib.npz for alpha and
tau by minimum log-loss, fired_valid.npz for the score), and scores the tuning split valid with the full distribution
over V, as tally.t0.readout does. It also recomputes the run's own counted and learned valid bits at the registered
settings as a check (the counted arm at alpha 0.01 must give the run's own counted valid bits). Dev-test is never opened.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.join(os.getcwd(), "src"))  # run from the tally study folder (python -I ignores PYTHONPATH)
from tally.t0 import data, readout  # noqa: E402

ALPHAS = (1e-5, 1e-4, 1e-3, 3e-3, 0.01, 0.1)
REGISTERED = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)


def load_fired(folder, split):
    z = np.load(os.path.join(folder, f"fired_{split}.npz"))
    C = int(z["C"])
    return z["bits"], C, z["y"].astype(np.int64), z["words"].astype(np.int64)


def fired_rows(bits, C, lo, hi):
    return np.unpackbits(bits[lo:hi], axis=1)[:, :C]


def counts(bits, C, y, K, chunk=50000):
    N = np.zeros((C, K), dtype=np.float64)
    for lo in range(0, len(y), chunk):
        f = sparse.csr_matrix(fired_rows(bits, C, lo, lo + chunk).astype(np.float64))
        Y = sparse.csr_matrix((np.ones(len(y[lo:lo + chunk])), (np.arange(len(y[lo:lo + chunk])), y[lo:lo + chunk])),
                              shape=(len(y[lo:lo + chunk]), K))
        N += (f.T @ Y).toarray()
    return N


def scores(bits, C, M):
    return readout.scores(lambda lo, hi: fired_rows(bits, C, lo, hi), M, len(bits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = data.build()
    K = data.K_TOP + 1
    res = {"note": "post hoc, exploratory, valid only; no rule; dev-test not opened", "alphas": list(ALPHAS), "seeds": {}}
    for seed in a.seeds:
        folder = [os.path.join("results", f) for f in sorted(os.listdir("results"))
                  if f.endswith(f"_t0_final_coal_c2000_s20_seed{seed}")]
        assert len(folder) == 1, folder
        folder = folder[0]
        met = json.load(open(os.path.join(folder, "metrics.json")))
        btr, C, ytr, _ = load_fired(folder, "train")
        bca, _, yca, _ = load_fired(folder, "calib")
        bva, _, _, wva = load_fired(folder, "valid")
        N = counts(btr, C, ytr, K)
        row = {"run": os.path.basename(folder), "run_counted_valid_bpw": met["counted_valid_bpw"],
               "run_learned_valid_bpw": met["learned_valid_bpw"], "run_alpha": met["fit"]["counted"]["alpha"], "by_alpha": {}}
        for alpha in ALPHAS:
            M = readout.counted_matrix(N, alpha)
            tau, nll = readout.fit_tau(scores(bca, C, M), yca)
            bits = readout.word_bits(scores(bva, C, M), tau, wva, d["cls"], d["u"], data.K_TOP)
            row["by_alpha"][f"{alpha:g}"] = {"tau": tau, "calib_nll": nll, "valid_bpw": float(bits.mean())}
            print(f"seed {seed} alpha {alpha:g}: calib nll {nll:.5f}, valid bpw {bits.mean():.5f}", flush=True)
        all_best = min(ALPHAS, key=lambda x: row["by_alpha"][f"{x:g}"]["calib_nll"])
        row["check_registered_alpha_reproduces_run"] = abs(row["by_alpha"]["0.01"]["valid_bpw"] - row["run_counted_valid_bpw"]) < 1e-9
        row["alpha_by_calib_within_extended_grid"] = all_best
        row["valid_bpw_at_that_alpha"] = row["by_alpha"][f"{all_best:g}"]["valid_bpw"]
        row["counted_minus_learned_valid_at_that_alpha"] = row["valid_bpw_at_that_alpha"] - row["run_learned_valid_bpw"]
        row["counted_minus_learned_valid_registered"] = row["run_counted_valid_bpw"] - row["run_learned_valid_bpw"]
        res["seeds"][str(seed)] = row
        print(json.dumps({k: v for k, v in row.items() if k != "by_alpha"}), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
