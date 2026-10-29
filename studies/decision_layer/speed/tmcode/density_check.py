"""Does the view's data change the per-element cost? Same CoTM config (N 256, s 8, 2 epochs, 2,048 rows, unchunked, CPU,
2 threads) on two bit sets: (a) the decision_layer cached potion view B of ABCD (first 2,048 train rows; the view of e60 / R19) and
(b) the C4 view B of ABCD rebuilt here (results/view_abcd_c4_dev_bits.npz: dev rows, used only as unlabeled-shape input:
first 2,048 rows train, next 1,024 rows dev; equal to e74c's dev bits). Reports wall time, the binomial's time per
element and the nonzero fractions of the two count tensors (read by a spy that only reads the counts)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prof_tm  # noqa: E402

import dl.tm as T  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402


def run(name, Xtr, ytr, Xdv, ydv, C):
    T.CHUNK_ELEMS = 50_000_000
    st = {"t": 0.0, "n": 0, "nz": []}
    orig = torch.binomial

    def spy(count, prob, generator=None):
        st["nz"].append(float((count > 0).float().mean()))
        t = time.perf_counter()
        r = orig(count, prob, generator=generator)
        st["t"] += time.perf_counter() - t
        st["n"] += count.numel()
        return r

    torch.binomial = spy
    try:
        cfg = TMConfig(n_clauses=256, T=256, s=8.0, epochs=2, batch_size=128, seed=1)
        tm = CoTM(C, Xtr.shape[1], cfg, device="cpu")
        t0 = time.perf_counter()
        tm.fit(Xtr, ytr, Xdv, ydv)
        wall = time.perf_counter() - t0
    finally:
        torch.binomial = orig
    nz = np.array(st["nz"])
    return {"view": name, "bits": int(Xtr.shape[1]), "density": float(Xtr.mean()), "wall_s": wall,
            "binomial_s": st["t"], "binomial_ns_per_elem": 1e9 * st["t"] / st["n"],
            "nonzero_inc": float(nz[0::2].mean()), "nonzero_dec": float(nz[1::2].mean())}


def main():
    torch.set_num_threads(2)
    out = []
    Xtr, ytr, Xdv, ydv, C = prof_tm.load("abcd", 2048, 1024)
    out.append(run("potion view B (cache)", Xtr, ytr, Xdv, ydv, C))
    d = np.load(HERE / "results" / "view_abcd_c4_dev_bits.npz")
    X, y = d["X"], d["y"]
    out.append(run("C4 view B (rebuilt, e74c dev bits)", X[:2048], y[:2048], X[2048:3072], y[2048:3072], 30))
    for r in out:
        print(json.dumps(r))
    Path(HERE / "results" / "density_check.json").write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                                      "runs": out}, indent=1))


if __name__ == "__main__":
    main()
