#!/usr/bin/env python
"""Self-tests of the TM side (torch env, plain asserts; run by tests/run_tests.sh as python -I -B):
the CPU loop learns a rule, its scores equal the packed CPU decision (dl cpu_tm_fast), it is deterministic, and it
needs no CUDA; then a timing projection of one K3c pilot at the real width (no assertion on the time)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A))
from astruct import paths

paths.check_env()
import torch

torch.set_num_threads(2)
from astruct.tmrun import tm_scores, train_tm

paths.add_study_paths(dl=True)
from dl import cpu_tm_fast

PASSED = []


def check(name, cond):
    if not cond:
        raise SystemExit(f"FAIL {name}")
    PASSED.append(name)
    print("ok", name, flush=True)


def auc(s, y):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def main() -> None:
    check("cuda not visible", not torch.cuda.is_available())
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, (3000, 12)).astype(np.uint8)
    y = (X[:, 0] & (1 - X[:, 1])).astype(np.int64)
    tm = train_tm(X, y, T=20, s=3.0, seed=1, epochs=5, n_clauses=60)
    s = tm_scores(tm, X)
    check("learns x0 AND NOT x1 (AUC > 0.95)", auc(s, y) > 0.95)
    st = tm.state()
    packed = cpu_tm_fast.pack(st["ta"], st["w"], n_states=tm.cfg.n_states, negation=True)
    S = cpu_tm_fast.class_sums(packed, X)
    check("scores equal the packed CPU decision", np.array_equal(S[:, 1] - S[:, 0], s.astype(np.int64)))
    tm2 = train_tm(X, y, T=20, s=3.0, seed=1, epochs=5, n_clauses=60)
    check("deterministic for a seed", np.array_equal(tm2.state()["ta"], st["ta"]) and np.array_equal(tm2.state()["w"], st["w"]))
    tm3 = train_tm(X, y, T=20, s=3.0, seed=2, epochs=5, n_clauses=60)
    check("another seed gives another machine", not np.array_equal(tm3.state()["ta"], st["ta"]))
    seen = []
    train_tm(X[:500], y[:500], T=20, s=3.0, seed=1, epochs=2, n_clauses=20, epoch_hook=lambda e, m: seen.append(e))
    check("epoch hook sees every epoch (epoch-1 state can be saved)", seen == [1, 2])
    # timing at the real width: 364 bits (about 40 ones per row), 4,000 clauses, T 8,000, s 15, batch 128
    Xr = (rng.random((128 * 12, 364)) < 0.11).astype(np.uint8)
    yr = (rng.random(128 * 12) < 0.1).astype(np.int64)
    t0 = time.time()
    tmr = train_tm(Xr, yr, T=8000, s=15.0, seed=1, epochs=1, n_clauses=4000)
    per_batch = (time.time() - t0) / 12
    pilot_rows = 210_000  # about 10% of about 2.1M training rows (plan, section 3)
    proj = per_batch * (pilot_rows / 128) * 10 / 3600
    out = {"seconds_per_batch_cpu_2_threads": round(per_batch, 4), "projected_hours_per_pilot_210k_rows_10_epochs": round(proj, 2),
           "clauses": 4000, "bits": 364, "empty_after_12_batches": tmr.stats()["empty"]}
    print("timing", json.dumps(out), flush=True)
    print(f"TM TESTS: {len(PASSED)} passed", flush=True)


if __name__ == "__main__":
    main()
