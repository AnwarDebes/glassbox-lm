#!/usr/bin/env python
"""Tests for b3_student.py on planted targets (a plain script, no pytest; CPU only, 2 threads):

    OMP_NUM_THREADS=2 nice -n 19 $PY -I -B tu/tests/test_b3_student.py

Planted job: 48 random input bits; 12 outputs, each an AND or an OR of 2 input bits. The student must learn them to at
least 99% balanced held-out accuracy (PLAN_3b.md "Code checks"). Also: the bit packing, the metrics on fixed cases,
the CLI train / predict round trip, the stop rule and the refusal to overwrite. Each run has its own folder in
tu/tests/runs/<UTC stamp>_b3student/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TU))
import b3_student as S

RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_b3student")
results = {"pass": 0, "fail": 0}
torch.set_num_threads(2)


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


# packing
rng = np.random.default_rng(11)
b = rng.integers(0, 2, (7, 21)).astype(np.uint8)
check("unpack(pack(x)) == x", np.array_equal(S.unpack(torch.as_tensor(S.pack(b)), 21).numpy(), b))

# metrics on fixed cases
Y = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]], np.uint8)
P = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1]], np.uint8)
m = S.metrics(P, Y)
check("metrics: constant outputs are left out and counted", m["constant_outputs"] == 2, str(m))
check("metrics: balanced accuracy of the one varying output", abs(m["balanced_acc"] - 75.0) < 1e-6, str(m))
check("metrics: plain bit accuracy", abs(m["bit_acc"] - 100 * 11 / 12) < 1e-3, str(m))
check("metrics: implied cosine of identical codes is 1", S.metrics(Y, Y)["implied_cos"] == 1.0)


def planted(n, seed):
    r = np.random.default_rng(seed)
    X = r.integers(0, 2, (n, 48)).astype(np.uint8)
    cols = []
    for k in range(12):
        i, j = 2 * k, 2 * k + 1
        cols.append((X[:, i] & X[:, j]) if k % 2 == 0 else (X[:, i] | X[:, j]))
    return X, np.stack(cols, 1).astype(np.uint8)


Xtr, Ytr = planted(6000, 1)
Xh, Yh = planted(2000, 2)
tr = {"X": S.pack(Xtr), "Y": S.pack(Ytr), "F": 48, "J": 12}
held = {"X": S.pack(Xh), "Y": S.pack(Yh), "F": 48, "J": 12}
cfg = S.TMConfig(n_clauses=200, T=30, s=3.0, epochs=8, batch_size=256, seed=1, half=False)
tm = S.CoTM(12, 48, cfg, device="cpu")
t0 = time.time()
curve = S.train(tm, tr, held, R=4, epochs=8, batch=256, seed=1, log=lambda s: None)
el = time.time() - t0
last = curve[-1]
check(f"planted AND / OR targets learned to >= 99% balanced accuracy ({last['balanced_acc']}, {el:.0f} s)",
      last["balanced_acc"] is not None and last["balanced_acc"] >= 99.0, json.dumps(curve[-2:]))
check("the curve has one row per epoch with the metrics", len(curve) == 8 and all("balanced_acc" in r for r in curve))
pred = S.predict_bits(tm, held["X"], 48)
check("predict_bits gives the held-out shape", pred.shape == (2000, 12) and pred.dtype == np.uint8)

# a fixed seed gives the same machine
tm2 = S.CoTM(12, 48, cfg, device="cpu")
S.train(tm2, tr, None, R=4, epochs=2, batch=256, seed=1, log=lambda s: None)
tm3 = S.CoTM(12, 48, cfg, device="cpu")
S.train(tm3, tr, None, R=4, epochs=2, batch=256, seed=1, log=lambda s: None)
check("same seed, same machine", torch.equal(tm2.ta, tm3.ta) and torch.equal(tm2.w, tm3.w))

# CLI round trip
RUNS.mkdir(parents=True, exist_ok=True)
np.savez(RUNS / "train.npz", X=tr["X"], Y=tr["Y"], n_features=48, n_outputs=12)
np.savez(RUNS / "held.npz", X=held["X"], Y=held["Y"], n_features=48, n_outputs=12)
np.savez(RUNS / "x.npz", X=held["X"], n_features=48)
cmd = [sys.executable, "-I", "-B", str(TU / "b3_student.py"), "train", "--train", str(RUNS / "train.npz"), "--held",
       str(RUNS / "held.npz"), "--out-dir", str(RUNS), "--tag", "cli", "--clauses", "200", "--T", "30", "--s", "3",
       "--R", "4", "--epochs", "8", "--batch", "256", "--seed", "1", "--device", "cpu"]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
check("CLI train: exit 0", p.returncode == 0, p.stderr[-800:])
rec = json.loads((RUNS / "cli.json").read_text()) if (RUNS / "cli.json").exists() else {}
check("CLI train: the record has the curve, the hashes and the state hash", len(rec.get("curve", [])) == 8
      and "state_sha256" in rec and len(rec.get("inputs_sha256", {})) == 2, str(rec)[:300])
check("CLI train: the same machine as the in-process run (same seed and settings)",
      bool(rec) and rec["curve"][-1]["balanced_acc"] == last["balanced_acc"], f"{rec.get('curve', [{}])[-1]} vs {last}")
p2 = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
check("CLI train: an existing record is refused (exit 2)", p2.returncode == 2 and "refusing to overwrite" in p2.stderr, p2.stderr)
pp = subprocess.run([sys.executable, "-I", "-B", str(TU / "b3_student.py"), "predict", "--state", str(RUNS / "cli.state.npz"),
                     "--x", str(RUNS / "x.npz"), "--out", str(RUNS / "pred.npz"), "--device", "cpu"],
                    capture_output=True, text=True, timeout=600, check=False)
check("CLI predict: exit 0", pp.returncode == 0, pp.stderr[-800:])
if pp.returncode == 0:
    z = np.load(RUNS / "pred.npz")
    got = S.unpack(torch.as_tensor(z["X"]), int(z["n_features"])).numpy()
    check("CLI predict: the saved state predicts as the trained machine", np.array_equal(got, pred), "")
# the stop rule: a tiny max-hours stops after epoch 1 with exit 3
ps = subprocess.run(cmd[:11] + ["--tag", "stop", "--clauses", "200", "--T", "30", "--s", "3", "--R", "4", "--epochs", "50",
                                 "--batch", "256", "--seed", "1", "--device", "cpu", "--max-hours", "0.00001"],
                    capture_output=True, text=True, timeout=900, check=False)
check("stop rule: exit 3 after the first epoch, no state written", ps.returncode == 3 and "STOP" in ps.stdout
      and not (RUNS / "stop.state.npz").exists(), (ps.stdout + ps.stderr)[-500:])

print(f"b3_student tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
