#!/usr/bin/env python
"""Tests for b3_twins.py on synthetic bits (a plain script, no pytest; CPU only):

    OMP_NUM_THREADS=2 nice -n 19 $PY -I -B tu/tests/test_b3_twins.py

Linear student: planted linear threshold targets must be learned to a high held-out balanced accuracy, the lam grid is
reported, and the closed form equals a direct least-squares solve. Counting student: m_ij and the thresholds on a
hand-checked case. CLI: train, predict (both models), refusal to overwrite. Each run has its own folder in
tu/tests/runs/<UTC stamp>_b3twins/.
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
import b3_twins as W
from b3_student import pack, unpack

RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_b3twins")
results = {"pass": 0, "fail": 0}
torch.set_num_threads(2)
dev = torch.device("cpu")


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


rng = np.random.default_rng(3)
F, J = 40, 10
Wt = rng.standard_normal((F, J))


def planted(n, seed):
    r = np.random.default_rng(seed)
    X = r.integers(0, 2, (n, F)).astype(np.uint8)
    s = (X - 0.5) @ Wt
    return X, (s > np.median(s, 0)).astype(np.uint8)


Xtr, Ytr = planted(4000, 1)
Xh, Yh = planted(1000, 2)
tr = {"X": pack(Xtr), "Y": pack(Ytr), "F": F, "J": J}
held = {"X": pack(Xh), "Y": pack(Yh), "F": F, "J": J}
lin, scores = W.fit_linear(tr, held, dev)
check("linear: every lam of the grid is scored", set(scores) == {str(f) for f in W.GRID}, str(scores.keys()))
check(f"linear: planted linear targets are learned (held-out balanced accuracy {scores[str(lin['lam_factor'])]['balanced_acc']})",
      scores[str(lin["lam_factor"])]["balanced_acc"] > 93.0, json.dumps(scores))
check("linear: the pick is the best of the grid", scores[str(lin["lam_factor"])]["balanced_acc"]
      == max(v["balanced_acc"] for v in scores.values()))
# closed form against a direct solve with the same lam
f = lin["lam_factor"]
n = Xtr.shape[0]
Xc = Xtr - Xtr.mean(0)
Yc = (2.0 * Ytr - 1) - (2.0 * Ytr - 1).mean(0)
Wd = np.linalg.solve(Xc.T @ Xc + f * n * np.eye(F), Xc.T @ Yc)
check("linear: the streamed closed form equals a direct ridge solve", np.allclose(lin["W"], Wd, atol=1e-4),
      f"max diff {np.abs(lin['W'] - Wd).max():.2e}")

# counting on a hand-checked case: 4 rows, 3 input bits, 2 outputs
Xs = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], np.uint8)
Ys = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], np.uint8)
m = W.fit_counting({"X": pack(Xs), "Y": pack(Ys), "F": 3, "J": 2}, dev)
check("counting: m_ij is the mean of bit j where input i is on (0.5 where i is never on)",
      np.allclose(m["M"], [[1.0, 0.5], [0.5, 1.0], [0.5, 0.5]]), str(m["M"]))
S = W.scores_counting(m, pack(Xs), 3, dev)
check("counting: a row with no bit on scores 0.5", np.allclose(S[3], [0.5, 0.5]), str(S))
check("counting: the thresholds give the training on-rates", np.allclose(W.predict_counting(m, pack(Xs), 3, dev).mean(0), Ys.mean(0)),
      f"{W.predict_counting(m, pack(Xs), 3, dev).mean(0)} vs {Ys.mean(0)}")

# CLI
RUNS.mkdir(parents=True, exist_ok=True)
np.savez(RUNS / "train.npz", X=tr["X"], Y=tr["Y"], n_features=F, n_outputs=J)
np.savez(RUNS / "held.npz", X=held["X"], Y=held["Y"], n_features=F, n_outputs=J)
np.savez(RUNS / "x.npz", X=held["X"], n_features=F)
cmd = [sys.executable, "-I", "-B", str(TU / "b3_twins.py"), "--train", str(RUNS / "train.npz"), "--held", str(RUNS / "held.npz"),
       "--out-dir", str(RUNS), "--tag", "t"]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
check("CLI: exit 0", p.returncode == 0, p.stderr[-600:])
rec = json.loads((RUNS / "t.twins.json").read_text()) if (RUNS / "t.twins.json").exists() else {}
check("CLI: the record has both students' held-out metrics and hashes", "held" in rec.get("linear", {})
      and "held" in rec.get("counting", {}) and len(rec.get("inputs_sha256", {})) == 2, str(rec)[:300])
check("CLI: the linear pick equals the in-process pick", rec.get("linear", {}).get("lam_factor") == lin["lam_factor"])
for kind, fn in (("linear", lambda x: W.predict_linear(lin, x, F, dev)), ("counting", None)):
    q = subprocess.run([sys.executable, "-I", "-B", str(TU / "b3_twins.py"), "predict", "--model", str(RUNS / f"t.{kind}.npz"),
                        "--x", str(RUNS / "x.npz"), "--out", str(RUNS / f"p_{kind}.npz")], capture_output=True, text=True,
                       timeout=600, check=False)
    check(f"CLI predict {kind}: exit 0", q.returncode == 0, q.stderr[-600:])
    if q.returncode == 0 and fn is not None:
        z = np.load(RUNS / f"p_{kind}.npz")
        check("CLI predict linear: equals the in-process prediction",
              np.array_equal(unpack(torch.as_tensor(z["X"]), int(z["n_features"])).numpy(), fn(held["X"])))
p2 = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
check("CLI: existing outputs are refused (exit 2)", p2.returncode == 2 and "refusing to overwrite" in p2.stderr, p2.stderr)

print(f"b3_twins tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
