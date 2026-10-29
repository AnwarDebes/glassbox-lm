#!/usr/bin/env python
"""Tests for b3_targets.py on synthetic vectors (a plain script, no pytest; CPU only):

    OMP_NUM_THREADS=2 nice -n 19 $PY -I -B tu/tests/test_b3_targets.py

The teacher paths (numerics, choose) are not run here: they need bge-base and are the approved CPU steps themselves.
Each run has its own folder in tu/tests/runs/<UTC stamp>_b3targets/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TU))
import b3_targets as B

RUNS = TU / "tests" / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_b3targets")
results = {"pass": 0, "fail": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))


rng = np.random.default_rng(7)
d, n = 12, 6000
A = rng.standard_normal((d, d)) * np.linspace(3.0, 0.3, d)  # unequal axes
T = (rng.standard_normal((n, d)) @ A + 0.5).astype(np.float16)
c = B.fit_codes(T, n_sub=4000, n_pt=4, iters=30, chunk=1000)
check("eigenvalues are in descending order", bool(np.all(np.diff(c["eigvals"]) <= 1e-6)), str(c["eigvals"]))
check("the eigenvectors are orthonormal", np.allclose(c["P"].T @ c["P"], np.eye(d), atol=1e-4))
check("each eigenvector's largest entry is positive", bool(np.all(c["P"][np.abs(c["P"]).argmax(0), np.arange(d)] > 0)))
check("the ITQ rotation is orthogonal", np.allclose(c["R_itq"].T @ c["R_itq"], np.eye(d), atol=1e-4))
check("mu is the mean", np.allclose(c["mu"], T.astype(np.float64).mean(0), atol=1e-3))
sh, pt, itq = (B.apply_code(c, k, T) for k in ("SH", "PT", "ITQ"))
check("shapes: SH and ITQ d bits, PT 3 x n_pt bits", sh.shape == (n, d) and itq.shape == (n, d) and pt.shape == (n, 12),
      f"{sh.shape} {pt.shape} {itq.shape}")
check("SH bits on centred vectors are balanced (0.4-0.6)", 0.4 < sh.mean() < 0.6, str(sh.mean()))
lv = pt.reshape(n, 4, 3).mean((0, 1))
check("PT levels are on for about 75 / 50 / 25%", abs(lv[0] - 0.75) < 0.03 and abs(lv[1] - 0.5) < 0.03 and abs(lv[2] - 0.25) < 0.03, str(lv))
check("PT bits are a thermometer (a higher level implies the lower ones)", bool(np.all(np.diff(pt.reshape(n, 4, 3).astype(int), axis=2) <= 0)))
V = (T.astype(np.float64) - c["mu"]) @ c["P"]
R0 = np.linalg.qr(np.random.default_rng(B.SEED_ITQ).standard_normal((d, d)))[0]


def qloss(R):
    return float(((np.where(V @ R >= 0, 1.0, -1.0) - V @ R) ** 2).sum())


check("ITQ lowers the quantisation loss from its start", qloss(c["R_itq"].astype(np.float64)) < qloss(R0),
      f"{qloss(c['R_itq'].astype(np.float64)):.1f} vs {qloss(R0):.1f}")
c2 = B.fit_codes(T, n_sub=4000, n_pt=4, iters=30, chunk=1000)
check("fit is deterministic", all(np.array_equal(c[k], c2[k]) for k in c), str([k for k in c if not np.array_equal(c[k], c2[k])]))
c3 = B.fit_codes(T, n_sub=4000, n_pt=4, iters=30, chunk=777)
check("the chunk size does not change the fit (to float32 rounding)", np.allclose(c["P"], c3["P"], atol=1e-4)
      and np.allclose(c["mu"], c3["mu"], atol=1e-6))
tr, cv = B.carve(1001)
check("carve: 90 / 10, disjoint, covering", len(cv) == 100 and len(tr) == 901 and not set(tr) & set(cv) and len(set(tr) | set(cv)) == 1001)
check("carve is deterministic", np.array_equal(B.carve(1001)[1], cv))
check("pick: SH within 0.1 of the best wins", B.pick({"SH": 90.0, "PT": 90.09, "ITQ": 89.0}) == "SH")
check("pick: otherwise the best", B.pick({"SH": 90.0, "PT": 90.2, "ITQ": 89.0}) == "PT" and B.pick({"SH": 80.0, "PT": 81.0, "ITQ": 82.0}) == "ITQ")


# choose() plumbing with a nearest-centroid stand-in for the LR (the real run uses dl.linear on the CPU)
def fit(X, y, k, C):
    M = np.stack([X[y == j].mean(0) if (y == j).any() else np.zeros(X.shape[1]) for j in range(k)])
    return M, C


def score(X, M, C):
    return ((X[:, None, :] - M[None]) ** 2).sum(2).argmin(1)


cent = rng.standard_normal((3, d)) * 2
data = {}
for s, m in (("a", 300), ("b", 200)):
    y = rng.integers(0, 3, m)
    data[s] = ((cent[y] + rng.standard_normal((m, d)) * 0.5).astype(np.float16), y, 3)
res = B.choose(c, data, fit, score)
check("choose: every set has every code", all(set(B.CODES) <= set(res["sets"][s]) for s in data), str(res["sets"].keys()))
check("choose: carved sizes", res["sets"]["a"]["n_train"] == 270 and res["sets"]["a"]["n_carved_val"] == 30)
check("choose: separable data gives high carved accuracy", min(res["sets"][s][k]["best"] for s in data for k in B.CODES) > 80,
      json.dumps(res["means"]))
check("choose: the pick is a code", res["pick"] in B.CODES and set(res["means"]) == set(B.CODES))

# the fit CLI on a small file of the teacher's width, then the refusal to overwrite
RUNS.mkdir(parents=True, exist_ok=True)
Tf = RUNS / "T_small.f16.npy"
G = np.random.default_rng(3).standard_normal((3000, 768)).astype(np.float32)
G /= np.linalg.norm(G, axis=1, keepdims=True)
np.save(Tf, G.astype(np.float16))
out = RUNS / "codes.npz"
p = subprocess.run([sys.executable, "-I", "-B", str(TU / "b3_targets.py"), "fit", "--train", str(Tf), "--out", str(out)],
                   capture_output=True, text=True, timeout=600, check=False)
check("fit CLI: exit 0", p.returncode == 0, p.stderr[-800:])
z = np.load(out) if out.exists() else None
check("fit CLI: 768-bit codes", z is not None and z["W_sh"].shape == (768, 768) and z["th_pt"].shape == (256, 3)
      and z["R_itq"].shape == (768, 768), "")
info = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {}
check("fit CLI: on-shares reported for the 3 codes", set(info.get("on_share_first_20000", {})) == set(B.CODES), p.stdout[-400:])
p2 = subprocess.run([sys.executable, "-I", "-B", str(TU / "b3_targets.py"), "fit", "--train", str(Tf), "--out", str(out)],
                    capture_output=True, text=True, timeout=600, check=False)
check("fit CLI: an existing output is refused (exit 2)", p2.returncode == 2 and "refusing to overwrite" in p2.stderr, p2.stderr)
bits = B.apply_code(dict(z), "PT", G[:10]) if z is not None else np.zeros((0, 0))
check("PT on the teacher's width gives 768 bits", bits.shape == (10, 768), str(bits.shape))

print(f"b3_targets tests: {results['pass']} passed, {results['fail']} failed")
sys.exit(0 if results["fail"] == 0 else 1)
