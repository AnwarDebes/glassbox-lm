"""Exact evaluation paths of a trained CoTM on the CPU: speed and integer equality (read-only use of the decision_layer code).

Model: results/explore/abcd_models/<--model>.npz (ta, w); rows: the dev bits that prof_view.py rebuilt (their SHA-256
equals the run's record). Paths, all giving int64 class sums [n, C]:
  torch_ref   CoTM.class_sums on the CPU (float32 matrix products; the training-time dev pass)
  torch_once  the same with the include mask built once per call (fast_tm.FastCoTM E1)
  cpu_tm      dl.cpu_tm.class_sums (64-bit packed literal masks)
  cpu_fast    dl.cpu_tm_fast.class_sums (CSR literal lists + a Python loop over rows for the vote)
  csr_mm      dl.cpu_tm_fast.fired + one float64 matrix product for the vote (exact: integer sums far below 2^53)
Checks: every path equal to torch_ref (np.array_equal), and argmax accuracy equal to the record's dev acc.
"""
from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(HERE))
import prof_tm  # noqa: E402,F401  (CPU patch of torch.cuda.synchronize, sys.path)
from fast_tm import FastCoTM  # noqa: E402

from dl import cpu_tm, cpu_tm_fast  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402


def csr_mm(model, X, budget=8_000_000):
    n = X.shape[0]
    W = model["w"].astype(np.float64)
    chunk = max(1, budget // max(1, model["n_literals"]))
    out = np.zeros((n, W.shape[1]), np.int64)
    for s in range(0, n, chunk):
        F = cpu_tm_fast.fired(model, X[s : s + chunk])
        out[s : s + chunk] = np.rint(F.astype(np.float64) @ W).astype(np.int64)
    return out


def timeit(fn, reps):
    ts, r = [], None
    for _ in range(reps):
        t = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t)
    return r, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e74c_abcd_Bc4_c4000_s16_seed1")
    ap.add_argument("--bits", required=True, help="prof_view.py's <out>_dev_bits.npz")
    ap.add_argument("--record-tag", default="e74c_abcd_Bc4_c4000_s16")
    ap.add_argument("--rows", type=int, default=0, help="0 = every dev row")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    z = np.load(ROOT / "results" / "explore" / "abcd_models" / f"{args.model}.npz")
    ta, w = z["ta"].astype(np.int32), z["w"].astype(np.int32)
    cfg = TMConfig(**json.loads(str(z["cfg"])))
    d = np.load(args.bits)
    X, y = d["X"], d["y"]
    if args.rows:
        X, y = X[: args.rows], y[: args.rows]
    assert ta.shape[1] == 2 * X.shape[1], (ta.shape, X.shape)
    rec = None
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("tag") == args.record_tag and r.get("arm") == "tm" and r.get("seed") == int(args.model.rsplit("seed", 1)[1]):
            rec = r
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": args.model, "n_rows": int(X.shape[0]),
           "n_bits": int(X.shape[1]), "n_clauses": int(ta.shape[0]), "n_classes": int(w.shape[1]), "threads": torch.get_num_threads()}
    tm = CoTM(w.shape[1], X.shape[1], cfg, device="cpu")
    tm.ta, tm.w = torch.as_tensor(ta), torch.as_tensor(w)
    fm = FastCoTM(w.shape[1], X.shape[1], cfg, device="cpu")
    fm.ta, fm.w = tm.ta, tm.w
    Xt = torch.as_tensor(X, dtype=torch.uint8)
    paths = {}
    S_ref, t = timeit(lambda: CoTM.class_sums(tm, Xt).numpy().astype(np.int64), args.reps)
    paths["torch_ref"] = t
    S1, t = timeit(lambda: fm.class_sums(Xt).numpy().astype(np.int64), args.reps)
    paths["torch_once"] = t
    t0 = time.perf_counter()
    m0 = cpu_tm.pack(ta, w, cfg.n_states)
    m1 = cpu_tm_fast.pack(ta, w, cfg.n_states)
    res["pack_s"] = time.perf_counter() - t0
    S2, t = timeit(lambda: cpu_tm.class_sums(m0, X), args.reps)
    paths["cpu_tm"] = t
    S3, t = timeit(lambda: cpu_tm_fast.class_sums(m1, X), args.reps)
    paths["cpu_fast"] = t
    S4, t = timeit(lambda: csr_mm(m1, X), args.reps)
    paths["csr_mm"] = t
    res["times_s"] = {k: [round(x, 4) for x in v] for k, v in paths.items()}
    res["equal_to_torch_ref"] = {k: bool(np.array_equal(S_ref, S)) for k, S in
                                 (("torch_once", S1), ("cpu_tm", S2), ("cpu_fast", S3), ("csr_mm", S4))}
    res["max_abs_class_sum"] = int(np.abs(S_ref).max())
    acc = float((S_ref.argmax(1) == y).mean())
    res["dev_acc_from_sums"] = acc
    res["record_dev_acc"] = rec.get("acc") if rec else None
    res["record_bits_sha256_dev"] = rec.get("bits_sha256", {}).get("dev") if rec else None
    res["n_clauses_kept"] = m1["n_clauses_kept"]
    res["n_literals_kept"] = m1["n_literals"]
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
