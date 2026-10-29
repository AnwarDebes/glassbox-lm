"""Time the readable agent's TM decision (batch 1, one CPU thread): the study's NumPy path (dl.cpu_tm_fast.decide)
against two plain C kernels with early exit (tmclause.c: literal lists; sparse 64-bit words).
Exactness first: the C class sums must equal dl.cpu_tm_fast.class_sums on every row checked.

usage:
  self-test (random small models):  python -I -B bench_c_clause.py --selftest --lib LIB.so
  benchmark on a saved model:        CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 python -I -B bench_c_clause.py \
                                         --dataset star --model MODEL.npz --lib LIB.so --out OUT.json [--n 1000]
Reads only; writes only --out. Benchmark only: no study code or file is changed.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import argparse
import ctypes
import json
import sys
import time

import numpy as np

ROOT = f"{_REPO_ROOT}/studies/decision_layer"
sys.path.insert(0, ROOT + "/src")
PARTS = ("mem", "words", "xcust", "xrecent")


def build_words(ta, n_states=256):
    """Sparse-word form of the kept clauses (the same clauses as cpu_tm_fast.pack keeps)."""
    incl = np.asarray(ta) > n_states
    f = incl.shape[1] // 2
    keep = np.flatnonzero(incl.any(1))
    nw = (f + 63) // 64

    def pk(B):
        b = np.packbits(B.astype(np.uint8), axis=1, bitorder="little")
        pad = nw * 8 - b.shape[1]
        if pad:
            b = np.concatenate([b, np.zeros((b.shape[0], pad), np.uint8)], 1)
        return np.ascontiguousarray(b).view(np.uint64)

    Pw, Qw = pk(incl[keep, :f]), pk(incl[keep, f:])
    nz = (Pw | Qw) != 0
    rows, cols = np.nonzero(nz)
    ws = np.concatenate([[0], np.cumsum(nz.sum(1))]).astype(np.int64)
    return {"f": f, "nw": nw, "ws": ws, "widx": cols.astype(np.int32), "pm": np.ascontiguousarray(Pw[rows, cols]),
            "nm": np.ascontiguousarray(Qw[rows, cols]), "n_words_total": int(len(cols))}


class CKernels:
    def __init__(self, lib_path, ta, w):
        from dl import cpu_tm_fast
        self.m = cpu_tm_fast.pack(ta, w)
        self.W = np.ascontiguousarray(self.m["w"], dtype=np.int64)
        self.nc, self.ncls = self.W.shape
        self.f = self.m["n_features"]
        self.starts = np.ascontiguousarray(np.append(self.m["starts"], len(self.m["lits"])).astype(np.int64))
        self.lits = np.ascontiguousarray(self.m["lits"].astype(np.int64))
        self.wd = build_words(ta)
        assert self.wd["f"] == self.f and len(self.wd["ws"]) == self.nc + 1
        lib = ctypes.CDLL(lib_path)
        vp, ci = ctypes.c_void_p, ctypes.c_int
        lib.sums_lits.argtypes = [vp, ci, ci, vp, vp, vp, ci, vp]
        lib.sums_words.argtypes = [vp, ci, vp, vp, vp, vp, vp, ci, vp]
        lib.pack_x.argtypes = [vp, ci, vp]
        self.lib = lib
        self.out = np.zeros(self.ncls, np.int64)
        self.xw = np.zeros(self.wd["nw"], np.uint64)
        p = lambda a: a.ctypes.data_as(vp)
        self._p = p
        self._args_l = (self.nc, p(self.starts), p(self.lits), p(self.W), self.ncls, p(self.out))
        wd = self.wd
        self._args_w = (self.nc, p(wd["ws"]), p(wd["widx"]), p(wd["pm"]), p(wd["nm"]), p(self.W), self.ncls, p(self.out))

    def sums_lits(self, x):
        x = np.ascontiguousarray(x[: self.f], dtype=np.uint8)
        nc, st, li, W, ncls, out = self._args_l
        self.lib.sums_lits(self._p(x), self.f, nc, st, li, W, ncls, out)
        return self.out.copy()

    def sums_words(self, x):
        x = np.ascontiguousarray(x[: self.f], dtype=np.uint8)
        self.lib.pack_x(self._p(x), self.f, self._p(self.xw))
        self.lib.sums_words(self._p(self.xw), *self._args_w)
        return self.out.copy()


def exact(ck, X):
    """True when both C kernels give dl.cpu_tm_fast.class_sums on every row of X."""
    from dl import cpu_tm_fast
    S = cpu_tm_fast.class_sums(ck.m, X)
    bad_l = sum(not np.array_equal(ck.sums_lits(X[i]), S[i]) for i in range(len(X)))
    bad_w = sum(not np.array_equal(ck.sums_words(X[i]), S[i]) for i in range(len(X)))
    return bad_l == 0 and bad_w == 0, bad_l, bad_w


def selftest(lib):
    rng = np.random.default_rng(7)
    for case, (nc, f, ncls, p_inc) in enumerate([(50, 130, 4, 0.03), (200, 64, 7, 0.05), (300, 1000, 30, 0.004),
                                                 (40, 1, 2, 0.5), (120, 333, 187, 0.01)]):
        ta = np.where(rng.random((nc, 2 * f)) < p_inc, 300, 100)
        ta[0] = 100  # an empty clause: dropped by pack
        w = rng.integers(-20, 21, size=(nc, ncls))
        X = (rng.random((300, f)) < 0.5).astype(np.uint8)
        X[0] = 0
        X[1] = 1
        ck = CKernels(lib, ta, w)
        ok, bl, bw = exact(ck, X)
        print(f"selftest case {case}: clauses kept {ck.nc} of {nc}, features {f}, classes {ncls}: "
              f"{'PASS' if ok else 'FAIL'} (rows differing: lits {bl}, words {bw})")
        if not ok:
            return 1
    print("selftest: all PASS")
    return 0


def timed(fn, rows, warm=20):
    for x in rows[:warm]:
        fn(x)
    t = []
    for x in rows:
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    return np.array(t)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"])
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    assert sys.byteorder == "little"
    if a.selftest:
        return selftest(a.lib)
    from dl import count_text, cpu_tm_fast
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = count_text.patch("c4words-S1000-v1")
    from dl.abcd import ABCDView, load_examples
    tr, blist = load_examples("train", a.dataset)
    dv, _ = load_examples("dev", a.dataset)
    view = ABCDView(parts=PARTS, lex_k=1000, seed=0).fit(tr, {}, blist)
    Xdv = np.ascontiguousarray(view.transform(dv, {}), dtype=np.uint8)
    z = np.load(a.model)
    ck = CKernels(a.lib, z["ta"], z["w"])
    ok, bl, bw = exact(ck, Xdv)
    assert ok, f"C class sums differ from the NumPy reference (rows: lits {bl}, words {bw})"
    idx = np.random.default_rng(0).choice(len(dv), size=min(a.n, len(dv)), replace=False)  # agent_speed.py's rows
    rows = [np.ascontiguousarray(Xdv[i]) for i in idx]
    t_view = timed(lambda i: view.transform([dv[i]], {}), list(idx))
    t_np = timed(lambda x: cpu_tm_fast.decide(ck.m, x), rows)
    t_cl = timed(lambda x: int(ck.sums_lits(x).argmax()), rows)
    t_cw = timed(lambda x: int(ck.sums_words(x).argmax()), rows)
    med = lambda t: float(np.median(t))
    res = {"dataset": a.dataset, "model": a.model, "table": meta.get("table"), "n_dev_rows_exact": int(len(Xdv)),
           "exact_lits": bl == 0, "exact_words": bw == 0, "n_timed": len(rows), "n_clauses_kept": ck.nc,
           "n_classes": ck.ncls, "n_features": ck.f, "n_literals": int(len(ck.lits)),
           "n_sparse_words": ck.wd["n_words_total"],
           "median_ms": {"view_text_to_bits": med(t_view), "tm_numpy_current": med(t_np), "tm_c_lits": med(t_cl),
                         "tm_c_words": med(t_cw)},
           "p90_ms": {"tm_numpy_current": float(np.percentile(t_np, 90)), "tm_c_lits": float(np.percentile(t_cl, 90)),
                      "tm_c_words": float(np.percentile(t_cw, 90))},
           "label": "rough: shared machine, one CPU thread, batch 1, nice 19; C built with gcc -O3 -march=native",
           "utc_start": t_start, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    best_c = min(res["median_ms"]["tm_c_lits"], res["median_ms"]["tm_c_words"])
    res["tm_speedup_best_c"] = res["median_ms"]["tm_numpy_current"] / best_c
    tot_now = res["median_ms"]["view_text_to_bits"] + res["median_ms"]["tm_numpy_current"]
    tot_c = res["median_ms"]["view_text_to_bits"] + best_c
    res["decision_total_ms"] = {"now": tot_now, "with_best_c": tot_c, "speedup": tot_now / tot_c}
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
