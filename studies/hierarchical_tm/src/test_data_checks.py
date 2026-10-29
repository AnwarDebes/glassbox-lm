"""CPU test of data_checks.length_feats (bug HTM-B2, 2026-10-25), on synthetic rows only: no data file is read.

The pair matrices are built as pairdata.load_pairs builds them (pairdata.ids_to_csr: 0/1 values of type uint8), and
scipy returns their row sums as uint64. Before the fix of 2026-10-25 the feature |len A - len B| wrapped around to about
1.8e19 whenever paragraph B was the longer one; the four other features were right.
Run from the study folder:
  PYTHONDONTWRITEBYTECODE=1 ../tm_language_model/.venv-tm/bin/python src/test_data_checks.py [--src DIR]
--src DIR imports data_checks.py from DIR instead of this src/ (for example the version before the fix); pairdata.py
always comes from this src/. Exit status 0 when every test passes, else 1.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from pairdata import ids_to_csr  # noqa: E402

F = 1024


def load_data_checks(src_dir):
    spec = importlib.util.spec_from_file_location("data_checks_under_test", os.path.join(src_dir, "data_checks.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pairs(lengths_a, lengths_b, rng):
    """A pair dict with sparse uint8 0/1 rows of the given word counts (distinct sorted word ids < F per row)."""
    out = {}
    for key, lens in (("A", lengths_a), ("B", lengths_b)):
        rows = [np.sort(rng.choice(F, size=n, replace=False)) for n in lens]
        ids = np.concatenate(rows).astype(np.uint16) if rows else np.zeros(0, np.uint16)
        off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
        out[key] = ids_to_csr(ids, off, F)
    return out


def expected(lengths_a, lengths_b):
    return np.array([[a, b, abs(a - b), min(a, b), max(a, b)] for a, b in zip(lengths_a, lengths_b)], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC, help="folder of the data_checks.py under test (default: this src/)")
    x = ap.parse_args()
    dc = load_data_checks(os.path.abspath(x.src))
    print(f"data_checks.py under test: {os.path.join(os.path.abspath(x.src), 'data_checks.py')}")
    rng, results = np.random.default_rng(20261025), []

    def check(name, cond, detail=""):
        results.append(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  [{detail}]"), flush=True)

    la, lb = [1, 2, 5, 0, 3, 0], [3, 2, 1, 4, 0, 0]
    p = pairs(la, lb, rng)
    check("the synthetic rows are uint8 like pairdata.ids_to_csr, and their row sums come back unsigned",
          p["A"].dtype == np.uint8 and np.asarray(p["A"].sum(1)).dtype.kind == "u", f"{p['A'].dtype}")
    got = dc.length_feats(p)
    check("hand-made rows: (len A, len B, |len A - len B|, min, max) exactly", np.array_equal(got, expected(la, lb)),
          f"got {got.tolist()}")
    check("no value above the largest possible length (a pair with lengths 1 and 3 gives 2, not 1.8e19)",
          np.isfinite(got).all() and got.max() <= F, f"max {got.max()}")
    q = {"A": p["B"], "B": p["A"]}
    g2 = dc.length_feats(q)
    check("swapping A and B swaps the two length columns and leaves the three others unchanged",
          np.array_equal(g2[:, [1, 0]], got[:, :2]) and np.array_equal(g2[:, 2:], got[:, 2:]), f"{g2.tolist()}")
    ra, rb = rng.integers(0, 61, 2000).tolist(), rng.integers(0, 61, 2000).tolist()
    g3 = dc.length_feats(pairs(ra, rb, rng))
    check("2,000 random rows with lengths 0 to 60 equal the reference computed with Python integers",
          np.array_equal(g3, expected(ra, rb)) and g3.dtype == np.float64,
          f"{int((g3 != expected(ra, rb)).any(axis=1).sum())} rows differ")
    n_fail = results.count(False)
    print(f"{len(results) - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
