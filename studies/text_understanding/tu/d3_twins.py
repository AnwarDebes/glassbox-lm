#!/usr/bin/env python
"""Step 3d, two control tables (tu/PLAN_3d.md "Arms"; CPU, seconds), both with a3B's word rows unchanged and d3J's
pair list:

- MID ("no new meaning"): a pair's row is the unit mean of its two words' a3B rows. existsT is a max over tokens, so an
  extra token can raise a concept bit without new meaning; MID measures that.
- BJ: a pair's row is d3J's fitted pair row. With a3B's word rows held fixed, BJ - MID is the fitted pair rows against
  their midpoints, and BJ - MW the fitted rows against rows embedded alone.

    python -I -B d3_twins.py --selftest   # tiny synthetic tables (no files)
    python -I -B d3_twins.py              # tu/d3/tables/d3MID_d768.npz, d3BJ_d768.npz, then tu/d3/twins_meta.json (last)

Output in d3_fit.py's format. Refuses to overwrite. Tests only: with $D3_SMOKE_DIR set, a3B_d768.npz is read from that
folder and d3J and the outputs live under its out/.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
sys.path.insert(0, str(HERE))
import d3_fit as DF

A3B = DF.A3 / "tables" / "a3B_d768.npz"


def mid_rows(U: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    m = U[w1].astype(np.float32) + U[w2].astype(np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    assert (n > 0).all(), "a pair of opposite word rows has no midpoint direction"
    return (m / n).astype(np.float16)


def build(za, zj) -> dict[str, dict]:
    assert np.array_equal(za["words"], zj["words"]), "a3B and d3J have different word lists"
    w1, w2 = zj["pair_w1"].astype(np.int64), zj["pair_w2"].astype(np.int64)
    keep = {k: za[k] for k in ("words", "counts", "vocab_ids")}
    pl = {"pair_w1": zj["pair_w1"], "pair_w2": zj["pair_w2"], "pair_counts": zj["pair_counts"]}
    return {"d3MID_d768": {"codes": za["codes"], "pair_codes": mid_rows(za["codes"], w1, w2), **pl, **keep},
            "d3BJ_d768": {"codes": za["codes"], "pair_codes": zj["pair_codes"], **pl, **keep}}


def selftest() -> None:
    rng = np.random.default_rng(4)
    U = rng.normal(size=(6, 4))
    U = (U / np.linalg.norm(U, axis=1, keepdims=True)).astype(np.float16)
    words = np.array([f"w{i}" for i in range(6)])
    za = {"codes": U, "words": words, "counts": np.ones(6), "vocab_ids": np.arange(6)}
    zj = {"codes": rng.normal(size=(6, 4)).astype(np.float16), "words": words, "pair_w1": np.array([0, 2], np.int32),
          "pair_w2": np.array([1, 3], np.int32), "pair_counts": np.array([30, 20]),
          "pair_codes": rng.normal(size=(2, 4)).astype(np.float16)}
    t = build(za, zj)
    m = t["d3MID_d768"]["pair_codes"].astype(np.float32)
    want = U[0].astype(np.float32) + U[1].astype(np.float32)
    assert np.allclose(m[0], want / np.linalg.norm(want), atol=2e-3)
    assert np.array_equal(t["d3MID_d768"]["codes"], U) and np.array_equal(t["d3BJ_d768"]["codes"], U), "word rows must be a3B's"
    assert np.array_equal(t["d3BJ_d768"]["pair_codes"], zj["pair_codes"]), "BJ's pair rows must be d3J's"
    try:
        build(za, {**zj, "words": np.array([f"x{i}" for i in range(6)])})
        raise SystemExit("SELFTEST FAILED: different word lists were accepted")
    except AssertionError:
        pass
    print("SELFTEST PASSED")


def main() -> None:
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        return
    assert sys.argv[1:] == [], sys.argv
    sd = os.environ.get("D3_SMOKE_DIR")
    a3b, outd = (Path(sd) / "a3B_d768.npz", Path(sd) / "out") if sd else (A3B, DF.D3)
    pj = outd / "tables" / "d3J_d768.npz"
    outs = {n: outd / "tables" / f"{n}.npz" for n in ("d3MID_d768", "d3BJ_d768")}
    meta_p = outd / "twins_meta.json"
    for p in [*outs.values(), meta_p]:
        assert not p.exists(), f"{p} exists"
    za, zj = np.load(a3b), np.load(pj)
    for n, arrays in build(za, zj).items():
        np.savez(outs[n], **arrays)
    meta = {"pairs": len(zj["pair_w1"]), "smoke": bool(sd),
            "sha256": {"d3_twins.py": DF.sha16(Path(__file__).resolve()), "a3B": DF.sha16(a3b), "d3J": DF.sha16(pj),
                       **{n: DF.sha16(p) for n, p in outs.items()}},
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    meta_p.write_text(json.dumps(meta, indent=1))  # written last
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
