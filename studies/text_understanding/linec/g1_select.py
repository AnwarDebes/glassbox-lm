#!/usr/bin/env python
"""Line C G1 (EXPLORATORY, validation only; rules in LOG.md before any run): job lists, selection and words,
mechanical.

    g1_select.py grid      # 6 jobs: T in {2000, 8000} x s in {2, 5, 15}, 4,000 clauses, seed 1
    g1_select.py wide      # the grid's best setting at 10,000 clauses, seed 1
    g1_select.py seeds     # seeds 2 and 3 at the pick
    g1_select.py words     # the words (needs tmlm: run with the thesis study's .venv-tm)

Selection uses CALIBRATION bits only (the mixed model's bits per token on the residue-2 rows; ties: the earlier job
in the listed order); the pick is the lower of the grid's best and its 10,000-clause twin (ties: 4,000 clauses).
Words, on the validation numbers at the pick:
- PASS: on each of seeds 1-3, TM bits per token <= count twin - 0.0291 (2% lower perplexity), the 95% document-
  bootstrap interval of (TM - count twin) bits per byte entirely below 0, and TM below the floor;
- STRONG: PASS, and the mean over seeds 1-3 of TM bits per token <= boosted trees';
- FAIL: otherwise. Logistic regression is printed beside.
A job line: stage|T|s|clauses|seed.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / "results_g1.jsonl"
GRID = [(T, s) for T in (2000, 8000) for s in (2.0, 5.0, 15.0)]
TWO_PCT = -np.log2(0.98)


def rows():
    return [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()] if ROWS.exists() else []


def find(R, T, s, clauses, seed):
    hit = [r for r in R if r["setting"]["T"] == T and float(r["setting"]["s"]) == float(s)
           and r["setting"]["clauses"] == clauses and r["seed"] == seed and r["setting"]["epochs"] == 10]
    return hit[0] if hit else None


def grid_best(R):
    sc = []
    for i, (T, s) in enumerate(GRID):
        r = find(R, T, s, 4000, 1)
        if r is None:
            sys.exit(f"missing grid row T {T} s {s}")
        sc.append((r["calibration_bits_per_token"], i, T, s))
    _, _, T, s = min(sc)
    return T, s


def pick(R):
    T, s = grid_best(R)
    a, b = find(R, T, s, 4000, 1), find(R, T, s, 10000, 1)
    if b is None:
        sys.exit("missing the 10,000-clause row")
    return (T, s, 10000) if b["calibration_bits_per_token"] < a["calibration_bits_per_token"] else (T, s, 4000)


def words(R) -> list:
    sys.path.insert(0, f"{_REPO_ROOT}/studies/tm_language_model/src")
    from tmlm.cvtm import protocol as P

    T, s, clauses = pick(R)
    vb = np.load(HERE / "g0" / "val_bits_per_position.npz")
    g0 = json.loads((HERE / "g0" / "results.json").read_text())["validation"]
    ev = P.load_eval_split("validation")
    doc_of = vb["doc_of"]
    nd = ev.n_docs
    twin_doc = np.bincount(doc_of, weights=vb["count_twin"], minlength=nd)
    idx = P.bootstrap_index(nd)
    twin, trees, lr, floor = (g0[k]["bits_per_token"] for k in ("count_twin", "trees", "logistic", "floor"))
    out = [f"G1 pick: T {T}, s {s}, {clauses} clauses (selected on calibration bits); floor {floor:.4f}, count twin "
           f"{twin:.4f}, logistic regression {lr:.4f}, boosted trees {trees:.4f} bits per token; PASS line {twin - TWO_PCT:.4f}"]
    ok_all, tm_bits = True, []
    for seed in (1, 2, 3):
        r = find(R, T, s, clauses, seed)
        if r is None:
            sys.exit(f"missing seed {seed} at the pick")
        v = r["validation_bits_per_token"]
        tm_bits.append(v)
        ci = P.bootstrap_bpb(np.asarray(r["validation_doc_bits"]), ev.doc_bytes, idx, twin_doc)
        ok = v <= twin - TWO_PCT + 1e-12 and ci["ci95"][1] < 0 and v < floor
        ok_all &= ok
        out.append(f"  seed {seed}: TM {v:.4f} bits per token ({100 * (1 - 2 ** (v - floor)):+.2f}% perplexity vs the floor; "
                   f"{100 * (1 - 2 ** (v - twin)):+.2f}% vs the count twin); TM - twin {ci['point']:+.4f} bits per byte "
                   f"[{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]; b {r['b']} -> {'meets' if ok else 'misses'}")
    strong = ok_all and float(np.mean(tm_bits)) <= trees + 1e-12
    word = "STRONG" if strong else ("PASS" if ok_all else "FAIL")
    out.append(f"words: TM mean {np.mean(tm_bits):.4f} vs trees {trees:.4f} and logistic regression {lr:.4f} -> {word}")
    return out


def main() -> None:
    what = sys.argv[1]
    R = rows()
    if what == "grid":
        for T, s in GRID:
            print(f"grid|{T}|{s}|4000|1")
    elif what == "wide":
        T, s = grid_best(R)
        print(f"wide|{T}|{s}|10000|1")
    elif what == "seeds":
        T, s, clauses = pick(R)
        for seed in (2, 3):
            print(f"seeds|{T}|{s}|{clauses}|{seed}")
    elif what == "words":
        txt = "\n".join(words(R))
        print(txt)
        p = HERE / "g1_words.txt"
        if p.exists():
            import time
            p = HERE / f"g1_words_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.txt"
        p.write_text(txt + "\n")
    else:
        sys.exit(f"unknown: {what}")


if __name__ == "__main__":
    main()
