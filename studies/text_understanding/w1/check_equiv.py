#!/usr/bin/env python
"""W1 harness-equivalence check (train and val features only; no TM is trained, no accuracy is computed).

    check_equiv.py --dataset clinc150 --new w1/feats/potion_check --ref emb

Rule (the GloVe arm's, LOG.md 18:51:16Z): the check passes when the 512 concepts are the same (names and member
counts) and every existsT bit that the TM reads after Thermo(8) is identical on train and val. Float differences
of the cosines are reported (GPU float32 rounding may move them by about 1e-7). One row in results_w1_checks.jsonl.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X  # noqa: E402


def bits(dataset: str, d: Path):
    F, meta = X.features(dataset, 512, d, "cuda")
    for sp in X.SPLITS:
        if f"{sp}__exists" in F:
            F[f"{sp}__existsT"] = F[f"{sp}__exists"]
    view = X.View("existsT", meta).fit(F)
    return F, meta, {sp: view.transform(F, sp) for sp in ("train", "val")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    args = ap.parse_args()
    Fa, ma, Ba = bits(args.dataset, args.new)
    Fb, mb, Bb = bits(args.dataset, args.ref)
    same_concepts = ma["concepts"] == mb["concepts"] and ma["members"] == mb["members"]
    diff_bits = {sp: int((Ba[sp] != Bb[sp]).sum()) for sp in Ba}
    n_bits = {sp: int(Ba[sp].size) for sp in Ba}
    fl = {}
    for k in sorted(set(Fa) & set(Fb)):
        a, b = np.asarray(Fa[k]), np.asarray(Fb[k])
        if a.dtype.kind == "f" and a.shape == b.shape:
            fl[k] = float(np.abs(a - b).max()) if a.size else 0.0
    rec = {"check": "harness equivalence (potion through w1_feats.py vs test 2's cache)", "dataset": args.dataset,
           "new": str(args.new), "ref": str(args.ref), "same_concepts_and_members": same_concepts,
           "existsT_bits_differing": diff_bits, "existsT_bits_total": n_bits, "max_abs_float_diff": fl,
           "passed": bool(same_concepts and all(v == 0 for v in diff_bits.values()))}
    print(json.dumps(rec))
    with open(HERE / "results_w1_checks.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
