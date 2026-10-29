#!/usr/bin/env python
"""Text understanding, step 2d's first measurement (DESCRIPTIVE, intent val; asked for at 22:04Z; logged in
LOG.md before the run): do the existsT OR bits saturate on longer texts? CPU only, one thread, no probe, no test split.

For CLINC150, Banking77 and HWU64 and both views (the C4 table ppw_S1000's existsT and potion's existsT, the caches the
probes read): per validation text, its length in words (glove_feats.glove_tokens) and its on-share (the share of its
existsT bits that are on, after the same Thermo(8) fit on train as every probe), plus the share of concepts at their
TOP thermometer level. Reported per length bin (1-5, 6-10, 11-15, 16-20, 21-30, 31-50, 51+ words): count, mean
on-share, mean top-level share; and the Spearman correlation of length with on-share. Writes tu/d2d_saturation.json and
.txt (refuses to overwrite).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EMB = HERE.parent
sys.path.insert(0, str(EMB))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import exists_tm as X  # noqa: E402
from glove_feats import glove_tokens  # noqa: E402
from dl.data import load  # noqa: E402

SETS = ("clinc150", "banking77", "hwu64")
DIRS = {"potion": EMB / "emb_potion_test2", "ppw_S1000": EMB / "w1" / "feats" / "ppw_S1000_d500_e0.5"}
BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 10**9)]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    t0 = time.time()
    oj, ot = HERE / "d2d_saturation.json", HERE / "d2d_saturation.txt"
    assert not oj.exists() and not ot.exists()
    res, lines = {}, []
    for ds in SETS:
        texts = load(ds).texts["val"]
        n_words = np.array([len(glove_tokens(t)) for t in texts])
        res[ds] = {"n": int(len(texts)), "max_words": int(n_words.max()), "median_words": float(np.median(n_words))}
        for name, d in DIRS.items():
            F, meta = X.features(ds, 512, d, "cpu")
            for sp in X.SPLITS:
                if f"{sp}__exists" in F:
                    F[f"{sp}__existsT"] = F[f"{sp}__exists"]
            v = X.View("existsT", meta).fit(F)
            B = v.transform(F, "val").astype(np.float32)
            on = B.mean(1)
            # top level per concept: the last threshold of each concept's thermometer
            th = v.th.th
            sims = F["val__existsT"]
            last = np.array([float(t[-1]) if len(t) else np.inf for t in th])
            top = (sims >= last[None, :]).mean(1)
            rows = []
            for lo, hi in BINS:
                m = (n_words >= lo) & (n_words <= hi)
                if m.sum() == 0:
                    continue
                rows.append({"bin": f"{lo}-{hi if hi < 10**9 else '+'}", "count": int(m.sum()),
                             "on_share": round(float(on[m].mean()), 4), "top_share": round(float(top[m].mean()), 4)})
            res[ds][name] = {"bins": rows, "spearman_length_on": round(spearman(n_words, on), 3),
                             "on_share_all": round(float(on.mean()), 4), "n_bits": int(B.shape[1])}
            lines.append(f"{ds} {name}: bits {B.shape[1]}, on-share {on.mean():.3f}, Spearman(length, on-share) "
                         f"{res[ds][name]['spearman_length_on']:+.3f}; by length: " +
                         "; ".join(f"{r['bin']} n {r['count']} on {r['on_share']:.3f} top {r['top_share']:.3f}" for r in rows))
    res["seconds"] = round(time.time() - t0, 1)
    oj.write_text(json.dumps(res, indent=1))
    ot.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
