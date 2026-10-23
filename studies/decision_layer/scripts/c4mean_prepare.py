#!/usr/bin/env python
"""e79 (plan logs/plan_e79_c4_prototypes.md): C4 mean-vector caches data/emb/<ds>__c4mean.npz (CPU; no neural network).

    OMP_NUM_THREADS=1 .venv-dl/bin/python scripts/c4mean_prepare.py --datasets star abcd multiwoz

The layout of the encoder caches ({train,dev,test}__{cust,recent,full}, float32, rows in record order), so that
ABCDView's intent and step parts read it unchanged. Each vector is the L2-normalised average of the L2-normalised C4
word vectors (dl.count_text's table c4words-S1000-v1) of the text's known tokens, "(nothing yet)" for an empty text, as
in the other caches. A text with no known token gets a zero row (counted in the provenance). These are lookups, an
average and a division. Train and dev only: the test arrays are empty, (0, 0), so no test text is read. Provenance in
the file: the table's meta, the examples' SHA-256s, the counts. Refuses to overwrite. Added 2026-10-20.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import load_examples  # noqa: E402
from dl.count_text import count_table  # noqa: E402

TABLE = "c4words-S1000-v1"
FIELDS = ("cust", "recent", "full")


def means(tb, texts: list[str]) -> tuple[np.ndarray, int]:
    ids = tb.ids(texts)
    M = np.zeros((len(ids), tb.En.shape[1]), np.float32)
    n_zero = 0
    for i, t in enumerate(ids):
        if not t:
            n_zero += 1
            continue
        v = tb.En[t].mean(0)
        M[i] = v / max(float(np.linalg.norm(v)), 1e-12)
    return M, n_zero


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()
    tb = count_table(TABLE)
    for ds in args.datasets:
        path = ROOT / "data" / "emb" / f"{ds}__c4mean.npz"
        if path.exists():
            raise SystemExit(f"{path} exists; not overwriting")
        t0 = time.time()
        arrays, prov = {}, {"what": "C4 mean vectors (e79)", "table": tb.meta(), "dataset": ds, "empty": "(nothing yet)",
                            "splits": ["train", "dev"], "examples_sha256": {}, "zero_rows": {}}
        for split in ("train", "dev"):
            rows, _ = load_examples(split, ds)
            prov["examples_sha256"][split] = hashlib.sha256((ROOT / "data" / ds / f"examples_{split}.json.gz").read_bytes()).hexdigest()
            for f in FIELDS:
                arrays[f"{split}__{f}"], z = means(tb, [r[f] for r in rows])
                prov["zero_rows"][f"{split}__{f}"] = z
        for f in FIELDS:
            arrays[f"test__{f}"] = np.zeros((0, 0), np.float32)
        np.savez(path, provenance=json.dumps(prov), **arrays)
        print(ds, {k: v.shape for k, v in arrays.items()}, "zero rows", prov["zero_rows"], f"{time.time() - t0:.0f}s",
              hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
