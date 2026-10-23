#!/usr/bin/env python
"""e79 self-test of the C4 mean caches (CPU; train and dev only; no model).

    OMP_NUM_THREADS=1 .venv-dl/bin/python scripts/c4mean_selftest.py | tee logs/c4mean_selftest.out

For STAR, ABCD and MultiWOZ, every split and field, 1,000 random rows (numpy default_rng(0)): the cache's vector equals
an independent float64 recomputation from the raw table file (tokens by dl.count_text.tokens, rows by the word list,
each row normalised, averaged, normalised; zero for a text with no known token), max absolute gap at most 1e-6. Also:
the row counts equal the examples', the test arrays are empty, the provenance names the C4 table file, and every
vector has norm 1 or 0. Added 2026-10-20.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import load_examples  # noqa: E402
from dl.count_text import TABLES, tokens  # noqa: E402


def need(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    sub, sha = TABLES["c4words-S1000-v1"]
    z = np.load(ROOT / "data" / "tables" / sub / "table.npz", allow_pickle=False)
    words = [str(w) for w in z["words"]]
    row = {w: i for i, w in enumerate(words)}
    E = z["codes"].astype(np.float64)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(0)
    for ds in ("star", "abcd", "multiwoz"):
        C = np.load(ROOT / "data" / "emb" / f"{ds}__c4mean.npz")
        prov = json.loads(str(C["provenance"]))
        need(prov["table"]["table_file_sha256"] == sha and prov["dataset"] == ds, f"{ds}: provenance names the C4 table file ({sha[:8]})")
        need(all(C[f"test__{f}"].shape == (0, 0) for f in ("cust", "recent", "full")), f"{ds}: test arrays are empty")
        for split in ("train", "dev"):
            rows, _ = load_examples(split, ds)
            for f in ("cust", "recent", "full"):
                M = C[f"{split}__{f}"]
                need(M.shape == (len(rows), 500), f"{ds} {split} {f}: {M.shape[0]:,} rows, 500 dims")
                nrm = np.linalg.norm(M.astype(np.float64), axis=1)
                need(bool(np.all((np.abs(nrm - 1) < 1e-5) | (nrm == 0))), f"{ds} {split} {f}: every vector has norm 1 or 0 ({int((nrm == 0).sum())} zero)")
                idx = rng.choice(len(rows), size=min(1000, len(rows)), replace=False)
                gap = 0.0
                for i in idx:
                    ids = [row[w] for w in tokens(rows[i][f] or "(nothing yet)") if w in row]
                    v = En[ids].mean(0) if ids else np.zeros(500)
                    n = np.linalg.norm(v)
                    v = v / n if n > 0 else v
                    gap = max(gap, float(np.abs(v - M[i]).max()))
                need(gap <= 1e-6, f"{ds} {split} {f}: 1,000 rows recomputed in float64, max gap {gap:.2e}")
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
