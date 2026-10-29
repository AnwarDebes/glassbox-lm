#!/usr/bin/env python
"""W1 references: potion-base-32M's own rows as a table file in mo_tm.py's format (a REFERENCE, transformer-distilled).

    make_potion_table.py                                   # tables/ref_potion_full.npz (all 63,091 rows)
    make_potion_table.py --cover tables/NAME.npz           # tables/ref_potion_cover_NAME.npz: only the tokens that
                                                           # have a vector in that table (the coverage-matched row)

The full file lets the harness-equivalence check run through w1_feats.py's table path (its own tokenizer file), and
the cover file gives the coverage-matched potion row of the controls ladder.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REV = "1e5a03f8eeb2c98b928fbbd846f22f816360919f"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", default=None)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("minishlab/potion-base-32M", device="cpu", revision=REV)
    full = st[0].embedding.weight.detach().float().numpy()
    vocab_ids = np.arange(full.shape[0], dtype=np.int32)
    codes = full
    name = "ref_potion_full"
    if args.cover:
        z = np.load(HERE / args.cover)
        has = np.abs(z["codes"].astype(np.float32)).sum(1) > 0
        vocab_ids = np.sort(z["vocab_ids"][has]).astype(np.int32)
        codes = full[vocab_ids]
        name = f"ref_potion_cover_{Path(args.cover).stem}"
    p = HERE / "tables" / f"{name}.npz"
    p.parent.mkdir(exist_ok=True)
    assert not p.exists(), f"{p} exists"
    np.savez(p, vocab_ids=vocab_ids, codes=codes.astype(np.float32), counts=np.zeros(len(vocab_ids), np.int64))
    print(p, len(vocab_ids), hashlib.sha256(p.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
