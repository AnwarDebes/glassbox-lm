#!/usr/bin/env python
"""Static-table caches for the next-action records: data/emb/<dataset>__potion-32m.npz (CPU, no GPU).

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv-dl/bin/python scripts/static_prepare.py --datasets abcd star

The same keys and layout as the bge-large caches ({train,dev,test}__{cust,recent,full}, float32,
L2-normalised, in record order, "(nothing yet)" for an empty text), so that ABCDView's intent and step
parts read it unchanged (view A of the no-network plan). Each vector is the normalised average of the
text's token vectors in potion-base-32M (src/dl/static_text.py): lookups, an average and a division,
no network. Records come from dl.abcd.load_examples, so the rows align with every script by
construction. Why not abcd_prepare.py: it stores int(model.max_seq_length), which is inf for a static
model (sentence_transformers StaticEmbedding), so it would crash after encoding; star_prepare.py rebuilds
its records from the raw files instead of reading them. Writes atomically and skips a file that exists.
Provenance in the file: revision, file and table SHA-256s, the empty-text rule, token lengths, the
example files' SHA-256s, library versions, and the largest gap to SentenceTransformer.encode on every
unique text (must be at most 1e-6). Added 2026-10-17 for the no-network agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import load_examples  # noqa: E402
from dl.static_text import EMPTY, FILE_SHA256, HF_ID, REV, one_thread, table  # noqa: E402

FIELDS = ("cust", "recent", "full")
SPLITS = ("train", "dev", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["abcd", "star"])
    args = ap.parse_args()
    import sentence_transformers
    import tokenizers
    import torch

    tb = table()
    for ds in args.datasets:
        path = ROOT / "data" / "emb" / f"{ds}__potion-32m.npz"
        if path.exists():
            print(f"{path.name} exists, skipped", flush=True)
            continue
        t0 = time.time()
        rows = {s: load_examples(s, ds)[0] for s in SPLITS}
        arrays, lens, gap = {}, {}, 0.0
        for fld in FIELDS:
            uniq = sorted({r[fld] or EMPTY for s in SPLITS for r in rows[s]})
            with one_thread():
                M = tb.mean(uniq)
                ref = tb.st.encode(uniq, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            gap = max(gap, float(np.abs(M - ref).max()))
            pos = {t: i for i, t in enumerate(uniq)}
            for s in SPLITS:
                arrays[f"{s}__{fld}"] = M[[pos[r[fld] or EMPTY] for r in rows[s]]].astype(np.float32)
            n_tok = np.array([len(x) for x in tb.ids([r[fld] or EMPTY for r in rows["train"]])])
            lens[fld] = {"train_median": float(np.median(n_tok)), "train_p90": float(np.percentile(n_tok, 90)), "train_max": int(n_tok.max())}
        if gap > 1e-6:
            raise SystemExit(f"{ds}: mean differs from SentenceTransformer.encode by {gap}")
        ex_dir = ROOT / "data" / ds
        prov = {"encoder": "potion-32m", "hf_id": HF_ID, "revision": REV, "file_sha256": FILE_SHA256, "table_sha256": tb.table_sha256,
                "empty_text": EMPTY, "fields": FIELDS, "normalized": True, "max_abs_gap_to_sentence_transformers": gap,
                "token_lengths": lens, "examples_sha256": {s: hashlib.sha256((ex_dir / f"examples_{s}.json.gz").read_bytes()).hexdigest() for s in SPLITS},
                "versions": {"sentence_transformers": sentence_transformers.__version__, "tokenizers": tokenizers.__version__,
                             "torch": torch.__version__, "numpy": np.__version__},
                "encode_seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")
        np.savez(tmp, provenance=json.dumps(prov), **arrays)
        os.replace(tmp, path)
        print(json.dumps({"file": path.name, **{k: v for k, v in prov.items() if k not in ("versions", "file_sha256")},
                          "shapes": {k: list(v.shape) for k, v in arrays.items()},
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
