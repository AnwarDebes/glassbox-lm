#!/usr/bin/env python
"""e77 (plan logs/plan_e77_multiwoz_dev.md): the bge-large cache for MultiWOZ 2.2, as star_prepare.py builds STAR's.

    multiwoz_embed.py              # dev stage: train and dev only
    multiwoz_embed.py --add-test   # R24 only, after its freeze and multiwoz_prepare.py --with-test

bge-large-en-v1.5 (scripts/embed.py's ENCODERS entry), L2-normalised, on the GPU; per split the fields cust, recent and
full, with "(nothing yet)" for an empty text, in the examples' row order (star_prepare.py's loop). Writes
data/emb/multiwoz__bge-large.npz and refuses to overwrite it.
--add-test embeds only the test texts. The train and dev arrays are copied from the dev-stage file, which is kept as
data/emb/multiwoz__bge-large.devstage.npz, and checked array by array against it; so the agents tuned on dev see the
same vectors at R24. Added 2026-10-19.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
EMB = ROOT / "data" / "emb"
FIELDS = ("cust", "recent", "full")


def examples(split: str) -> tuple[list[dict], str]:
    p = ROOT / "data" / "multiwoz" / f"examples_{split}.json.gz"
    with gzip.open(p, "rt") as f:
        return json.load(f)["rows"], hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-test", action="store_true")
    args = ap.parse_args()
    path, dev_stage = EMB / "multiwoz__bge-large.npz", EMB / "multiwoz__bge-large.devstage.npz"
    old, prov = {}, {}
    if args.add_test:
        if dev_stage.exists() or not path.exists():
            raise SystemExit(f"--add-test needs {path.name} (the dev stage) and no {dev_stage.name}")
        E = np.load(path)
        prov = json.loads(str(E["provenance"]))
        old = {k: E[k] for k in E.files if k != "provenance"}
        if set(old) != {f"{s}__{f}" for s in ("train", "dev") for f in FIELDS} or prov["splits"] != ["train", "dev"]:
            raise SystemExit(f"{path.name} is not the dev-stage file")
        for s in ("train", "dev"):
            if examples(s)[1] != prov["examples_sha256"][s]:
                raise SystemExit(f"examples_{s} changed since the dev stage")
        splits = ["test"]
    else:
        if path.exists():
            raise SystemExit(f"{path} exists; not overwriting")
        splits = ["train", "dev"]
    rows, shas = {}, dict(prov.get("examples_sha256", {}))
    for s in splits:
        rows[s], shas[s] = examples(s)
    import torch
    from sentence_transformers import SentenceTransformer

    from embed import ENCODERS

    hf_id, prefix, kw = ENCODERS["bge-large"]
    model = SentenceTransformer(hf_id, device="cuda", **kw)
    arrays, t0 = dict(old), time.time()
    for s in splits:
        for fld in FIELDS:
            texts = [r[fld] or "(nothing yet)" for r in rows[s]]
            uniq = sorted(set(texts))
            pos = {t: i for i, t in enumerate(uniq)}
            with torch.inference_mode():
                E = model.encode([prefix + t for t in uniq], batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)
            arrays[f"{s}__{fld}"] = E[[pos[t] for t in texts]].astype(np.float32)
            print("bge-large", s, fld, len(texts), "texts,", len(uniq), "unique", f"{time.time() - t0:.0f}s", flush=True)
    provenance = json.dumps({"encoder": "bge-large", "hf_id": hf_id, "dataset": "multiwoz", "splits": sorted(shas, key=["train", "dev", "test"].index),
                             "examples_sha256": shas, **({"train_dev_from": dev_stage.name} if args.add_test else {})})
    if args.add_test:
        tmp = EMB / "multiwoz__bge-large.new.npz"
        np.savez(tmp, provenance=provenance, **arrays)
        path.rename(dev_stage)
        tmp.rename(path)
        A, B = np.load(path), np.load(dev_stage)
        for k in B.files:
            if k != "provenance" and A[k].tobytes() != B[k].tobytes():
                raise SystemExit(f"{k} differs from the dev stage")
        print("train and dev arrays byte-identical to", dev_stage.name, hashlib.sha256(dev_stage.read_bytes()).hexdigest())
    else:
        np.savez(path, provenance=provenance, **arrays)
    print("wrote", path.relative_to(ROOT), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
