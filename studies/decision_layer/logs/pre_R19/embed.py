#!/usr/bin/env python
"""Frozen sentence embeddings for every split of a dataset, cached as .npz.

    embed.py --datasets clinc150 banking77 hwu64 --encoders minilm bge-large ...

Output: data/emb/<dataset>__<encoder>.npz with one float32 array per split (keys = split
names), labels per in-scope split (y_<split>), the intent names and a provenance record.
Embeddings are L2-normalised (cosine geometry) for every encoder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.data import load  # noqa: E402

ENCODERS = {
    # short name: (HF id, text prefix, extra kwargs for SentenceTransformer)
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", {}),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "", {}),
    "bge-base": ("BAAI/bge-base-en-v1.5", "", {}),
    "bge-large": ("BAAI/bge-large-en-v1.5", "", {}),
    "e5-large": ("intfloat/e5-large-v2", "query: ", {}),
    "mxbai-large": ("mixedbread-ai/mxbai-embed-large-v1", "", {}),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "", {}),
    # multilingual (phase 2, MASSIVE en-US / nb-NO)
    "bge-m3": ("BAAI/bge-m3", "", {}),
    "me5-large": ("intfloat/multilingual-e5-large", "query: ", {}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["clinc150", "banking77", "hwu64"])
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    out_dir = ROOT / "data" / "emb"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {name: load(name) for name in args.datasets}
    for enc in args.encoders:
        hf_id, prefix, kw = ENCODERS[enc]
        t0 = time.time()
        model = SentenceTransformer(hf_id, device="cuda", **kw)
        model.eval()
        load_s = time.time() - t0
        for name, ds in data.items():
            path = out_dir / f"{name}__{enc}.npz"
            if path.exists():
                print("exists", path)
                continue
            arrays: dict[str, np.ndarray] = {}
            t1 = time.time()
            for split, texts in ds.texts.items():
                with torch.inference_mode():
                    e = model.encode(
                        [prefix + t for t in texts],
                        batch_size=args.batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                arrays[split] = e.astype(np.float32)
            for split, y in ds.labels.items():
                arrays[f"y_{split}"] = y
            prov = {
                "dataset": name,
                "encoder": enc,
                "hf_id": hf_id,
                "prefix": prefix,
                "dim": int(arrays["train"].shape[1]),
                "normalized": True,
                "sources": ds.sources,
                "summary": ds.summary(),
                "encode_seconds": round(time.time() - t1, 1),
                "model_load_seconds": round(load_s, 1),
                "torch": torch.__version__,
            }
            np.savez(path, intents=np.array(ds.intents), provenance=json.dumps(prov), **arrays)
            print(json.dumps(prov))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
