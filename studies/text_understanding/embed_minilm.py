#!/usr/bin/env python
"""all-MiniLM-L6-v2 sentence embeddings for a dataset the decision-layer cache does not have (test 3: massive_en).

    embed_minilm.py --dataset massive_en --out-dir DIR

The procedure of the study's scripts/embed.py (L2-normalised, batch 128, no prefix), run here so that the
study's data folder is not written. Output: DIR/{dataset}__minilm.npz (train / val / test, y_*), the same
keys as the study's caches, plus the model snapshot that was used.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.data import load  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--out-dir", type=Path, required=True)
args = ap.parse_args()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
model.eval()
ds = load(args.dataset)
arrays = {}
for split, texts in ds.texts.items():
    with torch.inference_mode():
        arrays[split] = model.encode(texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=False).astype(np.float32)
for split, y in ds.labels.items():
    arrays[f"y_{split}"] = y
snap = sorted(Path.home().glob(".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))
path = args.out_dir / f"{args.dataset}__minilm.npz"
np.savez(path, **arrays)
print(json.dumps({"path": str(path), "snapshots": [p.name for p in snap], **{k: list(v.shape) for k, v in arrays.items()}}))
