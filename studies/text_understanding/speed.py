#!/usr/bin/env python
"""Rough CPU encode speed, one thread: CLINC150 val (3,000 queries), static models against transformers.
Rough because the machine is shared (6-core quota, other jobs running): read ratios, not absolute times."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.data import load
from sentence_transformers import SentenceTransformer
torch.set_num_threads(1)
texts = load("clinc150").texts["val"]
models = {"potion-8m": "minishlab/potion-base-8M", "potion-32m": "minishlab/potion-base-32M",
          "static-retr": "sentence-transformers/static-retrieval-mrl-en-v1",
          "minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge-base": "BAAI/bge-base-en-v1.5"}
out = {}
for k, hf in models.items():
    m = SentenceTransformer(hf, device="cpu")
    m.encode(texts[:64], batch_size=64)
    best = 1e9
    for _ in range(2):
        t = time.perf_counter(); m.encode(texts, batch_size=64, show_progress_bar=False); best = min(best, time.perf_counter() - t)
    out[k] = {"seconds": round(best, 3), "queries_per_s": round(len(texts) / best, 1)}
    print(k, out[k], flush=True)
Path(__file__).with_name("speed.json").write_text(json.dumps(out, indent=1))
