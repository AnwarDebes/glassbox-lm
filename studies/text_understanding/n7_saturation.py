#!/usr/bin/env python
"""N7 smoke diagnostic (train split only, test files never opened): does the max over tokens saturate on long texts?

    n7_saturation.py --dataset trec|mr|r52|r8|sst2|clinc150 --table potion|glove

The share of existsT bits that are on is about one half by construction (the thermometer thresholds are quantiles
of the training texts), so it cannot show saturation. This script measures it directly: the max cosine to each of
the 512 concepts over a text's tokens (the existsT similarities, same code path as n7_tm.exists_bits), and per
concept the spread of that value over the training texts (interquartile range, and the 10th-90th percentile range).
Saturated concepts are those every text matches about equally well. clinc150 (an intent set, short texts) is the
reference. Prints one JSON line; writes nothing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exists_tm as X  # noqa: E402
import glove_feats as GF  # noqa: E402
import n7_tm as N  # noqa: E402
from dl.boolean import _spherical_kmeans  # noqa: E402
from dl.data import load as dl_load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=N.DATASETS + ("clinc150",))
ap.add_argument("--table", required=True, choices=("potion", "glove"))
args = ap.parse_args()
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(0.1)
if args.dataset == "clinc150":
    texts = list(dl_load("clinc150").texts["train"])
else:
    texts = N.load(args.dataset, False)[0]["train"][0]
if args.table == "potion":
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(X.HF_ID, device="cpu", revision=X.REV)
    E = st[0].embedding.weight.detach().float().to(dev)
    ids = [e.ids for e in st[0].tokenizer.encode_batch(texts, add_special_tokens=False)]
else:
    M, words = GF.load_glove()
    row = {w: i for i, w in enumerate(words)}
    toks = [GF.glove_tokens(t) for t in texts]
    need = sorted({row[w] for x in toks for w in x if w in row})
    local = {g: j for j, g in enumerate(need)}
    E = torch.as_tensor(M[need], device=dev)
    ids = [[local[row[w]] for w in x if w in row] for x in toks]
En = torch.nn.functional.normalize(E, dim=1)
vocab = sorted({t for x in ids for t in x})
gen = torch.Generator(device=dev)
gen.manual_seed(0)
Cn = _spherical_kmeans(En[torch.as_tensor(vocab, device=dev)], 512, 25, gen)
S = X.pooled(ids, E, En, Cn, dev, bs=64 if args.dataset in ("r52", "r8") else 1024)["exists"]
q10, q25, q50, q75, q90 = np.percentile(S, [10, 25, 50, 75, 90], axis=0)
iqr, r1090 = q75 - q25, q90 - q10
print(json.dumps({"dataset": args.dataset, "table": args.table, "train_texts": len(texts),
                  "tokens_per_text_median": float(np.median([len(x) for x in ids])),
                  "median_concept_max_cos": round(float(np.median(q50)), 4),
                  "median_concept_iqr": round(float(np.median(iqr)), 4),
                  "median_concept_p10_p90": round(float(np.median(r1090)), 4),
                  "share_concepts_iqr_below_0.02": round(float((iqr < 0.02).mean()), 4)}))
