"""The dialogue design effect of a paired accuracy difference at 2,000 rows, measured on full dev (REG_agent_vs_llm).

usage: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python -I design_effect_dev.py > design_effect_dev.out
For each dataset: the agent (seed 1) minus fine-tuned bge-base, per dev row; 5 random 2,000-row draws (seeds 1000-1004);
for each, the bootstrap standard deviation over dialogues (2,000 draws) divided by the iid standard error.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import gzip
import json
import random

import numpy as np

D = f"{_REPO_ROOT}/studies/decision_layer/data"
PAIRS = {"star": ("runs/agent_star_c4_tm_s1/tm.npz", "runs/ft_star_bgebase_s1/dev_preds.npz"),
         "abcd": ("runs/agent_abcd_c4_tm_s1/tm.npz", "runs/ft_abcd_bgebase_s1/dev_preds.npz"),
         "multiwoz": ("runs/agent_multiwoz_c4_tm_s1/tm.npz", "runs/ft_mwoz_bgebase_s1/dev_preds.npz")}
for ds, (pa, pf) in PAIRS.items():
    rows = json.load(gzip.open(f"{D}/{ds}/examples_dev.json.gz", "rt"))["rows"]
    A, F = np.load(pa), np.load(pf)
    g = np.array([r["y"] for r in rows])
    assert (A["gold"] == g).all() and (F["gold"] == g).all()
    conv = np.array([r["convo_id"] for r in rows])
    d = (A["top_ids"][:, 0] == g).astype(float) - (F["top_ids"][:, 0] == g).astype(float)
    ratios, dials = [], []
    for rep in range(5):
        idx = np.array(sorted(random.Random(1000 + rep).sample(range(len(rows)), 2000)))
        dd, cc = d[idx], conv[idx]
        ids = np.unique(cc)
        groups = [np.where(cc == c)[0] for c in ids]
        rng = np.random.default_rng(rep)
        bs = np.array([dd[np.concatenate([groups[j] for j in rng.integers(0, len(ids), len(ids))])].mean()
                       for _ in range(2000)])
        ratios.append(round(float(bs.std() / (dd.std(ddof=1) / np.sqrt(len(dd)))), 3))
        dials.append(len(ids))
    print(json.dumps({"dataset": ds, "n_dev": len(rows), "dialogues": int(len(np.unique(conv))),
                      "se_ratio_cluster_over_iid": ratios, "dialogues_drawn": dials, "sd_row_difference": round(float(d.std()), 3)}))
