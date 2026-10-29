"""PLAN_logsize_dev.md: the nested pools of training dialogues (standard library only).

usage: python3.12 -I logsize_subsets.py [--out runs/logsize/subsets.json]
Each dataset's training dialogues (convo_id) are sorted and shuffled once with random.Random(20261028); the first N
are the pool of size N, so the pools are nested. Writes the order, the train file's sha256 and the rows per N. Refuses
to overwrite an existing file.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import gzip
import hashlib
import json
import os
import random
import sys

H = os.path.dirname(os.path.abspath(__file__))
DATA = f"{_REPO_ROOT}/studies/decision_layer/data"
DATASETS = ("star", "abcd", "multiwoz")
SIZES = (64, 256, 1024, 4096)
SEED = 20261028


def pools(rows, seed=SEED):
    """The shuffled order of the dialogues of ``rows`` (sorted first, so the file's row order does not matter)."""
    order = sorted({r["convo_id"] for r in rows})
    random.Random(seed).shuffle(order)
    return order


def keep_ids(subsets, dataset, n):
    """The convo_ids of the pool of size n ("full" or "whole": None, meaning every training row)."""
    if str(n) in ("full", "whole"):
        return None
    order = subsets[dataset]["order"]
    n = int(n)
    if not 0 < n <= len(order):
        raise ValueError(f"pool size {n} outside 1..{len(order)}")
    return set(order[:n])


def build(data=DATA, datasets=DATASETS, sizes=SIZES, seed=SEED):
    out = {"plan": "PLAN_logsize_dev.md", "seed": seed, "sizes": list(sizes)}
    for ds in datasets:
        path = os.path.join(data, ds, "examples_train.json.gz")
        with open(path, "rb") as h:
            raw = h.read()
        rows = json.loads(gzip.decompress(raw))["rows"]
        order = pools(rows, seed)
        rows_per = {}
        for n in sizes:
            keep = set(order[:n])
            rows_per[str(n)] = sum(r["convo_id"] in keep for r in rows)
        rows_per["whole"] = len(rows)
        out[ds] = {"train_file": path, "train_sha256": hashlib.sha256(raw).hexdigest(), "n_dialogues": len(order),
                   "n_rows": len(rows), "rows_per_size": rows_per, "order": order}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(H, "runs/logsize/subsets.json"))
    a = ap.parse_args(argv)
    d = build()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "x") as h:
        json.dump(d, h)
    for ds in DATASETS:
        print(ds, d[ds]["n_dialogues"], "dialogues", json.dumps(d[ds]["rows_per_size"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
