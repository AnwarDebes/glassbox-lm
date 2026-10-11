"""Read the clean Wikipedia pair data written by build_pairs.py.

data/{version}/{split}.npz, version in {sa, np, ns}, split in {train, dev, test}:
  a_ids, a_off  word ids of paragraph A: one flat uint16 array, row i is a_ids[a_off[i]:a_off[i+1]]
  b_ids, b_off  the same for paragraph B
  y             1 = same article (sa) or next paragraph (np, ns), 0 = two different articles
  art_a, art_b  WikiText-103 train-split article index of A and B (index into its titles.json)
  par_a, par_b  paragraph number inside that article (0 = first qualifying paragraph)
Word ids are ranks in data/vocab_F3072.json; the F-word vocabulary is ids < F (F=1024 is the main one).
For ns, shared words were removed at the 3,072-word level (overlap is 0 at both sizes), but the class
matching and Check 1b hold at F=1024 only, so load_pairs refuses ns at other F.

Use in a training driver:
    from pairdata import load_xy
    X, y = load_xy("sa", "train")          # X = [A bits | B bits], uint32, shape (n, 2*1024)
"""
import json
import os

import numpy as np
from scipy import sparse

DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def ids_to_csr(ids, off, F):
    """0/1 CSR matrix (rows x F) from flat sorted ids and row offsets, keeping ids < F."""
    rows = np.repeat(np.arange(len(off) - 1), np.diff(off))
    keep = ids < F
    return sparse.csr_matrix((np.ones(int(keep.sum()), np.uint8), (rows[keep], ids[keep].astype(np.int64))),
                             shape=(len(off) - 1, F))


def csr_to_ids(m):
    """Inverse of ids_to_csr: (ids uint16, offsets int64) of a 0/1 CSR matrix."""
    m = m.tocsr()
    m.eliminate_zeros()
    m.sort_indices()
    return m.indices.astype(np.uint16), m.indptr.astype(np.int64)


def load_pairs(version, split, F=1024, allow_unchecked=False):
    """Sparse 0/1 matrices A, B (n x F), labels y (int64) and the article / paragraph ids."""
    if version == "ns" and F != 1024 and not allow_unchecked:
        raise ValueError("ns passed Check 1b at F=1024 only; at other F its classes are not matched")
    d = np.load(os.path.join(DATA, version, f"{split}.npz"))
    out = {k: d[k] for k in ("art_a", "art_b", "par_a", "par_b")}
    out["A"] = ids_to_csr(d["a_ids"], d["a_off"], F)
    out["B"] = ids_to_csr(d["b_ids"], d["b_off"], F)
    out["y"] = d["y"].astype(np.int64)
    return out


def load_xy(version, split, F=1024):
    """TM-ready input: X = [A bits | B bits] as dense uint32 (n x 2F), y as uint32."""
    p = load_pairs(version, split, F)
    X = np.hstack([p["A"].toarray(), p["B"].toarray()]).astype(np.uint32)
    return X, p["y"].astype(np.uint32)


def load_vocab(F=1024):
    with open(os.path.join(DATA, f"vocab_F{F}.json")) as f:
        return json.load(f)
