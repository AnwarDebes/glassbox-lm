"""Penn Treebank in the five splits of the design (rule 2).

Units are contiguous blocks of 100 sentences of the standard training file. The blocks are
assigned once, by a seeded permutation: 5% calibration, 5% dev-test, the rest train. The
standard validation file is the tuning split ("valid"). The standard test file is never
opened here; there is no function that reads it.

Every sentence ends with the token <eos>. The vocabulary is the fixed 10,000-word list of the
Mikolov preprocessing (every word of the standard training file), sorted, so word ids do not
depend on the split.
"""
import hashlib
import os

import numpy as np

STUDIES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW = os.path.join(STUDIES, "tm_language_model", "data", "ptb", "raw")
FILES = {"train": "train", "valid": "validation"}
BLOCK = 100
SEED = 20261015
SHARES = {"calib": 0.05, "devtest": 0.05}
SPLITS = ("train", "calib", "valid", "devtest")


def path(name):
    return os.path.join(RAW, FILES[name], "text.utf8")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sentences(name):
    with open(path(name), encoding="utf8") as f:
        return [line.split() + ["<eos>"] for line in f]


def block_assignment(n_sentences):
    """Split name per block of the standard training file."""
    n_blocks = (n_sentences + BLOCK - 1) // BLOCK
    order = np.random.default_rng(SEED).permutation(n_blocks)
    n_cal = int(round(SHARES["calib"] * n_blocks))
    n_dev = int(round(SHARES["devtest"] * n_blocks))
    names = np.array(["train"] * n_blocks, dtype=object)
    names[order[:n_cal]] = "calib"
    names[order[n_cal:n_cal + n_dev]] = "devtest"
    return names


def load():
    """Returns (vocab, w2i, ids, blocks).

    ids[split] is one int32 token stream per split (blocks in file order, joined).
    blocks[split] is an int32 array, the same length, giving the block number of each token
    (for the block bootstrap and for cross-fitting by block).
    """
    train_sents = _sentences("train")
    vocab = sorted({w for s in train_sents for w in s})
    w2i = {w: i for i, w in enumerate(vocab)}
    assign = block_assignment(len(train_sents))
    toks = {s: [] for s in SPLITS}
    blks = {s: [] for s in SPLITS}
    for i, sent in enumerate(train_sents):
        b = i // BLOCK
        toks[assign[b]].extend(w2i[w] for w in sent)
        blks[assign[b]].extend([b] * len(sent))
    for i, sent in enumerate(_sentences("valid")):
        toks["valid"].extend(w2i[w] for w in sent)
        blks["valid"].extend([i // BLOCK] * len(sent))
    ids = {s: np.array(toks[s], dtype=np.int32) for s in SPLITS}
    blocks = {s: np.array(blks[s], dtype=np.int32) for s in SPLITS}
    return vocab, w2i, ids, blocks


def source_hashes():
    return {name: sha256(path(name)) for name in FILES}


if __name__ == "__main__":
    vocab, w2i, ids, blocks = load()
    print("V", len(vocab))
    for s in SPLITS:
        print(s, len(ids[s]), "tokens", len(np.unique(blocks[s])), "blocks")
    print(source_hashes())
