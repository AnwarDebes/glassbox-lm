"""PTB word-level data (README "Data"). One sentence per line; every sentence ends with <eos>;
the splits are read as one continuous token stream each (the standard PTB language-model setup)."""
import os
import numpy as np

PTB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tm_language_model", "data", "ptb", "raw")
SPLITS = {"train": "train", "valid": "validation", "test": "test"}


def path(split):
    return os.path.join(PTB, SPLITS[split], "text.utf8")


def tokens(split):
    out = []
    with open(path(split), encoding="utf8") as f:
        for line in f:
            out.extend(line.split())
            out.append("<eos>")
    return out


def load(splits=("train", "valid")):
    """Returns (vocab list, word->id dict, {split: int32 id array}). The vocabulary is built on
    train only (PTB is closed-vocabulary, so every valid/test word is in it)."""
    if "test" in splits:
        raise SystemExit("ptb.load: the test split is read only by final_test.py (README rule 5)")
    toks = {s: tokens(s) for s in splits}
    vocab = sorted(set(toks["train"]))
    w2i = {w: i for i, w in enumerate(vocab)}
    ids = {}
    for s, t in toks.items():
        missing = [w for w in set(t) if w not in w2i]
        assert not missing, f"{s} has out-of-vocabulary words: {missing[:5]}"
        ids[s] = np.array([w2i[w] for w in t], dtype=np.int32)
    return vocab, w2i, ids
