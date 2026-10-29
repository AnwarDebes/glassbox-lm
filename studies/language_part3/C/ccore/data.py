"""PTB in the tally splits (tally src/tally/ptb.py, read only): train 836,253, calib 46,925,
valid 73,760 and dev-test 46,411 tokens; <eos> ends every sentence. The standard test file is not
read here: no function of this package opens it (the test read of step 10 gets its own frozen script).

Dry runs never touch calib, valid or dev-test: dry_splits() rebuilds four small pseudo splits from
whole blocks of the train split only.
"""
import numpy as np

from . import paths

SPLITS = ("train", "calib", "valid", "devtest")
DRY_BLOCKS = {"train": 12, "calib": 3, "valid": 3, "devtest": 3}
DRY_SEED = 7


def load_ptb():
    paths.setup()
    from tally import ptb
    vocab, w2i, ids, blocks = ptb.load()
    eos = w2i["<eos>"]
    for s in SPLITS:
        assert ids[s][-1] == eos, s
    return {"vocab": vocab, "w2i": w2i, "ids": ids, "blocks": blocks, "V": len(vocab), "eos": eos,
            "source_hashes": ptb.source_hashes()}


def dry_splits(P, sizes=DRY_BLOCKS, seed=DRY_SEED):
    """Pseudo splits from whole train blocks (file order kept inside each split)."""
    ids, blocks = P["ids"]["train"], P["blocks"]["train"]
    ub = np.unique(blocks)
    perm = np.random.default_rng(seed).permutation(len(ub))
    out_ids, out_blocks, start = {}, {}, 0
    for s in SPLITS:
        chosen = np.sort(ub[perm[start:start + sizes[s]]])
        start += sizes[s]
        keep = np.isin(blocks, chosen)
        out_ids[s], out_blocks[s] = ids[keep], blocks[keep]
        assert out_ids[s][-1] == P["eos"]
    Q = dict(P)
    Q["ids"], Q["blocks"] = out_ids, out_blocks
    Q["dry"] = {"seed": seed, "blocks": dict(sizes), "from": "train split only"}
    return Q
