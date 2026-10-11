"""Export the artificial tests (pcp_world.py) in the format htm_driver.py reads for .npz files:
Xp = paragraph bits, P_train / P_held = index pairs into Xp, Y_train / Y_held, plus vocab_*.json.
'held' is the dev set. For L3 a second file (..._unseen.npz) has the held-out word pairs as 'held'."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcp_world import World  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pcp")


def export(rung, seed, n_train=10000, n_dev=4000):
    w = World(rung, seed)
    sets = [("train", "train", n_train), ("held", "dev", n_dev)]
    files = {"": sets}
    if w.holdout:
        files["_unseen"] = [("train", "train", n_train), ("held", "unseen", n_dev)]
    for suffix, ss in files.items():
        Xp, d, off = [], {}, 0
        for key, name, n in ss:
            X, y, _, _ = w.sample(n, name)
            Xp += [X[:, :w.V], X[:, w.V:]]
            d[f"P_{key}"] = np.c_[off + np.arange(n), off + n + np.arange(n)]
            d[f"Y_{key}"] = y
            off += 2 * n
        stem = f"{rung}_s{seed}{suffix}"
        np.savez_compressed(os.path.join(OUT, f"pairs_{stem}.npz"), Xp=np.vstack(Xp).astype(np.uint8), **d)
        json.dump(w.vocab, open(os.path.join(OUT, f"vocab_{stem}.json"), "w"))
        print(stem, "V", w.V, {k: v.shape for k, v in d.items()})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for rung in ("L1", "L2", "L3"):
        for seed in (1, 2, 3):
            export(rung, seed)
