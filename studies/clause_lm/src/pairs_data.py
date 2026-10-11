"""PTB adjacent word pairs for the pair TM (M1b): is word w a real next word after word u?

Both words must be in the top-K train words. Positives are adjacent pairs from the text, and
negatives keep u while w is drawn from the unigram^0.75 distribution over the top-K words
(noise-contrastive). Validation is labelled by whether the pair (u, w) occurs anywhere in
train ("seen"). The generalisation test uses UNSEEN positives against UNSEEN negatives, where a
pure count model has no information (AUC 0.5 by construction).
Output: data/ptb_pairs_K{K}.npz with int16 index arrays and labels, plus vocab_K{K}.json.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptb
from runlog import Run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make(K=1000, n_train_pos=0, seed=0):
    vocab, w2i, ids = ptb.load(("train", "valid"))
    V = len(vocab)
    tr, va = ids["train"].astype(np.int64), ids["valid"].astype(np.int64)
    counts = np.bincount(tr, minlength=V)
    top = np.argsort(-counts, kind="stable")[:K]
    remap = np.full(V, -1, np.int64); remap[top] = np.arange(K)
    q = counts[top].astype(np.float64) ** 0.75; q /= q.sum()
    rng = np.random.default_rng(seed)

    def pairs(stream):
        u, w = remap[stream[:-1]], remap[stream[1:]]
        keep = (u >= 0) & (w >= 0)
        return u[keep], w[keep]

    tu, tw = pairs(tr)
    seen = set((tu * K + tw).tolist())
    if n_train_pos:
        pick = rng.choice(len(tu), n_train_pos, replace=False)
        tu, tw = tu[pick], tw[pick]
    tn = rng.choice(K, len(tu), p=q)                     # negatives: same u, noise next word
    Xu = np.concatenate([tu, tu]); Xw = np.concatenate([tw, tn]); Y = np.r_[np.ones(len(tu)), np.zeros(len(tu))]
    perm = rng.permutation(len(Y)); Xu, Xw, Y = Xu[perm], Xw[perm], Y[perm]
    vu, vw = pairs(va)
    vn = rng.choice(K, len(vu), p=q)
    Vu = np.concatenate([vu, vu]); Vw = np.concatenate([vw, vn]); VY = np.r_[np.ones(len(vu)), np.zeros(len(vu))]
    Vseen = np.array([k in seen for k in (Vu * K + Vw).tolist()])
    out = os.path.join(ROOT, "data", f"ptb_pairs_K{K}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, Xu=Xu.astype(np.int16), Xw=Xw.astype(np.int16), Y=Y.astype(np.uint8),
                        Vu=Vu.astype(np.int16), Vw=Vw.astype(np.int16), VY=VY.astype(np.uint8), Vseen=Vseen,
                        top=top.astype(np.int32), q=q)
    json.dump([vocab[i] for i in top], open(os.path.join(ROOT, "data", f"vocab_K{K}.json"), "w"))
    stats = dict(K=K, n_train_pairs=int(len(Y)), n_valid_pairs=int(len(VY)),
                 valid_pos_unseen_share=float((~Vseen[VY == 1]).mean()),
                 valid_neg_unseen_share=float((~Vseen[VY == 0]).mean()),
                 n_unseen_pos=int(((~Vseen) & (VY == 1)).sum()), n_unseen_neg=int(((~Vseen) & (VY == 0)).sum()))
    return out, stats


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run = Run(f"pairs_data_K{K}", {"K": K, "negatives": "unigram^0.75 over top-K", "seed": 0},
              data_files=[ptb.path("train"), ptb.path("valid")])
    out, stats = make(K)
    run.print("wrote", out)
    run.finish(stats)
