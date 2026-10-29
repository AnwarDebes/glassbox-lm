"""src/pairs_data.py: the pair data of M1 on a toy stream, against an independent computation.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_pairs_data.py

Run from studies/clause_lm. CPU only, a few seconds. Toy sentences (tests/toyutil.py) in a temporary folder; the
module's ROOT is pointed there, so data/ptb_pairs_K{K}.npz and data/vocab_K{K}.json are written in the temporary
folder only. No Penn Treebank file is read.

Asserts, for K = 12 on a toy vocabulary of 30 words:
- top holds the K most frequent training words, ties broken by the lower id (stable sort), and q is
  count^0.75 normalised;
- the training positives are exactly the adjacent training pairs with both words among the K, and each negative keeps
  the first word of a positive (the multisets of first words are equal), with the second word among the K;
- the validation pairs are the adjacent validation pairs in stream order, followed by one negative per positive at
  the same offset with the same first word;
- the seen flag is true exactly when the pair occurs anywhere in the training stream, also when only a subset of
  training positives is kept (n_train_pos);
- the counts in the returned statistics match the arrays, the vocabulary file holds the words of top, and the same
  seed gives the same file;
- a property worth knowing: n_train_pos draws from the same random stream before the validation negatives are drawn,
  so a run with n_train_pos > 0 gets other validation negatives (the recorded data used n_train_pos = 0).
"""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()
K = 12


def main():
    with tempfile.TemporaryDirectory(prefix="clm_pairs_") as tmp:
        root = Path(tmp)
        train, valid = toyutil.toy_splits(seed=5, n_train=400, n_valid=120)
        ptb, _ = toyutil.use_src(root, train, valid)
        import pairs_data
        pairs_data.ROOT = str(root)

        vocab, w2i, ids = ptb.load(("train", "valid"))
        tr, va = ids["train"].astype(np.int64), ids["valid"].astype(np.int64)
        V = len(vocab)
        counts = np.bincount(tr, minlength=V)
        top_ref = sorted(range(V), key=lambda i: (-counts[i], i))[:K]
        remap = {w: k for k, w in enumerate(top_ref)}

        out, stats = pairs_data.make(K)
        d = np.load(out)
        top = d["top"].tolist()
        check(top == top_ref, "top = the K most frequent training words, ties to the lower id")
        q_ref = counts[top_ref].astype(float) ** 0.75
        check(np.allclose(d["q"], q_ref / q_ref.sum(), rtol=0, atol=1e-15), "q = count^0.75, normalised")

        def adj(stream):
            return [(remap[a], remap[b]) for a, b in zip(stream[:-1], stream[1:]) if a in remap and b in remap]

        tr_pairs, va_pairs = adj(tr.tolist()), adj(va.tolist())
        Xu, Xw, Y = d["Xu"].astype(int), d["Xw"].astype(int), d["Y"].astype(int)
        pos = Counter(zip(Xu[Y == 1].tolist(), Xw[Y == 1].tolist()))
        check(pos == Counter(tr_pairs), f"training positives = adjacent training pairs among the K ({len(tr_pairs)})")
        check(Counter(Xu[Y == 0].tolist()) == Counter(Xu[Y == 1].tolist()) and int((Y == 0).sum()) == len(tr_pairs),
              "each training negative keeps a positive's first word")
        check(bool(np.all((Xw >= 0) & (Xw < K))), "all second words are among the K")

        Vu, Vw, VY, Vseen = d["Vu"].astype(int), d["Vw"].astype(int), d["VY"].astype(int), d["Vseen"]
        n = len(va_pairs)
        check(list(zip(Vu[:n].tolist(), Vw[:n].tolist())) == va_pairs and VY[:n].tolist() == [1] * n,
              f"validation positives = adjacent validation pairs in stream order ({n})")
        check(VY[n:].tolist() == [0] * n and Vu[n:].tolist() == Vu[:n].tolist(),
              "validation negative i has positive i's first word, at offset n")
        seen_ref = set(tr_pairs)
        check(Vseen.dtype == bool and Vseen.tolist() == [(u, w) in seen_ref for u, w in zip(Vu.tolist(), Vw.tolist())],
              "seen flag = the pair occurs in the training stream")
        uns = ~Vseen
        check(stats["n_unseen_pos"] == int((uns & (VY == 1)).sum()) and stats["n_unseen_neg"] == int((uns & (VY == 0)).sum())
              and stats["n_valid_pairs"] == len(VY) and stats["n_train_pairs"] == len(Y),
              f"statistics match the arrays (unseen pos {stats['n_unseen_pos']}, neg {stats['n_unseen_neg']})")
        check(stats["n_unseen_pos"] > 0 and stats["n_unseen_neg"] > 0, "the toy data has unseen pairs of both kinds")
        words = json.loads((root / "data" / f"vocab_K{K}.json").read_text())
        check(words == [vocab[i] for i in top_ref], "the vocabulary file holds the words of top")

        first = {k: d[k].copy() for k in d.files}
        out2, _ = pairs_data.make(K)
        d2 = np.load(out2)
        check(all(np.array_equal(first[k], d2[k]) for k in first), "the same seed gives the same arrays")

        out3, stats3 = pairs_data.make(K, n_train_pos=50, seed=0)
        d3 = np.load(out3)
        seen3 = [(u, w) in seen_ref for u, w in zip(d3["Vu"].astype(int).tolist(), d3["Vw"].astype(int).tolist())]
        check(int((d3["Y"] == 1).sum()) == 50 and d3["Vseen"].tolist() == seen3,
              "with n_train_pos the seen flag still uses the whole training stream")
        check(d3["Vw"][:n].tolist() == first["Vw"][:n].tolist() and d3["Vw"][n:].tolist() != first["Vw"][n:].tolist(),
              "note: n_train_pos changes the random draws, so the validation negatives change with it")
    check.done()


if __name__ == "__main__":
    main()
