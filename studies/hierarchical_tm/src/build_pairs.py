"""Build the clean Wikipedia paragraph-pair data (design decision "clean data"; gates.md Check 1b).

1. Paragraphs (the probe's rule, feas_slide11/prep_data.py): WikiText-103 raw, train split.
   A paragraph is a non-heading line with at least 30 lowercase alphabetic tokens. Its words are the
   lowercase alphabetic tokens minus the probe's stop words and one-letter tokens, as a set.
2. Article-disjoint splits with a fixed seed: train 70%, dev 10%, final test 10%, NPMI reference 10%.
   NPMI-reference articles are never used for pairs.
3. Vocabulary: top 3,072 words by document frequency over train-split paragraphs. The main
   vocabulary is its first 1,024 words. Paragraphs are stored as sorted sets of these ids.
4. Three pair versions, each balanced (train 40k / dev 20k / test 10k pairs) with random A/B order.
   Only articles with at least 2 paragraphs are sampled.
   sa  same article. Positive: an article uniformly, then 2 distinct paragraphs uniformly.
       Negative: 2 distinct articles uniformly, then 1 paragraph uniformly from each.
   np  next paragraph. Positive: an article uniformly, then a consecutive pair (i, i+1) uniformly.
       Negative: 2 distinct articles uniformly, then from each one side of a uniformly chosen consecutive
       pair, so both classes have the same paragraph-position law (first and last paragraphs).
   ns  next paragraph, shared words removed: see build_ns_swap. Every word in both paragraphs is removed.
History (Check 1b runs in results/): v1 followed the study design (np negative paragraph uniform; ns
negatives picked from random pairs to match the positives' shared-word histogram) and failed for np and ns.
v2 fixed np and tried other ns negatives (still failed). v3 is this file. The exact v1 and v2 code is
copied in their run dirs (*_data_build_exploratory, *_data_build_v2_exploratory).
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from array import array

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairdata import DATA, csr_to_ids  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tm_language_model", "data", "wikitext103", "raw", "train")
STOP = set("""a about above after again against ain all am an and any are aren aren't as at be because been before being
below between both but by can couldn couldn't d did didn didn't do does doesn doesn't doing don don't down during each few for
from further had hadn hadn't has hasn hasn't have haven haven't having he her here hers herself him himself his how i if in into
is isn isn't it it's its itself just ll m ma me mightn mightn't more most mustn mustn't my myself needn needn't no nor not now o
of off on once only or other our ours ourselves out over own re s same shan shan't she she's should should've shouldn shouldn't
so some such t than that that'll the their theirs them themselves then there these they this those through to too under until up
ve very was wasn wasn't we were weren weren't what when where which while who whom why will with won won't wouldn wouldn't y you
you'd you'll you're you've your yours yourself yourselves unk also would could one two three first second new may
""".split())  # copied from the probe (NLTK English list + corpus fillers). Wikipedia pairs only, never slide 10.
WORD = re.compile(r"[a-z]+")
SIZES = {"train": 40000, "dev": 20000, "test": 10000}
F_MAIN, F_BIG, KB = 1024, 3072, 10  # k bins 0..9 and >=10


def read_paragraphs():
    """Per-article paragraph counts, flat word ids + row offsets of every paragraph, and the word list."""
    off = np.load(f"{RAW}/doc_bytes.npy")
    with open(f"{RAW}/text.utf8", "rb") as f:
        raw = f.read()
    w2g, gids, poff, npar = {}, array("i"), [0], np.zeros(len(off) - 1, np.int64)
    for i in range(len(off) - 1):
        for line in raw[off[i]:off[i + 1]].decode("utf8").split("\n"):
            s = line.strip()
            if not s or (s.startswith("=") and s.endswith("=")):
                continue
            toks = WORD.findall(s.lower())
            if len(toks) >= 30:
                words = dict.fromkeys(t for t in toks if t not in STOP and len(t) > 1)  # keeps first-seen order
                gids.extend(w2g.setdefault(t, len(w2g)) for t in words)
                poff.append(len(gids))
                npar[i] += 1
    return npar, np.frombuffer(gids, np.int32), np.array(poff, np.int64), list(w2g)


class Corpus:
    def __init__(self, npar, M):
        self.npar, self.M, self.M1 = npar, M, M[:, :F_MAIN].tocsr()
        self.start = np.concatenate([[0], np.cumsum(npar)])
        self.par_art = np.repeat(np.arange(len(npar)), npar)

    def rows(self, pairs, M=None):
        aA, pA, aB, pB = pairs
        M = self.M1 if M is None else M
        return M[self.start[aA] + pA], M[self.start[aB] + pB]

    def sample_pos(self, kind, arts, n, rng):
        a = arts[rng.integers(len(arts), size=n)]
        if kind == "sa":
            i = rng.integers(self.npar[a])
            j = rng.integers(self.npar[a] - 1)
            j += j >= i
        else:  # next paragraph
            i = rng.integers(self.npar[a] - 1)
            j = i + 1
        return a, i, a, j

    def sample_neg(self, arts, n, rng, posmatch=False):
        """2 distinct articles uniformly, then 1 paragraph from each: uniformly, or (posmatch) one side of a
        uniformly chosen consecutive pair, which is the position law of next-paragraph positives."""
        i1 = rng.integers(len(arts), size=n)
        i2 = rng.integers(len(arts) - 1, size=n)
        i2 += i2 >= i1
        a1, a2 = arts[i1], arts[i2]
        if posmatch:
            return (a1, rng.integers(self.npar[a1] - 1) + (rng.random(n) < 0.5),
                    a2, rng.integers(self.npar[a2] - 1) + (rng.random(n) < 0.5))
        return a1, rng.integers(self.npar[a1]), a2, rng.integers(self.npar[a2])

    def k_empty(self, pairs):
        """Shared-word count k (F=1024) and whether removal would leave an empty side (F=1024)."""
        A, B = self.rows(pairs)
        k = np.asarray(A.multiply(B).sum(1)).ravel()
        return k, (k == np.diff(A.indptr)) | (k == np.diff(B.indptr))


def take(pairs, idx):
    return tuple(x[idx] for x in pairs)


def cat(*ps):
    return tuple(np.concatenate(x) for x in zip(*ps))


def build_ns_swap(C, arts, n_cls, rng, draws=30, chunk=300):
    """ns rebuild (v3), partner swap. Positives: np-style draws (article uniform, then a consecutive pair (A, B)),
    duplicates removed, pairs that would get an empty side dropped. Group them by (shared words R, length of B)
    at F=1024. Inside a group, B paragraphs are permuted among positives from different articles, so that each
    new pair (A_i, B_j) shares exactly R (a cycle cover, solved as an assignment problem on random chunks of at
    most `chunk` members). Negatives are the permuted pairs. So every paragraph appears once in each class, in
    the same role (earlier / later), and a negative has the same k and the same two lengths as its positive.
    Whole cycles are then drawn at random until n_cls positives."""
    cand = C.sample_pos("np", arts, draws * n_cls, rng)
    _, first = np.unique(np.c_[cand[0], cand[1]], axis=0, return_index=True)
    cand = take(cand, np.sort(first))
    k, e = C.k_empty(cand)
    st = {"pos_draws": draws * n_cls, "pos_unique": len(k), "pos_dropped_empty": int(e.sum())}
    cand, k = take(cand, ~e), k[~e]
    A, B = C.rows(cand)
    S = A.multiply(B).tocsr()
    S.sort_indices()
    groups = {}
    for t, lb in enumerate(np.diff(B.indptr)):
        groups.setdefault((S.indices[S.indptr[t]:S.indptr[t + 1]].tobytes(), int(lb)), []).append(t)
    cycles = []
    for members in groups.values():
        for m in np.array_split(rng.permutation(members), -(-len(members) // chunk)):
            if len(m) < 2:
                continue
            X = (A[m].astype(np.int32) @ B[m].T.astype(np.int32)).toarray()
            cost = np.where((X == k[m[0]]) & (cand[0][m][:, None] != cand[0][m][None, :]), -1.0, 1e6)
            np.fill_diagonal(cost, 0.0)
            pi = linear_sum_assignment(cost)[1]
            seen = pi == np.arange(len(m))
            for i in np.flatnonzero(~seen):
                cyc = []
                while not seen[i]:
                    seen[i] = True
                    cyc.append(i)
                    i = pi[i]
                if cyc:
                    cycles.append((m[cyc], m[pi[cyc]]))
    st["matched_positives"] = int(sum(len(c[0]) for c in cycles))
    sel, total = [], 0
    for c in rng.permutation(len(cycles)):
        if total + len(cycles[c][0]) <= n_cls and n_cls - total - len(cycles[c][0]) != 1:  # 1 cannot be filled
            sel.append(cycles[c])
            total += len(cycles[c][0])
    i, j = np.concatenate([c[0] for c in sel]), np.concatenate([c[1] for c in sel])
    st["short_of_target"] = int(n_cls - total)
    st["k_hist_selected"] = np.bincount(np.minimum(k[i], KB), minlength=KB + 1).tolist()
    return take(cand, i), (cand[0][i], cand[1][i], cand[2][j], cand[3][j]), st


def assemble(C, pos, neg, rng, remove_shared):
    pairs = cat(pos, neg)
    y = np.r_[np.ones(len(pos[0]), np.uint8), np.zeros(len(neg[0]), np.uint8)]
    swap = rng.random(len(y)) < 0.5
    aA, pA, aB, pB = pairs
    pairs = (np.where(swap, aB, aA), np.where(swap, pB, pA), np.where(swap, aA, aB), np.where(swap, pA, pB))
    perm = rng.permutation(len(y))
    (aA, pA, aB, pB), y = take(pairs, perm), y[perm]
    A, B = C.rows((aA, pA, aB, pB), C.M)
    if remove_shared:
        S = A.multiply(B).tocsr()
        A, B = (A - S).tocsr(), (B - S).tocsr()
    a_ids, a_off = csr_to_ids(A)
    b_ids, b_off = csr_to_ids(B)
    return dict(a_ids=a_ids, a_off=a_off, b_ids=b_ids, b_off=b_off, y=y, art_a=aA.astype(np.int32),
                art_b=aB.astype(np.int32), par_a=pA.astype(np.int32), par_b=pB.astype(np.int32))


def check_pairs(d, version, split_arts):
    y, aA, aB, pA, pB = d["y"], d["art_a"], d["art_b"], d["par_a"], d["par_b"]
    assert 2 * int(y.sum()) == len(y), "labels not balanced"
    assert np.isin(aA, split_arts).all() and np.isin(aB, split_arts).all(), "pair uses an article outside its split"
    assert (aA[y == 0] != aB[y == 0]).all() and (aA[y == 1] == aB[y == 1]).all()
    assert (pA[y == 1] != pB[y == 1]).all()
    if version != "sa":
        assert (np.abs(pA[y == 1] - pB[y == 1]) == 1).all()
    if version == "ns":
        from pairdata import ids_to_csr
        A, B = ids_to_csr(d["a_ids"], d["a_off"], F_BIG), ids_to_csr(d["b_ids"], d["b_off"], F_BIG)
        assert A.multiply(B).nnz == 0, "ns overlap not 0"
        assert (np.diff(ids_to_csr(d["a_ids"], d["a_off"], F_MAIN).indptr) > 0).all()
        assert (np.diff(ids_to_csr(d["b_ids"], d["b_off"], F_MAIN).indptr) > 0).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    t0 = time.time()
    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        for f in ("build_pairs.py", "pairdata.py"):
            shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), run_dir)
    npar, gids, poff, words = read_paragraphs()
    n_art, n_par = len(npar), int(npar.sum())
    print(f"read {n_art} articles, {n_par} paragraphs, {len(words)} distinct words in {time.time() - t0:.0f}s", flush=True)

    perm = np.random.default_rng(a.seed).permutation(n_art)
    c = np.round(np.array([0.7, 0.8, 0.9]) * n_art).astype(int)
    splits = {"train": perm[:c[0]], "dev": perm[c[0]:c[1]], "test": perm[c[1]:c[2]], "npmi_ref": perm[c[2]:]}
    splits = {k: np.sort(v) for k, v in splits.items()}
    allv = np.concatenate(list(splits.values()))
    assert len(np.unique(allv)) == n_art == len(allv), "splits not disjoint or not complete"

    par_art = np.repeat(np.arange(n_art), npar)
    in_train = np.isin(par_art, splits["train"])
    df = np.bincount(gids[np.repeat(in_train, np.diff(poff))], minlength=len(words))
    order = np.argsort(-df, kind="stable")[:F_BIG]
    rank = np.full(len(words), -1, np.int64)
    rank[order] = np.arange(F_BIG)
    os.makedirs(a.out, exist_ok=True)
    for F in (F_MAIN, F_BIG):
        json.dump([words[g] for g in order[:F]], open(os.path.join(a.out, f"vocab_F{F}.json"), "w"))
    r = rank[gids]
    rows = np.repeat(np.arange(n_par), np.diff(poff))
    M = sparse.csr_matrix((np.ones(int((r >= 0).sum()), np.uint8), (rows[r >= 0], r[r >= 0])), shape=(n_par, F_BIG))
    M.sort_indices()
    C = Corpus(npar, M)
    ids, off = csr_to_ids(M)
    np.savez_compressed(os.path.join(a.out, "paragraphs_F3072.npz"), ids=ids, off=off, npar=npar.astype(np.int32))
    len1 = np.diff(C.M1.indptr)
    stats = {"seed": a.seed, "design": "v3", "n_articles": n_art, "n_paragraphs": n_par, "distinct_words": len(words),
             "vocab_df_min_F1024": int(df[order[F_MAIN - 1]]), "vocab_df_min_F3072": int(df[order[-1]]),
             "train_paragraphs": int(in_train.sum()),
             "vocab_words_per_paragraph_F1024": {"mean": float(len1.mean()), "median": float(np.median(len1)),
                                                 "zero": int((len1 == 0).sum())},
             "vocab_words_per_paragraph_F3072": float(np.diff(M.indptr).mean()), "splits": {}, "ns": {}}
    elig = {}
    for s, arts in splits.items():
        elig[s] = arts[npar[arts] >= 2]
        stats["splits"][s] = {"articles": len(arts), "eligible_articles": len(elig[s]),
                              "paragraphs": int(npar[arts].sum())}
    json.dump({"seed": a.seed, "source": RAW, "fractions": [0.7, 0.1, 0.1, 0.1],
               "note": "WikiText-103 train-split article indices (0-based, order of titles.json). "
                       "npmi_ref articles are never used for pairs.",
               **{k: v.tolist() for k, v in splits.items()}}, open(os.path.join(a.out, "splits.json"), "w"))

    for vi, version in enumerate(("sa", "np", "ns")):
        os.makedirs(os.path.join(a.out, version), exist_ok=True)
        for si, (split, n) in enumerate(SIZES.items()):
            rng = np.random.default_rng([a.seed, vi, si])
            arts = elig[split]
            if version == "ns":
                pos, neg, st = build_ns_swap(C, arts, n // 2, rng)
                stats["ns"][split] = st
            else:
                pos = C.sample_pos(version, arts, n // 2, rng)
                neg = C.sample_neg(arts, n // 2, rng, posmatch=(version == "np"))
            d = assemble(C, pos, neg, rng, remove_shared=(version == "ns"))
            check_pairs(d, version, arts)
            np.savez_compressed(os.path.join(a.out, version, f"{split}.npz"), **d)
            print(f"{version}/{split}: {len(d['y'])} pairs, {int(d['y'].sum())} positive, "
                  f"{len(np.unique(np.r_[d['art_a'], d['art_b']]))} articles  ({time.time() - t0:.0f}s)", flush=True)
    used = set()
    for version in ("sa", "np", "ns"):
        for split in SIZES:
            z = np.load(os.path.join(a.out, version, f"{split}.npz"))
            used |= set(np.unique(np.r_[z["art_a"], z["art_b"]]).tolist())
    assert not used & set(splits["npmi_ref"].tolist()), "an NPMI-reference article was used in a pair"
    stats["wall_s"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(a.out, "build_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    if run_dir:
        shutil.copy(os.path.join(a.out, "build_stats.json"), run_dir)
        m = {"n_articles": n_art, "n_paragraphs": n_par, "wall_s": stats["wall_s"]}
        m.update({f"ns_{s}_short_of_target": v["short_of_target"] for s, v in stats["ns"].items()})
        json.dump({k: {"value": v, "label": "measured"} for k, v in m.items()},
                  open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
