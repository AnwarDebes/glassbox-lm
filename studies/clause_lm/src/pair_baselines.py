"""Statistical baselines for the pair task (the bar the pair TM must beat), CPU only.

Task (data/ptb_pairs_K{K}.npz): is w a real next word after u? Pairs use the top-K words.
Metrics are AUC on validation pairs: all, seen-in-train, and UNSEEN (unseen positives vs
unseen negatives, the generalisation test where a word-bigram count model is at 0.5).
Baselines (all fitted on train text only):
  bigram_count     log c(u,w) (+0.5), the count model (0.5 on unseen by construction)
  class_bigram_k   k-means classes of u (right contexts) and w (left contexts) on PPMI-SVD
                   vectors; score = PMI(class(u), class(w)) from class-pair counts
  sim_smoothing    Dagan-Lee-Pereira-style: P(w|u) ~ sum over the 20 nearest u' (cosine of
                   right-context PPMI-SVD vectors) of sim(u,u') * c(u',w)/c(u')
"""
import json, os, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.stats import rankdata
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptb
from runlog import Run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def auc(score, y):
    r, n1 = rankdata(score), int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ppmi_svd(M, dim=100, seed=0):
    """Rows of the PPMI matrix of M, reduced to dim SVD dimensions and scaled to unit length.

    The start vector of the ARPACK iteration comes from a generator seeded with `seed` (added 2026-10-25). Without
    one, scipy.sparse.linalg.svds draws it from operating-system entropy, so every call gave slightly different
    vectors and, through k-means, other word classes: four calls of the k-means groups of clause_lm.py gave 158, 156,
    155 and 155 groups on 2026-10-25. The runs of 2026-10-02 used the unseeded call (their scipy version is not
    recorded), so their numbers are one draw that a rerun cannot repeat."""
    M = M.astype(np.float64)
    tot = M.sum()
    pr, pc = M.sum(1, keepdims=True) / tot, M.sum(0, keepdims=True) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M / tot) / (pr * pc))
    P = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)
    v0 = np.random.default_rng(seed).standard_normal(min(P.shape))
    U, S, _ = svds(csr_matrix(P), k=dim, v0=v0)
    E = U * np.sqrt(S)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def main(K=1000):
    d = np.load(os.path.join(ROOT, "data", f"ptb_pairs_K{K}.npz"))
    vocab, w2i, ids = ptb.load(("train", "valid"))
    V = len(vocab)
    tr = ids["train"].astype(np.int64)
    top = d["top"].astype(np.int64)
    remap = np.full(V, -1); remap[top] = np.arange(K)
    run = Run(f"pair_baselines_K{K}", {"K": K, "context_vocab": "top 2000 words", "svd_dim": 100,
                                       "kmeans_k": [50, 100, 200], "sim_neighbours": 20, "svd_seed": 0},
              data_files=[os.path.join(ROOT, "data", f"ptb_pairs_K{K}.npz"), ptb.path("train")])
    Vu, Vw, VY, Vseen = d["Vu"].astype(np.int64), d["Vw"].astype(np.int64), d["VY"].astype(int), d["Vseen"]
    uu = ~Vseen
    counts = np.bincount(tr, minlength=V)
    ctx = np.argsort(-counts)[:2000]
    cmap = np.full(V, -1); cmap[ctx] = np.arange(2000)
    a, b = tr[:-1], tr[1:]
    # right contexts of top-K words u (next word among the 2000 context words); left contexts of w
    m = (remap[a] >= 0) & (cmap[b] >= 0)
    R = np.zeros((K, 2000)); np.add.at(R, (remap[a[m]], cmap[b[m]]), 1)
    m = (remap[b] >= 0) & (cmap[a] >= 0)
    L = np.zeros((K, 2000)); np.add.at(L, (remap[b[m]], cmap[a[m]]), 1)
    # full top-K x top-K bigram counts
    m = (remap[a] >= 0) & (remap[b] >= 0)
    B = np.zeros((K, K)); np.add.at(B, (remap[a[m]], remap[b[m]]), 1)
    out = {}

    def report(name, sc):
        out[name] = dict(auc_all=auc(sc, VY), auc_seen=auc(sc[Vseen], VY[Vseen]), auc_unseen=auc(sc[uu], VY[uu]))
        run.print(name, out[name])

    report("bigram_count", np.log(B[Vu, Vw] + 0.5))
    Er, El = ppmi_svd(R), ppmi_svd(L)
    for k in (50, 100, 200):
        cu = KMeans(k, n_init=4, random_state=0).fit_predict(Er)
        cw = KMeans(k, n_init=4, random_state=0).fit_predict(El)
        N = np.zeros((k, k)); np.add.at(N, (cu[:, None], cw[None, :]), B)   # class-pair counts from word-pair counts
        N += 0.5
        pmi = np.log(N / N.sum()) - np.log(N.sum(1, keepdims=True) / N.sum()) - np.log(N.sum(0, keepdims=True) / N.sum())
        report(f"class_bigram_k{k}", pmi[cu[Vu], cw[Vw]])
    S = Er @ Er.T
    np.fill_diagonal(S, -1)
    nn = np.argsort(-S, axis=1)[:, :20]
    P = B / np.maximum(B.sum(1, keepdims=True), 1)
    sims = np.take_along_axis(S, nn, axis=1).clip(min=0)
    sc = np.einsum("ij,ij->i", sims[Vu], P[nn[Vu], Vw[:, None]]) / (sims[Vu].sum(1) + 1e-12)
    report("sim_smoothing_nn20", np.log(sc + 1e-6))
    run.finish(out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
