"""M2 evaluator: a clause-count language model on top of KN3 (PTB word level, validation).

A "clause" here is a context group G (a set of previous words u). Its vote is a count-based
next-word distribution over the full vocabulary:
  P_G(w) = max(c_G(w) - D, 0) / c_G + D * N_G / c_G * Pcont(w),   c_G(w) = sum_{u in G} c(u, w).
For a context (t, u), the clause component is the mean of P_G over the groups that contain u
(the firing clauses). Final model: P = (1 - lam) P_KN3 + lam P_clause, used only when u fires
at least one clause. lam is chosen on one half of validation and scored on the other half
(2-fold cross-fit by sentence parity), and the two halves are averaged.
Groups come from either:
  --groups kmeans:K   statistical word classes (k-means on right-context PPMI-SVD vectors of
                      the top-1000 words). This is the class-based baseline the TM must beat.
  --groups PATH       a pair-TM run's groups.json (the G1 lists of its positive clauses).
Both kinds of groups pass the same filter, keep_groups(): at least MIN_GROUP_WORDS = 2 words. Until 2026-10-25 a
pair-TM group of a single word was admitted while a k-means group needed two; the k-means run of 2026-10-02 (the M2
bar) used the filter that both arms now share. A pair-TM clause whose u-side has an empty OR-option fires for every
word, so it is no word group and is left out before the filter.
Reported: perplexity and top-1/top-5 for KN3 alone vs KN3 + clauses, overall and by context
type (trigram seen / trigram unseen).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptb
from kn import KN3, discount
from runlog import Run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAMS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
MIN_GROUP_WORDS = 2


def keep_groups(groups):
    """The one group filter of both arms: groups (lists of word ids) with at least MIN_GROUP_WORDS words."""
    return [list(g) for g in groups if len(g) >= MIN_GROUP_WORDS]


def kmeans_groups(K, tr, V, top, k):
    from pair_baselines import ppmi_svd
    from sklearn.cluster import KMeans
    remap = np.full(V, -1); remap[top] = np.arange(len(top))
    counts = np.bincount(tr, minlength=V)
    ctx = np.argsort(-counts)[:2000]; cmap = np.full(V, -1); cmap[ctx] = np.arange(2000)
    a, b = tr[:-1], tr[1:]
    m = (remap[a] >= 0) & (cmap[b] >= 0)
    R = np.zeros((len(top), 2000)); np.add.at(R, (remap[a[m]], cmap[b[m]]), 1)
    lab = KMeans(k, n_init=4, random_state=0).fit_predict(ppmi_svd(R))
    return keep_groups(top[lab == c].tolist() for c in range(k))


def tm_groups(gj, top):
    """The G1 word groups of a pair-TM groups.json: clauses with an empty u-side OR-option are left out (they fire
    for every word), then keep_groups()."""
    return keep_groups(top[np.asarray(g["G1_ids"], dtype=np.int64)].tolist()
                       for g in gj if g.get("empty_options_G1", 0) == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="kmeans:200")
    ap.add_argument("--K", type=int, default=1000)
    a = ap.parse_args()
    vocab, w2i, ids = ptb.load(("train", "valid"))
    V, eos = len(vocab), w2i["<eos>"]
    tr, va = ids["train"].astype(np.int64), ids["valid"].astype(np.int64)
    d = np.load(os.path.join(ROOT, "data", f"ptb_pairs_K{a.K}.npz"))
    top = d["top"].astype(np.int64)
    if a.groups.startswith("kmeans:"):
        groups = kmeans_groups(a.K, tr, V, top, int(a.groups.split(":")[1]))
    else:
        groups = tm_groups(json.load(open(a.groups)), top)
    run = Run(f"clause_lm_{a.groups.replace(':', '').replace('/', '_')[-40:]}", {**vars(a), "n_groups": len(groups),
              "min_group_words": MIN_GROUP_WORDS, "svd_seed": 0, "lams": LAMS, "crossfit": "2-fold by sentence parity"},
              data_files=[ptb.path("train"), ptb.path("valid")])
    # group vote distributions
    stream = np.concatenate([[eos], tr])
    k2, c2 = np.unique(stream[:-1] * V + stream[1:], return_counts=True)
    D = discount(c2)
    kn = KN3(tr, V, eos)
    member = [[] for _ in range(V)]
    votes = []
    for gi, G in enumerate(groups):
        Gs = np.array(G)
        m = np.isin(k2 // V, Gs)
        cw = np.bincount(k2[m] % V, weights=c2[m], minlength=V)
        tot, n = cw.sum(), (cw > 0).sum()
        if tot == 0:
            votes.append(None); continue
        votes.append(np.maximum(cw - D, 0) / tot + (D * n / tot) * kn.pcont)
        for u in G:
            member[u].append(gi)
    # sentence-parity halves for cross-fitting lam
    sent = np.cumsum(np.r_[0, (va[:-1] == eos)])
    half = sent % 2
    L = len(LAMS)
    lp = np.zeros((L, len(va))); h1 = np.zeros((L, len(va)), bool); h5 = np.zeros((L, len(va)), bool)
    fires = np.zeros(len(va), bool); triseen = np.zeros(len(va), bool)
    for pos, t, u, dk, seen in kn.walk(va):
        triseen[pos] = seen
        js = [j for j in member[u] if votes[j] is not None]
        if js:
            fires[pos] = True
            pc = np.mean([votes[j] for j in js], axis=0)
        for li, lam in enumerate(LAMS):
            dd = dk if (not js or lam == 0) else (1 - lam) * dk + lam * pc
            lp[li, pos] = np.log(dd[va[pos]])
            top5 = np.argpartition(-dd, 5)[:5]
            best = top5[np.argmax(dd[top5])]
            h1[li, pos] = va[pos] == best
            h5[li, pos] = np.isin(va[pos], top5)
    res = {"n_groups": len(groups), "fire_share": float(fires.mean()), "trigram_seen_share": float(triseen.mean())}
    picks = []
    for h in (0, 1):                                  # choose lam on half h, score on the other half
        tune, test = half == h, half != h
        li = int(np.argmin([-lp[i, tune].mean() for i in range(L)]))
        picks.append((LAMS[li], test, li))
    def score(mask_fn):
        out = {}
        for name, li_of in (("kn3", lambda li: 0), ("kn3+clauses", lambda li: li)):
            ppl_parts, a1, a5, n = [], 0, 0, 0
            for lam, test, li in picks:
                m = test & mask_fn
                ppl_parts.append(-lp[li_of(li), m].sum()); a1 += h1[li_of(li), m].sum(); a5 += h5[li_of(li), m].sum(); n += m.sum()
            out[name] = dict(ppl=float(np.exp(sum(ppl_parts) / n)), top1=float(a1 / n), top5=float(a5 / n), n=int(n))
        return out
    res["lam_chosen_per_half"] = [p[0] for p in picks]
    res["all"] = score(np.ones(len(va), bool))
    res["trigram_unseen"] = score(~triseen)
    res["trigram_seen"] = score(triseen)
    for k in ("all", "trigram_unseen", "trigram_seen"):
        run.print(k, json.dumps(res[k]))
    run.finish(res)


if __name__ == "__main__":
    main()
