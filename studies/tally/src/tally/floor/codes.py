"""Category codes for step A2: one class per word at one or more levels (module A of the design).

A code is an int array C of shape (V, n_levels), coarse level first. C[w, l] is the class of word
w at level l. <eos> has a class of its own at every level, so that class streams keep the
sentence ends (sentence mode). Codes are built from the training split only (design rule 3) and
saved under data/a2_codes/ with the sha256 of their source (the train token stream, the PTB
files, and any lexicon or group file) and the sha256 of the code array itself.

Builders:
- exchange_tree: exchange classes (Martin, Liermann, Ney 1998) at 1024 on train bigram counts,
  merged bottom-up by exchange clustering of the classes into 256, 64, 16 (floor/exchange.py).
- random_code: the exchange tree with the words permuted (seeded); every level keeps the exact
  class-size distribution (words per class) of the exchange tree.
- kmeans200: the clause_lm recipe (top 1,000 train words, right-context counts over the top
  2,000 words, PPMI, 100-dim SVD, k-means 200 with n_init 4 and random_state 0, groups of at
  least 2 words), rebuilt on the train split with a fixed SVD start vector. Words outside the
  groups are classes of their own.
- predefined: majority part-of-speech tag (universal tagset) from a perceptron tagger trained on
  the Brown corpus, then that tag plus the WordNet 3.0 lexicographer file (supersense) of the
  first synset of the word with that part of speech.
- topic: sliding-window code. Co-occurrence of each word with the context words 3 to 25 tokens
  away inside the same 100-sentence block (context words: train ranks 101 to 2,100), PPMI,
  100-dim SVD, k-means 256 on the words, then k-means 64 on the 256 centres (weighted by tokens).
- orgroups: the OR-groups of a hierarchical TM run of the hierarchical_tm study (word lists, WikiText-103
  paragraphs); each mapped word goes to the group with the highest group NPMI that holds it;
  other words are classes of their own.
"""
import hashlib
import json
import os
import re

import numpy as np

from . import exchange as ex

STUDY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STUDIES = os.path.dirname(STUDY)
CODE_DIR = os.path.join(STUDY, "data", "a2_codes")
EXTERNAL = os.path.join(STUDY, "data", "a2_external")
WORDNET_DIR = os.path.join(STUDIES, "tm_language_model", "data", "concept_eval", "nltk_data")
ORGROUPS_FILE = os.path.join(STUDIES, "hierarchical_tm", "results",
                             "20261002T114726Z_check4a_word_groups", "groups.tsv")
ORGROUPS_RUN = "20261002T050235Z_e4_head_sa_s1_scoreP100"
EXCHANGE_LEVELS = (16, 64, 256, 1024)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def stream_hash(train):
    return sha256_bytes(np.ascontiguousarray(train, dtype=np.int32).tobytes())


def normalise(C, eos):
    """Relabel every level to 0..n-1 by first appearance in word-id order, with <eos> in a class
    of its own, given the last id."""
    C = np.asarray(C, dtype=np.int64)
    if C.ndim == 1:
        C = C[:, None]
    out = np.zeros_like(C)
    for l in range(C.shape[1]):
        col = C[:, l]
        assert (col == col[eos]).sum() == 1, "eos must be alone in its class"
        keep = np.arange(len(col)) != eos
        _, first, inv = np.unique(col[keep], return_index=True, return_inverse=True)
        rank = np.argsort(np.argsort(first))
        out[keep, l] = rank[inv]
        out[eos, l] = len(first)
    return out


def n_classes(C):
    return [int(C[:, l].max()) + 1 for l in range(C.shape[1])]


def save(name, C, meta):
    os.makedirs(CODE_DIR, exist_ok=True)
    C = np.ascontiguousarray(C, dtype=np.int32)
    meta = dict(meta, name=name, shape=list(C.shape), n_classes=n_classes(C),
                code_sha256=sha256_bytes(C.tobytes()))
    np.save(os.path.join(CODE_DIR, name + ".npy"), C)
    with open(os.path.join(CODE_DIR, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def load(name, source):
    """Saved code if its source hashes equal source (dict), else None."""
    p = os.path.join(CODE_DIR, name + ".json")
    if not os.path.exists(p):
        return None, None
    with open(p) as f:
        meta = json.load(f)
    if meta.get("source") != source:
        return None, None
    C = np.load(os.path.join(CODE_DIR, name + ".npy")).astype(np.int64)
    if sha256_bytes(np.ascontiguousarray(C, dtype=np.int32).tobytes()) != meta["code_sha256"]:
        return None, None
    return C, meta


def get(name, source, build, log=print):
    """Load the saved code or build and save it. build() returns (C, extra meta)."""
    C, meta = load(name, source)
    if C is not None:
        log(f"code {name}: loaded {meta['code_sha256'][:12]} classes {meta['n_classes']}")
        return C, meta
    C, extra = build()
    meta = save(name, C, dict(extra, source=source))
    log(f"code {name}: built {meta['code_sha256'][:12]} classes {meta['n_classes']}")
    return C.astype(np.int64), meta


# ---- exchange tree and random codes ----

def exchange_tree(train, V, eos, levels=EXCHANGE_LEVELS, log=None):
    fine_first = tuple(sorted(levels, reverse=True))
    C, info = ex.tree(train, V, eos, fine_first, log=log)
    C = C[:, ::-1]                                # coarse level first
    return normalise(C, eos), {"builder": "exchange_tree", "levels": list(levels),
                               "init": "Martin et al.: K-1 most frequent own class, rest one",
                               "tree": "bottom-up, exchange clustering of the classes below",
                               "max_passes": 20, "stop_share": 0.001, "passes": info}


def random_code(base, eos, seed):
    """Words (all but <eos>) permuted: C[w] = base[perm[w]]."""
    V = base.shape[0]
    words = np.flatnonzero(np.arange(V) != eos)
    perm = np.arange(V)
    perm[words] = words[np.random.default_rng(seed).permutation(len(words))]
    return normalise(base[perm], eos), {"builder": "random_code", "seed": seed,
                                        "base": "exchange_tree", "rule": "C[w] = base[perm[w]]"}


# ---- k-means-200 (clause_lm recipe) ----

def ppmi_svd(M, dim=100, seed=0):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    M = np.asarray(M, dtype=np.float64)
    tot = M.sum()
    pr, pc = M.sum(1, keepdims=True) / tot, M.sum(0, keepdims=True) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M / tot) / (pr * pc))
    P = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)
    v0 = np.random.default_rng(seed).standard_normal(min(P.shape))
    U, S, _ = svds(csr_matrix(P), k=dim, v0=v0)
    E = U * np.sqrt(S)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def kmeans200(train, V, eos, K=1000, k=200, n_ctx=2000):
    from sklearn.cluster import KMeans
    tr = np.asarray(train, dtype=np.int64)
    counts = np.bincount(tr, minlength=V)
    top = np.argsort(-counts, kind="stable")[:K]
    remap = np.full(V, -1)
    remap[top] = np.arange(K)
    ctx = np.argsort(-counts, kind="stable")[:n_ctx]
    cmap = np.full(V, -1)
    cmap[ctx] = np.arange(n_ctx)
    a, b = tr[:-1], tr[1:]
    m = (remap[a] >= 0) & (cmap[b] >= 0)
    R = np.zeros((K, n_ctx))
    np.add.at(R, (remap[a[m]], cmap[b[m]]), 1)
    lab = KMeans(k, n_init=4, random_state=0).fit_predict(ppmi_svd(R))
    groups = [top[lab == c] for c in range(k) if (lab == c).sum() >= 2]
    C = np.arange(V) + len(groups)                 # own class for every word outside the groups
    for gi, G in enumerate(groups):
        C[G] = gi
    eos_group = [gi for gi, G in enumerate(groups) if eos in G]
    C[eos] = V + len(groups)                       # <eos> always alone
    sizes = sorted(len(G) for G in groups)
    return normalise(C, eos), {
        "builder": "kmeans200", "recipe": "clause_lm.kmeans_groups on the train split",
        "top_words": K, "context_words": n_ctx, "svd_dim": 100, "svd_v0_seed": 0,
        "kmeans": {"k": k, "n_init": 4, "random_state": 0}, "groups_min2": len(groups),
        "words_in_groups": int(sum(sizes)), "group_size_min_median_max":
        [sizes[0], sizes[len(sizes) // 2], sizes[-1]], "eos_was_in_group": bool(eos_group),
        "stream": "continuous (clause_lm convention), bigrams cross sentence ends"}


# ---- predefined: part of speech plus WordNet first-sense group ----

NUM_RE = re.compile(r"^[\d.,:/-]*\d[\d.,:/-]*$")
WN_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def _nltk_paths():
    import nltk
    for p in (os.path.join(EXTERNAL, "nltk_data"), WORDNET_DIR):
        if p not in nltk.data.path:
            nltk.data.path.insert(0, p)
    return nltk


def brown_sentences():
    """Brown corpus, universal tags, lowercased, numbers as N, punctuation removed (as PTB)."""
    nltk = _nltk_paths()
    from nltk.corpus import brown
    out = []
    for s in brown.tagged_sents(tagset="universal"):
        t = [("N" if NUM_RE.match(w) else w.lower(), tag) for w, tag in s if tag != "."]
        if t:
            out.append(t)
    return out


def predefined(train, vocab, eos, tagger_iters=5, log=print):
    nltk = _nltk_paths()
    from nltk.corpus import wordnet as wn
    from nltk.tag.perceptron import PerceptronTagger
    import random
    V = len(vocab)
    sents = brown_sentences()
    random.seed(0)
    tagger = PerceptronTagger(load=False)
    tagger.train(sents, nr_iter=tagger_iters)
    if log:
        log(f"tagger trained on Brown: {len(sents)} sentences")
    tr = np.asarray(train, dtype=np.int64)
    ends = np.flatnonzero(tr == eos)
    tagset = sorted({t for s in sents for _, t in s} - {"."})
    tix = {t: i for i, t in enumerate(tagset)}
    counts = np.zeros((V, len(tagset)), dtype=np.int64)
    start = 0
    for e in ends:
        words = [vocab[i] for i in tr[start:e]]
        if words:
            tags = tagger.tag(words)
            for i, (_, t) in zip(tr[start:e], tags):
                counts[i, tix[t]] += 1
        start = e + 1
    tag = np.empty(V, dtype=object)
    for w in range(V):
        if counts[w].sum() > 0:
            tag[w] = tagset[int(np.argmax(counts[w]))]
        else:
            tag[w] = tagger.tag([vocab[w]])[0][1]
    tag[vocab.index("<unk>")] = "UNK"
    tag[eos] = "EOS"
    lex = np.empty(V, dtype=object)
    for w in range(V):
        lex[w] = tag[w]
        if tag[w] in WN_POS:
            ss = wn.synsets(vocab[w], pos=WN_POS[tag[w]])
            if ss:
                lex[w] = tag[w] + ":" + ss[0].lexname()
    l1 = {t: i for i, t in enumerate(sorted(set(tag)))}
    l2 = {t: i for i, t in enumerate(sorted(set(lex)))}
    C = np.stack([[l1[t] for t in tag], [l2[t] for t in lex]], 1)
    agree = counts.max(1).sum() / max(counts.sum(), 1)
    return normalise(C, eos), {
        "builder": "predefined", "tagger": f"nltk {nltk.__version__} PerceptronTagger(load=False), "
        f"trained {tagger_iters} iterations on the Brown corpus, universal tagset, lowercased, "
        "numbers as N, punctuation removed, random.seed(0)",
        "tag_dictionary": "majority tag of each word over the tagged train split; words absent "
        "from train: the tag of the word alone; <unk> and <eos> own classes",
        "wordnet": f"WordNet {wn.get_version()}, lexname of synsets(word, pos)[0]",
        "majority_tag_token_share": float(agree), "levels": ["pos", "pos+lexname"],
        "n_tags": len(l1), "n_pos_lex": len(l2)}


# ---- sliding-window topic code ----

def window_counts(train, blocks, V, ctx_words, lo=3, hi=25):
    tr = np.asarray(train, dtype=np.int64)
    bl = np.asarray(blocks)
    F = len(ctx_words)
    cmap = np.full(V, -1, dtype=np.int64)
    cmap[ctx_words] = np.arange(F)
    acc = np.zeros(V * F, dtype=np.float64)
    for d in range(lo, hi + 1):
        a, b = tr[:-d], tr[d:]
        same = bl[:-d] == bl[d:]
        for u, v in ((a, b), (b, a)):
            m = same & (cmap[v] >= 0)
            acc += np.bincount(u[m] * F + cmap[v[m]], minlength=V * F)
    return acc.reshape(V, F)


def topic(train, blocks, V, eos, k_fine=256, k_coarse=64, lo=3, hi=25, skip_top=100, n_ctx=2000):
    from sklearn.cluster import KMeans
    tr = np.asarray(train, dtype=np.int64)
    counts = np.bincount(tr, minlength=V)
    rank = np.argsort(-counts, kind="stable")
    ctx = rank[skip_top:skip_top + n_ctx]
    M = window_counts(tr, blocks, V, ctx, lo, hi)
    words = np.flatnonzero((counts > 0) & (np.arange(V) != eos))
    E = ppmi_svd(M[words], 100, seed=0)
    km = KMeans(k_fine, n_init=4, random_state=0).fit(E)
    fine = np.zeros(V, dtype=np.int64)
    fine[words] = km.labels_
    rest = np.flatnonzero((counts == 0) & (np.arange(V) != eos))
    fine[rest] = km.predict(np.zeros((len(rest), E.shape[1]))) if len(rest) else fine[rest]
    w_tok = np.bincount(fine[words], weights=counts[words], minlength=k_fine)
    km2 = KMeans(k_coarse, n_init=4, random_state=0).fit(km.cluster_centers_, sample_weight=w_tok + 1e-9)
    coarse = km2.labels_[fine]
    fine[eos], coarse[eos] = k_fine, k_coarse
    C = np.stack([coarse, fine], 1)
    return normalise(C, eos), {
        "builder": "topic", "window": [lo, hi], "inside": "same 100-sentence block",
        "context_words": f"train ranks {skip_top + 1} to {skip_top + n_ctx}", "svd_dim": 100,
        "svd_v0_seed": 0, "kmeans_fine": {"k": k_fine, "n_init": 4, "random_state": 0},
        "kmeans_coarse": {"k": k_coarse, "on": "fine centres weighted by train tokens"},
        "zero_count_words": int(len(rest)), "levels": [k_coarse, k_fine]}


# ---- OR-groups of the hierarchical_tm study ----

def orgroups(vocab, eos, path=ORGROUPS_FILE, run=ORGROUPS_RUN):
    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}
    rows = []
    with open(path, encoding="utf8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            if r["role"] == "model" and r["run"] == run:
                rows.append(r)
    best = {}
    all_words, mapped = set(), set()
    for gi, r in enumerate(rows):
        npmi = float(r["npmi"])
        for w in r["words"].split():
            all_words.add(w)
            if w in w2i and w2i[w] != eos:
                mapped.add(w)
                if w not in best or npmi > best[w][0]:
                    best[w] = (npmi, gi)
    C = np.arange(V) + len(rows)
    for w, (_, gi) in best.items():
        C[w2i[w]] = gi
    C[eos] = V + len(rows)
    used = len({g for _, g in best.values()})
    return normalise(C, eos), {
        "builder": "orgroups", "file": os.path.relpath(path, STUDIES), "run": run,
        "groups_in_run": len(rows), "groups_used": used, "words_in_groups": len(all_words),
        "words_mapped_to_ptb": len(mapped), "rule": "each word to the group with the highest "
        "group NPMI that holds it; other words own class",
        "corpus": "WikiText-103 paragraphs (hierarchical_tm data card), vocab_F1024"}
