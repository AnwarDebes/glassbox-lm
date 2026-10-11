"""Build the AG News bag-of-words data for E5a (design decision 'Real words (E5)'; gates.md Check 2c).

Source: ONLY the AG News train parquet (fancyzhx/ag_news, revision eb185aa, 120,000 news texts, 4 topics with
30,000 each), read from the old project's Hugging Face cache (read-only). The dataset's own test file is never opened.
Cleaning: HTML entities decoded ('#39;' is an apostrophe, 'quot;' a quote), HTML tags and web addresses removed.
Words: lowercase alphabetic tokens, minus the NLTK English stop-word list (179 words: build_pairs.STOP without its
Wikipedia fillers) and one-letter tokens. A text is the SET of its words (presence only).
Splits: one random permutation of the 120,000 rows (seed 20261002): train 60,000 / dev 10,000 / final test 10,000
(used once, Day 5) / NPMI reference 40,000 (texts in no split, used only for NPMI coherence).
Vocabulary: the top 1,024 words by train document frequency (ties: alphabetical).
Writes data/agnews/: {train,dev,test,npmi_ref}.npz (CSR parts of X [n, 1024] 0/1, y, row = row in the parquet),
vocab.json, build_stats.json (sizes, per-class document frequencies, and which words can enter a rule at which s).
CPU only; needs pyarrow (.venv-tm).
"""
import hashlib, html, json, os, re, sys, time
from collections import Counter

import numpy as np
import scipy.sparse as sp

SRC = os.path.dirname(os.path.abspath(__file__))
PARQUET = (os.path.join(os.path.dirname(os.path.dirname(SRC)), "tm_language_model") + "/data/hf_cache/hub/datasets--fancyzhx--ag_news/snapshots/"
           "eb185aade064a813bc0b7f42de02595523103ca4/data/train-00000-of-00001.parquet")
OUT = os.path.join(os.path.dirname(SRC), "data", "agnews")
NAMES = ["World", "Sports", "Business", "Sci/Tech"]
SIZES = {"train": 60000, "dev": 10000, "test": 10000}  # the remaining 40,000 rows are the NPMI reference
F, SEED = 1024, 20261002
S_GRID = [21.1, 50, 60, 100, 200, 230, 400, 800]
WORD, TAG, URL = re.compile(r"[a-z]+"), re.compile(r"<[^>]*>"), re.compile(r"(https?://|www\.)\S+", re.I)
DEC, HEX = re.compile(r"&?#(\d{1,7});"), re.compile(r"&?#x([0-9a-f]{1,6});", re.I)
BARE = re.compile(r"(?<![&\w])(quot|amp|lt|gt|nbsp|apos);")  # entities that lost their '&', e.g. 'quot;' 


def stop_words():
    sys.path.insert(0, SRC)
    from build_pairs import STOP  # NLTK English list + 11 Wikipedia fillers
    return STOP - set("unk also would could one two three first second new may".split())


def _chr(n):
    return chr(n) if 0 < n <= 0x10FFFF else " "


def words(text, stop):
    s = HEX.sub(lambda m: _chr(int(m.group(1), 16)), DEC.sub(lambda m: _chr(int(m.group(1))), text))
    s = URL.sub(" ", TAG.sub(" ", html.unescape(html.unescape(BARE.sub(r"&\1;", s)))))
    return {t for t in WORD.findall(s.lower()) if len(t) > 1 and t not in stop}


def pct(v, qs=(0, 10, 25, 50, 75, 90, 100)):
    return {f"p{q}": round(float(np.percentile(v, q)), 5) for q in qs}


def main():
    import pyarrow.parquet as pq
    t0 = time.time()
    raw = open(PARQUET, "rb").read()
    tab = pq.read_table(PARQUET, columns=["text", "label"])
    texts, y = tab.column("text").to_pylist(), np.asarray(tab.column("label").to_pylist(), np.int64)
    stop = stop_words()
    docs = [words(s, stop) for s in texts]
    perm = np.random.default_rng(SEED).permutation(len(texts))
    cut = np.cumsum([0] + list(SIZES.values()))
    split = {k: perm[cut[i]:cut[i + 1]] for i, k in enumerate(SIZES)}
    split["npmi_ref"] = perm[cut[-1]:]
    assert sum(map(len, split.values())) == len(texts) == len(np.unique(np.concatenate(list(split.values()))))
    df = Counter(w for i in split["train"] for w in docs[i])
    vocab = sorted(df, key=lambda w: (-df[w], w))[:F]
    wid = {w: j for j, w in enumerate(vocab)}
    os.makedirs(OUT, exist_ok=True)
    st = {"source": PARQUET, "parquet_sha256": hashlib.sha256(raw).hexdigest(), "seed": SEED, "F": F,
          "class_names": NAMES, "n_stop_words": len(stop), "splits": {}}
    for k, rows in split.items():
        ind = [sorted(wid[w] for w in docs[i] if w in wid) for i in rows]
        lens = np.array([len(r) for r in ind])
        X = sp.csr_matrix((np.ones(lens.sum(), np.uint8), np.concatenate(ind).astype(np.int32),
                           np.r_[0, np.cumsum(lens)]), shape=(len(rows), F))
        np.savez_compressed(os.path.join(OUT, f"{k}.npz"), indices=X.indices, indptr=X.indptr, y=y[rows], row=rows)
        st["splits"][k] = {"n": len(rows), "per_class": np.bincount(y[rows], minlength=4).tolist(),
                           "vocab_words_per_text": pct(lens, (0, 10, 50, 90, 100)), "mean_vocab_words": round(float(lens.mean()), 2),
                           "empty_texts": int((lens == 0).sum()),
                           "mean_words_before_vocab_cut": round(float(np.mean([len(docs[i]) for i in rows])), 2)}
        if k == "train":
            Xtr, ytr = X, y[rows]
    n_c = np.bincount(ytr, minlength=4)
    p_c = np.vstack([np.asarray(Xtr[ytr == c].sum(0)).ravel() / n_c[c] for c in range(4)])  # [4, F] share of class-c texts
    p_all, p_max = np.asarray(Xtr.sum(0)).ravel() / Xtr.shape[0], p_c.max(0)
    st["df_train_share"] = pct(p_all)
    st["df_class_share"] = {NAMES[c]: pct(p_c[c]) for c in range(4)}  # per topic: share of its train texts with the word
    st["df_top_class_share"] = pct(p_max)  # p = the word's document frequency in the topic where it is most frequent
    st["s_needed_top_class"] = pct(1 / p_max)  # a word can enter a rule only if s > 1/p (TM book ch. 2)
    st["can_enter_at_s"] = {str(s): {"share_of_vocab_top_class": round(float((p_max > 1 / s).mean()), 3),
                                     "words_per_class": {NAMES[c]: int((p_c[c] > 1 / s).sum()) for c in range(4)}}
                            for s in S_GRID}
    st["top_words_per_class"] = {NAMES[c]: [f"{vocab[j]} {p_c[c, j]:.3f}" for j in np.argsort(-p_c[c])[:15]] for c in range(4)}
    st["top_words_overall"] = [f"{vocab[j]} {p_all[j]:.3f}" for j in range(30)]
    st["last_vocab_words"] = [f"{vocab[j]} {p_all[j]:.4f}" for j in range(F - 5, F)]
    st["wall_s"] = round(time.time() - t0, 1)
    json.dump(vocab, open(os.path.join(OUT, "vocab.json"), "w"))
    json.dump(st, open(os.path.join(OUT, "build_stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))
    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        keep = ["splits", "df_train_share", "df_class_share", "df_top_class_share", "s_needed_top_class", "can_enter_at_s",
                "parquet_sha256"]
        json.dump({k: {"value": st[k], "label": "measured"} for k in keep}, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
