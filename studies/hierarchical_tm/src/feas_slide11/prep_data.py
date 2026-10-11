"""Build slide-11 paragraph-pair data (same article Y=1 vs different article Y=0), word level.

Paragraph = non-heading line of WikiText-103 raw with >= 30 lowercase alphabetic tokens.
Vocab = top-F content words (stopwords and 'unk' removed) counted over TRAIN-split paragraphs.
Splits: the first N_ART train articles are split by article into train (85%) and heldout (15%);
the 60 WikiText validation articles form a second held-out set ("wval").
Output: pairs_F{F}.npz with X1/X2 index arrays into paragraph matrices, plus vocab json.
"""
import argparse, json, os, re, time
from collections import Counter
import numpy as np

STOP = set("""a about above after again against ain all am an and any are aren aren't as at be because been before being
below between both but by can couldn couldn't d did didn didn't do does doesn doesn't doing don don't down during each few for
from further had hadn hadn't has hasn hasn't have haven haven't having he her here hers herself him himself his how i if in into
is isn isn't it it's its itself just ll m ma me mightn mightn't more most mustn mustn't my myself needn needn't no nor not now o
of off on once only or other our ours ourselves out over own re s same shan shan't she she's should should've shouldn shouldn't
so some such t than that that'll the their theirs them themselves then there these they this those through to too under until up
ve very was wasn wasn't we were weren weren't what when where which while who whom why will with won won't wouldn wouldn't y you
you'd you'll you're you've your yours yourself yourselves unk also would could one two three first second new may
""".split())  # NLTK english stopword list (179) + a few corpus fillers (unk, also, would, could, one, two, three, first, second, new, may)

WORD = re.compile(r"[a-z]+")
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "tm_language_model", "data", "wikitext103", "raw")


def load_articles(split, n_art=None):
    off = np.load(f"{ROOT}/{split}/doc_bytes.npy")
    titles = json.load(open(f"{ROOT}/{split}/titles.json"))
    n = len(titles) if n_art is None else min(n_art, len(titles))
    with open(f"{ROOT}/{split}/text.utf8", "rb") as f:
        raw = f.read(int(off[n]))
    arts = []
    for i in range(n):
        txt = raw[int(off[i]):int(off[i + 1])].decode("utf8")
        paras = []
        for line in txt.split("\n"):
            s = line.strip()
            if not s or (s.startswith("=") and s.endswith("=")):
                continue
            toks = WORD.findall(s.lower())
            if len(toks) >= 30:
                paras.append([t for t in toks if t not in STOP and len(t) > 1])
        arts.append((titles[i], paras))
    return arts


def make_pairs(arts, n_pairs, rng):
    """arts: list of list-of-paragraph-ids. Balanced pairs."""
    multi = [a for a in arts if len(a) >= 2]
    allp = [(ai, p) for ai, a in enumerate(arts) for p in a]
    art_of = {p: ai for ai, a in enumerate(arts) for p in a}
    half = n_pairs // 2
    pos = []
    for _ in range(half):
        a = multi[rng.integers(len(multi))]
        i, j = rng.choice(len(a), 2, replace=False)
        pos.append((a[i], a[j]))
    neg = []
    while len(neg) < half:
        (a1, p1) = allp[rng.integers(len(allp))]
        (a2, p2) = allp[rng.integers(len(allp))]
        if a1 != a2:
            neg.append((p1, p2))
    P = np.array(pos + neg, dtype=np.int64)
    Y = np.array([1] * half + [0] * half, dtype=np.uint32)
    perm = rng.permutation(len(Y))
    return P[perm], Y[perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-art", type=int, default=8000)
    ap.add_argument("--F", type=int, nargs="+", default=[1024, 3072])
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-held", type=int, default=4000)
    ap.add_argument("--n-wval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    tr_arts = load_articles("train", a.n_art)
    va_arts = load_articles("validation")
    is_held = rng.random(len(tr_arts)) < 0.15
    # paragraph table
    paras, groups = [], {"train": [], "held": [], "wval": []}
    for k, (title, ps) in enumerate(tr_arts):
        ids = list(range(len(paras), len(paras) + len(ps)))
        paras.extend(ps)
        groups["held" if is_held[k] else "train"].append(ids)
    for title, ps in va_arts:
        ids = list(range(len(paras), len(paras) + len(ps)))
        paras.extend(ps)
        groups["wval"].append(ids)
    cnt = Counter()
    for ids in groups["train"]:
        for p in ids:
            cnt.update(set(paras[p]))  # document frequency over train paragraphs
    stats = {k: dict(articles=len(v), multi=sum(len(x) >= 2 for x in v), paragraphs=sum(len(x) for x in v)) for k, v in groups.items()}
    stats["mean_content_tokens_per_para"] = float(np.mean([len(p) for p in paras]))
    print(json.dumps(stats))
    pairs = {}
    for split, n in (("train", a.n_train), ("held", a.n_held), ("wval", a.n_wval)):
        pairs[split] = make_pairs(groups[split], n, rng)
    for F in a.F:
        vocab = [w for w, _ in cnt.most_common(F)]
        idx = {w: i for i, w in enumerate(vocab)}
        Xp = np.zeros((len(paras), F), dtype=np.uint8)
        for i, p in enumerate(paras):
            for t in p:
                j = idx.get(t)
                if j is not None:
                    Xp[i, j] = 1
        out = {"Xp": Xp}
        for split, (P, Y) in pairs.items():
            out[f"P_{split}"] = P
            out[f"Y_{split}"] = Y
        np.savez_compressed(f"{a.out}/pairs_F{F}.npz", **out)
        json.dump(vocab, open(f"{a.out}/vocab_F{F}.json", "w"))
        nz = Xp.sum(1)
        print(f"F={F}: active words per paragraph mean {nz.mean():.1f} median {np.median(nz):.0f}; vocab head {vocab[:25]}")
    json.dump(stats, open(f"{a.out}/data_stats.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
