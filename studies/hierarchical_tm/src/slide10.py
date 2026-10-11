"""Slide-10 sentences (E3): generator, data export and Check-2b scorer (design decision; gates.md Check 2b).

Input: slide10_templates.txt in the study folder (not distributed). One sentence per line ('#' lines and blank
lines are skipped): line 1 = the positive sentence of deck slide 10 with its brackets, line 2 = the negated sentence,
lines 3+ = near-misses (one word changed to a word outside its bracket, 'only' kept). Lines 1-2 must match the deck
text (not distributed), and line 2 must be line 1 with EDIT. Until the file exists, dry-run on the test fixture:
    python slide10.py build --templates src/tests/slide10_fixture.txt --out <scratch dir>

Tokens are lowercase words, and EVERY function word is kept ('would', 'not', 'be', 'only', 'for', 'after'). Never use
the stop-word list of feas_slide11/prep_data.py: it removes these words, and then each negated twin would
contain every word of its positive sentence, so no positive-literal rule could tell them apart.

Families (a family is a bracketed template; its sentences are all bracket combinations):
  slide10    line 1: 3x2x3x2x2x4 = 288 sentences. The deck's 216 = the alternatives shared with the negated sentence
             (the 3 time phrases other than 'for a short time'): flag in216. Coverage is reported for both.
  negated    the twin of each slide10 sentence (EDIT: 'would be' -> 'would not be', 'only' -> 'already',
             'for a short time' -> 'after a short time'); same combinations and split as slide10.
  nearmissJ  line 2+J. A line without brackets that is line 1 with one bracket replaced by a phrase outside it is
             expanded over the other brackets, so unseen near-miss combinations exist; any other line is one sentence.
  brokenK    slide10 with bracket K dropped and one neutral FILLERS word in its place (no doubled words).
  otherI     OTHER_FAMILIES (defined in this file): the same frame and length, their own brackets, no word of a slide-10
             bracket. They give the 'different family' pairs on other topics.
Split per family: train = a random minority (--train-share) of the combinations, with every alternative at least once;
the rest is halved into dev (the driver's held pairs) and eval (Check 2b only; never used to choose anything).

build writes to --out (default data/slide10, which needs the templates file):
  pairs_slide10.npz + vocab_slide10.json: the main version, the slide-11 way, in the old probe format htm_driver.py
      reads: Xp uint8 [sentences, F] (bag of words), P_train / P_held [pairs, 2] (rows of Xp), Y_train / Y_held
      (1 = same family). Extra keys: fam, split (0 train, 1 dev, 2 eval), in216. Every family is a 'document'. Pairs
      as in pcp_world.py: positive = family uniform, two different sentences of it; negative = two different families
      uniform, each sentence drawn as in a positive. So one sentence alone says nothing about the label. Check: a
      logistic regression on one sentence, fitted on 20,000 fresh train pairs and tested on 20,000 fresh dev pairs, has a
      95% Wilson CI that includes 0.50 (as Check 2a); the same on the exported pairs is reported for information.
  text/: the fallback with direct labels, in the format of text_driver.py (--data <out>/text): {train,dev,eval}.npz
      with indices, indptr (CSR rows of word ids), y (1 = slide10 sentence; 0 = negated twin, near-miss or broken
      sentence) and row (the sentence's row in pairs_slide10.npz and slide10_meta.json), plus vocab.json. Train and dev
      are balanced (half slide10, the negative kinds in equal shares, drawn with replacement); eval has every eval
      sentence once. text_driver.py loads only train and dev.
  slide10_meta.json: families (kind, template, brackets, fixed words, split sizes), every sentence's text, word
      frequencies (for s: a word must have frequency above 1/s among the examples a clause learns from), the checks.
check2b RUN_DIR: Check 2b from the include_bits.npz of an htm_driver.py run (pairs) or a text_driver.py run (one
sentence; the number of sides is read from the include bits). The best positive class-1 rule is chosen on TRAIN
sentences (slide10 coverage minus its highest firing rate on negated, near-miss and broken sentences) and scored on
EVAL sentences (--split dev: DEV sentences, for choosing settings). A pair rule covers a sentence if both of its halves
accept it, and fires on a negative if either half accepts it. A bracket is recovered if a counted OR-group (no empty
option, at least 2 distinct words) has Jaccard >= 0.8 with it after the family's fixed words (those in every slide10
sentence) are removed from both; the strict Jaccard without removal is reported too. The rule is printed with the
words shared by all options of an OR-group first, then its distinct alternatives.
"""
import argparse, itertools, json, os, re, sys
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(SRC)
TEMPLATES = os.path.join(HOME, "slide10_templates.txt")
FIXTURE = os.path.join(SRC, "tests", "slide10_fixture.txt")
DECK = os.path.join(HOME, "deck.txt")
EDIT = (("would be", "would not be"), ("only", "already"), ("for a short time", "after a short time"))
FILLERS = ("then", "also", "still", "indeed")  # 4, so no bracket has more alternatives than slide 10's largest
NEG_KINDS = ("negated", "nearmiss", "broken")
OTHER_FAMILIES = (
    "[Writing/Posting/Reporting] that the [soup/stew] was [delicious/tasty/simply superb] would be an [insult/injustice] and it [stayed warm/remained hot] only [until dessert / during the starter / briefly / around noon].",
    "[Announcing/Shouting/Declaring] that the [team/squad] was [unbeatable/dominant/truly brilliant] would be an [error/illusion] and it [thrilled the crowd/pleased the fans] only [before halftime / early on / right from kickoff].",
    "[Suggesting/Arguing/Insisting] that the [room/suite] was [spotless/cozy/very spacious] would be an [overreach/embellishment] and it [felt quiet/seemed calm] only [past midnight / occasionally / midweek].",
    "[Noting/Mentioning/Remarking] that the [band/orchestra] was [flawless/stunning/wonderfully tight] would be an [inaccuracy/oversimplification] and it [moved the audience/excited the listeners] only [near the end / sporadically / toward the finale / once or twice].",
    "[Believing/Assuming/Thinking] that the [novel/book] was [gripping/captivating/deeply moving] would be an [untruth/inflation] and it [engaged the reader/absorbed us] only [through chapter two / here and there / sometimes].",
    "[Promising/Swearing/Guaranteeing] that the [train/ferry] was [punctual/fast/remarkably smooth] would be an [overpromise/misrepresentation] and it [left promptly/reached the station] only [last summer / every other week / rarely].",
    "[Advertising/Boasting/Implying] that the [phone/tablet] was [durable/sturdy/extremely robust] would be an [overclaim/distortion] and it [lasted the day/survived the drop] only [under light use / with gentle care / by luck].",
    "[Grumbling/Complaining/Whispering] that the [lecture/seminar] was [tedious/monotonous/painfully slow] would be an [unkindness/overreaction] and it [lost the students/bored everyone] only [whenever slides changed / beyond the first hour / now and again].",
)


def words(text):
    """Lowercase word tokens; every function word is kept."""
    return re.findall(r"[a-z]+", text.lower())


def edit(text):
    """The slide-10 negation edit, on a template line or a sentence (lowercased)."""
    text = text.lower()
    for a, b in EDIT:
        text = re.sub(rf"\b{a}\b", b, text)
    return text


def parse(line):
    """Template -> (slots, is_bracket). A slot is a list of alternatives (token tuples); fixed text has one."""
    slots, br = [], []
    for i, part in enumerate(re.split(r"\[([^\]]*)\]", line)):
        alts = [tuple(words(a)) for a in part.split("/")] if i % 2 else [tuple(words(part))]
        if i % 2 or alts[0]:
            slots.append(alts)
            br.append(bool(i % 2))
    return slots, br


def expand(slots, br):
    """All sentences of a template: [(combo = the chosen alternative of each bracket, tokens)]."""
    out = []
    for pick in itertools.product(*[range(len(s)) for s in slots]):
        out.append((tuple(i for i, b in zip(pick, br) if b), [w for s, i in zip(slots, pick) for w in s[i]]))
    return out


def match_near_miss(slots, br, toks):
    """(k, phrase) if toks is the template with exactly bracket slot k replaced by a phrase outside it, else None."""
    alt = lambda s: "(?:" + "|".join(re.escape(" ".join(a)) for a in s) + ")"
    for k in [i for i, b in enumerate(br) if b]:
        m = re.fullmatch(" ".join("(.+)" if i == k else alt(s) for i, s in enumerate(slots)), " ".join(toks))
        if m and tuple(m.group(1).split()) not in slots[k]:
            return k, tuple(m.group(1).split())
    return None


def read_templates(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist yet. It is not distributed: line 1 = the positive "
                         "sentence of deck slide 10 with its brackets, line 2 = the negated sentence, then 5 near-misses. "
                         f"Dry run: --templates {FIXTURE} --out <scratch dir>")
    lines = [l.strip() for l in open(path, encoding="utf-8-sig") if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 3:
        raise SystemExit(f"{path}: needs the positive sentence, the negated sentence and at least one near-miss")
    return lines


def deck_sentences():
    """The two bracketed lines of slide 10 in the deck text, or None if the deck file is missing."""
    if not os.path.exists(DECK):
        return None
    text = open(DECK).read().split("slide10.xml ===")[1].split("===")[0]
    return [l.strip() for l in text.splitlines() if "[" in l][:2]


def families(lines):
    """[(name, kind, template text, slots, is_bracket)] (see the module docstring)."""
    pos, neg = lines[0], lines[1]
    S, br = parse(pos)
    if parse(edit(pos)) != parse(neg):
        raise SystemExit("line 2 must be line 1 with the slide-10 edit: " + "; ".join(f"'{a}' -> '{b}'" for a, b in EDIT))
    deck = deck_sentences()
    if deck and [parse(l) for l in deck] != [parse(pos), parse(neg)]:
        raise SystemExit(f"lines 1-2 differ from deck slide 10 ({DECK}); copy them exactly, with their brackets")
    fams = [("slide10", "slide10", pos, S, br), ("negated", "negated", neg, *parse(neg))]
    for j, line in enumerate(lines[2:], 1):
        m = None if "[" in line else match_near_miss(S, br, words(line))
        if "only" not in words(line):
            print(f"WARNING near-miss {j} does not keep 'only': {line}")
        if "[" in line:
            fams.append((f"nearmiss{j}", "nearmiss", line, *parse(line)))
        elif m:
            k, phrase = m
            fams.append((f"nearmiss{j}", "nearmiss", line, S[:k] + [[phrase]] + S[k + 1:], br[:k] + [False] + br[k + 1:]))
        else:
            print(f"WARNING near-miss {j} is not line 1 with one bracket replaced, so it stays one sentence: {line}")
            fams.append((f"nearmiss{j}", "nearmiss", line, [[tuple(words(line))]], [False]))
    for n, k in enumerate([i for i, b in enumerate(br) if b], 1):
        fams.append((f"broken{n}", "broken", f"line 1, bracket {n} -> filler", S[:k] + [[(f,) for f in FILLERS]] + S[k + 1:], br))
    return fams + [(f"other{i}", "other", t, *parse(t)) for i, t in enumerate(OTHER_FAMILIES, 1)]


def split_combos(combos, rng, share):
    """0 train / 1 dev / 2 eval per combination. Train: a random minority with every alternative at least once."""
    c = np.array(combos, dtype=np.int64).reshape(len(combos), -1)
    n_tr = max(1, int(round(share * len(c))))
    for _ in range(1000):
        o = rng.permutation(len(c))
        if all(len(np.unique(c[o[:n_tr], j])) == len(np.unique(c[:, j])) for j in range(c.shape[1])):
            break
    else:
        raise SystemExit("no train split covers every alternative; raise --train-share")
    sp = np.ones(len(c), np.int8)
    sp[o[:n_tr]] = 0
    sp[o[n_tr:][1::2]] = 2
    return sp


def sentence_table(fams, rng, share):
    """Every sentence of every family: tokens, fam, split, in216."""
    toks, fam, split, in216, s10 = [], [], [], [], {}
    pb, nb = ([s for s, b in zip(f[3], f[4]) if b] for f in fams[:2])
    shared = [[a in n for a in p] for p, n in zip(pb, nb)]  # slide10 alternatives that are also in the negated sentence
    for f, (_, kind, _, slots, br) in enumerate(fams):
        ex = expand(slots, br)
        sp = [s10[c] for c, _ in ex] if kind == "negated" else split_combos([c for c, _ in ex], rng, share)
        if kind == "slide10":
            s10 = {c: s for (c, _), s in zip(ex, sp)}
        for (c, t), s in zip(ex, sp):
            toks.append(t), fam.append(f), split.append(s)
            in216.append(kind in ("slide10", "negated") and all(shared[j][i] for j, i in enumerate(c)))
    return toks, np.array(fam), np.array(split, np.int8), np.array(in216)


def bits(toks, vocab):
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(toks), len(vocab)), np.uint8)
    for i, t in enumerate(toks):
        X[i, [idx[w] for w in t]] = 1
    return X


def superset_of(X, rows, of):
    """bool [len(rows)]: the sentence contains every word of at least one sentence in rows `of`."""
    hits = X[rows].astype(np.int32) @ X[of].T.astype(np.int32)
    return (hits == X[of].sum(1, dtype=np.int32)[None, :]).any(1)


def sample_pairs(groups, n, rng):
    """n pairs (pcp_world.py sampling): half positive (one family, two different sentences), half negative (two
    different families, one uniform sentence each); random side order; shuffled. groups: rows per family."""
    G, h = len(groups), n // 2
    pos = np.array([rng.choice(groups[t], 2, replace=False) for t in rng.integers(G, size=h)])
    t1 = rng.integers(G, size=h)
    t2 = (t1 + rng.integers(1, G, size=h)) % G
    neg = np.c_[[rng.choice(groups[t]) for t in t1], [rng.choice(groups[t]) for t in t2]]
    P, Y = np.vstack([pos, neg]), np.r_[np.ones(h, np.uint32), np.zeros(h, np.uint32)]
    swap = rng.random(2 * h) < 0.5
    P[swap] = P[swap][:, ::-1]
    o = rng.permutation(2 * h)
    return P[o], Y[o]


def one_sentence_lr(X, P_fit, Y_fit, P_ev, Y_ev):
    """Logistic regression on one sentence (side A) of each pair; its 95% Wilson CI must include 0.50."""
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, SRC)
    from data_checks import wilson
    acc = float((LogisticRegression(max_iter=2000).fit(X[P_fit[:, 0]], Y_fit).predict(X[P_ev[:, 0]]) == Y_ev).mean())
    ci = wilson(acc, len(Y_ev))
    return {"acc": round(acc, 4), "ci": ci, "ci_has_050": ci[0] <= 0.5 <= ci[1], "n_fit": len(Y_fit), "n_eval": len(Y_ev)}


def build(a):
    data_dir = os.path.join(HOME, "data", "slide10")
    if os.path.abspath(a.out) == data_dir and os.path.abspath(a.templates) != TEMPLATES:
        raise SystemExit(f"{data_dir} is only built from {TEMPLATES}")
    rng = np.random.default_rng([20261002, a.seed])
    fams = families(read_templates(a.templates))
    toks, fam, split, in216 = sentence_table(fams, rng, a.train_share)
    kind = np.array([f[1] for f in fams])[fam]
    vocab = sorted({w for t in toks for w in t})
    X = bits(toks, vocab)
    text = [" ".join(t) for t in toks]
    s10 = np.flatnonzero(kind == "slide10")
    for k in NEG_KINDS + ("other",):  # a positive-literal rule can reject a sentence only if it lacks a slide10 word
        r = np.flatnonzero(kind == k)
        bad = r[superset_of(X, r, s10)]
        if len(bad):
            raise SystemExit(f"{len(bad)} {k} sentence(s) contain every word of a slide10 sentence, so no positive-literal "
                             f"rule can reject them, e.g. '{text[bad[0]]}'. Change the template line.")
    if len(set(text)) != len(text):
        raise SystemExit("a sentence appears twice (in two families)")
    groups = lambda sp: [r for f in range(len(fams)) if len(r := np.flatnonzero((fam == f) & (split == sp))) >= 2]
    P_tr, Y_tr = sample_pairs(groups(0), a.n_train, rng)
    P_dv, Y_dv = sample_pairs(groups(1), a.n_dev, rng)
    lr = {"fresh": one_sentence_lr(X, *sample_pairs(groups(0), 20000, rng), *sample_pairs(groups(1), 20000, rng)),
          "exported": one_sentence_lr(X, P_tr, Y_tr, P_dv, Y_dv)}  # the check is 'fresh' (20,000 + 20,000 pairs, as Check 2a)
    from sklearn.metrics import roc_auc_score
    A_, B_ = X[P_dv[:, 0]].astype(bool), X[P_dv[:, 1]].astype(bool)
    overlap_auc = float(roc_auc_score(Y_dv, (A_ & B_).sum(1) / (A_ | B_).sum(1)))  # word-overlap (Jaccard) baseline
    # word frequency per bracket word: within its family's train sentences, and on one side of the positive train pairs
    side = X[P_tr[Y_tr == 1, 0]].mean(0)
    freq = {}
    for f, (name, kind_f, tpl, slots, br) in enumerate(fams):
        fixed = set.intersection(*[set(toks[i]) for i in np.flatnonzero(fam == f)])
        bw = sorted({w for s, b in zip(slots, br) if b for alt in s for w in alt} - fixed)
        own = X[(fam == f) & (split == 0)].mean(0)
        freq[name] = {w: [round(float(own[vocab.index(w)]), 4), round(float(side[vocab.index(w)]), 4)] for w in bw}
        fams[f] = dict(name=name, kind=kind_f, template=tpl, fixed=sorted(fixed),
                       brackets=[[" ".join(x) for x in s] for s, b in zip(slots, br) if b],
                       n=int((fam == f).sum()), n_split=[int(((fam == f) & (split == s)).sum()) for s in range(3)])
    fams[0]["brackets_216"] = [[x for x in b if x in nb] for b, nb in zip(fams[0]["brackets"], fams[1]["brackets"])]
    rarest = lambda names, j: min(((p[j], w, n) for n in names for w, p in freq[n].items()), default=None)
    near = [f["name"] for f in fams if f["kind"] in ("slide10",) + NEG_KINDS]
    summary = {"F": len(vocab), "n_sentences": len(toks), "n_families": len(fams),
               "count_slide10": int((kind == "slide10").sum()), "count_slide10_216": int(in216[kind == "slide10"].sum()),
               "count_negated": int((kind == "negated").sum()), "count_nearmiss": int((kind == "nearmiss").sum()),
               "count_broken": int((kind == "broken").sum()), "count_other": int((kind == "other").sum()),
               "slide10_split": fams[0]["n_split"], "n_train_pairs": len(Y_tr), "n_held_pairs": len(Y_dv),
               "one_sentence_lr": lr["fresh"], "one_sentence_lr_exported_pairs": lr["exported"],
               "word_overlap_auc_held": round(overlap_auc, 4),
               "rarest_slide10_word_in_slide10_train": rarest(["slide10"], 0),
               "rarest_slide10_word_on_positive_sides": rarest(["slide10"], 1),
               "rarest_near_family_word_on_positive_sides": rarest(near, 1),
               "rarest_word_on_positive_sides": rarest([f["name"] for f in fams], 1)}
    tdir = os.path.join(a.out, "text")
    os.makedirs(tdir, exist_ok=True)
    json.dump(vocab, open(os.path.join(tdir, "vocab.json"), "w"))
    for name, sp, n in (("train", 0, a.n_train), ("dev", 1, a.n_dev), ("eval", 2, 0)):
        rows = {k: np.flatnonzero((kind == k) & (split == sp)) for k in ("slide10",) + NEG_KINDS}
        if n:
            neg = [k for k in NEG_KINDS if len(rows[k])]  # a kind with no sentence in this split is skipped
            m = [(n - n // 2) // len(neg)] * len(neg)
            m[-1] += n - n // 2 - sum(m)
            parts = [rng.choice(rows["slide10"], n // 2)] + [rng.choice(rows[k], j) for k, j in zip(neg, m)]
        else:
            parts = list(rows.values())
        r = rng.permutation(np.concatenate(parts))
        np.savez_compressed(os.path.join(tdir, f"{name}.npz"), indices=X[r].nonzero()[1].astype(np.int32),
                            indptr=np.r_[0, np.cumsum(X[r].sum(1))].astype(np.int32),
                            y=(kind[r] == "slide10").astype(np.int64), row=r.astype(np.int64))
    np.savez_compressed(os.path.join(a.out, "pairs_slide10.npz"), Xp=X, P_train=P_tr, Y_train=Y_tr, P_held=P_dv,
                        Y_held=Y_dv, fam=fam, split=split, in216=in216)
    json.dump(vocab, open(os.path.join(a.out, "vocab_slide10.json"), "w"))
    json.dump({"templates": os.path.abspath(a.templates), "args": vars(a), "summary": summary, "families": fams,
               "frequencies": freq, "sentences": text}, open(os.path.join(a.out, "slide10_meta.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summary))
    print(f"DRIVER ARGS: --data {os.path.abspath(os.path.join(a.out, 'pairs_slide10.npz'))} --F {len(vocab)}")
    return summary


# ---------------------------------------------------------------- Check 2b
def accept(inc, X, A, K, sides):
    """bool [n, rules, sides]: half p of a rule accepts a sentence iff each of its K OR-groups has an option whose
    included words are all in the sentence (an empty option is always true). inc: [rules, sides*K*A, F]."""
    C, L, F = inc.shape
    hits = X.astype(np.float32) @ inc.reshape(C * L, F).T.astype(np.float32)
    return (hits == inc.sum(-1).reshape(1, -1)).reshape(len(X), C, sides, K, A).any(-1).all(-1)


def bracket_jaccard(inc_c, vocab, brackets, fixed, A, K, sides):
    """Per true bracket (a word set): the best Jaccard with a counted OR-group of one rule [sides*K*A, F], fixed words
    removed from both sets; and the strict best Jaccard without that removal."""
    counted = []
    for j in range(sides * K):
        opts = inc_c[j * A:(j + 1) * A]
        w = {vocab[i] for i in np.flatnonzero(opts.any(0))}
        if opts.any(1).all() and len(w) >= 2:
            counted.append(w)
    jac = lambda x, y: len(x & y) / max(1, len(x | y))
    return [(max([jac(w - fixed, W - fixed) for w in counted], default=0.0), max([jac(w, W) for w in counted], default=0.0))
            for W in brackets]


def check2b(run_dir, data=None, split_name="eval", thr=0.8):
    sys.path.insert(0, SRC)
    import htm_driver as hd
    z = np.load(os.path.join(run_dir, "include_bits.npz"))
    F, A, K = int(z["F"]), int(z["A"]), int(z["K"])
    inc = np.unpackbits(z["class1"], axis=-1, count=F).astype(bool)[0::2]  # the +1 (positive) rules of class 1
    sides = inc.shape[1] // (A * K)  # 2: htm_driver.py pairs; 1: text_driver.py single sentences
    if data is None:
        data = next(json.loads(l.split(" ", 1)[1]) for l in open(os.path.join(run_dir, "log.txt")) if l.startswith("CONFIG "))["data"]
    if os.path.isdir(data):  # a text_driver.py run: its --data is <out>/text
        data = os.path.join(os.path.dirname(os.path.normpath(data)), "pairs_slide10.npz")
    d, meta = np.load(data), json.load(open(os.path.join(os.path.dirname(data), "slide10_meta.json")))
    vocab = json.load(open(data.replace("pairs_", "vocab_").replace(".npz", ".json")))
    X, fam, split, in216 = d["Xp"], d["fam"], d["split"], d["in216"]
    kind = np.array([f["kind"] for f in meta["families"]])[fam]
    acc = accept(inc, X, A, K, sides)
    cover, fire = acc.all(-1), acc.any(-1)
    m = lambda k, sp: (kind == k) & (split == sp)
    rate = lambda v, sel: v[sel].mean(0) if sel.any() else np.full(v.shape[1:], np.nan)  # nan: no such sentence
    score = rate(cover, m("slide10", 0)) - np.nanmax([rate(fire, m(k, 0)) for k in NEG_KINDS], axis=0)
    b, e, sn = int(np.argmax(score)), {"dev": 1, "eval": 2}[split_name], split_name
    s10 = meta["families"][0]
    fixed = set(s10["fixed"])
    out = {"run": os.path.basename(os.path.normpath(run_dir)), "best_rule": 2 * b, "selection_score_train": float(score[b]),
           "coverage_train": float(rate(cover, m("slide10", 0))[b]),
           f"coverage_{sn}_288": float(rate(cover, m("slide10", e))[b]),
           f"coverage_{sn}_216": float(rate(cover, m("slide10", e) & in216)[b]),
           **{f"fires_{sn}_{k}": float(rate(fire, m(k, e))[b]) for k in NEG_KINDS + ("other",)},
           **{f"n_{sn}_{k}": int(m(k, e).sum()) for k in ("slide10",) + NEG_KINDS}}
    out[f"fires_{sn}_max"] = float(np.nanmax([out[f"fires_{sn}_{k}"] for k in NEG_KINDS]))
    for view, key in (("288", "brackets"), ("216", "brackets_216")):
        true = [set(" ".join(alts).split()) for alts in s10[key]]
        jac = bracket_jaccard(inc[b], vocab, true, fixed, A, K, sides)
        out[f"bracket_jaccard_{view}"] = [round(j, 3) for j, _ in jac]
        out[f"bracket_jaccard_strict_{view}"] = [round(j, 3) for _, j in jac]
        out[f"brackets_recovered_{view}"] = sum(j >= thr for j, _ in jac)
        out[f"check2b_{sn}_pass_{view}"] = bool(out[f"coverage_{sn}_{view}"] >= 0.9 and out[f"fires_{sn}_max"] <= 0.05
                                              and out[f"brackets_recovered_{view}"] >= 4)
    print("CHECK2B", json.dumps(out))
    for p in range(sides):  # each OR-group: the words in all of its options once, then its distinct alternatives
        for k in range(K):
            o = inc[b, A * (k + K * p):A * (k + K * p + 1)]
            common = o.all(0) if o.any(1).all() else np.zeros(F, bool)
            alts = "[" + " | ".join(dict.fromkeys(hd.option_text(x & ~common, vocab, m=12) or "-" for x in o if x.any())) + "]"
            line = (hd.option_text(common, vocab, m=12) + " & " + alts) if common.any() else alts
            if not o.any(1).all():
                line = f"ALWAYS-TRUE ({int((~o.any(1)).sum())} of {A} options empty; others {alts})"
            print(f"   {'AB'[p] if sides == 2 else 'S'}{k + 1}: {line}")
    if os.environ.get("TMLM_RUN_DIR"):  # nan (no sentence of that kind) is written as null
        json.dump({k: {"value": None if v != v else v, "label": "measured"} for k, v in out.items()},
                  open(os.path.join(os.environ["TMLM_RUN_DIR"], "metrics.json"), "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description="slide-10 sentences (E3); see the module docstring")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--templates", default=TEMPLATES)
    b.add_argument("--out", default=os.path.join(HOME, "data", "slide10"))
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--train-share", type=float, default=0.3, help="share of each family's combinations used in training")
    b.add_argument("--n-train", type=int, default=10000)
    b.add_argument("--n-dev", type=int, default=10000, help="held (dev) pairs; Check 1b used at least 10,000")
    c = sub.add_parser("check2b")
    c.add_argument("run_dir")
    c.add_argument("--data", default=None, help="pairs_slide10.npz (default: the run's --data)")
    c.add_argument("--split", default="eval", choices=["dev", "eval"], help="dev for choosing settings; eval for the gate")
    a = ap.parse_args()
    if a.cmd == "build":
        s = build(a)
        if os.environ.get("TMLM_RUN_DIR"):
            flat = {k: v for k, v in s.items() if not isinstance(v, (dict, tuple))}
            flat.update({f"{k}_{q}": s[k][q] for k in ("one_sentence_lr", "one_sentence_lr_exported_pairs") for q in ("acc", "ci", "ci_has_050")})
            json.dump({k: {"value": v, "label": "measured"} for k, v in flat.items()},
                      open(os.path.join(os.environ["TMLM_RUN_DIR"], "metrics.json"), "w"), indent=1)
    else:
        check2b(a.run_dir, a.data, a.split)


if __name__ == "__main__":
    main()
