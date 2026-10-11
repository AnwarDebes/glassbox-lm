"""Check 4a / 4b (gates.md): the Wikipedia word groups and the synonym test (E4s). CPU only.

Usage: python wiki_groups.py --runs DIR [DIR ...] [--shuffled DIR [DIR ...]] --data sa|np|ns_all
                             [--draws 1000] [--npmi-text paragraph|article]
  DIR has include_bits.npz (htm_driver.py, or score_saved.py at the chosen pass P). For Check 4b part 1 the model
  dirs must be score_saved.py dev outputs (metrics.json with the article bootstrap, and scores_dev.npz).
  Prints a plain report; writes metrics.json and groups.tsv (every counted group with its scores) to $TMLM_RUN_DIR
  (or --out). It refuses runs whose training CONFIG has other data, another label role, other settings or
  another pass (the pass is pass_P of a score_saved.py output, else the last pass in the training log).
Counted OR-groups (gates.md): the OR-groups of the class-1 positive clauses (even indices) that are not always-true
  (no empty option) and have at least 2 distinct words; a group's words = the union of its options' words (repeated
  words removed). Leaf index = o + A*(k + K*p): option o of OR-group k on paragraph p (0 = A, 1 = B).
NPMI (npmi_groups.npmi_matrix): the reference text is the 2,847 'npmi_ref' articles in data/splits.json (in no
  split and in no pair). One text = one whole article (2,847; the default and the pre-registered unit, registered on
  2 Oct 2026, 07:00 CEST (docs/registrations.md): the design defines NPMI as "how often a group's words appear together in new articles"), or, with
  --npmi-text paragraph, one paragraph (70,118; a secondary view only). Words count as a set.
WordNet score (gates.md): the share of a group's word pairs that are synonyms (share a synset) or hypernym-close (one
  is a hypernym of the other within 2 steps, or they share a direct hypernym). WordNet 3.0; nouns, verbs and
  adjectives (with satellites). Our readings: a word is looked up with WordNet's own base-form rule (began -> begin);
  any pair of senses counts; instance hypernyms count as hypernyms (NLTK's path rule). A word with no noun, verb or
  adjective sense is missing, and pairs with it are skipped: a group is scored on its covered words (at least 2).
  Its random groups are drawn from covered words of the same frequency bins, so they have as many scored pairs.
  Diagnostic, not a gate: 'wordnet_no_forms' counts two forms of one word (film, films) as not related.
Comparisons (npmi_groups.compare and compare_two: the frozen group-bootstrap gate plus the calibrated fields), per
  run and pooled over runs: groups vs frequency-matched random groups (the NPMI frequency bins for every measure),
  and groups vs the shuffled-label model's groups (pooled over its seeds). Each is done with all counted groups and
  without the groups that contain a skewed word (began, big, since, called; also left out of the random draws).
Check 4a (sa; also applied to np): NPMI with all counted groups, pooled over seeds (per-seed results are shown).
Check 4b (ns_all), per seed: part 1 = dev AUC CI above 0.5, and the partner-shuffle and length-only CIs (accuracy
  and AUC, article bootstrap) include 0.50; part 2 = WordNet and NPMI beat random and shuffled-label groups,
  without the skewed-word groups (decision of 1 Oct 2026, 20:10 CEST). It passes on 2 of 3 seeds.
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import npmi_groups as ng  # noqa: E402
from pairdata import DATA, ids_to_csr, load_pairs  # noqa: E402

NLTK_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tm_language_model", "data", "concept_eval", "nltk_data")
SKEWED = ("began", "big", "since", "called")
SAME = ("model", "clauses", "T", "s", "A", "K", "F", "data", "n_train", "batch", "label_mode")
NOTE = ("NOTE on the gate: the frozen group bootstrap is too narrow with few groups. In 1,000 no-effect toy cases it "
        "wrongly said 'beats random' 20% of the time with 6 groups and 8.7% with 30, against the intended 2.5% (two "
        "sets of 12 groups: 4.3%). The calibrated test did so 1.1%, 3.5% and 1.6%. Under the frozen settings "
        "the bootstrap decides and the calibrated result is printed next to it.")


def config_of(d):
    """(training CONFIG, last pass in its log) of a run dir; for a score_saved.py output dir, of its source run."""
    logs, m = [os.path.join(d, "log.txt")], os.path.join(d, "metrics.json")
    if os.path.exists(m):
        src = json.load(open(m)).get("run", {}).get("value")
        if src:
            logs.append(os.path.join(os.path.dirname(os.path.abspath(d)), src, "log.txt"))
    for f in filter(os.path.exists, logs):
        cfg = last = None
        for line in open(f):
            if line.startswith(("CONFIG ", "EPOCH ", "PASS ")):
                j = json.loads(line.split(" ", 1)[1])
                cfg, last = (j, last) if line.startswith("CONFIG") else (cfg, j["pass"])
        if cfg:
            return cfg, last
    return None, None


def counted_groups(inc, A, K):
    """Counted OR-groups of the class-1 positive clauses (gates.md) and the counts of dropped groups."""
    out, st = [], dict(or_groups=0, always_true=0, all_options_empty=0, fewer_than_2_words=0, counted=0,
                       with_repeated_words=0, options=0, options_2plus_words=0)
    for c in range(0, inc.shape[0], 2):
        for p in range(2):
            for k in range(K):
                opts = [np.flatnonzero(inc[c, o + A * (k + K * p)]).tolist() for o in range(A)]
                words, empty = sorted(set(sum(opts, []))), sum(not o for o in opts)
                st["or_groups"] += 1
                if empty:  # always-true (checked first)
                    st["always_true"] += 1
                    st["all_options_empty"] += empty == A
                elif len(words) < 2:
                    st["fewer_than_2_words"] += 1
                else:
                    st["counted"] += 1
                    st["with_repeated_words"] += len(words) < sum(map(len, opts))
                    st["options"] += A
                    st["options_2plus_words"] += sum(len(o) > 1 for o in opts)
                    out.append({"clause": c, "par": "AB"[p], "k": k, "words": words, "options": opts})
    return out, st


def load_run(d, role, data):
    z = np.load(os.path.join(d, "include_bits.npz"))
    F, A, K, name, (cfg, last) = int(z["F"]), int(z["A"]), int(z["K"]), os.path.basename(os.path.abspath(d)), config_of(d)
    if str(z["model"]) != "hier":
        raise SystemExit(f"{name}: not a hierarchical run, so it has no OR-groups")
    if cfg is not None and (cfg["data"] != data or bool(cfg.get("shuffle_labels")) != (role == "shuffled")):
        raise SystemExit(f"{name}: trained on '{cfg['data']}' with shuffle_labels={cfg.get('shuffle_labels')}, "
                         f"but given as a {role} run for --data {data}")
    inc = np.unpackbits(z["class1"], axis=-1, count=F).astype(bool)
    assert inc.shape[1] == 2 * K * A, (name, inc.shape)
    groups, st = counted_groups(inc, A, K)
    return {"dir": d, "name": name, "role": role, "vocab": [str(w) for w in z["vocab"]], "groups": groups, "stats": st,
            "cfg": cfg, "F": F, "pass": int(z["pass_P"]) if "pass_P" in z.files else last or (cfg or {}).get("passes")}


def reference(vocab, unit="paragraph"):
    """0/1 [texts, len(vocab)] of the NPMI reference text (one text = one paragraph, or one whole article), and the
    number of articles."""
    z = np.load(os.path.join(DATA, "paragraphs_F3072.npz"))
    start = np.r_[0, np.cumsum(z["npar"].astype(np.int64))]
    arts = json.load(open(os.path.join(DATA, "splits.json")))["npmi_ref"]
    rows = np.concatenate([np.arange(start[a], start[a + 1]) for a in arts])
    col = {w: i for i, w in enumerate(json.load(open(os.path.join(DATA, "vocab_F3072.json"))))}
    X = ids_to_csr(z["ids"], z["off"], len(col))[rows][:, [col[w] for w in vocab]]
    if unit == "article":  # a word is in an article if it is in any of its paragraphs
        owner = np.repeat(np.arange(len(arts)), [start[a + 1] - start[a] for a in arts])
        X = sp.csr_matrix(sp.csr_matrix((np.ones(len(rows)), (owner, np.arange(len(rows))))) @ X > 0, dtype=np.uint8)
    return X, len(arts)


def wordnet(vocab):
    """W [V, V] (1 = synonyms or hypernym-close), covered [V] (has a noun/verb/adjective sense), forms [V, V]."""
    import nltk
    nltk.data.path.insert(0, NLTK_DATA)
    from nltk.corpus import wordnet as wn
    assert wn.get_version() == "3.0"
    up, ids, rows = (lambda s: s.hypernyms() + s.instance_hypernyms()), {}, {k: [] for k in "SDHB"}
    for w in vocab:
        ss = {s for p in (wn.NOUN, wn.VERB, wn.ADJ) for s in wn.synsets(w, pos=p)}
        d1 = {h for s in ss for h in up(s)}
        d2 = d1 | {g for h in d1 for g in up(h)}
        base = {"#" + b for p in (wn.NOUN, wn.VERB, wn.ADJ) for b in wn._morphy(w, p)}  # all base forms
        for key, items in zip("SDHB", ({s.name() for s in ss}, {h.name() for h in d1}, {h.name() for h in d2}, base)):
            rows[key].append([ids.setdefault(i, len(ids)) for i in sorted(items)])

    def mat(r):
        return sp.csr_matrix((np.ones(sum(map(len, r))), (np.repeat(np.arange(len(r)), list(map(len, r))),
                              np.array([i for q in r for i in q], np.int64))), shape=(len(r), len(ids)))
    S, D, H, B = (mat(rows[k]) for k in "SDHB")
    near = (S @ H.T).toarray() > 0  # [u, v]: a sense of u is a hypernym of a sense of v, within 2 steps
    W = ((S @ S.T).toarray() > 0) | near | near.T | ((D @ D.T).toarray() > 0)
    return W.astype(np.float64), np.diff(S.indptr) > 0, (B @ B.T).toarray() > 0


def pick(groups, covered=None, skip=frozenset()):
    """Word-id lists for one measure: groups with a word in `skip` dropped; WordNet: covered words only, >= 2."""
    out = []
    for g in groups:
        w = [i for i in g["words"] if covered is None or covered[i]]
        if not skip & set(g["words"]) and len(w) >= 2:
            out.append(w)
    return out


def vs_random(g, M, bins, x):
    r = ng.compare(g, M, bins, x.draws, x.boots, 0, x.mc)
    if g:  # neutral names: the score is NPMI or the WordNet share
        r["score_groups"], r["score_random"] = r.pop("npmi_groups"), r.pop("npmi_random")
    return r


def vs_shuffled(a, b, M, x):
    if min(len(a), len(b)) < 2:
        return {"n_a": len(a), "n_b": len(b), "a_beats_b": None, "a_beats_b_calibrated": None,
                "n_groups_note": "fewer than 2 groups in a set: not computable"}
    return ng.compare_two(a, b, M, x.boots, 0, x.mc)


def comparisons(model, shuf, measures, skew, x):
    """{measure.version: {set: {"vs_random": ..., "vs_shuffled": ...}}} for each run, all runs pooled, and (a control,
    no verdict) the shuffled-label runs pooled."""
    res = {}
    for meas, (M, bins, cov) in measures.items():
        for ver in ("all", "no_skewed")[meas == "wordnet_no_forms":]:  # the diagnostic: the verdict version only
            skip, b = (skew if ver == "no_skewed" else frozenset()), bins.copy()
            b[sorted(skip)] = -2  # never drawn
            get = lambda rs: [w for r in rs for w in pick(r["groups"], cov, skip)]  # noqa: E731
            sh, out = get(shuf), {}
            for n, rs in [(r["name"], [r]) for r in model] + [("pooled", model)]:
                out[n] = {"vs_random": vs_random(get(rs), M, b, x), "vs_shuffled": vs_shuffled(get(rs), sh, M, x) if shuf else None}
            if shuf:
                out["shuffled_pooled"] = {"vs_random": vs_random(sh, M, b, x), "vs_shuffled": None}
            res[f"{meas}.{ver}"] = out
    return res


def inside(ci):
    return bool(ci[0] <= 0.5 <= ci[1])


def length_check(tr, dv):
    """Length-only logistic regression (Check 1b's features and model): trained on train, scored on dev, with the
    article bootstrap (score_saved.article_bootstrap: whole articles, 1,000 resamples; a pair belongs to art_a)."""
    import data_checks as dc, score_saved as ss
    Ltr, Ldv = dc.length_feats(tr), dc.length_feats(dv)
    sc = dc.StandardScaler().fit(Ltr)
    s = dc.LogisticRegression().fit(sc.transform(Ltr), tr["y"]).decision_function(sc.transform(Ldv))
    pred = (s > 0).astype(np.int64)
    b = ss.article_bootstrap(s, pred, s, pred, dv["y"], dv["art_a"])
    return {"acc": float((pred == dv["y"]).mean()), "auc": ss.hd.auc(s, dv["y"]), "acc_ci95": b["acc_ci95"],
            "auc_ci95": b["auc_ci95"], "articles": b["articles"], "n_dev": len(s),
            "ci_includes_0.50": inside(b["acc_ci95"]) and inside(b["auc_ci95"])}


def part1(d, length, data):
    """Check 4b part 1 for one seed, from a score_saved.py dev output dir."""
    f = os.path.join(d, "metrics.json")
    m = json.load(open(f)) if os.path.exists(f) else {}
    b = (m.get("bootstrap") or {}).get("value")
    if not b or not os.path.exists(os.path.join(d, "scores_dev.npz")):
        return {"pass": None, "note": "no article bootstrap or scores_dev.npz here: run score_saved.py (dev) on this "
                                      "seed's model at pass P and give its output dir"}
    if not m["data_file"]["value"].endswith(f"/{data}/dev.npz"):
        raise SystemExit(f"{d} was scored on {m['data_file']['value']}, not on {data}")
    import htm_driver as hd
    s = np.load(os.path.join(d, "scores_dev.npz"))
    r = {"auc": m["auc"]["value"], "auc_ci95": b["auc_ci95"], "partner_shuffle_acc_ci95": b["partner_shuffle_acc_ci95"],
         "partner_shuffle_auc_ci95": b["partner_shuffle_auc_ci95"], "scores_file_auc": hd.auc(s["score"], s["y"]),
         "dev_auc_ci_above_0.5": bool(b["auc_ci95"][0] > 0.5), "length_only_ci_includes_0.50": length["ci_includes_0.50"],
         "partner_shuffle_ci_includes_0.50": inside(b["partner_shuffle_acc_ci95"]) and inside(b["partner_shuffle_auc_ci95"])}
    r["scores_file_matches_metrics"] = bool(abs(r["scores_file_auc"] - r["auc"]) < 1e-9)
    r["pass"] = all(r[k] for k in ("dev_auc_ci_above_0.5", "partner_shuffle_ci_includes_0.50",
                                   "length_only_ci_includes_0.50", "scores_file_matches_metrics"))
    return r


def tri(flags):
    """no if any is no, else undecided if any is undecided, else yes."""
    return False if False in flags else None if None in flags else True


def flags(res, n, comps, cal):
    """The verdict components of one set: each comparison's gate (or calibrated) result."""
    k1, k2 = ("beats_random_calibrated", "a_beats_b_calibrated") if cal else ("beats_random", "a_beats_b")
    f = {}
    for meas, ver in comps:
        R = res[f"{meas}.{ver}"][n]
        f[f"{meas} vs random"], f[f"{meas} vs shuffled"] = R["vs_random"].get(k1), (R["vs_shuffled"] or {}).get(k2)
    return f


def group_text(g, vocab, skew, m=10):
    """Distinct options joined by OR, multi-word options in brackets, skewed words marked *."""
    wd = lambda i: vocab[i] + "*" * (i in skew)
    t = [wd(o[0]) if len(o) == 1 else "(" + " & ".join(map(wd, o)) + ")" for o in dict.fromkeys(map(tuple, g["options"]))]
    return " OR ".join(t[:m]) + (f" OR ...(+{len(t) - m})" if len(t) > m else "")


def yn(v):
    return {True: "yes", False: "no", None: "undecided"}[v]


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def line(name, vs, r):
    two = r is not None and "n_a" in r
    n = r and (f"{r['n_a']}/{r['n_b']}" if two else r["n_groups"])
    if not r or "diff" not in r:
        return f"   {name:<36} {vs:<9} {n or 0:>9}  " + (r or {}).get("n_groups_note", "no groups")
    lo, hi = r["ci95_group_bootstrap"]
    gate, cal, p = (r["a_beats_b"], r["a_beats_b_calibrated"], r["p_perm"]) if two else (r["beats_random"], r["beats_random_calibrated"], r["p_mc"])
    return (f"   {name:<36} {vs:<9} {n:>9} {r['diff']:+9.4f}  [{lo:+.4f}, {hi:+.4f}] {yn(gate):>4}   p {p:.4f} {yn(cal)}"
            + (f" ({r['n_groups_note']})" if r["n_groups_note"] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check 4a / 4b word-group scorer (see the module docstring)")
    ap.add_argument("--runs", nargs="+", required=True, help="the model's run dirs, one per seed")
    ap.add_argument("--shuffled", nargs="*", default=[], help="the shuffled-label model's run dirs (pooled)")
    ap.add_argument("--data", required=True, choices=["sa", "np", "ns_all"])
    ap.add_argument("--draws", type=int, default=1000, help="random groups per counted group (gates.md: 1,000)")
    ap.add_argument("--boots", type=int, default=1000, help="group-bootstrap resamples (gates.md: 1,000)")
    ap.add_argument("--mc", type=int, default=2000, help="null replicates of the calibrated test")
    ap.add_argument("--npmi-text", choices=["paragraph", "article"], default="article",
                    help="the NPMI text unit (fix it before the real scoring; see the module docstring)")
    ap.add_argument("--out", default=".", help="output dir when not run under run_logged.py")
    x = ap.parse_args(argv)
    t0, out_dir, P = time.time(), os.environ.get("TMLM_RUN_DIR", x.out), (lambda s: print(s, flush=True))
    os.makedirs(out_dir, exist_ok=True)
    model, shuf = [load_run(d, "model", x.data) for d in x.runs], [load_run(d, "shuffled", x.data) for d in x.shuffled]
    runs, vocab = model + shuf, model[0]["vocab"]
    if any(r["vocab"] != vocab for r in runs):
        raise SystemExit("the runs have different vocabularies")
    cfgs = [r["cfg"] for r in runs if r["cfg"]]
    diff = {k: v for k in SAME if len(v := sorted({str(c.get(k)) for c in cfgs})) > 1}
    if diff or len({r["pass"] for r in runs} - {None}) > 1:
        raise SystemExit(f"the runs differ in their settings {diff} or passes {[(r['name'], r['pass']) for r in runs]}")
    X, n_arts = reference(vocab, x.npmi_text)
    M, df = ng.npmi_matrix(X)
    bins, (W, covered, forms) = ng.freq_bins(df), wordnet(vocab)
    skew, wb = frozenset(vocab.index(w) for w in SKEWED if w in vocab), np.where(covered, bins, -1)
    res = comparisons(model, shuf, {"npmi": (M, bins, None), "wordnet": (W, wb, covered),
                                    "wordnet_no_forms": (W * ~forms, wb, covered)}, skew, x)
    for r in runs:  # per-group scores (groups.tsv, examples) and WordNet coverage
        for g in r["groups"]:
            cw = [i for i in g["words"] if covered[i]]
            g.update(cw=cw, npmi=ng.score(g["words"], M), wordnet=ng.score(cw, W) if len(cw) > 1 else None,
                     form_pairs=int(np.triu(forms[np.ix_(cw, cw)], 1).sum()))
        pairs = sum(len(g["words"]) * (len(g["words"]) - 1) // 2 for g in r["groups"])
        r["stats"].update(with_skewed_word=sum(bool(skew & set(g["words"])) for g in r["groups"]),
                          wordnet_groups_scored=sum(len(g["cw"]) > 1 for g in r["groups"]),
                          wordnet_pair_share=sum(len(g["cw"]) * (len(g["cw"]) - 1) // 2 for g in r["groups"]) / pairs if pairs else None)
    met = {"data": x.data, "runs": [r["name"] for r in model], "shuffled_runs": [r["name"] for r in shuf],
           "reference_text": {"articles": n_arts, "texts": X.shape[0], "text_unit": x.npmi_text, "vocab": len(vocab)},
           "wordnet_coverage": {"covered": int(covered.sum()), "vocab": len(vocab), "share": float(covered.mean()),
                                "missing": [w for w, c in zip(vocab, covered) if not c]},
           "group_counts": {r["name"]: r["stats"] for r in runs}, "comparisons": res, "undercoverage_note": NOTE}
    title = {"ns_all": "Check 4b, the synonym test (E4s)", "sa": "Check 4a, Wikipedia word groups (same article)",
             "np": "Check 4a rules on next-paragraph runs"}[x.data]
    pas = lambda rs: ", ".join(f"{r['name']} (pass {r['pass']})" for r in rs)
    P(f"WORD GROUPS: {title}, data {x.data}\nModel runs ({len(model)}): {pas(model)}\n"
      f"Shuffled-label runs ({len(shuf)}): {pas(shuf) or 'none given'}\n"
      f"Vocabulary: the runs' saved {len(vocab):,} words. NPMI reference text: the {n_arts:,} NPMI-reference articles "
      f"(in no split and in no pair); one {x.npmi_text} is one text ({X.shape[0]:,} texts).\n"
      f"WordNet 3.0: {int(covered.sum()):,} of {len(vocab):,} words ({covered.mean():.1%}) have a noun, verb or adjective "
      f"sense; pairs with the other words are skipped.\nSkewed words, marked * in every printed group: {', '.join(SKEWED)}.\n")
    P("1. OR-groups of the class-1 positive clauses. Dropped: always-true (an empty option; checked first) and fewer "
      "than 2 distinct words.\n" f"   {'run':<36} {'role':<9} {'groups':>6} {'always-true':>11} {'(all empty)':>11} "
      f"{'<2 words':>8} {'counted':>7} {'skewed':>6} {'repeated':>8} {'options 2+ words':>16}  WordNet: groups scored, pairs scored")
    for r in runs:
        s = r["stats"]
        P(f"   {r['name']:<36} {r['role']:<9} {s['or_groups']:>6} {s['always_true']:>11} {s['all_options_empty']:>11} "
          f"{s['fewer_than_2_words']:>8} {s['counted']:>7} {s['with_skewed_word']:>6} {s['with_repeated_words']:>8} "
          f"{s['options_2plus_words'] / max(s['options'], 1):>16.3f}  {s['wordnet_groups_scored']}, {s['wordnet_pair_share'] or 0:.1%}")
    P(f"\n2. Comparisons. diff = mean group score minus that of the random groups ({x.draws:,} frequency-matched draws "
      "per group) or of the shuffled-label groups. Gate = the frozen group bootstrap: 95% CI of diff above 0. Calibrated = "
      "p < 0.025 and t-interval above 0, with at least 12 groups per set.")
    label = {"npmi": "NPMI", "wordnet": "WordNet score", "wordnet_no_forms": "WordNet, two forms of one word not counted (diagnostic)",
             "pooled": "all model runs pooled", "shuffled_pooled": "shuffled-label runs pooled (control)"}
    for key, out in res.items():
        meas, ver = key.split(".")
        P(f"\n [{label[meas]}; {'all counted groups' if ver == 'all' else 'without groups that contain a skewed word'}]\n"
          f"   {'set':<36} {'vs':<9} {'groups':>9} {'diff':>9}  {'gate 95% CI':<20} {'gate':>4}   calibrated")
        for n, rr in out.items():
            P(line(label.get(n, n), "random", rr["vs_random"]) + ("\n" + line(label.get(n, n), "shuffled", rr["vs_shuffled"])
                                                                if rr["vs_shuffled"] is not None else ""))
    P(f"\n {NOTE}\n")
    names = [r["name"] for r in model]
    if x.data == "ns_all":
        length = length_check(*[load_pairs("ns_all", s, model[0]["F"]) for s in ("train", "dev")])
        p1 = {r["name"]: part1(r["dir"], length, x.data) for r in model}
        met.update(length_only=length, part1=p1)
        P(f"3. Check 4b\n   Part 1. Length-only logistic regression on {length['n_dev']:,} dev pairs: accuracy {length['acc']:.3f} "
          f"{ci(length['acc_ci95'])}, AUC {length['auc']:.3f} {ci(length['auc_ci95'])}; both include 0.50: "
          f"{yn(length['ci_includes_0.50'])}")
        for n, q in p1.items():
            P(f"   {n}: " + (q["note"] if q["pass"] is None else
              f"dev AUC {q['auc']:.3f} {ci(q['auc_ci95'])} above 0.5: {yn(q['dev_auc_ci_above_0.5'])}; partner-shuffle "
              f"accuracy {ci(q['partner_shuffle_acc_ci95'])} and AUC {ci(q['partner_shuffle_auc_ci95'])} include 0.50: "
              f"{yn(q['partner_shuffle_ci_includes_0.50'])}; scores file matches: {yn(q['scores_file_matches_metrics'])}"
              f" -> part 1 {yn(q['pass'])}"))
    for cal in (False, True):
        tag = "calibrated" if cal else "gate"
        if x.data == "ns_all":
            per = {n: {"part 1": p1[n]["pass"], **flags(res, n, [("wordnet", "no_skewed"), ("npmi", "no_skewed")], cal)} for n in names}
            ok = {n: tri(list(f.values())) for n, f in per.items()}
            v = list(ok.values())
            v = None if len(v) != 3 else True if v.count(True) >= 2 else None if v.count(True) + v.count(None) >= 2 else False
            P(f"   {tag.capitalize()} (parts 1 and 2; part 2 without the skewed-word groups):")
        else:
            per = {n: flags(res, n, [("npmi", "all")], cal) for n in names + ["pooled"]}
            ok = {n: tri(list(f.values())) for n, f in per.items()}
            v = ok["pooled"]
            P(("3. Check 4a (NPMI, all counted groups; decided on all model runs pooled)\n" if not cal else "")
              + f"   {tag.capitalize()}:")
        for n, f in per.items():
            P(f"   {label.get(n, n)}: " + ", ".join(f"{k} {yn(b)}" for k, b in f.items()) + f" -> {yn(ok[n])}")
        P(f"   -> Check {'4b' if x.data == 'ns_all' else '4a'} ({tag}): {yn(v).upper()}"
          + (f" ({list(ok.values()).count(True)} of {len(ok)} seeds pass; it needs 2 of 3)" if x.data == "ns_all" else "")
          + f". {NOTE}")
        met[f"check{'4b' if x.data == 'ns_all' else '4a'}_{tag}"] = {"components": per, "set_pass": ok, "pass": v}
    with open(os.path.join(out_dir, "groups.tsv"), "w") as f:
        f.write("role\trun\tclause\tgroup\tn_words\tskewed_words\tnpmi\twordnet\twordnet_pairs_scored\tpairs\t"
                "form_pairs\twords\toptions\n")
        for r in runs:
            for g in r["groups"]:
                n, c = len(g["words"]), len(g["cw"])
                f.write("\t".join(map(str, [r["role"], r["name"], g["clause"], f"{g['par']}{g['k'] + 1}", n,
                                            " ".join(vocab[i] for i in g["words"] if i in skew), f"{g['npmi']:.4f}",
                                            "" if g["wordnet"] is None else f"{g['wordnet']:.4f}", c * (c - 1) // 2,
                                            n * (n - 1) // 2, g["form_pairs"], " ".join(vocab[i] for i in g["words"]),
                                            group_text(g, vocab, skew, 10 ** 6)])) + "\n")
    key = "wordnet" if x.data == "ns_all" else "npmi"
    P(f"\n4. Examples: the 5 highest-scoring groups of each model run by {key} (not typical; every group is in groups.tsv).")
    for r in model:
        for g in sorted(r["groups"], key=lambda g: 9 if g[key] is None else -g[key])[:5]:
            P(f"   {r['name']} clause {g['clause']} {g['par']}{g['k'] + 1}: NPMI {g['npmi']:+.3f}, WordNet "
              f"{'-' if g['wordnet'] is None else format(g['wordnet'], '.2f')}: {group_text(g, vocab, skew)}")
    met["runtime_s"] = round(time.time() - t0, 1)
    P(f"\nwall {met['runtime_s']} s; groups file: {os.path.join(out_dir, 'groups.tsv')}")
    json.dump({k: {"value": v, "label": "derived" if k.startswith(("check4", "part1")) else "measured"} for k, v in met.items()},
              open(os.path.join(out_dir, "metrics.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return met


if __name__ == "__main__":
    main()
