"""CPU tests for wiki_groups.py (Checks 4a and 4b), on synthetic groups only: no real run's groups are scored.
Fake run dirs plant known OR-groups in include bits over the real 1,024-word vocabulary; WordNet 3.0 and the real
NPMI reference text are used as they will be in the real scoring.
Run from the study folder: ../tm_language_model/.venv-tm/bin/python src/test_wiki_groups.py
"""
import contextlib, io, json, os, sys, tempfile, time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npmi_groups as ng, wiki_groups as wg  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tm_language_model", "src"))
from tmlm.common.runlog import validate_metrics  # noqa: E402

VOCAB = json.load(open(os.path.join(wg.DATA, "vocab_F1024.json")))
WID = {w: i for i, w in enumerate(VOCAB)}
SYN = [["road", "route", "highway"], ["war", "battle"], ["city", "town"], ["school", "college", "university"],
       ["king", "queen"], ["game", "match"], ["president", "leader"], ["building", "house", "structure"],
       ["region", "area", "district"], ["country", "state"], ["film", "movie"], ["army", "troops", "forces"],
       ["big", "large"], ["began", "start"]]  # the last two contain a skewed word
CTX, RESULTS = {}, {}


def ctx():
    if not CTX:
        X, _ = wg.reference(VOCAB)
        M, df = ng.npmi_matrix(X)
        W, cov, forms = wg.wordnet(VOCAB)
        CTX.update(M=M, bins=ng.freq_bins(df), W=W, cov=cov, forms=forms, wb=np.where(cov, ng.freq_bins(df), -1))
    return CTX


def make_run(d, groups, role="model", data="ns_all", clauses=8, A=4, K=2, s=100.0, passes=100):
    """groups: {(clause, paragraph, k): [option word lists]}; every other OR-group stays all-empty (always-true)."""
    inc = np.zeros((clauses, 2 * K * A, 1024), bool)
    for (c, p, k), opts in groups.items():
        for o, words in enumerate(opts):
            inc[c, o + A * (k + K * p), [WID[w] for w in words]] = True
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "include_bits.npz"), class1=np.packbits(inc, axis=-1), class0=np.packbits(inc, axis=-1),
                        F=1024, A=A, K=K, model="hier", clauses=clauses, T=3200.0, s=s, vocab=np.array(VOCAB))
    cfg = dict(data=data, F=1024, model="hier", clauses=clauses, T=3200.0, s=s, A=A, K=K, n_train=0, batch=5000,
               label_mode="pair", shuffle_labels=role == "shuffled", passes=passes)
    open(os.path.join(d, "log.txt"), "w").write(f"[run_logged] run dir: fake\nCONFIG {json.dumps(cfg)}\n"
                                                f'PASS {{"pass": {passes - 1}}}\nEPOCH {{"pass": {passes}, "dev_auc": 0.6}}\n')
    return d


def fill(words, A):
    return [[words[o % len(words)]] for o in range(A)]


def place(groups, clauses, K, start=0):
    """Put word groups into consecutive positive-clause OR-groups (one option per word, repeated to fill A=4)."""
    slots = [(c, p, k) for c in range(0, clauses, 2) for p in range(2) for k in range(K)][start:]
    assert len(groups) <= len(slots)
    return {sl: fill(g, 4) for sl, g in zip(slots, groups)}


def fake_scored(d, auc_ci=None, ps_acc=(0.49, 0.51), ps_auc=(0.495, 0.512), data="ns_all", break_file=False):
    """A score_saved.py-style dev output: metrics.json with the article bootstrap block, and scores_dev.npz."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000).astype(np.uint32)
    sc = (rng.normal(size=2000) + 0.7 * y).astype(np.float32)
    import htm_driver as hd
    auc = hd.auc(sc, y) + (0.01 if break_file else 0.0)
    auc_ci = auc_ci or (round(auc - 0.03, 6), round(auc + 0.03, 6))
    boot = {"articles": 300, "resamples": 1000, "seed": 0, "pair_owner": "art_a", "auc_ci95": list(auc_ci),
            "partner_shuffle_acc_ci95": list(ps_acc), "partner_shuffle_auc_ci95": list(ps_auc)}
    m = {"auc": auc, "data_file": os.path.join(wg.DATA, data, "dev.npz"), "bootstrap": boot}
    json.dump({k: {"value": v, "label": "measured"} for k, v in m.items()}, open(os.path.join(d, "metrics.json"), "w"))
    np.savez_compressed(os.path.join(d, "scores_dev.npz"), score=sc, y=y, art_a=np.arange(2000) // 7)


def test_extract():
    g = {(0, 0, 0): [["road"], ["route"], ["highway"], ["road", "route"]],  # counted; repeated words; one 2-word option
         (0, 0, 1): [["war"], [], ["battle"], ["war"]],  # one empty option: always-true
         (0, 1, 0): [["city"]] * 4,  # one distinct word
         (0, 1, 1): fill(["big", "large"], 4),  # counted, skewed word
         (2, 0, 1): [["king"], ["queen"], ["film"], ["movie"]],  # counted
         (2, 1, 0): fill(["however", "although", "since"], 4),  # counted, skewed, no word in WordNet
         (1, 0, 0): fill(["school", "college"], 4), (3, 1, 1): fill(["war", "battle"], 4)}  # negative clauses: ignored
    for w in [w for opts in g.values() for o in opts for w in o]:
        assert w in WID, w
    with tempfile.TemporaryDirectory() as d:
        r = wg.load_run(make_run(d, g, clauses=4), "model", "ns_all")
    assert r["stats"] == dict(or_groups=8, always_true=3, all_options_empty=2, fewer_than_2_words=1, counted=4,
                              with_repeated_words=3, options=16, options_2plus_words=1), r["stats"]
    got = [(x["clause"], x["par"], x["k"], {VOCAB[i] for i in x["words"]}) for x in r["groups"]]
    assert got == [(0, "A", 0, {"road", "route", "highway"}), (0, "B", 1, {"big", "large"}),
                   (2, "A", 1, {"king", "queen", "film", "movie"}), (2, "B", 0, {"however", "although", "since"})], got
    skew = frozenset(WID[w] for w in wg.SKEWED)
    assert len(wg.pick(r["groups"])) == 4 and len(wg.pick(r["groups"], skip=skew)) == 2
    assert len(wg.pick(r["groups"], ctx()["cov"])) == 3  # the however/although/since group has no WordNet word
    rr = " & ".join(sorted(["road", "route"], key=WID.get))
    assert wg.group_text(r["groups"][0], VOCAB, skew) == f"road OR route OR highway OR ({rr})"
    assert wg.group_text(r["groups"][1], VOCAB, skew) == "big* OR large"
    RESULTS["extract_stats"] = r["stats"]


def test_reference_text():
    """The NPMI reference text: npmi_ref articles are in no split and in no train or dev pair (the final-test pair
    files are not opened); one text = one paragraph (default) or one article (= the OR of its paragraphs)."""
    s = json.load(open(os.path.join(wg.DATA, "splits.json")))
    ref = set(s["npmi_ref"])
    assert len(ref) == 2847 and not ref & set(s["train"] + s["dev"] + s["test"])
    for v in ("sa", "np", "ns_all"):
        for split in ("train", "dev"):
            z = np.load(os.path.join(wg.DATA, v, f"{split}.npz"))
            assert not ref & set(np.r_[z["art_a"], z["art_b"]].tolist()), (v, split)
    (Xp, n), (Xa, n2) = wg.reference(VOCAB), wg.reference(VOCAB, "article")
    assert Xp.shape == (70118, 1024) and Xa.shape == (2847, 1024) and n == n2 == 2847 and Xp.max() == Xa.max() == 1
    start = np.r_[0, np.cumsum(np.load(os.path.join(wg.DATA, "paragraphs_F3072.npz"))["npar"][s["npmi_ref"]])]
    for i in (0, 1234, 2846):
        assert (Xp[start[i]:start[i + 1]].max(axis=0).toarray().ravel() == Xa[i].toarray().ravel()).all()
    RESULTS["reference_text"] = {"paragraphs": Xp.shape[0], "articles": Xa.shape[0], "words_per_paragraph": float(Xp.sum(1).mean()),
                                 "words_per_article": float(Xa.sum(1).mean())}


def related_slow(u, v, wn):
    """Independent reference for the WordNet relation: loops over sense pairs (gates.md definition)."""
    ss = [{s for p in (wn.NOUN, wn.VERB, wn.ADJ) for s in wn.synsets(w, pos=p)} for w in (u, v)]
    if not ss[0] or not ss[1]:
        return None
    if ss[0] & ss[1]:
        return "synonym"
    up = lambda s: set(s.hypernyms() + s.instance_hypernyms())
    for a in ss[0]:
        for b in ss[1]:
            ha, hb = up(a), up(b)
            if a in hb or b in ha or any(a in up(h) for h in hb) or any(b in up(h) for h in ha):
                return "hypernym within 2 steps"
            if ha & hb:
                return "shared direct hypernym"
    return "unrelated"


def test_wordnet():
    c = ctx()
    from nltk.corpus import wordnet as wn
    W, cov, forms = c["W"], c["cov"], c["forms"]
    assert cov.sum() == 957 and not cov[WID["since"]] and not cov[WID["however"]] and cov[WID["began"]]
    for g in SYN:
        assert all(w in WID for w in g), g
        assert ng.score([WID[w] for w in g], W) >= 0.99, (g, ng.score([WID[w] for w in g], W))
    assert forms[WID["film"], WID["films"]] and W[WID["film"], WID["films"]] == 1 and not forms[WID["film"], WID["movie"]]
    rng, kinds = np.random.default_rng(5), {}
    pairs = [tuple(rng.choice(1024, 2, replace=False)) for _ in range(3000)] + [(WID[a], WID[b]) for g in SYN for a in g for b in g if a < b]
    for i, j in pairs:
        k = related_slow(VOCAB[i], VOCAB[j], wn)
        kinds[k] = kinds.get(k, 0) + 1
        assert (k is None) == (not (cov[i] and cov[j])), (VOCAB[i], VOCAB[j])
        if k is not None:
            assert bool(W[i, j]) == (k != "unrelated") and W[i, j] == W[j, i], (VOCAB[i], VOCAB[j], k, W[i, j])
    assert all(kinds.get(k, 0) > 10 for k in ("synonym", "hypernym within 2 steps", "shared direct hypernym", "unrelated")), kinds
    iu = np.triu_indices(int(cov.sum()), 1)
    RESULTS["wordnet"] = {"covered_words": int(cov.sum()), "related_share_of_covered_pairs": float(W[np.ix_(cov, cov)][iu].mean()),
                          "form_pairs": int(np.triu(forms, 1).sum()), "slow_check_pairs": len(pairs), "slow_check_kinds": kinds}


def null_groups(rng, k, bins, pool, lo=2, hi=7):
    """k groups drawn exactly like the random groups (the null): random templates, then one matched draw each."""
    return [list(ng.draw_matched(list(rng.choice(pool, z, replace=False)), bins, rng, 1)[0]) for z in rng.integers(lo, hi, k)]


def test_beats_random():
    c = ctx()
    syn = [[WID[w] for w in g] for g in SYN]
    r = ng.compare(syn, c["W"], c["wb"], 1000, 1000, 0, 2000)
    assert r["n_groups"] == 14 and r["beats_random"] and r["beats_random_calibrated"] is True and r["p_mc"] < 0.001, r
    RESULTS["planted_synonyms_vs_random"] = {"wordnet": r, "npmi": ng.compare(syn, c["M"], c["bins"], 1000, 1000, 0, 2000)}


def test_null_rates(reps_small=1000, reps_large=300):
    """No-effect cases on the real WordNet and NPMI matrices: 30 groups drawn exactly like their random groups.
    Measured: the share of cases with a false 'beats random' (frozen gate, Monte Carlo test, full calibrated verdict),
    and the mean diff, which must be about 0 (the random groups are drawn like the groups)."""
    c, out = ctx(), {}
    for name, M, bins, pool in (("wordnet", c["W"], c["wb"], np.flatnonzero(c["cov"])), ("npmi", c["M"], c["bins"], np.arange(1024))):
        for lo, hi, reps in ((2, 7, reps_small), (10, 41, reps_large)):
            hits, diffs = np.zeros((reps, 3), bool), np.zeros(reps)
            for i in range(reps):
                g = np.random.default_rng(300_000 + 10_000 * lo + i)
                q = ng.compare(null_groups(g, 30, bins, pool, lo, hi), M, bins, 200, 1000, i, 2000)
                hits[i], diffs[i] = (q["beats_random"], q["p_mc"] < 0.025, bool(q["beats_random_calibrated"])), q["diff"]
            r = {**dict(zip(["gate_bootstrap", "mc_test", "calibrated"], map(float, hits.mean(0)))), "replicates": reps,
                 "mean_diff": float(diffs.mean()), "se_mean_diff": float(diffs.std() / np.sqrt(reps))}
            assert abs(r["mean_diff"]) < 4 * r["se_mean_diff"] + 1e-9 and r["calibrated"] <= 0.08, r  # rates: measured
            out[f"{name}_30_groups_of_{lo}_to_{hi - 1}_words"] = r
    RESULTS["null_false_beats_random_rates_real_matrices"] = out


def test_compare_two(reps=1000):
    c = ctx()
    W, wb, pool = c["W"], c["wb"], np.flatnonzero(c["cov"])
    syn = [[WID[w] for w in g] for g in SYN]
    x = type("X", (), {"boots": 1000, "mc": 2000})
    r = wg.vs_shuffled(syn, null_groups(np.random.default_rng(1), 30, wb, pool), W, x)
    assert r["a_beats_b"] and r["a_beats_b_calibrated"] is True, r
    back = wg.vs_shuffled(null_groups(np.random.default_rng(1), 30, wb, pool), syn, W, x)
    assert not back["a_beats_b"] and back["a_beats_b_calibrated"] is False, back
    empty = wg.vs_shuffled(syn, [], W, x)
    assert empty["a_beats_b"] is None and empty["a_beats_b_calibrated"] is None and empty["n_b"] == 0
    hits = np.zeros((reps, 2), bool)
    for i in range(reps):
        g = np.random.default_rng(20_000 + i)
        q = wg.vs_shuffled(null_groups(g, 30, wb, pool), null_groups(g, 30, wb, pool), W, x)
        hits[i] = q["a_beats_b"], bool(q["a_beats_b_calibrated"])
    rate = dict(zip(["gate_bootstrap", "calibrated"], map(float, hits.mean(0))))
    assert rate["calibrated"] <= 0.05, rate
    RESULTS["compare_two"] = {"synonyms_vs_random_groups": r, "null_false_positive_rate_30_vs_30": {**rate, "replicates": reps}}


def test_skewed_split():
    c, seen = ctx(), []
    real = ng.compare

    def spy(groups, M, bins, *a):
        seen.append((len(groups), bins.copy()))
        return real(groups, M, bins, *a)
    with tempfile.TemporaryDirectory() as d:
        m = wg.load_run(make_run(os.path.join(d, "m"), place(SYN, 8, 2)), "model", "ns_all")
        s = wg.load_run(make_run(os.path.join(d, "s"), place([["war", "big"], ["city", "game"]], 8, 2), "shuffled"), "shuffled", "ns_all")
    skew = frozenset(WID[w] for w in wg.SKEWED)
    x = type("X", (), {"draws": 50, "boots": 200, "mc": 200})
    ng.compare = spy
    try:
        res = wg.comparisons([m], [s], {"npmi": (c["M"], c["bins"], None), "wordnet": (c["W"], c["wb"], c["cov"])}, skew, x)
    finally:
        ng.compare = real
    for key in ("npmi", "wordnet"):
        assert res[f"{key}.all"][m["name"]]["vs_random"]["n_groups"] == 14
        assert res[f"{key}.no_skewed"][m["name"]]["vs_random"]["n_groups"] == 12
        assert res[f"{key}.all"][m["name"]]["vs_shuffled"]["n_b"] == 2 and res[f"{key}.no_skewed"][m["name"]]["vs_shuffled"]["n_b"] == 1
    sk = sorted(skew)
    assert all((b[sk] == -2).all() for n, b in seen if n in (12, 1)) and not any((b[sk] == -2).any() for n, b in seen if n in (14, 2))
    RESULTS["skewed_split"] = "passed"


def test_part1():
    from pairdata import load_pairs
    rng = np.random.default_rng(3)

    def fake_pairs(n, informative):
        y = rng.integers(0, 2, n)
        la = rng.integers(5, 30, n) + (6 * y if informative else 0)
        A = sp.csr_matrix((np.ones(la.sum()), (np.repeat(np.arange(n), la), np.concatenate([rng.choice(1024, k, replace=False) for k in la]))), shape=(n, 1024))
        return {"A": A, "B": A[rng.permutation(n)], "y": y, "art_a": np.arange(n) // 5}
    good, bad = wg.length_check(fake_pairs(4000, False), fake_pairs(4000, False)), wg.length_check(fake_pairs(4000, True), fake_pairs(4000, True))
    assert good["ci_includes_0.50"] and not bad["ci_includes_0.50"] and bad["auc"] > 0.6, (good, bad)
    real = wg.length_check(load_pairs("ns_all", "train"), load_pairs("ns_all", "dev"))  # a data check, not a model
    assert real["ci_includes_0.50"] and real["n_dev"] == 20000, real
    RESULTS["length_only_ns_all"] = real
    with tempfile.TemporaryDirectory() as d:
        fake_scored(d)
        q = wg.part1(d, real, "ns_all")
        assert q["pass"] is True and q["scores_file_matches_metrics"], q
        assert wg.part1(d, bad, "ns_all")["pass"] is False
        for kw, key in ((dict(auc_ci=(0.49, 0.53)), "dev_auc_ci_above_0.5"), (dict(ps_acc=(0.505, 0.53)), "partner_shuffle_ci_includes_0.50"),
                        (dict(ps_auc=(0.45, 0.49)), "partner_shuffle_ci_includes_0.50"), (dict(break_file=True), "scores_file_matches_metrics")):
            fake_scored(d, **kw)
            q = wg.part1(d, real, "ns_all")
            assert q["pass"] is False and q[key] is False, (kw, q)
        fake_scored(d, data="sa")
        try:
            wg.part1(d, real, "ns_all")
            raise AssertionError("a run scored on other data must be refused")
        except SystemExit:
            pass
        os.remove(os.path.join(d, "metrics.json"))
        assert wg.part1(d, real, "ns_all")["pass"] is None
    RESULTS["part1"] = "passed"


def run_main(argv):
    out = io.StringIO()
    saved = os.environ.pop("TMLM_RUN_DIR", None)
    try:
        with contextlib.redirect_stdout(out):
            met = wg.main(argv)
    finally:
        if saved is not None:
            os.environ["TMLM_RUN_DIR"] = saved
    return met, out.getvalue()


def test_end_to_end():
    rng = np.random.default_rng(4)
    pool = [w for w in VOCAB if w not in wg.SKEWED]
    rand = lambda n: [list(rng.choice(pool, int(rng.integers(2, 5)), replace=False)) for _ in range(n)]
    with tempfile.TemporaryDirectory() as d:
        mods = [make_run(os.path.join(d, f"model_s{i}"), place(SYN + rand(4), 12, 2), clauses=12) for i in (1, 2, 3)]
        shuf = [make_run(os.path.join(d, f"shuf_s{i}"), place(rand(18), 12, 2), "shuffled", clauses=12) for i in (1, 2, 3)]
        for m in mods:
            fake_scored(m)
        met, text = run_main(["--runs", *mods, "--shuffled", *shuf, "--data", "ns_all", "--draws", "300", "--out", d])
        validate_metrics(json.load(open(os.path.join(d, "metrics.json"))))
        tsv = open(os.path.join(d, "groups.tsv")).read().splitlines()
        assert len(tsv) == 1 + 3 * 18 + 3 * 18 and any("big*" in l for l in tsv), len(tsv)
        assert met["check4b_gate"]["pass"] is True and met["check4b_calibrated"]["pass"] is True, met["check4b_gate"]
        assert met["group_counts"]["model_s1"]["with_skewed_word"] == 2 and "NOTE on the gate" in text
        assert text.count(wg.NOTE) >= 3 and "-> Check 4b (gate): YES" in text
        RESULTS["end_to_end_planted"] = {k: met[k] for k in ("check4b_gate", "check4b_calibrated")}
    with tempfile.TemporaryDirectory() as d:
        mods = [make_run(os.path.join(d, f"model_s{i}"), place(rand(18), 12, 2), clauses=12) for i in (1, 2, 3)]
        shuf = [make_run(os.path.join(d, f"shuf_s{i}"), place(rand(18), 12, 2), "shuffled", clauses=12) for i in (1, 2, 3)]
        for m in mods:
            fake_scored(m)
        met, text = run_main(["--runs", *mods, "--shuffled", *shuf, "--data", "ns_all", "--draws", "300", "--out", d])
        assert met["check4b_gate"]["pass"] is not True, met["check4b_gate"]
        RESULTS["end_to_end_random_model"] = {k: met[k] for k in ("check4b_gate", "check4b_calibrated")}
        sa = [make_run(os.path.join(d, f"sa_s{i}"), place(SYN + rand(4), 12, 2), clauses=12, data="sa") for i in (1, 2, 3)]
        sas = [make_run(os.path.join(d, f"sa_shuf_s{i}"), place(rand(18), 12, 2), "shuffled", clauses=12, data="sa") for i in (1, 2, 3)]
        met, text = run_main(["--runs", *sa, "--shuffled", *sas, "--data", "sa", "--draws", "300", "--out", d])
        assert met["check4a_gate"]["pass"] is True and "Check 4a" in text, met["check4a_gate"]
        RESULTS["end_to_end_check4a_planted"] = met["check4a_gate"]
        met, _ = run_main(["--runs", *sa, "--shuffled", *sas, "--data", "sa", "--draws", "300", "--npmi-text", "article", "--out", d])
        assert met["reference_text"]["text_unit"] == "article" and met["reference_text"]["texts"] == 2847, met["reference_text"]
        RESULTS["end_to_end_check4a_planted_article_npmi"] = met["check4a_gate"]
        bad = [  # wrong role, wrong data, other settings, another pass: each must be refused
            ["--runs", *mods[:1], "--shuffled", mods[1], "--data", "ns_all"],
            ["--runs", *mods, "--data", "sa"],
            ["--runs", *mods, "--shuffled", make_run(os.path.join(d, "s400"), place(rand(3), 12, 2), "shuffled", clauses=12, s=400.0), "--data", "ns_all"],
            ["--runs", *mods, "--shuffled", make_run(os.path.join(d, "p200"), place(rand(3), 12, 2), "shuffled", clauses=12, passes=200), "--data", "ns_all"]]
        for argv in bad:
            try:
                run_main(argv + ["--draws", "10", "--out", d])
                raise AssertionError(f"must be refused: {argv}")
            except SystemExit as e:
                assert str(e), argv


def test_runtime_real_size():
    """Granmo's regime shape (50 clauses, A 64, K 2): 100 OR-groups per run of 20 to 64 options, 3 + 3 runs, 1,000
    draws (the real setting). Measures the expected wall time of the real scoring."""
    rng = np.random.default_rng(6)
    with tempfile.TemporaryDirectory() as d:
        dirs = []
        for i, role in enumerate(["model"] * 3 + ["shuffled"] * 3):
            inc = np.zeros((50, 256, 1024), bool)
            for c in range(0, 50, 2):
                for leaf in range(256):
                    inc[c, leaf, rng.choice(1024, int(rng.integers(1, 3)), replace=False)] = True
            dd = make_run(os.path.join(d, f"{role}_{i}"), {}, role, clauses=50, A=64, K=2)
            z = dict(np.load(os.path.join(dd, "include_bits.npz")))
            z["class1"] = np.packbits(inc, axis=-1)
            np.savez_compressed(os.path.join(dd, "include_bits.npz"), **z)
            dirs.append(dd)
        t = time.time()
        met, text = run_main(["--runs", *dirs[:3], "--shuffled", *dirs[3:], "--data", "ns_all", "--out", d])
        RESULTS["runtime_real_size_s"] = round(time.time() - t, 1)
        RESULTS["runtime_real_size_groups_per_run"] = met["group_counts"]["model_0"]
        assert met["group_counts"]["model_0"]["counted"] == 100
        print(text[-3000:], flush=True)


if __name__ == "__main__":
    t0 = time.time()
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f} s)", flush=True)
    RESULTS["tests_passed"], RESULTS["wall_s"] = len(tests), round(time.time() - t0, 1)
    print("RESULTS", json.dumps(RESULTS, default=str))
    if os.environ.get("TMLM_RUN_DIR"):
        json.dump({k: {"value": v, "label": "measured"} for k, v in RESULTS.items()},
                  open(os.path.join(os.environ["TMLM_RUN_DIR"], "metrics.json"), "w"), indent=1, default=str)
