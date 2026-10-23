"""A2: which category code helps? Class members mixed with KN5 + cache, one arm per code.

    python scripts/a2_run.py --split valid                 # development run on valid
    python scripts/a2_run.py --split devtest --final       # the one dev-test look (needs a recorded decision)
    python scripts/a2_run.py --split valid --smoke         # tiny sizes, for testing the code

Every arm is the system KN5 + cache + the class members of one code. A class member is a class
5-gram (interpolated modified Kneser-Ney on the class stream, sentence mode) times P(w | class),
one member per level of the code (floor/classlm.py). The comparator is KN5 + cache without
class members. Counts and codes come from train only; the cache settings (half-life, <eos>
added or not) are chosen on valid without class members and kept for every arm; the mixing
weights of every arm are fitted by EM on calib (one global weight set). The cache resets at
every 100-sentence block and runs across sentences inside a block; n-gram members are in
sentence mode.

Codes (floor/codes.py): exchange tree 16/64/256/1024; 10 random codes (the exchange tree with
the words permuted); k-means-200 of clause_lm; predefined (part of speech, then part of speech
plus WordNet first-sense group); sliding-window topic code 64/256; kind of word plus topic
(the exchange members and the topic members together); the OR-groups of the hierarchical_tm study. No trained TM
auto-encoder exists in the repository, so that arm is not run.

Reported per arm: bits per token and perplexity on the eval split, on its positions whose
bigram is unseen in train, and without duplicate tokens; the paired difference to the
comparator with a 95% block-bootstrap interval. Reported per code, with no verdict: SimLex-999
and WS-353 correlation, the synonym-pair sense test, distinct n-gram growth.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tally import ptb  # noqa: E402
from tally.floor import cache, codeeval, codes, mix, stats  # noqa: E402
from tally.floor.classlm import ClassMember  # noqa: E402
from tally.floor.kn import KN  # noqa: E402
from tally.runlog import Run  # noqa: E402

HALF_LIVES = (25, 50, 100, 200, 400, 800, 1600, 3200, np.inf)
N_RANDOM = 10
CLASS_ORDER = 5
DELTA = 0.5
CANDIDATES = ("exchange", "kmeans200", "predefined", "topic", "kind_topic", "orgroups")
SUCCESS_PPL, KILL_PPL, MARGIN_PPL = 0.05, 0.02, 0.01


def arms_of(levels_of):
    ex = [("exchange", l) for l in range(levels_of["exchange"])]
    tp = [("topic", l) for l in range(levels_of["topic"])]
    A = {"none": [], "exchange": ex}
    for l in range(levels_of["exchange"]):
        A[f"exchange_level{l}"] = [("exchange", l)]
    for s in range(1, N_RANDOM + 1):
        A[f"random_s{s}"] = [(f"random_s{s}", l) for l in range(levels_of["exchange"])]
    A["kmeans200"] = [("kmeans200", 0)]
    A["predefined"] = [("predefined", l) for l in range(levels_of["predefined"])]
    A["topic"] = tp
    A["kind_topic"] = ex + tp
    A["orgroups"] = [("orgroups", 0)]
    return A


def tune_cache(V, eos, ids, blocks, p_cal, p_val):
    """Half-life and skip chosen by valid likelihood; weights by EM on calib (reset per block)."""
    best, rows = None, []
    for hl in HALF_LIVES:
        for skip in ((), (eos,)):
            c = cache.Cache(V, hl, "block", skip)
            pc, _ = c.prob(ids["calib"], blocks["calib"], p_cal)
            lam, _ = mix.em(np.stack([p_cal, pc], 1))
            pv, _ = c.prob(ids["valid"], blocks["valid"], p_val)
            llv = float(np.log(np.stack([p_val, pv], 1) @ lam).sum())
            rows.append({"half_life": float(hl), "skip_eos": bool(skip),
                         "valid_bits": -llv / len(pv) / np.log(2)})
            if best is None or llv > best[0]:
                best = (llv, c)
    return best[1], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("valid", "devtest"), required=True)
    ap.add_argument("--final", action="store_true", help="required for --split devtest")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, output labelled smoke")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.split == "devtest" and not a.final:
        sys.exit("devtest is scored only with --final (the one dev-test look, by a recorded decision)")
    if a.split == "devtest" and a.smoke:
        sys.exit("no smoke runs on devtest")
    ev = a.split
    name = a.name or ("a2_smoke" if a.smoke else ("a2_devtest_final" if a.final else "a2_valid_dev"))
    if a.smoke and "smoke" not in name:
        sys.exit("smoke runs must be labelled smoke")
    S = dict(levels=(4, 8, 16, 32), km=(200, 20, 400), topic=(32, 8), iters=1, blocks=(40, 4)) \
        if a.smoke else dict(levels=codes.EXCHANGE_LEVELS, km=(1000, 200, 2000), topic=(256, 64),
                             iters=5, blocks=None)
    ext = [os.path.join(codes.EXTERNAL, *p) for p in (
        ("simlex999", "SimLex-999.zip"), ("wordsim353", "wordsim353.zip"),
        ("nltk_data", "corpora", "brown.zip"), ("nltk_data", "taggers", "universal_tagset.zip"))]
    wn_zip = os.path.join(codes.WORDNET_DIR, "corpora", "wordnet.zip")
    cfg = {"split": ev, "smoke": a.smoke, "exchange_levels": list(S["levels"]),
           "kmeans": list(S["km"]), "topic_levels": list(S["topic"]), "tagger_iters": S["iters"],
           "class_order": CLASS_ORDER, "emission_delta": DELTA, "n_random": N_RANDOM,
           "kn_order": 5, "mode": "sent", "cache_reset": "block", "half_lives": [float(h) for h in HALF_LIVES],
           "cache_select": "valid, without class members", "em_fit": "calib, one global weight set",
           "bootstrap": {"n_boot": 1000, "level": 0.95, "seed": 0}, "dup_min_len": 8,
           "lines": {"success_ppl": SUCCESS_PPL, "kill_ppl": KILL_PPL, "margin_vs_random_mean_ppl": MARGIN_PPL},
           "candidates": list(CANDIDATES), "tm_autoencoder": "no trained one in the repository; arm not run"}
    run = Run(name, cfg, data_files=[ptb.path("train"), ptb.path("valid"), *ext, wn_zip,
                                     codes.ORGROUPS_FILE])
    T, M = {}, {"codes": {}, "members": {}}
    t = time.time()
    vocab, w2i, ids, blocks = ptb.load()
    T["load"] = time.time() - t
    V, eos = len(vocab), w2i["<eos>"]
    if a.smoke:
        nb_tr, nb_ev = S["blocks"]
        for s in ("train", "calib", "valid"):
            keep = np.isin(blocks[s], np.unique(blocks[s])[:nb_tr if s == "train" else nb_ev])
            ids[s], blocks[s] = ids[s][keep], blocks[s][keep]
    tr = ids["train"]
    run.print("tokens", {s: int(len(ids[s])) for s in ("train", "calib", ev)}, "V", V)
    pre = "smoke_" if a.smoke else ""
    src = {"train_stream_sha256": codes.stream_hash(tr), "ptb": ptb.source_hashes()}

    # ---- codes ----
    C = {}
    t = time.time()
    C["exchange"], M["codes"]["exchange"] = codes.get(
        pre + "exchange", src, lambda: codes.exchange_tree(tr, V, eos, S["levels"], log=run.print), run.print)
    T["code_exchange"] = time.time() - t
    for s in range(1, N_RANDOM + 1):
        C[f"random_s{s}"], M["codes"][f"random_s{s}"] = codes.get(
            pre + f"random_s{s}", dict(src, base=M["codes"]["exchange"]["code_sha256"]),
            lambda s=s: codes.random_code(C["exchange"], eos, s), run.print)
    t = time.time()
    K1, k, nc = S["km"]
    C["kmeans200"], M["codes"]["kmeans200"] = codes.get(
        pre + "kmeans200", src, lambda: codes.kmeans200(tr, V, eos, K1, k, nc), run.print)
    T["code_kmeans200"] = time.time() - t
    t = time.time()
    C["predefined"], M["codes"]["predefined"] = codes.get(
        pre + "predefined", dict(src, brown=codes.sha256_file(ext[2]), tagset=codes.sha256_file(ext[3]),
                                 wordnet=codes.sha256_file(wn_zip), iters=S["iters"]),
        lambda: codes.predefined(tr, vocab, eos, S["iters"], log=run.print), run.print)
    T["code_predefined"] = time.time() - t
    t = time.time()
    kf, kc = S["topic"]
    C["topic"], M["codes"]["topic"] = codes.get(
        pre + "topic", src, lambda: codes.topic(tr, blocks["train"], V, eos, kf, kc), run.print)
    T["code_topic"] = time.time() - t
    C["orgroups"], M["codes"]["orgroups"] = codes.get(
        pre + "orgroups", dict(src, groups=codes.sha256_file(codes.ORGROUPS_FILE)),
        lambda: codes.orgroups(vocab, eos), run.print)
    levels_of = {n: c.shape[1] for n, c in C.items()}

    # ---- word KN5 and cache ----
    t = time.time()
    kn5 = KN(tr, V, eos, order=5, mode="sent")
    T["kn5_build"] = time.time() - t
    splits = ("calib", "valid") if ev == "valid" else ("calib", "valid", "devtest")
    p_kn = {s: np.exp(kn5.logprob(ids[s])) for s in splits}
    t = time.time()
    cch, grid = tune_cache(V, eos, ids, blocks, p_kn["calib"], p_kn["valid"])
    T["cache_tune"] = time.time() - t
    M["cache"] = {"chosen": cch.config(), "grid": grid}
    run.print("cache chosen on valid:", cch.config())
    p_c = {s: cch.prob(ids[s], blocks[s], p_kn[s])[0] for s in ("calib", ev)}

    # ---- class members ----
    arms = arms_of(levels_of)
    needed = sorted({m for v in arms.values() for m in v})
    P_mem, members = {}, {}
    t = time.time()
    for code, l in needed:
        t1 = time.time()
        cm = ClassMember(tr, C[code][:, l], eos, CLASS_ORDER, DELTA)
        members[(code, l)] = cm
        P_mem[(code, l)] = {s: np.exp(cm.logprob(ids[s])) for s in ("calib", ev)}
        M["members"][f"{code}:{l}"] = dict(cm.summary(), seconds=round(time.time() - t1, 2))
    T["class_members_build_and_score"] = time.time() - t
    run.print(f"{len(needed)} class members built and scored in {T['class_members_build_and_score']:.1f} s")

    # ---- arms ----
    unseen = codeeval.unseen_bigram_mask(tr, ids[ev], eos, V)
    t = time.time()
    dup_r, dup = stats.duplicate_report(tr, blocks["train"], ids[ev], ev, 8)
    T["duplicates"] = time.time() - t
    M["duplicates"] = dup_r
    M["unseen_bigram_positions"] = {"tokens": int(unseen.sum()), "share": float(unseen.mean())}
    LP, R, lams = {}, {}, {}
    t = time.time()
    for arm, mem in arms.items():
        cols = lambda s: [p_kn[s], p_c[s]] + [P_mem[m][s] for m in mem]  # noqa: E731
        lam, trace = mix.em(np.stack(cols("calib"), 1))
        lp = mix.logprob(np.stack(cols(ev), 1), lam)
        LP[arm], lams[arm] = lp, lam
        R[arm] = {"members": ["kn5", "cache"] + [f"{c}:{l}" for c, l in mem], "lam": lam.round(5).tolist(),
                  "em_steps": len(trace), "all": stats.report(lp, ev),
                  "unseen_bigram": stats.report(lp[unseen], ev + " (bigram unseen in train)"),
                  "dup_removed": stats.report(lp[~dup], ev + " (duplicate tokens removed)")}
    T["em_all_arms"] = time.time() - t
    bv = blocks[ev]
    base = R["none"]["all"]["ppl"]
    for arm in arms:
        R[arm]["rel_ppl_vs_none"] = R[arm]["all"]["ppl"] / base - 1
        R[arm]["rel_ppl_vs_none_unseen"] = R[arm]["unseen_bigram"]["ppl"] / R["none"]["unseen_bigram"]["ppl"] - 1
        if arm != "none":
            R[arm]["bits_diff_vs_none"] = stats.block_bootstrap(stats.bits_diff(LP[arm], LP["none"]), bv)
    M["arms"] = R

    # ---- random codes, best code, margin, dev reading ----
    rnd = [f"random_s{s}" for s in range(1, N_RANDOM + 1)]
    r_ppl = np.array([R[r]["all"]["ppl"] for r in rnd])
    lp_rmean = np.mean([LP[r] for r in rnd], 0)
    best = min(CANDIDATES, key=lambda x: R[x]["all"]["ppl"])
    gain = 1 - R[best]["all"]["ppl"] / base
    marg = 1 - R[best]["all"]["ppl"] / r_ppl.mean()
    bb = stats.block_bootstrap(stats.bits_diff(LP[best], lp_rmean), bv)
    M["random"] = {"ppl_each": r_ppl.tolist(), "ppl_mean": float(r_ppl.mean()), "ppl_sd": float(r_ppl.std(ddof=1)),
                   "ppl_min": float(r_ppl.min()), "ppl_max": float(r_ppl.max()),
                   "rel_ppl_mean_vs_none": float(r_ppl.mean() / base - 1)}
    M["reading"] = {
        "label": "development reading on " + ev + ("" if a.final else "; not a verdict"),
        "best_code": best, "best_ppl": R[best]["all"]["ppl"], "none_ppl": base,
        "gain_vs_none": gain, "margin_vs_random_mean": marg, "best_minus_random_mean_bits": bb,
        "success_line": bool(gain >= SUCCESS_PPL and marg >= MARGIN_PPL and bb["hi"] < 0),
        "kill_line": bool(gain < KILL_PPL)}

    # ---- contract and future-change test ----
    t = time.time()
    M["contract"] = {}
    for arm in ("none", "kind_topic"):
        lam, mem = lams[arm], arms[arm]
        for s in ("calib", ev):
            def dfn(pos, s=s, lam=lam, mem=mem):
                Dk = kn5.dists(ids[s], pos)
                D = lam[0] * Dk + lam[1] * cch.dists(ids[s], blocks[s], pos, Dk)
                for j, m in enumerate(mem):
                    D += lam[2 + j] * members[m].dists(ids[s], pos)
                return D
            cols = [p_kn[s], p_c[s]] + [P_mem[m][s] for m in mem]
            fast = mix.logprob(np.stack(cols, 1), lam)
            r = stats.contract(dfn, fast, len(ids[s]), 1000, seed=11, true_ids=ids[s])
            M["contract"][f"{arm}_{s}"] = r
            run.print(f"contract {arm} {s}", r)
    T["contract"] = time.time() - t

    def system_logp(x, bl, arm):
        pk = np.exp(kn5.logprob(x))
        cols = [pk, cch.prob(x, bl, pk)[0]] + [np.exp(members[m].logprob(x)) for m in arms[arm]]
        return mix.logprob(np.stack(cols, 1), lams[arm])
    t = time.time()
    rng = np.random.default_rng(5)
    x0, b0 = ids[ev], blocks[ev]
    ref = system_logp(x0, b0, "kind_topic")
    worst = 0.0
    cuts = np.sort(rng.choice(np.arange(10, len(x0) - 10), size=10, replace=False))
    for cut in cuts:
        x1 = x0.copy()
        x1[cut + 1:] = rng.integers(0, V, len(x1) - cut - 1)
        x1[cut + 1:][x0[cut + 1:] == eos] = eos       # keep the sentence ends
        worst = max(worst, float(np.abs(system_logp(x1, b0, "kind_topic")[:cut + 1] - ref[:cut + 1]).max()))
    M["future_change_test"] = {"arm": "kind_topic", "cuts": cuts.tolist(), "max_abs_logp_change_before_cut": worst,
                               "pass": bool(worst == 0.0)}
    M["shuffled_label_test"] = "the 10 random codes are the shuffled-label control (word labels permuted)"
    T["future_change_test"] = time.time() - t
    run.print("future-change test", M["future_change_test"])

    # ---- code reports (no verdict) ----
    t = time.time()
    counts = np.bincount(tr, minlength=V)
    sl, ws = codeeval.load_simlex(), codeeval.load_ws353()
    syn = codeeval.synonym_pairs(vocab, counts, 10 if not a.smoke else 1)
    rep = {}
    for n in ["exchange", "random_s1", "kmeans200", "predefined", "topic", "orgroups"] + rnd[1:]:
        c = C[n]
        rep[n] = {"classes": codes.n_classes(c), "simlex": codeeval.similarity(c, w2i, sl),
                  "ws353": codeeval.similarity(c, w2i, ws), "sense_test": codeeval.sense_test(c, syn)}
    rep["kind_topic"] = {"note": "exchange and topic codes together",
                         "simlex": codeeval.similarity(np.concatenate([C["exchange"], C["topic"]], 1), w2i, sl),
                         "ws353": codeeval.similarity(np.concatenate([C["exchange"], C["topic"]], 1), w2i, ws)}
    for key in ("simlex", "ws353"):
        rep["random_mean_" + key] = float(np.mean([rep[r][key]["spearman"] for r in rnd]))
    rep["sense_test_synonym_pairs"] = len(syn)
    growth = {"words": codeeval.ngram_growth(tr, ids[ev], eos, V + 1)}
    for n in ("exchange", "random_s1", "kmeans200", "predefined", "topic", "orgroups"):
        for l in range(C[n].shape[1]):
            c = C[n][:, l]
            growth[f"{n}:{l}"] = codeeval.ngram_growth(c[tr], c[ids[ev]], int(c[eos]), int(c.max()) + 2)
    rep["ngram_growth"] = growth
    M["code_reports"] = rep
    T["code_reports"] = time.time() - t

    # ---- table ----
    run.print("")
    run.print(f"{'arm':18s} {'members':>7s} {'ppl':>9s} {'bits':>8s} {'vs none':>8s} {'95% CI bits diff':>21s} "
              f"{'unseen ppl':>11s} {'vs none':>8s}   ({ev}, {len(ids[ev])} tokens incl. <eos>, V = {V}; "
              f"unseen-bigram positions {int(unseen.sum())})")
    for arm in arms:
        r = R[arm]
        ci = r.get("bits_diff_vs_none")
        cis = f"{ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]" if ci else ""
        run.print(f"{arm:18s} {len(arms[arm]):7d} {r['all']['ppl']:9.3f} {r['all']['bits_per_token']:8.4f} "
                  f"{100 * r['rel_ppl_vs_none']:+7.2f}% {cis:>21s} {r['unseen_bigram']['ppl']:11.1f} "
                  f"{100 * r['rel_ppl_vs_none_unseen']:+7.2f}%")
    run.print("random codes:", {k: round(v, 3) if isinstance(v, float) else v for k, v in M["random"].items()
                                 if k != "ppl_each"})
    run.print("reading:", M["reading"])
    for n in ("exchange", "random_s1", "kmeans200", "predefined", "topic", "orgroups", "kind_topic"):
        run.print(f"code {n}: simlex {rep[n]['simlex']}, ws353 {rep[n]['ws353']}")
    M["timings_s"] = {k_: round(v, 2) for k_, v in T.items()}
    run.print("timings", M["timings_s"])
    run.finish(M)


if __name__ == "__main__":
    main()
