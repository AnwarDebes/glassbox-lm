"""A1 baseline check: valid perplexity of KN3, KN5 and KN5 + cache, own code and KenLM.

    python scripts/a1_baseline_check.py

A baseline measurement, not a registered run. Counts from train only; the mixing weights are
fitted by EM on calib; the cache settings (half-life, <eos> added or not) are hyperparameters and
are chosen on valid (design rule 2: valid is the tuning split); numbers are reported on valid.
Dev-test is not scored.
Also runs the section 4 contract (1,000 random positions per split: train, calib, valid), the
duplicate report for calib and valid, and two example block bootstraps on valid.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tally import ptb  # noqa: E402
from tally.floor import cache, kenlm, mix, stats  # noqa: E402
from tally.floor.kn import KN  # noqa: E402
from tally.runlog import Run  # noqa: E402

HALF_LIVES = (25, 50, 100, 200, 400, 800, 1600, 3200, np.inf)
CONTRACT_SPLITS = ("train", "calib", "valid")


def tune_cache(V, eos, ids, blocks, p_cal, p_val, reset):
    """Grid over half-life and skip (<eos> added or not). For every setting the weights of the
    KN and cache members are fitted by EM on calib; the setting is chosen by valid likelihood."""
    best = None
    rows = []
    for hl in HALF_LIVES:
        for skip in ((), (eos,)):
            c = cache.Cache(V, hl, reset, skip)
            pc, _ = c.prob(ids["calib"], blocks["calib"], p_cal)
            P = np.stack([p_cal, pc], 1)
            lam, tr = mix.em(P)
            pv, _ = c.prob(ids["valid"], blocks["valid"], p_val)
            llv = float(np.log(np.stack([p_val, pv], 1) @ lam).sum())
            rows.append({"half_life": float(hl), "skip_eos": bool(skip), "lam": lam.tolist(),
                         "calib_bits": -float(np.log(P @ lam).sum()) / len(pc) / np.log(2),
                         "valid_bits": -llv / len(pv) / np.log(2), "em_steps": len(tr)})
            if best is None or llv > best[0]:
                best = (llv, c, lam)
    return best[1], best[2], rows


def main():
    cfg = {"orders": [3, 5], "mode": "sent", "half_lives": [float(h) for h in HALF_LIVES],
           "resets": list(cache.RESETS), "contract_positions": 1000, "contract_splits": CONTRACT_SPLITS,
           "bootstrap": {"n_boot": 1000, "level": 0.95, "seed": 0}, "dup_min_len": 8,
           "cache_select": "valid", "em_fit": "calib", "rule_0p5_applies_to": "KN5 (KN3 reported only)",
           "count_edges": list(mix.COUNT_EDGES), "bucket_tau": 20.0}
    run = Run("a1_baseline_check", cfg, data_files=[ptb.path("train"), ptb.path("valid")])
    T = {}
    t = time.time()
    vocab, w2i, ids, blocks = ptb.load()
    T["load"] = time.time() - t
    V, eos = len(vocab), w2i["<eos>"]
    M = {"splits": {s: int(len(ids[s])) for s in ("train", "calib", "valid")}, "V": V}
    run.print("tokens", M["splits"], "V", V)

    # own KN
    own, lp = {}, {}
    for N in (3, 5):
        t = time.time()
        m = KN(ids["train"], V, eos, order=N, mode="sent")
        T[f"own_kn{N}_build"] = time.time() - t
        own[N] = m
        for s in ("calib", "valid"):
            t = time.time()
            lp[("own", N, s)] = m.logprob(ids[s])
            T[f"own_kn{N}_score_{s}"] = time.time() - t
        run.print(f"own kn{N}", m.summary())
    # contract for KN
    M["contract"] = {}
    for N in (3, 5):
        for s in CONTRACT_SPLITS:
            m = own[N]
            fast = m.logprob(ids[s])
            t = time.time()
            r = stats.contract(lambda pos: m.dists(ids[s], pos), fast, len(ids[s]), 1000, seed=N,
                               true_ids=ids[s])
            r["seconds"] = time.time() - t
            M["contract"][f"own_kn{N}_{s}"] = r
            run.print(f"contract own kn{N} {s}", r)

    # KenLM on the same reduced train text
    D = os.path.join(kenlm.STUDY, "data", "a1_kenlm")
    os.makedirs(D, exist_ok=True)
    for s in ("train", "calib", "valid"):
        kenlm.write_text(ids[s], vocab, eos, os.path.join(D, f"{s}.sent.txt"))
    M["kenlm"] = {}
    for N in (3, 5):
        arpa, binf = os.path.join(D, f"kn{N}.arpa"), os.path.join(D, f"kn{N}.bin")
        t = time.time()
        disc = kenlm.lmplz(N, os.path.join(D, "train.sent.txt"), arpa, os.path.join(D, "tmp"))
        kenlm.build_binary(arpa, binf)
        T[f"kenlm_kn{N}_build"] = time.time() - t
        n_uni = None
        with open(arpa) as f:
            for line in f:
                if line.startswith("ngram 1="):
                    n_uni = int(line.split("=")[1])
                    break
        oovs = {}
        for s in ("calib", "valid"):
            t = time.time()
            l, oov = kenlm.query(binf, os.path.join(D, f"{s}.sent.txt"))
            T[f"kenlm_kn{N}_query_{s}"] = time.time() - t
            assert len(l) == len(ids[s])
            lp[("kenlm", N, s)] = l
            oovs[s] = int(oov.sum())
        # diagnosis: own model with KenLM's uniform floor (its unigrams minus <s>)
        diag = KN(ids["train"], V, eos, order=N, n_uniform=n_uni - 1).logprob(ids["valid"])
        k, o = lp[("kenlm", N, "valid")], lp[("own", N, "valid")]
        M["kenlm"][f"kn{N}"] = {
            "discounts_kenlm": disc,
            "discounts_own": {str(k_): own[N].summary()["discounts"][k_] for k_ in range(1, N + 1)},
            "unigrams_in_arpa": n_uni, "oov_tokens": oovs,
            "valid_rel_ppl_own_vs_kenlm": float(np.exp(k.mean() - o.mean()) - 1),
            "valid_max_abs_logp_diff": float(np.abs(k - o).max()),
            "valid_rel_ppl_own_kenlm_floor_vs_kenlm": float(np.exp(k.mean() - diag.mean()) - 1),
            "valid_max_abs_logp_diff_kenlm_floor": float(np.abs(k - diag).max())}
        run.print(f"kenlm kn{N}", M["kenlm"][f"kn{N}"])

    # cache: weights by EM on calib, settings chosen on valid, for own KN5 and KenLM KN5
    M["cache"] = {}
    for src in ("own", "kenlm"):
        for reset in cache.RESETS:
            t = time.time()
            p_cal = np.exp(lp[(src, 5, "calib")])
            p_val = np.exp(lp[(src, 5, "valid")])
            c, lam, rows = tune_cache(V, eos, ids, blocks, p_cal, p_val, reset)
            pc, empty = c.prob(ids["valid"], blocks["valid"], p_val)
            lp[(src, "cache_" + reset, "valid")] = mix.logprob(np.stack([p_val, pc], 1), lam)
            T[f"{src}_cache_{reset}_tune_and_score"] = time.time() - t
            M["cache"][f"{src}_{reset}"] = {"chosen": c.config(), "lam": lam.tolist(), "grid": rows,
                                           "valid_empty_cache_tokens": int(empty.sum())}
            run.print(f"cache {src} {reset}: chosen {c.config()} lam {lam.round(4).tolist()}")
            if src == "own" and reset == "block":
                best_c, best_lam = c, lam

    # contract for the mixture own KN5 + cache (block reset)
    m5 = own[5]
    for s in CONTRACT_SPLITS:
        pk = np.exp(m5.logprob(ids[s]))
        pc, _ = best_c.prob(ids[s], blocks[s], pk)
        fast = mix.logprob(np.stack([pk, pc], 1), best_lam)

        def dfn(pos, s=s):
            Dk = m5.dists(ids[s], pos)
            return best_lam[0] * Dk + best_lam[1] * best_c.dists(ids[s], blocks[s], pos, Dk)
        t = time.time()
        r = stats.contract(dfn, fast, len(ids[s]), 1000, seed=7, true_ids=ids[s])
        r["seconds"] = time.time() - t
        M["contract"][f"own_kn5_cache_block_{s}"] = r
        run.print(f"contract own kn5+cache {s}", r)

    # bucketed interpolation: correctness only (fit on calib, contract on calib)
    t = time.time()
    pk = np.exp(lp[("own", 5, "calib")])
    pc, _ = best_c.prob(ids["calib"], blocks["calib"], pk)
    P = np.stack([pk, pc], 1)
    L, cnt = m5.match(ids["calib"])
    b = mix.buckets(L, cnt)
    nb = mix.n_buckets(5)
    lamb, trace = mix.em_bucketed(P, b, nb, best_lam, tau=cfg["bucket_tau"])
    fastb = mix.logprob(P, lamb, b)

    def dfb(pos):
        Dk = m5.dists(ids["calib"], pos)
        Dc = best_c.dists(ids["calib"], blocks["calib"], pos, Dk)
        lb = lamb[b[pos]]
        return lb[:, :1] * Dk + lb[:, 1:] * Dc
    rb = stats.contract(dfb, fastb, len(ids["calib"]), 1000, seed=9, true_ids=ids["calib"])
    T["bucketed_fit_and_contract"] = time.time() - t
    M["bucketed_correctness"] = {
        "contract_calib": rb, "em_monotone": bool(np.all(np.diff(trace) >= -1e-6)),
        "em_steps": len(trace), "weights_sum_to_one": bool(np.allclose(lamb.sum(1), 1)),
        "buckets_used": int(len(np.unique(b))), "buckets": nb}
    run.print("bucketed (correctness only)", M["bucketed_correctness"])

    # perplexity table (valid)
    table = {}
    for src in ("own", "kenlm"):
        for key, name in ((3, "KN3"), (5, "KN5"), ("cache_block", "KN5 + cache (reset per block)"),
                          ("cache_none", "KN5 + cache (no reset)")):
            table[f"{src}: {name}"] = stats.report(lp[(src, key, "valid")], "valid")
    M["valid_table"] = table
    run.print("")
    run.print(f"{'model':42s} {'ppl':>9s} {'bits/token':>11s}  (valid, {len(ids['valid'])} tokens incl. <eos>, V = {V})")
    for k_, r in table.items():
        run.print(f"{k_:42s} {r['ppl']:9.3f} {r['bits_per_token']:11.5f}")

    # example block bootstraps on valid
    bv = blocks["valid"]
    M["bootstrap_examples"] = {
        "own_kn5_cache_block_minus_own_kn5": stats.block_bootstrap(
            stats.bits_diff(lp[("own", "cache_block", "valid")], lp[("own", 5, "valid")]), bv),
        "own_kn5_minus_kenlm_kn5": stats.block_bootstrap(
            stats.bits_diff(lp[("own", 5, "valid")], lp[("kenlm", 5, "valid")]), bv)}
    for k_, r in M["bootstrap_examples"].items():
        run.print("bootstrap", k_, r)

    # duplicate report (calib and valid)
    t = time.time()
    M["duplicates"] = {}
    for s in ("calib", "valid"):
        r, mask = stats.duplicate_report(ids["train"], blocks["train"], ids[s], s, 8)
        r["own_kn5_ppl_dup_removed"] = stats.report(lp[("own", 5, s)][~mask], s)["ppl"]
        M["duplicates"][s] = r
        run.print("duplicates", r)
    T["duplicates"] = time.time() - t
    M["timings_s"] = {k_: round(v, 3) for k_, v in T.items()}
    run.print("timings", M["timings_s"])
    run.finish(M)


if __name__ == "__main__":
    main()
