"""NPMI coherence of learned OR-groups vs frequency-matched random groups (gates.md Check 2c; CPU only).

NPMI(i,j) = log(P(i,j) / (P(i) P(j))) / -log P(i,j) on the NPMI reference texts (data/agnews/npmi_ref.npz, texts in
no split), P = share of texts that contain the word(s); -1 when two words never occur together. A group's score is
the mean NPMI over its word pairs.
Counted OR-groups (gates.md): OR-groups of positive clauses, not always-true, repeated words removed, at least 2
distinct words. Top OR-groups (E5a, pre-registered): the counted groups of each topic's top 10 positive clauses by
dev fire-rate difference (rank <= 10 in groups.json, the order of rules.txt). Flat runs have no OR-groups.
Random groups: per counted group i, K = 1,000 random groups of the same size; each word is replaced by a vocabulary
word from the same frequency bin (10 equal-count bins of reference document frequency), distinct within a group.
GATE METHOD (frozen, gates.md): d_i = s_i - mean(r_i); statistic T = mean d over groups; percentile group
  bootstrap (1,000 resamples of the groups) -> "ci95_group_bootstrap"; "beats_random" = its lower bound > 0.
  This interval is too narrow with few groups (null false-positive rates below, nominal 0.025).
EXTRA, CALIBRATED (reported next to the gate and not used for the verdict; method added on 2 Oct 2026):
  "p_mc": exact Monte Carlo randomization test. Per group the pool v_i = [s_i, r_i1..r_iK]; in each of B = 2,000 null
     replicates u is drawn uniformly from 0..K for every group, d_i^b = v_i[u] - mean(v_i without u),
     T_b = mean_i d_i^b; one-sided p = (1 + #{T_b >= T}) / (1 + B) (fixed seed).
  "t_ci95": two-sided 95% t-interval over groups, mean d +- t(0.975, n-1) sd / sqrt(n).
  "beats_random_calibrated" = p_mc < 0.025 and t lower bound > 0; None with fewer than 12 groups
     ("n_groups_note": "fewer than 12 groups: weak").
compare_two(a, b) (one set of groups vs another, e.g. a shuffled-label model's groups; for Checks 4a/4b): difference
  of mean group scores. Gate: group bootstrap of the difference (groups resampled within each set, 1,000 times),
  "a_beats_b" = lower bound > 0. Extra: one-sided permutation test over group labels ("p_perm", B = 2,000), Welch 95%
  t-interval ("welch_ci95"), "a_beats_b_calibrated" (both sets >= 12 groups, p < 0.025, Welch lower bound > 0).
Calibration (toy_check, logged in results/*_e5a_cpu_tests_v2_exploratory; null groups drawn like the random groups,
1,000 replicates each; share of replicates with a false "beats random", nominal 0.025):
  6 groups:  gate bootstrap 0.204; calibrated: MC test 0.019, t-interval alone 0.138, both together 0.011;
  30 groups: gate bootstrap 0.087; calibrated: MC test 0.035, t-interval alone 0.085, both together 0.035;
  compare_two, 12 vs 12 groups: gate bootstrap 0.043; permutation test 0.025, full calibrated verdict 0.016.
  So the frozen bootstrap says "beats random" far too often with few groups; the calibrated test stays near 0.025.
Usage: python npmi_groups.py RUN_DIR [RUN_DIR ...]   (each hierarchical run, then all of them pooled)
       python npmi_groups.py --toy                    (self-test on a toy corpus; test_e5a.py runs it too)
"""
import argparse, json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy import stats

SRC = os.path.dirname(os.path.abspath(__file__))
MIN_GROUPS, ALPHA = 12, 0.025


def npmi_matrix(X):
    """X: 0/1 [texts, V]. Returns (NPMI [V, V], document frequencies [V]); the diagonal is not used."""
    X = sp.csr_matrix(X, dtype=np.float64)
    co = (X.T @ X).toarray()
    df, n = np.diag(co).copy(), X.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.log(co * n / np.outer(df, df)) / -np.log(co / n)
    m[co == 0] = -1.0
    m[co == n] = 1.0  # both words in every text (0/0)
    return m, df


def score(ids, M):
    iu = np.triu_indices(len(ids), 1)
    return float(M[np.ix_(ids, ids)][iu].mean())


def freq_bins(df, n_bins=10):
    """Equal-count frequency bins (by rank, so ties split evenly)."""
    rank = np.argsort(np.argsort(df, kind="stable"), kind="stable")
    return rank * n_bins // len(df)


def draw_matched(ids, bins, rng, draws):
    """[draws, len(ids)] word ids; column i is drawn from the frequency bin of ids[i]; distinct within a row."""
    ids = np.asarray(ids)
    R = np.empty((draws, len(ids)), np.int64)
    for b in np.unique(bins[ids]):
        pos, members = np.flatnonzero(bins[ids] == b), np.flatnonzero(bins == b)
        assert len(pos) <= len(members), "group has more words in one frequency bin than the bin holds"
        R[:, pos] = members[np.argsort(rng.random((draws, len(members))), axis=1)[:, :len(pos)]]
    return R


def random_scores(ids, bins, M, rng, draws=1000):
    """Scores of `draws` random groups matched word by word on frequency bin."""
    R, iu = draw_matched(ids, bins, rng, draws), np.triu_indices(len(ids), 1)
    return np.concatenate([M[R[c:c + 100, :, None], R[c:c + 100, None, :]][:, iu[0], iu[1]].mean(1)
                           for c in range(0, draws, 100)])


def mc_test(V, B=2000, seed=0):
    """V [n, K+1]: column 0 = each group's score, columns 1..K = its matched random scores. Returns (T, one-sided p)."""
    n, K = V.shape[0], V.shape[1] - 1
    S = V.sum(1)
    T = float(np.mean(V[:, 0] * (1 + 1 / K) - S / K))  # = mean_i (s_i - mean r_i)
    U = np.random.default_rng(seed).integers(0, K + 1, (B, n))
    Tb = (V[np.arange(n), U] * (1 + 1 / K) - S / K).mean(1)
    return T, float((1 + (Tb >= T).sum()) / (1 + B))


def t_interval(d):
    if len(d) < 2:
        return [float("nan"), float("nan")]
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - h), float(d.mean() + h)]


def calibrated(enough, p, lo):
    return (bool(p < ALPHA and lo > 0) if enough else None), ("" if enough else f"fewer than {MIN_GROUPS} groups: weak")


def compare(groups, M, bins, draws=1000, boots=1000, seed=0, B=2000):
    """Groups vs frequency-matched random groups: the gate (group bootstrap) and the calibrated extra fields."""
    if not groups:
        return {"n_groups": 0, "beats_random": None, "beats_random_calibrated": None, "n_groups_note": "no counted groups"}
    rng = np.random.default_rng(seed)
    V = np.array([np.r_[score(g, M), random_scores(g, bins, M, rng, draws)] for g in groups])  # [n, 1 + K]
    d = V[:, 0] - V[:, 1:].mean(1)
    lo, hi = np.percentile(d[rng.integers(0, len(d), (boots, len(d)))].mean(1), [2.5, 97.5])
    T, p = mc_test(V, B, seed + 1)
    tci = t_interval(d)
    ok, note = calibrated(len(groups) >= MIN_GROUPS, p, tci[0])
    return {"n_groups": len(groups), "mean_size": round(float(np.mean([len(g) for g in groups])), 2),
            "npmi_groups": float(V[:, 0].mean()), "npmi_random": float(V[:, 1:].mean()), "diff": float(d.mean()),
            "ci95_group_bootstrap": [float(lo), float(hi)], "beats_random": bool(lo > 0),
            "p_mc": p, "t_ci95": tci, "beats_random_calibrated": ok, "n_groups_note": note}


def compare_two(groups_a, groups_b, M, boots=1000, seed=0, B=2000):
    """Mean group score of set a minus set b: the gate (group bootstrap) and the calibrated extra fields."""
    a, b = np.array([score(g, M) for g in groups_a]), np.array([score(g, M) for g in groups_b])
    na, nb, T = len(a), len(b), float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    bd = a[rng.integers(0, na, (boots, na))].mean(1) - b[rng.integers(0, nb, (boots, nb))].mean(1)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    pool = np.r_[a, b]
    perm = pool[np.argsort(np.random.default_rng(seed + 1).random((B, na + nb)), axis=1)]
    p = float((1 + ((perm[:, :na].mean(1) - perm[:, na:].mean(1)) >= T).sum()) / (1 + B))
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    dfw = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)) if va + vb > 0 else 1.0
    h = stats.t.ppf(0.975, dfw) * np.sqrt(va + vb)
    ok, note = calibrated(min(na, nb) >= MIN_GROUPS, p, T - h)
    return {"n_a": na, "n_b": nb, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": T,
            "ci95_group_bootstrap": [float(lo), float(hi)], "a_beats_b": bool(lo > 0),
            "p_perm": p, "welch_ci95": [float(T - h), float(T + h)], "a_beats_b_calibrated": ok, "n_groups_note": note}


def counted_groups(gj, top):
    """(topic, words) of the counted OR-groups in each topic's positive clauses of rank <= top."""
    return [(c["name"], g["words"]) for c in gj["classes"] for d in c["clauses"] if d["rank"] <= top
            for g in d["groups"] if not g["always_true"] and len(g["words"]) >= 2]


def toy_check(reps=1000):
    """Exact NPMI values, frequency matching, planted clusters (must pass under both methods; 6 groups: no calibrated
    verdict), and the null false-positive rates of both methods (groups drawn like the random groups). Asserted: the
    calibrated tests stay at or below 0.04. The bootstrap's rates are only reported (known undercoverage)."""
    M, _ = npmi_matrix(np.array([[1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0], [0, 0, 0, 0, 1]]))
    assert abs(M[0, 1]) < 1e-12 and abs(M[2, 3] - 1) < 1e-12 and M[2, 4] == -1  # independent, always together, never
    rng = np.random.default_rng(7)
    n, topic = 6000, rng.integers(0, 2, 6000)
    X = rng.random((n, 60)) < 0.12  # words 20-59: noise at rate 0.12; words 0-9 / 10-19: topic 0 / 1 words, same rate
    X[:, :20] = (rng.random((n, 20)) < 0.24) & (topic[:, None] == np.repeat([0, 1], 10)[None, :])
    M, df = npmi_matrix(X.astype(np.uint8))
    bins = freq_bins(df)
    assert np.bincount(bins).min() == np.bincount(bins).max() == 6
    R = draw_matched([0, 1, 2, 30], bins, np.random.default_rng(0), 200)
    assert (bins[R] == bins[[0, 1, 2, 30]]).all() and all(len(set(r)) == 4 for r in R.tolist())

    def null(g, k):
        return [list(draw_matched(list(g.choice(60, z, replace=False)), bins, g, 1)[0]) for z in g.integers(3, 6, k)]

    p6 = compare([[0, 1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18, 19]], M, bins, 300, 1000, 1)
    assert p6["beats_random"] and p6["p_mc"] < ALPHA and p6["t_ci95"][0] > 0, p6
    assert p6["beats_random_calibrated"] is None and p6["n_groups_note"] == "fewer than 12 groups: weak", p6
    base = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0], [1, 3, 5], [5, 7, 9]]
    planted14 = base + [[w + 10 for w in g] for g in base]
    p14 = compare(planted14, M, bins, 300, 1000, 2)
    assert p14["beats_random"] and p14["beats_random_calibrated"] is True, p14
    t14 = compare_two(planted14, null(np.random.default_rng(3), 14), M, 1000, 3)
    assert t14["a_beats_b"] and t14["a_beats_b_calibrated"] is True, t14
    fpr = {}
    for k in (6, 30):
        hits = np.zeros((reps, 4), bool)
        for i in range(reps):
            r = compare(null(np.random.default_rng(100_000 * k + i), k), M, bins, 200, 1000, i)
            hits[i] = r["beats_random"], r["p_mc"] < ALPHA, r["t_ci95"][0] > 0, r["p_mc"] < ALPHA and r["t_ci95"][0] > 0
        fpr[f"{k}_groups"] = dict(zip(["group_bootstrap_gate", "mc_test", "t_interval", "calibrated_without_minimum"],
                                      map(float, hits.mean(0))))
        assert fpr[f"{k}_groups"]["mc_test"] <= 0.04 and fpr[f"{k}_groups"]["calibrated_without_minimum"] <= 0.04, fpr
    two = np.zeros((reps, 3), bool)
    for i in range(reps):
        g = np.random.default_rng(900_000 + i)
        r = compare_two(null(g, 12), null(g, 12), M, 1000, i)
        two[i] = r["a_beats_b"], r["p_perm"] < ALPHA, bool(r["a_beats_b_calibrated"])
    fpr["compare_two_12_vs_12"] = dict(zip(["group_bootstrap_gate", "permutation_test", "calibrated_full"], map(float, two.mean(0))))
    assert fpr["compare_two_12_vs_12"]["permutation_test"] <= 0.04 and fpr["compare_two_12_vs_12"]["calibrated_full"] <= 0.04, fpr
    return {"planted_6": p6, "planted_14": p14, "planted_14_vs_null_14": t14, "null_false_positive_rates": fpr,
            "null_replicates": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--draws", type=int, default=1000, help="random groups per counted group (K)")
    ap.add_argument("--boots", type=int, default=1000, help="group bootstrap resamples (the gate method)")
    ap.add_argument("--mc", type=int, default=2000, help="Monte Carlo null replicates (B) of the calibrated test")
    a = ap.parse_args()
    out = {}
    if a.toy:
        out = toy_check()
    else:
        sys.path.insert(0, SRC)
        from text_driver import DATA, load_split
        vocab = json.load(open(os.path.join(DATA, "vocab.json")))
        wid = {w: i for i, w in enumerate(vocab)}
        M, df = npmi_matrix(load_split(DATA, "npmi_ref")[0])
        bins, pooled = freq_bins(df), []
        for d in a.runs:
            gj = json.load(open(os.path.join(d, "groups.json")))
            if gj["model"] != "hier":
                continue
            top = [[wid[w] for w in g] for _, g in counted_groups(gj, a.top)]
            alls = [[wid[w] for w in g] for _, g in counted_groups(gj, 10 ** 9)]
            out[os.path.basename(d.rstrip("/"))] = {"top": compare(top, M, bins, a.draws, a.boots, B=a.mc),
                                                    "all": compare(alls, M, bins, a.draws, a.boots, B=a.mc)}
            pooled += top
        out["pooled_top"] = compare(pooled, M, bins, a.draws, a.boots, B=a.mc)
    print(json.dumps(out, indent=1))
    run_dir = os.environ.get("TMLM_RUN_DIR")
    if run_dir:
        json.dump({k: {"value": v, "label": "measured"} for k, v in out.items()}, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
