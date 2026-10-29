"""Step 0 of C-core (PLAN_part3.md section 4; proposal C section 4): free and exploratory, no claim and
no gate. Each frozen T0 arm (tally results, 10 seeds; the untrained banks 10 draws per seed) is mixed
with the floor F: P = lambda[b] F(w | h) + (1 - lambda[b]) P_T0(w | h), weights per bucket fitted on calib
(the floor's 40 buckets, tau 20, MAP-EM). This is exact through the true-word probabilities alone,
because every T0 vector over V sums to 1 (T0's contract). Scored on valid and dev-test; per arm the
per-position bits are averaged over seeds (and draws) and compared with F by the paired block
bootstrap (100-sentence blocks, 1,000 resamples, seed 0). First, every loaded array's mean must equal
T0's logged bits per word (|difference| <= 1e-9) and its block ids must equal the tally splits'."""
import json
import os

import numpy as np

from . import hashing, systems

SPLITS = ("calib", "valid", "devtest")
SCORED = ("valid", "devtest")
UNTRAINED = tuple(f"untrained{d}" for d in range(10))
ARMS = {
    "coal_learned": ("coal", ("learned",)),
    "coal_counted": ("coal", ("counted",)),
    "coal_untrained": ("coal", UNTRAINED),
    "logreg_on_clauses": ("logreg", ("logreg",)),
    "flat_learned": ("flat", ("learned",)),
    "flat_counted": ("flat", ("counted",)),
}
CONTRASTS = (("coal_learned", "coal_untrained"), ("coal_counted", "coal_untrained"),
             ("coal_learned", "coal_counted"), ("logreg_on_clauses", "coal_learned"))
MEAN_TOL = 1e-9


def load_t0(P, log=print):
    """{(kind, seed): {key__split: bits array}} with the mean and block checks done."""
    runs = hashing.t0_runs()
    out, checks = {}, []
    for (kind, seed), d in sorted(runs.items()):
        z = np.load(os.path.join(d, "bits.npz"))
        with open(os.path.join(d, "metrics.json")) as f:
            met = json.load(f)
        keys = ("logreg",) if kind == "logreg" else (("learned", "counted") + (UNTRAINED if kind == "coal" else ()))
        arrs = {}
        for k in keys:
            for s in SPLITS:
                a = z[f"{k}__{s}"].astype(np.float64)
                logged = met[f"{k}_{s}_bpw"]
                err = abs(float(a.mean()) - float(logged))
                assert len(a) == len(P["ids"][s]), (kind, seed, k, s)
                if err > MEAN_TOL:
                    raise SystemExit(f"T0 {kind} seed {seed} {k} {s}: mean {a.mean()} != logged {logged}")
                checks.append(err)
                arrs[f"{k}__{s}"] = a
        if kind != "logreg":
            for s in SPLITS:
                if not np.array_equal(z[f"blocks__{s}"], P["blocks"][s]):
                    raise SystemExit(f"T0 {kind} seed {seed}: block ids of {s} differ from the tally split")
        out[(kind, seed)] = arrs
    log(f"T0 arrays loaded: {len(out)} runs, {len(checks)} arrays, max |mean - logged| = {max(checks):.3g}")
    return out, {"runs": len(out), "arrays": len(checks), "max_mean_err": float(max(checks))}


def mix_with_floor(bits_by_split, F):
    """F: {split: (pF, b)}. Returns per-position bits of the mixture on the scored splits, the weights."""
    p_cal = 2.0 ** (-bits_by_split["calib"])
    P = np.stack([F["calib"][0], p_cal], 1)
    lam_g, lamb, em = systems.fit_mixture(P, F["calib"][1])
    out = {}
    for s in SCORED:
        Q = np.stack([F[s][0], 2.0 ** (-bits_by_split[s])], 1)
        out[s] = systems.bits(np.log((Q * lamb[F[s][1]]).sum(1)))
    w_t0 = {s: float(lamb[F[s][1], 1].mean()) for s in SCORED}
    return out, {"lam_global": lam_g.tolist(), "mean_t0_weight": w_t0, **em}


def run(T0, F, seeds, block_ids, bootstrap, log=print):
    """bootstrap(d, blocks) -> interval dict. Returns metrics and per-arm mean bits per split."""
    F_bits = {s: systems.bits(np.log(F[s][0])) for s in SCORED}
    res = {"F": {s: float(F_bits[s].mean()) for s in SCORED}, "arms": {}, "contrasts": {}}
    mean_bits = {}
    for arm, (kind, keys) in ARMS.items():
        acc = {s: [] for s in SCORED}
        per_seed = []
        for seed in seeds:
            arrs = T0[(kind, seed)]
            for k in keys:
                mixed, info = mix_with_floor({s: arrs[f"{k}__{s}"] for s in SPLITS}, F)
                for s in SCORED:
                    acc[s].append(mixed[s])
                per_seed.append({"seed": seed, "member": k,
                                 **{f"{s}_bpw": float(mixed[s].mean()) for s in SCORED},
                                 **{f"{s}_minus_F": float(mixed[s].mean() - F_bits[s].mean()) for s in SCORED},
                                 "mean_t0_weight": info["mean_t0_weight"]})
        mean_bits[arm] = {s: np.mean(acc[s], 0) for s in SCORED}
        r = {"members_averaged": len(acc["valid"]), "per_seed": per_seed}
        for s in SCORED:
            r[s] = {"bpw": float(mean_bits[arm][s].mean()),
                    "minus_F": bootstrap(mean_bits[arm][s] - F_bits[s], block_ids[s])}
        res["arms"][arm] = r
        log(f"step0 {arm}: valid {r['valid']['bpw']:.4f} (minus F {r['valid']['minus_F']['mean']:+.4f} "
            f"[{r['valid']['minus_F']['lo']:+.4f}, {r['valid']['minus_F']['hi']:+.4f}]); devtest "
            f"{r['devtest']['bpw']:.4f} (minus F {r['devtest']['minus_F']['mean']:+.4f})")
    for a, b in CONTRASTS:
        res["contrasts"][f"{a} - {b}"] = {s: bootstrap(mean_bits[a][s] - mean_bits[b][s], block_ids[s])
                                         for s in SCORED}
    return res, mean_bits, F_bits
