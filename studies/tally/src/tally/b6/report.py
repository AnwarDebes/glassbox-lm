"""B6 verdict from the run folders.

    PYTHONPATH=src python -m tally.b6.report [--label b6] [--seeds 1-10] [--boot 10000]

Reads the finished runs of the label, rebuilds every arm's dev-test predictions from the saved
class sums, and writes report.json and report.md in a run folder of its own.

Primary statistic: mean over seeds of the paired dev-test accuracy gain, rule T minus the blind
composite, in accuracy points, with a 95% percentile interval from a two-level paired bootstrap
(seeds resampled, and dev-test examples resampled with one draw shared by both arms and all
seeds of a replicate). Verdict (design B6): success if on one dataset the mean gain is at least
0.5 points with the interval above zero and on the other dataset the interval is not entirely
below zero; kill if the mean gain is under 0.2 points on both; otherwise no claim.
"""
import argparse
import glob
import json
import os

import numpy as np

from tally.b6 import combine, config
from tally.runlog import RESULTS, Run

SUCCESS, KILL = 0.5, 0.2


def finished(label, arm, ds, seed, member=None):
    name = f"{label}_{arm}_{ds}_s{seed}" + (f"_{member}" if member else "")
    hits = []
    for d in sorted(glob.glob(os.path.join(RESULTS, f"*_{name}"))):
        try:
            with open(os.path.join(d, "run.json")) as f:
                if json.load(f).get("state") == "finished":
                    hits.append(d)
        except OSError:
            pass
    if len(hits) > 1:
        raise RuntimeError(f"more than one finished run {name}: {hits}")
    return hits[0] if hits else None


def scores(d):
    z = np.load(os.path.join(d, "scores.npz"))
    return {k: z[k] for k in z.files}


def labels(d):
    with open(os.path.join(d, "labels.json")) as f:
        return {k: np.asarray(v) for k, v in json.load(f).items()}


def arms_for_seed(label, ds, seed):
    """Correct/incorrect vector on dev-test per arm (None when a run is missing)."""
    specs = config.MEMBERS[ds]
    names = [s["name"] for s in specs]
    out, missing, y = {}, [], None

    def team_paper(sc, nm, split="devtest"):
        al = [combine.alpha(sc[f"{n}__calib"]) for n in nm]
        return combine.paper([sc[f"{n}__{split}"] for n in nm], al)

    # blind members and the composites built from them
    bd = {n: finished(label, "member", ds, seed, n) for n in names}
    if all(bd.values()):
        sc = {}
        for n in names:
            sc.update(scores(bd[n]))
        lab = labels(bd[names[0]])
        y = lab["devtest"]
        val = {n: combine.acc(sc[f"{n}__valid"], lab["valid"]) for n in names}
        best = max(names, key=lambda n: val[n])
        out["best_member"] = sc[f"{best}__devtest"].argmax(1) == y
        out["blind"] = team_paper(sc, names).argmax(1) == y
        out["blind_library_norm"] = combine.library([sc[f"{n}__devtest"] for n in names]).argmax(1) == y
        out["blind_confidence"] = combine.confidence([sc[f"{n}__devtest"] for n in names]).argmax(1) == y
        accs = [combine.acc(sc[f"{n}__calib"], lab["calib"]) for n in names]
        out["blind_accweighted"] = combine.accuracy_weighted([sc[f"{n}__devtest"] for n in names], accs).argmax(1) == y
        out["_best_member_name"] = best
    else:
        missing += [f"member_{n}" for n, d in bd.items() if not d]

    for arm in ("ruleT", "seqboost", "coupled"):
        d = finished(label, arm, ds, seed)
        if not d:
            missing.append(arm)
            continue
        sc, lab = scores(d), labels(d)
        y = lab["devtest"]
        out[arm] = team_paper(sc, names).argmax(1) == y
        if arm == "coupled":
            joint = sum(sc[f"{s['name']}__devtest"] / config.T_of(s) for s in specs)
            out["coupled_joint"] = joint.argmax(1) == y
    nd = {n: finished(label, "nocomm", ds, seed, n) for n in names}
    if all(nd.values()):
        sc = {}
        for n in names:
            sc.update(scores(nd[n]))
        y = labels(nd[names[0]])["devtest"]
        out["nocomm"] = team_paper(sc, names).argmax(1) == y
    else:
        missing += [f"nocomm_{n}" for n, d in nd.items() if not d]
    for arm, key in (("single", "single__devtest"), ("fusion", "head__devtest"), ("gate", "team__devtest")):
        d = finished(label, arm, ds, seed)
        if not d:
            missing.append(arm)
            continue
        y = labels(d)["devtest"]
        out[arm] = scores(d)[key].argmax(1) == y
    return out, missing


def boot_gain(A, B, n_boot, rng, chunk=500):
    """A, B: (seeds, n) correct vectors. Mean gain in points and 95% two-level bootstrap interval."""
    D = (A.astype(np.float64) - B.astype(np.float64))
    S, n = D.shape
    mean = 100 * D.mean()
    reps = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=b).T / n          # (n, b) example weights
        per_seed = D @ W                                                  # (S, b)
        pick = rng.integers(0, S, size=(b, S))
        reps.append(100 * per_seed[pick, np.arange(b)[:, None]].mean(axis=1))
    reps = np.concatenate(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    # seed-only t interval, shown beside it
    g = 100 * D.mean(axis=1)
    from scipy import stats
    half = stats.t.ppf(0.975, S - 1) * g.std(ddof=1) / np.sqrt(S) if S > 1 else float("nan")
    return dict(mean=float(mean), lo=float(lo), hi=float(hi), per_seed=g.round(3).tolist(),
                t_lo=float(g.mean() - half), t_hi=float(g.mean() + half))


def verdict(prim):
    ds = list(prim)
    for a, b in ((ds[0], ds[1]), (ds[1], ds[0])):
        if prim[a]["mean"] >= SUCCESS and prim[a]["lo"] > 0 and prim[b]["hi"] > 0:
            return "success"
    if all(prim[d]["mean"] < KILL for d in ds):
        return "kill"
    return "no claim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="b6")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    run = Run(f"{a.label}report", vars(a))
    rng = np.random.default_rng(12345)
    rep, prim, complete = {}, {}, True
    for ds in config.MEMBERS:
        per, miss = {}, {}
        for s in seeds:
            per[s], m = arms_for_seed(a.label, ds, s)
            if m:
                miss[s] = m
        if miss:
            complete = False
        arms = sorted({k for v in per.values() for k in v if not k.startswith("_")})
        full = [k for k in arms if all(k in per[s] for s in seeds)]
        acc = {k: [round(100 * float(per[s][k].mean()), 3) for s in seeds] for k in full}
        comp = {}
        if "blind" in full:
            B = np.stack([per[s]["blind"] for s in seeds])
            for k in full:
                if k != "blind":
                    comp[k] = boot_gain(np.stack([per[s][k] for s in seeds]), B, a.boot, rng)
            if len(seeds) >= 2:
                h = len(seeds) // 2      # A/A: blind composites of two disjoint seed halves, paired
                comp["AA_blind_seed_halves"] = boot_gain(B[:h], B[h:2 * h], a.boot, rng)
        dup_path = os.path.join(os.path.dirname(RESULTS), "data", "b6", ds, "devtest_dup_in_train.npy")
        if "ruleT" in comp and os.path.exists(dup_path):
            keep = ~np.load(dup_path)
            if all(len(per[s]["blind"]) == len(keep) for s in seeds):
                comp["ruleT_no_duplicates"] = boot_gain(np.stack([per[s]["ruleT"][keep] for s in seeds]),
                                                        np.stack([per[s]["blind"][keep] for s in seeds]), a.boot, rng)
        if "ruleT" in comp:
            prim[ds] = comp["ruleT"]
        rep[ds] = dict(missing=miss, accuracy=acc, gain_vs_blind=comp,
                       best_member={s: per[s].get("_best_member_name") for s in seeds})
    v = verdict(prim) if complete and len(prim) == 2 else "incomplete: no verdict"
    rep["primary"] = prim
    rep["verdict"] = v
    with open(run.path("report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    lines = [f"# B6 report ({a.label}, seeds {a.seeds})", "", f"Verdict: **{v}**", ""]
    for ds in config.MEMBERS:
        lines += [f"## {ds}", "", "| arm | mean acc (%) | gain vs blind (points) | 95% interval | t interval (seeds) |",
                  "|---|---|---|---|---|"]
        for k, accs in rep[ds]["accuracy"].items():
            g = rep[ds]["gain_vs_blind"].get(k)
            gs = f"{g['mean']:+.2f} | [{g['lo']:+.2f}, {g['hi']:+.2f}] | [{g['t_lo']:+.2f}, {g['t_hi']:+.2f}]" if g else "- | - | -"
            lines.append(f"| {k} | {np.mean(accs):.2f} | {gs} |")
        for k in ("AA_blind_seed_halves", "ruleT_no_duplicates"):
            g = rep[ds]["gain_vs_blind"].get(k)
            if g:
                lines.append(f"| {k} | - | {g['mean']:+.2f} | [{g['lo']:+.2f}, {g['hi']:+.2f}] | "
                             f"[{g['t_lo']:+.2f}, {g['t_hi']:+.2f}] |")
        if rep[ds]["missing"]:
            lines.append(f"\nMissing runs: {json.dumps(rep[ds]['missing'])}")
        lines.append("")
    with open(run.path("report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    run.finish(dict(verdict=v, primary=prim))


if __name__ == "__main__":
    main()
