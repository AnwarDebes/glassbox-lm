"""Train and score the OR-group Tsetlin machine (orgroup_tm.py) on the planted-group pair tests L2 and L3 of the
hierarchical_tm study (its files data/pcp/pairs_<rung>_s<seed>.npz, read only), or on a world of a new seed
built by the study's pcp_world.World (--fresh).

Usage (from the study folder):
  .venv/bin/python -I src/run_pcp.py --rung L2 --seed 1 --mode near --name e0_L2_near_s1
Writes runs/<UTC>_<name>/{config.json, metrics.json, groups.txt, state.npz} and appends one row to logs/RUNS.md
(not with --no-log). Scores: plain accuracy (vote sum >= 1), tuned accuracy, AUC, group recovery by the study's own
pcp_world.recovery, purity and coverage as in pcp_purity.py; at L3 the same accuracies on the unseen word pairs.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
STUDY = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "hierarchical_tm"))
sys.path.insert(0, SRC)
sys.dont_write_bytecode = True  # nothing is written into the study folder
sys.path.insert(1, os.path.join(STUDY, "src"))

from pcp_world import World, recovery

from metrics import auc, group_stats, groups_to_include_bits, plain_acc, tuned_acc
from orgroup_tm import OrGroupTM, pack_pairs


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rung", choices=["L2", "L3"], required=True)
    ap.add_argument("--seed", type=int, required=True, help="world seed (and model seed)")
    ap.add_argument("--fresh", action="store_true", help="generate the world of this seed with pcp_world.World "
                                                         "(10,000 train, 4,000 dev, 4,000 unseen pairs) instead of reading the study's files")
    ap.add_argument("--mode", choices=["near", "side", "all"], default="near")
    ap.add_argument("--clauses", type=int, default=40)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--T", type=float, default=4)
    ap.add_argument("--init-state", type=int, default=-1)
    ap.add_argument("--seed-state", type=int, default=0)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--p-rec", type=float, default=1.0)
    ap.add_argument("--p-rep", type=float, default=1.0)
    ap.add_argument("--p-reinf", type=float, default=1.0)
    ap.add_argument("--q-seed", type=float, default=1.0)
    ap.add_argument("--neg-frac", type=float, default=0.0)
    ap.add_argument("--reject-all", action="store_true")
    ap.add_argument("--reject", choices=["random", "all", "weakest"], default=None)
    ap.add_argument("--reinforce", choices=["all", "critical", "share"], default="all")
    ap.add_argument("--max-seed", type=int, default=0, help="at most this many clauses seeded per example (0 = no limit)")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=0, help="first n train pairs only (0 = all)")
    ap.add_argument("--n-dev", type=int, default=0, help="first n dev pairs only (0 = all)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--no-log", action="store_true", help="do not append to logs/RUNS.md (tests)")
    return ap.parse_args(argv)


def load(a):
    """Train and dev (and unseen) pairs as (ids, bounds, y), the world and the data provenance."""
    w = World(a.rung, a.seed)
    out, prov = {}, {}
    if a.fresh:
        for name, n in (("train", 10000), ("dev", 4000)) + ((("unseen", 4000),) if w.holdout else ()):
            X, y, _, _ = w.sample(n, name)
            Xp = np.vstack([X[:, :w.V], X[:, w.V:]]).astype(np.uint8)
            P = np.c_[np.arange(n), n + np.arange(n)]
            out[name] = (*pack_pairs(Xp, P), y.astype(np.int64))
        prov["source"] = f"pcp_world.World({a.rung!r}, {a.seed}).sample"
    else:
        f = os.path.join(STUDY, "data", "pcp", f"pairs_{a.rung}_s{a.seed}.npz")
        d = np.load(f)
        out["train"] = (*pack_pairs(d["Xp"], d["P_train"]), d["Y_train"].astype(np.int64))
        out["dev"] = (*pack_pairs(d["Xp"], d["P_held"]), d["Y_held"].astype(np.int64))
        prov[os.path.basename(f)] = sha(f)
        if w.holdout:
            fu = f.replace(".npz", "_unseen.npz")
            du = np.load(fu)
            out["unseen"] = (*pack_pairs(du["Xp"], du["P_held"]), du["Y_held"].astype(np.int64))
            prov[os.path.basename(fu)] = sha(fu)
    for name, n in (("train", a.n_train), ("dev", a.n_dev)):
        if n:
            ids, bounds, y = out[name]
            out[name] = (ids, bounds[:n], y[:n])
    return out, w, prov


def score_set(tm, ids, bounds, y):
    v = tm.votes(ids, bounds)
    return {"plain_acc": plain_acc(v, y), "tuned_acc": tuned_acc(v, y), "auc": auc(v, y),
            "pred_pos": float((v >= 1).mean())}


def group_report(tm, w):
    pos = [gs for gs, p in zip(tm.groups(), tm.pol) if p > 0]
    inc, A = groups_to_include_bits(pos, tm.G, w.V)
    rec = recovery(inc, w.groups, A, tm.K)
    flat = [g for gs in pos for g in gs]
    st = group_stats(flat, w.G, w.m)
    return {"recovery": rec["recovery"], "n_counted_groups": rec["n_counted_groups"],
            "purity_all": st["purity"], "coverage_all": st["coverage"], "mean_words": st["mean_words"],
            "n_groups_2plus": st["n"], "empty_group_share": float(np.mean([len(g) == 0 for g in flat])),
            "blank_clause_share": float(np.mean([all(len(g) == 0 for g in gs) for gs in pos]))}


def main(argv=None):
    a = get_args(argv)
    t0 = time.time()
    data, w, prov = load(a)
    tm = OrGroupTM(a.clauses, w.V, sides=2, K=a.K, T=a.T, mode=a.mode, init_state=a.init_state,
                   seed_state=a.seed_state, r=a.r, p_rec=a.p_rec, p_rep=a.p_rep, p_reinf=a.p_reinf,
                   q_seed=a.q_seed, neg_frac=a.neg_frac, reject_all=a.reject_all, max_seed=a.max_seed,
                   reject=a.reject, reinforce=a.reinforce, seed=a.seed)
    hist = []
    for p in range(1, a.passes + 1):
        t1 = time.time()
        tm.fit_pass(*data["train"])
        fit_s = time.time() - t1
        row = {"pass": p, "fit_s": round(fit_s, 3)}
        row.update({f"dev_{k}": v for k, v in score_set(tm, *data["dev"]).items()})
        row.update(group_report(tm, w))
        hist.append(row)
        print("PASS", json.dumps(row), flush=True)
    assert tm.nmem_consistent(), "member counts out of step with the states"
    n_tr = min(2000, len(data["train"][2]))
    ids, bounds, y = data["train"]
    final = dict(hist[-1])
    final.update({f"train_{k}": v for k, v in score_set(tm, ids, bounds[:n_tr], y[:n_tr]).items()})
    if "unseen" in data:
        final.update({f"unseen_{k}": v for k, v in score_set(tm, *data["unseen"]).items()})
    final["events"] = tm.event_counts()
    final["n_train"], final["n_dev"] = len(data["train"][2]), len(data["dev"][2])
    final["wall_s"] = round(time.time() - t0, 2)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(a.out_root, f"{stamp}_{a.name}")
    os.makedirs(run_dir, exist_ok=False)
    cfg = {**vars(a), "V": w.V, "hidden_groups": w.G, "words_per_group": w.m, "data": prov,
           "orgroup_tm_sha256": sha(os.path.join(SRC, "orgroup_tm.py")), "run_utc": stamp}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"final": final, "history": hist}, f, indent=1)
    np.savez_compressed(os.path.join(run_dir, "state.npz"), state=tm.state, pol=tm.pol)
    with open(os.path.join(run_dir, "groups.txt"), "w") as f:
        for c, gs in enumerate(tm.groups()):
            f.write(f"clause {c} ({'+' if tm.pol[c] > 0 else '-'}):\n")
            f.writelines(f"   {'AB'[g // tm.K]}{g % tm.K + 1}: [" + " | ".join(w.vocab[i] for i in words) + "]\n" for g, words in enumerate(gs))
    print("FINAL", json.dumps(final), flush=True)
    if not a.no_log:
        log = os.path.join(ROOT, "logs", "RUNS.md")
        new = not os.path.exists(log)
        with open(log, "a") as f:
            if new:
                f.write("# Runs of this study (every run, newest last; times UTC from the clock of the run)\n\n"
                        "| run folder | rung, seed | mode, clauses, K, T, init, r, passes (+ options) | dev plain / tuned / AUC | "
                        "recovery | purity / coverage | unseen plain |\n|---|---|---|---|---|---|---|\n")
            un = final.get("unseen_plain_acc")
            opts = "".join(f", {k}={v}" for k, v in (("max_seed", a.max_seed), ("neg_frac", a.neg_frac), ("reject_all", a.reject_all), ("reject", a.reject), ("reinforce", a.reinforce), ("q_seed", a.q_seed), ("p_rec", a.p_rec), ("p_rep", a.p_rep)) if not (v is False or v is None or v == "all" or (not isinstance(v, bool) and v in (0, 0.0, 1.0))))
            f.write(f"| {os.path.basename(run_dir)} | {a.rung}, {a.seed}{' fresh' if a.fresh else ''} | {a.mode}, "
                    f"{a.clauses}, {a.K}, {a.T:g}, {a.init_state}, {a.r}, {a.passes}{opts} | {final['dev_plain_acc']:.4f} / "
                    f"{final['dev_tuned_acc']:.4f} / {final['dev_auc']:.4f} | {final['recovery']:.3f} | "
                    f"{final['purity_all']} / {final['coverage_all']} | {'' if un is None else format(un, '.4f')} |\n")
    return run_dir, final


if __name__ == "__main__":
    main()
