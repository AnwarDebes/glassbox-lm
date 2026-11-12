"""Train and score the OR-group Tsetlin machine (orgroup_tm.py) on the Wikipedia paragraph pairs of the
hierarchical_tm study (its files data/{sa,np,ns_all}/{train,dev}.npz, read only): two paragraphs, y = 1 iff they
come from the same article (sa) or follow each other (np; ns_all with every shared word removed).

Usage (from the study folder):
  .venv/bin/python -I src/run_wiki.py --data sa --seed 1 --name e4_sa_first_s1
Scores on the dev pairs: AUC and tuned accuracy of the vote sum (as the study's driver), and the same with the
B paragraphs shuffled between pairs (the partner-shuffle control, which must stay near 0.5).
--split test is refused unless --registered names an existing frozen record.
Writes runs/<UTC>_<name>/{config.json, metrics.json, groups.txt, state.npz, scores_<split>.npz} and appends one
row to logs/RUNS_wiki.md (not with --no-log).
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

from metrics import auc, tuned_acc
from orgroup_tm import OrGroupTM, pack_csr


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", choices=["sa", "np", "ns_all"], required=True)
    ap.add_argument("--F", type=int, default=1024)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--clauses", type=int, default=2000)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--T", type=float, default=40)
    ap.add_argument("--mode", choices=["near", "side", "all"], default="side")
    ap.add_argument("--init-state", type=int, default=-1)
    ap.add_argument("--seed-state", type=int, default=0)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--p-rec", type=float, default=1.0)
    ap.add_argument("--p-rep", type=float, default=1.0)
    ap.add_argument("--q-seed", type=float, default=1.0)
    ap.add_argument("--max-seed", type=int, default=1)
    ap.add_argument("--reject", choices=["random", "all", "weakest"], default="weakest")
    ap.add_argument("--reinforce", choices=["all", "critical", "share"], default="critical")
    ap.add_argument("--p-decay", type=float, default=0.0)
    ap.add_argument("--seed-all", action="store_true")
    ap.add_argument("--tied", action="store_true", help="one group for both paragraphs")
    ap.add_argument("--wmax", type=int, default=0, help="clause weights in 0..wmax as extra readings (0 = off)")
    ap.add_argument("--neg-frac", type=float, default=0.0)
    ap.add_argument("--smin", type=int, default=-32)
    ap.add_argument("--smax", type=int, default=31)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=0, help="first n train pairs only (0 = all)")
    ap.add_argument("--n-dev", type=int, default=0, help="first n dev pairs only (0 = all)")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--registered", default="", help="frozen record that allows --split test")
    ap.add_argument("--top", type=int, default=40, help="clauses printed in groups.txt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args(argv)
    if a.split == "test" and not (a.registered and os.path.exists(a.registered)):
        ap.error("--split test needs --registered <existing frozen record>")
    if a.data == "ns_all" and a.F != 1024:
        ap.error("ns_all holds only at F = 1024 (the study's loader refuses other sizes)")
    return a


def load_split(version, split, F, n=0):
    """ids, bounds, y of one split, keeping word ids below F; and the file's sha256."""
    f = os.path.join(STUDY, "data", version, f"{split}.npz")
    d = np.load(f)
    sides = []
    for k in ("a", "b"):
        ids, off = d[f"{k}_ids"], d[f"{k}_off"]
        keep = ids < F
        csum = np.concatenate([[0], np.cumsum(keep)])
        sides.append((ids[keep].astype(np.int32), csum[off].astype(np.int64)))
    ids, bounds = pack_csr(sides)
    y = d["y"].astype(np.int64)
    if n:
        bounds, y = bounds[:n], y[:n]
    return ids, bounds, y, sha(f)


def shuffled_partner(ids, bounds, seed):
    """The same pairs with the B paragraphs permuted between pairs."""
    n = bounds.shape[0]
    perm = np.random.default_rng([seed, 7]).permutation(n)
    a = [(ids[bounds[e, 0]:bounds[e, 1]]) for e in range(n)]
    b = [(ids[bounds[e, 1]:bounds[e, 2]]) for e in perm]
    off_a = np.concatenate([[0], np.cumsum([len(x) for x in a])])
    off_b = np.concatenate([[0], np.cumsum([len(x) for x in b])])
    return pack_csr([(np.concatenate(a), off_a), (np.concatenate(b), off_b)])


def evaluate(tm, ids, bounds, y):
    """Scores of the unit vote sum ('auc', 'tuned_acc') and, with weights, of the two weighted readings."""
    sc = tm.scores(ids, bounds)
    v = sc["unit"]
    out = {"auc": auc(v, y), "tuned_acc": tuned_acc(v, y), "mean_votes_pos": float(v[y == 1].mean()),
           "mean_votes_neg": float(v[y == 0].mean()), "zero_vote_share": float((v == 0).mean())}
    for k in ("ratio", "weight", "logodds"):
        if k in sc:
            out[f"auc_{k}"] = auc(sc[k], y)
            out[f"tuned_acc_{k}"] = tuned_acc(sc[k], y)
    return out, sc


def main(argv=None):
    a = get_args(argv)
    t0 = time.time()
    tr_ids, tr_b, tr_y, h_tr = load_split(a.data, "train", a.F, a.n_train)
    ev_ids, ev_b, ev_y, h_ev = load_split(a.data, a.split, a.F, a.n_dev if a.split == "dev" else 0)
    with open(os.path.join(STUDY, "data", f"vocab_F{a.F}.json")) as f:
        vocab = json.load(f)
    tm = OrGroupTM(a.clauses, a.F, sides=2, K=a.K, T=a.T, mode=a.mode, init_state=a.init_state,
                   seed_state=a.seed_state, r=a.r, p_rec=a.p_rec, p_rep=a.p_rep, q_seed=a.q_seed, neg_frac=a.neg_frac,
                   max_seed=a.max_seed, reject=a.reject, reinforce=a.reinforce, p_decay=a.p_decay, seed_all=a.seed_all,
                   tied=a.tied, wmax=a.wmax, smin=a.smin, smax=a.smax, seed=a.seed)
    print("CONFIG", json.dumps({**vars(a), "n_train": len(tr_y), "n_eval": len(ev_y)}), flush=True)
    hist = []
    for p in range(1, a.passes + 1):
        t1 = time.time()
        tm.fit_pass(tr_ids, tr_b, tr_y)
        row = {"pass": p, "fit_s": round(time.time() - t1, 2)}
        if p % a.eval_every == 0 or p == a.passes:
            m, _ = evaluate(tm, ev_ids, ev_b, ev_y)
            row.update({f"{a.split}_{k}": v for k, v in m.items()})
            nm = tm.nmem
            row.update(members=int(nm.sum()), mean_group=round(float(nm[nm > 0].mean()) if (nm > 0).any() else 0.0, 2),
                       empty_groups=int((nm == 0).sum()), blank_clauses=int((nm.sum(1) == 0).sum()))
        hist.append(row)
        print("PASS", json.dumps(row), flush=True)
    assert tm.nmem_consistent() and tm.twins_consistent()
    final_m, sc = evaluate(tm, ev_ids, ev_b, ev_y)
    sh_ids, sh_b = shuffled_partner(ev_ids, ev_b, a.seed)
    sh_m, _ = evaluate(tm, sh_ids, sh_b, ev_y)
    n_tr = min(5000, len(tr_y))
    trm, _ = evaluate(tm, tr_ids, tr_b[:n_tr], tr_y[:n_tr])
    final = {"passes": a.passes, **{f"{a.split}_{k}": val for k, val in final_m.items()},
             "partner_shuffle_auc": sh_m["auc"], "partner_shuffle_tuned_acc": sh_m["tuned_acc"],
             "train_auc": trm["auc"], "members": int(tm.nmem.sum()), "events": tm.event_counts(),
             "best_pass_by_auc": max((h for h in hist if f"{a.split}_auc" in h), key=lambda h: h[f"{a.split}_auc"])["pass"],
             "fit_s_per_pass": round(float(np.mean([h["fit_s"] for h in hist[1:] or hist])), 2),
             "n_train": len(tr_y), "n_eval": len(ev_y), "wall_s": round(time.time() - t0, 1)}
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(a.out_root, f"{stamp}_{a.name}")
    os.makedirs(run_dir, exist_ok=False)
    cfg = {**vars(a), "data_sha256": {f"{a.data}/train.npz": h_tr, f"{a.data}/{a.split}.npz": h_ev},
           "orgroup_tm_sha256": sha(os.path.join(SRC, "orgroup_tm.py")), "run_utc": stamp}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"final": final, "history": hist}, f, indent=1)
    np.savez_compressed(os.path.join(run_dir, "state.npz"), state=tm.state, pol=tm.pol, wts=tm.wts)
    np.savez_compressed(os.path.join(run_dir, f"scores_{a.split}.npz"), votes=sc["unit"], y=ev_y,
                        **{k: v for k, v in sc.items() if k != "unit"})
    # the clauses with the largest difference of firing rates between the classes, with their groups in words
    _, fire = tm.votes(ev_ids, ev_b[:4000], return_fire=True)
    yy = ev_y[:4000]
    f1, f0 = fire[yy == 1].mean(0), fire[yy == 0].mean(0)
    order = np.argsort(-(f1 - f0), kind="stable")[:a.top]
    groups = tm.groups()
    with open(os.path.join(run_dir, "groups.txt"), "w") as f:
        f.write(f"top {len(order)} clauses by firing-rate difference on the first 4,000 {a.split} pairs\n")
        for c in order:
            f.write(f"clause {c}: fires on y=1 {f1[c]:.4f}, y=0 {f0[c]:.4f}\n")
            f.writelines(f"   {('G' if a.tied else 'AB'[g // tm.K])}{g % tm.K + 1}: [" + " | ".join(vocab[i] for i in words) + "]\n"
                         for g, words in enumerate(groups[c]) if not (a.tied and g >= tm.K))
    print("FINAL", json.dumps(final), flush=True)
    if not a.no_log:
        log = os.path.join(ROOT, "logs", "RUNS_wiki.md")
        new = not os.path.exists(log)
        with open(log, "a") as f:
            if new:
                f.write("# Wikipedia pair runs of this study (every run, newest last)\n\n"
                        "| run folder | data, F, seed | clauses, K, T, mode, reinforce, reject, r, init, passes (+ options) | split | "
                        "AUC / tuned acc (unit votes; then AUC of the weight and log-odds readings) | partner-shuffle AUC | members, s per pass |\n|---|---|---|---|---|---|---|\n")
            wtxt = f" ; ratio {final[f'{a.split}_auc_ratio']:.4f}" + (
                f" ; {final[f'{a.split}_auc_weight']:.4f} / {final[f'{a.split}_auc_logodds']:.4f}" if a.wmax else "")
            opts = "".join(f", {k}={val}" for k, val in (("p_decay", a.p_decay), ("seed_all", a.seed_all), ("neg_frac", a.neg_frac),
                                                         ("n_train", a.n_train), ("smin", a.smin), ("p_rec", a.p_rec), ("p_rep", a.p_rep), ("q_seed", a.q_seed),
                                                         ("tied", a.tied), ("wmax", a.wmax))
                           if not (val is False or (not isinstance(val, bool) and val in (0, 0.0, 1.0, -32))))
            f.write(f"| {os.path.basename(run_dir)} | {a.data}, {a.F}, {a.seed} | {a.clauses}, {a.K}, {a.T:g}, {a.mode}, {a.reinforce}, "
                    f"{a.reject}, {a.r}, {a.init_state}, {a.passes}{opts} | {a.split} | {final[f'{a.split}_auc']:.4f} / "
                    f"{final[f'{a.split}_tuned_acc']:.4f}{wtxt} | {final['partner_shuffle_auc']:.4f} | {final['members']}, "
                    f"{final['fit_s_per_pass']} |\n")
    return run_dir, final


if __name__ == "__main__":
    main()
