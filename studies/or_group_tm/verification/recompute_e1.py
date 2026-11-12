"""Independent recomputation of E1 (registrations/E1_planted_confirmation.md).

For each of the ten E1 runs (rungs L2 and L3, world seeds 11 to 15) this script
  1. regenerates the world and its pair sets with the study's pcp_world.World (the same call the runner makes);
  2. retrains the model with a second implementation of the rule, written here in plain Python from the text of
     the plan (mode near, one seed event per example, reinforce every present member, reject one random group),
     without numba and without importing src/orgroup_tm.py, and compares the final states with the stored
     state.npz, element by element;
  3. recomputes, from the STORED states and with its own code, the vote sums, the plain accuracy on the dev pairs
     and on the unseen pairs (L3), and the group recovery (share of hidden groups matched with Jaccard >= 0.8 by a
     group of at least 2 words), and compares them with metrics.json;
  4. applies the rules R1, R2 and R3 of the plan.
The random draws of the gate use numpy's legacy generator seeded as the engine seeds its own; if the two streams
differed, step 2 would report a mismatch and steps 3 and 4 would still stand on the stored states.

Usage (from the study folder):  .venv/bin/python -I verification/recompute_e1.py > verification/recompute_e1.out
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STUDY = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "hierarchical_tm"))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(STUDY, "src"))
from pcp_world import World

C, T, INIT, SEED_STATE, R, SMAX, SMIN, PASSES = 40, 16.0, -1, 0, 2, 31, -32, 5


def pairs(w, n, name):
    X, y, _, _ = w.sample(n, name)
    V = w.V
    a = [np.flatnonzero(X[i, :V]) for i in range(n)]
    b = [np.flatnonzero(X[i, V:]) for i in range(n)]
    return a, b, y.astype(int)


def retrain(w, a, b, y, seed):
    """The rule of E1 in plain Python. state[c][side] is a dict-like array of automaton states."""
    V = w.V
    st = np.full((C, 2, V), INIT, np.int16)
    for p in range(PASSES):
        order = np.random.default_rng([seed, p, 7]).permutation(len(y))
        np.random.seed((seed * 1000003 + p * 7919 + 13) % (2 ** 31 - 1))
        for e in order:
            words = (a[e], b[e])
            true = np.array([[bool((st[c, s, words[s]] >= 0).any()) for s in (0, 1)] for c in range(C)])
            fires = true.all(1)
            v = min(max(int(fires.sum()), -T), T)
            p_up = (T - v) / (2 * T) if y[e] == 1 else (T + v) / (2 * T)
            seeded = False
            for c in range(C):
                if np.random.random() >= p_up:
                    continue
                if y[e] == 1:  # the clause should fire (all clauses have positive polarity)
                    if fires[c]:
                        for s in (0, 1):
                            for x in words[s]:
                                if 0 <= st[c, s, x] < SMAX:
                                    st[c, s, x] += 1
                        continue
                    empty = [(st[c, s] >= 0).sum() == 0 for s in (0, 1)]
                    if all(empty[s] for s in (0, 1) if not true[c, s]):  # every false group is empty: seed
                        if seeded:
                            continue
                        seeded = True
                        for s in (0, 1):
                            if empty[s]:
                                for x in words[s]:
                                    if INIT <= st[c, s, x] < SEED_STATE:
                                        st[c, s, x] = SEED_STATE
                        continue
                    false = [s for s in (0, 1) if not true[c, s]]
                    if len(false) == 1:  # near miss: recruit
                        s = false[0]
                        for x in words[s]:
                            if st[c, s, x] < 0:
                                st[c, s, x] += 1
                else:  # the clause should not fire
                    if fires[c]:
                        s = np.random.randint(2)
                        for x in words[s]:
                            if st[c, s, x] >= 0:
                                st[c, s, x] = max(st[c, s, x] - R, SMIN)
                        continue
                    false = [s for s in (0, 1) if not true[c, s]]
                    if len(false) == 1:  # near miss on a negative: repress
                        s = false[0]
                        for x in words[s]:
                            if SMIN < st[c, s, x] < 0:
                                st[c, s, x] -= 1
    return st


def votes(st, a, b):
    mem = st >= 0
    return np.array([sum(bool(mem[c, 0, a[e]].any() and mem[c, 1, b[e]].any()) for c in range(st.shape[0]))
                     for e in range(len(a))])


def recovery(st, groups):
    mem = st >= 0
    found = [set(np.flatnonzero(mem[c, s]).tolist()) for c in range(st.shape[0]) for s in (0, 1)]
    found = [f for f in found if len(f) >= 2]
    hit = [any(len(set(g) & f) / len(set(g) | f) >= 0.8 for f in found) for g in np.asarray(groups).tolist()]
    return float(np.mean(hit))


def main():
    rows, ok_all = [], True
    for rung in ("L2", "L3"):
        for seed in (11, 12, 13, 14, 15):
            d = sorted(glob.glob(os.path.join(ROOT, "runs", f"*_e1_{rung}_s{seed}")))
            assert len(d) == 1, (rung, seed, d)
            with open(os.path.join(d[0], "metrics.json")) as f:
                m = json.load(f)["final"]
            with open(os.path.join(d[0], "config.json")) as f:
                cfg = json.load(f)
            assert (cfg["mode"], cfg["max_seed"], cfg["T"], cfg["clauses"], cfg["K"], cfg["passes"], cfg["fresh"]) == \
                ("near", 1, 16.0, 40, 1, 5, True), cfg
            stored = np.load(os.path.join(d[0], "state.npz"))["state"]
            w = World(rung, seed)
            tr = pairs(w, 10000, "train")
            dv = pairs(w, 4000, "dev")
            mine = retrain(w, *tr, seed)
            same = bool((mine == stored).all())
            vd = votes(stored, dv[0], dv[1])
            acc = float(((vd >= 1).astype(int) == dv[2]).mean())
            rec = recovery(stored, w.groups)
            row = {"rung": rung, "seed": seed, "retrained_state_equals_stored": same,
                   "dev_plain_acc": acc, "dev_plain_acc_reported": m["dev_plain_acc"],
                   "recovery": rec, "recovery_reported": m["recovery"]}
            match = same and abs(acc - m["dev_plain_acc"]) < 1e-12 and abs(rec - m["recovery"]) < 1e-12
            if rung == "L3":
                un = pairs(w, 4000, "unseen")
                vu = votes(stored, un[0], un[1])
                row["unseen_plain_acc"] = float(((vu >= 1).astype(int) == un[2]).mean())
                row["unseen_plain_acc_reported"] = m["unseen_plain_acc"]
                match = match and abs(row["unseen_plain_acc"] - m["unseen_plain_acc"]) < 1e-12
            row["match"] = bool(match)
            ok_all &= match
            rows.append(row)
            print(json.dumps(row), flush=True)
    r1 = sum(r["dev_plain_acc"] >= 0.90 and r["recovery"] >= 0.80 for r in rows if r["rung"] == "L2")
    r2 = sum(r["dev_plain_acc"] >= 0.90 and r["recovery"] >= 0.80 for r in rows if r["rung"] == "L3")
    r3 = sum(r["unseen_plain_acc"] >= 0.85 for r in rows if r["rung"] == "L3")
    out = {"R1_L2_seeds_meeting": r1, "R2_L3_seeds_meeting": r2, "R3_unseen_seeds_meeting": r3,
           "R1_met": r1 >= 4, "R2_met": r2 >= 4, "R3_met": r3 >= 4, "E1_met": r1 >= 4 and r2 >= 4 and r3 >= 4,
           "all_numbers_match_and_states_reproduced": bool(ok_all)}
    print("VERDICT", json.dumps(out))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
