"""M1b: the pair TM. Granmo's hierarchical TM (the slide-11 hierarchy of the unpublished project proposal) on PTB adjacent word pairs.

Input X = [one-hot u | one-hot w] (2K bits). Hierarchy:
  ((AND_GROUP, K), (OR_ALTERNATIVES, A), (AND_ALTERNATIVES, 1), (AND_GROUP, 2))
so each clause is "u in group G1 AND w in group G2": a readable class-bigram rule.
Model: MultiClassTsetlinMachine (PyHierarchicalTsetlinMachineCUDA 5b4a223; src/launch_gpu.sh runs it with
../decision_layer/.venv-htm/bin/python), positive literals only, 8 state bits, boost 0. Score = class_sum[1] - class_sum[0].
Metrics on validation pairs: AUC on all pairs, on SEEN pairs, and on UNSEEN pairs (unseen
positives vs unseen negatives; a count model is at 0.5 there by construction), each on the whole validation set and
on two fixed halves, tune and check (eval_halves). By default every validation pair is scored, the same 93,766 pairs
that src/pair_baselines.py scores (changed 2026-10-25; before, a subset of 40,000 was scored, so only the unseen AUC
was comparable with the baselines). The run reports two passes separately: best_on_tune, the evaluated pass with the
highest tune-half unseen AUC, with its check-half values, and last_pass. Which split carries a registered claim
(PTB test pairs through a final_test.py, or a dev-test carved from train) is a design decision of the M1
registration; this script does not make it.
Saves: model.pkl (package save() at the best-on-tune pass), model_last.pkl (at the last pass), groups.json (each
positive clause's G1/G2 words at the best-on-tune pass; clause_groups() says which options count) and
include_bits.npz (the same pass).
"""
import argparse, json, os, pickle, sys, time
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import Run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--clauses", type=int, default=400)
    ap.add_argument("--A", type=int, default=32)
    ap.add_argument("--T", type=float, default=400)
    ap.add_argument("--s", type=float, default=2000)
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=200000, help="training pairs per pass (fixed subset; 0 = all)")
    ap.add_argument("--n-eval", type=int, default=0, help="validation pairs scored each eval: 0 = all (default; the "
                    "pairs of pair_baselines.py); a smaller number keeps every unseen pair plus a fixed draw of seen "
                    "pairs (smoke tests only)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-minutes", type=float, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default="pair_tm")
    return ap.parse_args()


def auc(score, y):
    r, n1 = rankdata(score), int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def dense(u, w, K):
    X = np.zeros((len(u), 2 * K), np.uint32)
    X[np.arange(len(u)), u] = 1
    X[np.arange(len(u)), K + w] = 1
    return X


N_STATE_BITS = 8


def eval_halves(n_pairs):
    """Tune mask over the validation pairs of pairs_data.py, which stores the n positives first and the n negatives
    after them, negative n + i with the first word of positive i. Both share the matched-pair index i = index mod n:
    even i -> tune half (True), odd i -> check half (False), so a positive and its negative stay together."""
    n = n_pairs // 2
    assert 2 * n == n_pairs, "pairs_data.py writes one negative per positive"
    return (np.arange(n_pairs) % n) % 2 == 0


def eval_pick(Vseen, n_eval):
    """Indices of the validation pairs to score: all of them when n_eval is 0 or covers them all; otherwise every
    unseen pair plus a fixed random draw of seen pairs (the selection used before 2026-10-25; smoke tests only)."""
    n = len(Vseen)
    if n_eval <= 0 or n_eval >= n:
        return np.arange(n)
    unseen, seen = np.flatnonzero(~Vseen), np.flatnonzero(Vseen)
    extra = np.random.default_rng(1).choice(seen, min(len(seen), max(0, n_eval - len(unseen))), replace=False)
    return np.sort(np.r_[unseen, extra])


def auc_block(sc, y, seen):
    uu = ~seen
    return dict(auc_all=auc(sc, y), auc_seen=auc(sc[seen], y[seen]), auc_unseen=auc(sc[uu], y[uu]))


def eval_metrics(sc, y, seen, tune):
    """AUCs on all scored pairs and on the tune and check halves."""
    return {"all": auc_block(sc, y, seen), "tune": auc_block(sc[tune], y[tune], seen[tune]),
            "check": auc_block(sc[~tune], y[~tune], seen[~tune])}


def pick_best_on_tune(hist):
    """The evaluated pass with the highest tune-half unseen AUC; the earliest pass on a tie; None if none is a number."""
    best = None
    for r in hist:
        v = r["tune"]["auc_unseen"]
        if v == v and (best is None or v > best["tune"]["auc_unseen"]):
            best = r
    return best


def include_bits_from_state(flat, n_clauses, n_teams, n_chunks, n_state_bits, n_literals):
    """Include bits [clause, TA team, literal] from the flat uint32 automaton state of one class machine of
    PyHierarchicalTsetlinMachineCUDA 5b4a223 (the 'ta_state_hierarchy' of TsetlinMachine.save()). Layout as in
    tm.py:232-236: [clause, TA team, literal chunk, state bit]; literal k of a team is bit k % 32 of chunk k // 32,
    and it is included when its most significant state bit is set (tm.py:236, ta_action)."""
    raw = np.asarray(flat, dtype=np.uint32)
    assert raw.size == n_clauses * n_teams * n_chunks * n_state_bits, "state size does not match the hierarchy"
    msb = np.ascontiguousarray(raw.reshape(n_clauses, n_teams, n_chunks, n_state_bits)[..., n_state_bits - 1])
    bits = np.unpackbits(msb.astype("<u4").view(np.uint8), axis=-1, bitorder="little")
    return bits[..., :n_literals].astype(bool)


def clause_groups(inc, A, vocab, n_clauses):
    """The rule 'u in G1 AND w in G2' of each positive (even, weight +1) clause of the class-1 machine.

    inc is [clause, 2A TA teams, K]: teams 0..A-1 are the OR-options over u and teams A..2A-1 those over w (team
    index g*A + a, kernels.py:186-210). An option is the AND of its included literals (kernels.py:213-227), so over a
    one-hot half an option with one included word fires exactly for that word, an option with two or more words never
    fires (dead), and an empty option always fires (that side then accepts every word). G1 and G2 hold the words of
    the single-word options; empty and dead options are counted. Changed 2026-10-25: before, the words of every option
    were listed, those of dead options included."""
    groups = []
    for c in range(0, n_clauses, 2):
        g = dict(clause=c)
        for side, teams in (("1", slice(0, A)), ("2", slice(A, 2 * A))):
            opts = inc[c, teams]
            n_inc = opts.sum(1)
            live = sorted(set(np.flatnonzero(opts[n_inc == 1].any(0)).tolist()))
            g[f"G{side}"] = [vocab[i] for i in live]
            g[f"G{side}_ids"] = live
            g[f"empty_options_G{side}"] = int((n_inc == 0).sum())
            g[f"dead_options_G{side}"] = int((n_inc >= 2).sum())
        groups.append(g)
    return groups


def main():
    a = args()
    d = np.load(os.path.join(ROOT, "data", f"ptb_pairs_K{a.K}.npz"))
    vocab = json.load(open(os.path.join(ROOT, "data", f"vocab_K{a.K}.json")))
    K, A = a.K, a.A
    run = Run(f"{a.name}_s{int(a.s)}_C{a.clauses}_A{a.A}_seed{a.seed}", vars(a),
              data_files=[os.path.join(ROOT, "data", f"ptb_pairs_K{K}.npz")])
    Xu, Xw, Y = d["Xu"].astype(np.int64), d["Xw"].astype(np.int64), d["Y"].astype(np.uint32)
    if a.n_train and a.n_train < len(Y):
        keep = np.sort(np.random.default_rng(0).choice(len(Y), a.n_train, replace=False))
        Xu, Xw, Y = Xu[keep], Xw[keep], Y[keep]
    Vu, Vw, VY, Vseen = d["Vu"].astype(np.int64), d["Vw"].astype(np.int64), d["VY"], d["Vseen"]
    n_valid = len(VY)
    tune = eval_halves(n_valid)
    pick = eval_pick(Vseen, a.n_eval)
    Vu, Vw, VY, Vseen, tune = Vu[pick], Vw[pick], VY[pick], Vseen[pick], tune[pick]

    import PyHierarchicalTsetlinMachineCUDA.tm as htm
    hs = ((htm.AND_GROUP, K), (htm.OR_ALTERNATIVES, A), (htm.AND_ALTERNATIVES, 1), (htm.AND_GROUP, 2))
    tm = htm.MultiClassTsetlinMachine(a.clauses, a.T, a.s, hierarchy_structure=hs, append_negated=False,
                                      number_of_state_bits=N_STATE_BITS, boost_true_positive_feedback=0, seed=a.seed)
    run.print("CONFIG", json.dumps({**vars(a), "n_train": int(len(Y)), "n_eval": int(len(VY)),
                                    "eval_all_pairs": bool(len(VY) == n_valid), "n_eval_tune": int(tune.sum()),
                                    "eval_unseen_pos": int(((~Vseen) & (VY == 1)).sum()),
                                    "eval_unseen_neg": int(((~Vseen) & (VY == 0)).sum())}))

    def score(u, w):
        out = []
        for i in range(0, len(u), 5000):
            cs = tm.score(dense(u[i:i + 5000], w[i:i + 5000], K))
            out.append(cs[1] - cs[0])
        return np.concatenate(out)

    hist, t0, n_batches = [], time.time(), -(-len(Y) // 5000)
    best_state = None
    for p in range(1, a.passes + 1):
        np.random.seed(a.seed + p)
        t1 = time.time()
        for b in np.array_split(np.random.permutation(len(Y)), n_batches):
            tm.fit(dense(Xu[b], Xw[b], K), Y[b])
        fit_s = time.time() - t1
        last = p == a.passes or (a.max_minutes and time.time() - t0 > 60 * a.max_minutes)
        if p % a.eval_every == 0 or last or p == 1:
            sc = score(Vu, Vw)
            r = dict(pass_=p, **eval_metrics(sc, VY, Vseen, tune), fit_s=round(fit_s, 1),
                     ms_per_pair=round(1000 * fit_s / len(Y), 4), wall_s=round(time.time() - t0, 1))
            hist.append(r)
            run.print("EPOCH", json.dumps(r))
            if pick_best_on_tune(hist) is r:
                best_state = tm.save()                # host copy of the state at the best tune pass so far
        if last:
            break

    last_state = tm.save()
    best = pick_best_on_tune(hist)
    if best is None:                                  # no tune-half unseen AUC is a number: keep the last pass
        best, best_state = hist[-1], last_state
    inc = include_bits_from_state(best_state["tms"][1]["ta_state_hierarchy"], a.clauses, 2 * A, (K - 1) // 32 + 1,
                                  N_STATE_BITS, K)    # [C, 2A teams, K]
    groups = clause_groups(inc, A, vocab, a.clauses)
    json.dump(groups, open(os.path.join(run.dir, "groups.json"), "w"), indent=0)
    np.savez_compressed(os.path.join(run.dir, "include_bits.npz"), class1=np.packbits(inc, axis=-1), K=K, A=A,
                        pass_=best["pass_"])
    with open(os.path.join(run.dir, "model.pkl"), "wb") as f:
        pickle.dump(best_state, f)
    with open(os.path.join(run.dir, "model_last.pkl"), "wb") as f:
        pickle.dump(last_state, f)
    for g in groups[:12]:
        run.print(f"RULE clause {g['clause']}: u in [{' | '.join(g['G1'][:10])}{' ...' if len(g['G1']) > 10 else ''}]"
                  f" (empty options {g['empty_options_G1']}, dead {g['dead_options_G1']})  AND  w in "
                  f"[{' | '.join(g['G2'][:10])}{' ...' if len(g['G2']) > 10 else ''}]"
                  f" (empty options {g['empty_options_G2']}, dead {g['dead_options_G2']})")
    run.finish({"eval_pairs": int(len(VY)), "eval_all_pairs": bool(len(VY) == n_valid),
                "picked_on": "tune half, auc_unseen", "best_on_tune": best, "last_pass": hist[-1], "history": hist})


if __name__ == "__main__":
    main()
