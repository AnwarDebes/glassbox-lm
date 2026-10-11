"""M1b: the pair TM. Granmo's hierarchical TM (the slide-11 hierarchy of the unpublished project proposal) on PTB adjacent word pairs.

Input X = [one-hot u | one-hot w] (2K bits). Hierarchy:
  ((AND_GROUP, K), (OR_ALTERNATIVES, A), (AND_ALTERNATIVES, 1), (AND_GROUP, 2))
so each clause is "u in group G1 AND w in group G2": a readable class-bigram rule.
Model: MultiClassTsetlinMachine (PyHierarchicalTsetlinMachineCUDA 5b4a223, .venv-htm), positive
literals only, 8 state bits, boost 0. Score = class_sum[1] - class_sum[0].
Metrics on validation pairs: AUC on all pairs, on SEEN pairs, and on UNSEEN pairs (unseen
positives vs unseen negatives; a count model is at 0.5 there by construction).
Saves: model.pkl (package save()), groups.json (each positive clause's G1/G2 words with
validation fire rates), include_bits.npz.
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
    ap.add_argument("--n-eval", type=int, default=40000, help="validation pairs scored each eval (fixed subset; all unseen kept)")
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


def include_bits(t):
    import pycuda.driver as cuda
    C, L, ch, b = t.number_of_clauses, t.number_of_ta_teams, t.number_of_literal_chunks_per_leaf, t.number_of_state_bits
    raw = np.empty(C * L * ch * b, np.uint32)
    cuda.memcpy_dtoh(raw, t.ta_state_hierarchy_gpu)
    msb = np.ascontiguousarray(raw.reshape(C, L, ch, b)[..., b - 1])
    bits = np.unpackbits(msb.view(np.uint8), axis=-1, bitorder="little")
    return bits[..., :t.number_of_literals_per_leaf].astype(bool)


def main():
    a = args()
    d = np.load(os.path.join(ROOT, "data", f"ptb_pairs_K{a.K}.npz"))
    vocab = json.load(open(os.path.join(ROOT, "data", f"vocab_K{a.K}.json")))
    K = a.K
    run = Run(f"{a.name}_s{int(a.s)}_C{a.clauses}_A{a.A}_seed{a.seed}", vars(a),
              data_files=[os.path.join(ROOT, "data", f"ptb_pairs_K{K}.npz")])
    rng = np.random.default_rng(a.seed)
    Xu, Xw, Y = d["Xu"].astype(np.int64), d["Xw"].astype(np.int64), d["Y"].astype(np.uint32)
    if a.n_train and a.n_train < len(Y):
        keep = np.sort(np.random.default_rng(0).choice(len(Y), a.n_train, replace=False))
        Xu, Xw, Y = Xu[keep], Xw[keep], Y[keep]
    Vu, Vw, VY, Vseen = d["Vu"].astype(np.int64), d["Vw"].astype(np.int64), d["VY"], d["Vseen"]
    unseen = np.flatnonzero(~Vseen)
    seen = np.flatnonzero(Vseen)
    pick = np.sort(np.r_[unseen, np.random.default_rng(1).choice(seen, max(0, a.n_eval - len(unseen)), replace=False)])
    Vu, Vw, VY, Vseen = Vu[pick], Vw[pick], VY[pick], Vseen[pick]

    import PyHierarchicalTsetlinMachineCUDA.tm as htm
    hs = ((htm.AND_GROUP, K), (htm.OR_ALTERNATIVES, a.A), (htm.AND_ALTERNATIVES, 1), (htm.AND_GROUP, 2))
    tm = htm.MultiClassTsetlinMachine(a.clauses, a.T, a.s, hierarchy_structure=hs, append_negated=False,
                                      number_of_state_bits=8, boost_true_positive_feedback=0, seed=a.seed)
    run.print("CONFIG", json.dumps({**vars(a), "n_train": int(len(Y)), "n_eval": int(len(VY)),
                                    "eval_unseen_pos": int(((~Vseen) & (VY == 1)).sum()),
                                    "eval_unseen_neg": int(((~Vseen) & (VY == 0)).sum())}))

    def score(u, w):
        out = []
        for i in range(0, len(u), 5000):
            cs = tm.score(dense(u[i:i + 5000], w[i:i + 5000], K))
            out.append(cs[1] - cs[0])
        return np.concatenate(out)

    hist, t0, n_batches = [], time.time(), -(-len(Y) // 5000)
    for p in range(1, a.passes + 1):
        np.random.seed(a.seed + p)
        t1 = time.time()
        for b in np.array_split(np.random.permutation(len(Y)), n_batches):
            tm.fit(dense(Xu[b], Xw[b], K), Y[b])
        fit_s = time.time() - t1
        last = p == a.passes or (a.max_minutes and time.time() - t0 > 60 * a.max_minutes)
        if p % a.eval_every == 0 or last or p == 1:
            sc = score(Vu, Vw)
            uu = ~Vseen
            r = dict(pass_=p, auc_all=auc(sc, VY), auc_seen=auc(sc[Vseen], VY[Vseen]), auc_unseen=auc(sc[uu], VY[uu]),
                     fit_s=round(fit_s, 1), ms_per_pair=round(1000 * fit_s / len(Y), 4), wall_s=round(time.time() - t0, 1))
            hist.append(r)
            run.print("EPOCH", json.dumps(r))
        if last:
            break

    # groups of the positive (even, weight +1) clauses of the class-1 machine
    inc = include_bits(tm.tms[1])                     # [C, leaves=2A, K]
    A = a.A
    sc_fire_pos, sc_fire_neg = {}, {}
    groups = []
    for c in range(0, a.clauses, 2):
        g1 = sorted(set(np.flatnonzero(inc[c, :A].any(0)).tolist()))
        g2 = sorted(set(np.flatnonzero(inc[c, A:].any(0)).tolist()))
        empty1 = int((~inc[c, :A].any(1)).sum()); empty2 = int((~inc[c, A:].any(1)).sum())
        groups.append(dict(clause=c, G1=[vocab[i] for i in g1], G2=[vocab[i] for i in g2],
                           G1_ids=g1, G2_ids=g2, empty_options_G1=empty1, empty_options_G2=empty2))
    json.dump(groups, open(os.path.join(run.dir, "groups.json"), "w"), indent=0)
    np.savez_compressed(os.path.join(run.dir, "include_bits.npz"), class1=np.packbits(inc, axis=-1), K=K, A=A)
    with open(os.path.join(run.dir, "model.pkl"), "wb") as f:
        pickle.dump(tm.save(), f)
    for g in groups[:12]:
        run.print(f"RULE clause {g['clause']}: u in [{' | '.join(g['G1'][:10])}{' ...' if len(g['G1']) > 10 else ''}]"
                  f" (empty options {g['empty_options_G1']})  AND  w in [{' | '.join(g['G2'][:10])}{' ...' if len(g['G2']) > 10 else ''}]"
                  f" (empty options {g['empty_options_G2']})")
    run.finish({**hist[-1], "history": hist})


if __name__ == "__main__":
    main()
