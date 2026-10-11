"""Slide-11 feasibility probe: hierarchical TM (and flat TM) on same-article paragraph pairs.

Input layout (verified against PyHierarchicalTsetlinMachineCUDA tm.py/kernels.py at 5b4a223):
  X = [x1 (F bits) | x2 (F bits)], number_of_features_hierarchy = F * 2 (only AND_GROUP/OR_GROUP
  factors span features; OR_/AND_ALTERNATIVES reuse the same feature slice with separate TA teams).
  hier: ((AND_GROUP,F),(OR_ALTERNATIVES,A),(AND_ALTERNATIVES,K),(AND_GROUP,2))
     TA team (leaf) index comp = a + A*(k + K*p), p=0 paragraph 1, p=1 paragraph 2.
  flat: ((AND_GROUP,F),(AND_GROUP,2)) -> one conjunction over P1 words and one over P2 words, ANDed.
Leaf semantics: a leaf with no included literal outputs 1 (also at inference).
Clause polarity (vanilla TsetlinMachine): weight = +1 for even clause index, -1 for odd.
"""
import argparse, json, os, sys, time
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))


def build(d, split, n=None):
    P, Y = d[f"P_{split}"], d[f"Y_{split}"]
    if n is not None:
        P, Y = P[:n], Y[:n]
    Xp = d["Xp"]
    X = np.hstack([Xp[P[:, 0]], Xp[P[:, 1]]]).astype(np.uint32)
    return X, Y.astype(np.uint32)


def leaf_outputs(inc, Xpar):
    """inc: (L, F) 0/1 include matrix; Xpar: (N, F) 0/1. returns (N, L) leaf truth."""
    need = inc.sum(1)
    hit = Xpar.astype(np.float32) @ inc.T.astype(np.float32)
    return (np.abs(hit - need[None, :]) < 0.5)


def clause_outputs_hier(lits, X, F, A, K, npar=2):
    """lits: (C, A*K*2, F) includes. returns (N, C) bool clause truth (binary)."""
    C = lits.shape[0]
    out = np.ones((X.shape[0], C), dtype=bool)
    for p in range(npar):
        Xpar = X[:, p * F:(p + 1) * F]
        for k in range(K):
            idx = [a + A * (k + K * p) for a in range(A)]
            inc = lits[:, idx, :].reshape(C * A, F)
            lo = leaf_outputs(inc, Xpar).reshape(X.shape[0], C, A)
            out &= lo.any(2)
    return out


def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hier", "flat"], default="hier")
    ap.add_argument("--F", type=int, default=1024)
    ap.add_argument("--clauses", type=int, default=200)
    ap.add_argument("--A", type=int, default=8)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--T", type=float, default=100)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-held", type=int, default=2000)
    ap.add_argument("--n-train-eval", type=int, default=1000)
    ap.add_argument("--binary-inference", type=int, default=0)
    ap.add_argument("--and-group-normalization", type=int, default=0)
    ap.add_argument("--constant-update-p", type=int, default=0)
    ap.add_argument("--no-clipping", type=int, default=0)
    ap.add_argument("--append-negated", type=int, default=0)
    ap.add_argument("--boost", type=int, default=0)
    ap.add_argument("--multiclass", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=30)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--swap-aug", type=int, default=0)
    ap.add_argument("--input-mode", choices=["pair", "inter"], default="pair")
    ap.add_argument("--label-mode", choices=["pair", "p1word", "overlap2"], default="pair")
    a = ap.parse_args()
    t_start = time.time()
    import PyHierarchicalTsetlinMachineCUDA.tm as htm
    d = dict(np.load(f"{D}/pairs_F{a.F}.npz"))
    vocab = json.load(open(f"{D}/vocab_F{a.F}.json"))
    Xtr, Ytr = build(d, "train", a.n_train)
    Xho, Yho = build(d, "held", a.n_held)
    Xwv, Ywv = build(d, "wval")
    F = a.F
    if a.label_mode != "pair":
        def relabel(X):
            if a.label_mode == "p1word":
                return X[:, vocab.index("war")].astype(np.uint32)
            ov = (X[:, :F] & X[:, F:]).sum(1)
            return (ov >= 2).astype(np.uint32)
        Ytr, Yho, Ywv = relabel(Xtr), relabel(Xho), relabel(Xwv)
        print(f"LABEL-MODE {a.label_mode}: positive rate train {Ytr.mean():.3f} held {Yho.mean():.3f}", flush=True)
    NP = 2
    if a.input_mode == "inter":  # NOT slide-11: relation precomputed outside the TM
        Xtr, Xho, Xwv = [(X[:, :F] & X[:, F:]).astype(np.uint32) for X in (Xtr, Xho, Xwv)]
        NP = 1
    if a.model == "hier":
        hs = ((htm.AND_GROUP, F), (htm.OR_ALTERNATIVES, a.A), (htm.AND_ALTERNATIVES, a.K), (htm.AND_GROUP, NP))
    else:
        hs = ((htm.AND_GROUP, F), (htm.AND_GROUP, NP))
    kw = dict(no_clipping=bool(a.no_clipping), constant_update_p=a.constant_update_p, binary_inference=bool(a.binary_inference),
              and_group_normalization=bool(a.and_group_normalization), hierarchy_structure=hs,
              boost_true_positive_feedback=a.boost, number_of_state_bits=8, append_negated=bool(a.append_negated), seed=a.seed)
    if a.multiclass:
        tm = htm.MultiClassTsetlinMachine(a.clauses, a.T, a.s, **kw)
    else:
        tm = htm.TsetlinMachine(a.clauses, a.T, a.s, **kw)
    cfg = vars(a).copy()
    cfg["hierarchy_structure"] = [[int(x) for x in h] for h in hs]
    print("CONFIG", json.dumps(cfg), "Xtr", Xtr.shape, flush=True)
    hist = []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        perm = np.random.default_rng(a.seed * 100 + ep).permutation(len(Ytr))
        Xe = Xtr[perm]
        if a.swap_aug:  # randomly swap paragraph order per pair (symmetric task)
            sw = np.random.default_rng(a.seed * 1000 + ep).random(len(perm)) < 0.5
            Xe = Xe.copy(); Xe[sw] = np.hstack([Xe[sw, F:], Xe[sw, :F]])
        tm.fit(Xe, Ytr[perm])
        tfit = time.time() - t0
        t1 = time.time()
        str_ = np.asarray(tm.score(Xtr[:a.n_train_eval])).reshape(-1) if not a.multiclass else None
        sho = np.asarray(tm.score(Xho)).reshape(-1) if not a.multiclass else None
        if a.multiclass:
            ptr = tm.predict(Xtr[:a.n_train_eval]).reshape(-1); pho = tm.predict(Xho).reshape(-1)
        else:
            ptr = (str_ >= 0).astype(np.uint32); pho = (sho >= 0).astype(np.uint32)
        tev = time.time() - t1
        r = dict(epoch=ep, train_acc=float((ptr == Ytr[:a.n_train_eval]).mean()), held_acc=float((pho == Yho).mean()),
                 held_pred_pos=float(pho.mean()), fit_s=round(tfit, 1), ms_per_pair=round(1000 * tfit / len(Ytr), 3),
                 eval_s=round(tev, 1))
        if not a.multiclass:
            ytr = Ytr[:a.n_train_eval]
            cands = np.unique(str_)
            accs = [((str_ >= c) == ytr).mean() for c in cands]
            thr = float(cands[int(np.argmax(accs))])
            r["held_acc_tuned_thr"] = float(((sho >= thr) == Yho).mean())
            r["held_auc"] = auc(sho, Yho)
            r["held_score_distinct"] = int(len(np.unique(sho)))
        hist.append(r)
        print("EPOCH", json.dumps(r), flush=True)
        if (time.time() - t_start) / 60 > a.max_minutes:
            print("STOP: max-minutes reached", flush=True)
            break
    pwv = tm.predict(Xwv).reshape(-1)
    final = dict(wval_acc=float((pwv == Ywv).mean()), wall_s=round(time.time() - t_start, 1))
    print("FINAL", json.dumps(final), flush=True)

    # ---- interpretation ----
    if not a.multiclass:
        lits = tm.get_literals()  # (C, teams, literals_per_leaf)
        lits = lits[:, :, :F] if not a.append_negated else lits[:, :, :F]  # positive literals only
        nteams = lits.shape[1]
        nlit = lits.sum(2)
        stats = dict(teams_per_clause=int(nteams), frac_empty_leaves=float((nlit == 0).mean()),
                     mean_literals_nonempty_leaf=float(nlit[nlit > 0].mean()) if (nlit > 0).any() else 0.0,
                     median_literals_nonempty_leaf=float(np.median(nlit[nlit > 0])) if (nlit > 0).any() else 0.0)
        print("LEAFSTATS", json.dumps(stats), flush=True)
        final.update(stats)
        sign = np.where(np.arange(a.clauses) % 2 == 0, 1, -1)
        nev = min(2000, len(Yho))
        if a.model == "hier":
            co = clause_outputs_hier(lits, Xho[:nev], F, a.A, a.K, NP)
        elif NP == 1:
            co = leaf_outputs(lits[:, 0, :], Xho[:nev, :F])
        else:
            co = leaf_outputs(lits[:, 0, :], Xho[:nev, :F]) & leaf_outputs(lits[:, 1, :], Xho[:nev, F:])
        y = Yho[:nev].astype(bool)
        f1 = co[y].mean(0); f0 = co[~y].mean(0)
        # sanity: reconstructed binary vote vs package prediction
        vote = (co * sign[None, :]).sum(1)
        agree = float(((vote >= 0) == (tm.predict(Xho[:nev]).reshape(-1) == 1)).mean())
        print(f"RECON agreement of numpy binary-vote with package predict on {nev} held pairs: {agree:.3f}"
              f" (exact only when binary_inference=1); numpy-binary-vote acc {float(((vote >= 0) == y).mean()):.4f}", flush=True)
        final["recon_agreement"] = agree
        final["numpy_binary_vote_acc"] = float(((vote >= 0) == y).mean())
        final["frac_clauses_always_true_on_held"] = float((co.mean(0) == 1.0).mean())
        final["frac_clauses_never_true_on_held"] = float((co.mean(0) == 0.0).mean())
        score = sign * (f1 - f0)
        order = np.argsort(-score)
        lines = []
        def words(ix):
            return [vocab[j] for j in np.nonzero(ix)[0]]
        for c in order[:a.show]:
            pol = "+" if sign[c] > 0 else "-"
            head = f"clause {c} ({pol}) fires on held Y=1 {f1[c]:.3f} / Y=0 {f0[c]:.3f}"
            lines.append(head)
            if a.model == "hier":
                for p in range(NP):
                    for k in range(a.K):
                        alts = []
                        for al in range(a.A):
                            w = words(lits[c, al + a.A * (k + a.K * p)])
                            alts.append("*" if not w else " & ".join(w[:6]) + (" &..." if len(w) > 6 else ""))
                        lines.append(f"   P{p + 1} concept {k}: [" + " | ".join(alts) + "]")
            else:
                for p in range(NP):
                    w = words(lits[c, p])
                    lines.append(f"   P{p + 1}: " + (" & ".join(w[:15]) if w else "*"))
        # also negative-polarity clauses most predictive of Y=0
        lines.append("--- top negative-polarity clauses ---")
        for c in order[::-1][:0]:
            pass
        neg_order = [c for c in np.argsort(-(f0 - f1)) if sign[c] < 0][: max(3, a.show // 3)]
        for c in neg_order:
            lines.append(f"clause {c} (-) fires on held Y=1 {f1[c]:.3f} / Y=0 {f0[c]:.3f}")
            if a.model == "hier":
                for p in range(NP):
                    for k in range(a.K):
                        alts = []
                        for al in range(a.A):
                            w = words(lits[c, al + a.A * (k + a.K * p)])
                            alts.append("*" if not w else " & ".join(w[:6]) + (" &..." if len(w) > 6 else ""))
                        lines.append(f"   P{p + 1} concept {k}: [" + " | ".join(alts) + "]")
            else:
                for p in range(NP):
                    w = words(lits[c, p])
                    lines.append(f"   P{p + 1}: " + (" & ".join(w[:15]) if w else "*"))
        txt = "\n".join(lines)
        print("CLAUSES\n" + txt, flush=True)
        tag = a.tag or f"{a.model}_F{a.F}_C{a.clauses}_A{a.A}_K{a.K}_T{int(a.T)}_s{a.s}_b{a.binary_inference}n{a.and_group_normalization}c{a.constant_update_p}x{a.no_clipping}"
        np.savez_compressed(f"{D}/lits_{tag}.npz", lits=lits.astype(np.uint8))
        open(f"{D}/clauses_{tag}.txt", "w").write(txt + "\n")
    out = dict(config=cfg, history=hist, final=final)
    tag = a.tag or f"{a.model}_F{a.F}_C{a.clauses}_A{a.A}_K{a.K}_T{int(a.T)}_s{a.s}_b{a.binary_inference}n{a.and_group_normalization}c{a.constant_update_p}x{a.no_clipping}"
    json.dump(out, open(f"{D}/result_{tag}.json", "w"), indent=1)
    rd = os.environ.get("TMLM_RUN_DIR")
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
