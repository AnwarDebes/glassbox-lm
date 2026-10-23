"""T0: counted against learned votes on the same clauses (coalesced TM, tmu, CPU).

    python -m tally.t0.run --mode tune --seed 1 --s 10 --epochs 10
    python -m tally.t0.run --mode final --seed S --s S* --epochs E*

tune: after every epoch, arms 1 (learned) and 2 (counted) on the tuning split (valid), with
      tau and alpha fitted on calib. Stops after `patience` epochs without a better arm-1 score.
final: after the last epoch, arms 1, 2 and 3 (untrained banks, 10 draws) on calib, valid and
      devtest; per-position bits saved; clause outputs saved for arm 4 (logistic regression,
      t0/logreg.py). Per-position bits are stored so the report can pair them.
"""
import argparse
import os
import pickle
import time

import numpy as np

from tally import runlog
from tally.t0 import data, readout
from tally.t0.clauses import Bank
from tally.t0.fast_coalesced import FastCoalesced
from scipy import sparse
from tmu.models.classification.vanilla_classifier import TMClassifier

ALPHAS = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)
FIT_CHUNK = 100 * (data.K_TOP + 1)  # tmu trains in rounds of K examples; a multiple keeps them whole
N_UNTRAINED = 10


def actives(codes):
    return (codes + np.arange(data.CONTEXT, dtype=np.int32) * (data.K_TOP + 1)).astype(np.int32)


def score_split(bank, M, act):
    return readout.scores(lambda lo, hi: bank.fired(act[lo:hi]), M, len(act))


def counted_arm(bank, act, y, K, splits, run, label):
    """alpha and tau on calib; returns per-split bits and the fitted numbers."""
    N, _ = bank.counts(act["train"], y["train"], K)
    best = None
    for a in ALPHAS:
        M = readout.counted_matrix(N.astype(np.float64), a)
        S = score_split(bank, M, act["calib"])
        tau, nll = readout.fit_tau(S, y["calib"])
        if best is None or nll < best[2]:
            best = (a, tau, nll, M)
    a, tau, nll, M = best
    run.print(f"{label}: alpha={a} tau={tau:.4g} calib_nll={nll:.4f}")
    return M, a, tau


def machine_bank(tm, machine):
    """The machine's clauses as one Bank and its learned votes as a (clauses, K) matrix."""
    F = data.CONTEXT * (data.K_TOP + 1)
    if machine == "coalesced":
        return Bank(tm.clause_bank.get_literals(), F), tm.weights().T.astype(np.float64)
    K = tm.number_of_classes
    lits = np.concatenate([tm.clause_banks[k].get_literals() for k in range(K)])
    c = tm.clause_banks[0].number_of_clauses
    w = np.concatenate([tm.weight_banks[k].get_weights() for k in range(K)]).astype(np.float64)
    Ml = sparse.csr_matrix((w, (np.arange(K * c), np.repeat(np.arange(K), c))), shape=(K * c, K))
    return Bank(lits, F), Ml


def evaluate(tm, machine, d, act, y, splits, run, untrained=False, seed=0):
    K = data.K_TOP + 1
    bank, Ml = machine_bank(tm, machine)
    out, fit = {}, {}
    rng = np.random.default_rng(seed)
    # arm 1: learned votes
    tau_l, nll_l = readout.fit_tau(score_split(bank, Ml, act["calib"]), y["calib"])
    fit["learned"] = dict(tau=tau_l, calib_nll=nll_l)
    # arm 2: counted read-out on the same bank
    Mc, a_c, tau_c = counted_arm(bank, act, y, K, splits, run, "counted")
    fit["counted"] = dict(alpha=a_c, tau=tau_c)
    for sp in splits:
        for arm, M, tau in (("learned", Ml, tau_l), ("counted", Mc, tau_c)):
            S = score_split(bank, M, act[sp])
            ok, dev, mn = readout.check_distribution(S, tau, d["cls"], d["u"], data.K_TOP, rng)
            assert ok, f"distribution check failed: {arm} {sp} max|sum-1|={dev} min={mn}"
            out[(arm, sp)] = readout.word_bits(S, tau, d["ids"][sp], d["cls"], d["u"], data.K_TOP)
    if untrained:
        for r in range(N_UNTRAINED):
            rb = Bank.random_like(bank, np.random.default_rng([seed, r]))
            Mr, a_r, tau_r = counted_arm(rb, act, y, K, splits, run, f"untrained{r}")
            fit[f"untrained{r}"] = dict(alpha=a_r, tau=tau_r)
            for sp in splits:
                S = score_split(rb, Mr, act[sp])
                ok, dev, mn = readout.check_distribution(S, tau_r, d["cls"], d["u"], data.K_TOP, rng)
                assert ok, f"distribution check failed: untrained{r} {sp}"
                out[(f"untrained{r}", sp)] = readout.word_bits(S, tau_r, d["ids"][sp], d["cls"], d["u"], data.K_TOP)
    stats = dict(clauses_empty=int((bank.nlit == 0).sum()), clauses_no_positive=int(len(bank.free)),
                 mean_literals=float(bank.nlit.mean()))
    return out, fit, stats, bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tune", "final", "smoke"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--machine", choices=("coalesced", "flat"), default="coalesced")
    ap.add_argument("--clauses", type=int, default=2000, help="coalesced: in all; flat: per class")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--registration", default=None)
    ap.add_argument("--smoke_rows", type=int, default=30030)
    args = ap.parse_args()

    backend = ("tmu 5605ff0 TMCoalescedClassifier via FastCoalesced" if args.machine == "coalesced"
               else "tmu 5605ff0 TMClassifier")
    cfg = dict(vars(args), T=args.clauses, platform="CPU", backend=backend,
               context=data.CONTEXT, k_top=data.K_TOP, fit_chunk=FIT_CHUNK, alphas=ALPHAS, n_untrained=N_UNTRAINED,
               ptb_split_seed=data.ptb.SEED, ptb_block=data.ptb.BLOCK)
    tag = "coal" if args.machine == "coalesced" else "flat"
    name = f"t0_{args.mode}_{tag}_c{args.clauses}_s{args.s:g}_seed{args.seed}"
    run = runlog.Run(name, cfg, data_files=[data.ptb.path("train"), data.ptb.path("valid")],
                     registration=args.registration)
    d = data.build()
    splits = ("calib", "valid") if args.mode != "final" else ("calib", "valid", "devtest")
    codes, act, y = {}, {}, {}
    for sp in ("train",) + splits:
        codes[sp], y[sp] = data.split_xy(d, sp)
        act[sp] = actives(codes[sp])
    if args.mode == "smoke":
        codes["train"], y["train"], act["train"] = codes["train"][:args.smoke_rows], y["train"][:args.smoke_rows], act["train"][:args.smoke_rows]
        for sp in splits:
            codes[sp], y[sp], act[sp] = codes[sp][:3000], y[sp][:3000], act[sp][:3000]
            d["ids"][sp] = d["ids"][sp][:3000]
    n = len(y["train"])
    dropped = (n % FIT_CHUNK) % (data.K_TOP + 1) if args.machine == "coalesced" else 0
    run.print(f"train positions {n}; dropped per epoch by tmu's rounds of K: {dropped}")

    if args.machine == "coalesced":
        tm = FastCoalesced(args.clauses, args.clauses, args.s, platform="CPU", seed=args.seed)
    else:
        tm = TMClassifier(args.clauses, args.clauses, args.s, platform="CPU", seed=args.seed)
    history, best, since = [], None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = np.random.default_rng([args.seed, ep]).permutation(n)
        for lo in range(0, n, FIT_CHUNK):
            rows = perm[lo:lo + FIT_CHUNK]
            tm.fit(data.to_bits(codes["train"], rows), y["train"][rows].astype(np.uint32))
        t_fit = time.time() - t0
        with open(run.path("model_last.pkl"), "wb") as f:
            pickle.dump(dict(epoch=ep, tm=tm), f)
        rec = dict(epoch=ep, fit_s=round(t_fit, 1))
        if args.mode in ("tune", "smoke"):
            t1 = time.time()
            out, fit, stats, _ = evaluate(tm, args.machine, d, act, y, splits, run, seed=args.seed)
            rec.update({f"{a}_{sp}_bpw": float(v.mean()) for (a, sp), v in out.items()}, fit=fit, **stats,
                       eval_s=round(time.time() - t1, 1))
            score = rec["learned_valid_bpw"]
            if best is None or score < best[1]:
                best, since = (ep, score), 0
            else:
                since += 1
        history.append(rec)
        run.print("EPOCH " + str(rec))
        runlog.json.dump(history, open(run.path("history.json"), "w"), indent=1)
        if args.mode == "tune" and since >= args.patience:
            run.print(f"stop: no better learned valid score for {args.patience} epochs")
            break

    metrics = dict(history=history)
    if args.mode == "tune":
        metrics.update(best_epoch=best[0], best_learned_valid_bpw=best[1])
    if args.mode in ("final", "smoke"):
        t1 = time.time()
        out, fit, stats, bank = evaluate(tm, args.machine, d, act, y, splits, run, untrained=args.machine == "coalesced", seed=args.seed)
        np.savez_compressed(run.path("bits.npz"), **{f"{a}__{sp}": v for (a, sp), v in out.items()},
                            **{f"blocks__{sp}": d["blocks"][sp][:len(y[sp])] for sp in splits})
        for sp in (("train",) + splits) if args.machine == "coalesced" else ():
            np.savez_compressed(run.path(f"fired_{sp}.npz"),
                                bits=np.packbits(np.concatenate([bank.fired(act[sp][lo:lo + 50000])
                                                                 for lo in range(0, len(act[sp]), 50000)]), axis=1),
                                y=y[sp], words=d["ids"][sp][:len(y[sp])], C=bank.C)
        summ = {}
        for (a, sp), v in out.items():
            summ[f"{a}_{sp}_bpw"] = float(v.mean())
        unt = [k for k in summ if k.startswith("untrained") and k.endswith("_devtest_bpw")]
        if unt:
            summ["untrained_mean_devtest_bpw"] = float(np.mean([summ[k] for k in unt]))
        metrics.update(summ, fit=fit, **stats, eval_s=round(time.time() - t1, 1))
    run.finish(metrics)


if __name__ == "__main__":
    main()
