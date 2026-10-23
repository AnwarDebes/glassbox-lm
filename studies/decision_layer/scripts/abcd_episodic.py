#!/usr/bin/env python
"""Next action learned from ONE score per conversation (delayed reward): credit assignment.

R10 gave the agent a reward after every press. Here it sees nothing while the conversation runs;
at the end it gets one number: how many of its presses were the ones the human agent made. It
has to work out which presses earned the score. The action turns of a conversation are decided
in order, and the memory bits come from the agent's own earlier presses (``--loop closed``, as
R17) or from the logged presses (``--loop open``, as R10's stream). As in R13 and R17, the text
bits stay as logged. Inputs: R10's label-free view (ABCDFreeView: mem, words, rclust, cclust).

Credit (how the one score becomes a 0/1 signal for each press; ``--credit``):
- ``step``: the per-press reward itself (R10's signal; the reference that the score hides);
- ``mc``: each press gets 1 with probability score / presses (uniform Monte Carlo credit; when
  every press or no press was right, this is exact);
- ``exact``: learn only from conversations where the score pins every press down (all right or
  none right); skip the rest;
- ``exactneg``: as ``exact``, and in the other conversations each press gets a 0 with probability
  1 - score / presses (never a 1: a false 1 makes a wrong action the target, the costly error);
- ``sign``: every press gets 1 if the conversation's fraction right beats a running mean, else 0;
- ``loo``: the score minus the learner's own predicted reward of the conversation's other presses
  (a counterfactual baseline; clipped to [0, 1]);
- ``hybrid``: presses of conversations whose score pins them down are labels (R10's update); the
  others get the ``loo`` credit (with ``--update reg``);
- ``posterior``: the probability that each press was right given the score, from a calibrated
  per-press confidence (a logistic map of the chosen action's standardised vote, fitted to the
  scores by the Poisson-binomial likelihood), then a 0/1 draw.

Learners: ``tm`` (TMBandit, Boltzmann exploration; ``--update sym``: a 1 only raises the pressed
action and a 0 only lowers it, instead of R10's rule where a 1 also pushes the hardest other action
down; ``--update reg``: the pressed action's vote regresses toward the credit, read as a reward
in [0, 1], as in a regression TM), ``neural`` (NeuralBandit), ``linucb``.
One update per round of ``--convs`` conversations (about 32 presses on ABCD, as R10's rounds).
After every pass: greedy accuracy on dev (or test in registered runs) in closed and open loop.
Writes one JSON line per run to results/<out>/abcd_episodic.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abcd_closed_loop import closed_loop  # noqa: E402
from dl.abcd import ABCDFreeView, load_examples, memory_from_rows  # noqa: E402
from dl.bitcache import free_view_bits  # noqa: E402
from dl.bandit import LinUCBFast, NeuralBandit, TMBandit  # noqa: E402
from dl.tm import TMConfig  # noqa: E402


def pb_dist(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Poisson-binomial distribution of the number of successes, per row. q, mask: [c, K]."""
    c, K = q.shape
    d = np.zeros((c, K + 1))
    d[:, 0] = 1.0
    for t in range(K):
        qt = np.where(mask[:, t], q[:, t], 0.0)[:, None]
        shifted = np.concatenate([np.zeros((c, 1)), d[:, :-1]], 1)
        d = d * (1 - qt) + shifted * qt
    return d


class Calibrator:
    """q = sigmoid(a + b * z): chance that a press was right, z = the chosen action's standardised
    vote. Fitted to the observed scores only (no per-press label): every ``refit`` rounds, (a, b)
    maximise the Poisson-binomial likelihood of each conversation's score over a ring buffer of
    the last ``buf`` conversations (scipy L-BFGS-B, on the CPU)."""

    def __init__(self, buf: int = 2000, width: int = 64, refit: int = 10):
        self.p = np.array([0.0, 1.0])
        self.buf, self.refit, self.n, self.i, self.rounds = buf, refit, 0, 0, 0
        self.Z = np.zeros((buf, width))
        self.M = np.zeros((buf, width), dtype=bool)
        self.G = np.zeros(buf, dtype=np.int64)

    def q(self, Z, p=None):
        p = self.p if p is None else p
        return 1.0 / (1.0 + np.exp(-(p[0] + p[1] * Z)))

    def update(self, Z, M, G):
        from scipy.optimize import minimize

        c, w = Z.shape
        rows = (np.arange(c) + self.i) % self.buf
        self.Z[rows] = 0.0
        self.M[rows] = False
        self.Z[rows, :w] = Z
        self.M[rows, :w] = M
        self.G[rows] = G
        self.i = (self.i + c) % self.buf
        self.n = min(self.buf, self.n + c)
        self.rounds += 1
        if self.rounds % self.refit:
            return
        wmax = int(np.flatnonzero(self.M[: self.n].any(0)).max()) + 1
        Zb, Mb, Gb = self.Z[: self.n, :wmax], self.M[: self.n, :wmax], self.G[: self.n]

        def nll(p):
            d = pb_dist(self.q(Zb, p), Mb)
            return -np.log(np.clip(d[np.arange(len(Gb)), Gb], 1e-12, None)).mean()

        self.p = minimize(nll, self.p, method="L-BFGS-B", bounds=[(-10, 10), (-10, 10)]).x

    def posterior(self, Z, M, G):
        """P(press t right | the conversation's score), [c, K]."""
        q = self.q(Z)
        c, K = q.shape
        out = np.zeros_like(q)
        pG = np.clip(pb_dist(q, M)[np.arange(c), G], 1e-300, None)
        for t in range(K):
            m2 = M.copy()
            m2[:, t] = False
            loo = pb_dist(q, m2)
            p_rest = np.where(G >= 1, loo[np.arange(c), np.maximum(G - 1, 0)], 0.0)
            out[:, t] = np.where(M[:, t], q[:, t] * p_rest / pG, 0.0)
        return np.clip(out, 0, 1)


def mem_bits(done: torch.Tensor, last: torch.Tensor, cnt: torch.Tensor, max_count: int = 6) -> torch.Tensor:
    """The memory block of memory_from_rows (done(b), last action = b, at least k actions so far) from
    per-conversation state on the GPU; tests/test_episodic_credit.py checks that the two agree."""
    ar = torch.arange(1, max_count + 1, device=cnt.device)
    return torch.cat([done, last, (cnt.unsqueeze(1) >= ar).to(torch.uint8)], 1)


def std_vote(S: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    z = (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-6)
    return z.gather(1, a.view(-1, 1)).squeeze(1)


def cap_gpu() -> None:
    """Cap this process's share of the shared GPU (env DL_GPU_FRAC, default 0.15 of the card), so one job
    cannot starve other runs of memory."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("DL_GPU_FRAC", "0.15")))


def main() -> None:
    cap_gpu()
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", default="tm", choices=["tm", "neural", "linucb"])
    ap.add_argument("--credit", required=True, choices=["step", "mc", "exact", "exactneg", "sign", "posterior", "loo", "hybrid"])
    ap.add_argument("--loop", default="closed", choices=["closed", "open"])
    ap.add_argument("--dataset", default="abcd", choices=["abcd", "star"])
    ap.add_argument("--encoder", default="bge-large")
    ap.add_argument("--parts", default="mem,words,rclust,cclust")
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--boltzmann", type=float, default=0.5)
    ap.add_argument("--soft", action="store_true", help="mc and posterior credit as a probability in [0, 1] instead of a 0/1 draw "
                    "(for --update reg and the neural learner)")
    ap.add_argument("--update", default="target", choices=["target", "sym", "reg"],
                    help="TM: 'target' = TMBandit.learn (a 1 makes the press the target and pushes the hardest other action "
                         "down, as R10); 'sym' = a 1 only raises the pressed action, a 0 only lowers it; 'reg' = the pressed "
                         "action's vote regresses toward the credit (as a regression TM), no other action is touched")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--nn-lr", type=float, default=1e-3)
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--convs", type=int, default=9, help="conversations per round (ABCD: about 32 presses)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default="explore")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()
    if args.eval == "test" and not args.out.startswith("confirm"):
        raise SystemExit("test split only in registered runs (--out confirm/<registration>)")
    parts = tuple(p for p in args.parts.split(",") if p)
    assert parts[0] == "mem", "memory bits must be the first block"

    t0 = time.time()
    torch.manual_seed(args.seed)
    tr, blist = load_examples("train", args.dataset)
    ev, _ = load_examples(args.eval, args.dataset)
    E = np.load(ROOT / "data" / "emb" / f"{args.dataset}__{args.encoder}.npz")
    etr = {f: E[f"train__{f}"] for f in ("cust", "recent")}
    eev = {f: E[f"{args.eval}__{f}"] for f in ("cust", "recent")}
    K = len(blist)
    n_mem = 2 * K + 6
    if args.eval == "dev":
        Xc, names, _, _ = free_view_bits(args.dataset, parts)  # cached (src/dl/bitcache.py); the same view as R10
        Xtr_np, Xev_np = Xc["train"], Xc["dev"]
    else:
        view = ABCDFreeView(parts=parts, seed=0).fit(tr, etr, blist)
        Xtr_np, Xev_np, names = view.transform(tr, etr), view.transform(ev, eev), view.names
    Btr = torch.as_tensor(Xtr_np, dtype=torch.uint8, device="cuda")
    Bev = torch.as_tensor(Xev_np, dtype=torch.uint8, device="cuda")
    ytr = torch.as_tensor([r["y"] for r in tr], device="cuda")
    yev = np.array([r["y"] for r in ev])
    convs: dict[int, list[int]] = {}
    for i, r in enumerate(tr):
        convs.setdefault(r["convo_id"], []).append(i)
    conv_list = [sorted(v, key=lambda i: tr[i]["k"]) for v in convs.values()]
    n_conv = len(conv_list)

    if args.learner == "tm":
        cfg = TMConfig(n_clauses=args.clauses, T=args.clauses, s=args.s, epochs=args.passes, seed=args.seed)
        ln = TMBandit(K, Btr.shape[1], cfg, eps=0.0)
        ln.boltzmann = args.boltzmann
        votes = lambda X: ln.tm.class_sums(X)
        as_in = lambda X: X
    elif args.learner == "neural":
        ln = NeuralBandit(K, Btr.shape[1], lr=args.nn_lr, boltzmann=args.boltzmann, seed=args.seed)
        votes = lambda X: ln.net(X.float()).detach()
        as_in = lambda X: X.float()
    else:
        ln = LinUCBFast(K, Btr.shape[1], alpha=args.alpha)
        votes = lambda X: X.float() @ ln._theta().T
        as_in = lambda X: X.float()
    cal = Calibrator(width=max(len(c) for c in conv_list)) if args.credit == "posterior" else None

    def votes_as_reward(Xr, ar):
        """The learner's own predicted reward of each press: the TM's vote as (clip(S, -T, T) / T + 1) / 2 (the scale
        --update reg trains), the neural learner's sigmoid, LinUCB's clipped estimate."""
        S_ = votes(Xr).gather(1, ar.view(-1, 1)).squeeze(1)
        if args.learner == "tm":
            T_ = ln.tm.cfg.T
            return (S_.clamp(-T_, T_) / T_ + 1) / 2
        if args.learner == "neural":
            return torch.sigmoid(S_)
        return S_.clamp(0, 1)
    soft = args.soft
    rng = torch.Generator(device="cuda")
    rng.manual_seed(args.seed + 101)
    gperm = np.random.default_rng(args.seed)
    base, curve = 0.5, []
    tot_ok = tot_n = 0
    for p in range(args.passes):
        order = gperm.permutation(n_conv)
        ok_pass = n_pass = 0
        used = 0
        for st in range(0, n_conv, args.convs):
            cs = [conv_list[j] for j in order[st : st + args.convs]]
            lens = np.array([len(c) for c in cs])
            Ipad = np.full((len(cs), lens.max()), -1, dtype=np.int64)
            for j, c in enumerate(cs):
                Ipad[j, : len(c)] = c
            It = torch.as_tensor(Ipad, device="cuda")
            # the agent's own presses so far, per conversation (closed loop), as the memory bits
            done = torch.zeros((len(cs), K), dtype=torch.uint8, device="cuda")
            last = torch.zeros((len(cs), K), dtype=torch.uint8, device="cuda")
            cnt = torch.zeros(len(cs), dtype=torch.long, device="cuda")
            Xs, As, Ys, Cid, Pos, Zs = [], [], [], [], [], []
            for k in range(int(lens.max())):
                live = torch.as_tensor(np.flatnonzero(lens > k), device="cuda")
                ids = It[live, k]
                X = Btr[ids].clone()
                if args.loop == "closed":
                    X[:, :n_mem] = mem_bits(done[live], last[live], cnt[live])
                with torch.no_grad():
                    a = ln.act(as_in(X))
                    Z = std_vote(votes(X), a) if cal is not None else None
                done[live, a] = 1
                last[live] = 0
                last[live, a] = 1
                cnt[live] += 1
                Xs.append(X); As.append(a); Ys.append(ytr[ids])
                Cid.append(live); Pos.append(torch.full((len(live),), k, device="cuda"))
                if Z is not None:
                    Zs.append(Z)
            X, a, y = torch.cat(Xs), torch.cat(As), torch.cat(Ys)
            cid, pos = torch.cat(Cid), torch.cat(Pos)
            ok = (a == y)  # hidden from the learner except through the credit rule below
            nc = len(cs)
            G = torch.zeros(nc, device="cuda", dtype=torch.long).index_add_(0, cid, ok.long())
            Kc = torch.zeros(nc, device="cuda", dtype=torch.long).index_add_(0, cid, torch.ones_like(cid))
            frac = (G.float() / Kc.float())[cid]
            keep = None
            if args.credit == "step":
                r = ok.float()
            elif args.credit == "mc":
                r = frac.clone() if soft else (torch.rand(len(a), generator=rng, device="cuda") < frac).float()
            elif args.credit == "exact":
                keep = (frac == 0) | (frac == 1)
                r = frac
            elif args.credit == "exactneg":
                # exact where the score pins the presses down; elsewhere no positive signal (a false positive is the
                # costly error: it makes a wrong action the target), only a 0 with probability 1 - score / presses
                draw0 = torch.rand(len(a), generator=rng, device="cuda") < (1 - frac)
                keep = (frac == 0) | (frac == 1) | draw0
                r = (frac == 1).float()
            elif args.credit == "hybrid":
                # where the score pins the presses down (all right or none right) they are labels: R10's update on them;
                # elsewhere the loo credit with the regression update (below)
                with torch.no_grad():
                    phat = votes_as_reward(X, a)
                Sc = torch.zeros(nc, device="cuda").index_add_(0, cid, phat)
                r = (G.float()[cid] - (Sc[cid] - phat)).clamp(0, 1)
                ex = (frac == 0) | (frac == 1)
                if bool(ex.any()):
                    ln.learn(as_in(X[ex]), a[ex], frac[ex])
                    used += int(ex.sum())
                keep = ~ex
            elif args.credit == "loo":
                # leave the others out: the score minus what the learner itself predicts for the conversation's
                # other presses (a counterfactual baseline); with calibrated predictions its mean is the press's own reward
                with torch.no_grad():
                    phat = votes_as_reward(X, a)
                Sc = torch.zeros(nc, device="cuda").index_add_(0, cid, phat)
                r = (G.float()[cid] - (Sc[cid] - phat)).clamp(0, 1)
                if not soft:
                    r = (torch.rand(len(a), generator=rng, device="cuda") < r).float()
            elif args.credit == "sign":
                r = (frac > base).float()
                base = 0.99 * base + 0.01 * float(frac.mean())
            else:
                ci, pi, Gn = cid.cpu().numpy(), pos.cpu().numpy(), G.cpu().numpy()
                Zm = np.zeros((nc, int(lens.max())))
                Mm = np.zeros((nc, int(lens.max())), dtype=bool)
                Zm[ci, pi] = torch.cat(Zs).cpu().numpy()
                Mm[ci, pi] = True
                cal.update(Zm, Mm, Gn)
                post = torch.as_tensor(cal.posterior(Zm, Mm, Gn)[ci, pi], dtype=torch.float32, device="cuda")
                r = post if soft else (torch.rand(len(a), generator=rng, device="cuda") < post).float()
            if keep is not None:
                X, a, r = X[keep], a[keep], r[keep]
            if len(a):
                if args.learner == "tm" and args.update == "sym":
                    hit = r > 0.5
                    ln.tm.feedback(X, a, hit, a, ~hit)
                    ln.t += len(a)
                elif args.learner == "tm" and args.update == "reg":
                    # regression on the credit: the pressed action's vote, read as a predicted reward
                    # (clip(S, -T, T) / T + 1) / 2, moves toward r; the error sets the feedback probability
                    T_ = ln.tm.cfg.T
                    with torch.no_grad():
                        Sa = ln.tm.class_sums(X).gather(1, a.view(-1, 1)).squeeze(1)
                    err = r - (Sa.clamp(-T_, T_) / T_ + 1) / 2
                    ln.tm.feedback(X, a, err > 0, a, err < 0, p_pos=err.clamp(min=0), p_neg=(-err).clamp(min=0))
                    ln.t += len(a)
                else:
                    ln.learn(as_in(X), a, r)
                used += len(a)
            ok_pass += int(ok.sum()); n_pass += len(ok)
        tot_ok += ok_pass; tot_n += n_pass
        with torch.no_grad():
            S_open = votes(Bev).cpu().numpy()

            def score(Zn):
                return votes(torch.as_tensor(Zn, dtype=torch.uint8, device="cuda")).cpu().numpy()

            p_closed = closed_loop(score, Xev_np, ev, blist, n_mem)
        curve.append({"pass": p + 1, "online_acc_pass": ok_pass / n_pass, "presses_learned": used,
                      "eval_open": float((S_open.argmax(1) == yev).mean()), "eval_closed": float((p_closed == yev).mean()),
                      "t": round(time.time() - t0, 1)} | ({"cal": [round(float(v), 3) for v in cal.p]} if cal else {}))
        print(json.dumps(curve[-1]), flush=True)
    rec = {"tag": args.tag, "dataset": args.dataset, "learner": args.learner, "credit": args.credit, "loop": args.loop, "parts": parts,
           "update": args.update, "soft": args.soft,
           "clauses": args.clauses, "s": args.s, "boltzmann": args.boltzmann, "convs": args.convs, "passes": args.passes, "seed": args.seed,
           "eval": args.eval, "n_bits": int(Btr.shape[1]), "final_eval_closed": curve[-1]["eval_closed"], "final_eval_open": curve[-1]["eval_open"],
           "cum_online_acc": tot_ok / tot_n, "curve": curve, "seconds": round(time.time() - t0, 1),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.learner == "tm":
        rec["stats"] = ln.tm.stats()
        if args.save_model:
            st = ln.tm.state()
            mdir = ROOT / "results" / args.out
            mdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mdir / f"model_episodic_{args.tag}_seed{args.seed}.npz", ta=st["ta"], w=st["w"], cfg=st["cfg"],
                                names=np.array(names))
    out = ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "abcd_episodic.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    if args.eval == "test":
        np.savez(out / f"preds_episodic_{args.tag}_seed{args.seed}.npz", pred_closed=p_closed, pred_open=S_open.argmax(1), y=yev)
    print(json.dumps({k: v for k, v in rec.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
