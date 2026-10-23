#!/usr/bin/env python
"""R8: clarify-or-route over H = 4 prefixes with delayed reward (CLINC150), one run.

    r8_run.py --learner tm_q --seed 1 [--eval val|test] [--no-memory]

Learners:
  tm_q        coalesced TM, one class per action, class sum / T = Q; fitted-Q iteration toward
              r + gamma * max_a' Q_k(s', a') with regression feedback (as R6's tm_q --fqi).
  tm_myopic   the same with gamma = 0.
  lin_q       linear Q on the continuous twin of the same information, fitted-Q iteration (Adam).
  rule        thresholds on the router margin per prefix (route if >= a_t, else ask; at the last
              prefix route if >= b, else escalate; escalate at once if < c), tuned on val by
              coordinate ascent from several starts, with the out-of-scope episodes weighted to
              the same share as the TD learners train with (--oos-share).
  rule2       the same with a second threshold per step on the top class sum (both must hold).
  route, ask_route  fixed: route at once / ask until the last prefix, then route.
Router: a supervised TM (thermo:8 on MiniLM, 4,000 clauses, 40 epochs), 5-fold cross-fitted on
train, trained once (router seed 1) and cached in results/r8_router_seed1.pt; every agent seed uses
the same router. Val/test episodes use the mean of the 5 fold routers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dl.boolean import Thermo  # noqa: E402
from dl.data import load  # noqa: E402
from dl.mdp import Rewards  # noqa: E402
from dl.mdp_h import ASK, ESCALATE, ROUTE, EpisodesH, MetaCoderH, cont_features_h, evaluate_h, masked_greedy_h, prefix_text, returns_h, step_rewards_h  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402
from r6_run import TMQ, LinQ  # noqa: E402

DEV = "cuda"
H = 4
SPLITS = ("train", "val", "test", "oos_train", "oos_val", "oos_test")


def prefix_embeddings(ds) -> list[dict]:
    """MiniLM embeddings of the k/H prefixes, k = 1 .. H (k = H is the full query)."""
    out = []
    for k in range(1, H + 1):
        p = ROOT / "data" / "emb" / f"clinc150__minilm__prefix{k}of{H}.npz"
        if k == H:
            d = np.load(ROOT / "data" / "emb" / "clinc150__minilm.npz")
            out.append({s: d[s] for s in SPLITS})
            continue
        if not p.exists():
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
            e = {s: m.encode([prefix_text(t, k, H) for t in ds.texts[s]], batch_size=256, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False).astype(np.float32) for s in SPLITS}
            np.savez(p, **e)
        d = np.load(p)
        out.append({s: d[s] for s in SPLITS})
    return out


def router_scores(seed: int = 1) -> dict:
    cache = ROOT / "results" / f"r8_router_seed{seed}.pt"
    if cache.exists():
        return torch.load(cache)
    ds = load("clinc150")
    views = prefix_embeddings(ds)
    full = views[-1]
    d = np.load(ROOT / "data" / "emb" / "clinc150__minilm.npz")
    ytr, yva = d["y_train"], d["y_val"]
    therm = Thermo(8).fit(full["train"])
    enc = lambda X: torch.as_tensor(therm.transform(X), dtype=torch.uint8, device=DEV)  # noqa: E731
    Btr, Bva = therm.transform(full["train"]), therm.transform(full["val"])
    n = len(ytr)
    folds = np.random.default_rng(seed).permutation(n) % 5
    res = {"train": [torch.zeros((n, 150), device=DEV) for _ in range(H)]}
    other = [s for s in SPLITS if s != "train"]
    for s in other:
        res[s] = [0.0] * H
    for k in range(5):
        tr = folds != k
        cfg = TMConfig(n_clauses=4000, T=4000, s=8.0, epochs=40, seed=seed * 10 + k)
        tm = CoTM(150, Btr.shape[1], cfg).fit(Btr[tr], ytr[tr], Bva, yva)
        te = np.flatnonzero(~tr)
        for v in range(H):
            res["train"][v][te] = tm.class_sums(enc(views[v]["train"][te]))
            for s in other:
                res[s][v] = res[s][v] + tm.class_sums(enc(views[v][s])) / 5.0
    torch.save(res, cache)
    return res


def build(scores: dict, split: str, ds) -> EpisodesH:
    y = ds.labels[split]
    n_oos = scores[f"oos_{split}"][0].shape[0]
    S = [torch.cat([scores[split][v], scores[f"oos_{split}"][v]]) for v in range(H)]
    return EpisodesH(S, np.r_[y, -np.ones(n_oos, dtype=np.int64)], np.r_[np.zeros(len(y), bool), np.ones(n_oos, bool)])


def collect(ep: EpisodesH, R: Rewards, n_episodes: int, seed: int, oos_share: float):
    """Fitted-Q data: for sampled queries, every step reached by asking, one uniformly random allowed
    action at each step (all steps are covered: the state at step t is reached only by t asks)."""
    g = torch.Generator(device=DEV)
    g.manual_seed(seed + 99)
    in_idx = (~ep.is_oos).nonzero().squeeze(1)
    oos_idx = ep.is_oos.nonzero().squeeze(1)
    n_oos = int(round(oos_share * n_episodes))
    idx = torch.cat([in_idx[torch.randint(0, len(in_idx), (n_episodes - n_oos,), generator=g, device=DEV)],
                     oos_idx[torch.randint(0, len(oos_idx), (n_oos,), generator=g, device=DEV)]])
    I, S_, A, Rr, D = [], [], [], [], []
    for t in range(H):
        st = torch.full_like(idx, t)
        hi = 3 if t < H - 1 else 2
        a = torch.randint(0, hi, idx.shape, generator=g, device=DEV)
        if t == H - 1:
            a = a * 2  # ROUTE or ESCALATE
        r, done = step_rewards_h(ep, idx, st, a, R)
        I.append(idx); S_.append(st); A.append(a); Rr.append(r); D.append(done)
    return (torch.cat(x) for x in (I, S_, A, Rr, D))


def train_fqi(agent, feat, ep: EpisodesH, R: Rewards, gamma, iters, epochs, batch, seed, n_episodes, oos_share):
    I, S_, A, Rr, D = collect(ep, R, n_episodes, seed, oos_share)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed + 7)
    X = feat(ep, I, S_)
    nd = ~D
    Xn = feat(ep, I[nd], S_[nd] + 1)
    sn = S_[nd] + 1
    losses = []
    for _ in range(iters):
        y = Rr.clone()
        if gamma > 0 and nd.any():
            qn = agent.q(Xn)
            qn[sn == H - 1, ASK] = -1e9
            y[nd] = Rr[nd] + gamma * qn.max(1).values
        for _ in range(epochs):
            perm = torch.randperm(X.shape[0], generator=g, device=DEV)
            for st in range(0, X.shape[0], batch):
                j = perm[st : st + batch]
                losses.append(agent.update(X[j], A[j], y[j]))
    return losses


class LinQH(LinQ):
    pass


def rule_policy(params: dict, ep: EpisodesH, use_top: bool):
    def pol(i, s):
        m = ep.conf[i, s, 1]
        tp = ep.conf[i, s, 0]
        a_t = params["a"][s]  # route threshold per step (the last entry is b)
        ok = m >= a_t
        if use_top:
            ok = ok & (tp >= params["u"][s])
        esc = (m < params["c"]) & (s < H - 1)
        act = torch.where(ok, torch.full_like(i, ROUTE), torch.where(esc, torch.full_like(i, ESCALATE), torch.full_like(i, ASK)))
        return torch.where((s == H - 1) & ~ok, torch.full_like(i, ESCALATE), act)
    return pol


def tune_rule(ep: EpisodesH, R: Rewards, use_top: bool, seed: int, oos_share: float, starts: int = 6, sweeps: int = 6):
    """Coordinate ascent on val over quantile grids; returns the best parameters found. The objective
    weights the out-of-scope episodes to ``oos_share`` of the total, the share the TD learners train
    with, so that the rule and the learners trade off the same two kinds of episode."""
    qm = torch.quantile(ep.conf[:, :, 1].reshape(-1), torch.linspace(0, 1, 41, device=DEV))
    qt = torch.quantile(ep.conf[:, :, 0].reshape(-1), torch.linspace(0, 1, 41, device=DEV))
    gm = torch.cat([torch.tensor([-1e9], device=DEV), qm, torch.tensor([1e9], device=DEV)])
    gt = torch.cat([torch.tensor([-1e9], device=DEV), qt, torch.tensor([1e9], device=DEV)])
    rng = np.random.default_rng(seed)

    w = torch.where(ep.is_oos, oos_share / ep.is_oos.sum(), (1 - oos_share) / (~ep.is_oos).sum())

    def value(p):
        return float((returns_h(rule_policy(p, ep, use_top), ep, R) * w).sum())

    best = (-1e9, None)
    for st in range(starts):
        if st == 0:
            p = {"a": gm[[-1] * (H - 1) + [len(gm) // 2]].clone(), "u": gt[[0] * H].clone(), "c": gm[0].clone()}
        else:
            p = {"a": gm[rng.integers(0, len(gm), H)].clone(), "u": gt[rng.integers(0, len(gt), H)].clone(), "c": gm[rng.integers(0, len(gm) // 2)].clone()}
        v = value(p)
        for _ in range(sweeps):
            improved = False
            coords = [("a", t) for t in range(H)] + ([("u", t) for t in range(H)] if use_top else []) + [("c", None)]
            for key, t in coords:
                grid = gt if key == "u" else gm
                for val in grid:
                    q = {k: (x.clone() if torch.is_tensor(x) else x) for k, x in p.items()}
                    if t is None:
                        q[key] = val.clone()
                    else:
                        q[key][t] = val
                    vq = value(q)
                    if vq > v + 1e-12:
                        p, v, improved = q, vq, True
            if not improved:
                break
        if v > best[0]:
            best = (v, p)
    return best[1]


def strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--router-seed", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eval", default="val")
    ap.add_argument("--out", default="explore")
    ap.add_argument("--clauses", type=int, default=2000)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--s", type=float, default=4.0)
    ap.add_argument("--kappa", type=float, default=0.003)
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--oos-share", type=float, default=0.15)
    ap.add_argument("--ask-cost", type=float, default=0.1)
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--fqi-iters", type=int, default=6)
    ap.add_argument("--fqi-epochs", type=int, default=10)
    ap.add_argument("--fqi-episodes", type=int, default=60000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.eval == "test" and args.out == "explore":
        raise SystemExit("test split only in registered runs (--out confirm/R8)")
    t0 = time.time()
    out = ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)
    ds = load("clinc150")
    scores = router_scores(args.router_seed)
    ep_tr, ep_va, ep_ev = build(scores, "train", ds), build(scores, "val", ds), build(scores, args.eval, ds)
    R = Rewards(ask=-args.ask_cost)
    rec = {"tag": args.tag, "learner": args.learner, "seed": args.seed, "router_seed": args.router_seed, "eval": args.eval,
           "H": H, "rewards": asdict(R), "memory": not args.no_memory,
           "router_acc": {s: [float(e.ok[~e.is_oos, v].float().mean()) for v in range(H)] for s, e in (("train_oof", ep_tr), ("val", ep_va), (args.eval, ep_ev))}}
    torch.manual_seed(args.seed)
    if args.learner in ("tm_q", "tm_myopic", "lin_q"):
        gamma = 0.0 if args.learner == "tm_myopic" else args.gamma
        if args.learner == "lin_q":
            mu = ep_tr.conf.reshape(-1, 2).mean(0)
            sd = ep_tr.conf.reshape(-1, 2).std(0)
            feat = lambda ep, i, s: cont_features_h(ep, i, s, mu, sd, memory=not args.no_memory)  # noqa: E731
            dim = feat(ep_tr, torch.zeros(1, dtype=torch.long, device=DEV), torch.zeros(1, dtype=torch.long, device=DEV)).shape[1]
            agent = LinQH(dim, args.lr, args.seed)
            rec.update({"lr": args.lr})
        else:
            coder = MetaCoderH(H, args.bins, memory=not args.no_memory).fit(ep_tr)
            feat = lambda ep, i, s: coder.encode(ep, i, s)  # noqa: E731
            agent = TMQ(len(coder.names), args.clauses, args.T, args.s, args.seed, args.kappa)
            rec.update({"clauses": args.clauses, "T": args.T, "s": args.s, "kappa": args.kappa, "bins": args.bins})
        losses = train_fqi(agent, feat, ep_tr, R, gamma, args.fqi_iters, args.fqi_epochs, args.batch, args.seed,
                           args.fqi_episodes, args.oos_share)
        pol = lambda ep: (lambda i, s: masked_greedy_h(agent.q(feat(ep, i, s)), s, H))  # noqa: E731
        rec.update({"gamma": gamma, "fqi_iters": args.fqi_iters, "fqi_epochs": args.fqi_epochs, "fqi_episodes": args.fqi_episodes,
                    "oos_share": args.oos_share, "loss_last": float(np.mean(losses[-100:]))})
        rv, re_ = evaluate_h(pol(ep_va), ep_va, R), evaluate_h(pol(ep_ev), ep_ev, R)
        if args.learner != "lin_q":
            rec["stats"] = agent.tm.stats()
            mem = "" if not args.no_memory else "_nomem"
            np.savez_compressed(out / f"r8_{args.learner}{mem}_seed{args.seed}{('_' + args.tag) if args.tag else ''}.npz",
                                **agent.tm.state(), names=np.array(coder.names))
    elif args.learner in ("rule", "rule2"):
        p = tune_rule(ep_va, R, use_top=args.learner == "rule2", seed=args.seed, oos_share=args.oos_share)
        rec["params"] = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in p.items()}
        rv, re_ = evaluate_h(rule_policy(p, ep_va, args.learner == "rule2"), ep_va, R), evaluate_h(rule_policy(p, ep_ev, args.learner == "rule2"), ep_ev, R)
    elif args.learner in ("route", "ask_route"):
        fixed = ROUTE if args.learner == "route" else ASK
        pol = lambda i, s: torch.where(s < H - 1, torch.full_like(i, fixed), torch.full_like(i, ROUTE))  # noqa: E731
        rv, re_ = evaluate_h(pol, ep_va, R), evaluate_h(pol, ep_ev, R)
    else:
        raise SystemExit(args.learner)
    for res, ep in ((rv, ep_va), (re_, ep_ev)):
        ret = res["_returns"]
        res["mean_return_at_oos_share"] = float((1 - args.oos_share) * ret[~ep.is_oos].mean() + args.oos_share * ret[ep.is_oos].mean())
    rec["val"] = strip(rv)
    rec["eval_result"] = strip(re_)
    rec["seconds"] = round(time.time() - t0, 1)
    np.save(out / f"r8_returns_{args.learner}{'_nomem' if args.no_memory else ''}_seed{args.seed}{('_' + args.tag) if args.tag else ''}_{args.eval}.npy",
            re_["_returns"].cpu().numpy())
    with open(out / "r8.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "stats"}))


if __name__ == "__main__":
    main()
