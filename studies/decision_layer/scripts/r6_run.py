#!/usr/bin/env python
"""R6: clarify-or-route with delayed reward (CLINC150), one run.

    r6_run.py --learner tm_q --seed 1 [--gamma 1.0] [--eval val|test]

Learners:
  tm_q        Coalesced TM, one class per action; class sum / T is the Q-value; trained toward the
              TD target r + gamma * max_a' Q_target(s', a') with feedback probability ~ |TD error|
              (regression-TM style), epsilon-greedy behaviour, replay buffer, target copy.
  tm_myopic   the same with gamma = 0 (immediate reward only).
  lin_q       linear Q on the continuous features (same information as the bits), same TD loop.
  rule        the best 3-threshold rule (route / ask / escalate on the router margin), grid on val.
  route, ask_route  fixed policies.
Router: supervised TM (thermo:8, 4,000 clauses, 40 epochs), 5-fold cross-fitted on train so the
training episodes see held-out router scores; val/test episodes use the mean of the 5 fold routers.
Router outputs are cached per seed in results/<out>/r6_router_seed<k>.pt.
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

from dl.boolean import Thermo  # noqa: E402
from dl.data import load  # noqa: E402
from dl.mdp import ACTIONS, ASK, ESCALATE, ROUTE, Episodes, MetaCoder, Rewards, evaluate, half_text, step_rewards  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402

DEV = "cuda"


def half_embeddings(ds) -> dict:
    p = ROOT / "data" / "emb" / "clinc150__minilm__half.npz"
    if p.exists():
        d = np.load(p)
        return {k: d[k] for k in d.files}
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
    out = {}
    for split, texts in ds.texts.items():
        out[split] = m.encode([half_text(t) for t in texts], batch_size=256, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.savez(p, **out)
    return out


def router_scores(seed: int, out_dir: Path) -> dict:
    cache = out_dir / f"r6_router_seed{seed}.pt"
    if cache.exists():
        return torch.load(cache)
    ds = load("clinc150")
    full = np.load(ROOT / "data" / "emb" / "clinc150__minilm.npz")
    half = half_embeddings(ds)
    ytr, yva = full["y_train"], full["y_val"]
    therm = Thermo(8).fit(full["train"])
    enc = lambda X: torch.as_tensor(therm.transform(X), dtype=torch.uint8, device=DEV)  # noqa: E731
    Btr, Bva = therm.transform(full["train"]), therm.transform(full["val"])
    n = len(ytr)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % 5
    oof = {"half": torch.zeros((n, 150), device=DEV), "full": torch.zeros((n, 150), device=DEV)}
    views = {}
    for split in ("val", "test", "oos_train", "oos_val", "oos_test"):
        views[split] = {"half": enc(half[split]), "full": enc(full[split])}
    acc = {s: {"half": 0, "full": 0} for s in views}
    for k in range(5):
        tr = folds != k
        cfg = TMConfig(n_clauses=4000, T=4000, s=8.0, epochs=40, seed=seed * 10 + k)
        tm = CoTM(150, Btr.shape[1], cfg).fit(Btr[tr], ytr[tr], Bva, yva)
        te = np.flatnonzero(~tr)
        oof["half"][te] = tm.class_sums(enc(half["train"][te]))
        oof["full"][te] = tm.class_sums(enc(full["train"][te]))
        for s, v in views.items():
            for view in ("half", "full"):
                acc[s][view] = acc[s][view] + tm.class_sums(v[view]) / 5.0
    res = {"train": oof, **acc}
    torch.save(res, cache)
    return res


def build_episodes(scores: dict, split: str, ds) -> Episodes:
    lab = {"train": "train", "val": "val", "test": "test"}[split]
    y = ds.labels[lab]
    S_half = torch.cat([scores[lab]["half"], scores[f"oos_{lab}"]["half"]])
    S_full = torch.cat([scores[lab]["full"], scores[f"oos_{lab}"]["full"]])
    n_in, n_oos = len(y), scores[f"oos_{lab}"]["half"].shape[0]
    yy = np.r_[y, -np.ones(n_oos, dtype=np.int64)]
    is_oos = np.r_[np.zeros(n_in, bool), np.ones(n_oos, bool)]
    return Episodes(S_half, S_full, yy, is_oos)


# ----------------------------------------------------------------------------------------------
class TMQ:
    def __init__(self, n_bits, clauses, T, s, seed, kappa):
        cfg = TMConfig(n_clauses=clauses, T=T, s=s, seed=seed, epochs=1)
        self.tm = CoTM(3, n_bits, cfg)
        self.T, self.kappa = T, kappa
        self.target = None

    def q(self, X, target=False):
        tm = self.tm
        if target and self.target is not None:
            ta, w = tm.ta, tm.w
            tm.ta, tm.w = self.target
            S = tm.class_sums(X)
            tm.ta, tm.w = ta, w
        else:
            S = tm.class_sums(X)
        return (S / self.T).clamp(-1.5, 1.5)

    def sync(self):
        self.target = (self.tm.ta.clone(), self.tm.w.clone())

    def update(self, X, a, y_td):
        # train-rule outputs inside feedback; Q of the chosen action at inference rule for the error
        err = y_td - self.q(X).gather(1, a.view(-1, 1)).squeeze(1)
        p = (self.kappa * err.abs()).clamp(max=1.0)
        up = err > 0
        self.tm.feedback(X, a, up, a, ~up, p_pos=torch.where(up, p, torch.zeros_like(p)),
                         p_neg=torch.where(~up, p, torch.zeros_like(p)))
        return float(err.pow(2).mean())


class LinQ:
    def __init__(self, dim, lr, seed):
        g = torch.Generator().manual_seed(seed)
        self.W = (0.01 * torch.randn((dim, 3), generator=g)).to(DEV).requires_grad_(True)
        self.b = torch.zeros(3, device=DEV, requires_grad=True)
        self.opt = torch.optim.Adam([self.W, self.b], lr=lr)
        self.target = None

    def q(self, X, target=False):
        if target and self.target is not None:
            W, b = self.target
            return X @ W + b
        with torch.no_grad():
            return X @ self.W + self.b

    def sync(self):
        self.target = (self.W.detach().clone(), self.b.detach().clone())

    def update(self, X, a, y_td):
        self.opt.zero_grad()
        qa = (X @ self.W + self.b).gather(1, a.view(-1, 1)).squeeze(1)
        loss = (qa - y_td).pow(2).mean()
        loss.backward()
        self.opt.step()
        return float(loss)


def cont_features(ep: Episodes, idx, step, mu, sd):
    cur = (ep.conf[idx, step] - mu) / sd
    prev = (ep.conf[idx, 0] - mu) / sd
    s1 = (step == 1).float().unsqueeze(1)
    f = torch.cat([s1, cur, cur.pow(2), prev * s1, (prev * s1).pow(2), cur * s1], 1)
    return f


def masked_greedy(Q, step):
    Q = Q.clone()
    Q[step == 1, ASK] = -1e9
    return Q.argmax(1)


def train_td(agent, feat, ep_tr: Episodes, R: Rewards, gamma, n_updates, batch, eps, seed, sync_every, oos_share):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    in_idx = (~ep_tr.is_oos).nonzero().squeeze(1)
    oos_idx = ep_tr.is_oos.nonzero().squeeze(1)
    buf_s, buf_i, buf_a, buf_r, buf_d = [], [], [], [], []
    losses = []
    agent.sync()
    for it in range(n_updates):
        # collect one batch of fresh episodes (both steps) with the behaviour policy
        n_oos = int(round(oos_share * batch))
        idx = torch.cat([in_idx[torch.randint(0, len(in_idx), (batch - n_oos,), generator=g, device=DEV)],
                         oos_idx[torch.randint(0, len(oos_idx), (n_oos,), generator=g, device=DEV)]])
        step = torch.zeros_like(idx)
        a0 = masked_greedy(agent.q(feat(ep_tr, idx, step)), step)
        rnd = torch.randint(0, 3, a0.shape, generator=g, device=DEV)
        a0 = torch.where(torch.rand(a0.shape, generator=g, device=DEV) < eps, rnd, a0)
        r0, d0 = step_rewards(ep_tr, idx, step, a0, R)
        buf_i.append(idx); buf_s.append(step); buf_a.append(a0); buf_r.append(r0); buf_d.append(d0)
        cont = ~d0
        if cont.any():
            i1 = idx[cont]
            s1 = torch.ones_like(i1)
            a1 = masked_greedy(agent.q(feat(ep_tr, i1, s1)), s1)
            rnd = torch.randint(0, 2, a1.shape, generator=g, device=DEV) * 2  # ROUTE or ESCALATE
            a1 = torch.where(torch.rand(a1.shape, generator=g, device=DEV) < eps, rnd, a1)
            r1, d1 = step_rewards(ep_tr, i1, s1, a1, R)
            buf_i.append(i1); buf_s.append(s1); buf_a.append(a1); buf_r.append(r1); buf_d.append(torch.ones_like(d1))
        if len(buf_i) > 64:
            buf_i = [torch.cat(buf_i)[-200000:]]; buf_s = [torch.cat(buf_s)[-200000:]]
            buf_a = [torch.cat(buf_a)[-200000:]]; buf_r = [torch.cat(buf_r)[-200000:]]; buf_d = [torch.cat(buf_d)[-200000:]]
        I, S_, A, Rr, D = (torch.cat(x) for x in (buf_i, buf_s, buf_a, buf_r, buf_d))
        j = torch.randint(0, I.shape[0], (batch,), generator=g, device=DEV)
        i, st, a, r, d = I[j], S_[j], A[j], Rr[j], D[j]
        y = r.clone()
        nd = ~d
        if gamma > 0 and nd.any():
            s_next = torch.ones_like(i[nd])
            qn = agent.q(feat(ep_tr, i[nd], s_next), target=True)
            qn[:, ASK] = -1e9
            y[nd] = r[nd] + gamma * qn.max(1).values
        losses.append(agent.update(feat(ep_tr, i, st), a, y))
        if (it + 1) % sync_every == 0:
            agent.sync()
    return losses


def collect_random(ep_tr: Episodes, R: Rewards, n_episodes: int, seed: int, oos_share: float):
    """Exploration data for fitted-Q iteration: uniformly random allowed actions."""
    g = torch.Generator(device=DEV)
    g.manual_seed(seed + 99)
    in_idx = (~ep_tr.is_oos).nonzero().squeeze(1)
    oos_idx = ep_tr.is_oos.nonzero().squeeze(1)
    n_oos = int(round(oos_share * n_episodes))
    idx = torch.cat([in_idx[torch.randint(0, len(in_idx), (n_episodes - n_oos,), generator=g, device=DEV)],
                     oos_idx[torch.randint(0, len(oos_idx), (n_oos,), generator=g, device=DEV)]])
    st0 = torch.zeros_like(idx)
    a0 = torch.randint(0, 3, idx.shape, generator=g, device=DEV)
    r0, d0 = step_rewards(ep_tr, idx, st0, a0, R)
    i1 = idx[~d0]
    st1 = torch.ones_like(i1)
    a1 = torch.randint(0, 2, i1.shape, generator=g, device=DEV) * 2
    r1, _ = step_rewards(ep_tr, i1, st1, a1, R)
    I = torch.cat([idx, i1]); S_ = torch.cat([st0, st1]); A = torch.cat([a0, a1]); Rr = torch.cat([r0, r1])
    D = torch.cat([d0, torch.ones_like(r1, dtype=torch.bool)])
    return I, S_, A, Rr, D


def train_fqi(agent, feat, ep_tr: Episodes, R: Rewards, gamma, iters, epochs, batch, seed, n_episodes, oos_share):
    """Fitted-Q iteration: targets r + gamma * max_a' Q_k(s', a') are fixed per iteration; the TM is
    fitted to them by regression feedback for ``epochs`` passes."""
    I, S_, A, Rr, D = collect_random(ep_tr, R, n_episodes, seed, oos_share)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed + 7)
    X = feat(ep_tr, I, S_)
    nd = ~D
    Xn = feat(ep_tr, I[nd], torch.ones_like(I[nd]))
    losses = []
    for k in range(iters):
        y = Rr.clone()
        if gamma > 0 and nd.any():
            qn = agent.q(Xn)
            qn[:, ASK] = -1e9
            y[nd] = Rr[nd] + gamma * qn.max(1).values
        for e in range(epochs):
            perm = torch.randperm(X.shape[0], generator=g, device=DEV)
            for st in range(0, X.shape[0], batch):
                j = perm[st : st + batch]
                losses.append(agent.update(X[j], A[j], y[j]))
    return losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eval", default="val")
    ap.add_argument("--out", default="explore")
    ap.add_argument("--clauses", type=int, default=2000)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--s", type=float, default=4.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--sync", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--oos-share", type=float, default=0.15)
    ap.add_argument("--ask-cost", type=float, default=0.1)
    ap.add_argument("--no-memory", action="store_true", help="ablation: no step-0 literals at step 1")
    ap.add_argument("--fqi", action="store_true", help="tm_q/tm_myopic/lin_q: fitted-Q iteration instead of online TD")
    ap.add_argument("--fqi-iters", type=int, default=3)
    ap.add_argument("--fqi-epochs", type=int, default=3)
    ap.add_argument("--fqi-episodes", type=int, default=60000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.eval == "test" and args.out == "explore":
        raise SystemExit("test split only in registered runs (--out confirm)")
    t0 = time.time()
    out = ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)
    ds = load("clinc150")
    scores = router_scores(args.seed, out)
    ep_tr = build_episodes(scores, "train", ds)
    ep_va = build_episodes(scores, "val", ds)
    ep_ev = build_episodes(scores, args.eval, ds)
    R = Rewards(ask=-args.ask_cost)
    rec = {"tag": args.tag, "learner": args.learner, "seed": args.seed, "eval": args.eval, "rewards": asdict(R),
           "router_acc": {s: {v: float(e.ok[~e.is_oos, k].float().mean()) for k, v in enumerate(("half", "full"))}
                          for s, e in (("train_oof", ep_tr), ("val", ep_va), (args.eval, ep_ev))}}

    if args.learner in ("tm_q", "tm_myopic"):
        coder = MetaCoder(args.bins, memory=not args.no_memory).fit(ep_tr)
        feat = lambda ep, i, s: coder.encode(ep, i, s)  # noqa: E731
        n_bits = len(coder.names)
        agent = TMQ(n_bits, args.clauses, args.T, args.s, args.seed, args.kappa)
        gamma = 0.0 if args.learner == "tm_myopic" else args.gamma
        if args.fqi:
            losses = train_fqi(agent, feat, ep_tr, R, gamma, args.fqi_iters, args.fqi_epochs, args.batch, args.seed,
                               args.fqi_episodes, args.oos_share)
        else:
            losses = train_td(agent, feat, ep_tr, R, gamma, args.updates, args.batch, args.eps, args.seed, args.sync, args.oos_share)
        pol = lambda ep: (lambda i, s: masked_greedy(agent.q(feat(ep, i, s)), s))  # noqa: E731
        rec.update({"fqi": args.fqi, "fqi_iters": args.fqi_iters, "fqi_epochs": args.fqi_epochs, "fqi_episodes": args.fqi_episodes,
                    "memory": not args.no_memory, "gamma": gamma, "clauses": args.clauses, "T": args.T, "s": args.s, "kappa": args.kappa, "bins": args.bins,
                    "updates": args.updates, "loss_last": float(np.mean(losses[-100:])), "stats": agent.tm.stats()})
        rec["val"] = evaluate(pol(ep_va), ep_va, R)
        rec["eval_result"] = evaluate(pol(ep_ev), ep_ev, R)
        st = agent.tm.state()
        mem = "" if not args.no_memory else "_nomem"
        np.savez_compressed(out / f"r6_{args.learner}{mem}_seed{args.seed}{('_' + args.tag) if args.tag else ''}.npz", **st,
                            names=np.array(coder.names))
    elif args.learner == "lin_q":
        mu = ep_tr.conf.reshape(-1, 2).mean(0)
        sd = ep_tr.conf.reshape(-1, 2).std(0)
        feat = lambda ep, i, s: cont_features(ep, i, s, mu, sd)  # noqa: E731
        agent = LinQ(feat(ep_tr, torch.zeros(1, dtype=torch.long, device=DEV), torch.zeros(1, dtype=torch.long, device=DEV)).shape[1], args.lr, args.seed)
        if args.fqi:
            losses = train_fqi(agent, feat, ep_tr, R, args.gamma, args.fqi_iters, args.fqi_epochs, args.batch, args.seed,
                               args.fqi_episodes, args.oos_share)
        else:
            losses = train_td(agent, feat, ep_tr, R, args.gamma, args.updates, args.batch, args.eps, args.seed, args.sync, args.oos_share)
        pol = lambda ep: (lambda i, s: masked_greedy(agent.q(feat(ep, i, s)), s))  # noqa: E731
        rec.update({"fqi": args.fqi, "gamma": args.gamma, "lr": args.lr, "updates": args.updates, "loss_last": float(np.mean(losses[-100:]))})
        rec["val"] = evaluate(pol(ep_va), ep_va, R)
        rec["eval_result"] = evaluate(pol(ep_ev), ep_ev, R)
    elif args.learner == "rule":
        # route at step 0 if margin >= a; else escalate if margin < c; else ask; after asking,
        # route if margin >= b else escalate. Grid over quantiles of the val margins; the first
        # best in the order a (outer), b (middle), c (inner) is chosen. Vectorised on the GPU:
        # the return of every grid point is computed from the episodes' margins and outcomes.
        qs = torch.quantile(ep_va.conf[:, :, 1].reshape(-1), torch.linspace(0, 1, 41, device=DEV)).tolist()
        grid = sorted(set([-1e9] + qs + [1e9]))
        ga, gb, gc = torch.tensor(grid, device=DEV), torch.tensor(grid[::2], device=DEV), torch.tensor(grid[::4], device=DEV)
        m0, m1 = ep_va.conf[:, 0, 1], ep_va.conf[:, 1, 1]
        r_route0 = torch.where(ep_va.ok[:, 0], torch.full_like(m0, R.correct), torch.full_like(m0, R.wrong))
        r_route1 = torch.where(ep_va.ok[:, 1], torch.full_like(m1, R.correct), torch.full_like(m1, R.wrong))
        # after asking: return as a function of b  -> [nb, n]
        after = R.ask + torch.where(m1[None, :] >= gb[:, None], r_route1[None, :], torch.full_like(r_route1, R.escalate)[None, :])
        route_now = (m0[None, :] >= ga[:, None])  # [na, n]
        esc_now = (m0[None, :] < gc[:, None])  # [nc, n]
        # total[a, b, c] = mean over episodes
        tot = torch.empty((len(grid), len(gb), len(gc)), device=DEV)
        for ia in range(len(grid)):
            rn = route_now[ia]
            base = torch.where(rn, r_route0, torch.zeros_like(r_route0))  # routed now
            # not routed: escalate if m0 < c else ask
            ask_part = after[:, None, :] * (~rn & ~esc_now)[None, :, :]  # [nb, nc, n]
            esc_part = (R.escalate * (~rn & esc_now).float())[None, :, :]
            tot[ia] = (base[None, None, :] + ask_part + esc_part).mean(2)
        valid = gc[None, None, :] <= ga[:, None, None]  # c > a is skipped, as in the loop version
        flat = torch.where(valid.expand_as(tot), tot, torch.full_like(tot, -1e9)).reshape(-1)
        best_v = flat.max()
        k = int((flat >= best_v).nonzero()[0])  # first maximum in (a, b, c) order
        ia, rem = divmod(k, len(gb) * len(gc))
        ib, ic = divmod(rem, len(gc))
        a, b, c = grid[ia], grid[::2][ib], grid[::4][ic]
        def mk(a, b, c):
            def pol(i, s, ep):
                m = ep.conf[i, s, 1]
                act0 = torch.where(m >= a, torch.full_like(i, ROUTE), torch.where(m < c, torch.full_like(i, ESCALATE), torch.full_like(i, ASK)))
                act1 = torch.where(m >= b, torch.full_like(i, ROUTE), torch.full_like(i, ESCALATE))
                return torch.where(s == 0, act0, act1)
            return pol
        pol = mk(a, b, c)
        rec["thresholds"] = {"route_if_margin_ge": a, "after_ask_route_if_ge": b, "escalate_if_lt": c}
        rec["val"] = evaluate(lambda i, s: pol(i, s, ep_va), ep_va, R)
        rec["eval_result"] = evaluate(lambda i, s: pol(i, s, ep_ev), ep_ev, R)
    elif args.learner in ("route", "ask_route"):
        fixed = ROUTE if args.learner == "route" else ASK
        pol = lambda i, s: torch.where(s == 0, torch.full_like(i, fixed), torch.full_like(i, ROUTE))  # noqa: E731
        rec["val"] = evaluate(pol, ep_va, R)
        rec["eval_result"] = evaluate(pol, ep_ev, R)
    else:
        raise SystemExit(args.learner)
    rec["seconds"] = round(time.time() - t0, 1)
    with open(out / "r6.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
