#!/usr/bin/env python
"""W1 engine, autoencoder task: the recipe of the TM autoencoder (Bhattarai et al. 2024) on the GPU CoTM.

    mo_tm_ae.py --slice S1 --tag NAME [--clauses 4000 --T 8000 --s 5 --u 25 --rounds 200 --batch 1024]

Machine: as mo_tm.py (dl/tm.py CoTM read only and unchanged; one shared clause pool; one integer weight per
(clause, output token); no negated literals; empty start; optional exact literal budget).
Task (self-supervised, per output balanced, as the paper's Algorithm 1 with accumulation u): a round gives every
output token t one positive and one negative example, in a random order, --batch examples per minibatch:
  positive  the OR of the context sets of u random occurrences of t (window W on each side, inside the document),
            with t's own type masked; CoTM target-class feedback for class t only (pos_mask);
  negative  the OR of the context sets of u random positions of the stream, t's type masked; negative-class
            feedback for class t only (neg_mask).
  One CoTM.feedback(X, pos, pos_mask, neg, neg_mask) call per minibatch.
Held-out check (self-supervised, never an evaluation set): up to 2,000 output tokens with at least u occurrences in
  the held-out slice H; input = the OR of u of their H contexts (t masked); the rank of t among t and 99 unigram^0.75
  negatives by class sum; mean reciprocal rank (ties at their average rank). After every --eval-every rounds; the
  best state is kept; --patience checks without a better MRR stop the run.
Readout R3a (test 5's rule), files and records as mo_tm.py (rows in results_w1_train.jsonl, task "ae").
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_tm as M  # noqa: E402
from dl.tm import CHUNK_ELEMS, TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402


class AEStream(M.Stream):
    """mo_tm.Stream plus an index from output token to its positions, and accumulated context bags."""

    def __init__(self, ids, V, window, dev):
        super().__init__(ids, V, window, dev)
        n_out = int((V["out_index"] >= 0).sum())
        yp = self.y[self.positions]
        order = torch.argsort(yp, stable=True)
        self.pos_sorted = self.positions[order]
        self.cnt = torch.bincount(yp, minlength=n_out)
        self.start = torch.cumsum(self.cnt, 0) - self.cnt

    def occurrences(self, cls: torch.Tensor, u: int, gen) -> torch.Tensor:
        r = torch.rand((cls.shape[0], u), generator=gen, device=self.dev)
        c = self.cnt[cls].unsqueeze(1)
        k = (r * c).long().clamp(max=(c - 1).clamp(min=0))
        return self.pos_sorted[(self.start[cls].unsqueeze(1) + k).clamp(max=self.pos_sorted.shape[0] - 1)]

    def anywhere(self, k: int, u: int, gen) -> torch.Tensor:
        i = torch.randint(0, self.positions.shape[0], (k, u), generator=gen, device=self.dev)
        return self.positions[i]

    def bag(self, P: torch.Tensor, mask_tok: torch.Tensor) -> torch.Tensor:
        k, u = P.shape
        n = self.ids.shape[0]
        Q = P.unsqueeze(2) + self.off.view(1, 1, -1)
        inside = (Q >= 0) & (Q < n)
        Qc = Q.clamp(0, n - 1)
        tq = self.ids[Qc]
        valid = inside & (tq >= 0) & (self.doc[Qc] == self.doc[P].unsqueeze(2))
        valid &= tq != mask_tok.view(k, 1, 1)
        f = self.ctx_index[tq.clamp(min=0)]
        valid &= f >= 0
        X = torch.zeros((k, self.F), dtype=torch.uint8, device=self.dev)
        rows = torch.arange(k, device=self.dev).view(k, 1, 1).expand_as(f)
        X[rows[valid], f[valid]] = 1
        return X


def heldout_ae(V, args, dev):
    h = M.subsample(M.load_slice("H"), V["counts"], args.sub_t, args.sub_seed + 1)
    st = AEStream(h, V, args.window, dev)
    g = torch.Generator(device=dev)
    g.manual_seed(7)
    elig = torch.nonzero(st.cnt >= args.u).squeeze(1)
    cls = elig[torch.randperm(elig.shape[0], generator=g, device=dev)[:2000]]
    out_ids = torch.as_tensor(V["out_ids"], device=dev)
    X = st.bag(st.occurrences(cls, args.u, g), out_ids[cls])
    q = M.neg_dist(V, dev)
    g.manual_seed(11)
    cand = torch.multinomial(q, cls.shape[0] * 99, replacement=True, generator=g).view(-1, 99)
    while bool((cand == cls.unsqueeze(1)).any()):
        bad = cand == cls.unsqueeze(1)
        cand[bad] = torch.multinomial(q, int(bad.sum()), replacement=True, generator=g)
    return X, cls, cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True, choices=("S1", "S10", "S100"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--u", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--init", default="empty", choices=("empty", "half"))
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--features", type=int, default=12000)
    ap.add_argument("--max-outputs", type=int, default=0)
    ap.add_argument("--max-literals", type=int, default=0)
    ap.add_argument("--sub-t", type=float, default=1e-4)
    ap.add_argument("--sub-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=10, help="rounds between held-out checks")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--mem", type=float, default=0.25)
    ap.add_argument("--ckpt-minutes", type=float, default=20.0)
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem)
    free0, total = torch.cuda.mem_get_info()
    ids = M.load_slice(args.slice)
    V = M.vocab(ids, args.min_count, args.features, args.max_outputs)
    sub = M.subsample(ids, V["counts"], args.sub_t, args.sub_seed)
    st = AEStream(sub, V, args.window, dev)
    n_out, F = len(V["out_ids"]), st.F
    if args.max_literals:
        assert args.clauses * F <= CHUNK_ELEMS, f"clauses x features must be <= {CHUNK_ELEMS:,} with a literal budget"
    cfg = TMConfig(n_clauses=args.clauses, T=args.T, s=args.s, batch_size=args.batch, neg_sampling="random",
                   negation=False, max_literals=args.max_literals, budget_mode="exact", budget_select="rank",
                   half=True, seed=args.seed)
    tm = make_tm(n_out, F, cfg, args.init)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    out_ids = torch.as_tensor(V["out_ids"], device=dev)
    live = torch.nonzero(st.cnt > 0).squeeze(1)
    info = {"tag": args.tag, "task": "ae", "slice": args.slice, "tokens_in_slice": int((ids >= 0).sum()),
            "tokens_after_subsampling": int((sub >= 0).sum()), "outputs": n_out, "outputs_with_positions": int(live.shape[0]),
            "features": F, "cfg": vars(args), "gpu_total_gib": round(total / 2**30, 1), "gpu_free_gib_at_start": round(free0 / 2**30, 1),
            "sha256": {"mo_tm_ae.py": M.sha16(Path(__file__).resolve()), "mo_tm.py": M.sha16(HERE / "mo_tm.py"),
                       "dl/tm.py": M.sha16(Path(f"{_REPO_ROOT}/studies/decision_layer/src/dl/tm.py"))}}
    print(json.dumps({k: v for k, v in info.items() if k != "cfg"}), flush=True)
    del ids
    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "ckpt").mkdir(exist_ok=True)
    table_path = HERE / "tables" / f"{args.tag}.npz"
    assert not table_path.exists(), f"{table_path} exists: a new version needs a new tag"
    Xh, yh, ch = heldout_ae(V, args, dev)
    half = args.batch // 2
    curve, best, best_state, stale, best_at, last_ck = [], -1.0, None, 0, None, time.time()

    def checkpoint(name, rnd):
        p = HERE / "ckpt" / f"{args.tag}_{name}.npz"
        np.savez(p.with_suffix(".tmp.npz"), ta=tm.ta.cpu().numpy().astype(np.int16), w=tm.w.cpu().numpy(), round=rnd,
                 curve=json.dumps(curve), out_ids=V["out_ids"], ctx_ids=V["ctx_ids"])
        os.replace(p.with_suffix(".tmp.npz"), p)

    t_train, n_ex, stop = time.time(), 0, False
    for rnd in range(args.rounds):
        order = live[torch.randperm(live.shape[0], generator=gen, device=dev)]
        for a in range(0, order.shape[0], half):
            c = order[a : a + half]
            k = c.shape[0]
            Xp = st.bag(st.occurrences(c, args.u, gen), out_ids[c])
            Xn = st.bag(st.anywhere(k, args.u, gen), out_ids[c])
            X = torch.cat([Xp, Xn])
            cc = torch.cat([c, c])
            is_pos = torch.cat([torch.ones(k, dtype=torch.bool, device=dev), torch.zeros(k, dtype=torch.bool, device=dev)])
            tm.feedback(X, cc, is_pos, cc, ~is_pos)
            n_ex += 2 * k
        if (rnd + 1) % args.eval_every == 0 or rnd == args.rounds - 1:
            sc = M.score_heldout(tm, Xh, yh, ch)
            curve.append({"round": rnd + 1, "examples": n_ex, **sc, "t": round(time.time() - t_train, 1)})
            print(json.dumps(curve[-1]), flush=True)
            if sc["mrr"] > best:
                best, stale, best_at = sc["mrr"], 0, len(curve)
                best_state = (tm.ta.cpu(), tm.w.cpu())
            else:
                stale += 1
                if args.patience and stale >= args.patience:
                    stop = True
        if time.time() - last_ck > 60 * args.ckpt_minutes:
            checkpoint("last", rnd + 1)
            last_ck = time.time()
        if stop:
            break
    torch.cuda.synchronize()
    train_s = time.time() - t_train
    if best_state is not None:
        tm.ta, tm.w = best_state[0].to(dev), best_state[1].to(dev)
    checkpoint("final", rnd + 1)
    codes = M.readout_r3a(tm, args.T)
    np.savez(table_path, vocab_ids=V["out_ids"].astype(np.int32), codes=codes, counts=V["counts"][V["out_ids"]].astype(np.int64))
    rec = {**info, "utc": M.utc(), "curve": curve, "best_heldout_mrr": best, "best_check": best_at, "stopped_early": stop,
           "rounds_done": rnd + 1, "examples": n_ex, "train_s": round(train_s, 1), "examples_per_s": round(n_ex / max(train_s, 1e-9), 1),
           "total_s": round(time.time() - t0, 1), "stats": tm.stats(), "rows_without_vector": int((np.abs(codes).sum(1) == 0).sum()),
           "table": str(table_path.relative_to(HERE)), "table_sha256": M.sha16(table_path),
           "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(json.dumps({k: v for k, v in rec.items() if k != "curve"}), flush=True)
    with open(HERE / "results_w1_train.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
