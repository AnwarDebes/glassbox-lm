#!/usr/bin/env python
"""W1 engine: a multi-output Tsetlin machine on the GPU that learns a word table from raw text alone.

    mo_tm.py --slice S1 --tag NAME [--clauses 4000 --T 4000 --s 5 --batch 1024 --epochs 3]   # train, write the table
    mo_tm.py --slice S1 --pilot 200 --tag NAME [...]   # speed pilot: timing and memory only, no table

Machine: the decision-layer CoTM (dl/tm.py, read only and unchanged) with one shared pool of clauses and one integer
weight per (clause, output token): n_classes = the output vocabulary, n_features = the context vocabulary, no
negated literals (as the TM autoencoder of Bhattarai et al. 2023), started empty (dl.tm_init "empty", as tmu).
Task (self-supervised, CBOW-like; no label of any dataset is used): an example is one position of the token stream.
  input     the set of context tokens within a dynamic window (b drawn uniformly from 1..W on each side, inside the
            document), the target's own token type left out (masked);
  positive  the token at that position (CoTM target-class feedback);
  negative  one output token drawn from the unigram^0.75 distribution (word2vec's negative distribution; the
            CoTM's default hardest negative would push near-synonyms apart, which a word table must not do); a draw
            equal to the target gets no feedback.
  Training calls CoTM.feedback(X, pos, None, neg, neg_mask) once per minibatch (dl/tm.py's batch rule).
Stream: the slice's potion-base-32M token ids (w1/corpus); frequent tokens are subsampled once before the windows
  ("dirty", word2vec's rule keep = min(1, sqrt(t / f) + t / f), numpy default_rng(--sub-seed)); [PAD] [UNK] [CLS]
  [SEP] [MASK] (and, in H, tokens the slice never saw) are dropped from the stream.
Vocabularies: outputs = tokens with at least --min-count occurrences in the slice; contexts = the --features most
  frequent tokens. The count twin (ppmi_svd.py) uses the same two lists and the same subsampled stream.
Held-out check (self-supervised, never an evaluation set): --heldout positions of the held-out slice H (same
  subsampling rule with the slice's counts, full window W), each with 99 negatives from unigram^0.75: rank of the
  true token by class sum; mean reciprocal rank (ties at their average rank) and strict top-1 share.
Readout R3a (test 5's rule): the row of output token w = clip(W[:, w], 0, T) * [W[:, w] >= 2], L2-normalized
  (an all-zero row = no vector). Written as tables/{tag}.npz (vocab_ids = potion token ids, codes float16).
Records: one JSON row per run in results_w1_train.jsonl (pilot rows in results_w1_pilot.jsonl), checkpoints
  ckpt/{tag}_*.npz (TA states int16, weights int32, the step) so that a stopped run can resume (--resume).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
from dl.tm import TMConfig  # noqa: E402
from dl.tm_init import make_tm  # noqa: E402

CORPUS = HERE / "corpus"
V_POTION = 63091
SPECIAL = (0, 1, 2, 3, 4)  # [PAD] [UNK] [CLS] [SEP] [MASK] (potion-base-32M token ids)


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_slice(name: str) -> np.ndarray:
    man = json.loads((CORPUS / "c4_slices.json").read_text())
    if name == "H":
        a = np.load(CORPUS / "c4_H.ids.npy")
    else:
        a = np.load(CORPUS / "c4_S100.ids.npy", mmap_mode="r")[: man["slices"][name]["end_offset_in_S100"]]
        a = np.ascontiguousarray(a)
    got = hashlib.sha256(a.astype(np.int32).tobytes()).hexdigest()
    assert got == man["slices"][name]["sha256_int32"], f"slice {name} does not match c4_slices.json"
    return a


def vocab(ids: np.ndarray, min_count: int, n_features: int, max_outputs: int = 0) -> dict:
    """Outputs = tokens with at least min_count occurrences (the max_outputs most frequent of them if > 0);
    contexts = the n_features most frequent tokens."""
    cnt = np.bincount(ids[ids >= 0], minlength=V_POTION).astype(np.int64)
    cnt[list(SPECIAL)] = 0
    order = np.argsort(-cnt, kind="stable")
    out_ids = order[cnt[order] >= min_count]
    if max_outputs:
        out_ids = out_ids[:max_outputs]
    ctx_ids = order[: n_features][cnt[order[:n_features]] > 0]
    out_index = np.full(V_POTION, -1, np.int64)
    out_index[out_ids] = np.arange(len(out_ids))
    ctx_index = np.full(V_POTION, -1, np.int64)
    ctx_index[ctx_ids] = np.arange(len(ctx_ids))
    return {"counts": cnt, "out_ids": out_ids, "ctx_ids": ctx_ids, "out_index": out_index, "ctx_index": ctx_index}


def subsample(ids: np.ndarray, counts: np.ndarray, t: float, seed: int) -> np.ndarray:
    """Word2vec's frequent-token subsampling, applied once; document separators (-1) are always kept. Tokens with
    count 0 in the training slice (the five special tokens, and in H the tokens the slice never saw) are dropped."""
    total = counts.sum()
    f = np.where(counts > 0, counts / max(total, 1), 1.0)
    keep_p = np.where(counts > 0, np.minimum(1.0, np.sqrt(t / f) + t / f), 0.0)
    u = np.random.default_rng(seed).random(len(ids))
    tok = np.maximum(ids, 0)
    keep = (ids < 0) | (u < keep_p[tok])
    return ids[keep]


class Stream:
    """The subsampled token stream on the GPU and the minibatch builder (all index work on the GPU)."""

    def __init__(self, ids: np.ndarray, V: dict, window: int, dev: torch.device):
        self.dev = dev
        self.W = window
        self.ids = torch.as_tensor(ids.astype(np.int64), device=dev)
        self.doc = torch.cumsum((self.ids < 0).long(), 0)
        self.out_index = torch.as_tensor(V["out_index"], device=dev)
        self.ctx_index = torch.as_tensor(V["ctx_index"], device=dev)
        tok = self.ids.clamp(min=0)
        self.y = torch.where(self.ids >= 0, self.out_index[tok], torch.full_like(self.ids, -1))
        self.positions = torch.nonzero(self.y >= 0).squeeze(1)
        o = torch.arange(1, window + 1, device=dev)
        self.off = torch.cat([-o.flip(0), o])  # [-W..-1, 1..W]
        self.dist = self.off.abs()
        self.F = int((V["ctx_index"] >= 0).sum())

    def batch(self, P: torch.Tensor, gen: torch.Generator | None):
        n = self.ids.shape[0]
        Q = P.unsqueeze(1) + self.off.unsqueeze(0)
        inside = (Q >= 0) & (Q < n)
        Qc = Q.clamp(0, n - 1)
        tq = self.ids[Qc]
        valid = inside & (tq >= 0) & (self.doc[Qc] == self.doc[P].unsqueeze(1))
        if gen is not None:  # dynamic window
            b = torch.randint(1, self.W + 1, (P.shape[0], 1), generator=gen, device=self.dev)
            valid &= self.dist.unsqueeze(0) <= b
        tgt = self.ids[P]
        valid &= tq != tgt.unsqueeze(1)  # the target's own type is masked
        f = self.ctx_index[tq.clamp(min=0)]
        valid &= f >= 0
        X = torch.zeros((P.shape[0], self.F), dtype=torch.uint8, device=self.dev)
        r = torch.arange(P.shape[0], device=self.dev).unsqueeze(1).expand_as(f)
        X[r[valid], f[valid]] = 1
        return X, self.y[P]


def neg_dist(V: dict, dev) -> torch.Tensor:
    c = V["counts"][V["out_ids"]].astype(np.float64) ** 0.75
    return torch.as_tensor(c / c.sum(), dtype=torch.float32, device=dev)


def heldout_set(V: dict, args, dev):
    """Fixed held-out examples from slice H: inputs, targets and 99 negative candidates each."""
    h = subsample(load_slice("H"), V["counts"], args.sub_t, args.sub_seed + 1)
    st = Stream(h, V, args.window, dev)
    g = torch.Generator(device=dev)
    g.manual_seed(7)
    pick = st.positions[torch.randperm(st.positions.shape[0], generator=g, device=dev)[: args.heldout]]
    X, y = st.batch(pick, None)  # full window
    q = neg_dist(V, dev)
    g.manual_seed(11)
    cand = torch.multinomial(q, X.shape[0] * 99, replacement=True, generator=g).view(-1, 99)
    while bool((cand == y.unsqueeze(1)).any()):
        bad = cand == y.unsqueeze(1)
        cand[bad] = torch.multinomial(q, int(bad.sum()), replacement=True, generator=g)
    return X, y, cand


def score_heldout(tm, X, y, cand, bs: int = 512) -> dict:
    rr, top1 = [], []
    wf = tm.w.float()
    for a in range(0, X.shape[0], bs):
        co = tm.clause_outputs(X[a : a + bs]).float()  # inference rule: empty clauses are silent
        allc = torch.cat([y[a : a + bs].unsqueeze(1), cand[a : a + bs]], 1)  # [b, 100]
        S = (co @ wf).gather(1, allc)  # class sums of the true token and its 99 candidates
        s0 = S[:, :1]
        gt, eq = (S[:, 1:] > s0).sum(1).float(), (S[:, 1:] == s0).sum(1).float()
        rr.append(1.0 / (1.0 + gt + 0.5 * eq))
        top1.append(((gt == 0) & (eq == 0)).float())
    return {"mrr": round(float(torch.cat(rr).mean()), 4), "top1": round(float(torch.cat(top1).mean()), 4)}


def readout_r3a(tm, T: int) -> np.ndarray:
    W = tm.w.T.float()  # [outputs, clauses]
    R = torch.clamp(W, 0, T) * (W >= 2).float()
    return torch.nn.functional.normalize(R, dim=1).cpu().numpy().astype(np.float16)


def readout_r5(tm, st: Stream, n_out: int, n_per: int = 100, seed: int = 5, bs: int = 4096) -> np.ndarray:
    """R5 (reported, no bar): per output token, the mean clause output (inference rule) over up to n_per of its
    training positions (full window; the first n_per in a random order with a fixed seed), minus the mean over all
    sampled positions, L2-normalized. The TM's own learned features counted around each word."""
    g = torch.Generator(device=st.dev)
    g.manual_seed(seed)
    pos = st.positions[torch.randperm(st.positions.shape[0], generator=g, device=st.dev)]
    y = st.y[pos]
    order = torch.argsort(y, stable=True)
    ys = y[order]
    rank = torch.arange(ys.shape[0], device=st.dev) - torch.searchsorted(ys, ys)
    keep = order[rank < n_per]
    P = pos[keep]
    acc = torch.zeros((n_out, tm.cfg.n_clauses), device=st.dev)
    cnt = torch.zeros(n_out, device=st.dev)
    for a in range(0, P.shape[0], bs):
        X, yy = st.batch(P[a : a + bs], None)
        acc.index_add_(0, yy, tm.clause_outputs(X).float())
        cnt.index_add_(0, yy, torch.ones(yy.shape[0], device=st.dev))
    glob = acc.sum(0) / cnt.sum().clamp(min=1)
    R = (acc / cnt.clamp(min=1).unsqueeze(1) - glob.unsqueeze(0)) * (cnt > 0).float().unsqueeze(1)
    return torch.nn.functional.normalize(R, dim=1).cpu().numpy().astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True, choices=("S1", "S10", "S100"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--clauses", type=int, default=4000)
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--init", default="empty", choices=("empty", "half"))
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--features", type=int, default=20000)
    ap.add_argument("--max-outputs", type=int, default=0, help="keep only the most frequent output tokens (0 = all)")
    ap.add_argument("--max-literals", type=int, default=0, help="dl/tm.py literal budget per clause, exact mode (0 = none)")
    ap.add_argument("--sub-t", type=float, default=1e-4)
    ap.add_argument("--sub-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--heldout", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=0, help="minibatches between held-out checks (0: once per epoch)")
    ap.add_argument("--patience", type=int, default=2, help="held-out checks without a better MRR before stopping")
    ap.add_argument("--mem", type=float, default=0.25, help="cap on this process's share of the GPU memory")
    ap.add_argument("--pilot", type=int, default=0, help="speed pilot: time this many minibatches, write no table")
    ap.add_argument("--ckpt-minutes", type=float, default=20.0)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--r5", action="store_true", help="also write the R5 table, tables/{tag}_R5.npz")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem)
    free0, total = torch.cuda.mem_get_info()
    ids = load_slice(args.slice)
    V = vocab(ids, args.min_count, args.features, args.max_outputs)
    sub = subsample(ids, V["counts"], args.sub_t, args.sub_seed)
    st = Stream(sub, V, args.window, dev)
    n_out, F = len(V["out_ids"]), st.F
    if args.max_literals:  # dl/tm.py supports literal budgets only in its unchunked automata update
        from dl.tm import CHUNK_ELEMS
        assert args.clauses * F <= CHUNK_ELEMS, f"clauses x features must be <= {CHUNK_ELEMS:,} with a literal budget"
    cfg = TMConfig(n_clauses=args.clauses, T=args.T, s=args.s, epochs=args.epochs, batch_size=args.batch,
                   neg_sampling="random", negation=False, max_literals=args.max_literals, budget_mode="exact",
                   budget_select="rank", half=True, seed=args.seed, eval_every=1)
    tm = make_tm(n_out, F, cfg, args.init)
    q = neg_dist(V, dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    info = {"tag": args.tag, "slice": args.slice, "tokens_in_slice": int((ids >= 0).sum()),
            "tokens_after_subsampling": int((sub >= 0).sum()), "examples_per_epoch": int(st.positions.shape[0]),
            "outputs": n_out, "features": F, "cfg": {k: v for k, v in vars(args).items() if k not in ("resume",)},
            "gpu_total_gib": round(total / 2**30, 1), "gpu_free_gib_at_start": round(free0 / 2**30, 1),
            "sha256": {"mo_tm.py": sha16(Path(__file__).resolve()),
                       "dl/tm.py": sha16(Path(f"{_REPO_ROOT}/studies/decision_layer/src/dl/tm.py")),
                       "dl/tm_init.py": sha16(Path(f"{_REPO_ROOT}/studies/decision_layer/src/dl/tm_init.py"))}}
    print(json.dumps({k: v for k, v in info.items() if k != "cfg"}), flush=True)
    del ids

    def step(P):
        X, pos = st.batch(P, gen)
        neg = torch.multinomial(q, P.shape[0], replacement=True, generator=gen)
        tm.feedback(X, pos, None, neg, neg != pos)

    if args.pilot:
        perm = st.positions[torch.randperm(st.positions.shape[0], generator=gen, device=dev)]
        warm = 20
        for i in range(warm):
            step(perm[i * args.batch : (i + 1) * args.batch])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        for i in range(warm, warm + args.pilot):
            step(perm[i * args.batch : (i + 1) * args.batch])
        torch.cuda.synchronize()
        dt = time.time() - t1
        rec = {**info, "utc": utc(), "pilot_minibatches": args.pilot, "seconds": round(dt, 2),
               "ms_per_minibatch": round(1000 * dt / args.pilot, 2),
               "examples_per_s": round(args.pilot * args.batch / dt, 1),
               "projected_hours_per_epoch": {"S1": None, "S10": None, "S100": None},
               "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "stats": tm.stats()}
        man = json.loads((CORPUS / "c4_slices.json").read_text())
        frac = info["examples_per_epoch"] / max(info["tokens_in_slice"], 1)
        for k in rec["projected_hours_per_epoch"]:
            rec["projected_hours_per_epoch"][k] = round(man["slices"][k]["tokens"] * frac / rec["examples_per_s"] / 3600, 3)
        print(json.dumps(rec), flush=True)
        with open(HERE / "results_w1_pilot.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return

    (HERE / "ckpt").mkdir(exist_ok=True)
    (HERE / "tables").mkdir(exist_ok=True)
    table_path = HERE / "tables" / f"{args.tag}.npz"
    assert not table_path.exists(), f"{table_path} exists: a new version needs a new tag"
    Xh, yh, ch = heldout_set(V, args, dev)
    start_epoch, start_batch, curve, best, best_state, stale, best_at = 0, 0, [], -1.0, None, 0, None
    if args.resume is not None:
        z = np.load(args.resume, allow_pickle=False)
        tm.ta = torch.as_tensor(z["ta"].astype(np.int32), device=dev)
        tm.w = torch.as_tensor(z["w"], device=dev)
        start_epoch, start_batch = int(z["epoch"]), int(z["batch"])
        curve = json.loads(str(z["curve"]))
        gen.manual_seed(args.seed + 1000 * start_epoch + start_batch)  # a resumed run draws a new stream
        print(f"resumed from {args.resume} at epoch {start_epoch} batch {start_batch}", flush=True)
    last_ck = time.time()

    def checkpoint(epoch, batch, name):
        p = HERE / "ckpt" / f"{args.tag}_{name}.npz"
        w = tm.w.cpu().numpy()
        np.savez(p.with_suffix(".tmp.npz"), ta=tm.ta.cpu().numpy().astype(np.int16), w=w, epoch=epoch, batch=batch,
                 curve=json.dumps(curve), out_ids=V["out_ids"], ctx_ids=V["ctx_ids"])
        os.replace(p.with_suffix(".tmp.npz"), p)

    t_train = time.time()
    n_pos = st.positions.shape[0]
    nb = (n_pos + args.batch - 1) // args.batch
    every = args.eval_every or nb
    stop = False
    for epoch in range(start_epoch, args.epochs):
        g_ep = torch.Generator(device=dev)
        g_ep.manual_seed(args.seed * 100003 + epoch)
        perm = st.positions[torch.randperm(n_pos, generator=g_ep, device=dev)]
        for bi in range(start_batch if epoch == start_epoch else 0, nb):
            step(perm[bi * args.batch : (bi + 1) * args.batch])
            done = epoch * nb + bi + 1
            if done % every == 0 or bi == nb - 1:
                sc = score_heldout(tm, Xh, yh, ch)
                curve.append({"epoch": epoch + 1, "minibatch": bi + 1, "examples": done * args.batch, **sc,
                              "t": round(time.time() - t_train, 1)})
                print(json.dumps(curve[-1]), flush=True)
                if sc["mrr"] > best:
                    best, stale = sc["mrr"], 0
                    best_state = (tm.ta.cpu(), tm.w.cpu(), len(curve))  # kept off the GPU
                else:
                    stale += 1
                    if args.patience and stale >= args.patience:
                        stop = True
                        break
            if time.time() - last_ck > 60 * args.ckpt_minutes:
                checkpoint(epoch, bi + 1, "last")
                last_ck = time.time()
        if stop:
            break
    torch.cuda.synchronize()
    train_s = time.time() - t_train
    if best_state is not None:
        tm.ta, tm.w, best_at = best_state[0].to(dev), best_state[1].to(dev), best_state[2]
    checkpoint(args.epochs, 0, "final")
    codes = readout_r3a(tm, args.T)
    np.savez(table_path, vocab_ids=V["out_ids"].astype(np.int32), codes=codes,
             counts=V["counts"][V["out_ids"]].astype(np.int64))
    r5 = {}
    if args.r5:
        p5 = HERE / "tables" / f"{args.tag}_R5.npz"
        assert not p5.exists(), f"{p5} exists"
        np.savez(p5, vocab_ids=V["out_ids"].astype(np.int32), codes=readout_r5(tm, st, n_out),
                 counts=V["counts"][V["out_ids"]].astype(np.int64))
        r5 = {"table_R5": str(p5.relative_to(HERE)), "table_R5_sha256": sha16(p5)}
    rec = {**info, "utc": utc(), "curve": curve, "best_heldout_mrr": best, "best_check": best_at,
           "stopped_early": stop, "train_s": round(train_s, 1), "total_s": round(time.time() - t0, 1),
           "examples_per_s": round(len(curve) and curve[-1]["examples"] / max(train_s, 1e-9), 1),
           "stats": tm.stats(), "rows_without_vector": int((np.abs(codes).sum(1) == 0).sum()),
           "table": str(table_path.relative_to(HERE)), "table_sha256": sha16(table_path), **r5,
           "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(json.dumps({k: v for k, v in rec.items() if k != "curve"}), flush=True)
    with open(HERE / "results_w1_train.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
