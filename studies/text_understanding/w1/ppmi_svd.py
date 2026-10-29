#!/usr/bin/env python
"""W1 count-only twin: PPMI-SVD on the same slice, the same subsampled stream and the same two vocabularies as the
Tsetlin machine (mo_tm.py), with the settings Levy, Goldberg and Dagan (2015) recommend for count models.

    ppmi_svd.py --slice S10 --tag NAME [--dims 300,500,1000 --eigs 0,0.5]   # one table per (dim, eig)

Counts: for every pair of positions at distance d <= W inside one document of the subsampled stream (mo_tm.subsample,
  same t and seed), target = an output token, context = a context token, the two of different token types (the TM
  masks the target's own type too), weight (W - d + 1) / W: the expected weight of word2vec's dynamic window, which
  is what the TM's dynamic window samples. Both directions.
PPMI: PMI(w, c) = log(P(w, c) / (P(w) P_a(c))) with context distribution smoothing P_a(c) = n(c)^0.75 / sum n^0.75;
  PPMI = max(PMI, 0); a pair never seen gets 0.
SVD: torch.svd_lowrank (q = max dim + 16, 6 power iterations, torch.manual_seed(0); GPU when it fits, else CPU),
  rows U_d * S_d^eig, L2-normalized (a token with an all-zero PPMI row gets no vector). One table per (dim, eig):
  tables/{tag}_d{dim}_e{eig}.npz, the format of mo_tm.py's tables.
"""

from __future__ import annotations

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


def cooc(sub: np.ndarray, V: dict, W: int, dev) -> torch.Tensor:
    ids = torch.as_tensor(sub.astype(np.int64), device=dev)
    doc = torch.cumsum((ids < 0).long(), 0)
    oi = torch.as_tensor(V["out_index"], device=dev)
    ci = torch.as_tensor(V["ctx_index"], device=dev)
    n_out, F = int((V["out_index"] >= 0).sum()), int((V["ctx_index"] >= 0).sum())
    Mx = torch.zeros(n_out * F, dtype=torch.float32, device=dev)
    n = ids.shape[0]
    for d in range(1, W + 1):
        wgt = (W - d + 1) / W
        for sa, sb in ((slice(0, n - d), slice(d, n)), (slice(d, n), slice(0, n - d))):  # target a, context b
            a, b, da, db = ids[sa], ids[sb], doc[sa], doc[sb]
            ok = (a >= 0) & (b >= 0) & (da == db) & (a != b)
            ta, cb = oi[a.clamp(min=0)], ci[b.clamp(min=0)]
            ok &= (ta >= 0) & (cb >= 0)
            flat = ta[ok] * F + cb[ok]
            Mx.index_add_(0, flat, torch.full_like(flat, 1, dtype=torch.float32) * wgt)
    return Mx.view(n_out, F)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True, choices=("S1", "S10", "S100"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dims", default="500")
    ap.add_argument("--eigs", default="0.5")
    ap.add_argument("--cds", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--features", type=int, default=20000)
    ap.add_argument("--sub-t", type=float, default=1e-4)
    ap.add_argument("--sub-seed", type=int, default=0)
    ap.add_argument("--mem", type=float, default=0.25)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    dims = [int(x) for x in args.dims.split(",")]
    eigs = [float(x) for x in args.eigs.split(",")]
    dev = torch.device("cpu" if args.cpu else "cuda")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem)
    else:
        torch.set_num_threads(2)
    ids = M.load_slice(args.slice)
    V = M.vocab(ids, args.min_count, args.features)
    sub = M.subsample(ids, V["counts"], args.sub_t, args.sub_seed)
    C = cooc(sub, V, args.window, dev)
    tot = C.sum()
    pw = C.sum(1) / tot
    nc = C.sum(0)
    pc = nc.pow(args.cds) / nc.pow(args.cds).sum()
    nz = float((C > 0).float().mean())
    for r in range(0, C.shape[0], 4096):  # PPMI in place, a block of rows at a time
        blk = C[r : r + 4096]
        pmi = torch.log(blk / tot) - torch.log(pw[r : r + 4096]).unsqueeze(1) - torch.log(pc).unsqueeze(0)
        C[r : r + 4096] = torch.where(blk > 0, pmi.clamp(min=0), torch.zeros_like(blk))
    torch.manual_seed(0)
    q = max(dims) + 16
    U, S, _ = torch.svd_lowrank(C, q=q, niter=6)
    sq = float((C * C).sum())
    rows_zero = int((C.abs().sum(1) == 0).sum())
    (HERE / "tables").mkdir(exist_ok=True)
    out = []
    for d in dims:
        for e in eigs:
            R = U[:, :d] * S[:d].pow(e).unsqueeze(0)
            R = torch.nn.functional.normalize(R, dim=1).cpu().numpy().astype(np.float16)
            R[(C.abs().sum(1) == 0).cpu().numpy()] = 0
            p = HERE / "tables" / f"{args.tag}_d{d}_e{e:g}.npz"
            assert not p.exists(), f"{p} exists: a new version needs a new tag"
            np.savez(p, vocab_ids=V["out_ids"].astype(np.int32), codes=R, counts=V["counts"][V["out_ids"]].astype(np.int64))
            out.append({"table": str(p.relative_to(HERE)), "dim": d, "eig": e, "table_sha256": M.sha16(p),
                        "sq_share_kept": round(float((S[:d] ** 2).sum()) / sq, 4)})
    rec = {"tag": args.tag, "kind": "PPMI-SVD twin", "slice": args.slice, "utc": M.utc(), "cfg": vars(args),
           "outputs": len(V["out_ids"]), "features": int((V["ctx_index"] >= 0).sum()),
           "tokens_after_subsampling": int((sub >= 0).sum()), "nonzero_share": round(nz, 4),
           "rows_without_vector": rows_zero, "tables": out, "device": dev.type, "total_s": round(time.time() - t0, 1),
           "sha256": {"ppmi_svd.py": M.sha16(Path(__file__).resolve()), "mo_tm.py": M.sha16(HERE / "mo_tm.py")}}
    print(json.dumps(rec), flush=True)
    with open(HERE / "results_w1_train.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
