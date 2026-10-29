#!/usr/bin/env python
"""W1-S count twin at scale: ppmi_svd.py's computation with the stream processed in chunks (so S1000 fits), for the
potion-token stream or the C4 word stream (the vocabulary control).

    ppmi_svd2.py --stream potion --slice S10|S100|S300|S1000 --tag NAME [--dims 500 --eigs 0.5 --features 20000]
    ppmi_svd2.py --stream words --slice S1000 --tag NAME [...]
    ppmi_svd2.py --check-against tables/pp_S10_d500_e0.5.npz --stream potion --slice S10 --tag check_S10

Stream and slice: potion S10 / S100 from corpus/c4_S100.ids.npy (mo_tm.load_slice, hash-checked); potion S300 / S1000
from corpus/c4_S1000.ids.npy (c4_ext_slices.json offsets, hash-checked); words S1000 from corpus/c4_S1000.words.npy
(out-of-vocabulary words removed before anything else; vocabulary = c4_words_vocab.json, 63,091 words, no special
tokens). Vocabularies, subsampling, counts, PPMI and SVD exactly as ppmi_svd.py (outputs >= --min-count occurrences,
the --features most frequent tokens as contexts, word2vec's subsampling with t and seed, pairs at distance d <= W in
one document of different types with weight (W - d + 1) / W, both directions, context smoothing 0.75, PPMI, then
torch.svd_lowrank q = max dim + 16, 6 iterations, torch.manual_seed(0); rows U_d S_d^eig, unit length). The only
change: each pair (p, p + d) is counted once while p runs over chunks of --chunk positions (the same set of pairs).
--check-against: the new table against an old one on the same slice, per row cosine after aligning each SVD
component's sign; passes when every row has cosine >= 0.999. Tables: tables/{tag}_d{dim}_e{eig}.npz (vocab_ids, codes,
counts; words tables also hold the word strings). Rows in results_w1_train.jsonl (checks in results_w1_checks.jsonl).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_tm as M  # noqa: E402

CORPUS = HERE / "corpus"


def load(stream: str, name: str):
    if stream == "potion" and name in ("S1", "S10", "S100"):
        return M.load_slice(name), M.V_POTION, M.SPECIAL, None
    man = json.loads((CORPUS / "c4_ext_slices.json").read_text())
    if stream == "potion":
        a = np.ascontiguousarray(np.load(CORPUS / "c4_S1000.ids.npy", mmap_mode="r")[: man["slices"][name]["end_offset"]])
        assert hashlib.sha256(a.astype(np.int32).tobytes()).hexdigest() == man["slices"][name]["sha256_int32"], "slice hash"
        return a, M.V_POTION, M.SPECIAL, None
    assert name == "S1000", "the word control is at S1000 only"
    a = np.load(CORPUS / "c4_S1000.words.npy")
    assert hashlib.sha256(a.astype(np.int32).tobytes()).hexdigest() == man["words"]["sha256_int32"], "word stream hash"
    a = a[a != -2]  # out-of-vocabulary words are removed before anything else
    words = json.loads((CORPUS / "c4_words_vocab.json").read_text())["words"]
    return a, len(words), (), words


def vocab(ids, V, special, min_count, n_features):
    cnt = np.bincount(ids[ids >= 0], minlength=V).astype(np.int64)
    cnt[list(special)] = 0
    order = np.argsort(-cnt, kind="stable")
    out_ids = order[cnt[order] >= min_count]
    ctx_ids = order[:n_features][cnt[order[:n_features]] > 0]
    out_index = np.full(V, -1, np.int64)
    out_index[out_ids] = np.arange(len(out_ids))
    ctx_index = np.full(V, -1, np.int64)
    ctx_index[ctx_ids] = np.arange(len(ctx_ids))
    return {"counts": cnt, "out_ids": out_ids, "ctx_ids": ctx_ids, "out_index": out_index, "ctx_index": ctx_index}


def cooc(sub: np.ndarray, Vd: dict, W: int, dev, chunk: int) -> torch.Tensor:
    oi = torch.as_tensor(Vd["out_index"], device=dev)
    ci = torch.as_tensor(Vd["ctx_index"], device=dev)
    n_out, F = int((Vd["out_index"] >= 0).sum()), int((Vd["ctx_index"] >= 0).sum())
    Mx = torch.zeros(n_out * F, dtype=torch.float32, device=dev)
    doc_all = np.cumsum(sub < 0).astype(np.int64)
    n = len(sub)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        seg = torch.as_tensor(sub[a : min(n, b + W)].astype(np.int64), device=dev)
        dseg = torch.as_tensor(doc_all[a : min(n, b + W)], device=dev)
        for d in range(1, W + 1):
            m = min(b - a, seg.shape[0] - d)
            if m <= 0:
                continue
            wgt = (W - d + 1) / W
            x, y, dx, dy = seg[:m], seg[d : d + m], dseg[:m], dseg[d : d + m]
            for t, c in ((x, y), (y, x)):  # both directions of the pair (p, p + d)
                ok = (t >= 0) & (c >= 0) & (dx == dy) & (t != c)
                tt, cc = oi[t.clamp(min=0)], ci[c.clamp(min=0)]
                ok &= (tt >= 0) & (cc >= 0)
                flat = tt[ok] * F + cc[ok]
                Mx.index_add_(0, flat, torch.full_like(flat, 1, dtype=torch.float32) * wgt)
        del seg, dseg
    return Mx.view(n_out, F)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=("potion", "words"))
    ap.add_argument("--slice", required=True, choices=("S1", "S10", "S100", "S300", "S1000"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dims", default="500")
    ap.add_argument("--eigs", default="0.5")
    ap.add_argument("--cds", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--features", type=int, default=20000)
    ap.add_argument("--sub-t", type=float, default=1e-4)
    ap.add_argument("--sub-seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=50_000_000)
    ap.add_argument("--mem", type=float, default=0.4)
    ap.add_argument("--check-against", default=None)
    args = ap.parse_args()
    t0 = time.time()
    dims = [int(x) for x in args.dims.split(",")]
    eigs = [float(x) for x in args.eigs.split(",")]
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem)
    ids, V, special, words = load(args.stream, args.slice)
    Vd = vocab(ids, V, special, args.min_count, args.features)
    sub = M.subsample(ids, Vd["counts"], args.sub_t, args.sub_seed)
    n_tok = int((ids >= 0).sum())
    del ids
    t1 = time.time()
    C = cooc(sub, Vd, args.window, dev, args.chunk)
    tot = C.sum()
    pw = C.sum(1) / tot
    nc = C.sum(0)
    pc = nc.pow(args.cds) / nc.pow(args.cds).sum()
    nz = float((C > 0).float().mean())
    for r in range(0, C.shape[0], 4096):
        blk = C[r : r + 4096]
        pmi = torch.log(blk / tot) - torch.log(pw[r : r + 4096]).unsqueeze(1) - torch.log(pc).unsqueeze(0)
        C[r : r + 4096] = torch.where(blk > 0, pmi.clamp(min=0), torch.zeros_like(blk))
    torch.manual_seed(0)
    U, S, _ = torch.svd_lowrank(C, q=max(dims) + 16, niter=6)
    zero_rows = (C.abs().sum(1) == 0).cpu().numpy()
    sq = float((C * C).sum())
    build_s = time.time() - t1
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (HERE / "tables").mkdir(exist_ok=True)
    out = []
    for d in dims:
        for e in eigs:
            R = U[:, :d] * S[:d].pow(e).unsqueeze(0)
            R = torch.nn.functional.normalize(R, dim=1).cpu().numpy().astype(np.float16)
            R[zero_rows] = 0
            p = HERE / "tables" / f"{args.tag}_d{d}_e{e:g}.npz"
            assert not p.exists(), f"{p} exists: a new version needs a new tag"
            extra = {"words": np.asarray([words[i] for i in Vd["out_ids"]])} if words is not None else {}
            np.savez(p, vocab_ids=Vd["out_ids"].astype(np.int32), codes=R, counts=Vd["counts"][Vd["out_ids"]].astype(np.int64), **extra)
            out.append({"table": str(p.relative_to(HERE)), "dim": d, "eig": e, "table_sha256": M.sha16(p),
                        "sq_share_kept": round(float((S[:d] ** 2).sum()) / sq, 4)})
    rec = {"tag": args.tag, "kind": "PPMI-SVD (W1-S, chunked)", "stream": args.stream, "slice": args.slice, "utc": M.utc(),
           "cfg": vars(args), "tokens_in_slice": n_tok, "tokens_after_subsampling": int((sub >= 0).sum()),
           "outputs": len(Vd["out_ids"]), "features": int((Vd["ctx_index"] >= 0).sum()), "nonzero_share": round(nz, 4),
           "rows_without_vector": int(zero_rows.sum()), "tables": out, "build_s": round(build_s, 1), "peak_gpu_gib": peak,
           "total_s": round(time.time() - t0, 1),
           "sha256": {"ppmi_svd2.py": M.sha16(Path(__file__).resolve()), "mo_tm.py": M.sha16(HERE / "mo_tm.py")}}
    print(json.dumps(rec), flush=True)
    with open(HERE / "results_w1_train.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if args.check_against:
        old = np.load(HERE / args.check_against)
        new = np.load(HERE / out[0]["table"])
        assert np.array_equal(old["vocab_ids"], new["vocab_ids"]), "different output lists"
        A, Bm = old["codes"].astype(np.float32), new["codes"].astype(np.float32)
        sign = np.sign((A * Bm).sum(0))
        sign[sign == 0] = 1
        cos = (A * (Bm * sign)).sum(1)
        live = np.abs(A).sum(1) > 0
        chk = {"check": "ppmi_svd2.py against ppmi_svd.py on the same slice (row cosine after component sign alignment)",
               "slice": args.slice, "old": args.check_against, "new": out[0]["table"], "rows": int(live.sum()),
               "min_row_cosine": round(float(cos[live].min()), 6), "mean_row_cosine": round(float(cos[live].mean()), 6),
               "passed": bool(cos[live].min() >= 0.999)}
        print(json.dumps(chk))
        with open(HERE / "results_w1_checks.jsonl", "a") as fh:
            fh.write(json.dumps(chk) + "\n")


if __name__ == "__main__":
    main()
