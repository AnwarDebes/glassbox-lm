#!/usr/bin/env python
"""W2 H-B step 0, part 2 (EXPLORATORY, VAL only; rules in LOG.md before any run): the counting twin with the
question -> answer pair signal of part 1.

    python -I pair_table.py --selftest     # the pair counting, the shift, PPMI on a block and the SVD rows, on
                                           # synthetic data on the CPU; reads no corpus file
    python -I pair_table.py                # writes w2/tables/{J,Jshift,P}_d500_e0.5.npz (refuses to overwrite)

1. The C4 block: w1/ppmi_svd2.py's own functions (imported, unchanged) on the same C4 word stream of S1000 with the
   same settings as the W1 C4-word table ppw_S1000 (63,091 output words, the 20,000 most frequent as contexts,
   subsampling t 1e-4 seed 0, window 5 with weights (W - d + 1) / W in both directions, context smoothing 0.75, PPMI).
   Check: its SVD alone (q 516, 6 iterations, torch.manual_seed(0); rows U_500 S_500^0.5, unit length) must give the
   rows of w1/tables/ppw_S1000_d500_e0.5.npz again (every row cosine >= 0.999 after aligning each component's sign,
   the W1 check), else the run stops and writes no table.
2. The pair block (the same rows and the same 20,000 context words): for each pair of a FAQ-like document of
   w2/corpus/pairs_S1000.jsonl.gz (faq_like true; 276,661 pairs), Q = the distinct vocabulary words of the question
   and A = those of the answer (glove_feats.glove_tokens, the tokenizer of the word stream); every (w in Q, v in A)
   with w != v adds 1 at (row w, context v) when v is a context word, and 1 at (row v, context w) when w is a context
   word (both directions, once per pair). Then PPMI as in step 1.
   Control 'shift': the same questions, each with the answer of the NEXT pair of its own document (cyclically; every
   FAQ-like document holds at least 3 pairs): the document's topic stays, the matching answer goes.
3. Tables: J = the SVD of [C4 block | lambda * pair block], lambda = ||C4 block||_F / ||pair block||_F (equal
   energy); Jshift = the same with the shift block (its own lambda); P = the SVD of the pair block alone (diagnostic;
   a word without any pair count gets a zero row, which w1/wordtab_feats.py drops as out of vocabulary). SVD and rows
   as step 1. Each table holds vocab_ids, codes (float16), counts and words like the W1 word tables, so
   w1/wordtab_feats.py reads it unchanged.
One row per run in w2/results_w2_tables.jsonl. GPU through gpu_run.sh shared; memory fraction --mem.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/w2
EMB = HERE.parent
W1 = EMB / "w1"
sys.path.insert(0, str(W1))
sys.path.insert(0, str(EMB))
import mo_tm as M  # noqa: E402
import ppmi_svd2 as P  # noqa: E402
import glove_feats as GF  # noqa: E402

PAIRS = HERE / "corpus" / "pairs_S1000.jsonl.gz"
PAIRS_SHA = "11f98abae10e5a3a"
PPW = W1 / "tables" / "ppw_S1000_d500_e0.5.npz"
PPW_SHA = "e1771d9eaf4a07f7"
N_FAQ = 276_661
# = ppw_S1000's settings (w1/results_w1_train.jsonl, tag ppw_S1000)
CFG = {"dims": 500, "eig": 0.5, "cds": 0.75, "window": 5, "min_count": 10, "features": 20000, "sub_t": 1e-4,
       "sub_seed": 0, "chunk": 50_000_000}
ROWS = 4096


def ppmi_(A: torch.Tensor, cds: float) -> None:
    """In place: ppmi_svd2.py's PPMI with context smoothing cds, on a matrix or a column block of one (a view)."""
    tot = A.sum()
    pw = A.sum(1) / tot
    nc = A.sum(0)
    pc = nc.pow(cds) / nc.pow(cds).sum()
    for r in range(0, A.shape[0], ROWS):
        blk = A[r : r + ROWS]
        pmi = torch.log(blk / tot) - torch.log(pw[r : r + ROWS]).unsqueeze(1) - torch.log(pc).unsqueeze(0)
        A[r : r + ROWS] = torch.where(blk > 0, pmi.clamp(min=0), torch.zeros_like(blk))


def sq_norm(A: torch.Tensor) -> float:
    return float(sum(float((A[r : r + ROWS] * A[r : r + ROWS]).sum()) for r in range(0, A.shape[0], ROWS)))


def zero_rows_of(A: torch.Tensor) -> np.ndarray:
    return np.concatenate([(A[r : r + ROWS].abs().sum(1) == 0).cpu().numpy() for r in range(0, A.shape[0], ROWS)])


def svd_rows(A: torch.Tensor, d: int, e: float):
    """ppmi_svd2.py's rows: torch.svd_lowrank q = d + 16, 6 iterations, seed 0; U_d S_d^e, unit length; zero rows 0."""
    torch.manual_seed(0)
    U, S, _ = torch.svd_lowrank(A, q=d + 16, niter=6)
    zr = zero_rows_of(A)
    R = U[:, :d] * S[:d].pow(e).unsqueeze(0)
    R = torch.nn.functional.normalize(R, dim=1).cpu().numpy().astype(np.float16)
    R[zr] = 0
    kept = round(float((S[:d] ** 2).sum()) / sq_norm(A), 4)
    del U, S
    return R, zr, kept


def row_cosines(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    A, B = old.astype(np.float32), new.astype(np.float32)
    sign = np.sign((A * B).sum(0))
    sign[sign == 0] = 1
    return (A * (B * sign)).sum(1)


def read_pairs(word_index: dict) -> list:
    """The FAQ-like pairs in file order: (doc, Q, A), Q and A the sorted distinct vocabulary ids of each side."""
    out = []
    with gzip.open(PAIRS, "rt") as f:
        for line in f:
            r = json.loads(line)
            if not r["faq_like"]:
                continue
            q = sorted({word_index[w] for w in GF.glove_tokens(r["q"]) if w in word_index})
            a = sorted({word_index[w] for w in GF.glove_tokens(r["a"]) if w in word_index})
            out.append((int(r["doc"]), q, a))
    return out


def shifted(pairs: list) -> list:
    """Each question with the answer of the next pair of its own document (cyclically)."""
    by_doc = defaultdict(list)
    for i, (d, _, _) in enumerate(pairs):
        by_doc[d].append(i)
    out = list(pairs)
    for d, idx in by_doc.items():
        assert len(idx) >= 3, f"document {d} has {len(idx)} pairs; FAQ-like documents have at least 3"
        for j, i in enumerate(idx):
            out[i] = (d, pairs[i][1], pairs[idx[(j + 1) % len(idx)]][2])
    return out


def add_pairs_(Jt: torch.Tensor, col0: int, pairs: list, out_index: np.ndarray, ctx_index: np.ndarray,
               chunk: int = 20_000_000) -> dict:
    """Adds the cross-pair counts at Jt[row, col0 + context] (both directions, once per pair, w != v)."""
    dev = Jt.device
    rows, cols, n, total, used = [], [], 0, 0, 0

    def flush():
        if rows:
            r = torch.as_tensor(np.concatenate(rows), device=dev)
            c = torch.as_tensor(np.concatenate(cols), device=dev) + col0
            Jt.index_put_((r, c), torch.ones(len(r), dtype=Jt.dtype, device=dev), accumulate=True)
            rows.clear()
            cols.clear()

    for _, q, a in pairs:
        if not q or not a:
            continue
        used += 1
        W, Vv = np.meshgrid(np.asarray(q, np.int64), np.asarray(a, np.int64), indexing="ij")
        W, Vv = W.ravel(), Vv.ravel()
        keep = W != Vv
        W, Vv = W[keep], Vv[keep]
        for t, c in ((W, Vv), (Vv, W)):  # (row w, context v) and (row v, context w)
            ro, co = out_index[t], ctx_index[c]
            m = (ro >= 0) & (co >= 0)
            rows.append(ro[m])
            cols.append(co[m])
            n += int(m.sum())
            total += int(m.sum())
        if n >= chunk:
            flush()
            n = 0
    flush()
    return {"pairs_with_words_on_both_sides": used, "cross_entries": total}


def selftest() -> None:
    rng = np.random.default_rng(0)
    V, F = 60, 24
    out_index = np.arange(V)
    out_index[[3, 7]] = -1  # two words without a row
    ctx_index = np.full(V, -1)
    ctx_index[:F] = np.arange(F)
    pairs = []
    for d in range(5):
        for _ in range(3 + d % 2):
            pairs.append((d, sorted(set(rng.integers(0, V, 6).tolist())), sorted(set(rng.integers(0, V, 9).tolist()))))
    # counting against a plain dictionary
    Jt = torch.zeros(V, 2 * F)
    info = add_pairs_(Jt, F, pairs, out_index, ctx_index, chunk=7)
    ref = defaultdict(float)
    for _, q, a in pairs:
        for w in q:
            for v in a:
                if w == v:
                    continue
                if out_index[w] >= 0 and ctx_index[v] >= 0:
                    ref[(out_index[w], ctx_index[v])] += 1
                if out_index[v] >= 0 and ctx_index[w] >= 0:
                    ref[(out_index[v], ctx_index[w])] += 1
    R = np.zeros((V, F))
    for (i, j), x in ref.items():
        R[i, j] = x
    assert np.array_equal(Jt[:, F:].numpy(), R), "pair counts differ from the dictionary count"
    assert float(Jt[:, :F].abs().sum()) == 0.0, "counts leaked into the left block"
    assert info["cross_entries"] == int(R.sum())
    # the shift: same questions, answers from the next pair of the same document, never the own answer
    sh = shifted(pairs)
    for i, (d, q, a) in enumerate(sh):
        assert d == pairs[i][0] and q == pairs[i][1]
        same_doc = [j for j, p in enumerate(pairs) if p[0] == d]
        k = same_doc.index(i)
        assert a == pairs[same_doc[(k + 1) % len(same_doc)]][2]
    # PPMI on a column-block view = PPMI on a contiguous copy = the formula in numpy
    Jt[:, :F] = torch.as_tensor(rng.integers(0, 4, (V, F)).astype(np.float32))
    right = Jt[:, F:].clone()
    ppmi_(Jt[:, F:], 0.75)
    ppmi_(right, 0.75)
    assert torch.allclose(Jt[:, F:], right)
    C = R.copy()
    tot, pw, nc = C.sum(), C.sum(1) / C.sum(), C.sum(0)
    pc = nc ** 0.75 / (nc ** 0.75).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = np.log(C / tot) - np.log(pw)[:, None] - np.log(pc)[None, :]
    pm = np.where(C > 0, np.maximum(pm, 0), 0)
    assert np.allclose(right.numpy(), pm, atol=1e-5), "PPMI differs from the formula"
    assert torch.isfinite(Jt).all()
    # SVD rows: a scaled block gives the same unit rows (the scale cancels), zero rows stay zero
    blk = Jt[:, F:].clone()
    r1, z1, _ = svd_rows(blk, 4, 0.5)
    r2, z2, _ = svd_rows(blk * 3.7, 4, 0.5)
    assert np.array_equal(z1, z2) and np.allclose(r1.astype(np.float32), r2.astype(np.float32), atol=2e-3)
    assert (r1[z1] == 0).all() and z1[3] and z1[7]
    print("selftest passed:", json.dumps(info))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mem", type=float, default=0.4)
    ap.add_argument("--min-free-gib", type=float, default=24.0)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    t0 = time.time()
    out_dir = HERE / "tables"
    outs = {k: out_dir / f"{k}_d{CFG['dims']}_e{CFG['eig']:g}.npz" for k in ("J", "Jshift", "P")}
    for p in outs.values():
        assert not p.exists(), f"{p} exists: a new version needs a new name"
    assert M.sha16(PAIRS) == PAIRS_SHA, "pairs file changed"
    assert M.sha16(PPW) == PPW_SHA, "ppw_S1000 table changed"
    dev = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < args.min_free_gib:  # exit 5: the chain waits and tries again
        print(f"only {free / 2**30:.1f} GiB free on the card (need {args.min_free_gib})", flush=True)
        sys.exit(5)
    torch.cuda.set_per_process_memory_fraction(args.mem)
    # 1. the C4 block, as ppw_S1000
    ids, V, special, words = P.load("words", "S1000")
    Vd = P.vocab(ids, V, special, CFG["min_count"], CFG["features"])
    sub = M.subsample(ids, Vd["counts"], CFG["sub_t"], CFG["sub_seed"])
    del ids
    C = P.cooc(sub, Vd, CFG["window"], dev, CFG["chunk"])
    del sub
    ppmi_(C, CFG["cds"])
    R, zr, kept = svd_rows(C, CFG["dims"], CFG["eig"])
    old = np.load(PPW)
    assert np.array_equal(old["vocab_ids"], Vd["out_ids"].astype(np.int32)), "a different output list than ppw_S1000"
    live = np.abs(old["codes"].astype(np.float32)).sum(1) > 0
    cos = row_cosines(old["codes"], R)
    check = {"rows": int(live.sum()), "min_row_cosine": round(float(cos[live].min()), 6),
             "mean_row_cosine": round(float(cos[live].mean()), 6), "passed": bool(cos[live].min() >= 0.999),
             "sq_share_kept": kept}
    print(json.dumps({"c4_block_check": check}), flush=True)
    rec = {"step": "W2 H-B step 0 part 2 tables (exploratory, val only)", "utc_start": M.utc(), "cfg": CFG,
           "card_gib": round(total / 2**30, 1), "free_gib_at_start": round(free / 2**30, 1), "c4_block_check": check}
    if not check["passed"]:
        rec["stopped"] = "the C4 block does not reproduce ppw_S1000 (rule: every row cosine >= 0.999); no table"
        with open(HERE / "results_w2_tables.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        sys.exit(4)
    n_out, F = C.shape
    Ccpu = C.cpu()
    del C, R
    torch.cuda.empty_cache()
    Jt = torch.zeros((n_out, 2 * F), dtype=torch.float32, device=dev)
    for r in range(0, n_out, ROWS):
        Jt[r : r + ROWS, :F] = Ccpu[r : r + ROWS].to(dev)
    del Ccpu
    c4_sq = sq_norm(Jt[:, :F])
    # 2. the pairs
    word_index = {w: i for i, w in enumerate(words)}
    t1 = time.time()
    pairs = read_pairs(word_index)
    assert len(pairs) == N_FAQ, f"{len(pairs)} FAQ-like pairs, expected {N_FAQ}"
    rec["pairs"] = {"faq_like_pairs": len(pairs), "read_s": round(time.time() - t1, 1),
                    "mean_distinct_words_question": round(float(np.mean([len(q) for _, q, _ in pairs])), 2),
                    "mean_distinct_words_answer": round(float(np.mean([len(a) for _, _, a in pairs])), 2)}
    variants = {"J": pairs, "Jshift": shifted(pairs)}
    oi, ci = Vd["out_index"], Vd["ctx_index"]
    extra = {"words": np.asarray([words[i] for i in Vd["out_ids"]])}
    out_dir.mkdir(exist_ok=True)
    rec["tables"] = {}
    for name, prs in variants.items():
        Jt[:, F:].zero_()
        info = add_pairs_(Jt, F, prs, oi, ci)
        blk = Jt[:, F:]
        info["rows_with_pair_counts"] = int((~zero_rows_of(blk)).sum())
        info["nonzero_share"] = round(sum(float((blk[r : r + ROWS] > 0).sum()) for r in range(0, n_out, ROWS)) / (n_out * F), 5)
        ppmi_(blk, CFG["cds"])
        lam = (c4_sq / sq_norm(blk)) ** 0.5
        blk.mul_(lam)
        info["lambda"] = round(lam, 6)
        todo = [(name, Jt)] + ([("P", blk)] if name == "J" else [])
        for tname, A in todo:
            R, zr, kept = svd_rows(A, CFG["dims"], CFG["eig"])
            np.savez(outs[tname], vocab_ids=Vd["out_ids"].astype(np.int32), codes=R,
                     counts=Vd["counts"][Vd["out_ids"]].astype(np.int64), **extra)
            rec["tables"][tname] = {"table": str(outs[tname].relative_to(EMB)), "table_sha256": M.sha16(outs[tname]),
                                    "rows_without_vector": int(zr.sum()), "sq_share_kept": kept,
                                    **({"pair_block": info} if tname != "P" else {"from": "the J pair block alone"})}
            print(json.dumps({tname: rec["tables"][tname]}), flush=True)
            del R
    rec["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    rec["total_s"] = round(time.time() - t0, 1)
    rec["utc_end"] = M.utc()
    rec["sha256"] = {"pair_table.py": M.sha16(Path(__file__).resolve()), "ppmi_svd2.py": M.sha16(W1 / "ppmi_svd2.py"),
                     "mo_tm.py": M.sha16(W1 / "mo_tm.py"), "glove_feats.py": M.sha16(EMB / "glove_feats.py"),
                     "pairs_S1000.jsonl.gz": PAIRS_SHA, "ppw_S1000_d500_e0.5.npz": PPW_SHA}
    with open(HERE / "results_w2_tables.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "cfg"}))


if __name__ == "__main__":
    main()
