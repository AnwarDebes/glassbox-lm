#!/usr/bin/env python
"""Step 3d, the fit (tu/PLAN_3d.md "Fit"): 3a's least squares with one row per listed word pair added. The solver is
tu/a3_fit.py's fit(), imported unchanged.

    python -I -B d3_fit.py --selftest      # synthetic tests on the CPU (no files)
    python -I -B d3_fit.py --check-a3B     # the GPU job's first step: the same fit with NO pairs must reproduce 3a's
                                           # held-out cosines, lambda* and a3B's rows (tu/d3/check_P0.json; exit 3 if not)
    python -I -B d3_fit.py --check-a3B --device cpu   # the same check on the CPU, before the GPU job, to test the
                                           # gate's tolerances across devices (tu/d3/check_P0_cpu.json)
    python -I -B d3_fit.py                 # the GPU job: tu/d3/tables/d3J_d768.npz and tu/d3/fit_meta_d3J.json

Pairs (pair_list): the rows of tu/d3/pair_counts_c5.npz (d3_pairs.py) whose two tokens are both word forms (TOK's
[a-z0-9]+, n't or 'x; a pair with a punctuation token is not listed) and whose count in 3a's 1,999,932 training
sentences is at least MIN_COUNT = 20, in that file's order. C4 data only: no intent set plays any part.
Tokens of a sentence (tokens_of): its live words, repeats counted (exactly 3a's rows), then its listed adjacent pairs
(d3_pairs.pair_keys), repeats counted, as rows V + pair index. A[n, r] = count of r in n / all tokens of n: the
sentence vector is the mean of its word rows and pair rows, each pair one token.
Start rows W0 (step A): ppw_S1000's rows for the words; a pair's row is the mean of its two words' rows.
Then a3_fit.fit() as in 3a: step A maps W0 to W_A (one 500 x 768 map); step B solves
(A^T A + lambda I) W = A^T T + lambda W_A by CG for lambda 10, 1, 0.1, 0.01; lambda* = the highest held-out cosine
(ties to the larger lambda). With no pairs this is 3a's fit exactly.
Output: unit rows in float16: "codes" (the 63,091 words, ppw_S1000's format: words, counts and vocab_ids copied),
"pair_codes", "pair_w1", "pair_w2" (word row ids) and "pair_counts". Only lambda*'s table is written; the meta, written
last, keeps every lambda's held-out cosine, CG steps and residual, and "converged" per lambda. a3_fit.fit stops
silently after 400 CG steps, so: exit 4 (the chain stops) when lambda*'s CG did not reach the tolerance 1e-4; another
lambda's miss is a caveat in the meta. The check also needs lambda* converged. Refuses to overwrite.
Tests only: with $D3_SMOKE_DIR set, every input and output is under that folder and the device is the CPU
(tests/test_d3.py's end-to-end smoke on tiny data); d3_chain.sh refuses any D3_* variable in the real run.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
sys.path.insert(0, str(HERE))
import a3_fit as AF
import d3_pairs as DP

A3 = HERE / "a3"
D3 = HERE / "d3"
TABLE = AF.TABLE
COUNTS = D3 / "pair_counts_c5.npz"
MIN_COUNT = 20
WORD_FORM = re.compile(r"[a-z0-9]+|n't|'[a-z]+")
# the check against 3a (--check-a3B): held-out cosines within CHECK_COS of tu/a3/fit_meta.json (stored to 5 dp), the
# same lambda*, and every live row's cosine to a3B's row at least CHECK_ROW
CHECK_COS, CHECK_ROW = 1e-4, 0.999


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def pair_list(words: list[str], counts_npz: Path = COUNTS, min_count: int = MIN_COUNT):
    """(w1, w2, counts) of the listed pairs, in the counts file's order."""
    z = np.load(counts_npz)
    assert int(z["V"]) == len(words), "the counts file was made for another word list"
    form = np.array([bool(WORD_FORM.fullmatch(w)) for w in words])
    w1, w2, c = z["w1"].astype(np.int64), z["w2"].astype(np.int64), z["counts"].astype(np.int64)
    keep = (c >= min_count) & form[w1] & form[w2]
    return w1[keep], w2[keep], c[keep]


def tokens_of(text: str, tokenize, row: dict[str, int], pair_row: dict[int, int], V: int) -> np.ndarray:
    toks = tokenize(text)
    w = [row[t] for t in toks if t in row]
    p = [V + pair_row[k] for k in DP.pair_keys(toks, row, V) if k in pair_row] if pair_row else []
    return np.asarray(w + p, dtype=np.int64)


def start_rows(W0_words: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return np.concatenate([W0_words, 0.5 * (W0_words[w1] + W0_words[w2])]).astype(np.float32)


def selftest(tmp: Path) -> None:
    import torch

    rng = np.random.default_rng(1)
    V, d0, D, N, NH = 120, 12, 10, 30000, 3000
    words = [f"w{i}" for i in range(V - 2)] + [",", "."]
    row = {w: i for i, w in enumerate(words)}
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()

    def sents(n):
        return [" ".join(words[j] for j in rng.choice(V, size=rng.integers(3, 12), p=p)) for _ in range(n)]

    tr, ho = sents(N), sents(NH)
    tok = str.split
    # the pair list from counts, as d3_pairs writes them
    keys_u, cnt, _ = DP.count_pairs(tr, tok, row, V, min_keep=1)
    np.savez(tmp / "counts.npz", w1=(keys_u // V).astype(np.int32), w2=(keys_u % V).astype(np.int32), counts=cnt,
             V=np.asarray(V))
    w1, w2, c = pair_list(words, tmp / "counts.npz", min_count=30)
    assert len(w1) > 50 and (c >= 30).all() and list(c) == sorted(c, reverse=True)
    assert not any(words[a] in ",." or words[b] in ",." for a, b in zip(w1, w2)), "punctuation pairs must not be listed"
    P = len(w1)
    pair_row = {int(a) * V + int(b): j for j, (a, b) in enumerate(zip(w1, w2))}
    # tokens: with no pairs, exactly 3a's rows (words only, repeats kept, in text order)
    t3a = [np.array([row[t] for t in tok(s) if t in row], dtype=np.int64) for s in tr[:200]]
    t0 = [tokens_of(s, tok, row, {}, V) for s in tr[:200]]
    assert all(np.array_equal(a, b) for a, b in zip(t3a, t0)), "no-pair tokens differ from 3a's rows"
    j = int(np.flatnonzero(w1 != w2)[0])
    a, b = int(w1[j]), int(w2[j])
    got = tokens_of(f"{words[a]} {words[b]} {words[a]} {words[b]}", tok, row, pair_row, V)
    assert list(got[:4]) == [a, b, a, b] and int((got == V + j).sum()) == 2 and (got[4:] >= V).all(), got
    # start rows: a pair's row is the mean of its two words' rows
    W0w = rng.normal(size=(V, d0)).astype(np.float32)
    W0 = start_rows(W0w, w1, w2)
    assert W0.shape == (V + P, d0) and np.allclose(W0[V + 3], (W0w[w1[3]] + W0w[w2[3]]) / 2)
    # recovery: plant word AND pair rows; targets = the mean over all tokens; a3_fit.fit must recover the seen rows
    W_true = rng.normal(size=(V + P, D)).astype(np.float32)
    rows_tr = [tokens_of(x, tok, row, pair_row, V) for x in tr]
    rows_ho = [tokens_of(x, tok, row, pair_row, V) for x in ho]
    pool = lambda rows: np.stack([W_true[r].mean(0) for r in rows])
    T_tr, T_ho = pool(rows_tr).astype(np.float16), pool(rows_ho).astype(np.float16)
    torch.set_num_threads(1)
    _, sols, _, res = AF.fit(rows_tr, T_tr, rows_ho, T_ho, W0, "cpu", lambdas=[1e-6], tol=1e-9, maxit=4000,
                              log=lambda s: None)
    m = T_tr.astype(np.float32).mean(0)
    seen = np.zeros(V + P, dtype=bool)
    for r in rows_tr:
        seen[r] = True
    W = sols[1e-6]
    err_w = np.linalg.norm(W[:V][seen[:V]] - (W_true[:V] - m)[seen[:V]]) / np.linalg.norm((W_true[:V] - m)[seen[:V]])
    err_p = np.linalg.norm(W[V:] - (W_true[V:] - m)) / np.linalg.norm(W_true[V:] - m)
    # negative check: the same targets fit WITHOUT pair rows cannot recover the word rows (pairs carry signal)
    rows_np = [r[r < V] for r in rows_tr]
    rows_hp = [r[r < V] for r in rows_ho]
    _, sols_np, _, res_np = AF.fit(rows_np, T_tr, rows_hp, T_ho, W0w, "cpu", lambdas=[1e-6], tol=1e-9, maxit=4000,
                                   log=lambda s: None)
    err_np = np.linalg.norm(sols_np[1e-6][seen[:V]] - (W_true[:V] - m)[seen[:V]]) / np.linalg.norm((W_true[:V] - m)[seen[:V]])
    out = {"pairs": P, "recover_words_rel_err": float(err_w), "recover_pairs_rel_err": float(err_p),
           "no_pair_fit_words_rel_err": float(err_np), "heldout_cos": res["heldout_cos"]["1e-06"],
           "heldout_cos_no_pairs": res_np["heldout_cos"]["1e-06"]}
    print(json.dumps(out))
    assert err_w < 1e-3 and err_p < 1e-3, "SELFTEST FAILED: planted rows not recovered"
    assert err_np > 0.05 and out["heldout_cos_no_pairs"] < out["heldout_cos"], "SELFTEST FAILED: the negative check"
    print("SELFTEST PASSED")



def io_paths() -> dict[str, Path]:
    """The real inputs and outputs, or (tests only) everything under $D3_SMOKE_DIR, on the CPU."""
    import os

    sd = os.environ.get("D3_SMOKE_DIR")
    if not sd:
        return {"table": TABLE, "counts": COUNTS, "a3": A3, "a3B": A3 / "tables" / "a3B_d768.npz",
                "fit_meta_3a": A3 / "fit_meta.json", "out": D3, "smoke": False}
    d = Path(sd)
    return {"table": d / "table.npz", "counts": d / "counts.npz", "a3": d, "a3B": d / "a3B_d768.npz",
            "fit_meta_3a": d / "fit_meta.json", "out": d / "out", "smoke": True}


def converged(res: dict, tol: float = AF.TOL) -> dict[str, bool]:
    """Per lambda: did CG reach the tolerance (a3_fit.fit stops silently after its step limit)?"""
    return {lam: bool(v["rel_residual"] <= tol) for lam, v in res["cg"].items()}


def load_rows(words_row: dict[str, int], pair_row: dict[int, int], V: int, a3: Path = A3):
    sys.path.insert(0, str(EMB))
    import glove_feats as GF

    def rows_of(sp):
        texts = (a3 / f"c4_sent_{sp}_f.txt").read_text(encoding="utf-8").split("\n")[:-1]
        rows = [tokens_of(s, GF.glove_tokens, words_row, pair_row, V) for s in texts]
        assert min(int((r < V).sum()) for r in rows) >= 3
        return rows

    return rows_of("train"), rows_of("held")


def main() -> None:
    if sys.argv[1:] == ["--selftest"]:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            selftest(Path(td))
        return
    argv = sys.argv[1:]
    check = argv[:1] == ["--check-a3B"]
    P_ = io_paths()
    dev = "cpu" if P_["smoke"] or (check and argv[1:] == ["--device", "cpu"]) else "cuda"
    assert argv in ([], ["--check-a3B"], ["--check-a3B", "--device", "cpu"]), argv
    if dev == "cpu":
        import torch

        torch.set_num_threads(2)
    t0 = time.time()
    a3, outd = P_["a3"], P_["out"]
    (outd / "tables").mkdir(parents=True, exist_ok=True)
    out = outd / "tables" / "d3J_d768.npz"
    cpu_check = check and not P_["smoke"] and dev == "cpu"
    meta_p = outd / (("check_P0_cpu.json" if cpu_check else "check_P0.json") if check else "fit_meta_d3J.json")
    for f in ([meta_p] if check else [meta_p, out]):
        assert not f.exists(), f"{f} exists"
    z = np.load(P_["table"])
    W0w = z["codes"].astype(np.float32)
    dead = np.abs(W0w).sum(1) == 0
    words = [str(w) for w in z["words"]]
    V = len(words)
    row = {w: i for i, w in enumerate(words) if not dead[i]}
    if check:
        w1 = w2 = c = np.zeros(0, np.int64)
    else:
        w1, w2, c = pair_list(words, P_["counts"])
    pair_row = {int(a) * V + int(b): j for j, (a, b) in enumerate(zip(w1, w2))}
    rows_tr, rows_ho = load_rows(row, pair_row, V, a3)
    T_tr = np.load(a3 / "T_train.f16.npy")
    T_ho = np.load(a3 / "T_held.f16.npy")
    assert len(T_tr) == len(rows_tr) and len(T_ho) == len(rows_ho)
    n_pair_tok = int(sum(int((r >= V).sum()) for r in rows_tr))
    print(f"{len(rows_tr):,} training and {len(rows_ho):,} held-out sentences, {len(w1):,} pairs, {n_pair_tok:,} pair "
          f"tokens in training ({time.time() - t0:.0f}s)", flush=True)
    W0 = start_rows(W0w, w1, w2)
    _, sols, _, res = AF.fit(rows_tr, T_tr, rows_ho, T_ho, W0, dev, log=lambda s: print(s, flush=True))
    conv = converged(res)
    res.update({"device": dev, "smoke": P_["smoke"], "converged": conv, "word_rows": V, "pair_rows": len(w1),
                "note_words_field": "a3_fit's 'words' counts every row, words and pairs"})
    dead_all = np.concatenate([dead, np.zeros(len(w1), dtype=bool)])
    U = AF.unit_rows(sols[res["lambda_star"]], dead_all)
    res.update({"pairs": len(w1), "pair_tokens_train": n_pair_tok, "min_count": MIN_COUNT, "dead_words": int(dead.sum()),
                "train": len(rows_tr), "held": len(rows_ho),
                "sha256": {"d3_fit.py": sha16(Path(__file__).resolve()), "a3_fit.py": sha16(HERE / "a3_fit.py"),
                           "d3_pairs.py": sha16(HERE / "d3_pairs.py"), "table": sha16(P_["table"]),
                           "counts": sha16(P_["counts"]) if not check else None,
                           "T_train": sha16(a3 / "T_train.f16.npy"), "T_held": sha16(a3 / "T_held.f16.npy")}})
    if check:
        ref = json.loads(P_["fit_meta_3a"].read_text())
        za = np.load(P_["a3B"])
        Ua = za["codes"].astype(np.float32)
        Uc = U.astype(np.float32)
        live = ~dead
        rc = (Ua[live] * Uc[live]).sum(1)
        dcos = {k: abs(res["heldout_cos"][k] - ref["heldout_cos"][k]) for k in ref["heldout_cos"]}
        ok = (max(dcos.values()) <= CHECK_COS and res["lambda_star"] == ref["lambda_star"] and float(rc.min()) >= CHECK_ROW
              and np.array_equal(za["words"], z["words"]) and conv[str(res["lambda_star"])])
        res["check"] = {"heldout_cos_abs_diff": dcos, "lambda_star_3a": ref["lambda_star"],
                        "row_cos_min": round(float(rc.min()), 6), "row_cos_mean": round(float(rc.mean()), 8),
                        "max_abs_diff": float(np.abs(Ua - Uc).max()), "tol": {"cos": CHECK_COS, "row": CHECK_ROW},
                        "a3B_sha256": sha16(P_["a3B"]), "pass": bool(ok)}
    else:
        keep = {k: z[k] for k in ("words", "counts", "vocab_ids")}
        np.savez(out, codes=U[:V], pair_codes=U[V:], pair_w1=w1.astype(np.int32), pair_w2=w2.astype(np.int32),
                 pair_counts=c, **keep)
        res["sha256"]["d3J_d768.npz"] = sha16(out)
    res.update({"total_seconds": round(time.time() - t0, 1),
                "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    meta_p.write_text(json.dumps(res, indent=1))  # written last: the chain skips this step only when the meta exists
    print(json.dumps({k: v for k, v in res.items() if k != "sha256"}))
    if check and not res["check"]["pass"]:
        sys.exit(3)
    if not conv[str(res["lambda_star"])]:
        sys.exit(4)  # lambda*'s CG did not reach the tolerance: the chain stops


if __name__ == "__main__":
    main()
