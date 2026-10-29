#!/usr/bin/env python
"""Step 3d, the MW twin (tu/PLAN_3d.md "Arms"): the same pair list as d3J, but each pair row is the teacher's embedding
of the pair's own string, the way Model2Vec embeds a token on its own; the word rows are a3B's, unchanged.

    python -I -B d3_mw.py --selftest   # CPU: the strings and the row construction on synthetic vectors (no model)
    python -I -B d3_mw.py              # the GPU job: tu/d3/tables/d3MW_d768.npz, then tu/d3/mw_meta.json (last)

Pairs: d3_fit.pair_list (C4 counts only). Pair string (pair_strings): the two tokens joined by one space ("turn off"),
except that a token "n't" or "'x" joins its left neighbour with no space, as in the written text ("do n't" -> "don't",
"it 's" -> "it's"; 1,483 of the 152,882 pairs).
Teacher: tu/a3_teacher.py's verify_files, load and encode, unchanged (bge-base-en-v1.5 at a5beb1e3; the [CLS] vector
of the last layer, L2-normalised; fp16 on the GPU). Before the full encoding, the first 64 pair strings are encoded on
the CPU in fp32 and on the GPU in fp16; the job stops unless they agree (min cosine >= 0.99).
Row = unit(e - mu), where mu is the mean of the pair strings' own embeddings (Model2Vec's PCA step also centres on its
vocabulary's own mean). The pre-review draft centred on 3a's C4 sentence mean, which left a shared offset (the mean
pairwise cosine of 400 such rows was 0.114, against -0.002 when centred on their own mean; review_3d_2026-10-23).
Output in d3_fit.py's format. Refuses to overwrite.
Tests only: with $D3_SMOKE_DIR set, the inputs and outputs are under that folder (a3B_d768.npz, table.npz,
counts.npz; out/) and both encodings run on the CPU in fp32.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
sys.path.insert(0, str(HERE))
import d3_fit as DF

A3, D3 = DF.A3, DF.D3
A3B = A3 / "tables" / "a3B_d768.npz"


def pair_strings(words: list[str], w1: np.ndarray, w2: np.ndarray) -> list[str]:
    out = []
    for a, b in zip(w1, w2):
        x, y = words[int(a)], words[int(b)]
        out.append(x + ("" if y == "n't" or y.startswith("'") else " ") + y)
    return out


def mw_rows(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = E.astype(np.float32)
    mu = X.mean(0)
    X = X - mu[None, :]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    assert (n > 0).all(), "a pair vector equals the mean"
    return (X / n).astype(np.float16), mu


def selftest() -> None:
    rng = np.random.default_rng(2)
    words = ["turn", "off", "on", "do", "n't", "it", "'s"]
    got = pair_strings(words, np.array([0, 0, 3, 5]), np.array([1, 2, 4, 6]))
    assert got == ["turn off", "turn on", "don't", "it's"], got
    E = (rng.normal(size=(50, 8)) + 1.5).astype(np.float16)  # a shared offset, as the teacher's vectors have
    R, mu = mw_rows(E)
    R = R.astype(np.float32)
    assert R.shape == (50, 8) and np.allclose(np.linalg.norm(R, axis=1), 1, atol=2e-3)
    assert np.allclose(mu, E.astype(np.float32).mean(0), atol=1e-5)
    G = R @ R.T
    mean_cos = (G.sum() - 50) / (50 * 49)
    U = E.astype(np.float32) / np.linalg.norm(E.astype(np.float32), axis=1, keepdims=True)
    raw_cos = ((U @ U.T).sum() - 50) / (50 * 49)
    assert abs(mean_cos) < 0.05 and raw_cos > 0.5, (mean_cos, raw_cos)  # negative check: uncentred rows share a direction
    try:
        mw_rows(np.ones((3, 4), np.float16))
        raise SystemExit("SELFTEST FAILED: equal rows were accepted")
    except AssertionError:
        pass
    print("SELFTEST PASSED")


def main() -> None:
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        return
    assert sys.argv[1:] == [], sys.argv
    import torch

    import a3_teacher as AT

    sd = os.environ.get("D3_SMOKE_DIR")
    smoke = bool(sd)
    a3b, table, counts, outd = ((Path(sd) / "a3B_d768.npz", Path(sd) / "table.npz", Path(sd) / "counts.npz",
                                 Path(sd) / "out") if smoke else (A3B, DF.TABLE, DF.COUNTS, D3))
    t0 = time.time()
    out, meta_p = outd / "tables" / "d3MW_d768.npz", outd / "mw_meta.json"
    for p in (out, meta_p):
        assert not p.exists(), f"{p} exists"
    (outd / "tables").mkdir(parents=True, exist_ok=True)
    za = np.load(a3b)
    zc = np.load(table)
    assert np.array_equal(za["words"], zc["words"])
    words = [str(w) for w in za["words"]]
    w1, w2, c = DF.pair_list(words, counts)
    texts = pair_strings(words, w1, w2)
    AT.verify_files()
    torch.set_num_threads(2)
    tok, model = AT.load("cpu", torch.float32)
    E32, _ = AT.encode(tok, model, texts[:64], "cpu", batch=16)
    if smoke:
        dev = "cpu"
        E16 = E32
    else:
        dev = "cuda"
        del model
        tok, model = AT.load("cuda", torch.float16)
        E16, _ = AT.encode(tok, model, texts[:64], "cuda", batch=64)
    cos = (E16.astype(np.float32) * E32.astype(np.float32)).sum(1)
    assert cos.min() >= 0.99, f"fp16 GPU vs fp32 CPU: min cosine {cos.min():.4f}"
    E, st = AT.encode(tok, model, texts, dev, batch=512, report_every=100)
    R, mu = mw_rows(E)
    keep = {k: za[k] for k in ("words", "counts", "vocab_ids")}
    np.savez(out, codes=za["codes"], pair_codes=R, pair_w1=w1.astype(np.int32), pair_w2=w2.astype(np.int32),
             pair_counts=c, **keep)
    G = R[:400].astype(np.float32) @ R[:400].astype(np.float32).T
    k = min(400, len(R))
    meta = {"pairs": len(texts), "smoke": smoke, "check_min_cos_fp16_vs_fp32": round(float(cos.min()), 5), "encode": st,
            "mean_norm": round(float(np.linalg.norm(mu)), 4),
            "mean_pair_cos_first400": round(float((G.sum() - k) / max(1, k * (k - 1))), 4), "examples": texts[:10],
            "teacher": "bge-base-en-v1.5@a5beb1e3",
            "sha256": {"d3_mw.py": DF.sha16(Path(__file__).resolve()), "d3_fit.py": DF.sha16(HERE / "d3_fit.py"),
                       "a3_teacher.py": DF.sha16(HERE / "a3_teacher.py"), "a3B": DF.sha16(a3b), "counts": DF.sha16(counts),
                       "d3MW_d768.npz": DF.sha16(out)},
            "seconds": round(time.time() - t0, 1), "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    meta_p.write_text(json.dumps(meta, indent=1))  # written last
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
