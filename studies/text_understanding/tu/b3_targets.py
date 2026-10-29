#!/usr/bin/env python
"""Step 3b, the target codes (tu/PLAN_3b.md, "Teacher numerics" and "Target bits"). CPU only.

    python -I -B b3_targets.py fit --out tu/b3/codes.npz
    python -I -B b3_targets.py numerics --codes tu/b3/codes.npz --out tu/b3/numerics.json
    python -I -B b3_targets.py choose --codes tu/b3/codes.npz --vec-dir tu/b3/teacher_train --out tu/b3/choice.json

fit: three 768-bit codes from 3a's C4 teacher vectors T_train (fp16, made on the GPU), read in chunks:
- the mean mu and the covariance of T_train over all rows (float64 sums), its eigenvectors (np.linalg.eigh, largest
  eigenvalue first, each eigenvector's sign fixed so that its largest-magnitude entry is positive);
- SH: the signs of (x - mu) @ W, W [768, 768] Gaussian from numpy default_rng(20261025) (SimHash on centred vectors);
- PT: the top 256 axes, x -> (x - mu) @ P; per axis a thermometer at the 25 / 50 / 75% quantiles of a fixed subsample
  (200,000 rows of T_train, numpy default_rng(20261026).permutation); bits axis by axis, low level first;
- ITQ: all 768 axes, then the rotation of Gong and Lazebnik (2011) fit on the same subsample: R0 = Q of a Gaussian
  matrix from default_rng(20261027), then 50 rounds of B = sign(V R) and R = Shat S^T from the SVD B^T V = S Omega
  Shat^T; bits = ((x - mu) @ P @ R > 0).
numerics: the first 500 sentences of c4_sent_held_f.txt are encoded on the CPU in fp32 (a3_teacher.load/encode, the
pinned revision after verify_files(); encode stores fp16) and compared with their fp16 GPU rows of T_held: the
minimum and mean cosine, and each code's bit disagreement rate.
choose: each intent set's TRAINING split only (dl.data.load(name).texts["train"], labels["train"]), carved 90 / 10 by
a numpy default_rng(20261025) permutation per set (the first 10% of it is the carved val); the vectors are encoded on
the CPU in fp32 (stored fp16, saved to --vec-dir with their sha256); per code, logistic regression on the code bits
(dl.linear.fit_logreg on the CPU, C 1 / 10 / 100, best on the carved val); the code with the best mean over the 3
sets wins, and SH wins when it is within 0.1 points of the best.
Every output is refused when it exists; every input and output is hashed in the JSON.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
A3 = HERE / "a3"
SETS = ("clinc150", "banking77", "hwu64")
SEED_SH, SEED_SUB, SEED_ITQ, SEED_CARVE = 20261025, 20261026, 20261027, 20261025
N_SUB, N_PT, ITQ_ITERS, N_NUM = 200_000, 256, 50, 500
QUANTS = (25.0, 50.0, 75.0)
CODES = ("SH", "PT", "ITQ")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fit_codes(T: np.ndarray, n_sub: int = N_SUB, n_pt: int = N_PT, iters: int = ITQ_ITERS, chunk: int = 100_000) -> dict:
    """T: [n, d] float array (memory-mapped is fine). Returns the code parameters (float32)."""
    n, d = T.shape
    s1, s2 = np.zeros(d), np.zeros((d, d))
    for i in range(0, n, chunk):
        x = np.asarray(T[i : i + chunk], np.float64)
        s1 += x.sum(0)
        s2 += x.T @ x
    mu = s1 / n
    cov = s2 / n - np.outer(mu, mu)
    ev, P = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, P = ev[order], P[:, order]
    P *= np.sign(P[np.abs(P).argmax(0), np.arange(d)])
    W = np.random.default_rng(SEED_SH).standard_normal((d, d))
    sub = np.sort(np.random.default_rng(SEED_SUB).permutation(n)[: min(n_sub, n)])
    Xs = np.asarray(T[sub], np.float64) - mu
    th = np.percentile(Xs @ P[:, :n_pt], QUANTS, axis=0).T  # [n_pt, 3]
    V = (Xs @ P).astype(np.float32)
    R = np.linalg.qr(np.random.default_rng(SEED_ITQ).standard_normal((d, d)))[0]
    for _ in range(iters):
        B = np.where(V @ R.astype(np.float32) >= 0, np.float32(1.0), np.float32(-1.0))
        S, _, Sh_t = np.linalg.svd((B.T @ V).astype(np.float64))
        R = Sh_t.T @ S.T
    f = np.float32
    return {"mu": mu.astype(f), "eigvals": ev.astype(f), "P": P.astype(f), "W_sh": W.astype(f), "th_pt": th.astype(f),
            "R_itq": R.astype(f), "n_pt": np.asarray(n_pt), "sub_sha16": np.asarray(hashlib.sha256(sub.tobytes()).hexdigest()[:16])}


def apply_code(c: dict, name: str, X: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """X: [n, d] vectors -> [n, 768] uint8 bits of code `name`."""
    out = []
    n_pt = int(c["n_pt"])
    for i in range(0, X.shape[0], chunk):
        x = np.asarray(X[i : i + chunk], np.float32) - c["mu"]
        if name == "SH":
            b = x @ c["W_sh"] > 0
        elif name == "PT":
            pr = x @ c["P"][:, :n_pt]
            b = (pr[:, :, None] >= c["th_pt"][None, :, :]).reshape(x.shape[0], -1)
        elif name == "ITQ":
            b = (x @ c["P"]) @ c["R_itq"] > 0
        else:
            raise ValueError(name)
        out.append(b.astype(np.uint8))
    return np.concatenate(out) if out else np.zeros((0, 768), np.uint8)


def carve(n: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(SEED_CARVE).permutation(n)
    k = round(n * 0.1)
    return np.sort(perm[k:]), np.sort(perm[:k])


def best_lr(Btr, ytr, Bcv, ycv, n_classes, fit, score, grid=(1.0, 10.0, 100.0)) -> dict:
    accs = {}
    for C in grid:
        W, b = fit(Btr.astype(np.float32), ytr, n_classes, C)
        accs[str(C)] = round(100.0 * float((score(Bcv.astype(np.float32), W, b) == ycv).mean()), 3)
    best = max(accs, key=lambda k: accs[k])
    return {"acc_by_C": accs, "best_C": float(best), "best": accs[best]}


def pick(means: dict) -> str:
    top = max(means.values())
    return "SH" if means["SH"] >= top - 0.1 - 1e-9 else max(means, key=lambda k: means[k])


def choose(codes: dict, data: dict, fit, score) -> dict:
    """data[set] = (vectors [n, d], labels [n], n_classes). Returns per set and code the carved-val LR results."""
    res = {"sets": {}, "means": {}}
    for s, (vec, y, n_classes) in data.items():
        tr, cv = carve(len(y))
        res["sets"][s] = {"n_train": len(tr), "n_carved_val": len(cv)}
        for name in CODES:
            B = apply_code(codes, name, vec)
            res["sets"][s][name] = best_lr(B[tr], y[tr], B[cv], y[cv], n_classes, fit, score)
    for name in CODES:
        res["means"][name] = round(float(np.mean([res["sets"][s][name]["best"] for s in data])), 3)
    res["pick"] = pick(res["means"])
    return res


def torch_lr():
    import torch

    sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
    from dl.linear import fit_logreg, logits

    torch.set_num_threads(2)

    def fit(X, y, n_classes, C):
        return fit_logreg(X, y, n_classes, C, device="cpu")

    def score(X, W, b):
        return logits(X, W, b, device="cpu").argmax(1).numpy()

    return fit, score


def teacher():
    import torch

    sys.path.insert(0, str(HERE))
    import a3_teacher as TEA

    torch.set_num_threads(2)
    TEA.verify_files()
    tok, model = TEA.load("cpu", torch.float32)
    return TEA, tok, model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--train", type=Path, default=A3 / "T_train.f16.npy")
    f.add_argument("--out", type=Path, required=True)
    nu = sub.add_parser("numerics")
    nu.add_argument("--codes", type=Path, required=True)
    nu.add_argument("--out", type=Path, required=True)
    ch = sub.add_parser("choose")
    ch.add_argument("--codes", type=Path, required=True)
    ch.add_argument("--vec-dir", type=Path, required=True)
    ch.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.cmd == "fit":
        T = np.load(a.train, mmap_mode="r")
        c = fit_codes(T)
        with open(a.out, "xb") as fo:
            np.savez(fo, **c, train_sha16=np.asarray(sha16(a.train)), b3_targets_sha16=np.asarray(sha16(Path(__file__).resolve())))
        bits = {k: apply_code(c, k, np.asarray(T[:20000], np.float32)) for k in CODES}
        print(json.dumps({"out": str(a.out), "sha16": sha16(a.out), "train": str(a.train), "train_sha16": sha16(a.train),
                          "rows": int(T.shape[0]), "seconds": round(time.time() - t0, 1),
                          "on_share_first_20000": {k: round(float(b.mean()), 4) for k, b in bits.items()},
                          "top_eigvals": [round(float(x), 6) for x in c["eigvals"][:5]],
                          "pt_variance_share": round(float(c["eigvals"][: int(c["n_pt"])].sum() / c["eigvals"].sum()), 4),
                          "utc": utc()}))
        return 0
    codes = dict(np.load(a.codes))
    TEA, tok, model = teacher()
    if a.cmd == "numerics":
        held = A3 / "c4_sent_held_f.txt"
        with open(held) as fh:
            texts = [next(fh).rstrip("\n") for _ in range(N_NUM)]
        cpu, info = TEA.encode(tok, model, texts, "cpu", batch=64)
        gpu = np.load(A3 / "T_held.f16.npy", mmap_mode="r")[:N_NUM]
        a32, b32 = cpu.astype(np.float32), np.asarray(gpu, np.float32)
        cos = (a32 * b32).sum(1) / (np.linalg.norm(a32, axis=1) * np.linalg.norm(b32, axis=1))
        dis = {k: round(float((apply_code(codes, k, a32) != apply_code(codes, k, b32)).mean()), 6) for k in CODES}
        res = {"n": N_NUM, "texts": f"the first {N_NUM} lines of {held}", "held_sha16": sha16(held),
               "T_held_sha16": sha16(A3 / "T_held.f16.npy"), "codes_sha16": sha16(a.codes),
               "cpu": "fp32 compute on the CPU, 2 threads, stored fp16 by a3_teacher.encode", "gpu": "T_held: fp16 on cuda (3a)",
               "cos_min": round(float(cos.min()), 6), "cos_mean": round(float(cos.mean()), 6), "bit_disagreement": dis,
               "encode_info": info, "a3_teacher_sha16": sha16(HERE / "a3_teacher.py"),
               "b3_targets_sha16": sha16(Path(__file__).resolve()), "seconds": round(time.time() - t0, 1), "utc": utc()}
    else:
        sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
        from dl.data import load

        fit, score = torch_lr()
        a.vec_dir.mkdir(parents=True, exist_ok=True)
        data, vec_files = {}, {}
        for s in SETS:
            ds = load(s)
            texts, y = ds.texts["train"], np.asarray(ds.labels["train"])
            vp = a.vec_dir / f"{s}__train_bge-base_cpu32.npz"
            if vp.exists():
                z = np.load(vp)
                assert int(z["n"]) == len(texts), (s, int(z["n"]), len(texts))
                vec = z["vec"]
            else:
                vec, info = TEA.encode(tok, model, texts, "cpu", batch=64)
                with open(vp, "xb") as fo:
                    np.savez(fo, vec=vec, n=np.asarray(len(texts)), info=json.dumps(info))
            vec_files[str(vp)] = sha16(vp)
            data[s] = (vec, y, len(ds.intents))
        res = choose(codes, data, fit, score)
        res.update({"codes_sha16": sha16(a.codes), "vector_files": vec_files, "split": "TRAINING split only, carved 90 / 10",
                    "teacher": "bge-base-en-v1.5@a5beb1e3, fp32 compute on the CPU, stored fp16",
                    "a3_teacher_sha16": sha16(HERE / "a3_teacher.py"), "b3_targets_sha16": sha16(Path(__file__).resolve()),
                    "seconds": round(time.time() - t0, 1), "utc": utc()})
    with open(a.out, "x") as fo:
        fo.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res if a.cmd == "numerics" else {"means": res["means"], "pick": res["pick"], "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
