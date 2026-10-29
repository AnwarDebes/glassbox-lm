#!/usr/bin/env python
"""Test 5 code tables keyed by potion token id (REGISTRATION_T5.md, "Readouts" and "Controls").

    python -I codes.py --ckpt t5/tmae_ckpt_r2000.npz              # all five tables
    python -I codes.py --ckpt PATH --tables R1,R3a                  # only the TM-AE tables
    python -I codes.py --controls-only                              # potionV, ppmiV, wordsV (no checkpoint)
    python -I codes.py --ckpt t5/pilot_state_A6.npz --suffix _smoke # smoke test on a stand-in checkpoint

Rows of every table = the 4,564 corpus-(a) vocabulary tokens, in the order of corpus_a.npz vocab_ids:
  R1      uint8   [W >= 2]                                   4,564 x 600 (an all-zero row = no code)
  R3a     float32 clip(W, 0, T) * [W >= 2], L2-normalized    4,564 x 600 (all-zero rows stay zero = no code)
  potionV float32 the potion-base-32M rows (model.safetensors "embeddings", revision 1e5a03f8), 4,564 x 512.
          Check: for 200 CLINC150 training sentences, normalize(mean of their token rows) equals the study's
          cached potion-32m sentence vectors (glassbox-lm/studies/text_understanding/emb/clinc150__potion-32m.npz, train only).
  ppmiV   float32 PPMI-SVD (REGISTRATION_T5.md Amendment 1): the PPMI of the token-by-token document co-occurrence
          counts of corpus (a): count(a, b) = number of texts holding both a and b, for a != b (a token is not its
          own context: the diagonal is 0), context smoothing alpha 0.75 (Levy, Goldberg and Dagan 2015):
          PMI = log(P(a,b) / (P(a) P_alpha(b))), P_alpha(b) = count(b)^0.75 / sum_c count(c)^0.75, PPMI = max(PMI, 0);
          then a full numpy SVD (LAPACK gesdd, float64, one thread: deterministic) of the dense 4,564 x 4,564 PPMI,
          truncated to 600 dimensions, rows U_600 * S_600^0.5 (each singular vector's sign fixed so that its
          largest-magnitude entry is positive; cosines do not depend on it), L2-normalized; 4,564 x 600. The
          manifest records the share of the singular-value sum (and of the squared sum) that is kept.
          The raw PPMI rows are not saved (Amendment 1: not run).
  wordsV  uint8   identity, 4,564 x 4,564
Files: t5/codes_{name}{suffix}.npz (arrays: codes, vocab_ids) and t5/codes_manifest{suffix}.json (sha256, shape,
dtype, sources, and for R1/R3a a short report: weight histogram, empty rows, bits per token, probe neighbours,
readable clauses). An existing manifest is updated, not replaced. CPU only (numpy); the checkpoint may be a
train_tmae.py checkpoint (W, include) or a pilot state (W, inc_pos). Never reads val or test text.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
_HOME = _os.path.expanduser("~")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

T5 = Path(__file__).resolve().parent
EMB = T5.parent
CORPUS, CORPUS_MAN = T5 / "corpus_a.npz", T5 / "corpus_a.manifest.json"
REV = "1e5a03f8eeb2c98b928fbbd846f22f816360919f"
SNAP = Path(f"{_HOME}/.cache/huggingface/hub/models--minishlab--potion-base-32M/snapshots") / REV
DL_SRC = Path(f"{_REPO_ROOT}/studies/decision_layer/src")
ALL = ("R1", "R3a", "potionV", "ppmiV", "wordsV")
CONTROLS = ("potionV", "ppmiV", "wordsV")
ALPHA = 0.75
SVD_DIMS = 600  # Amendment 1: the TM-AE's 600 dimensions


def utc() -> str:
    return subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_hist(W: np.ndarray) -> dict:
    labels = ["<=-100", "-99..-10", "-9..-2", "-1", "0", "1", "2..9", "10..99", ">=100"]
    edges = [-np.inf, -99.5, -9.5, -1.5, -0.5, 0.5, 1.5, 9.5, 99.5, np.inf]
    counts, _ = np.histogram(W, bins=edges)
    return dict(zip(labels, (int(c) for c in counts)))


def load_ckpt(path: Path, F: int, vocab_ids: np.ndarray) -> dict:
    z = np.load(path)
    W = z["W"].astype(np.int64)
    inc = z["include"][:, :F] if "include" in z.files else z["inc_pos"][:, :F]
    targets = z["targets"].astype(np.int64)
    assert W.shape[0] == F and np.array_equal(targets, np.arange(F)), "checkpoint targets must be all F tokens in order"
    if "vocab_ids" in z.files:
        assert np.array_equal(z["vocab_ids"], vocab_ids), "checkpoint vocabulary differs from corpus_a.npz"
    T = int(z["T"])
    rounds = int(z["rounds"]) if "rounds" in z.files else None
    return {"W": W, "inc": inc.astype(bool), "T": T, "rounds": rounds}


def report(ck: dict, R1: np.ndarray, R3a: np.ndarray, vocab_ids, tok, probes: dict) -> dict:
    W, inc = ck["W"], ck["inc"]
    bits = R1.sum(1)
    lits = inc.sum(1)
    name = lambda r: tok.id_to_token(int(vocab_ids[int(r)]))  # noqa: E731
    out = {"weight_hist": weight_hist(W), "R1_empty_rows": int((bits == 0).sum()),
           "R1_empty_share": round(float((bits == 0).mean()), 4), "R1_bits_per_token_mean": round(float(bits.mean()), 2),
           "R1_bits_per_token_max": int(bits.max()), "clauses_without_literal": int((lits == 0).sum()),
           "literals_per_clause_mean": round(float(lits.mean()), 2), "literals_per_clause_max": int(lits.max()),
           "probes": {}}
    for w, info in probes.items():
        r = info.get("column")
        if r is None:
            continue
        p: dict = {"R1_bits": int(bits[r])}
        if R3a[r].any():
            cos = R3a @ R3a[r]
            cos[r] = -np.inf
            top = np.argsort(-cos, kind="stable")[:5]
            p["R3a_neighbours"] = [[name(j), round(float(cos[j]), 3)] for j in top]
        if bits[r]:
            inter = (R1.astype(np.int32) @ R1[r].astype(np.int32)).astype(np.float64)
            union = bits + bits[r] - inter
            jac = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
            jac[r] = -np.inf
            top = np.argsort(-jac, kind="stable")[:5]
            p["R1_jaccard_neighbours"] = [[name(j), round(float(jac[j]), 3)] for j in top]
            cl = np.argsort(-W[r], kind="stable")[:3]
            p["top_clauses"] = [{"clause": int(j), "weight": int(W[r, j]),
                                 "literals": [name(i) for i in np.flatnonzero(inc[j])[:12]],
                                 "n_literals": int(inc[j].sum())} for j in cl]
        out["probes"][w] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--tables", default=None, help="comma list of " + ",".join(ALL))
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    tables = CONTROLS if args.controls_only else tuple((args.tables or ",".join(ALL)).split(","))
    assert all(t in ALL for t in tables), tables
    if any(t in ("R1", "R3a") for t in tables):
        assert args.ckpt is not None and args.ckpt.exists(), "R1 / R3a need --ckpt"
    t0 = time.time()
    from tokenizers import Tokenizer

    z = np.load(CORPUS)
    n_docs, F = (int(v) for v in z["shape"])
    vocab_ids = z["vocab_ids"].astype(np.int32)
    cman = json.loads(CORPUS_MAN.read_text())
    assert cman["outputs"]["npz_sha256"] == sha256_file(CORPUS)
    tok = Tokenizer.from_file(cman["inputs"]["tokenizer"]["path"])
    man_path = T5 / f"codes_manifest{args.suffix}.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"tables": {}}
    man["corpus_npz_sha256"] = cman["outputs"]["npz_sha256"]
    man["codes_py_sha256"] = sha256_file(Path(__file__).resolve())[:16]
    out_tables: dict = {}

    def save(name: str, codes: np.ndarray, extra: dict) -> None:
        p = T5 / f"codes_{name}{args.suffix}.npz"
        np.savez_compressed(p, codes=codes, vocab_ids=vocab_ids)
        has = codes.any(1) if codes.dtype != np.float32 else (np.abs(codes).sum(1) > 0)
        rec = {"file": p.name, "sha256": sha256_file(p), "shape": list(codes.shape), "dtype": str(codes.dtype),
               "rows_with_code": int(has.sum()), "written_utc": utc(), **extra}
        man["tables"][name] = rec
        out_tables[name] = {k: rec[k] for k in ("file", "shape", "dtype", "rows_with_code")} | {"sha256": rec["sha256"][:16]}

    if any(t in ("R1", "R3a") for t in tables):
        ck = load_ckpt(args.ckpt, F, vocab_ids)
        src = {"ckpt": str(args.ckpt), "ckpt_sha256": sha256_file(args.ckpt), "ckpt_rounds": ck["rounds"], "T": ck["T"]}
        W = ck["W"]
        R1 = (W >= 2).astype(np.uint8)
        R3a = (np.clip(W, 0, ck["T"]) * (W >= 2)).astype(np.float32)
        nrm = np.linalg.norm(R3a, axis=1)
        R3a[nrm > 0] /= nrm[nrm > 0, None]
        rep = report(ck, R1, R3a, vocab_ids, tok, cman["probes"])
        if "R1" in tables:
            save("R1", R1, {"source": src, "rule": "[W >= 2]", "report": rep})
        if "R3a" in tables:
            save("R3a", R3a, {"source": src, "rule": "clip(W, 0, T) * [W >= 2], L2-normalized"})
    if "potionV" in tables:
        from safetensors import safe_open

        st_path = SNAP / "model.safetensors"
        with safe_open(str(st_path), "np") as f:
            E = f.get_tensor("embeddings").astype(np.float32)
        P = E[vocab_ids]
        # check against the study's cached sentence vectors (normalize(mean of the token rows); train only)
        sys.path.insert(0, str(DL_SRC))
        from dl.data import load

        texts = list(load("clinc150").texts["train"][:200])
        ref = np.load(EMB / "emb" / "clinc150__potion-32m.npz")["train"][:200]
        ids = [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
        mine = np.stack([E[i].mean(0) if len(i) else np.zeros(E.shape[1], np.float32) for i in ids])
        mine /= np.maximum(np.linalg.norm(mine, axis=1, keepdims=True), 1e-12)
        diff = float(np.abs(mine - ref).max())
        assert diff < 1e-4, f"potion rows do not reproduce the cached sentence vectors (max abs diff {diff})"
        save("potionV", P, {"source": {"safetensors": str(st_path), "sha256": sha256_file(st_path), "revision": REV},
                            "check_vs_cached_clinc150_train_200_maxabs": diff})
    if "ppmiV" in tables:
        X = csr_matrix((np.ones(len(z["indices"]), dtype=np.float64), z["indices"], z["indptr"]), shape=(n_docs, F))
        C = (X.T @ X).toarray()
        np.fill_diagonal(C, 0.0)
        total = C.sum()
        row = C.sum(1)
        col_a = C.sum(0) ** ALPHA
        Pw = row / total
        Pc = col_a / col_a.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(C / total) - np.log(Pw)[:, None] - np.log(Pc)[None, :]
        ppmi = np.where((C > 0) & np.isfinite(pmi), np.maximum(pmi, 0.0), 0.0)  # float64
        del C, pmi
        ts = time.time()
        U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
        svd_s = time.time() - ts
        k = SVD_DIMS
        Uk = U[:, :k]
        big = np.argmax(np.abs(Uk), axis=0)
        sgn = np.sign(Uk[big, np.arange(k)])
        sgn[sgn == 0] = 1.0
        rows = Uk * sgn[None, :] * np.sqrt(S[:k])[None, :]
        nrm = np.linalg.norm(rows, axis=1)
        rows[nrm > 0] /= nrm[nrm > 0, None]
        save("ppmiV", rows.astype(np.float32), {
            "rule": "PPMI-SVD (Amendment 1): U_600 * S_600^0.5 of the PPMI, L2-normalized",
            "source": {"corpus_npz_sha256": cman["outputs"]["npz_sha256"]}, "alpha": ALPHA,
            "diagonal": "excluded (count of distinct-token pairs only)",
            "ppmi_nonzero_share": round(float((ppmi > 0).mean()), 4),
            "svd": "numpy.linalg.svd (LAPACK gesdd), full, float64, one thread", "svd_s": round(svd_s, 1),
            "dims": k, "singular_value_sum_share_kept": round(float(S[:k].sum() / S.sum()), 4),
            "singular_value_squared_sum_share_kept": round(float((S[:k] ** 2).sum() / (S ** 2).sum()), 4),
            "singular_values_first_last": [round(float(S[0]), 4), round(float(S[k - 1]), 4)],
            "zero_rows": int((nrm == 0).sum())})
    if "wordsV" in tables:
        save("wordsV", np.eye(F, dtype=np.uint8), {"rule": "identity: one bit per vocabulary token present"})
    man["updated_utc"] = utc()
    tmp = man_path.with_name(man_path.name + ".partial")  # atomic: a parallel t5_tm.py run never reads a half-written file
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, man_path)
    print(json.dumps({"manifest": man_path.name, "tables": out_tables, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
