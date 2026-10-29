"""Shared helpers for the self-tests: small corpora built in memory or through S0 on the synthetic files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parents[1]
if str(A) not in sys.path:
    sys.path.insert(0, str(A))

from astruct.data import Corpus, _csr, hash64
from astruct.paths import TMP

TEST_TMP = TMP / "selftest"


def random_corpus(n_rep: int = 300, n_words: int = 40, n_ph: int = 4, n_cond: int = 6, seed: int = 0,
                  split_share=(0.7, 0.15, 0.15)) -> Corpus:
    """Random replies over a small vocabulary with placeholders, conditions and expected placeholders."""
    rng = np.random.default_rng(seed)
    phs = [f"[d-s{i}]" for i in range(n_ph)]
    words = [f"w{i}" for i in range(n_words - 1)] + ["?"]
    vocab = ["<end>", "<unk>"] + phs + words
    V = len(vocab)
    zipf = 1.0 / np.arange(1, V) ** 1.1
    zipf /= zipf.sum()
    toks, exp, acts, pairs, S, Ah, split, dlg = [], [], [], [], [], [], [], []
    for r in range(n_rep):
        cond = int(rng.integers(n_cond))
        L = int(rng.integers(1, 12))
        body = [int(x) for x in rng.choice(np.arange(1, V), L, p=zipf)]
        if cond % 2 == 0:  # a condition-specific habit, so conditions carry information
            body = [2 + cond % n_ph] + body[: L - 1]
        e = sorted({int(2 + (cond + k) % n_ph) for k in range(int(rng.integers(0, 3)))})
        toks.append(body + [0])
        exp.append(e)
        acts.append([cond % 3, 3 + cond % 2])
        pairs.append([cond, n_cond + cond % 2])
        S.append(hash64(f"S:{cond}"))
        Ah.append(hash64(f"A:{cond % 3}"))
        u = rng.random()
        split.append(0 if u < split_share[0] else (1 if u < split_share[0] + split_share[1] else 2))
        dlg.append(r // 3)
    tok_off, tok = _csr(toks)
    exp_off, exp_ids = _csr(exp)
    act_off, act_ids = _csr(acts)
    pair_off, pair_ids = _csr(pairs)
    split = np.array(split, np.int8)
    a = {"r_dlg": np.array(dlg, np.int32), "r_turn": np.ones(n_rep, np.int16), "r_h": np.where(split == 0, 50, 3).astype(np.int16),
         "r_split": split, "r_train": (split == 0) & (np.arange(n_rep) % 2 == 0), "tok_off": tok_off, "tok": tok.astype(np.int32),
         "act_off": act_off, "act_ids": act_ids.astype(np.int16), "pair_off": pair_off, "pair_ids": pair_ids.astype(np.int16),
         "exp_off": exp_off, "exp_ids": exp_ids.astype(np.int32), "S_hash": np.array(S, np.int64), "A_hash": np.array(Ah, np.int64),
         "concept32": np.where(np.arange(V) >= 2 + n_ph, np.arange(V) % 5, -1).astype(np.int16),
         "concept64": np.where(np.arange(V) >= 2 + n_ph, np.arange(V) % 7, -1).astype(np.int16)}
    meta = {"vocab": vocab, "act_names": [f"D{i}-Act" for i in range(5)],
            "pair_names": [[f"D{i}-Act", "slot"] for i in range(n_cond + 2)],
            "dialogue_ids": [f"R{i}.json" for i in range(n_rep // 3 + 1)],
            "lists": {"top32": words[:8], "cue40": words[:10], "pairs64": [[f"D{i}-Act", "slot"] for i in range(4)]}}
    return Corpus(a, meta)


def synth_dirs(name: str, n: int = 300, seed: int = 0) -> tuple[Path, Path]:
    from synth import write_c4_stub, write_mwoz
    root = TEST_TMP / name
    mw, c4 = root / "mwoz", root / "c4"
    if not (mw / "MANIFEST.sha256").exists():
        write_mwoz(mw, n=n, seed=seed)
    if not (c4 / "vectors.npy").exists():
        write_c4_stub(c4, seed=seed)
    return mw, c4


def synth_corpus(name: str = "base", n: int = 300, seed: int = 0) -> Corpus:
    from astruct.data import make_corpus
    from astruct.paths import load_tokenize_ref
    mw, c4 = synth_dirs(name, n, seed)
    tk = load_tokenize_ref().tokens
    arrays, meta = make_corpus(mw, c4, tk, log=lambda *a: None)
    return Corpus(arrays, meta)
