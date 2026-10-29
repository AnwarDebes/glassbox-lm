"""Synthetic C trees for the step-9 self-tests (not a test itself; imported by tests/selftest_step9.py and
tests/selftest_step9_e2e.py). Synthetic data only: no PTB text, no split of C, no number of C.

build_tree(root, twin_gain_nats) writes, under a test root inside C/tmp/:
  .step9_test_root (marker), .gpu_go (a fake go file), fake_nvidia_smi.sh (prints a process line when the file
  gpu_busy exists, exits 9 when gpu_broken exists, else prints nothing), R1_C-core.md (a stand-in text);
  data/ and data_dry/ with prep.npz, twin.npz, learners.npz and step6_decision.json of the same synthetic data; and
  finished run records of prep, twin, learners and gates (real and dry) whose outputs.json hold the files' hashes.
The token stream: 129 words in 64 classes plus <eos> (class 64); the next word's class is, with probability 0.85,
one of 4 classes picked by the top 2 E bits of the previous word, so a small TM can learn it. The floor F is a
positive vector over the words at every position (0.3 the generating process + 0.7 the train unigram)."""
import json
import os
from pathlib import Path

import numpy as np

V = 130
EOS = 0
SENT_PER_BLOCK = 10


def _stream(rng, n_tokens, cls_words):
    out = []
    while len(out) < n_tokens:
        L = int(rng.integers(3, 10))
        c = int(rng.integers(0, 64))
        for _ in range(L):
            out.append(int(rng.choice(cls_words[c])))
            c = 16 * (c >> 4) + int(rng.integers(0, 4)) if rng.random() < 0.85 else int(rng.integers(0, 64))
        out.append(EOS)
    return np.array(out, dtype=np.int64)


def _blocks(x):
    sent = np.concatenate([[0], np.cumsum(x[:-1] == EOS)])
    return (sent // SENT_PER_BLOCK).astype(np.int32)


def _true_dist(x, cls, cls_words):
    """The generating process at every position: <eos> after k words with hazard 1 / (10 - k) for k >= 3 (lengths
    3-9, uniform); else the next class (uniform at a sentence start; else 0.85 within the previous class's group of
    4, 0.15 uniform) and a uniform word inside it."""
    n = len(x)
    start = np.concatenate([[True], x[:-1] == EOS])
    sid = np.cumsum(start) - 1
    k = np.arange(n) - np.flatnonzero(start)[sid]
    h = np.where(k < 3, 0.0, 1.0 / np.maximum(10 - k, 1))
    pc = np.full((n, 64), 1.0 / 64)
    prev = np.where(k > 0, cls[np.roll(x, 1)], -1)
    inner = k > 0
    g = 16 * (prev[inner] >> 4)
    pc[inner] = 0.15 / 64
    rows = np.flatnonzero(inner)
    for j in range(4):
        pc[rows, g + j] += 0.85 / 4
    size = np.array([len(w) for w in cls_words], dtype=np.float64)
    D = np.zeros((n, V))
    words = np.arange(1, V)
    D[:, words] = (1 - h)[:, None] * pc[:, cls[words]] / size[cls[words]][None, :]
    D[:, EOS] = h
    return D


def _floor(x, cls, cls_words, unigram):
    """The floor: 0.3 x the generating process + 0.7 x the train unigram (positive everywhere). A context-free class
    member (the class unigram, or a TM trained on shuffled targets) cannot improve it, as with KN5 + cache on PTB,
    while a class model that learned the transitions can (checked when this file was written: class unigram
    +0.0000, shuffled smoke TM +0.0000, a 64-clause TM on all train positions -0.2400 bpw against F)."""
    D = 0.3 * _true_dist(x, cls, cls_words) + 0.7 * unigram[None, :]
    D /= D.sum(1, keepdims=True)
    return D


def emission(train, cls, delta=0.5):
    """ccore/codes.py's emission rule (n(w) + 0.5) / (n(c) + 0.5 |c|), written out here because codes.py imports
    numba, which .venv-dl lacks (checked equal to codes.emission in .venv-tm when this file was written)."""
    n = np.bincount(np.asarray(train), minlength=len(cls)).astype(np.float64)
    K = int(cls.max()) + 1
    size = np.bincount(cls, minlength=K).astype(np.float64)
    ncls = np.bincount(cls, weights=n, minlength=K)
    return (n + delta) / (ncls[cls] + delta * size[cls])


def make_data(seed=0, n_train=20000, n_calib=3000, n_valid=3000, n_pos=400):
    """Arrays of a synthetic prep.npz (the keys the TM stages read)."""
    from ccore import layout
    rng = np.random.default_rng(seed)
    cls = np.full(V, 64, dtype=np.int64)
    cls[1:] = (np.arange(1, V) - 1) % 64
    cls_words = [np.flatnonzero(cls == c) for c in range(64)]
    path6 = np.where(np.arange(V) == EOS, 0, cls % 64).astype(np.int64)
    path8 = rng.integers(0, 256, size=V).astype(np.int64)
    x = {"train": _stream(rng, n_train, cls_words), "calib": _stream(rng, n_calib, cls_words),
         "valid": _stream(rng, n_valid, cls_words)}
    counts = np.bincount(x["train"], minlength=V).astype(np.float64)
    unigram = (counts + 1.0) / (counts.sum() + V)
    Z = {"cls": cls, "path6": path6, "path8": path8, "counts": counts,
         "emission": emission(x["train"], cls)}
    for s, xs in x.items():
        Z[f"X_{s}"] = layout.features(xs, EOS, path6, path8)
        Z[f"y_{s}"] = cls[xs]
        Z[f"w_{s}"] = xs.astype(np.int32)
        Z[f"blocks_{s}"] = _blocks(xs)
        if s == "train":
            continue
        D = _floor(xs, cls, cls_words, unigram)
        n = len(xs)
        M = np.zeros((V, layout.N_CLASSES))
        M[np.arange(V), cls] = 1.0
        Z[f"pF_{s}"] = D[np.arange(n), xs]
        Z[f"Fc_{s}"] = D @ M
        Z[f"b_{s}"] = rng.integers(0, 40, size=n).astype(np.int64)
        Z[f"L_{s}"] = rng.integers(0, 6, size=n).astype(np.int64)
        pos = np.sort(rng.choice(n, size=min(n_pos, n), replace=False))
        Z[f"pos_{s}"] = pos
        Z[f"D_{s}"] = D[pos]
    return Z


def _bpw(lp):
    return float(-np.mean(lp) / np.log(2.0))


def write_data(d, Z, twin_gain_nats):
    """prep.npz, twin.npz, learners.npz and step6_decision.json in folder d; returns their paths."""
    os.makedirs(d, exist_ok=True)
    np.savez(d / "prep.npz", **Z)
    F = np.log(Z["pF_valid"])
    np.savez(d / "twin.npz", F_valid=F, best_valid=F + twin_gain_nats)
    np.savez(d / "learners.npz", F_valid=F, lr_valid=F + 0.5 * twin_gain_nats, lgbm_valid=F + 0.8 * twin_gain_nats)
    dec = {"decision": "PASS: K0 and K0b passed; the GPU TM step may be scheduled",
           "valid_bpw": {"F": _bpw(F), "twin": _bpw(F + twin_gain_nats), "LR": _bpw(F + 0.5 * twin_gain_nats),
                         "LightGBM": _bpw(F + 0.8 * twin_gain_nats), "class_unigram_diagnostic": _bpw(F) - 1e-4},
           "AA": {"diff_bpw": 0.001, "max_abs": 0.0036, "pass": True}, "label": "synthetic self-test data"}
    (d / "step6_decision.json").write_text(json.dumps(dec, indent=1, sort_keys=True))
    return {"prep": [d / "prep.npz"], "twin": [d / "twin.npz"], "learners": [d / "learners.npz"],
            "gates": [d / "step6_decision.json"]}


def write_runs(runs, files, dry):
    from ccore import hashing
    os.makedirs(runs, exist_ok=True)
    for i, (st, fs) in enumerate(files.items()):
        rd = runs / f"20261023T00000{i}Z_{st}{'_dry' if dry else ''}"
        os.makedirs(rd)
        (rd / "run.json").write_text(json.dumps({"state": "finished", "wall_s": 1.0}))
        (rd / "metrics.json").write_text("{}")
        (rd / "outputs.json").write_text(json.dumps(hashing.manifest(fs)))


FAKE_NVSMI = """#!/bin/bash
# fake GPU process list of the step-9 self-tests
here=$(dirname "$0")
[ -e "$here/gpu_broken" ] && { echo "fake nvidia-smi failure"; exit 9; }
[ -e "$here/gpu_busy" ] && echo "424242, python, 1234 MiB"
exit 0
"""


def build_tree(root, twin_gain_nats=0.0, seed=0, **kw):
    root = Path(root)
    os.makedirs(root, exist_ok=False)
    (root / ".step9_test_root").write_text("step-9 self-test root\n")
    (root / ".gpu_go").write_text("fake go file of a self-test\n")
    (root / "fake_nvidia_smi.sh").write_text(FAKE_NVSMI)
    (root / "R1_C-core.md").write_text("stand-in registration text of a step-9 self-test (synthetic data)\n")
    os.makedirs(root / "tmp")
    Z = make_data(seed=seed, **kw)
    write_runs(root / "runs", write_data(root / "data", Z, twin_gain_nats), dry=False)
    write_runs(root / "runs", write_data(root / "data_dry", Z, twin_gain_nats), dry=True)
    return Z


def freeze_into(root, logmd_line=True):
    """The first freeze of the test root (scripts/freeze.py, run in-process with every record path redirected) and
    a LOG.md line holding its hash. Returns the record's SHA-256."""
    import importlib.util
    from ccore import hashing, paths, tmcore
    saved = {k: getattr(paths, k) for k in ("DATA", "DATA_DRY", "RUNS", "LISTINGS", "REG", "FROZEN")}
    saved_log = hashing.LOGMD
    try:
        tmcore.redirect_to_test_root(Path(root))
        spec = importlib.util.spec_from_file_location("c_freeze_selftest", paths.C / "scripts" / "freeze.py")
        fr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fr)
        fr.main([])
        h = hashing.sha256(paths.FROZEN)
        if logmd_line:
            hashing.LOGMD.write_text(f"- self-test: R1 [C-core] stand-in line, frozen record {h}\n")
        return h
    finally:
        for k, v in saved.items():
            setattr(paths, k, v)
        hashing.LOGMD = saved_log
