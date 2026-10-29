#!/usr/bin/env python
"""Test 5 timing pilot for the Tsetlin machine autoencoder (tmu 0.8.3), DESIGN_NOTES.md Sec. 4.4.

    python -I pilot_tmae.py --cell A1|A2|A3|A4|A5|A6|O1|R [--tiny] [--masked] [--seed 1] [--budget-s 780]

Cells (the design notes' P1..P6 renamed A1..A6, because "P1" names another registered test of this study):
  A1  shared pool (Bhattarai et al.): n 600, T 1200, s 5.0, u 25, negation off; 100 targets x 100 rounds
      (10 fit calls of 10 rounds); checkpoints at rounds 50 and 100
  A2  = A1 with n 64, T 128
  A3  = A1 with u 10
  A4  = A1 with reuse_random_feedback=1
  A5  = A1 late stage: 100 targets x 1,000 rounds (10 calls of 100), a checkpoint after every call; saves state
  A6  = A1 with every vocabulary word as a target, 20 rounds (2 calls of 10); saves state
  O1  Omni per word (Kadhim et al.): 20 words, one machine each: 32 clauses, T 20000, s 1.0, u 24, negation on,
      8 state bits, 4 fit calls x 2,000 examples; Omni vectors (mean signed state over positive clauses); saves
  R   readouts on the saved states of A5, A6 and O1 (needs those cells first): R1 bits (W >= 2), E rows, literal
      membership (R2), literal-count vectors (R3b), Omni vectors (R3c); spherical k-means K 512 and the
      existsT-style thermometer (7 per-concept quantiles over the corpus texts); nearest neighbours of the probes
Targets: the 10 probe words of the manifest (if single tokens in the vocabulary) + df-stratified words (seed 0).
--tiny: 5 targets and 2 rounds (2 fit calls of 1 round; O1: 2 calls of 1 example per word); results go to
        pilot_results_tiny.jsonl and pilot_state_*_tiny.npz, never into the real files; tiny A1 and O1 also
        retrain once to check that the run repeats bit for bit, and tiny A1 runs the R readout code on its own
        state (so the dry run touches the readout code too). Tiny A cells always save pilot_state_<cell>_tiny.npz;
        R --tiny reads pilot_state_A1_tiny.npz (for both A5 and A6) and pilot_state_O1_tiny.npz.
--masked: use tmae_fixed.MaskedClassSumAE (the paper's masked class sum) instead of tmae_fixed.TMAEFixed.
Classes: tmae_fixed.TMAEFixed = stock TMAutoEncoder + a numpy-2-safe activate_literals (stock fails with
feature_negation=False under numpy 2: OverflowError, tiny dry run 2026-10-17 23:54Z); same random draws.

Fixes of DESIGN_NOTES.md Sec. 3.8 applied here: CUDA_VISIBLE_DEVICES="" before importing tmu (tmu imports
pycuda.autoinit); OMP/BLAS/rayon threads 1; output_balancing=0.5 (the paper's balanced targets); libc srand()
before every fit call, with a value derived from (seed, fit call) so that calls do not replay the same document
draws; odd seeds only (tmu's pcg32 is a multiplicative generator: seed 0 would make fast_rand() return 0
forever); B rows thresholded at W >= 2 (all targets start from one shared random +-1 weight vector).
Each cell appends one JSON line to pilot_results.jsonl (or the tiny file) and prints it. CPU only, one thread.
Never prints corpus text: only token strings of single vocabulary types (probes, neighbours, top literals).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before tmu: tmu.util.cuda_profiler runs `import pycuda.autoinit`
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse  # noqa: E402
import ctypes  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

T5 = Path(__file__).resolve().parent
sys.path.insert(0, str(T5))  # python -I leaves the script folder off sys.path; tmae_fixed.py lives here
CORPUS, MANIFEST = T5 / "corpus_a.npz", T5 / "corpus_a.manifest.json"
LIBC = ctypes.CDLL("libc.so.6")  # the same libc whose rand() tmu's Tools.c uses for the document draws
LEVELS = [i / 8 for i in range(1, 8)]  # thermometer: 7 interior quantiles (as thermo:8)

BASE_A = dict(kind="A", n=600, T=1200, s=5.0, u=25, negation=False, reuse=0, targets="pilot100",
              rounds=100, calls=10, ckpt_calls=[5, 10], save=False)
CELLS = {
    "A1": dict(BASE_A),
    "A2": dict(BASE_A, n=64, T=128),
    "A3": dict(BASE_A, u=10),
    "A4": dict(BASE_A, reuse=1),
    "A5": dict(BASE_A, rounds=1000, calls=10, ckpt_calls=list(range(1, 11)), save=True),
    "A6": dict(BASE_A, targets="all", rounds=20, calls=2, ckpt_calls=[1, 2], save=True),
    "O1": dict(kind="O", n=32, T=20000, s=1.0, u=24, negation=True, words=20, examples=2000, epochs=4, save=True),
    "R": dict(kind="R", K=512, iters=25),
}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rss_mb() -> dict:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB on Linux
    cur = None
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return {"rss_peak_mb": round(peak, 1), "rss_now_mb": None if cur is None else round(cur, 1)}


def cpu_model() -> str | None:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def seed_libc(seed: int, call: int) -> int:
    v = (seed * 1_000_003 + call * 7_919 + 12_345) % (2**32)
    LIBC.srand(ctypes.c_uint(v))
    return v


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def load_corpus():
    z = np.load(CORPUS)
    n_docs, F = (int(v) for v in z["shape"])
    X = csr_matrix((np.ones(len(z["indices"]), dtype=np.uint32), z["indices"], z["indptr"]), shape=(n_docs, F))
    assert X.indices.dtype == np.int32 and X.indptr.dtype == np.int32, (X.indices.dtype, X.indptr.dtype)
    man = json.loads(MANIFEST.read_text())
    return X, z["vocab_ids"], z["df"], man


def probe_cols(man) -> list[int]:
    return [int(v["column"]) for v in man["probes"].values() if v.get("in_vocab")]


def stratified(df: np.ndarray, exclude: list[int], k: int, seed: int = 0) -> list[int]:
    rest = np.setdiff1d(np.arange(len(df)), np.asarray(exclude, dtype=np.int64))
    order = rest[np.argsort(df[rest], kind="stable")]
    quarters = np.array_split(order, 4)
    sizes = [len(x) for x in np.array_split(np.arange(k), 4)]
    rng = np.random.RandomState(seed)
    out: list[int] = []
    for q, m in zip(quarters, sizes):
        out.extend(int(x) for x in rng.choice(q, size=min(m, len(q)), replace=False))
    return out


def a_targets(cfg, df, man) -> np.ndarray:
    if cfg["targets"] == "all":
        return np.arange(len(df), dtype=np.uint32)
    p = probe_cols(man)
    return np.asarray(p + stratified(df, p, 100 - len(p)), dtype=np.uint32)


def o_targets(cfg, df, man) -> np.ndarray:
    p = probe_cols(man)
    return np.asarray(p + stratified(df, p, cfg["words"] - len(p)), dtype=np.uint32)


def make_ae(cfg, targets, seed: int, masked: bool):
    # TMAEFixed = stock TMAutoEncoder with a numpy-2-safe activate_literals (needed for feature_negation=False);
    # same random draws as stock, so with negation on it trains exactly like the stock class
    from tmae_fixed import MaskedClassSumAE, TMAEFixed

    Cls = MaskedClassSumAE if masked else TMAEFixed
    return Cls(number_of_clauses=cfg["n"], T=cfg["T"], s=cfg["s"],
               output_active=np.ascontiguousarray(np.asarray(targets, dtype=np.uint32)),
               accumulation=cfg["u"], feature_negation=cfg["negation"], platform="CPU", output_balancing=0.5,
               reuse_random_feedback=cfg.get("reuse", 0), number_of_state_bits_ta=8, seed=seed)


def weight_hist(W: np.ndarray) -> dict:
    labels = ["<=-100", "-99..-10", "-9..-2", "-1", "0", "1", "2..9", "10..99", ">=100"]
    edges = [-np.inf, -99.5, -9.5, -1.5, -0.5, 0.5, 1.5, 9.5, 99.5, np.inf]
    counts, _ = np.histogram(W, bins=edges)
    return dict(zip(labels, (int(c) for c in counts)))


def decode_states(cb) -> np.ndarray:
    """TA states (0..2^b-1) of every (clause, literal) from tmu's bit planes; checked against get_ta_state."""
    n, C, b = cb.number_of_clauses, cb.number_of_ta_chunks, cb.number_of_state_bits_ta
    planes = cb.clause_bank.reshape(n, C, b)
    sh = np.arange(32, dtype=np.uint32)
    st = np.zeros((n, C * 32), dtype=np.int16)
    for q in range(b):
        bits = ((planes[:, :, q, None] >> sh) & np.uint32(1)).reshape(n, C * 32).astype(np.int16)
        st |= bits << q
    return st[:, : cb.number_of_literals]


def decode_check(cb, st, rng, k=50) -> bool:
    n, L = st.shape
    for _ in range(k):
        j, l = int(rng.randint(n)), int(rng.randint(L))
        if int(cb.get_ta_state(j, l)) != int(st[j, l]):
            return False
    return True


def tokstr(tok, vocab_ids, col: int) -> str:
    return tok.id_to_token(int(vocab_ids[int(col)]))


def a_diag(tm, targets, F, B_prev, tok, vocab_ids) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    m = len(targets)
    W = np.stack([tm.get_weights(i).copy() for i in range(m)]).astype(np.int64)  # targets x clauses
    B = W >= 2
    inc = tm.clause_bank.get_literals()  # clauses x 2F (uint32 0/1)
    inc_pos = inc[:, :F].astype(bool)
    tcols = np.asarray(targets, dtype=np.int64)
    selfref = B & inc_pos[:, tcols].T  # target i, clause j: B[i, j] and clause j includes target i's literal
    bits = B.sum(1)
    n_inc = inc_pos.sum(1)
    lit_freq = inc_pos.sum(0)
    top = np.argsort(-lit_freq, kind="stable")[:10]
    d = {
        "weight_hist": weight_hist(W),
        "B_bits_per_target_mean": round(float(bits.mean()), 2),
        "B_bits_per_target_max": int(bits.max()) if m else 0,
        "B_empty_row_share": round(float((bits == 0).mean()), 4),
        "B_rows_over_10pct_share": round(float((bits > 0.1 * W.shape[1]).mean()), 4),
        "selfref_clause_links": int(selfref.sum()),
        "selfref_targets": int((selfref.sum(1) > 0).sum()),
        "included_literals_per_clause_mean": round(float(n_inc.mean()), 2),
        "included_literals_per_clause_max": int(n_inc.max()),
        "clauses_without_literal": int((n_inc == 0).sum()),
        "targets_as_literal_share": round(float(inc_pos[:, tcols].any(0).mean()), 4),
        "top_included_literals": [[tokstr(tok, vocab_ids, c), int(lit_freq[c])] for c in top if lit_freq[c] > 0],
    }
    if B_prev is not None:
        inter, union = (B & B_prev).sum(1), (B | B_prev).sum(1)
        ok = union > 0
        d["B_jaccard_to_prev_mean"] = round(float((inter[ok] / union[ok]).mean()), 4) if ok.any() else None
        d["B_both_empty_rows"] = int((~ok).sum())
    return d, B, W, inc_pos


# ----------------------------------------------------------------- readouts (cell R; also tiny A1 self-check)
def normalize_rows(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Z = Z.astype(np.float32, copy=False)
    nrm = np.linalg.norm(Z, axis=1)
    ok = nrm > 0
    out = np.zeros_like(Z)
    out[ok] = Z[ok] / nrm[ok, None]
    return out, ok


def skmeans(Zn: np.ndarray, K: int, iters: int, seed: int = 0) -> np.ndarray:
    """Spherical k-means on unit rows (numpy; the study's version is torch, dl.boolean._spherical_kmeans)."""
    rng = np.random.RandomState(seed)
    K = min(K, len(Zn))
    C = Zn[rng.choice(len(Zn), K, replace=False)].copy()
    for _ in range(iters):
        a = (Zn @ C.T).argmax(1)
        M = csr_matrix((np.ones(len(Zn), np.float32), (a, np.arange(len(Zn)))), shape=(K, len(Zn)))
        S = np.asarray(M @ Zn)
        nrm = np.linalg.norm(S, axis=1)
        ok = nrm > 0
        C[ok] = S[ok] / nrm[ok, None]
    return C


def thermo_bits(X: csr_matrix, codes: np.ndarray, cols: np.ndarray, K: int, iters: int) -> dict:
    """existsT-style bits from per-word codes: k-means concepts on the codes, sim = max over a text's words of
    cos(word, concept), 7 per-concept quantile thresholds over the corpus texts; a word's bit (k, t) is
    cos(word, concept k) >= t, so the sentence bits are the OR of its words' bits."""
    F = X.shape[1]
    Zn, ok = normalize_rows(codes)
    out = {"words_with_code": int(ok.sum())}
    if ok.sum() == 0:
        out["skipped"] = "no nonzero code"
        return out
    t0 = time.perf_counter()
    C = skmeans(Zn[ok], K, iters)
    out["kmeans_s"] = round(time.perf_counter() - t0, 3)
    out["K"] = int(C.shape[0])
    t0 = time.perf_counter()
    S = np.full((F, C.shape[0]), -2.0, dtype=np.float32)
    rows = np.asarray(cols, dtype=np.int64)[ok]
    S[rows] = Zn[ok] @ C.T
    docmax = np.empty((X.shape[0], C.shape[0]), dtype=np.float32)
    for a in range(0, X.shape[0], 2000):
        b = min(a + 2000, X.shape[0])
        lo, hi = X.indptr[a], X.indptr[b]
        vals = S[X.indices[lo:hi]]
        docmax[a:b] = np.maximum.reduceat(vals, X.indptr[a:b] - lo, axis=0)
    thr = []
    for k in range(C.shape[0]):
        v = docmax[:, k]
        v = v[v > -1.5]  # texts with at least one coded word (uncoded words carry -2 and never fire)
        thr.append(np.unique(np.quantile(v, LEVELS)).astype(np.float32) if len(v) else np.zeros(0, np.float32))
    nbits = int(sum(len(t) for t in thr))
    word_on = np.zeros(F, dtype=np.int64)
    sent_on = 0.0
    for k, t in enumerate(thr):
        word_on += (S[:, k : k + 1] >= t[None, :]).sum(1)
        sent_on += float((docmax[:, k : k + 1] >= t[None, :]).sum())
    out.update({"bits": nbits, "thermo_s": round(time.perf_counter() - t0, 3),
                "word_bits_on_mean": round(float(word_on[rows].mean()), 2),
                "sentence_bits_on_share": round(sent_on / max(1, X.shape[0] * nbits), 4)})
    return out


def neighbours(codes, cols, probes, tok, vocab_ids, k=5) -> dict:
    Zn, ok = normalize_rows(codes)
    pos = {int(c): i for i, c in enumerate(cols)}
    out = {}
    for p in probes:
        if p not in pos:
            continue
        i = pos[p]
        name = tokstr(tok, vocab_ids, p)
        if not ok[i]:
            out[name] = "empty code"
            continue
        sims = Zn @ Zn[i]
        sims[i] = -np.inf
        top = np.argsort(-sims, kind="stable")[:k]
        out[name] = [[tokstr(tok, vocab_ids, cols[j]), round(float(sims[j]), 3)] for j in top if np.isfinite(sims[j])]
    return out


def readouts(X, small: dict, full: dict | None, omni: dict | None, man, tok, vocab_ids, K, iters, budget_end):
    """small: a trained shared-pool state with the probes among its targets (A5); full: one with all words as
    targets (A6); omni: O1 vectors. Each is a dict of W (targets x clauses), inc_pos (clauses x F), targets."""
    probes = probe_cols(man)
    stages: dict = {}

    def over() -> bool:
        return time.perf_counter() > budget_end

    for tag, st in (("small", small), ("full", full)):
        if st is None:
            continue
        if over():
            stages[f"{tag}"] = "skipped: budget"
            continue
        W, inc_pos, cols = st["W"], st["inc_pos"], np.asarray(st["targets"], dtype=np.int64)
        t0 = time.perf_counter()
        B = W >= 2
        Eth = np.where(W >= 2, W, 0)  # E at the B threshold (drops the shared +-1 start)
        Epaper = np.clip(W, 0, None)  # the paper's clip(W, 0, T) (T cap is never reached here)
        V = B.astype(np.float32) @ inc_pos.astype(np.float32)  # R3b: literal counts over context words
        r2 = inc_pos[:, cols].any(0)
        s = {"targets": int(len(cols)), "clauses": int(W.shape[1]), "R1_bits": int(W.shape[1]),
             "R1_empty_share": round(float((B.sum(1) == 0).mean()), 4),
             "R1_bits_per_word_mean": round(float(B.sum(1).mean()), 2),
             "R2_bits": int(W.shape[1]), "R2_words_with_bits_share": round(float(r2.mean()), 4),
             "R3b_dims": int(V.shape[1]), "codes_s": round(time.perf_counter() - t0, 3)}
        if tag == "small":
            s["neighbours_E_theta2"] = neighbours(Eth, cols, probes, tok, vocab_ids)
            s["neighbours_E_paper"] = neighbours(Epaper, cols, probes, tok, vocab_ids)
            s["neighbours_R3b"] = neighbours(V, cols, probes, tok, vocab_ids)
        if not over():
            s["R3a_thermo"] = thermo_bits(X, Eth, cols, K, iters)
        if not over():
            s["R3b_thermo"] = thermo_bits(X, V, cols, K, iters)
        stages[tag] = s
    if omni is not None and not over():
        vec, cols = omni["vectors"], np.asarray(omni["targets"], dtype=np.int64)
        t0 = time.perf_counter()
        s = {"words": int(len(cols)), "dims": int(vec.shape[1]),
             "neighbours_omni": neighbours(vec, cols, probes, tok, vocab_ids)}
        s["R3c_thermo"] = thermo_bits(X, vec, cols, K, iters)
        s["s"] = round(time.perf_counter() - t0, 3)
        stages["omni"] = s
    return stages


# ----------------------------------------------------------------- cells
def run_a(name, cfg, args, X, df, vocab_ids, man, tok, row):
    F = X.shape[1]
    targets = a_targets(cfg, df, man)
    rounds, calls, ckpt = cfg["rounds"], cfg["calls"], set(cfg["ckpt_calls"])
    if args.tiny:
        targets, rounds, calls, ckpt = targets[:5], 2, 2, {1, 2}
    per_call = rounds // calls
    m = len(targets)
    row.update({"targets": int(m), "rounds_planned": rounds, "fit_calls_planned": calls, "examples_per_call": per_call,
                "probe_targets": [tokstr(tok, vocab_ids, c) for c in probe_cols(man) if c in set(targets.tolist())]})

    def train(seed):
        tm = make_ae(cfg, targets, seed, args.masked)
        times, ckpts, B_prev, srands, done = [], [], None, [], 0
        t_start = time.perf_counter()
        for c in range(calls):
            srands.append(seed_libc(seed, c))
            w0, c0 = time.perf_counter(), time.process_time()
            tm.fit(X, number_of_examples=per_call)
            times.append((time.perf_counter() - w0, time.process_time() - c0))
            done = c + 1
            if done in ckpt:
                d0 = time.perf_counter()
                d, B_prev, W, inc_pos = a_diag(tm, targets, F, B_prev, tok, vocab_ids)
                d.update({"round": done * per_call, "train_wall_s": round(sum(t[0] for t in times), 3),
                          "train_cpu_s": round(sum(t[1] for t in times), 3),
                          "call_wall_s": round(times[-1][0], 3), "diag_s": round(time.perf_counter() - d0, 3)})
                ckpts.append(d)
            if time.perf_counter() - t_start > args.budget_s:
                break
        return tm, times, ckpts, srands, done

    tm, times, ckpts, srands, done = train(args.seed)
    wall, cpu = sum(t[0] for t in times), sum(t[1] for t in times)
    upd = done * per_call * m
    steady = times[1:] if len(times) > 1 else times
    steady_upd = (len(steady)) * per_call * m
    s_up_steady = sum(t[0] for t in steady) / steady_upd
    row.update({
        "fit_calls_done": done, "rounds_done": done * per_call, "updates": int(upd),
        "stopped_by_budget": done < calls,
        "first_call_wall_s": round(times[0][0], 4),
        "train_wall_s": round(wall, 3), "train_cpu_s": round(cpu, 3),
        "s_per_update_wall": wall / upd, "s_per_update_cpu": cpu / upd,
        "s_per_update_wall_steady": s_up_steady,
        "s_per_round_wall_steady": s_up_steady * m,
        "extrapolated_hours_2000_rounds_all_F_targets": round(s_up_steady * 2000 * F / 3600, 3),
        "libc_srand_values": srands[:3],
        "checkpoints": ckpts,
    })
    W = np.stack([tm.get_weights(i).copy() for i in range(m)]).astype(np.int32)
    inc_pos = tm.clause_bank.get_literals()[:, :F].astype(bool)
    if cfg.get("save") or args.tiny:  # tiny A cells always save, so that R --tiny can read A1's output
        p = T5 / f"pilot_state_{name}{'_masked' if args.masked else ''}{'_tiny' if args.tiny else ''}.npz"
        np.savez(p, W=W, inc_pos=inc_pos, targets=targets, n=cfg["n"], T=cfg["T"], F=F)
        row["state_file"] = p.name
    if args.tiny:
        tm2, _, _, _, _ = train(args.seed)
        W2 = np.stack([tm2.get_weights(i).copy() for i in range(m)]).astype(np.int32)
        row["repeat_identical"] = bool(sha(W) == sha(W2) and sha(tm.clause_bank.clause_bank) == sha(tm2.clause_bank.clause_bank))
        st = {"W": W, "inc_pos": inc_pos, "targets": targets}
        t0 = time.perf_counter()
        row["tiny_readout_check"] = readouts(X, st, st, None, man, tok, vocab_ids, 512, 2,
                                             time.perf_counter() + 25)
        row["tiny_readout_check_s"] = round(time.perf_counter() - t0, 3)


def run_o(name, cfg, args, X, df, vocab_ids, man, tok, row):
    F = X.shape[1]
    targets = o_targets(cfg, df, man)
    examples, epochs = cfg["examples"], cfg["epochs"]
    if args.tiny:
        targets, examples, epochs = targets[:5], 1, 2
    row.update({"words": int(len(targets)), "examples_per_call": examples, "fit_calls_per_word": epochs,
                "words_list": [tokstr(tok, vocab_ids, c) for c in targets]})

    def train_word(wi, f):
        seed = args.seed + 2 * wi  # odd, different per word
        tm = make_ae(cfg, [int(f)], seed, args.masked)
        w0, c0 = time.perf_counter(), time.process_time()
        for ep in range(epochs):
            seed_libc(seed, ep)
            tm.fit(X, number_of_examples=examples)
        return tm, time.perf_counter() - w0, time.process_time() - c0

    t_start = time.perf_counter()
    walls, cpus, vecs, Ws, selfref, pos_share, n_inc, dec_ok = [], [], [], [], 0, [], [], True
    rng = np.random.RandomState(0)
    for wi, f in enumerate(targets):
        tm, dw, dc = train_word(wi, f)
        walls.append(dw)
        cpus.append(dc)
        st = decode_states(tm.clause_bank)
        dec_ok = dec_ok and decode_check(tm.clause_bank, st, rng)
        W = tm.get_weights(0).copy().astype(np.int64)
        pos = W > 0
        if pos.any():
            v = np.floor_divide((st[pos, :F].astype(np.int64) - st[pos, F:].astype(np.int64)).sum(0), int(pos.sum()))
        else:
            v = np.zeros(F, dtype=np.int64)
        vecs.append(v.astype(np.int16))
        Ws.append(W)
        inc = st >= 128
        selfref += int(inc[:, int(f)].sum())
        pos_share.append(float(pos.mean()))
        n_inc.append(float(inc.sum(1).mean()))
        if time.perf_counter() - t_start > args.budget_s:
            break
    done = len(walls)
    vecs, Wst = np.stack(vecs), np.stack(Ws)
    upd_per_word = examples * epochs
    row.update({
        "words_done": done, "stopped_by_budget": done < len(targets),
        "s_per_word_wall_mean": float(np.mean(walls)), "s_per_word_wall_median": float(np.median(walls)),
        "s_per_word_wall_min": float(np.min(walls)), "s_per_word_wall_max": float(np.max(walls)),
        "s_per_word_cpu_mean": float(np.mean(cpus)),
        "s_per_update_wall": float(np.mean(walls)) / upd_per_word,
        "extrapolated_hours_all_F_words_1proc": round(float(np.mean(walls)) * F / 3600, 3),
        "extrapolated_hours_all_F_words_2proc": round(float(np.mean(walls)) * F / 7200, 3),
        "weight_hist": weight_hist(Wst), "positive_clause_share_mean": round(float(np.mean(pos_share)), 4),
        "included_literals_per_clause_mean": round(float(np.mean(n_inc)), 2),
        "selfref_clauses": selfref, "state_decode_check": bool(dec_ok),
        "omni_vector_nonzero_share": round(float((vecs != 0).mean()), 4),
        "omni_vector_min": int(vecs.min()), "omni_vector_max": int(vecs.max()),
    })
    if cfg.get("save"):
        p = T5 / f"pilot_state_{name}{'_masked' if args.masked else ''}{'_tiny' if args.tiny else ''}.npz"
        np.savez(p, vectors=vecs, W=Wst, targets=targets[:done], F=F)
        row["state_file"] = p.name
    row["neighbours_omni_among_trained"] = neighbours(vecs, targets[:done], probe_cols(man), tok, vocab_ids, k=3)
    if args.tiny:
        tm2, _, _ = train_word(0, targets[0])
        st2 = decode_states(tm2.clause_bank)
        tm1, _, _ = train_word(0, targets[0])
        st1 = decode_states(tm1.clause_bank)
        row["repeat_identical"] = bool(sha(st1) == sha(st2) and sha(tm1.get_weights(0)) == sha(tm2.get_weights(0)))


def run_r(name, cfg, args, X, df, vocab_ids, man, tok, row):
    sfx = "_tiny" if args.tiny else ""
    paths = {k: T5 / f"pilot_state_{k}{sfx}.npz" for k in ("A5", "A6", "O1")}
    if args.tiny:  # R --tiny runs on A1's tiny output, which stands in for both A5 and A6
        paths["A5"] = paths["A6"] = T5 / "pilot_state_A1_tiny.npz"
    missing = [p.name for p in paths.values() if not p.exists()]
    row["inputs"] = {k: p.name for k, p in paths.items()}
    if missing:
        row["error"] = f"missing state files: {missing}"
        return
    a5, a6, o1 = (dict(np.load(paths[k])) for k in ("A5", "A6", "O1"))
    t0 = time.perf_counter()
    row["stages"] = readouts(X, a5, a6, o1, man, tok, vocab_ids, cfg["K"], cfg["iters"],
                             time.perf_counter() + args.budget_s)
    row["readout_wall_s"] = round(time.perf_counter() - t0, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget-s", type=float, default=780.0, help="soft budget; checked after every fit call")
    args = ap.parse_args()
    if args.seed % 2 == 0:
        sys.exit("--seed must be odd: tmu's pcg32 is a multiplicative generator (seed 0 never moves)")
    t0w = time.perf_counter()
    utc_start = now()
    load0 = os.getloadavg()
    import tmu  # noqa: F401  (after the environment fixes above)
    from importlib.metadata import version
    from tokenizers import Tokenizer

    X, vocab_ids, df, man = load_corpus()
    tok = Tokenizer.from_file(man["inputs"]["tokenizer"]["path"])
    cfg = CELLS[args.cell]
    row = {"cell": args.cell, "tiny": args.tiny, "masked": args.masked, "seed": args.seed,
           "utc_start": utc_start, "config": {k: v for k, v in cfg.items() if k != "ckpt_calls"},
           "corpus": {"npz_sha256": man["outputs"]["npz_sha256"][:16], "docs": int(X.shape[0]),
                      "F": int(X.shape[1]), "nnz": int(X.nnz)},
           "fixes": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "output_balancing": 0.5,
                     "libc_srand": "before every fit: (seed*1000003 + call*7919 + 12345) mod 2^32",
                     "B_threshold": "W >= 2"},
           "versions": {"python": platform.python_version(), "tmu": version("tmu"), "numpy": np.__version__,
                        "scipy": scipy.__version__},
           "cpu_model": cpu_model(), "loadavg_start": [round(x, 2) for x in load0]}
    {"A": run_a, "O": run_o, "R": run_r}[cfg["kind"]](args.cell, cfg, args, X, df, vocab_ids, man, tok, row)
    row.update(rss_mb())
    row["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
    row["wall_s_total"] = round(time.perf_counter() - t0w, 3)
    row["utc_end"] = now()
    out = T5 / ("pilot_results_tiny.jsonl" if args.tiny else "pilot_results.jsonl")
    line = json.dumps(row)
    with open(out, "a") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
