"""H9.6 of docs/13 amendment A4 (pre-registered) and the gate G1 T1: the vote equilibrium of the
signed-vote engine, S = T tanh(0.5 ln(p / b)) with b = (1 - p) Q / (V - 1).

Registered row (docs/13 A4): "CP bank, 200,000 validation positions, the 50 most probable KN-5
classes per position; bins of x = ln(p_KN / b); mean S/T per bin against tanh(x/2). Weighted R^2 at
least 0.8, AND Spearman(S, ln p) at most 0.2 among classes with p > 20 b. Prediction 0.55."
Gate G1 T1 (docs/13 A4): a weighted R^2 of at least 0.8 opens a same-data bank trained on the
positions the core predicts poorly (the NCE-style engine change follows only if that bank beats a
uniform-negative bank by at least 0.005 bits per byte). docs/13 A5: H9.6 met is row D10, not met D11.

The derivation from the engine's update rules (docs/11 sections 4, 5.2, 5.3 and 8c; tm_core.h) is
the constant DERIVATION below; the run prints it into its log before any score is computed.

Inputs (read only): the CP anchor bank `results/20260923T042350Z_CP_anchor/model_seed1.npz` (its state
hash must equal the logged epoch-4 checkpoint hash); the validation store at V 8000; the validation
group of the frozen CVTM cache `results/20260924T085728Z_cvtm_cache_build/cache/` (top-64 of the full-data
KN-5, i.e. the KN-5 fitted on every training document, docs/11 11.6), whose first 50 entries are the 50
most probable KN-5 classes (descending, ties to the lower id). The test split is never loaded.

Phases (every analysis choice is fixed in the config before any class score was computed):
  gpu       (child process under scripts/gpu_run.sh shared, so the GPU lock is held only here) the CUDA
            engine's predict_scores over EVERY validation position: the raw top-1 must reproduce the
            logged 0.14151747957266259 exactly; for the 200,000 selected positions the integer class
            scores S of the 50 KN classes, the gold score, the number of fired clauses and the numbers
            of fired clauses holding a positive or a negative vote for each class (diagnostics).
  analysis  (CPU, in the parent) the registered statistics, the document-block bootstrap, secondary
            diagnostics labelled as such, the plot data and a PNG, the verdict and the gate.
  planted   (separate config and run, EXPLORATORY, seeds >= 100) the derivation checked on the real CPU
            engine in a planted world where the class distribution p is known and the same in every
            context (tabular mode, random features, labels independent of the features).

Launch (confirmatory; copies the pattern of results/20260924T062704Z_cvtm_H8_1_causal_cp20M, with the
GPU lock taken by the child only):
    scripts/cpu_slot.sh 1 -- .venv-tm/bin/python scripts/mem_guard.py -- scripts/run_logged.py \\
        --name cvtm_H9_6_vote_equilibrium --config experiments/cvtm/configs/cvtm_H9_6_vote_equilibrium.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_vote_equilibrium.py --config experiments/cvtm/configs/cvtm_H9_6_vote_equilibrium.json
Planted check (exploratory):
    CUDA_VISIBLE_DEVICES= scripts/cpu_slot.sh 1 -- .venv-tm/bin/python scripts/mem_guard.py -- scripts/run_logged.py \\
        --no-gpu-sampler --name cvtm_H9_6_planted_exploratory --config experiments/cvtm/configs/cvtm_H9_6_planted_exploratory.json -- \\
        .venv-tm/bin/python -u experiments/cvtm_vote_equilibrium.py --config experiments/cvtm/configs/cvtm_H9_6_planted_exploratory.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from engine_common import ROOT, load_json  # noqa: E402

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

GIB = float(1 << 30)
CG = Path("/sys/fs/cgroup")
CORPUS = "wikitext103"

DERIVATION = r"""
DERIVATION of the vote equilibrium (docs/13 A4, H9.6), from docs/11 sections 4, 5.1, 5.2, 5.3 and 8c and
the code of src/tmlm/engine/core/tm_core.h (tm_target_feedback, tm_negative_feedback, tm_negative_update,
tm_pc_q32, tm_vote_add), for the configuration of the CP anchor bank: causal, V = 8000, T = 64,
neg_mode = binomial, Q = cfg.q = 1.0, v_slots = 64, MARGIN = T / 4 = 16, P_VETO = 1/16, B = 1024,
E_CAP = 256.

Notation. A training position with left context h; F(h) the clauses that fire there; w_jc the vote of
clause j for class c (0 when j has no vote slot for c); S_c(h) = sum over j in F(h) of w_jc, the integer
class score of spec section 4 step 4 (what predict_scores returns); clamp(S) = min(T, max(-T, S));
p = P(c | h), the probability that the next token after h is c under the training distribution. Every
fired clause receives one event per training example (spec 5.1), and the gates of an example use the
batch-start scores of that example (spec 4 step 5).

Rules that change w_jc for j in F(h), per training example with context h:
 (R+) y = c, probability p. Gate bern(Py) with Py = (T - clamp(S_c)) / (2T) (spec 4 step 5). If it passes:
      vote_add(j, c, +1) (a missing slot is created in a free slot or by evicting a slot with |w| <= 1),
      together with Type Ia or Type II on the literals, which does not change w.
 (R-) y != c, probability 1 - p. Class c is in the negative set drawn for (example, clause j) with
      probability Q / (V - 1) (spec 8c: independent Bernoulli(Q/(V-1)) per class c != y, truncated at
      NEG_K_MAX = 32). If drawn: gate bern(Pc) with Pc = (T + clamp(S_c)) / (2T) (tm_pc_q32, the error gate
      of the standard algorithm). If it passes and w_jc != 0: vote_add(j, c, -1) (w reaching 0 frees the
      slot). If w_jc == 0: no decrement; with probability 1/16, and only if S_c >= S_y - T/4, a vote -1 is
      created in a FREE slot (the veto of spec 5.3).

Expected change of S_c per training visit of h. With n+ the number of clauses of F(h) that can take the
+1 and n- the number that hold a nonzero vote for c:
    E[dS_c] = n+ p (T - clamp S_c) / (2T) - n- (1 - p) (Q / (V - 1)) (T + clamp S_c) / (2T).
Assumption A1 (same voters): n+ = n- = n. Then, with b = (1 - p) Q / (V - 1),
    E[dS_c] = n / (2T) [ p (T - clamp S_c) - b (T + clamp S_c) ] = n (p + b) / (2T) (S* - clamp S_c),
    S* = T (p - b) / (p + b).
The drift is linear and decreasing in clamp(S_c), so S* is the unique stable rest point. Since
(p - b) / (p + b) = (r - 1) / (r + 1) = tanh(ln(r) / 2) with r = p / b,
    S* = T tanh(x / 2),   x = ln(p / b) = ln(p (V - 1) / ((1 - p) Q)).
S* does not depend on n, on s or on the literal feedback: it is fixed by the ratio of the two gated
update rates. At V = 8000 and Q = 1, b is about 1.25e-4.

Consequence (the saturation account of docs/13 A4 and A5): tanh(x/2) >= 19/21 (0.9048) whenever p >= 20 b
(x >= ln 20), i.e. for every class with p above about 0.0025. All probable classes then sit within
10 percent of T, so the signed score cannot order them by probability. Hence the second registered
statistic: Spearman(S, ln p) at most 0.2 among the classes with p > 20 b.

Assumptions (each is a way the prediction can fail on the real bank):
 A1 Same voters. Broken by (a) the vote-slot limit: 64 slots per clause for 8000 classes, and a new
    class takes a slot only when one is free or holds |w| <= 1, so rare classes lack voters and S_c is
    pulled toward 0; (b) negatives decrement existing votes only: a clause whose vote for c reached 0
    stops receiving negatives for c, and S_c < 0 is reachable only through vetoes (probability 1/16,
    margin condition, free slot only). The prediction S* < 0 for p < b is therefore not reachable in
    general; the engine's floor for such classes is near 0.
 A2 Representability. A clause fires in many contexts with different p(c | h); its vote rests when the
    drift terms summed over those contexts cancel. The per-context law S_c(h) = S*(h) is a rest point of
    every clause at once only if the fired sets can represent S*(h) in all contexts together; otherwise
    the rest point is a compromise across contexts and S_c(h) deviates from S*(h).
 A3 Convergence. Four epochs of 135,246,414 examples bring the votes of the evaluated contexts to their
    rest point, and the literals and fired sets (Type I and II feedback, lazy decay, insertion, seeding,
    recycling) move slowly compared with the votes.
 A4 Clamping and overshoot. The gates read clamp(S), not S: for a linear drift the rest condition
    E[p (T - clamp S) - b (T + clamp S)] = 0 gives E[clamp S_c] = S* exactly. The registered statistic uses
    the unclamped S that predict_scores returns. All fired clauses step together (up to about 290 unit
    steps per example on this bank, and every example of a batch of 1024 reads the batch-start score),
    so S can overshoot T; above T only the negatives act, at the small rate b, so for p >> b the raw S/T
    can stay above 1 and above the prediction. The clamped statistic is reported beside the registered
    one as a diagnostic.
 A5 p is the training distribution P(c | h), estimated here by P_KN5(c | h) of the full-data KN-5 (the
    frozen cache); the evaluation positions are validation positions.
 A6 Negligible: Q32 and Q16 rounding of the gates; truncation at NEG_K_MAX; the event cap E_CAP = 256
    (it drops events without regard to class; the logged epoch-4 rate is 0.0 dropped events per
    example); the vote clamp at +-32767.
"""


# ------------------------------------------------------------------ bookkeeping
class Recorder:
    """Labelled metrics (measured / derived / unverified) flushed to the run dir of run_logged.py, or to
    <out>/metrics_unlogged.json outside it. In smoke mode every entry carries "smoke": true."""

    def __init__(self, smoke: bool, tag: str, out: Path | None = None) -> None:
        self.smoke = bool(smoke)
        self.tag = tag
        self.out = out
        self.metrics: dict = {}
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        print(f"[{self.tag} {time.strftime('%H:%M:%S', time.gmtime())} +{time.time() - self.t0:7.1f}s] {msg}", flush=True)

    def m(self, name: str, value, label: str = "measured", **extra) -> None:
        e = {"value": _plain(value), "label": label, **{k: _plain(v) for k, v in extra.items()}}
        if self.smoke:
            e["smoke"] = True
        self.metrics[name] = e

    def flush(self) -> None:
        if os.environ.get("TMLM_RUN_DIR"):
            from tmlm.common import runlog
            runlog.write_metrics(self.metrics, on_conflict="replace")
        elif self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            p = self.out / f"metrics_unlogged_{self.tag}.json"
            p.write_text(json.dumps(self.metrics, indent=1, default=str) + "\n")

    def datasets(self, hashes: dict) -> None:
        if os.environ.get("TMLM_RUN_DIR"):
            from tmlm.common import runlog
            runlog.write_dataset_hashes(hashes, on_conflict="replace")


def _plain(v):
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()] if v.dtype == object else v.tolist()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def run_dir_or(out: str | None) -> Path:
    d = os.environ.get("TMLM_RUN_DIR")
    if d:
        return Path(d)
    if not out:
        raise SystemExit("not under run_logged.py: give --out-dir (a scratch directory)")
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cgroup_memory() -> tuple[int, int]:
    cur = int((CG / "memory.current").read_text().strip())
    mx_txt = (CG / "memory.max").read_text().strip()
    return cur, (int(mx_txt) if mx_txt != "max" else 1 << 62)


def wait_for_memory(peak_gib: float, tag: str, rec: Recorder, poll_s: int = 120, max_wait_s: int = 6 * 3600) -> None:
    """Memory rule: start a step above 8 GiB only when memory.current + its peak < 0.80 memory.max (poll 2 minutes)."""
    waited = 0
    while True:
        cur, mx = cgroup_memory()
        if cur + peak_gib * GIB < 0.80 * mx:
            rec.log(f"memory before {tag}: current {cur / GIB:.1f} GiB + peak {peak_gib:.0f} GiB < 0.80 x {mx / GIB:.0f} GiB (waited {waited} s)")
            rec.m(f"memory.wait_s.{tag}", waited)
            return
        if waited >= max_wait_s:
            raise RuntimeError(f"memory never allowed {tag}")
        rec.log(f"memory before {tag}: current {cur / GIB:.1f} GiB + {peak_gib:.0f} GiB >= 0.80 x max; waiting {poll_s} s")
        time.sleep(poll_s)
        waited += poll_s


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


# ------------------------------------------------------------------ public helpers (tests/test_cvtm_vote_equilibrium.py)
def b_of(p: np.ndarray, Q: float, V: int) -> np.ndarray:
    """b = (1 - p) Q / (V - 1): the rate at which class c is drawn as a gated-in negative candidate per
    visit (before the error gate), docs/11 8c."""
    return (1.0 - np.asarray(p, np.float64)) * float(Q) / float(V - 1)


def x_of(p: np.ndarray, Q: float, V: int) -> np.ndarray:
    """x = ln(p / b), computed as ln p - ln b."""
    p = np.asarray(p, np.float64)
    return np.log(p) - np.log(b_of(p, Q, V))


def equilibrium(p: np.ndarray, Q: float, V: int, T: float) -> np.ndarray:
    """S* = T (p - b) / (p + b) (the rest point of the derivation; equal to T tanh(x/2))."""
    p = np.asarray(p, np.float64)
    b = b_of(p, Q, V)
    return float(T) * (p - b) / (p + b)


def bin_index(x: np.ndarray, width: float) -> np.ndarray:
    return np.floor(np.asarray(x, np.float64) / float(width)).astype(np.int64)


def weighted_r2(n: np.ndarray, ybar: np.ndarray, fbar: np.ndarray) -> float:
    """1 - sum n (ybar - fbar)^2 / sum n (ybar - mean)^2 over bins with n > 0; mean = sum n ybar / sum n.
    No fitted parameter: fbar is the theory curve."""
    n = np.asarray(n, np.float64)
    ok = n > 0
    n, ybar, fbar = n[ok], np.asarray(ybar, np.float64)[ok], np.asarray(fbar, np.float64)[ok]
    mean = float((n * ybar).sum() / n.sum())
    ss_tot = float((n * (ybar - mean) ** 2).sum())
    ss_res = float((n * (ybar - fbar) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def bin_sums(bins: np.ndarray, group: np.ndarray, values: list[np.ndarray], n_groups: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Per (group, bin) sums. Returns (bin_ids ascending, counts [G, K], [sums [G, K] per value])."""
    bins = np.asarray(bins, np.int64)
    ids, inv = np.unique(bins, return_inverse=True)
    k = ids.shape[0]
    key = np.asarray(group, np.int64) * k + inv
    cnt = np.bincount(key, minlength=n_groups * k).reshape(n_groups, k).astype(np.float64)
    sums = [np.bincount(key, weights=np.asarray(v, np.float64), minlength=n_groups * k).reshape(n_groups, k) for v in values]
    return ids, cnt, sums


class RankPrep:
    """Average ranks of one variable under integer multiplicity weights w (a resample of documents):
    the rank of an element equals its average rank in the data set expanded by the multiplicities, so
    Pearson(ranks, weights w) is the Spearman correlation of the expanded data (ties get average ranks)."""

    def __init__(self, v: np.ndarray) -> None:
        v = np.asarray(v)
        order = np.argsort(v, kind="stable")
        vs = v[order]
        new = np.empty(v.shape[0], dtype=bool)
        if v.shape[0]:
            new[0] = True
            new[1:] = vs[1:] != vs[:-1]
        gid_sorted = np.cumsum(new) - 1
        self.gid = np.empty(v.shape[0], dtype=np.int64)
        self.gid[order] = gid_sorted
        self.n_groups = int(gid_sorted[-1] + 1) if v.shape[0] else 0

    def ranks(self, w: np.ndarray) -> np.ndarray:
        W = np.bincount(self.gid, weights=w, minlength=self.n_groups)
        before = np.cumsum(W) - W
        return (before + (W + 1.0) / 2.0)[self.gid]


def weighted_pearson(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    sw = float(w.sum())
    if sw <= 0:
        return float("nan")
    ma = float((w * a).sum() / sw)
    mb = float((w * b).sum() / sw)
    da, db = a - ma, b - mb
    va, vb = float((w * da * da).sum()), float((w * db * db).sum())
    if va <= 0 or vb <= 0:
        return float("nan")
    return float((w * da * db).sum() / math.sqrt(va * vb))


def weighted_spearman(pa: RankPrep, pb: RankPrep, w: np.ndarray) -> float:
    w = np.asarray(w, np.float64)
    return weighted_pearson(pa.ranks(w), pb.ranks(w), w)


def doc_multiplicities(n_docs: int, resamples: int, seed) -> np.ndarray:
    """float64 [resamples, n_docs]: how often each document is drawn (with replacement), from
    tmlm.cvtm.protocol.bootstrap_index (the document-block bootstrap of the CVTM runs)."""
    from tmlm.cvtm.protocol import bootstrap_index
    idx = bootstrap_index(int(n_docs), int(resamples), seed)
    out = np.zeros((idx.shape[0], int(n_docs)), dtype=np.float64)
    for r in range(idx.shape[0]):
        out[r] = np.bincount(idx[r], minlength=int(n_docs))
    return out


@njit(cache=False)
def _avg_ranks_small(v):
    n = v.shape[0]
    r = np.empty(n, np.float64)
    for i in range(n):
        less = 0.0
        eq = 0.0
        for j in range(n):
            if v[j] < v[i]:
                less += 1.0
            elif v[j] == v[i]:
                eq += 1.0
        r[i] = less + (eq + 1.0) / 2.0
    return r


@njit(cache=False)
def k_within_row_spearman(S, lnp, L):
    """Spearman of S[i, :L[i]] and lnp[i, :L[i]] per row; NaN when L < 3 or a column is constant."""
    n = S.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        m = L[i]
        if m < 3:
            continue
        a = _avg_ranks_small(S[i, :m].astype(np.float64))
        b = _avg_ranks_small(lnp[i, :m])
        ma = a.mean()
        mb = b.mean()
        num = 0.0
        va = 0.0
        vb = 0.0
        for k in range(m):
            num += (a[k] - ma) * (b[k] - mb)
            va += (a[k] - ma) ** 2
            vb += (b[k] - mb) ** 2
        if va > 0 and vb > 0:
            out[i] = num / math.sqrt(va * vb)
    return out


def percentile_ci(stats: np.ndarray) -> list:
    s = np.asarray(stats, np.float64)
    s = s[np.isfinite(s)]
    if s.shape[0] == 0:
        return [None, None]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def verdict_h96(r2: float, rho: float, r2_min: float, rho_max: float) -> str:
    return "MET" if (r2 >= r2_min and rho <= rho_max) else "NOT MET"


def verdict_g1t1(r2: float, r2_min: float) -> str:
    return "OPEN" if r2 >= r2_min else "CLOSED"


# ------------------------------------------------------------------ gpu phase (child process)
def load_bank(cfg: dict, rec: Recorder) -> tuple:
    from tmlm.common import hashing
    from tmlm.decode.scorer import TrainedModel
    path = ROOT / cfg["bank"]["model"]
    cpm = json.loads((ROOT / cfg["bank"]["metrics"]).read_text())
    t0 = time.time()
    model = TrainedModel.load(path)
    want = cpm["epoch4.checkpoint"]["value"]["state_hash"]
    rec.m("bank.state_hash", model.state_hash, expected=want, source=cfg["bank"]["metrics"])
    if model.state_hash != want:
        raise SystemExit(f"bank state hash {model.state_hash} differs from the logged epoch-4 hash {want}")
    ec = model.cfg
    want_cfg = cfg["bank"]["engine_must_be"]
    for k, v in want_cfg.items():
        got = getattr(ec, k)
        if (float(got) if isinstance(v, float) else got) != v:
            raise SystemExit(f"bank config {k} = {got}, expected {v}")
    rec.m("bank.engine_config", ec.to_dict())
    rec.m("bank.file_sha256", hashing.sha256_file(path))
    rec.log(f"bank loaded and verified in {time.time() - t0:.1f} s: state hash {model.state_hash[:16]}..., T {ec.T}, q {ec.q}, V {ec.vocab}")
    return model, cpm


def load_validation_and_kn(cfg: dict, rec: Recorder, model) -> dict:
    from cp_causal_perplexity import contiguous
    from tmlm.common import hashing
    from tmlm.cvtm import cache as CC
    from tmlm.cvtm import protocol as P
    from tmlm.data.corpus_store import load_split
    V = int(model.cfg.vocab)
    val = contiguous(load_split(CORPUS, V, "validation"))
    for name in ("kind_map", "stop", "sym_freq"):
        if not np.array_equal(np.asarray(getattr(val, name)), np.asarray(getattr(model.tables, name))):
            raise SystemExit(f"validation store {name} differs from the bank's static tables")
    hashes = P.dataset_hashes(("train", "validation"), CORPUS, V)
    cache_dir = ROOT / cfg["kn"]["cache"]
    meta = CC.read_meta(cache_dir)
    names = list(cfg["kn"]["arrays"])
    for name in names:
        rel = f"validation/{name}.npy"
        got = hashing.sha256_file(cache_dir / rel)
        if got != meta["files"][rel]["sha256"]:
            raise SystemExit(f"cache file {rel}: sha256 differs from cache_meta.json")
        hashes[f"cvtm_cache_validation_{name}"] = got
    arr = CC.load_arrays(cache_dir, "validation", names, mmap=True)
    tok = np.asarray(val.tok).astype(np.int64)
    ds = np.asarray(val.doc_start).astype(np.int64)
    if not (np.array_equal(np.asarray(arr["gold"]).astype(np.int64), tok) and np.array_equal(np.asarray(arr["doc_start"]).astype(np.int64), ds)):
        raise SystemExit("cache validation stream differs from the validation store")
    rec.datasets(hashes)
    rec.m("data.dataset_hashes", hashes)
    return {"val": val, "tok": tok, "doc_start": ds, "arrays": arr, "cache_dir": cache_dir}


def select_positions(n_val: int, count: int, seed: list) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(s) for s in seed]))
    return np.sort(rng.choice(n_val, size=int(count), replace=False)).astype(np.int64)


def run_gpu(cfg: dict, args) -> int:
    out = run_dir_or(args.out_dir)
    rec = Recorder(args.smoke, "h96-gpu", out)
    import cupy as cp
    from tmlm.common import locks
    from tmlm.engine.cuda.cuda_engine import CudaEngine
    comp = cfg["compute"]
    pool = cp.get_default_memory_pool()
    pool.set_limit(size=int(float(comp["gpu_pool_limit_gib"]) * GIB))
    model, cpm = load_bank(cfg, rec)
    D = load_validation_and_kn(cfg, rec, model)
    V, T = int(model.cfg.vocab), int(model.cfg.T)
    K = int(cfg["kn"]["top"])
    tok, ds = D["tok"], D["doc_start"]
    n_val = int(tok.shape[0])
    n_sel = int(args.n_positions or cfg["positions"]["count"])
    if args.n_positions and not args.smoke:
        raise SystemExit("--n-positions is a smoke-only override")
    sel = select_positions(n_val, n_sel, cfg["positions"]["seed"])
    kn_ids = np.asarray(D["arrays"]["kn_top_ids"][sel, :K]).astype(np.int64)
    kn_p = np.asarray(D["arrays"]["kn_top_p"][sel, :K]).astype(np.float64)
    rec.m("positions.count", n_sel)
    rec.m("positions.first_ten", sel[:10])
    rec.log(f"validation {n_val} positions; selected {n_sel}; engine V {V}, T {T}, C {model.cfg.n_clauses}")

    epoch = int(cfg["engine"]["epoch_argument"])
    t0 = time.time()
    eng = CudaEngine(model.cfg, D["val"])
    eng.load_state(model.state)
    rec.m("gpu.engine_build_s", time.time() - t0)
    rec.flush()

    # ---- pass 1: predict_scores over every validation position (top-1 check) and S at the KN classes
    chunk = int(comp["score_chunk"])
    S = np.zeros((n_sel, K), np.int32)
    S_gold = np.zeros(n_sel, np.int32)
    S_max = np.zeros(n_sel, np.int32)
    hits = 0
    t0 = time.time()
    for lo in range(0, n_val, chunk):
        locks.yield_gpu_if_requested()
        hi = min(n_val, lo + chunk)
        sc = eng.predict_scores(epoch, np.arange(lo, hi, dtype=np.uint32))
        hits += int((sc.argmax(axis=1) == tok[lo:hi]).sum())
        a, b = np.searchsorted(sel, lo), np.searchsorted(sel, hi)
        if b > a:
            rows = sel[a:b] - lo
            S[a:b] = sc[rows[:, None], kn_ids[a:b]]
            S_gold[a:b] = sc[rows, tok[sel[a:b]]]
            S_max[a:b] = sc[rows].max(axis=1)
        del sc
    top1 = hits / n_val
    logged = float(cpm["epoch4.validation.causal_top1_raw_scores"]["value"])
    rec.m("check.signed_raw_top1_reproduced", top1, expected=logged, source=cfg["bank"]["metrics"],
          note="argmax of predict_scores (ties to the lowest id) over every validation position, CUDA engine")
    rec.m("gpu.score_pass_s", time.time() - t0)
    rec.log(f"pass 1: raw top-1 {top1!r} (logged {logged!r}) in {time.time() - t0:.1f} s")
    rec.flush()
    if top1 != logged:
        rec.log("PIPELINE CHECK FAILED: the bank's raw top-1 is not reproduced; stop")
        return 3

    # ---- pass 2 (diagnostics): fired clauses and voters per (position, KN class); the forward's scores must equal pass 1
    dchunk = int(comp["diag_chunk"])
    n_fired = np.zeros(n_sel, np.int32)
    n_posv = np.zeros((n_sel, K), np.int16)
    n_negv = np.zeros((n_sel, K), np.int16)
    vc_d, vw_d = eng.d["vote_cls"], eng.d["vote_w"]
    mism = 0
    t0 = time.time()
    for lo in range(0, n_sel, dchunk):
        locks.yield_gpu_if_requested()
        hi = min(n_sel, lo + dchunk)
        m = hi - lo
        eng._sync_state_struct()
        fw = eng._forward(epoch, cp.asarray(sel[lo:hi].astype(np.uint32)), None, 0)
        eng._marks = []
        fe = fw.fired_ex.astype(cp.int64)
        fj = fw.fired_j.astype(cp.int64)
        n_fired[lo:hi] = cp.bincount(fe, minlength=m).get()
        ids_d = cp.asarray(kn_ids[lo:hi])
        rows_d = cp.arange(m)[:, None]
        got = fw.scores[rows_d, ids_d].get()
        mism += int((got != S[lo:hi]).any(axis=1).sum())
        vc = vc_d[fj].astype(cp.int64)
        vw = vw_d[fj]
        key = fe[:, None] * V + vc
        for sign, dest in ((1, n_posv), (-1, n_negv)):
            sel_k = key[(vw > 0) if sign > 0 else (vw < 0)]
            cnt = cp.bincount(sel_k, minlength=m * V).reshape(m, V)
            dest[lo:hi] = cnt[rows_d, ids_d].astype(cp.int16).get()
            del cnt, sel_k
        del fw, fe, fj, vc, vw, key, ids_d
        pool.free_all_blocks()
    rec.m("check.forward_scores_equal_predict_scores_mismatch_rows", mism, note="selected positions, the 50 KN classes")
    rec.m("gpu.diag_pass_s", time.time() - t0)
    rec.m("gpu.pool_limit_gib", float(comp["gpu_pool_limit_gib"]))
    rec.m("gpu.host_peak_rss_gib", peak_rss_gib())
    rec.log(f"pass 2: fired and voter counts in {time.time() - t0:.1f} s, score mismatches {mism}")
    if mism:
        rec.flush()
        rec.log("PIPELINE CHECK FAILED: forward scores differ from predict_scores; stop")
        return 3
    doc_of = (np.searchsorted(ds, sel, side="right") - 1).astype(np.int32)
    np.savez_compressed(out / cfg["outputs"]["pairs_npz"], positions=sel, doc=doc_of, kn_ids=kn_ids.astype(np.uint16),
                        kn_p=kn_p, S=S, S_gold=S_gold, S_max=S_max, gold=tok[sel].astype(np.uint16), n_fired=n_fired,
                        n_pos_voters=n_posv, n_neg_voters=n_negv, T=np.int64(T), V=np.int64(V), Q=np.float64(model.cfg.q))
    rec.m("gpu.exit", "ok")
    rec.flush()
    return 0


# ------------------------------------------------------------------ KN cross-check (CPU, parent)
def kn_crosscheck(cfg: dict, rec: Recorder, pairs: dict, tok: np.ndarray, ds: np.ndarray, n_check: int) -> bool:
    """The cached top-50 against KNModel.dist of kn5_all.npz (an independent code path: the cache used the
    vote form of docs/11 11.2): probabilities at the cached ids within 1e-9 relative, and no class outside
    the list more probable than the list's last entry by more than 1e-9 relative (pass rule); the row sums
    of KNModel.dist are recorded."""
    from tmlm.cvtm import cache as CC
    wait_for_memory(float(cfg["kn"]["cross_check"]["peak_gib"]), "kn_crosscheck", rec)
    t0 = time.time()
    kn = CC.load_kn_variant(ROOT / cfg["kn"]["cache"], "all", verify=True)
    rec.m("kn.all.load_s", time.time() - t0)
    rng = np.random.default_rng(np.random.SeedSequence([int(s) for s in cfg["kn"]["cross_check"]["seed"]]))
    n = pairs["positions"].shape[0]
    rows = np.sort(rng.choice(n, size=min(int(n_check), n), replace=False))
    pos = pairs["positions"][rows]
    ctx, valid = kn.contexts(tok, ds, pos)
    dist = kn.dist(ctx, valid)
    ids = pairs["kn_ids"][rows].astype(np.int64)
    pc = pairs["kn_p"][rows]
    pd = np.take_along_axis(dist, ids, axis=1)
    rel = np.abs(pd - pc) / np.maximum(pc, 1e-300)
    last = pc[:, -1]
    mask = np.ones_like(dist, dtype=bool)
    np.put_along_axis(mask, ids, False, axis=1)
    outside_max = np.where(mask, dist, -1.0).max(axis=1)
    excess = (outside_max - last) / np.maximum(last, 1e-300)
    sums = dist.sum(axis=1)
    ok = bool(rel.max() <= 1e-9 and excess.max() <= 1e-9)
    rec.m("check.kn_top50_vs_KNModel_dist.max_rel_diff", float(rel.max()))
    rec.m("check.kn_top50_vs_KNModel_dist.max_outside_excess_rel", float(excess.max()))
    rec.m("check.kn_top50_vs_KNModel_dist.max_abs_row_sum_minus_1", float(np.abs(sums - 1.0).max()),
          note="recorded, not part of the pass rule")
    rec.m("check.kn_top50_vs_KNModel_dist.positions", int(rows.shape[0]))
    rec.m("check.kn_top50_vs_KNModel_dist.passed", ok)
    rec.log(f"KN cross-check on {rows.shape[0]} positions: max rel diff {rel.max():.3g}, outside excess {excess.max():.3g}, passed {ok}")
    del kn, dist
    return ok


# ------------------------------------------------------------------ analysis (CPU, parent)
def analyse(cfg: dict, rec: Recorder, pairs: dict, out: Path, smoke: bool) -> dict:
    st = cfg["statistics"]
    T, V, Q = int(pairs["T"]), int(pairs["V"]), float(pairs["Q"])
    assert (T, V, Q) == (int(st["T"]), int(st["V"]), float(st["Q"])), "config and bank disagree on T, V or Q"
    S = pairs["S"].astype(np.int64)
    p = pairs["kn_p"].astype(np.float64)
    n, K = S.shape
    if not (np.all(p > 0) and np.all(p < 1)):
        raise SystemExit("a KN probability is outside (0, 1)")
    if not np.all(np.diff(p, axis=1) <= 0):
        raise SystemExit("the KN top list is not in descending probability")
    b = b_of(p, Q, V)
    x = x_of(p, Q, V)
    y = S / float(T)
    yc = np.clip(S, -T, T) / float(T)
    f = np.tanh(x / 2.0)
    Sstar = equilibrium(p, Q, V, T)
    assert np.allclose(Sstar / T, f, rtol=0, atol=1e-12), "(p - b)/(p + b) differs from tanh(x/2)"
    doc = np.repeat(pairs["doc"].astype(np.int64), K)
    n_docs = int(pairs["doc"].max()) + 1
    width = float(st["bins"]["width"])
    kb = bin_index(x.ravel(), width)

    # ---- registered statistic 1: weighted R^2 of the bin means of S/T against the theory
    ids, cnt, (sy, sf, syc, sx) = bin_sums(kb, doc, [y.ravel(), f.ravel(), yc.ravel(), x.ravel()], n_docs)
    N = cnt.sum(axis=0)
    ybar, fbar, ycbar, xbar = sy.sum(0) / N, sf.sum(0) / N, syc.sum(0) / N, sx.sum(0) / N
    r2 = weighted_r2(N, ybar, fbar)
    r2_tanh_of_mean = weighted_r2(N, ybar, np.tanh(xbar / 2.0))
    r2_clamped = weighted_r2(N, ycbar, fbar)
    # pair level (no binning), diagnostic
    yy = y.ravel()
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2_pairs = 1.0 - float(((yy - f.ravel()) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    # ---- registered statistic 2: pooled Spearman(S, ln p) over pairs with p > 20 b
    hi_mask = p > float(st["spearman"]["p_over_b_min"]) * b
    s_hi = S[hi_mask]
    lp_hi = np.log(p[hi_mask])
    doc_hi = np.repeat(pairs["doc"].astype(np.int64)[:, None], K, axis=1)[hi_mask]
    pa, pb = RankPrep(s_hi), RankPrep(lp_hi)
    ones = np.ones(s_hi.shape[0])
    rho = weighted_spearman(pa, pb, ones)
    from scipy.stats import spearmanr
    rho_scipy = float(spearmanr(s_hi, lp_hi).statistic) if s_hi.shape[0] > 2 else float("nan")
    assert abs(rho - rho_scipy) < 1e-9 or not np.isfinite(rho_scipy), "weighted-rank Spearman differs from scipy"

    # ---- bootstrap over validation documents
    bs = st["bootstrap"]
    mult = doc_multiplicities(n_docs, int(bs["resamples"]), bs["seed"])
    t0 = time.time()
    Nb = mult @ cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        yb = (mult @ sy) / Nb
        fb = (mult @ sf) / Nb
        ycb = (mult @ syc) / Nb
    r2_bs = np.array([weighted_r2(Nb[r], np.nan_to_num(yb[r]), np.nan_to_num(fb[r])) for r in range(mult.shape[0])])
    r2c_bs = np.array([weighted_r2(Nb[r], np.nan_to_num(ycb[r]), np.nan_to_num(fb[r])) for r in range(mult.shape[0])])
    rho_bs = np.empty(mult.shape[0])
    for r in range(mult.shape[0]):
        rho_bs[r] = weighted_spearman(pa, pb, mult[r][doc_hi])
    rec.m("bootstrap.wall_s", time.time() - t0)

    # ---- secondary diagnostics (not registered)
    L = hi_mask.sum(axis=1).astype(np.int64)             # the classes with p > 20 b form a prefix of the top list
    assert np.all(hi_mask == (np.arange(K)[None, :] < L[:, None])), "p > 20 b is not a prefix of the descending list"
    rho_within = k_within_row_spearman(S, np.log(p), L)
    ok_w = np.isfinite(rho_within)
    rho_all = weighted_spearman(RankPrep(S.ravel()), RankPrep(np.log(p).ravel()), np.ones(S.size))
    lo_mask = ~hi_mask
    rho_lo = weighted_spearman(RankPrep(S[lo_mask]), RankPrep(np.log(p[lo_mask])), np.ones(int(lo_mask.sum()))) if lo_mask.sum() > 2 else float("nan")
    npv = pairs["n_pos_voters"].astype(np.int64)
    nnv = pairs["n_neg_voters"].astype(np.int64)
    # the fitted-shift diagnostic: S/T against tanh((x - ln kappa)/2), kappa on a grid, bins weighted as above (post hoc)
    grid = np.arange(-8.0, 8.0001, 0.05)
    r2_shift = np.array([weighted_r2(N, ybar, np.tanh((xbar - g) / 2.0)) for g in grid])
    g_best = float(grid[int(np.nanargmax(r2_shift))])
    by_rank = {"mean_S_over_T": y.mean(axis=0), "mean_theory": f.mean(axis=0), "mean_p": p.mean(axis=0),
               "frac_S_ge_T": (S >= T).mean(axis=0), "frac_S_eq_0": (S == 0).mean(axis=0)}

    verdict = verdict_h96(r2, rho, float(st["threshold"]["r2_min"]), float(st["threshold"]["spearman_max"]))
    gate = verdict_g1t1(r2, float(st["threshold"]["r2_min"]))
    res = {"r2": r2, "rho": rho, "r2_ci": percentile_ci(r2_bs), "rho_ci": percentile_ci(rho_bs), "verdict": verdict, "gate": gate}

    # ---- metrics
    lab = "measured"
    rec.m("h96.n_pairs", int(S.size))
    rec.m("h96.n_positions", int(n))
    rec.m("h96.n_bins", int(ids.shape[0]))
    rec.m("h96.weighted_r2", r2, label=lab, ci95=res["r2_ci"], ci_label="derived: document-block bootstrap, 2,000 resamples",
          threshold_min=float(st["threshold"]["r2_min"]), note="registered: bin means of raw S/T vs the bin means of tanh(x/2); weights = pairs per bin")
    rec.m("h96.spearman_S_lnp_p_gt_20b", rho, label=lab, ci95=res["rho_ci"], ci_label="derived: document-block bootstrap, 2,000 resamples",
          threshold_max=float(st["threshold"]["spearman_max"]), n_pairs=int(s_hi.shape[0]),
          note="registered: pooled over every (position, class) pair with p > 20 b; average ranks for ties")
    rec.m("h96.spearman_scipy_crosscheck", rho_scipy, label="measured")
    rec.m("h96.verdict", verdict, label="derived", rule="MET iff weighted R^2 >= 0.8 AND Spearman <= 0.2 (docs/13 A4)")
    rec.m("g1_t1.verdict", gate, label="derived", rule="OPEN iff weighted R^2 >= 0.8 (docs/13 A4 gates)")
    rec.m("a5.row", "D10" if verdict == "MET" else "D11", label="derived", rule="docs/13 A5: D10 H9.6 met, D11 not met")
    rec.m("secondary.weighted_r2_theory_tanh_of_bin_mean_x", r2_tanh_of_mean, note="not registered: theory per bin = tanh(mean x / 2)")
    rec.m("secondary.weighted_r2_clamped_S", r2_clamped, ci95=percentile_ci(r2c_bs), note="not registered: clamp(S, -T, T)/T (the quantity the gates read, assumption A4)")
    rec.m("secondary.r2_pairs_unbinned", r2_pairs, note="not registered: every pair, raw S/T vs tanh(x/2), no binning")
    rec.m("secondary.spearman_all_top50_pairs", rho_all, note="not registered")
    rec.m("secondary.spearman_pairs_p_le_20b", rho_lo, note="not registered")
    rec.m("secondary.within_position_spearman_mean", float(np.nanmean(rho_within)) if ok_w.any() else None,
          n_positions=int(ok_w.sum()), median=float(np.nanmedian(rho_within)) if ok_w.any() else None,
          note="not registered: per position over its classes with p > 20 b (>= 3 classes, non-constant S), then averaged")
    rec.m("secondary.fitted_shift_ln_kappa", g_best, r2_at_best=float(np.nanmax(r2_shift)),
          note="post hoc, exploratory: S/T vs tanh((x - ln kappa)/2), ln kappa on a 0.05 grid in [-8, 8]; kappa multiplies b")
    rec.m("secondary.mean_S_over_T_all_pairs", float(y.mean()))
    rec.m("secondary.mean_theory_all_pairs", float(f.mean()))
    rec.m("secondary.frac_pairs_S_ge_T", float((S >= T).mean()))
    rec.m("secondary.frac_pairs_S_gt_T", float((S > T).mean()))
    rec.m("secondary.frac_pairs_S_eq_0", float((S == 0).mean()))
    rec.m("secondary.frac_pairs_S_lt_0", float((S < 0).mean()))
    rec.m("secondary.frac_pairs_p_gt_20b", float(hi_mask.mean()))
    rec.m("secondary.frac_pairs_no_positive_voter", float((npv == 0).mean()))
    rec.m("secondary.frac_pairs_p_gt_20b_no_positive_voter", float((npv[hi_mask] == 0).mean()) if hi_mask.any() else None)
    rec.m("secondary.mean_positive_voters_p_gt_20b", float(npv[hi_mask].mean()) if hi_mask.any() else None)
    rec.m("secondary.mean_negative_voters_all", float(nnv.mean()))
    rec.m("secondary.mean_fired_per_position", float(pairs["n_fired"].mean()))
    rec.m("secondary.gold_in_top50_frac", float((pairs["kn_ids"].astype(np.int64) == pairs["gold"].astype(np.int64)[:, None]).any(axis=1).mean()))
    rec.m("secondary.mean_S_gold_over_T", float(pairs["S_gold"].mean() / T))
    rec.m("secondary.mean_S_max_over_T", float(pairs["S_max"].mean() / T))
    rec.m("secondary.by_rank", {k: v.tolist() for k, v in by_rank.items()}, note="not registered: column r = the r-th most probable KN class")

    # ---- plot data
    table = []
    for k in range(ids.shape[0]):
        in_bin = kb == ids[k]
        table.append({"bin": int(ids[k]), "x_lo": float(ids[k] * width), "x_hi": float((ids[k] + 1) * width), "n_pairs": int(N[k]),
                      "mean_x": float(xbar[k]), "mean_S_over_T": float(ybar[k]), "mean_theory_tanh_x_half": float(fbar[k]),
                      "tanh_mean_x_half": float(np.tanh(xbar[k] / 2.0)), "mean_clamped_S_over_T": float(ycbar[k]),
                      "sd_S_over_T": float(yy[in_bin].std()), "frac_S_ge_T": float((S.ravel()[in_bin] >= T).mean()),
                      "frac_S_eq_0": float((S.ravel()[in_bin] == 0).mean()),
                      "mean_positive_voters": float(npv.ravel()[in_bin].mean())})
    pd_path = out / cfg["outputs"]["plot_data_json"]
    pd_path.write_text(json.dumps({"description": "H9.6 plot data: bins of x = ln(p_KN / b), width 0.25; weights = pairs per bin",
                                   "T": T, "V": V, "Q": Q, "bins": table,
                                   "bootstrap": {"weighted_r2": r2_bs.tolist(), "spearman": rho_bs.tolist()}}, indent=1) + "\n")
    with open(out / cfg["outputs"]["plot_data_csv"], "w") as fh:
        cols = list(table[0].keys())
        fh.write(",".join(cols) + "\n")
        for row in table:
            fh.write(",".join(repr(row[c]) for c in cols) + "\n")
    rec.m("outputs.plot_data", [str(pd_path.name), cfg["outputs"]["plot_data_csv"]])
    try:
        plot(out / cfg["outputs"]["plot_png"], table, res, by_rank, T)
        rec.m("outputs.plot_png", cfg["outputs"]["plot_png"])
    except Exception as exc:  # noqa: BLE001  (the plot is a convenience; the data files are the record)
        rec.m("outputs.plot_png_error", repr(exc), label="unverified")
    return res


def plot(path: Path, table: list, res: dict, by_rank: dict, T: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.array([r["mean_x"] for r in table])
    ym = np.array([r["mean_S_over_T"] for r in table])
    yc = np.array([r["mean_clamped_S_over_T"] for r in table])
    nn = np.array([r["n_pairs"] for r in table])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    xx = np.linspace(xs.min() - 0.5, xs.max() + 0.5, 400)
    ax[0].plot(xx, np.tanh(xx / 2), "k-", lw=1.2, label="theory tanh(x/2)")
    ax[0].scatter(xs, ym, s=8 + 40 * nn / nn.max(), c="C0", label="mean S/T (registered)")
    ax[0].plot(xs, yc, "C1.", ms=4, label="mean clamp(S)/T (diagnostic)")
    ax[0].axvline(math.log(20), color="grey", ls=":", lw=1, label="p = 20 b")
    ax[0].set_xlabel("x = ln(p_KN / b)")
    ax[0].set_ylabel("S / T")
    ax[0].set_title(f"H9.6: weighted R^2 {res['r2']:.3f}, Spearman {res['rho']:.3f}: {res['verdict']}")
    ax[0].legend(fontsize=8)
    ax[1].bar(xs, nn, width=0.22)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("pairs per bin")
    r = np.arange(1, len(by_rank["mean_S_over_T"]) + 1)
    ax[2].plot(r, by_rank["mean_S_over_T"], "C0.-", label="mean S/T")
    ax[2].plot(r, by_rank["mean_theory"], "k-", label="mean tanh(x/2)")
    ax[2].set_xlabel("rank of the class under KN-5")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_main(cfg: dict, args) -> int:
    out = run_dir_or(args.out_dir)
    rec = Recorder(args.smoke, "h96", out)
    rec.log("H9.6 vote equilibrium (docs/13 A4) and gate G1 T1; the derivation follows")
    for line in DERIVATION.strip("\n").splitlines():
        print(line, flush=True)
    rec.m("derivation", DERIVATION.strip(), label="derived", note="written before any class score was computed; printed into log.txt")
    rec.m("setup.config", cfg)
    rec.flush()

    # GPU phase in a child that holds the shared GPU lock only for its own lifetime
    pairs_path = out / cfg["outputs"]["pairs_npz"]
    if args.reuse_pairs:
        if not args.smoke:
            raise SystemExit("--reuse-pairs is a smoke-only override")
    else:
        cmd = [str(ROOT / "scripts" / "gpu_run.sh"), "shared", "--", sys.executable, "-u", str(Path(__file__).resolve()),
               "--config", str(args.config), "--phase", "gpu"]
        if args.out_dir:
            cmd += ["--out-dir", str(out)]
        if args.smoke:
            cmd += ["--smoke"]
        if args.n_positions:
            cmd += ["--n-positions", str(args.n_positions)]
        rec.log("GPU phase: " + " ".join(cmd))
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        rec.m("gpu_phase.exit_code", rc)
        rec.m("gpu_phase.wall_s_including_lock_wait", time.time() - t0)
        rec.flush()
        if rc != 0:
            rec.log(f"GPU phase failed with exit code {rc}: stop")
            return rc if rc > 0 else 1
    with np.load(pairs_path) as z:
        pairs = {k: z[k] for k in z.files}
    from tmlm.common import hashing
    rec.m("outputs.pairs_npz_sha256", hashing.sha256_file(pairs_path))

    # the KN cross-check needs the validation stream
    from tmlm.data.corpus_store import load_split
    val = load_split(CORPUS, int(pairs["V"]), "validation")
    tok = np.asarray(val.tok).astype(np.int64)
    ds = np.asarray(val.doc_start).astype(np.int64)
    if not np.array_equal(tok[pairs["positions"]], pairs["gold"].astype(np.int64)):
        raise SystemExit("pairs file does not match the validation stream")
    n_check = int(args.kn_check_positions or cfg["kn"]["cross_check"]["positions"])
    if args.kn_check_positions and not args.smoke:
        raise SystemExit("--kn-check-positions is a smoke-only override")
    ok = kn_crosscheck(cfg, rec, pairs, tok, ds, n_check)
    rec.flush()
    if not ok:
        rec.log("PIPELINE CHECK FAILED: the cached KN top-50 differs from KNModel.dist; stop")
        return 3

    t0 = time.time()
    res = analyse(cfg, rec, pairs, out, args.smoke)
    rec.m("analysis.wall_s", time.time() - t0)
    rec.m("host_peak_rss_gib", peak_rss_gib())
    rec.flush()
    if args.smoke:
        rec.log("smoke: statistics computed and written to the smoke metrics file; not printed")
        return 0
    st = cfg["statistics"]["threshold"]
    rec.log(f"H9.6 weighted R^2 = {res['r2']:.4f} (threshold >= {st['r2_min']}), 95% CI {res['r2_ci']} [derived, document bootstrap]")
    rec.log(f"H9.6 Spearman(S, ln p | p > 20 b) = {res['rho']:.4f} (threshold <= {st['spearman_max']}), 95% CI {res['rho_ci']} [derived]")
    rec.log(f"H9.6 verdict: {res['verdict']}")
    rec.log(f"G1 T1 verdict: {res['gate']} (weighted R^2 {res['r2']:.4f} vs 0.8)")
    rec.log(f"docs/13 A5 row: {'D10' if res['verdict'] == 'MET' else 'D11'}")
    return 0


# ------------------------------------------------------------------ planted check (exploratory)
def zipf_p(V: int, a: float) -> np.ndarray:
    w = 1.0 / np.arange(1, V + 1, dtype=np.float64) ** float(a)
    return w / w.sum()


def planted_one(pc: dict, V: int, B: int, seed: int) -> dict:
    """One planted run on the CPU engine: labels independent of the random features, so every context has
    the same class distribution p (A2 and A5 hold exactly; A1 holds when v_slots >= V)."""
    from tmlm.engine.api import TabularData
    from tmlm.engine.config import EngineConfig
    from tmlm.engine.cpu.cpu_engine import CpuEngine
    F, N, n_test = int(pc["n_features"]), int(pc["n_train"]), int(pc["n_test"])
    p = zipf_p(V, float(pc["zipf_a"]))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20260924, 96, V, B]))
    x = (rng.random((N, F)) < 0.5).astype(np.uint8)
    y = rng.choice(V, size=N, p=p).astype(np.uint16)
    xt = (rng.random((n_test, F)) < 0.5).astype(np.uint8)
    yt = rng.choice(V, size=n_test, p=p).astype(np.uint16)
    ecfg = EngineConfig(mode="tabular", vocab=V, n_features=F, n_clauses=int(pc["n_clauses"]), seed=int(seed), T=int(pc["T"]),
                        batch=int(B), q=float(pc["Q"]), neg_mode="binomial", n_bag_k=0)
    eng = CpuEngine(ecfg, tabular=TabularData(x=x, y=y))
    probe = CpuEngine(ecfg, tabular=TabularData(x=xt, y=yt))
    epochs = int(pc["epochs"])
    nb = eng.n_batches()
    snaps = set(np.linspace(0, nb - 1, int(pc["snapshots_last_epoch"])).round().astype(int).tolist())
    rows = np.arange(n_test, dtype=np.uint32)
    acc = np.zeros(V)
    acc_c = np.zeros(V)
    ns = 0
    t0 = time.time()
    diag_tot: dict = {}
    for e in range(epochs):
        for bi in range(nb):
            d = eng.train_batch(e, bi)
            for k, v in d.items():
                diag_tot[k] = diag_tot.get(k, 0) + v
            if e == epochs - 1 and bi in snaps:
                probe.load_state(eng.state())
                s = probe.predict_scores(0, rows).astype(np.float64)
                acc += s.mean(axis=0)
                acc_c += np.clip(s, -ecfg.T, ecfg.T).mean(axis=0)
                ns += 1
    T = float(ecfg.T)
    mean_S = acc / ns
    mean_Sc = acc_c / ns
    xq = x_of(p, float(pc["Q"]), V)
    f = np.tanh(xq / 2.0)
    y_ = mean_S / T
    yc = mean_Sc / T

    def r2(obs):
        ss_tot = float(((obs - obs.mean()) ** 2).sum())
        return 1.0 - float(((obs - f) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    from scipy.stats import spearmanr
    hi = p > 20.0 * b_of(p, float(pc["Q"]), V)
    st = eng.state()
    return {"V": V, "B": B, "seed": seed, "wall_s": time.time() - t0, "p": p.tolist(), "x": xq.tolist(), "theory": f.tolist(),
            "mean_S_over_T": y_.tolist(), "mean_clamped_S_over_T": yc.tolist(), "r2_raw": r2(y_), "r2_clamped": r2(yc),
            "max_abs_dev_raw": float(np.abs(y_ - f).max()), "max_abs_dev_clamped": float(np.abs(yc - f).max()),
            "spearman_p_gt_20b": float(spearmanr(mean_S[hi], np.log(p[hi])).statistic) if hi.sum() > 2 else None,
            "n_classes_p_gt_20b": int(hi.sum()), "snapshots": ns, "alive_clauses": int(st["alive"].sum()),
            "diag_per_example": {k: v / max(1, diag_tot.get("n_examples", 1)) for k, v in diag_tot.items()}}


def run_planted(cfg: dict, args) -> int:
    out = run_dir_or(args.out_dir)
    rec = Recorder(args.smoke, "h96-planted", out)
    pc = cfg["planted"]
    rec.log("EXPLORATORY planted check of the H9.6 derivation on the CPU engine (supports no registered verdict)")
    for line in DERIVATION.strip("\n").splitlines():
        print(line, flush=True)
    rec.m("setup.config", cfg, label="measured")
    runs = []
    seeds = pc["seeds"] if not args.smoke else pc["seeds"][:1]
    for V in pc["vocab"]:
        for B in pc["batch"]:
            for sd in seeds:
                r = planted_one(pc if not args.smoke else {**pc, "n_train": 4000, "epochs": 2}, int(V), int(B), int(sd))
                runs.append(r)
                rec.log(f"planted V {V} B {B} seed {sd}: R^2 raw {r['r2_raw']:.4f}, clamped {r['r2_clamped']:.4f}, max |dev| raw "
                        f"{r['max_abs_dev_raw']:.3f}, clamped {r['max_abs_dev_clamped']:.3f}, Spearman(p > 20b) {r['spearman_p_gt_20b']}, "
                        f"alive {r['alive_clauses']}, {r['wall_s']:.0f} s")
                rec.m(f"planted.V{V}.B{B}.seed{sd}", r, label="measured", status="exploratory")
                rec.flush()
    summary = {}
    for V in pc["vocab"]:
        for B in pc["batch"]:
            rr = [r for r in runs if r["V"] == int(V) and r["B"] == int(B)]
            summary[f"V{V}.B{B}"] = {"r2_raw": [r["r2_raw"] for r in rr], "r2_clamped": [r["r2_clamped"] for r in rr],
                                     "max_abs_dev_raw": [r["max_abs_dev_raw"] for r in rr],
                                     "max_abs_dev_clamped": [r["max_abs_dev_clamped"] for r in rr],
                                     "spearman_p_gt_20b": [r["spearman_p_gt_20b"] for r in rr]}
    rec.m("planted.summary", summary, label="measured", status="exploratory")
    (out / "planted_runs.json").write_text(json.dumps(_plain(runs), indent=1) + "\n")
    rec.flush()
    return 0


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", choices=("main", "gpu"), default="main", help="gpu: the child process (internal)")
    ap.add_argument("--out-dir", default=None, help="scratch output when not under run_logged.py")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-positions", type=int, default=0, help="smoke only")
    ap.add_argument("--kn-check-positions", type=int, default=0, help="smoke only")
    ap.add_argument("--reuse-pairs", action="store_true", help="smoke only: analyse an existing pairs file in --out-dir")
    args = ap.parse_args()
    cfg = load_json(args.config)
    mode = cfg.get("mode", "confirmatory")
    if mode == "planted":
        return run_planted(cfg, args)
    if args.phase == "gpu":
        return run_gpu(cfg, args)
    return run_main(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
