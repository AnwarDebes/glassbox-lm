"""Step 9 of C-core, the numpy side (R1 [C-core] section 6, the TM, C0 and early-checkpoint rows; section 9, C-K1,
C-K2, C-K3 and C-shuffled for the TM at smoke scale). Written under the step-9 amendment; numpy and scipy only, so it
imports in both environments (torch lives in ccore/tmtorch.py).

Registered settings (R1 section 6, fixed before any number of C was seen):
  our CoTM (dl/tm.py f649043f..., workspace code, not a CAIR library), 65 classes, the 60 bits with negation
  (120 literals), 8,000 clauses, neg_sampling "random", batch 128, at most 20 epochs, other TMConfig defaults;
  read-out softmax(S / tau + b): tau and 65 biases fitted on calib by log-loss after each epoch; the epoch is the one
  with the lowest calib class log-loss of its own read-out (ties to the earlier epoch); valid is scored only at that
  checkpoint; grid T {8,000, 16,000} x s {3, 10, 30, 90} at seed 1, picked by the lowest valid system bpw (ties to
  the lower trial index); the pick then runs seeds 2-10; per-position bits are averaged over the seeds.
  C0: the picked setting on a seeded 1% of the train positions, 10 seeds; early checkpoint: each seed's state after
  epoch 1 with its own calib read-out.
  C-K2 smoke: 10% of the train blocks (subset seed 1), 2 epochs, T 8,000, s 10, TM seed 1.
Implementation choices of the amendment (flagged in the step-9 amendment, which was never frozen and is not distributed): every run trains all its epochs
(no patience rule); the read-out fit is L-BFGS-B from a cold start at every epoch (a >= 0, biases in [-50, 50]);
C0 uses one 1% subset of positions (subset seed 1) for its 10 TM seeds; the shuffled-label TM uses the smoke subset
with targets permuted across its positions (seed 1, as learners.PERM_SEED); K2's class unigram is the one every C
stage uses (systems.null_class_model of all train positions); C-K3's line is 230 GPU-minutes (twice the top of the
50-115 estimate) on measured plus projected stage wall time.

Self-test mode (tests only): C_STEP9_TEST_ROOT names a folder under C/tmp/ that holds the marker file
.step9_test_root. Then the data, runs, registration, frozen record and LOG.md paths point into that folder, the GPU
is hidden, the CoTM runs on the CPU and the TEST profile (tiny settings) applies. A test-mode run cannot read or write
C/data, C/data_dry or C/runs.
"""
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import env, gates, hashing, layout, learners, paths, runlog, systems

N_CLASSES = layout.N_CLASSES          # 65
N_FEATURES = layout.N_BITS            # 60 (120 literals with negation)
LN2 = math.log(2.0)

# ---------------------------------------------------------------- registered settings
CLAUSES = 8000
BATCH = 128
NEG_SAMPLING = "random"
MAX_EPOCHS = 20
GRID = tuple((T, s) for T in (8000, 16000) for s in (3.0, 10.0, 30.0, 90.0))   # trial index order = tie order
GRID_SEED = 1
SEEDS = tuple(range(1, 11))           # seed 1 is the grid run of the pick; seeds 2-10 run after the pick
SMOKE = {"share": 0.10, "subset_seed": 1, "epochs": 2, "T": 8000, "s": 10.0, "seed": 1}
SHUFFLE_PERM_SEED = learners.PERM_SEED   # 1, as for the twin, LR and LightGBM
C0 = {"share": 0.01, "subset_seed": 1}
K2_MAX_EMPTY_SHARE = 0.5
K2_MAX_MEDIAN_LITERALS = 30
K2_MAX_NO_POSITIVE_SHARE = 0.5
K3_ESTIMATE_MIN = (50.0, 115.0)
K3_LIMIT_MIN = 2 * K3_ESTIMATE_MIN[1]    # 230 GPU-minutes
READOUT_B_BOUND = 50.0
LENGTH_BINS = (0, 1, 2, 3, 5, 10, 20, 30, 50, 80, 121)
STAGES = ("tm_smoke", "tm_grid", "tm_seeds", "tm_c0")
DECISION_FILES = {"tm_smoke": "smoke_decision.json", "tm_grid": "grid_decision.json",
                  "tm_seeds": "seeds_decision.json", "tm_c0": "c0_decision.json"}
TORCH_ROLE = ("torch is the tensor library of our CoTM (dl/tm.py); no neural-network layer is built; the no-torch "
              "rule of R1 section 7 holds for every CPU stage, and tensorflow, jax, transformers, "
              "sentence_transformers and model2vec are refused here too")


@dataclass(frozen=True)
class Profile:
    name: str
    clauses: int
    grid: tuple
    max_epochs: int
    smoke: dict
    seeds: tuple
    c0: dict
    c0_seeds: tuple
    device: str
    extra: dict = field(default_factory=dict)


REGISTERED = Profile("registered", CLAUSES, GRID, MAX_EPOCHS, SMOKE, SEEDS, C0, SEEDS, "cuda")
# --dry on the pseudo splits of data_dry/ (train blocks only, written by the earlier dry stages): the registered
# machine, fewer epochs and seeds; its numbers mean nothing and its decisions are not enforced.
DRY = Profile("dry", CLAUSES, GRID, 2, dict(SMOKE), (1, 2, 3), C0, (1, 2), "cuda")
# self-tests on synthetic data (C_STEP9_TEST_ROOT): a tiny machine on the CPU.
TEST = Profile("test", 64, tuple((T, s) for T in (16, 32) for s in (3.0, 10.0, 30.0, 90.0)), 3,
               {"share": 0.10, "subset_seed": 1, "epochs": 2, "T": 32, "s": 3.0, "seed": 1}, (1, 2, 3), C0, (1, 2),
               "cpu")

TEST_ENV = "C_STEP9_TEST_ROOT"
TEST_MARKER = ".step9_test_root"
GO_FILE = paths.C / ".gpu_go"


# ---------------------------------------------------------------- mode, run rules and test-root redirection
class Mode:
    """What kind of run this is: test (synthetic data in a test root, CPU), dry (data_dry/, GPU) or real."""

    def __init__(self, test_root, dry, profile, go_file):
        self.test_root, self.dry, self.profile, self.go_file = test_root, dry, profile, go_file

    @property
    def test(self):
        return self.test_root is not None

    @property
    def device(self):
        return self.profile.device

    def describe(self):
        return {"test_root": str(self.test_root) if self.test else None, "dry": self.dry, "profile": self.profile.name,
                "device": self.device, "go_file": str(self.go_file)}


def test_root_from_env():
    root = os.environ.get(TEST_ENV)
    if not root:
        return None
    r = Path(os.path.realpath(root))
    tmp = Path(os.path.realpath(paths.TMP))
    if tmp not in r.parents:
        raise SystemExit(f"{TEST_ENV}={root} is not a folder under {paths.TMP}")
    if not (r / TEST_MARKER).exists():
        raise SystemExit(f"{TEST_ENV}={root} has no {TEST_MARKER} marker file")
    return r


def redirect_to_test_root(root):
    """Point every data, run and record path of C into the test root (the code paths stay where they are)."""
    paths.DATA = root / "data"
    paths.DATA_DRY = root / "data_dry"
    paths.RUNS = root / "runs"
    paths.LISTINGS = root / "runs" / "listings"
    paths.REG = root / "R1_C-core.md"
    paths.FROZEN = root / "R1_C-core.frozen.json"
    hashing.LOGMD = root / "LOG.md"


def stage_problems(test, environ=None):
    """Run rules of a TM stage: env.problems() without its no-GPU line, plus the GPU runner's settings."""
    environ = os.environ if environ is None else environ
    out = [p for p in env.problems() if not p.startswith("CUDA_VISIBLE_DEVICES")]
    cvd = environ.get("CUDA_VISIBLE_DEVICES")
    if test and cvd != "":
        out.append("a self-test run needs CUDA_VISIBLE_DEVICES set to the empty string (the GPU hidden)")
    if not test and not cvd:
        out.append("a GPU stage needs CUDA_VISIBLE_DEVICES set to the GPU (scripts/run_gpu_stage.sh sets 0)")
    if environ.get("CUDA_CACHE_DISABLE") != "1":
        out.append("CUDA_CACHE_DISABLE is not 1")
    if environ.get("C_GPU_RUNNER") != "1":
        out.append("not started by scripts/run_gpu_stage.sh (C_GPU_RUNNER is not 1)")
    return out


def guard(dry):
    """Run rules, test-root redirection, the go file and the profile of a GPU stage. Called before any data is read."""
    global _LIMITS
    root = test_root_from_env()
    test = root is not None
    p = stage_problems(test)
    if p:
        raise SystemExit("run rules not met:\n  " + "\n  ".join(p))
    try:
        from threadpoolctl import threadpool_limits
        _LIMITS = threadpool_limits(limits=env.THREADS)   # kept referenced for the life of the job
    except ImportError:                                    # the environment variables already cap the pools
        pass
    paths.setup()
    if test:
        redirect_to_test_root(root)
    go = (root / ".gpu_go") if test else GO_FILE
    if not go.exists():
        raise SystemExit(f"no go file {go}: the GPU queue creates it when it gives the slot (never this code)")
    profile = TEST if test else (DRY if dry else REGISTERED)
    return Mode(root, dry, profile, go)


_LIMITS = None


# ---------------------------------------------------------------- run record that allows the CoTM's torch
class TMRun(runlog.Run):
    """runlog.Run for the stages that run our CoTM: torch is allowed (dl/tm.py is written in torch) and recorded;
    every other neural-network library still fails the run. Also appends per-run GPU-time rows (C-K3)."""

    def finish(self, metrics):
        loaded = env.neural_modules_loaded()
        others = [m for m in loaded if m != "torch"]
        if others:
            raise SystemExit(f"neural-network libraries were imported: {others}")
        self.meta["neural_modules_loaded"] = loaded
        self.meta["torch_role"] = TORCH_ROLE
        self.dump("metrics.json", metrics)
        self.dump("outputs.json", hashing.manifest(self.outputs))
        self.meta.update(state="finished", end_utc=runlog.utc(), wall_s=round(time.time() - self.t0, 1))
        self.dump("run.json", self.meta)
        gpu_time_row({"utc": self.meta["end_utc"], "stage": self.name, "run": self.id, "what": "stage wall time",
                      "device": metrics.get("device", ""), "epochs": "", "batches": "", "train_s": "",
                      "readout_s": "", "wall_s": self.meta["wall_s"]})
        self.print("finished in", self.meta["wall_s"], "s")
        self._log.close()


GPU_TIME_COLS = ("utc", "stage", "run", "what", "device", "epochs", "batches", "train_s", "readout_s", "wall_s")


def gpu_time_row(row):
    """One line of runs/gpu_time.tsv (C-K3's record): every training run and every stage's wall time."""
    p = paths.RUNS / "gpu_time.tsv"
    new = not p.exists()
    with open(p, "a") as f:
        if new:
            f.write("\t".join(GPU_TIME_COLS) + "\n")
        f.write("\t".join(str(row.get(c, "")) for c in GPU_TIME_COLS) + "\n")


# ---------------------------------------------------------------- subsets and targets
def smoke_subset(blocks_train, profile):
    """Positions of the smoke: 10% of the train blocks (learners.subset_blocks with the smoke's subset seed)."""
    sm = profile.smoke
    mask = learners.subset_blocks(blocks_train, share=sm["share"], seed=sm["subset_seed"])
    if not mask.any():
        raise SystemExit("the smoke subset is empty")
    return np.flatnonzero(mask)


def c0_subset(n_train, profile):
    """C0's positions: a seeded 1% of the train positions (one subset, sorted, for all C0 seeds)."""
    k = int(round(profile.c0["share"] * n_train))
    if k < 1:
        raise SystemExit("the C0 subset is empty")
    return np.sort(np.random.default_rng(profile.c0["subset_seed"]).choice(n_train, size=k, replace=False))


def shuffled_targets(y):
    """Class targets permuted across the given positions (seed 1)."""
    return learners.permuted(y, seed=SHUFFLE_PERM_SEED)


# ---------------------------------------------------------------- read-out: softmax(S / tau + b) on calib
def _softmax_rows(L):
    L = L - L.max(1, keepdims=True)
    np.exp(L, out=L)
    L /= L.sum(1, keepdims=True)
    return L


def readout_proba(S, ro):
    """Class probabilities (n, 65) of the read-out ro at class sums S (any split)."""
    Z = np.asarray(S, dtype=np.float64) / ro["sigma"]
    return _softmax_rows(Z * ro["a_scaled"] + np.asarray(ro["b"], dtype=np.float64))


def class_logloss_bits(PX, y):
    """Mean class log-loss in bits (floored at 1e-300 only against an exact 0)."""
    y = np.asarray(y, dtype=np.int64)
    return float(-np.mean(np.log2(np.maximum(PX[np.arange(len(y)), y], 1e-300))))


def fit_readout(S, y, K=N_CLASSES):
    """Fit 1 / tau (>= 0) and K biases by log-loss on (S, y): logits = (S / sigma) a + b with sigma = std(S) (a
    scale for the optimiser only; tau = sigma / a). The loss is convex in (a, b); L-BFGS-B from a cold start (a 0,
    b the smoothed class log-frequencies), so the result depends only on (S, y). Returns the read-out and its calib
    class log-loss in bits, recomputed from the probabilities it gives."""
    from scipy.optimize import minimize
    S = np.asarray(S, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = S.shape[0]
    if S.shape != (n, K) or len(y) != n:
        raise ValueError(f"read-out shapes {S.shape} and {y.shape}")
    sigma = float(S.std())
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    Z = S / sigma
    ar = np.arange(n)
    Zy = Z[ar, y]
    cnt = np.bincount(y, minlength=K).astype(np.float64)
    b0 = np.log((cnt + 0.5) / (n + 0.5 * K))
    x0 = np.concatenate([[0.0], b0 - b0.mean()])
    nev = [0]

    def f(x):
        nev[0] += 1
        a, b = x[0], x[1:]
        L = Z * a
        L += b
        m = L.max(1)
        L -= m[:, None]
        np.exp(L, out=L)
        s = L.sum(1)
        L /= s[:, None]
        nll = (m + np.log(s)).sum() - (a * Zy + b[y]).sum()
        ga = (L * Z).sum() - Zy.sum()
        gb = L.sum(0) - cnt
        return nll / n, np.concatenate([[ga], gb]) / n

    r = minimize(f, x0, jac=True, method="L-BFGS-B",
                 bounds=[(0.0, None)] + [(-READOUT_B_BOUND, READOUT_B_BOUND)] * K,
                 options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-9})
    a_s = float(r.x[0])
    ro = {"a_scaled": a_s, "sigma": sigma, "b": [float(v) for v in r.x[1:]],
          "tau": (sigma / a_s) if a_s > 0 else float("inf"), "converged": bool(r.success), "nit": int(r.nit),
          "nfev": int(nev[0]), "message": str(r.message), "grad_max_abs": float(np.abs(r.jac).max())}
    ro["calib_class_bits"] = class_logloss_bits(readout_proba(S, ro), y)
    return ro


def pick_epoch(losses):
    """1-based epoch with the lowest calib class log-loss; ties to the earlier epoch."""
    return int(min(range(len(losses)), key=lambda i: (losses[i], i))) + 1


def pick_trial(bpws):
    """Index of the lowest valid system bpw; ties to the lower trial index."""
    return int(min(range(len(bpws)), key=lambda i: (bpws[i], i)))


# ---------------------------------------------------------------- clause statistics and C-K2
def clause_lengths(ta, n_states=256):
    return (np.asarray(ta) > n_states).sum(1)


def length_histogram(ta, n_states=256):
    h, e = np.histogram(clause_lengths(ta, n_states), bins=LENGTH_BINS)
    return {"edges": list(LENGTH_BINS), "counts": h.tolist()}


def clause_stats(ta, n_features=N_FEATURES, n_states=256):
    """The dl CoTM's own stats() figures, from a stored state (equal to CoTM.stats(); tests check it)."""
    A = np.asarray(ta) > n_states
    lens = A.sum(1)
    pos = A[:, :n_features].sum(1)
    return {"n_clauses": int(A.shape[0]), "empty": int((lens == 0).sum()), "median_literals": float(np.median(lens)),
            "mean_literals": float(lens.mean()), "max_literals": int(lens.max()),
            "p95_literals": float(np.percentile(lens, 95)),
            "no_positive_literal": int(((pos == 0) & (lens > 0)).sum()), "mean_positive_literals": float(pos.mean())}


def k2(stats, tm_class_bits_valid, unigram_class_bits_valid):
    """C-K2 (R1 section 9): stop and diagnose if more than 50% of clauses are empty, the median clause has more than
    30 literals, more than 50% of clauses have no positive literal (stats() count: non-empty clauses without one),
    or the calibrated valid class bits are worse than the class unigram's."""
    n = stats["n_clauses"]
    empty = stats["empty"] / n
    nopos = stats["no_positive_literal"] / n
    checks = {
        "empty_share": {"value": empty, "max": K2_MAX_EMPTY_SHARE, "stop": bool(empty > K2_MAX_EMPTY_SHARE)},
        "median_literals": {"value": stats["median_literals"], "max": K2_MAX_MEDIAN_LITERALS,
                            "stop": bool(stats["median_literals"] > K2_MAX_MEDIAN_LITERALS)},
        "no_positive_share": {"value": nopos, "max": K2_MAX_NO_POSITIVE_SHARE,
                              "stop": bool(nopos > K2_MAX_NO_POSITIVE_SHARE)},
        "class_bits_vs_unigram": {"tm": tm_class_bits_valid, "unigram": unigram_class_bits_valid,
                                  "stop": bool(tm_class_bits_valid > unigram_class_bits_valid)},
    }
    reported = {"empty_or_no_positive_share": (stats["empty"] + stats["no_positive_literal"]) / n}
    stops = [k for k, v in checks.items() if v["stop"]]
    return {"checks": checks, "reported_no_rule": reported, "stops": stops, "pass": not stops}


# ---------------------------------------------------------------- C-K3: GPU minutes, measured and projected
def batches_per_epoch(n, batch=BATCH):
    return int(math.ceil(n / batch))


def remaining_plan(after_stage, profile, n_train, n_c0):
    """Training runs and system fits of the stages after `after_stage` (None = before the smoke):
    [(stage, runs, epochs, positions, systems)]."""
    plan = [("tm_grid", len(profile.grid), profile.max_epochs, n_train, len(profile.grid) + 1),
            ("tm_seeds", len(profile.seeds) - 1, profile.max_epochs, n_train, 2 * (len(profile.seeds) - 1)),
            ("tm_c0", len(profile.c0_seeds), profile.max_epochs, n_c0, len(profile.c0_seeds))]
    order = [None, "tm_smoke", "tm_grid", "tm_seeds", "tm_c0"]
    done = order.index(after_stage)
    return [p for p in plan if order.index(p[0]) > done]


def project_minutes(plan, t_batch, t_readout_epoch, t_system, t_stage_overhead):
    """Projected wall minutes of a plan from measured unit times (seconds)."""
    s = 0.0
    for _, runs, epochs, n, n_sys in plan:
        s += runs * epochs * (batches_per_epoch(n) * t_batch + t_readout_epoch) + n_sys * t_system + t_stage_overhead
    return s / 60.0


def measured_minutes(stages, dry):
    """Wall minutes of the GPU stage runs so far, as C-K3 counts them: finished and crashed runs alike (both held the
    GPU slot; run.json wall_s). A run folder still marked "running" (a killed job) has no wall time and is listed."""
    import glob
    total, rows = 0.0, {}
    for st in stages:
        for d in sorted(glob.glob(str(paths.RUNS / f"*_{st}{'_dry' if dry else ''}"))):
            try:
                meta = runlog.read_json(os.path.join(d, "run.json"))
            except (OSError, ValueError):
                continue
            name = os.path.basename(d)
            if meta.get("state") in ("finished", "crashed") and "wall_s" in meta:
                rows[name] = float(meta["wall_s"])
                total += float(meta["wall_s"])
            else:
                rows[name] = f"no wall time (state {meta.get('state')})"
    return total / 60.0, rows


def k3(measured_min, projected_min):
    total = measured_min + projected_min
    return {"measured_min": measured_min, "projected_remaining_min": projected_min, "total_min": total,
            "estimate_min": list(K3_ESTIMATE_MIN), "limit_min": K3_LIMIT_MIN,
            "pass": bool(total <= K3_LIMIT_MIN + gates.EPS)}


# ---------------------------------------------------------------- systems and seed means
def split(Z, s, e):
    return systems.Split(Z[f"w_{s}"], Z[f"y_{s}"], Z[f"pF_{s}"], Z[f"Fc_{s}"], Z[f"b_{s}"], Z[f"L_{s}"],
                         Z[f"blocks_{s}"], e)


def system_score(PX, sp, Z, e, name):
    """System {F, f_TM, s_TM} (ccore/systems.py): bucketed MAP-EM on calib, the contract on the fixed positions of
    calib and valid and the all-position check (each stops the stage), then the valid per-position log-probs."""
    lam_g, lamb, em = systems.fit_system(PX["calib"], sp["calib"])
    rep = {}
    for s in ("calib", "valid"):
        con = systems.contract(PX[s], sp[s], lamb, Z[f"pos_{s}"], Z[f"D_{s}"], Z["cls"], e)
        allp = systems.all_positions(PX[s], lamb, {"pass": True})
        if not (con["pass"] and allp["pass"]):
            raise SystemExit(f"{name}: contract or all-position check failed on {s}: {con} {allp}")
        rep[s] = {"contract": con, "all_positions": allp}
    lp = systems.logp(PX["valid"], sp["valid"], lamb)
    return lp, lamb, {"valid_bpw": systems.bpw(lp), "class_bits_valid": systems.class_bits(PX["valid"], sp["valid"]),
                      "class_bits_calib": systems.class_bits(PX["calib"], sp["calib"]),
                      "standalone_bpw_valid": systems.standalone_bpw(PX["valid"], sp["valid"]),
                      "rare_split": systems.rare_split(lp, sp["valid"]), "checks": rep,
                      "lam_global": np.round(lam_g, 6).tolist(),
                      "mean_member_weights_valid": np.round(lamb[sp["valid"].b].mean(0), 6).tolist(), **em}


def seed_mean_bits(lps):
    """Per-position bits averaged over seeds (bits, never probabilities): (mean bits (n,), bits (seeds, n),
    per-seed bpw)."""
    B = np.stack([systems.bits(lp) for lp in lps])
    return B.mean(0), B, [float(b.mean()) for b in B]


def true_class_prob(PX, y):
    return PX[np.arange(len(y)), np.asarray(y, dtype=np.int64)]


# ---------------------------------------------------------------- inputs of the TM stages
def load_prep(dry):
    z = np.load(runlog.out_dir(dry) / "prep.npz")
    return {k: z[k] for k in z.files}


def tm_dir(dry):
    d = runlog.out_dir(dry) / "tm"
    os.makedirs(d, exist_ok=True)
    return d


def step6(dry, Z, e, sp):
    """The step-6 decision (exactly one finished real gates run, outputs verified) and the twin, F, LR and LightGBM
    valid per-position log-probs, each checked against the decision file's bpw (|difference| <= 1e-12)."""
    d = runlog.out_dir(dry)
    dec = runlog.read_json(d / "step6_decision.json")
    T = np.load(d / "twin.npz")
    L = np.load(d / "learners.npz")
    lp = {"F": np.log(sp["valid"].pF), "twin": T["best_valid"], "LR": L["lr_valid"], "LightGBM": L["lgbm_valid"]}
    if not np.allclose(T["F_valid"], lp["F"], rtol=0, atol=1e-12):
        raise SystemExit("twin.npz F_valid differs from the floor of prep.npz")
    mism = {k: (systems.bpw(v), dec["valid_bpw"][k]) for k, v in lp.items()
            if abs(systems.bpw(v) - dec["valid_bpw"][k]) > 1e-12}
    if mism:
        raise SystemExit(f"valid bpw differ from step6_decision.json: {mism}")
    return dec, lp


def decision_of(stage, dry):
    """The decision of a finished stage run, read from its decision file (an output whose hash check_inputs has just
    verified) and checked against the run's metrics.json: (decision, decision record, run folder)."""
    d = runlog.check_inputs([stage], dry)[stage]
    outs = runlog.read_json(os.path.join(d, "outputs.json"))
    hits = [p for p in outs if Path(p).name == DECISION_FILES[stage]]
    if len(hits) != 1:
        raise SystemExit(f"{d}: expected one {DECISION_FILES[stage]} among its outputs, found {hits}")
    rec = runlog.read_json(paths.ROOT / hits[0])
    if runlog.read_json(os.path.join(d, "metrics.json")).get("decision") != rec["decision"]:
        raise SystemExit(f"{d}: metrics.json and {DECISION_FILES[stage]} disagree")
    return rec["decision"], rec, d


def require_pass(stage, dry):
    """A real stage runs only after the earlier stage decided PASS (dry runs: reported, not enforced)."""
    dec, m, d = decision_of(stage, dry)
    if not dry and not dec.startswith("PASS"):
        raise SystemExit(f"{stage} decided {dec!r} ({d}): this stage does not run")
    return dec, m, d


def exit_code(decision):
    """3 for a registered STOP or a C-K3 HOLD, else 0 (scripts/run_gpu_stage.sh passes it on)."""
    return 3 if decision.startswith(("STOP", "HOLD")) else 0


def profile_dict(profile):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(profile).items()}
