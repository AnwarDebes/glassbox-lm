"""CPU profile of one CoTM training run (the decision_layer src/dl/tm.py, imported read-only; nothing of the decision_layer is written).

Fixed configs (--config): see CONFIGS below. Data: real view-B bits of ABCD (mem, words, xcust, xrecent on potion-32M,
the decision_layer cache data/abcd/bits_mem-words-xcust-xrecent__potion-32m_{train,dev}.npz, read-only), the first --rows train
rows and the first --dev dev rows, every bit column; or synthetic bits (--data synth).

Three passes on the same config and seed, each in a fresh CoTM:
  1. wall   : tm.fit timed with perf_counter (no profiler attached)
  2. cprof  : cProfile of tm.fit -> <out>_cprofile.txt
  3. tprof  : torch.profiler (CPU activity, record_shapes) of tm.fit, with record_function ranges around
              CoTM.feedback / CoTM._automata_chunked / CoTM._neg_classes / CoTM.predict (wrappers in this process
              only) -> <out>_ops_by_name.txt, <out>_ops_by_shape.txt, <out>_phases.json
The final TA states and weights of the three passes are compared (they must be equal: profiling changes nothing).
Run: CUDA_VISIBLE_DEVICES= nice -n 19 <venv python> -I prof_tm.py --config C1 --out results/C1
"""
from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import cProfile
import functools
import hashlib
import io
import json
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SRC = f"{_REPO_ROOT}/studies/decision_layer/src"
DATA = Path(f"{_REPO_ROOT}/studies/decision_layer/data")
sys.path.insert(0, SRC)
import dl.tm as T  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402

torch.cuda.synchronize = lambda *a, **k: None  # CoTM.fit ends with it; this process has no GPU (CUDA_VISIBLE_DEVICES=)

# name: (data, rows, dev, clauses, epochs, batch, s, chunk_elems or 0 = the module default)
CONFIGS = {
    # unchunked path (N * m below dl.tm.CHUNK_ELEMS)
    "C1": ("abcd", 2048, 1024, 256, 2, 128, 16.0, 0),
    # the chunked path that every e82 run takes on the GPU (N * m > 50M there): forced here with 2 chunks
    "C2": ("abcd", 2048, 1024, 256, 2, 128, 16.0, -2),
    # host overhead floor: tiny tensors, many steps
    "C3": ("synth16", 25600, 1024, 8, 1, 128, 8.0, 0),
    # 5 chunks, like the 16k-clause MultiWOZ / STAR runs
    "C4": ("abcd", 2048, 1024, 320, 2, 128, 8.0, -5),
}


def load(data: str, rows: int, dev: int, seed: int = 0):
    if data == "abcd":
        tr = np.load(DATA / "abcd" / "bits_mem-words-xcust-xrecent__potion-32m_train.npz")
        dv = np.load(DATA / "abcd" / "bits_mem-words-xcust-xrecent__potion-32m_dev.npz")
        Xtr, ytr, Xdv, ydv = tr["X"][:rows], tr["y"][:rows], dv["X"][:dev], dv["y"][:dev]
        C = int(max(tr["y"].max(), dv["y"].max())) + 1
        return np.ascontiguousarray(Xtr, np.uint8), ytr.astype(np.int64), np.ascontiguousarray(Xdv, np.uint8), ydv.astype(np.int64), C
    if data == "synth16":
        rng = np.random.default_rng(seed)
        X = (rng.random((rows + dev, 16)) < 0.3).astype(np.uint8)
        y = (X[:, :3].sum(1) + X[:, 3:5].sum(1)) % 4
        return X[:rows], y[:rows], X[rows:], y[rows:], 4
    raise ValueError(data)


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def make(cfg_name: str, seed: int):
    data, rows, dev, N, ep, bs, s, ck = CONFIGS[cfg_name]
    Xtr, ytr, Xdv, ydv, C = load(data, rows, dev)
    f = Xtr.shape[1]
    m = 2 * f
    if ck < 0:  # force -ck chunks: CHUNK_ELEMS just below N * m / (-ck - 1) ... choose cs = ceil(N / -ck)
        k = -ck
        cs = -(-N // k)
        T.CHUNK_ELEMS = cs * m  # cs = CHUNK_ELEMS // m -> exactly cs clauses per chunk; N * m > CHUNK_ELEMS
    else:
        T.CHUNK_ELEMS = 50_000_000
    cfg = TMConfig(n_clauses=N, T=N, s=s, epochs=ep, batch_size=bs, seed=seed)
    tm = CoTM(C, f, cfg, device="cpu")
    return tm, (Xtr, ytr, Xdv, ydv), dict(data=data, rows=rows, dev=dev, n_clauses=N, epochs=ep, batch_size=bs, s=s,
                                          n_features=f, n_literals=m, n_classes=C, chunk_elems=T.CHUNK_ELEMS,
                                          chunked=bool(N * m > T.CHUNK_ELEMS),
                                          n_chunks=(-(-N // max(1, T.CHUNK_ELEMS // m)) if N * m > T.CHUNK_ELEMS else 1),
                                          steps_per_epoch=-(-rows // bs), density_train=float(Xtr.mean()))


def wrap_ranges(stats: dict):
    """record_function ranges + call counts around the CoTM methods (this process only)."""
    from torch.profiler import record_function

    def deco(name, fn):
        @functools.wraps(fn)
        def w(*a, **k):
            stats[name] = stats.get(name, 0) + 1
            with record_function(f"phase::{name}"):
                return fn(*a, **k)
        return w

    orig = {n: getattr(CoTM, n) for n in ("feedback", "_automata_chunked", "_neg_classes", "predict")}
    for n, fn in orig.items():
        setattr(CoTM, n, deco(n, fn))
    return orig


def unwrap(orig):
    for n, fn in orig.items():
        setattr(CoTM, n, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-cprofile", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"config": args.config, "seed": args.seed, "threads": torch.get_num_threads(), "torch": torch.__version__,
           "numpy": np.__version__, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # binomial count statistics (a wrapper that only reads the counts; same call, same generator)
    bstats = []
    orig_binom = torch.binomial

    def binom_spy(count, prob, generator=None):
        nz = count > 0
        bstats.append((float(nz.float().mean()), float(count.sum()), float(count.max()), int(count.numel())))
        return orig_binom(count, prob, generator=generator)

    # 1. wall
    tm, (Xtr, ytr, Xdv, ydv), info = make(args.config, args.seed)
    res["info"] = info
    t0 = time.perf_counter()
    tm.fit(Xtr, ytr, Xdv, ydv)
    res["wall_fit_s"] = time.perf_counter() - t0
    res["history"] = tm.history
    res["best_epoch"] = getattr(tm, "best_epoch", None)
    ref = (sha(tm.ta.numpy()), sha(tm.w.numpy()))
    res["state_sha"] = {"wall": ref}
    # eval alone (the per-eval cost): predict on dev, timed 3 times
    Xv = torch.as_tensor(Xdv, dtype=torch.uint8)
    te = []
    for _ in range(3):
        t1 = time.perf_counter()
        tm.predict(Xv)
        te.append(time.perf_counter() - t1)
    res["predict_dev_s"] = te

    # 2. cProfile
    if not args.skip_cprofile:
        tm2, _, _ = make(args.config, args.seed)
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        tm2.fit(Xtr, ytr, Xdv, ydv)
        pr.disable()
        res["cprofile_fit_s"] = time.perf_counter() - t0
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(40)
        s.write("\n\n==== by cumulative ====\n")
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(40)
        Path(f"{out}_cprofile.txt").write_text(s.getvalue())
        res["state_sha"]["cprofile"] = (sha(tm2.ta.numpy()), sha(tm2.w.numpy()))

    # 3. torch.profiler with phase ranges and the binomial spy
    from torch.profiler import ProfilerActivity, profile

    calls = {}
    orig = wrap_ranges(calls)
    try:
        tm3, _, _ = make(args.config, args.seed)
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            tm3.fit(Xtr, ytr, Xdv, ydv)
        res["tprofile_fit_s"] = time.perf_counter() - t0
    finally:
        unwrap(orig)
    res["state_sha"]["tprofile"] = (sha(tm3.ta.numpy()), sha(tm3.w.numpy()))
    # 4. binomial count statistics in a pass of its own (the spy's reads would otherwise be charged to feedback)
    torch.binomial = binom_spy
    try:
        tm4, _, _ = make(args.config, args.seed)
        tm4.fit(Xtr, ytr, Xdv, ydv)
    finally:
        torch.binomial = orig_binom
    res["state_sha"]["spy"] = (sha(tm4.ta.numpy()), sha(tm4.w.numpy()))
    res["same_state_all_passes"] = len(set(map(tuple, res["state_sha"].values()))) == 1
    res["method_calls"] = calls
    Path(f"{out}_ops_by_name.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60))
    Path(f"{out}_ops_by_shape.txt").write_text(
        prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=80))

    # phase attribution from the event tree: every op is charged (self time) to its innermost phase range,
    # and inside feedback / _automata_chunked to an op class by name and shapes
    ev = prof.events()
    N, m, bs, C = info["n_clauses"], info["n_literals"], info["batch_size"], info["n_classes"]
    phase_self = defaultdict(float)
    op_class = defaultdict(float)
    op_count_feedback = defaultdict(int)
    VIEW_OPS = {"aten::t", "aten::transpose", "aten::unsqueeze", "aten::slice", "aten::select", "aten::expand",
                "aten::as_strided", "aten::view", "aten::alias", "aten::numpy_T", "aten::permute", "aten::empty",
                "aten::empty_strided", "aten::resolve_conj", "aten::resolve_neg", "aten::lift_fresh",
                "aten::detach", "aten::_unsafe_view", "aten::reshape", "aten::expand_as", "aten::squeeze",
                "aten::result_type", "aten::item", "aten::_local_scalar_dense", "aten::is_nonzero"}

    def phase_of(e):
        p = e.cpu_parent
        while p is not None:
            if p.name.startswith("phase::"):
                return p.name[7:]
            p = p.cpu_parent
        return "outside"

    def top_op(e):
        """True when no aten op encloses e (its parent is a phase range or nothing)."""
        p = e.cpu_parent
        return p is None or p.name.startswith("phase::") or not p.name.startswith("aten::")

    n_feedback_calls = calls.get("feedback", 0)
    for e in ev:
        if e.name.startswith("phase::"):
            continue
        ph = phase_of(e)
        st = e.self_cpu_time_total / 1e6
        phase_self[ph] += st
        if ph in ("feedback", "_automata_chunked", "_neg_classes"):
            shp = e.input_shapes or []
            nm = e.name
            if nm in ("aten::mm", "aten::matmul", "aten::addmm"):
                s0 = tuple(shp[0]) if shp and shp[0] else ()
                s1 = tuple(shp[1]) if len(shp) > 1 and shp[1] else ()
                if s1 and s1[-1] == C:
                    k = "mm_class_sums [b,N]x[N,C]"
                elif s0 and s0[0] == bs and s1 and s1[-1] != C and s0[-1] == m:
                    k = "mm_clause_outputs [b,m]x[m,N]"
                else:
                    k = "mm_automata_counts [N,b]x[b,m]"
            elif nm.startswith("aten::binomial") or nm.startswith("aten::_s_binomial"):
                k = "binomial"
            elif nm in ("aten::rand", "aten::uniform_", "aten::randint", "aten::random_", "aten::randperm"):
                k = "rng_uniform_or_int"
            elif nm in ("aten::index_add_", "aten::index_add"):
                k = "weights_index_add"
            else:
                k = f"other[{ph}]"
            op_class[k] += st
            if top_op(e) and e.name not in VIEW_OPS and e.name.startswith("aten::"):
                op_count_feedback[e.name] += 1
        elif ph == "predict":
            op_class["predict (dev eval)"] += st
        else:
            op_class[f"outside: {e.name}"] += st
    res["phase_self_s"] = dict(sorted(phase_self.items(), key=lambda kv: -kv[1]))
    res["op_class_self_s"] = dict(sorted(op_class.items(), key=lambda kv: -kv[1])[:40])
    res["n_feedback_calls"] = n_feedback_calls
    res["top_level_nonview_aten_ops_per_feedback"] = {k: v / max(1, n_feedback_calls) for k, v in
                                                      sorted(op_count_feedback.items(), key=lambda kv: -kv[1])}
    res["top_level_nonview_aten_ops_per_feedback_total"] = sum(op_count_feedback.values()) / max(1, n_feedback_calls)
    if bstats:
        a = np.array([b[0] for b in bstats])
        res["binomial_calls"] = len(bstats)
        res["binomial_nonzero_frac_mean"] = float(a.mean())
        res["binomial_nonzero_frac_inc_dec"] = [float(a[0::2].mean()), float(a[1::2].mean())]
        res["binomial_mean_count_per_element"] = float(np.mean([b[1] / b[3] for b in bstats]))
        res["binomial_max_count"] = float(max(b[2] for b in bstats))
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(f"{out}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("config", "wall_fit_s", "same_state_all_passes", "phase_self_s", "op_class_self_s",
                                          "top_level_nonview_aten_ops_per_feedback_total") if k in res}, indent=1))


if __name__ == "__main__":
    main()
