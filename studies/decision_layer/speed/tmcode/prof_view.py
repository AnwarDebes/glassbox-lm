"""CPU profile of the no-network view build (ABCDView parts mem, words, xcust, xrecent on the C4 count table), as
scripts/table_run.py + abcd_explore.py build it for e72 / e74c / e77 (view B on c4words-S1000-v1). Read-only use of the decision_layer
code and data; the patch of dl.static_text (count_text.patch) lives in this process only.

Writes <out>.json (timings per part, bits SHA-256 vs the e74c record), <out>_cprofile.txt, and the dev bits to
<out>_dev_bits.npz (this folder) for bench_eval.py.
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

ROOT = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(ROOT / "src"))
from dl import count_text  # noqa: E402

E74C = {"train": "d7b0c2c7064aa8d319e871bfce4c589e8dbcfb67ff6874b2bd462925d78ab1ce",
        "dev": "56297e10edd0532f15823b602508976eaa28fbeb5fd0f14893eb862620014d33"}  # results/explore/abcd.jsonl, tag e74c_abcd_Bc4_c4000_s16


def sha(X):
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-cprofile", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    t_all = time.perf_counter()
    times = defaultdict(float)
    t0 = time.perf_counter()
    meta = count_text.patch("c4words-S1000-v1")
    times["table_load_and_hash"] = time.perf_counter() - t0
    from dl import abcd, boolean, static_text  # noqa: E402
    from dl.abcd import ABCDView, load_examples  # noqa: E402

    # timers around the parts (this process only)
    def timed(name, fn):
        @functools.wraps(fn)
        def w(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                times[name] += time.perf_counter() - t
        return w

    boolean.Lex.fit = timed("lex_fit", boolean.Lex.fit)
    boolean.Lex.transform = timed("lex_transform", boolean.Lex.transform)
    count_text.ExistsTC.fit = timed("existsT_fit", count_text.ExistsTC.fit)
    count_text.ExistsTC.transform = timed("existsT_transform", count_text.ExistsTC.transform)
    boolean._spherical_kmeans = timed("existsT_fit.kmeans", boolean._spherical_kmeans)
    count_text.CountTable.ids = timed("tokenize_ids (fit + transform)", count_text.CountTable.ids)
    abcd.memory_from_rows = timed("mem_bits", abcd.memory_from_rows)
    boolean.Thermo.fit = timed("existsT_fit.thermo", boolean.Thermo.fit)

    t0 = time.perf_counter()
    tr, blist = load_examples("train", "abcd")
    dv, _ = load_examples("dev", "abcd")
    E = np.load(ROOT / "data" / "emb" / "abcd__none.npz")
    etr = {f: E[f"train__{f}"] for f in ("cust", "recent", "full")}
    edv = {f: E[f"dev__{f}"] for f in ("cust", "recent", "full")}
    times["load_examples"] = time.perf_counter() - t0

    pr = cProfile.Profile()
    if not args.no_cprofile:
        pr.enable()
    t0 = time.perf_counter()
    view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, etr, blist)
    times["view_fit_total"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xtr = view.transform(tr, etr)
    times["view_transform_train_total"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xdv = view.transform(dv, edv)
    times["view_transform_dev_total"] = time.perf_counter() - t0
    if not args.no_cprofile:
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(35)
        s.write("\n\n==== by cumulative ====\n")
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(35)
        Path(f"{args.out}_cprofile.txt").write_text(s.getvalue())
    h = {"train": sha(Xtr), "dev": sha(Xdv)}
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cprofile_on": not args.no_cprofile,
           "times_s": {k: round(v, 3) for k, v in sorted(times.items(), key=lambda kv: -kv[1])},
           "total_s": round(time.perf_counter() - t_all, 3), "n_bits": int(Xtr.shape[1]), "n_train": int(Xtr.shape[0]),
           "n_dev": int(Xdv.shape[0]), "bits_sha256": h, "equals_e74c_record": {k: h[k] == E74C[k] for k in h},
           "static_info": view.static_info(), "density_train": float(Xtr.mean())}
    ydv = np.array([r["y"] for r in dv])
    np.savez_compressed(f"{args.out}_dev_bits.npz", X=Xdv, y=ydv)
    Path(f"{args.out}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("times_s", "total_s", "n_bits", "equals_e74c_record")}, indent=1))


if __name__ == "__main__":
    main()
