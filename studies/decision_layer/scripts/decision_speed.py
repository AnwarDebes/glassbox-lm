#!/usr/bin/env python
"""Rough CPU time of one decision of the word-table agent (view B), with the fast exact TM decision.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 .venv-dl/bin/python scripts/decision_speed.py

Dev rows only (1,000 per dataset, drawn with numpy default_rng(0)), batch 1, one thread, after warm-up. Per
decision it times the view's bits for that row (memory, word bits, existsT lookup + OR) and then the TM decision,
separately, with src/dl/cpu_tm.py (as in R19's speed test) and with src/dl/cpu_tm_fast.py.
- Models: STAR e60_star_B_c16000 (seed 1); ABCD R19abcd tm_mem_static seed 1.
- Checks before timing: the view's train and dev bits equal the recorded hashes, both decisions give equal class
  sums on every dev row, and the dev accuracy is as recorded.
Output: results/explore/decision_speed.json. Rough: shared machine; read the ratios, not the times. The bge-large
agent's time is in results/explore/static_speed_R19.json (not re-measured here). Added 2026-10-18 after R19.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import cpu_tm, cpu_tm_fast  # noqa: E402
from dl.abcd import ABCDView, load_examples  # noqa: E402

PARTS = ("mem", "words", "xcust", "xrecent")
CASES = [("star", "results/explore/abcd_models/e60_star_B_c16000_seed1.npz", "results/confirm/R19star/tm_mem_static/seed1.json"),
         ("abcd", "results/confirm/R19abcd/tm_mem_static/model_seed1.npz", "results/confirm/R19abcd/tm_mem_static/seed1.json")]


def sha(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()


def stats(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)), "mean_ms": float(a.mean())}


def main() -> None:
    out = {"label": "rough: shared machine, one CPU thread, batch 1", "parts": PARTS, "datasets": {}}
    for ds, mpath, jpath in CASES:
        tr, blist = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        ydv = np.array([r["y"] for r in dv])
        j = json.loads((ROOT / jpath).read_text())
        view = ABCDView(parts=PARTS, seed=0).fit(tr, {}, blist)
        assert sha(view.transform(tr, {})) == j["bits_sha256"]["train"], ds
        Xdv = view.transform(dv, {})
        assert sha(Xdv) == j["bits_sha256"]["dev"], ds
        z = np.load(ROOT / mpath)
        m0, m1 = cpu_tm.pack(z["ta"], z["w"]), cpu_tm_fast.pack(z["ta"], z["w"])
        S0, S1 = cpu_tm.class_sums(m0, Xdv), cpu_tm_fast.class_sums(m1, Xdv)
        assert np.array_equal(S0, S1), ds
        acc = float((S1.argmax(1) == ydv).mean())
        assert acc == j["dev_acc"], (ds, acc, j["dev_acc"])
        idx = np.random.default_rng(0).choice(len(dv), size=min(1000, len(dv)), replace=False)
        T = {k: [] for k in ("view", "tm_cpu_tm", "tm_fast", "total_cpu_tm", "total_fast")}
        for warm in (True, False):
            for i in (idx[:20] if warm else idx):
                t0 = time.perf_counter()
                x = view.transform([dv[i]], {})
                t1 = time.perf_counter()
                p0 = int(cpu_tm.class_sums(m0, x).argmax(1)[0])
                t2 = time.perf_counter()
                p1 = cpu_tm_fast.decide(m1, x[0])
                t3 = time.perf_counter()
                assert p0 == p1
                if warm:
                    continue
                ms = lambda a, b: (b - a) * 1000  # noqa: E731
                T["view"].append(ms(t0, t1)); T["tm_cpu_tm"].append(ms(t1, t2)); T["tm_fast"].append(ms(t2, t3))
                T["total_cpu_tm"].append(ms(t0, t2)); T["total_fast"].append(ms(t0, t1) + ms(t2, t3))
        s = {k: stats(v) for k, v in T.items()}
        out["datasets"][ds] = {"model": mpath, "n": len(idx), "dev_acc": acc, "class_sums_equal_dev_rows": int(len(Xdv)),
                               "n_clauses_kept": m1["n_clauses_kept"], "n_literals": m1["n_literals"], "stages": s,
                               "ratio_tm_cpu_tm_over_fast_median": s["tm_cpu_tm"]["median_ms"] / s["tm_fast"]["median_ms"]}
        print(ds, json.dumps(out["datasets"][ds]), flush=True)
    out["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (ROOT / "results" / "explore" / "decision_speed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
