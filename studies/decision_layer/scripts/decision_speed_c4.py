#!/usr/bin/env python
"""Rough CPU time of one decision of the agent on the C4-word table (R21's view B), with the exact CPU decision.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 .venv-dl/bin/python scripts/decision_speed_c4.py

The same protocol as scripts/decision_speed.py (R19's agent), on R21's models. Dev rows only (1,000 per dataset, drawn
with numpy default_rng(0)), batch 1, one thread, after a warm-up of 20 rows. Per decision it times two stages
separately:
- the view's bits for that row: the memory, the word bits, and the existsT lookup + OR on the C4 table with its word
  tokenizer;
- then the TM decision, with src/dl/cpu_tm.py and with src/dl/cpu_tm_fast.py.
- Models: R21star and R21abcd tm_mem_c4, seed 1 (model_seed1.npz).
- Checks before timing: the view's train and dev bits equal the seed-1 json's hashes, both decisions give equal class
  sums on every dev row, and the dev accuracy equals the json's.
- Output: results/explore/decision_speed_c4.json (refuses to overwrite). Rough: a shared machine; read the ratios,
  not the times. R19's numbers are not carried over, because the tokenizer differs. Added 2026-10-19 after R21.
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

from dl import count_text, cpu_tm, cpu_tm_fast  # noqa: E402

PARTS = ("mem", "words", "xcust", "xrecent")
OUT = ROOT / "results" / "explore" / "decision_speed_c4.json"


def sha(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()


def stats(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)), "mean_ms": float(a.mean())}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    meta = count_text.patch("c4words-S1000-v1")
    from dl.abcd import ABCDView, load_examples  # noqa: E402

    out = {"label": "rough: shared machine, one CPU thread, batch 1", "parts": PARTS, "table": meta, "datasets": {}}
    for ds, reg in (("star", "R21star"), ("abcd", "R21abcd")):
        tr, blist = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        ydv = np.array([r["y"] for r in dv])
        folder = ROOT / "results" / "confirm" / reg / "tm_mem_c4"
        j = json.loads((folder / "seed1.json").read_text())
        view = ABCDView(parts=PARTS, seed=0).fit(tr, {}, blist)
        assert sha(view.transform(tr, {})) == j["bits_sha256"]["train"], ds
        Xdv = view.transform(dv, {})
        assert sha(Xdv) == j["bits_sha256"]["dev"], ds
        z = np.load(folder / "model_seed1.npz")
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
        out["datasets"][ds] = {"model": str((folder / "model_seed1.npz").relative_to(ROOT)), "n": len(idx), "dev_acc": acc,
                               "class_sums_equal_dev_rows": int(len(Xdv)), "n_clauses_kept": m1["n_clauses_kept"],
                               "n_literals": m1["n_literals"], "stages": s,
                               "ratio_tm_cpu_tm_over_fast_median": s["tm_cpu_tm"]["median_ms"] / s["tm_fast"]["median_ms"]}
        print(ds, json.dumps(out["datasets"][ds]), flush=True)
    out["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
