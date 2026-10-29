"""Bitwise identity at the automaton bounds: both machines start from the same state near 2 * n_states (every automaton
in [2n - 7, 2n], i.e. every literal included), so the upper clamp is hit from the first steps; plus one start near the
lower bound ([1, 8]). Same comparisons as test_identity.py (ta, w, generator state, history, best_epoch, class sums).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prof_tm  # noqa: E402
from fast_tm import FastCoTM  # noqa: E402
from test_identity import same  # noqa: E402

import dl.tm as T  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402


def main():
    torch.set_num_threads(2)
    Xtr, ytr, Xdv, ydv, C = prof_tm.load("abcd", 1024, 1024)
    f = Xtr.shape[1]
    m = 2 * f
    N = 256
    cases = []
    # "firing_top": each clause includes 3 plain literals that are 1 in >= 95% of the rows, at states 505-512, every
    # other automaton excluded just below the boundary: the clauses fire, Type I feedback raises included 1-literals,
    # so the upper clamp (2 * n_states) is exercised (near_max cannot: a clause with x and NOT x never fires)
    freq = Xtr.mean(0)
    hot = np.flatnonzero(freq >= 0.95)
    print(f"{len(hot)} plain literals are 1 in >= 95% of the {len(Xtr)} rows", flush=True)
    for start in ("firing_top", "near_max", "near_min"):
        for chunk_name, chunk in (("unchunked", 50_000_000), ("chunks2", (N // 2) * m)):
            for s in (16.0, 2.0):
                cfg = TMConfig(n_clauses=N, T=N, s=s, epochs=2, batch_size=128, seed=11)
                rng = np.random.default_rng(5)
                if start == "firing_top":
                    ta0 = cfg.n_states - rng.integers(0, 8, size=(N, m))
                    for j in range(N):
                        ta0[j, rng.choice(hot, 3, replace=False)] = 2 * cfg.n_states - rng.integers(0, 8, 3)
                elif start == "near_max":
                    ta0 = 2 * cfg.n_states - rng.integers(0, 8, size=(N, m))
                else:
                    ta0 = 1 + rng.integers(0, 8, size=(N, m))
                outs = []
                for cls in (CoTM, FastCoTM):
                    T.CHUNK_ELEMS = chunk
                    tm = cls(C, f, cfg, device="cpu")
                    tm.ta = torch.as_tensor(ta0, dtype=torch.int32)
                    tm.fit(Xtr, ytr, Xdv, ydv)
                    outs.append(tm)
                ref, fast = outs
                r = same(ref, fast)
                Xv = torch.as_tensor(Xdv, dtype=torch.uint8)
                r["class_sums"] = bool(np.array_equal(CoTM.class_sums(ref, Xv).numpy(), fast.class_sums(Xv).numpy()))
                ta = ref.ta.numpy()
                ok = all(r.values())
                c = {"case": f"{start}/{chunk_name}/s{s:g}", "pass": ok, **r, "ta_at_max": int((ta == 2 * cfg.n_states).sum()),
                     "ta_at_1": int((ta == 1).sum()), "ta_elems": int(ta.size)}
                cases.append(c)
                print(f"{'PASS' if ok else 'FAIL'} {c}", flush=True)
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "all_pass": all(c["pass"] for c in cases), "cases": cases}
    Path(HERE / "results" / "identity_bounds.json").write_text(json.dumps(res, indent=1))
    print("ALL PASS" if res["all_pass"] else "SOME FAIL", len(cases), "cases")


if __name__ == "__main__":
    main()
