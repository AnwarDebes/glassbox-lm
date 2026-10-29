"""Step-by-step identity from the 'firing_top' start of test_identity_bounds.py: after EACH of the first 8 steps, ta,
w and the generator state of the prototype must equal the original's, and the number of automata at the upper bound
(2 * n_states = 512) is printed, so the upper clamp is seen to be exercised. Unchunked and 2 chunks, s 16 and s 2."""
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

import dl.tm as T  # noqa: E402
from dl.tm import CoTM, TMConfig  # noqa: E402


def main():
    torch.set_num_threads(2)
    Xtr, ytr, _, _, C = prof_tm.load("abcd", 1024, 8)
    f = Xtr.shape[1]
    m = 2 * f
    N = 256
    hot = np.flatnonzero(Xtr.mean(0) >= 0.95)
    out = []
    for chunk_name, chunk in (("unchunked", 50_000_000), ("chunks2", (N // 2) * m)):
        for s in (16.0, 2.0):
            cfg = TMConfig(n_clauses=N, T=N, s=s, epochs=1, batch_size=128, seed=11)
            rng = np.random.default_rng(5)
            ta0 = cfg.n_states - rng.integers(0, 8, size=(N, m))
            for j in range(N):
                ta0[j, rng.choice(hot, 3, replace=False)] = 2 * cfg.n_states - rng.integers(0, 8, 3)
            T.CHUNK_ELEMS = chunk
            a, b = CoTM(C, f, cfg, device="cpu"), FastCoTM(C, f, cfg, device="cpu")
            for tm in (a, b):
                tm.ta = torch.as_tensor(ta0, dtype=torch.int32)
            Xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr)
            steps = []
            for k in range(8):
                Xb, yb = Xt[128 * k : 128 * (k + 1)], yt[128 * k : 128 * (k + 1)]
                a._step(Xb, yb)
                b._step(Xb, yb)
                eq = (bool(torch.equal(a.ta, b.ta)), bool(torch.equal(a.w, b.w)), bool(torch.equal(a.gen.get_state(), b.gen.get_state())))
                steps.append({"step": k + 1, "equal_ta_w_gen": eq, "ta_at_max": int((a.ta == 2 * cfg.n_states).sum()),
                              "ta_at_1": int((a.ta == 1).sum())})
            ok = all(all(x["equal_ta_w_gen"]) for x in steps)
            out.append({"case": f"{chunk_name}/s{s:g}", "pass": ok, "steps": steps})
            print(("PASS" if ok else "FAIL"), chunk_name, s, [(x["step"], x["ta_at_max"], x["ta_at_1"]) for x in steps], flush=True)
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "all_pass": all(c["pass"] for c in out), "cases": out}
    Path(HERE / "results" / "identity_steps.json").write_text(json.dumps(res, indent=1))
    print("ALL PASS" if res["all_pass"] else "SOME FAIL", len(out), "cases")


if __name__ == "__main__":
    main()
