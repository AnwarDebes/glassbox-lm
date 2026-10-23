# diagnostic (dev bits only, no model, no score): on-share of per-concept count bits n>=2/4/8 when counting at xcust level j
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dl.abcd import load_examples
from dl.count_text import ExistsTC, count_table
tb = count_table("c4words-S1000-v1")
for ds in ("star", "abcd"):
    tr, _ = load_examples("train", ds); dv, _ = load_examples("dev", ds)
    x = ExistsTC(seed=0, what="cust").fit([r["cust"] for r in tr])
    ids = tb.ids([r["cust"] for r in dv])
    nlev = np.array([len(t) for t in x.th])
    nk = np.array([len(t) for t in ids])
    top = nk >= np.percentile(nk, 90)
    for j in (0, 2, 3, 4, 6):
        ok = nlev > j
        lows = np.array([t[j] if len(t) > j else np.inf for t in x.th])
        hit = x._S_all >= lows[None, :]
        N = np.stack([hit[t].sum(0) if t else np.zeros(x.K, int) for t in ids])[:, ok]
        s = {m: round(float((N >= m).mean()), 3) for m in (1, 2, 4, 8)}
        st = {m: round(float((N[top] >= m).mean()), 3) for m in (2, 4, 8)}
        print(f"{ds} level {j}: concepts with that level {int(ok.sum())}/512; on-share n>=1/2/4/8 {s}; longest 10% n>=2/4/8 {st}", flush=True)
