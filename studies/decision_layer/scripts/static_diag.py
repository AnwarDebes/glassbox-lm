#!/usr/bin/env python
"""Saturation check of the existsT bits on the dialogue texts (CPU, DEV only).

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv-dl/bin/python scripts/static_diag.py

existsT says "some word of the text is close to concept k at level j". A long text has many words, so
for a long text most bits may be on, and then they say little. For each dataset and field (customer's
words so far, last 3 turns), fitted on train: the share of bits on per dev text, by token-length decile,
and the share of dev texts with 90% or more of their bits on. Writes results/explore/static_diag.json.
Added 2026-10-17 for the no-network agent.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import load_examples  # noqa: E402
from dl.static_text import ExistsT, table  # noqa: E402


def main() -> None:
    out = {"time": None, "rows": []}
    for ds in ("star", "abcd"):
        tr, _ = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        for fld in ("cust", "recent"):
            x = ExistsT(what=fld).fit([r[fld] for r in tr])
            B = x.transform([r[fld] for r in dv])
            share = B.mean(1)
            n_tok = np.array([len(i) for i in table().ids([r[fld] for r in dv])])
            edges = np.unique(np.percentile(n_tok, np.linspace(0, 100, 11)))
            dec = np.clip(np.searchsorted(edges, n_tok, side="right") - 1, 0, len(edges) - 2)
            by = [{"tokens": f"{int(edges[d])}-{int(edges[d + 1])}", "n": int((dec == d).sum()),
                   "share_on": float(share[dec == d].mean())} for d in range(len(edges) - 1) if (dec == d).any()]
            row = {"dataset": ds, "field": fld, "n_bits": int(x.n_bits), "share_on_mean": float(share.mean()),
                   "texts_90pct_on": float((share >= 0.9).mean()), "by_token_decile": by, "info": x.info}
            out["rows"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "info"}), flush=True)
    out["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (ROOT / "results" / "explore" / "static_diag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
