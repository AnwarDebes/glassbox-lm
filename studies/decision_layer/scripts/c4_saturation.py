#!/usr/bin/env python
"""Saturation check of view B's OR bits on the C4 table (CPU, DEV only; no model; no rule). Added 2026-10-19 for the
text-stack map in the project notes (not published; P7, step 2d): scripts/static_diag.py's R19-era check redone for the C4 view.

    OMP_NUM_THREADS=1 .venv-dl/bin/python scripts/c4_saturation.py

existsT says "some word of the text is close to concept k at level j"; its thermometer thresholds come from the train
texts' max cosines. A long text has many words, so most of its bits may be on, and then they say little. For STAR and
ABCD, fields cust (view B's xcust) and recent (xrecent), the block is fitted on train exactly as ABCDView fits it
(dl.count_text.ExistsTC on c4words-S1000-v1, K 512, Thermo(8), seed 0). Per dev text:
- the share of the block's bits that are on;
- its token count: the C4 tokens that have a vector (what the OR sees; as static_diag.py), and all C4 tokens.
Reported per dataset and field: the deciles of the known-token count with the mean share on per decile, the share of
dev texts with 90% or more of the bits on (and 75% or more), and the potion numbers of results/explore/static_diag.json
beside them. Writes results/explore/c4_saturation.json (refuses to overwrite).
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
from dl.count_text import ExistsTC, count_table, tokens  # noqa: E402

OUT = ROOT / "results" / "explore" / "c4_saturation.json"
TABLE = "c4words-S1000-v1"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    t0 = time.time()
    tb = count_table(TABLE)
    potion = {(r["dataset"], r["field"]): r for r in json.loads((ROOT / "results" / "explore" / "static_diag.json").read_text())["rows"]}
    out = {"what": "saturation of view B's OR bits on the C4 table (dev; P7 step 2d)", "table": tb.meta(), "rows": []}
    for ds in ("star", "abcd"):
        tr, _ = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        for fld in ("cust", "recent"):
            x = ExistsTC(seed=0, what=fld, table_name=TABLE).fit([r[fld] for r in tr])
            texts = [r[fld] for r in dv]
            B = x.transform(texts)
            share = B.mean(1)
            n_known = np.array([len(i) for i in tb.ids(texts)])
            n_all = np.array([len(tokens(t or "(nothing yet)")) for t in texts])
            edges = np.unique(np.percentile(n_known, np.linspace(0, 100, 11)))
            dec = np.clip(np.searchsorted(edges, n_known, side="right") - 1, 0, len(edges) - 2)
            by = [{"tokens": f"{int(edges[d])}-{int(edges[d + 1])}", "n": int((dec == d).sum()), "share_on": round(float(share[dec == d].mean()), 4),
                   "share_texts_90pct_on": round(float((share[dec == d] >= 0.9).mean()), 4)} for d in range(len(edges) - 1) if (dec == d).any()]
            p = potion.get((ds, fld), {})
            row = {"dataset": ds, "field": fld, "n_dev": len(dv), "n_bits": int(x.n_bits), "share_on_mean": round(float(share.mean()), 4),
                   "share_on_max": round(float(share.max()), 4), "texts_90pct_on": round(float((share >= 0.9).mean()), 4),
                   "texts_75pct_on": round(float((share >= 0.75).mean()), 4),
                   "known_tokens": {"median": float(np.median(n_known)), "p90": float(np.percentile(n_known, 90)), "max": int(n_known.max())},
                   "all_tokens": {"median": float(np.median(n_all)), "p90": float(np.percentile(n_all, 90)), "max": int(n_all.max())},
                   "by_token_decile": by,
                   "potion_R19era": {"n_bits": p.get("n_bits"), "share_on_mean": p.get("share_on_mean"), "texts_90pct_on": p.get("texts_90pct_on"),
                                     "by_token_decile": p.get("by_token_decile")},
                   "info": x.info}
            out["rows"].append(row)
            print(f"{ds} {fld:6} bits {x.n_bits:>5}  mean on {row['share_on_mean']:.3f}  max {row['share_on_max']:.3f}  >=90% on {100 * row['texts_90pct_on']:.2f}% "
                  f"(potion {100 * (p.get('texts_90pct_on') or 0):.2f}%)  known tokens median {row['known_tokens']['median']:.0f} max {row['known_tokens']['max']}", flush=True)
            print("   deciles:", ", ".join(f"{b['tokens']}: {b['share_on']:.3f}" for b in by), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    out["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
