"""Paired scale comparison ACROSS runs of experiments/ckm_posthoc_probe_causal_controls.py (EXPLORATORY; validation
only; reads the per-token bits that those logged runs saved, trains and evaluates nothing).

Why a separate step: the controls run measured the young-bank control and the trained bank at 5M counting tokens
(plus one larger prefix chosen by its own projection); a second run of the same script measured the trained bank at
another prefix size. Both score the SAME validation positions, so paired document-block bootstrap differences
between the runs are well defined. The definitions are those of the controls script (its `comparisons` function is
imported and reused unchanged), so every quantity means exactly what it means inside a single run.

    scripts/run_logged.py --name ckm_probe_causal_scale_compare_exploratory --config <scratch json> --no-gpu-sampler -- \\
        .venv-tm/bin/python -u experiments/ckm_posthoc_probe_causal_scale_compare.py --runs <run dir> <run dir> \\
        --pairs trained5M:trained20M trained15M:trained20M --young young5M
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import ckm_posthoc_probe_causal_controls as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories of ckm_posthoc_probe_causal_controls.py")
    ap.add_argument("--pairs", nargs="+", required=True, help="small:large tag pairs, e.g. trained5M:trained20M")
    ap.add_argument("--young", default=None, help="young tag to include in the per-tag statistics of the first pair")
    args = ap.parse_args()
    new: dict = {}
    doc = None
    n_bytes = None
    sources = {}
    for d in args.runs:
        d = Path(d)
        rows = json.loads((d / "rows.json").read_text())
        met = json.loads((d / "metrics.json").read_text())
        z = np.load(d / "per_token_bits.npz")
        if doc is None:
            doc = z["doc"].astype(np.int64)
            n_bytes = int(met["data.validation_bytes_utf8"]["value"])
        else:
            assert np.array_equal(doc, z["doc"]), "runs score different validation positions"
            assert n_bytes == int(met["data.validation_bytes_utf8"]["value"])
        for k in z.files:
            if k in ("positions", "doc"):
                continue
            if k in C.PER_TOK:
                assert np.array_equal(C.PER_TOK[k], z[k]), f"row {k} differs between runs"
            C.PER_TOK[k] = z[k]
            sources[k] = str(d)
        C.ROWS.update(rows)
        for k, v in met.items():
            if k.endswith(".best_row_by_heldout_bits"):
                C.METRICS[k] = v
        run = json.loads((d / "run.json").read_text())
        new[f"inputs.{d.name}"] = {"value": {"run_dir": str(d), "source_tree_hash": run.get("source_tree_hash"), "exit_status": run.get("exit_status"),
                                             "tags": met.get("tags", {}).get("value")}, "label": "measured", "exploratory": True}
    n = int(doc.shape[0])
    X = dict(doc_of_val=doc, n_val_docs=int(doc.max() + 1), n_va=n, n_va_tok=n, n_bytes=n_bytes)
    before = set(C.METRICS)
    for i, pair in enumerate(args.pairs):
        small, large = pair.split(":")
        tags = {"small": small, "large": large}
        if i == 0 and args.young:
            tags["young"] = args.young
        C.comparisons(X, tags)
        for k in [k for k in C.METRICS if k.startswith("paired.scale.")]:
            new[k.replace("paired.scale.", f"paired.scale_{small}_to_{large}.")] = C.METRICS.pop(k)
    for k in C.METRICS:
        if k not in before and k.startswith("paired.") and k not in new:
            new[k] = C.METRICS[k]
    # derived ratio: TM share (KN-5 + induction + cache variant) over the best row's margin below KN-5, per tag
    for tag in sorted({t for p in args.pairs for t in p.split(":")} | ({args.young} if args.young else set())):
        sh = new.get(f"paired.{tag}.tm_share.kn5+lin+induction+doc_cache", {}).get("value")
        mg = new.get(f"paired.{tag}.margin.best_vs_kn5", {}).get("value")
        if sh and mg:
            new[f"derived.{tag}.tm_share_over_best_margin_below_kn5"] = {"value": sh["bits_per_token_a_minus_b"] / mg["bits_per_token_a_minus_b"],
                                                                        "label": "derived", "exploratory": True}
    new["sources_of_rows"] = {"value": sources, "label": "measured", "exploratory": True}
    if os.environ.get("TMLM_RUN_DIR"):
        from tmlm.common import runlog
        runlog.write_metrics(new, on_conflict="replace")
    for k, v in new.items():
        if k.startswith("paired.scale_"):
            x = v["value"]
            print(f"{k:90s} tok {x['bits_per_token_a_minus_b']:+.4f} {[round(c, 4) for c in x['ci95_bits_per_token']]} "
                  f"bpb {x['bits_per_byte_a_minus_b']:+.4f} {[round(c, 4) for c in x['ci95_bits_per_byte']]}", flush=True)
        elif k.startswith("derived."):
            print(k, v["value"], flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
