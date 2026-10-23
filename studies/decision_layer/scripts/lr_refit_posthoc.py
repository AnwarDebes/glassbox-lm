#!/usr/bin/env python
"""Post hoc, no rule: R19's LR baseline refit at its dev-best C, after the registered runs; one test read.

    lr_refit_posthoc.py --dataset star --C 0.01

Rules: registrations/R19_amendment_2026-10-18.md, section "Post-hoc LR refit at the dev-best C" (written before
e67b's numbers). C was chosen on dev only (e67, e67b, e67c in results/explore/abcd.jsonl; this script checks that
the given C is the dev-best of {0.1, 0.03, 0.01, 0.003, 0.001} with ties to the larger C, and that it beats C 0.1
by at least 0.5 points on dev). The view and bits are R19's view B (mem, words, xcust, xrecent; potion-32m), built
as scripts/abcd_confirm.py builds them, and their train, dev and test hashes must equal R19's. One LR fit on train
(dl.linear.fit_logreg, as abcd_confirm.py's lr arm), seed 0, one test read. Writes
results/confirm/<R19star|R19abcd>/lr_mem_static_posthocC/seed0.npz and .json in the format of scripts/report_abcd.py,
and refuses to overwrite. Added 2026-10-18 after R19 (internal audit 9, e67).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl.abcd import ABCDView, load_examples  # noqa: E402
from dl.linear import fit_logreg, logits  # noqa: E402

GRID = (0.1, 0.03, 0.01, 0.003, 0.001)  # larger first: ties go to the larger C
PARTS = ("mem", "words", "xcust", "xrecent")


def sha(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("DL_GPU_FRAC", "0.15")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["abcd", "star"])
    ap.add_argument("--C", type=float, required=True)
    args = ap.parse_args()
    reg = {"star": "R19star", "abcd": "R19abcd"}[args.dataset]
    out = ROOT / "results" / "confirm" / reg / "lr_mem_static_posthocC"
    if (out / "seed0.json").exists():
        raise SystemExit(f"exists: {out} (one test read only)")
    # the dev curve and the choice of C (dev only)
    dev = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("script") == "lr_c_check.py" and d["dataset"] == args.dataset and tuple(d["parts"]) == PARTS and d["encoder"] == "potion-32m":
            dev[d["C"]] = d["acc"]
    reg_json = json.loads((ROOT / "results" / "confirm" / reg / "lr_mem_static" / "seed0.json").read_text())
    dev.setdefault(0.1, reg_json["dev_acc"])
    assert set(GRID) <= set(dev), f"dev curve incomplete: {sorted(dev)}"
    best = max(GRID, key=lambda c: (dev[c], c))
    assert best == args.C, (best, args.C, dev)
    assert dev[best] - dev[0.1] >= 0.005, ("dev gain below 0.5 points: no refit", dev)
    tm_json = json.loads((ROOT / "results" / "confirm" / reg / "tm_mem_static" / "seed1.json").read_text())
    want = tm_json["bits_sha256"]
    # the view, as abcd_confirm.py builds it
    tr, blist = load_examples("train", args.dataset)
    dv, _ = load_examples("dev", args.dataset)
    te, _ = load_examples("test", args.dataset)
    E = np.load(ROOT / "data" / "emb" / f"{args.dataset}__potion-32m.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev", "test")}
    view = ABCDView(parts=PARTS, lex_k=1000, seed=0).fit(tr, em["train"], blist)
    X = {s: view.transform(rows, em[s]) for s, rows in (("train", tr), ("dev", dv), ("test", te))}
    for s in ("train", "dev", "test"):
        assert sha(X[s]) == want[s], (s, "bits differ from R19's")
    y = {s: np.array([r["y"] for r in rows]) for s, rows in (("train", tr), ("dev", dv), ("test", te))}
    torch.manual_seed(0)
    t0 = time.time()
    W, b = fit_logreg(X["train"], y["train"], len(blist), best)
    p_dv = logits(X["dev"], W, b).argmax(1).cpu().numpy()
    dev_acc = float((p_dv == y["dev"]).mean())
    assert abs(dev_acc - dev[best]) < 1e-12, (dev_acc, dev[best])
    p_te = logits(X["test"], W, b).argmax(1).cpu().numpy()
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "seed0.npz", pred=p_te, y=y["test"], convo_id=np.array([r["convo_id"] for r in te]),
                        k=np.array([r["k"] for r in te]), w_bert=np.array([r["ast_count_bert"] for r in te]),
                        w_roberta=np.array([r["ast_count_roberta"] for r in te]))
    rec = {"registration": reg, "name": "lr_mem_static_posthocC", "arm": "lr", "parts": PARTS, "encoder": "potion-32m", "seed": 0,
           "label": "post hoc, no rule, LR baseline refit at the dev-best C, after the registered runs",
           "C": best, "dev_curve": {str(c): dev[c] for c in GRID}, "dev_acc": dev_acc, "test_acc": float((p_te == y["test"]).mean()),
           "bits_sha256": want, "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "seed0.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
