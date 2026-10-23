#!/usr/bin/env python
"""Self-test of 2d-a (src/dl/count_bits.py) on the real data (CPU; train and dev only; no model; no score).

    OMP_NUM_THREADS=1 .venv-dl/bin/python scripts/count_bits_selftest.py | tee logs/count_bits_selftest.out

For STAR and ABCD, view B (mem, words, xcust, xrecent) on the C4 table is built twice, fitted on train: as ABCDView,
and as ABCDViewN with the part xcustn. Checked, on dev and on the first 5,000 train rows (exits at the first failure):
1. ABCDViewN's first columns are ABCDView's bits, byte for byte, and those dev bits equal the tuned dev row's
   (e74c_<ds>_Bc4_..._s16), so the block is additive and the old view is untouched;
2. the count block has 3 bits per concept (1,536 for K 512), and its names line up with the bits;
3. n_k >= 1 equals xcust's level-3 bit (or its highest-level bit where k has no level 3) for every concept, row for
   row (the spec's test, as amended in f8f552fd);
4. the block's bits are n_k >= 2, 4, 8 of ABCDViewN.counts, and n >= 8 <= n >= 4 <= n >= 2 <= that xcust bit;
5. the per-word hit matrix equals xcust's own codes at those levels for every word of the table (an independent path);
6. no count name matches verify_policy's concept regex (so e70's domain rows do not misread them), and every row of
   count_domain_rows holds on the bits.
Reported: the block's on-share per level, by known-token decile of the customer's words. Writes nothing else.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import count_bits, count_text  # noqa: E402

CONCEPT = re.compile(r"^some word of the (customer's words so far|last 3 turns) is close to concept (\d+) .* at level (\d+) \(cos >= ([-0-9.]+)\)$")
REF = {"star": "e74c_star_Bc4_c16000_s16", "abcd": "e74c_abcd_Bc4_c4000_s16"}
PARTS = ("mem", "words", "xcust", "xrecent")


def need(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main() -> None:
    t0 = time.time()
    count_text.patch("c4words-S1000-v1")
    from dl.abcd import ABCDView, load_examples

    rows = {d["tag"]: d for d in map(json.loads, (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines())
            if d.get("arm") == "tm" and d.get("tag") in REF.values()}
    report = {}
    for ds in ("star", "abcd"):
        tr, blist = load_examples("train", ds)
        dv, _ = load_examples("dev", ds)
        E = np.load(ROOT / "data" / "emb" / f"{ds}__none.npz")
        emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
        v0 = ABCDView(parts=PARTS, lex_k=1000, seed=0).fit(tr, emb["train"], blist)
        v1 = count_bits.ABCDViewN(parts=PARTS + (count_bits.PART,), lex_k=1000, seed=0).fit(tr, emb["train"], blist)
        x = v1.xcust
        K = x.K
        F0 = len(v0.names)
        # the xcust block's position and its level-0 columns
        o_cust = next(i for i, n in enumerate(v1.names) if n.startswith("some word of the customer's words so far"))
        first_cols = o_cust + v1.count_xcust_cols  # k's level-j_k bit (3, or k's highest level)
        lvl_ok = all(f" at level {j} (cos >= " in x.names[int(c)] and f"concept {k} (" in x.names[int(c)] for k, (j, c) in enumerate(zip(v1.count_at, v1.count_xcust_cols)))
        need(len(first_cols) == K and lvl_ok, f"{ds}: one counted xcust bit per concept ({K}); {int((v1.count_at < 3).sum())} concepts use their highest level (no level 3)")
        # 5: the hit matrix against xcust's own codes, every word of the table
        codes = np.unpackbits(x.packed, axis=1)[:, : x.n_bits]
        need(np.array_equal(codes[:, v1.count_xcust_cols].astype(bool), v1._hit), f"{ds}: hit matrix = xcust's codes at the counted levels for all {codes.shape[0]:,} words")
        for split, rr, e in (("dev", dv, emb["dev"]), ("train[:5000]", tr[:5000], {f: a[:5000] for f, a in emb["train"].items()})):
            X0 = v0.transform(rr, e)
            X1 = v1.transform(rr, e)
            need(X1.shape[1] == F0 + 3 * K == len(v1.names) and len(v1.count_names) == 3 * K == 1536,
                 f"{ds} {split}: {X0.shape[1]} + {3 * K} bits, names line up ({len(v1.names)})")
            need(np.array_equal(X1[:, :F0], X0), f"{ds} {split}: the first {F0} bits equal ABCDView's, byte for byte")
            if split == "dev":
                h = hashlib.sha256(np.ascontiguousarray(X0, dtype=np.uint8).tobytes()).hexdigest()
                need(h == rows[REF[ds]]["bits_sha256"]["dev"], f"{ds} dev: ABCDView's bits equal {REF[ds]}'s ({h[:8]})")
            N = v1.counts([r["cust"] for r in rr])
            need(np.array_equal((N >= 1).astype(np.uint8), X1[:, first_cols]), f"{ds} {split}: n_k >= 1 equals xcust's level-3 (or highest) bit, all {K} concepts, all {len(rr)} rows")
            C = X1[:, F0:].reshape(len(rr), K, 3)
            need(all(np.array_equal(C[:, :, j], (N >= m).astype(np.uint8)) for j, m in enumerate(count_bits.COUNT_LEVELS)),
                 f"{ds} {split}: the block is n_k >= 2, 4, 8")
            need(bool((C[:, :, 2] <= C[:, :, 1]).all() and (C[:, :, 1] <= C[:, :, 0]).all() and (C[:, :, 0] <= X1[:, first_cols]).all()),
                 f"{ds} {split}: n >= 8 <= n >= 4 <= n >= 2 <= the counted xcust bit")
            dr = count_bits.count_domain_rows(v1.names)
            need(len(dr) == 3 * K and all(float(sum(c * X1[:, i].astype(float) for i, c in co.items()).max()) <= hi for co, lo, hi in dr),
                 f"{ds} {split}: all {len(dr)} domain rows hold")
            if split == "dev":
                need(not any(CONCEPT.match(n) for n in v1.count_names), f"{ds}: no count name matches verify_policy's concept regex")
                nk = np.array([len(t) for t in x._table().ids([r["cust"] for r in rr])])
                edges = np.unique(np.percentile(nk, np.linspace(0, 100, 11)))
                dec = np.clip(np.searchsorted(edges, nk, side="right") - 1, 0, len(edges) - 2)
                report[ds] = {"n_bits_view_B": F0, "n_bits_block": 3 * K,
                              "on_share": {f"n>={m}": round(float(C[:, :, j].mean()), 4) for j, m in enumerate(count_bits.COUNT_LEVELS)},
                              "on_share_by_decile": [{"tokens": f"{int(edges[d])}-{int(edges[d + 1])}",
                                                      **{f"n>={m}": round(float(C[dec == d][:, :, j].mean()), 3) for j, m in enumerate(count_bits.COUNT_LEVELS)}}
                                                     for d in range(len(edges) - 1) if (dec == d).any()]}
                print(f"   {ds} dev on-share: {report[ds]['on_share']}; by decile: "
                      + "; ".join(f"{b['tokens']}: {b['n>=2']:.2f}/{b['n>=4']:.2f}/{b['n>=8']:.2f}" for b in report[ds]["on_share_by_decile"]), flush=True)
    print(json.dumps({"all checks passed": True, "seconds": round(time.time() - t0, 1), "report": report}), flush=True)


if __name__ == "__main__":
    main()
