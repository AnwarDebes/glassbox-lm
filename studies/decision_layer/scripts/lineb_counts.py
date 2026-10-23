#!/usr/bin/env python
"""Line B (verification) pre-work on DEV: the guideline properties, and how often the human agents and R19's
word-table TM break them on real conversations. Decides nothing; the test split is not read.

    lineb_counts.py --out results/explore/lineb_counts.json

The property list (the rule is in logs/plan_e70_verify.md, fixed before any solver run):
- Sources: the 55 subflows of data/untrusted/abcd/guidelines.json, mapped onto the 30 buttons by
  scripts/abcd_inject.guideline_steps (the study's mapping since e40), and the 55 sequences of kb.json.
  A property is kept only if it holds in both sources.
- P_first: with an empty history the chosen button starts a subflow (the union of both sources' first buttons).
- P_prec(A, B): A comes before the first B in every subflow that contains B, in both sources: the agent
  chooses B only if done(A).

Counts:
- human: the logged next button against the logged history (train and dev);
- TM: R19abcd tm_mem_static seeds 1 to 5 (view B, the registered models), on dev, open loop (logged memory) and
  closed loop (its own earlier presses, as scripts/abcd_closed_loop.py), with the CPU decision (src/dl/cpu_tm_fast.py).
Added 2026-10-19 for Line B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from abcd_closed_loop import closed_loop  # noqa: E402
from abcd_inject import guideline_steps  # noqa: E402
from dl import cpu_tm_fast  # noqa: E402
from dl.abcd import ABCD_DIR, ABCDView, load_examples  # noqa: E402

PARTS = ("mem", "words", "xcust", "xrecent")
DEV_SHA = "2f3223b2d9ca726ce7102990fca2cba1e0cbcb5ec5cf4cd727095b99503ce343"  # R19abcd's dev bits (view B)


def derive(seqs: dict[str, list[str]], buttons: list[str]) -> tuple[set, dict]:
    first = {s[0] for s in seqs.values() if s}
    prec = {}
    for b in buttons:
        subs = [s for s in seqs.values() if b in s]
        pred = None
        for s in subs:
            before = set(s[: s.index(b)])
            pred = before if pred is None else pred & before
        if pred:
            prec[b] = pred
    return first, prec


def properties(buttons: list[str]) -> dict:
    g = guideline_steps(buttons)
    k = json.loads((ABCD_DIR / "kb.json").read_text())
    fg, pg = derive(g, buttons)
    fk, pk = derive(k, buttons)
    allowed_first = sorted(fg | fk)
    assert all(s[0] in allowed_first for s in list(g.values()) + list(k.values()) if s)
    prec = sorted((a, b) for b in buttons for a in (pg.get(b, set()) & pk.get(b, set())))
    return {"allowed_first": allowed_first, "forbidden_first": [b for b in buttons if b not in allowed_first],
            "prec": [list(p) for p in prec],
            "dropped_prec": sorted([a, b] for b in buttons for a in (pg.get(b, set()) ^ pk.get(b, set()))),
            "first_guidelines": sorted(fg), "first_kb": sorted(fk)}


def count(prev_lists: list[list[str]], chosen: list[str], props: dict) -> dict:
    """Violations of each property, given the history and the chosen button of every action turn."""
    out = {"first": 0, "first_turns": 0}
    for prev, c in zip(prev_lists, chosen):
        if not prev:
            out["first_turns"] += 1
            out["first"] += c not in props["allowed_first"]
    for a, b in props["prec"]:
        out[f"{a}->{b}"] = sum(c == b and a not in prev for prev, c in zip(prev_lists, chosen))
    out[f"{'any_prec'}"] = sum(any(c == b and a not in prev for a, b in props["prec"]) for prev, c in zip(prev_lists, chosen))
    return out


def per_conv(prev_lists, chosen, rows, props) -> int:
    """Conversations with at least one violation of any property."""
    bad = set()
    for prev, c, r in zip(prev_lists, chosen, rows):
        if (not prev and c not in props["allowed_first"]) or any(c == b and a not in prev for a, b in props["prec"]):
            bad.add(r["convo_id"])
    return len(bad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/explore/lineb_counts.json")
    args = ap.parse_args()
    out_path = ROOT / args.out
    if out_path.exists():
        raise SystemExit(f"exists: {out_path}")
    t0 = time.time()
    tr, blist = load_examples("train")
    dv, _ = load_examples("dev")
    props = properties(blist)
    rec = {"script": "lineb_counts.py", "properties": props, "n_prec": len(props["prec"]),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name, rows in (("train", tr), ("dev", dv)):
        prev = [list(r["prev"]) for r in rows]
        chosen = [blist[r["y"]] for r in rows]
        rec[f"human_{name}"] = {"turns": len(rows), "conversations": len({r["convo_id"] for r in rows}),
                                "violations": count(prev, chosen, props), "conversations_with_violation": per_conv(prev, chosen, rows, props)}
    E = np.load(ROOT / "data" / "emb" / "abcd__potion-32m.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=PARTS, lex_k=1000, seed=0).fit(tr, em["train"], blist)
    X = view.transform(dv, em["dev"])
    assert hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest() == DEV_SHA, "dev bits differ from R19's"
    n_mem = 2 * len(blist) + 6
    y = np.array([r["y"] for r in dv])
    src = ROOT / "results" / "confirm" / "R19abcd" / "tm_mem_static"
    assert json.loads((src / "bit_names.json").read_text()) == view.names
    by_conv: dict[int, list[int]] = {}
    for i, r in enumerate(dv):
        by_conv.setdefault(r["convo_id"], []).append(i)
    for idx in by_conv.values():
        idx.sort(key=lambda i: dv[i]["k"])
    rec["tm_dev"] = {}
    for seed in (1, 2, 3, 4, 5):
        z = np.load(src / f"model_seed{seed}.npz")
        m = cpu_tm_fast.pack(z["ta"], z["w"])
        score = lambda Z: cpu_tm_fast.class_sums(m, Z)  # noqa: E731
        p_open = score(X).argmax(1)
        p_closed = closed_loop(score, X, dv, blist, n_mem)
        own_prev = [None] * len(dv)
        for idx in by_conv.values():
            for j, i in enumerate(idx):
                own_prev[i] = [blist[p_closed[q]] for q in idx[:j]]
        logged_prev = [list(r["prev"]) for r in dv]
        rec["tm_dev"][str(seed)] = {
            "acc_open": float((p_open == y).mean()), "acc_closed": float((p_closed == y).mean()), "n_clauses_kept": m["n_clauses_kept"],
            "n_literals": m["n_literals"],
            "open": {"violations": count(logged_prev, [blist[p] for p in p_open], props),
                     "conversations_with_violation": per_conv(logged_prev, [blist[p] for p in p_open], dv, props)},
            "closed": {"violations": count(own_prev, [blist[p] for p in p_closed], props),
                       "conversations_with_violation": per_conv(own_prev, [blist[p] for p in p_closed], dv, props)}}
        print(seed, json.dumps({k: rec["tm_dev"][str(seed)][k] for k in ("acc_open", "acc_closed", "n_clauses_kept", "n_literals")}), flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("n_prec", "human_dev", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
