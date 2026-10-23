#!/usr/bin/env python
"""Readable decisions of the agent with no neural network at any stage (R21abcd tm_mem_c4, seed 1), on ABCD dev.

    demo_agent_c4.py --n 4 --seed 0 > logs/demo_agent_c4_seed1_dev.txt

Dev dialogues drawn with numpy default_rng(seed) from the sorted dev conversation ids, not picked by hand. Open loop,
with the memory from the logged history. Every action turn shows:
- what the agent reads (the customer's latest utterance, the last 3 turns, the buttons done);
- its choice, the runner-up and the margin, as integer votes;
- the 3 firing clauses with the largest vote for the chosen button, in words (bit names), with their votes;
- whether a guard (e70's veto clauses, M recomputed for this model) fires, and whether the guarded choice differs;
- the logged button.
Long clauses are cut after 6 literals; positive literals come first.
Added 2026-10-19 after R21.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

import verify_policy as vp  # noqa: E402
from dl import count_text, cpu_tm_fast  # noqa: E402

MODELS = ROOT / "results" / "confirm" / "R21abcd" / "tm_mem_c4"


def short(name: str) -> str:
    return name.split(" (cos >= ")[0] if name.startswith("some word of") else name


def clause_text(lits: np.ndarray, names: list[str], F: int, max_lits: int = 6) -> str:
    pos = [short(names[l]) for l in lits if l < F]
    neg = [f"NOT {short(names[l - F])}" for l in lits if l >= F]
    parts = pos + neg
    more = len(parts) - max_lits
    return " AND ".join(parts[:max_lits]) + (f" AND ... (+{more} more)" if more > 0 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=1)
    args = ap.parse_args()
    meta = count_text.patch("c4words-S1000-v1")
    from dl.abcd import ABCDView, load_examples  # noqa: E402
    from lineb_counts import properties  # noqa: E402

    tr, blist = load_examples("train")
    dv, _ = load_examples("dev")
    names = json.loads((MODELS / "bit_names.json").read_text())
    view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, {}, blist)
    assert view.names == names
    vp.SRC = MODELS
    tm = vp.load_tm(args.model_seed)
    pack = tm["pack"]
    F = pack["n_features"]
    guards = vp.guards_for(tm, vp.checks_for(properties(blist), blist), blist)
    g = dict(tm, guards=guards)
    convs = sorted({r["convo_id"] for r in dv})
    pick = sorted(np.random.default_rng(args.seed).choice(len(convs), size=args.n, replace=False).tolist())
    print(f"The agent with no neural network at any stage (R21abcd tm_mem_c4, seed {args.model_seed}; table {meta['table']},"
          f" {meta['table_file_sha256'][:16]}). ABCD dev, open loop. Dialogues drawn with default_rng({args.seed}) from"
          f" {len(convs)} dev conversations: {[convs[i] for i in pick]}.\n")
    right = total = 0
    for ci in pick:
        cid = convs[ci]
        rows = sorted((r for r in dv if r["convo_id"] == cid), key=lambda r: r["k"])
        print(f"=== conversation {cid} ({rows[0]['flow']} / {rows[0]['subflow']}), {len(rows)} action turns")
        X = view.transform(rows, {})
        S = cpu_tm_fast.class_sums(pack, X)
        SG = vp.guard_sums(g, X)
        Fd = cpu_tm_fast.fired(pack, X).astype(bool)
        for t, r in enumerate(rows):
            s = S[t]
            order = np.argsort(-s, kind="stable")
            a, b = int(order[0]), int(order[1])
            gi = [(lit, cls) for lit, cls, _ in guards if (X[t, lit] if lit < F else 1 - X[t, lit - F])]
            ga = int(SG[t].argmax())
            u = r["cust"].split(" | ")[-1] if r["cust"] else ""
            print(f"\n turn {r['k']}: customer said: \"{u[:160]}\"")
            print(f"   last 3 turns: \"{r['recent'][-220:]}\"")
            print(f"   buttons done so far: {r['prev'] if r['prev'] else 'none'}")
            print(f"   CHOICE: {blist[a]} (votes {int(s[a])}); runner-up {blist[b]} ({int(s[b])}); margin {int(s[a] - s[b])}")
            on = np.flatnonzero(Fd[t])
            top = on[np.argsort(-pack["w"][on, a], kind="stable")[:3]]
            for j in top:
                lits = pack["lits"][pack["starts"][j] : (pack["starts"][j + 1] if j + 1 < len(pack["starts"]) else len(pack["lits"]))]
                print(f"     because {int(pack['w'][j, a]):+d}: IF {clause_text(lits, names, F)}")
            print(f"     ({len(on)} clauses fire; the 3 above carry the largest votes for {blist[a]})")
            vetoed = sorted({blist[c] for _, c in gi})
            print(f"   guards: {len(gi)} veto clauses fire (vetoed: {', '.join(vetoed) if vetoed else 'none'}); "
                  + (f"guarded choice {blist[ga]} differs" if ga != a else "the guarded choice is the same"))
            print(f"   logged button: {r['button']} -> {'right' if blist[a] == r['button'] else 'wrong'}")
            right += blist[a] == r["button"]
            total += 1
        print()
    print(f"{right} of {total} turns right in these dialogues (open loop).")


if __name__ == "__main__":
    main()
