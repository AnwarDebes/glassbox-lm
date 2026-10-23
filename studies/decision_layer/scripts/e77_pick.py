#!/usr/bin/env python
"""e77 picks (plan logs/plan_e77_multiwoz_dev.md): the TM's s on MultiWOZ dev for both agents.

    e77_pick.py --stage screen   # the 4k picks; writes logs/jobs_e77b.txt (s 8 and the pick at 16k, per agent)
    e77_pick.py --stage final    # after e77b: s* per agent and the tuned dev gap at 16k

Rules (fixed in the plan), for each agent:
- screen: the highest 4k dev score among s 8, 12, 16 and 24, if it is at least 0.3 points above the agent's own s 8;
  otherwise s 8; an exact tie goes to the first of [8, 12, 16, 24];
- final: s* is the pick, unless its 16k dev is below the agent's own s 8 at 16k; then s 8.
Every row is checked: one row per tag, MultiWOZ, seed 1, the agent's view, after the freeze, and the same bits for
all of an agent's rows (the C4 agent's equal to scripts/e77_view_smoke.py's). Read only, except that it writes the
e77b job file; it refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = [8, 12, 16, 24]
AGENTS = {"C4": "e77_mwoz_Bc4", "bge-large": "e77_mwoz_bge"}
VIEW = {"C4": ("none", ["mem", "words", "xcust", "xrecent"]), "bge-large": ("bge-large", ["mem", "intent", "step", "words"])}
# the C4 view's bits from scripts/e77_view_smoke.py (logs/e77_view_smoke.out; equal with 1 and 2 threads)
C4_BITS = {"train": "a8c92a30430ec501d95cd7519e84847fb448d240356745a7d6474ae9dad71c66",
           "dev": "311e8933d2724bf8fc3ee1a5cef412010bd3a580c076217e3a964bcd286d68ba"}
SEEN_BITS: dict = {}
CMD = {"C4": ("HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 DL_GPU_FRAC=0.25 .venv-dl/bin/python scripts/table_run.py "
              "--table c4words-S1000-v1 scripts/abcd_explore.py --encoder none --arms tm --seed 1 --parts mem,words,xcust,xrecent --dataset multiwoz"),
       "bge-large": ("HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 DL_GPU_FRAC=0.25 .venv-dl/bin/python scripts/abcd_explore.py "
                     "--dataset multiwoz --encoder bge-large --parts mem,intent,step,words --arms tm --seed 1")}


def rows() -> dict:
    out = {}
    for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag", "").startswith("e77_"):
            out.setdefault(d["tag"], []).append(d)
    return out


def one(r: dict, agent: str, tag: str, frozen: str, clauses: int, s: int) -> dict:
    got = r.get(tag, [])
    if len(got) != 1:
        raise SystemExit(f"{tag}: {len(got)} rows (need exactly 1)")
    d = got[0]
    assert d["dataset"] == "multiwoz" and d["cfg"]["s"] == float(s) and d["cfg"]["n_clauses"] == clauses and d["cfg"]["T"] == clauses and d["seed"] == 1, tag
    assert (d["encoder"], list(d["parts"]), d["lex_k"]) == (*VIEW[agent], 1000), tag
    assert d["time"] > frozen, f"{tag} before the freeze"
    assert d["bits_sha256"] == SEEN_BITS.setdefault(agent, C4_BITS if agent == "C4" else d["bits_sha256"]), f"{tag}: other bits"
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["screen", "final"])
    args = ap.parse_args()
    frozen = json.loads((ROOT / "logs" / "plan_e77_multiwoz_dev.frozen.json").read_text())["frozen_utc"]
    r = rows()
    picks = {}
    for agent, pre in AGENTS.items():
        acc = {s: one(r, agent, f"{pre}_c4000_s{s}", frozen, 4000, s)["acc"] for s in S}
        for s in S:
            print(f"{agent} 4k s {s:>2}: dev {acc[s]:.4f}")
        best = max(S, key=lambda s: (acc[s], -S.index(s)))
        picks[agent] = best if acc[best] - acc[8] >= 0.003 - 1e-12 else 8
        print(f"{agent}: best s {best} ({acc[best]:.4f}); s 8 {acc[8]:.4f}; gap {100 * (acc[best] - acc[8]):+.2f}; pick s {picks[agent]}")
    if args.stage == "screen":
        out = ROOT / "logs" / "jobs_e77b.txt"
        if out.exists():
            raise SystemExit(f"{out} exists; not overwriting")
        lines = ["# e77b (plan logs/plan_e77_multiwoz_dev.md): MultiWOZ dev at 16,000 clauses, seed 1: each agent at s 8 and at its pick. Written by scripts/e77_pick.py."]
        for agent, pre in AGENTS.items():
            for s in sorted({8, picks[agent]}):
                lines.append(f"{CMD[agent]} --clauses 16000 --s {s} --save --tag {pre}_c16000_s{s} >> logs/{pre}_c16000_s{s}.log 2>&1")
        out.write_text("\n".join(lines) + "\n")
        print(f"wrote {out.relative_to(ROOT)} ({len(lines) - 1} jobs)")
        return
    final = {}
    for agent, pre in AGENTS.items():
        a8 = one(r, agent, f"{pre}_c16000_s8", frozen, 16000, 8)["acc"]
        p = picks[agent]
        ap_ = a8 if p == 8 else one(r, agent, f"{pre}_c16000_s{p}", frozen, 16000, p)["acc"]
        final[agent] = (p if ap_ >= a8 - 1e-12 else 8, ap_ if ap_ >= a8 - 1e-12 else a8)
        print(f"{agent}: 16k s 8 {a8:.4f}; 16k pick s {p} {ap_:.4f} -> s* {final[agent][0]}")
    print(f"tuned dev gap at 16k (C4 at s* {final['C4'][0]} minus bge-large at s* {final['bge-large'][0]}): {100 * (final['C4'][1] - final['bge-large'][1]):+.2f} points")


if __name__ == "__main__":
    main()
