#!/usr/bin/env python
"""e81 (plan logs/plan_e81_T_screen.md): the TM's vote threshold T on the agents' readers. Picks, follow-up jobs, bars.
Read only, except that it writes job files (refusing to overwrite). Written before any e81 number.

    e81_pick.py --stage smoke     # after jobs_e81_smoke.txt: the --T plumbing on each runner path (before jobs_e81a.txt)
    e81_pick.py --stage screen    # after jobs_e81a.txt: the T pick per (agent, dataset); writes logs/jobs_e81b.txt
    e81_pick.py --stage final     # after jobs_e81b.txt: the bars, with the epoch table and TM minus LR beside them

Rules (fixed in the plan):
- the T ratio r in [1, 1/2, 1/4] (T = r x clauses) at 4,000 clauses, seed 1, at each agent's own tuned s; the pick is the
  highest 4k dev, if it is at least 0.3 points above r = 1 (T = clauses, the existing row), otherwise r = 1; an exact tie
  goes to the first in the list (r = 1);
- for a pick r != 1: on STAR and MultiWOZ the pick runs at 16,000 clauses (T = r x 16,000) at seeds 1 and 2; on ABCD the
  4k pick stands and runs at seed 2; for the bge-large agent the seed-2 row at r = 1 is run too (it has no dev seed 2);
- B1, "T helps the reader" on (agent, dataset): the pick's full-size mean of seeds 1-2 minus r = 1's is at least +0.5;
- B2, "T closes the gap" for the refined agent on MultiWOZ: its TM at the pick (mean of seeds 1-2, 16k) minus LR on
  the same bits (0.3643, e82's LR) is at least half of today's gap, i.e. at least -2.675 points.
The C4 agent is screened at 4k and reported with no rule (no follow-up).
Checks on every row it reads: the settings above, the other TM settings (BASE_CFG), the a3B table for the refined agent,
e81 rows written after the plan's freeze, one dev bits hash per (agent, dataset); LR rows on those bits with GRID_C.
Tested by tests/test_e81_pick.py (made-up rows in a temporary folder).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATIOS = [1.0, 0.5, 0.25]
FULL = {"star": 16000, "abcd": 4000, "multiwoz": 16000}
A3B_SHA = "4a1ed2aed7aa4ce726892446a2cdf34b23a6e43d3c2e943f0fb46c5210d7d855"
TH = "HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false DL_GPU_FRAC=0.25 .venv-dl/bin/python"
AGENTS = {
    "a3B": {"enc": "none", "parts": ["mem", "words", "xcust", "xrecent"], "s": {"star": 16, "abcd": 16, "multiwoz": 8},
            "run": f"{TH} scripts/newtable_run.py --table-dir data/tables/c4words_a3B_v1 --name c4words-a3B-v1 scripts/abcd_explore.py --encoder none "
                   "--arms tm --parts mem,words,xcust,xrecent --save",
            "ref4k": {"star": "e82r3_star_Bt_c4000_s16", "abcd": "e82_abcd_Bt_c4000_s16", "multiwoz": "e82r3_multiwoz_Bt_c4000_s8"},
            "reffull": {"star": [("e82_star_Bt_c16000_s16", 1), ("e82_star_Bt_c16000_s16", 2)],
                        "abcd": [("e82_abcd_Bt_c4000_s16", 1), ("e82_abcd_Bt_c4000_s16", 2)],
                        "multiwoz": [("e82_multiwoz_Bt_c16000_s8", 1), ("e82_multiwoz_Bt_c16000_s8", 2)]},
            "follow": True},
    "bge": {"enc": "bge-large", "parts": ["mem", "intent", "step", "words"], "s": {"star": 16, "abcd": 24, "multiwoz": 8},
            "run": f"{TH} scripts/abcd_explore.py --encoder bge-large --parts mem,intent,step,words --arms tm --save",
            "ref4k": {"star": "e74d_star_bge_c4000_s16", "abcd": "e75_abcd_bge_c4000_s24", "multiwoz": "e77_mwoz_bge_c4000_s8"},
            "reffull": {"star": [("e74d_star_bge_c16000_s16", 1), ("e81_star_bge_c16000_s16_T16000", 2)],
                        "abcd": [("e75_abcd_bge_c4000_s24", 1), ("e81_abcd_bge_c4000_s24_T4000", 2)],
                        "multiwoz": [("e77_mwoz_bge_c16000_s8", 1), ("e81_multiwoz_bge_c16000_s8_T16000", 2)]},
            "follow": True},
    "C4": {"enc": "none", "parts": ["mem", "words", "xcust", "xrecent"], "s": {"star": 16, "abcd": 16, "multiwoz": 8},
           "run": f"{TH} scripts/table_run.py --table c4words-S1000-v1 scripts/abcd_explore.py --encoder none --arms tm --parts mem,words,xcust,xrecent --save",
           "ref4k": {"star": "e74c_star_Bc4_c4000_s16", "abcd": "e74c_abcd_Bc4_c4000_s16", "multiwoz": "e77_mwoz_Bc4_c4000_s8"},
           "reffull": {"star": [("e74c_star_Bc4_c16000_s16", 1), ("e78_star_B_c16000_s16", 2)],
                       "abcd": [("e74c_abcd_Bc4_c4000_s16", 1), ("e78_abcd_B_c4000_s16", 2)],
                       "multiwoz": [("e77_mwoz_Bc4_c16000_s8", 1), ("e79_mwoz_B_c16000_s8", 2)]},
           "follow": False},
}
LR = {("a3B", "multiwoz"): "dry_e82mwoz_lr_mem_a3B", ("a3B", "star"): "dry_e81star_lr_a3B", ("a3B", "abcd"): "dry_e81abcd_lr_a3B",
      ("bge", "star"): "dry_e81star_lr_bge", ("bge", "abcd"): "dry_e81abcd_lr_bge", ("bge", "multiwoz"): "dry_e81multiwoz_lr_bge",
      ("C4", "star"): "dry_R21star_lr_mem_c4", ("C4", "abcd"): "dry_R21abcd_lr_mem_c4", ("C4", "multiwoz"): "dry_e77mwoz_lr_mem_c4"}
B2_LR, B2_TODAY = 0.3643158037776872, (0.3122706889523033 + 0.3092811523304797) / 2
# every TM setting other than clauses, T, s and seed, as in all 21 reference rows (checked 2026-10-21 before the freeze)
BASE_CFG = {"n_states": 256, "epochs": 40, "batch_size": 128, "neg_sampling": "hardest", "negation": True, "max_literals": 0, "budget_mode": "exact",
            "max_negated": 0, "budget_select": "rank", "clause_drop": 0.0, "half": True, "eval_every": 2}
GRID_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def tag(ds: str, agent: str, clauses: int, s: int, T: int) -> str:
    return f"e81_{ds}_{agent}_c{clauses}_s{s}_T{T}"


def job(ds: str, agent: str, clauses: int, s: int, T: int, seed: int) -> str:
    t = tag(ds, agent, clauses, s, T)
    return f"{AGENTS[agent]['run']} --dataset {ds} --clauses {clauses} --T {T} --s {s} --seed {seed} --tag {t} >> logs/{t}_seed{seed}.log 2>&1"


class Rows:
    def __init__(self) -> None:
        self.frozen = json.loads((ROOT / "logs" / "plan_e81_T_screen.frozen.json").read_text())["frozen_utc"]
        self.bits: dict = {}  # (agent, dataset) -> the dev bits' sha256 that every row of it must share (T and seed do not change the bits)
        self.r: dict = {}
        for line in (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("arm") == "tm" and d.get("tag"):
                self.r.setdefault((d["tag"], d["seed"]), []).append(d)

    def get(self, t: str, seed: int, agent: str, ds: str, clauses: int, T: int) -> dict:
        got = self.r.get((t, seed), [])
        if len(got) != 1:
            raise SystemExit(f"{t} seed {seed}: {len(got)} rows (need exactly 1)")
        d, a = got[0], AGENTS[agent]
        if not (d["dataset"] == ds and d["encoder"] == a["enc"] and list(d["parts"]) == a["parts"] and d["cfg"]["n_clauses"] == clauses
                and d["cfg"]["T"] == T and d["cfg"]["s"] == float(a["s"][ds]) and d["cfg"]["seed"] == seed):
            raise SystemExit(f"{t} seed {seed}: settings differ")
        if ({k: v for k, v in d["cfg"].items() if k not in ("n_clauses", "T", "s", "seed")} != BASE_CFG or d.get("lex_k") != 1000
                or d.get("init") != "half"):
            raise SystemExit(f"{t} seed {seed}: other TM settings differ from the reference rows'")
        h = d["bits_sha256"]["dev"]
        if self.bits.setdefault((agent, ds), h) != h:
            raise SystemExit(f"{t} seed {seed}: dev bits {h[:8]} differ from the other {agent} {ds} rows' {self.bits[(agent, ds)][:8]}")
        if agent == "a3B" and not all(d["static_info"][k]["table_file_sha256"] == A3B_SHA for k in ("xcust", "xrecent")):
            raise SystemExit(f"{t} seed {seed}: not on the a3B table")
        if t.startswith("e81_") and d["time"] <= self.frozen:
            raise SystemExit(f"{t} seed {seed}: written before the freeze")
        return d


def row4k(R: Rows, agent: str, ds: str, r: float) -> dict:
    s = AGENTS[agent]["s"][ds]
    if r == 1.0:
        return R.get(AGENTS[agent]["ref4k"][ds], 1, agent, ds, 4000, 4000)
    return R.get(tag(ds, agent, 4000, s, int(4000 * r)), 1, agent, ds, 4000, int(4000 * r))


def lr_acc(R: Rows, agent: str, ds: str) -> float | None:
    """LR's dev accuracy on the same bits (C chosen on dev from GRID_C), or None if it was not run."""
    p = ROOT / "results" / "explore" / "dryrun" / LR[(agent, ds)] / "seed0.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    if j["bits_sha256"]["dev"] != R.bits[(agent, ds)] or j["grid_c"] != GRID_C or j["arm"] != "lr":
        raise SystemExit(f"{LR[(agent, ds)]}: not LR on the same dev bits with the wide grid")
    return j["dev_acc"]


def epochs(d: dict) -> tuple[float, float, float]:
    v = [x["val_acc"] for x in d["history"]]
    return d["acc"], v[-1], sum(v[-10:]) / len(v[-10:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "screen", "final"])
    args = ap.parse_args()
    R = Rows()
    if args.stage == "smoke":
        bad = 0
        for agent, a in AGENTS.items():
            ref = R.get(a["ref4k"]["abcd"], 1, agent, "abcd", 4000, 4000)
            got = R.r.get((f"e81smoke_{agent}", 1), [])
            d = got[0] if len(got) == 1 else None
            ok = (d is not None and d["dataset"] == "abcd" and d["encoder"] == a["enc"] and list(d["parts"]) == a["parts"]
                  and (d["cfg"]["n_clauses"], d["cfg"]["T"], d["cfg"]["epochs"], d["cfg"]["s"]) == (200, 50, 2, float(a["s"]["abcd"]))
                  and d["time"] > R.frozen and d["bits_sha256"]["dev"] == ref["bits_sha256"]["dev"]
                  and (agent != "a3B" or all(d["static_info"][k]["table_file_sha256"] == A3B_SHA for k in ("xcust", "xrecent"))))
            bad += not ok
            print(f"smoke {agent}: {'ok' if ok else 'FAIL'} ({len(got)} row(s); needs 200 clauses, T 50, 2 epochs, s {a['s']['abcd']}, written after "
                  f"the freeze, the dev bits of {a['ref4k']['abcd']}" + (", the a3B table)" if agent == "a3B" else ")"))
        if bad:
            raise SystemExit(f"smoke: {bad} of {len(AGENTS)} failed")
        return
    picks = {}
    for agent in AGENTS:
        for ds in ("star", "abcd", "multiwoz"):
            acc = {r: row4k(R, agent, ds, r)["acc"] for r in RATIOS}
            best = max(RATIOS, key=lambda r: (acc[r], -RATIOS.index(r)))
            picks[(agent, ds)] = best if acc[best] - acc[1.0] >= 0.003 - 1e-12 else 1.0
            print(f"{agent:4} {ds:8} 4k s {AGENTS[agent]['s'][ds]}: " + ", ".join(f"T {int(4000 * r)} {acc[r]:.4f}" for r in RATIOS)
                  + f"; pick T {int(4000 * picks[(agent, ds)])}")
    if args.stage == "screen":
        out = ROOT / "logs" / "jobs_e81b.txt"
        if out.exists():
            raise SystemExit(f"{out} exists; not overwriting")
        lines = []
        for (agent, ds), r in picks.items():
            if r == 1.0 or not AGENTS[agent]["follow"]:
                continue
            s, c = AGENTS[agent]["s"][ds], FULL[ds]
            seeds = (1, 2) if c == 16000 else (2,)
            lines += [job(ds, agent, c, s, int(c * r), sd) for sd in seeds]
            if agent == "bge":
                lines.append(job(ds, agent, c, s, c, 2))  # bge-large's seed 2 at T = clauses (no dev seed 2 exists)
        out.write_text("\n".join(["# e81b (plan logs/plan_e81_T_screen.md): the follow-ups of the T picks that are not T = clauses. "
                                  "Written by scripts/e81_pick.py --stage screen. One at a time through seq_queue.sh."] + lines) + "\n")
        print(f"wrote {out.relative_to(ROOT)} ({len(lines)} jobs)")
        return
    for (agent, ds), r in picks.items():
        s, c = AGENTS[agent]["s"][ds], FULL[ds]
        lr = lr_acc(R, agent, ds)
        if r == 1.0 or not AGENTS[agent]["follow"]:
            have = [R.get(t, sd, agent, ds, c, c) for t, sd in AGENTS[agent]["reffull"][ds] if len(R.r.get((t, sd), [])) == 1]
            tm = sum(d["acc"] for d in have) / len(have)
            print(f"{agent:4} {ds:8}: pick T {int(4000 * r)} ({'no follow-up' if not AGENTS[agent]['follow'] else 'T = clauses'}); TM at T = clauses, "
                  f"{c} clauses, mean of {len(have)} seed(s) {tm:.4f}; TM minus LR " + (f"{100 * (tm - lr):+.2f}" if lr is not None else "(LR not run)"))
            continue
        ref = [R.get(t, sd, agent, ds, c, c) for t, sd in AGENTS[agent]["reffull"][ds]]
        if c == 16000:
            new = [R.get(tag(ds, agent, c, s, int(c * r)), sd, agent, ds, c, int(c * r)) for sd in (1, 2)]
        else:
            new = [row4k(R, agent, ds, r), R.get(tag(ds, agent, c, s, int(c * r)), 2, agent, ds, c, int(c * r))]
        e_new, e_ref = [epochs(d) for d in new], [epochs(d) for d in ref]
        diff = [100 * ((e_new[0][i] + e_new[1][i]) / 2 - (e_ref[0][i] + e_ref[1][i]) / 2) for i in range(3)]
        b1 = diff[0] >= 0.5 - 1e-9
        tm = (new[0]["acc"] + new[1]["acc"]) / 2
        print(f"{agent:4} {ds:8}: T {int(c * r)} against T {c} at {c} clauses, mean of seeds 1-2: {diff[0]:+.2f} (B1 needs >= +0.5: {'MET' if b1 else 'not met'}); "
              f"beside it: final epoch {diff[1]:+.2f}, last-10 mean {diff[2]:+.2f}; TM minus LR " + (f"{100 * (tm - lr):+.2f}" if lr is not None else "(LR not run)"))
    old = [R.get(t, sd, "a3B", "multiwoz", 16000, 16000) for t, sd in AGENTS["a3B"]["reffull"]["multiwoz"]]
    if lr_acc(R, "a3B", "multiwoz") != B2_LR or (old[0]["acc"] + old[1]["acc"]) / 2 != B2_TODAY:
        raise SystemExit("B2's constants differ from the rows")
    r = picks[("a3B", "multiwoz")]
    new = old if r == 1.0 else [R.get(tag("multiwoz", "a3B", 16000, 8, int(16000 * r)), sd, "a3B", "multiwoz", 16000, int(16000 * r)) for sd in (1, 2)]
    tm = (new[0]["acc"] + new[1]["acc"]) / 2
    gap_today, gap_now = 100 * (B2_TODAY - B2_LR), 100 * (tm - B2_LR)
    print(f"B2 (refined agent, MultiWOZ): TM minus LR {gap_now:+.2f} (today {gap_today:+.2f}; needs >= {gap_today / 2:+.2f}): "
          f"{'MET' if gap_now >= gap_today / 2 - 1e-9 else 'not met'}")
    l10 = [100 * ((epochs(a)[2] + epochs(b)[2]) / 2 - B2_LR) for a, b in (old, new)]
    print(f"B2 beside it (no rule): last-10 mean TM minus LR {l10[1]:+.2f} (today {l10[0]:+.2f}; half of it {l10[0] / 2:+.2f}): "
          f"{'steady' if l10[1] >= l10[0] / 2 - 1e-9 else 'not steady'}")


if __name__ == "__main__":
    main()
