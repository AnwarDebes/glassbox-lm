"""Independent recompute of e81's picks and bars (private verification code; written from the frozen plan's text, not
from scripts/e81_pick.py; standard library only; read only).

    python3 -I recompute_e81.py <study root> [<e81_pick final output>]

From results/explore/abcd.jsonl it rebuilds:
- the T pick of every (agent, dataset): +0.3 points over T = clauses, ties to the larger T;
- B1's full-size differences at best-dev, the final epoch and the last-10 mean;
- B2 for the refined agent on MultiWOZ, against LR 0.3643158037776872.
If an e81_pick.py final output is given, every number printed there is compared with the recompute, and the result
line says MATCH or MISMATCH.
"""

import json
import re
import sys
from pathlib import Path

# from the plan's tables (logs/plan_e81_T_screen.md, frozen 2026-10-21T05:47:50Z)
S = {"a3B": {"star": 16, "abcd": 16, "multiwoz": 8}, "bge": {"star": 16, "abcd": 24, "multiwoz": 8}, "C4": {"star": 16, "abcd": 16, "multiwoz": 8}}
REF4K = {("a3B", "star"): "e82r3_star_Bt_c4000_s16", ("a3B", "abcd"): "e82_abcd_Bt_c4000_s16", ("a3B", "multiwoz"): "e82r3_multiwoz_Bt_c4000_s8",
         ("bge", "star"): "e74d_star_bge_c4000_s16", ("bge", "abcd"): "e75_abcd_bge_c4000_s24", ("bge", "multiwoz"): "e77_mwoz_bge_c4000_s8",
         ("C4", "star"): "e74c_star_Bc4_c4000_s16", ("C4", "abcd"): "e74c_abcd_Bc4_c4000_s16", ("C4", "multiwoz"): "e77_mwoz_Bc4_c4000_s8"}
REFFULL = {("a3B", "star"): [("e82_star_Bt_c16000_s16", 1), ("e82_star_Bt_c16000_s16", 2)],
           ("a3B", "abcd"): [("e82_abcd_Bt_c4000_s16", 1), ("e82_abcd_Bt_c4000_s16", 2)],
           ("a3B", "multiwoz"): [("e82_multiwoz_Bt_c16000_s8", 1), ("e82_multiwoz_Bt_c16000_s8", 2)],
           ("bge", "star"): [("e74d_star_bge_c16000_s16", 1), ("e81_star_bge_c16000_s16_T16000", 2)],
           ("bge", "abcd"): [("e75_abcd_bge_c4000_s24", 1), ("e81_abcd_bge_c4000_s24_T4000", 2)],
           ("bge", "multiwoz"): [("e77_mwoz_bge_c16000_s8", 1), ("e81_multiwoz_bge_c16000_s8_T16000", 2)],
           ("C4", "star"): [("e74c_star_Bc4_c16000_s16", 1), ("e78_star_B_c16000_s16", 2)],
           ("C4", "abcd"): [("e74c_abcd_Bc4_c4000_s16", 1), ("e78_abcd_B_c4000_s16", 2)],
           ("C4", "multiwoz"): [("e77_mwoz_Bc4_c16000_s8", 1), ("e79_mwoz_B_c16000_s8", 2)]}
FULL = {"star": 16000, "abcd": 4000, "multiwoz": 16000}
LR_A3B_MWOZ = 0.3643158037776872


def main() -> None:
    root = Path(sys.argv[1])
    rows = {}
    for line in (root / "results" / "explore" / "abcd.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append(d)

    def one(t, sd, T):
        v = rows.get((t, sd), [])
        assert len(v) == 1, (t, sd, len(v))
        assert v[0]["cfg"]["T"] == T, (t, sd, v[0]["cfg"]["T"], T)
        return v[0]

    def ep(d):
        h = [x["val_acc"] for x in d["history"]]
        return d["acc"], h[-1], sum(h[-10:]) / len(h[-10:])

    got = {}
    for (agent, ds), ref in REF4K.items():
        s = S[agent][ds]
        a = {4000: one(ref, 1, 4000)["acc"]}
        for T in (2000, 1000):
            a[T] = one(f"e81_{ds}_{agent}_c4000_s{s}_T{T}", 1, T)["acc"]
        best = 4000
        for T in (2000, 1000):  # strict '>' keeps the earlier (larger) T on a tie
            if a[T] > a[best]:
                best = T
        pick = best if a[best] - a[4000] >= 0.003 - 1e-12 else 4000
        got[(agent, ds, "pick")] = pick
        if pick == 4000 or agent == "C4":
            continue
        c = FULL[ds]
        Tf = c * pick // 4000
        new = [one(f"e81_{ds}_{agent}_c{c}_s{s}_T{Tf}", sd, Tf) for sd in (1, 2)] if c == 16000 else \
              [one(f"e81_{ds}_{agent}_c4000_s{s}_T{Tf}", 1, Tf), one(f"e81_{ds}_{agent}_c4000_s{s}_T{Tf}", 2, Tf)]
        ref_rows = [one(t, sd, c) for t, sd in REFFULL[(agent, ds)]]
        e_n, e_r = [ep(d) for d in new], [ep(d) for d in ref_rows]
        got[(agent, ds, "b1")] = [round(100 * ((e_n[0][i] + e_n[1][i]) / 2 - (e_r[0][i] + e_r[1][i]) / 2), 2) for i in range(3)]
    today = [one(t, sd, 16000)["acc"] for t, sd in REFFULL[("a3B", "multiwoz")]]
    pick = got[("a3B", "multiwoz", "pick")]
    if pick == 4000:
        tm = sum(today) / 2
    else:
        Tf = 4 * pick
        tm = sum(one(f"e81_multiwoz_a3B_c16000_s8_T{Tf}", sd, Tf)["acc"] for sd in (1, 2)) / 2
    gap_today, gap = 100 * (sum(today) / 2 - LR_A3B_MWOZ), 100 * (tm - LR_A3B_MWOZ)
    got["b2"] = (round(gap, 2), gap >= gap_today / 2 - 1e-9)
    for k, v in sorted(got.items(), key=str):
        print("recompute", k, v)
    if len(sys.argv) > 2:
        txt = Path(sys.argv[2]).read_text()
        bad = []
        for (agent, ds, what), v in [(k, v) for k, v in got.items() if k != "b2"]:
            if what == "pick":
                m = re.search(rf"^{re.escape(f'{agent:4} {ds:8}')} 4k s \d+: .*; pick T (\d+)$", txt, re.M)
                if not m or int(m.group(1)) != v:
                    bad.append((agent, ds, what, v, m.group(1) if m else None))
            else:
                m = re.search(rf"^{re.escape(f'{agent:4} {ds:8}')}: T \d+ against T \d+ at \d+ clauses, mean of seeds 1-2: ([+-]\d+\.\d+) .*final epoch ([+-]\d+\.\d+), last-10 mean ([+-]\d+\.\d+);", txt, re.M)
                if not m or [float(x) for x in m.groups()] != v:
                    bad.append((agent, ds, what, v, m.groups() if m else None))
        m = re.search(r"^B2 \(refined agent, MultiWOZ\): TM minus LR ([+-]\d+\.\d+) .*: (MET|not met)$", txt, re.M)
        if not m or float(m.group(1)) != got["b2"][0] or (m.group(2) == "MET") != got["b2"][1]:
            bad.append(("b2", got["b2"], m.groups() if m else None))
        print("MATCH" if not bad else f"MISMATCH {bad}")


if __name__ == "__main__":
    main()
