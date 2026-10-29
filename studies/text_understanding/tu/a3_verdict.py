#!/usr/bin/env python
"""Step 3a, the verdict (tu/PLAN_3a.md "Bar", fixed before any run; VAL only). Writes tu/a3/verdict_a3.json and .txt
(refuses to overwrite).

TM probe: the C4 view's s picked by 0b (tu/b0_picks.json), seeds 4-6; the table's rows (tag "3a-tm",
tu/a3/results_a3_val.jsonl) minus 0b's ppw_S1000 rows at the same s and seeds (tag "0b-seeds"), paired by seed.
LR probe: the table's row (tu/a3/results_a3_lr.jsonl) minus ppw_S1000's FRESH row from the same job (a3_lr.py on
ppw_S1000's own feature files; review of 19:20Z, fix 3); its W2 row (w2/results_w2_lr.jsonl) is reported beside it.
The bar compares the unrounded gains with 0.5 - 1e-9 (decided 19:20Z, fix 2).
PASS when a3B minus ppw_S1000 is at least +0.5 with BOTH probes on at least 2 of the 3 sets; else FAIL. Descriptive:
a3A, and the share of the C4-to-potion gap closed (TM: potion at its own 0b pick, seeds 4-6; LR: potion's W2 row).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
A3 = HERE / "a3"
SETS = ["clinc150", "banking77", "hwu64"]
SEEDS = [4, 5, 6]
C4, POT = "ppw_S1000_d500_e0.5", "potion-base-32M"
ARMS = ["a3B_d768", "a3A_d768"]


def rows(p: Path, tag: str | None = None):
    out = [json.loads(l) for l in open(p)]
    return [r for r in out if tag is None or r.get("tag") == tag]


def tm(rs, t, d, s):
    v = []
    for sd in SEEDS:
        m = [r for r in rs if r["table"] == t and r["dataset"] == d and float(r["s"]) == float(s) and r["seed"] == sd
             and r["view"] == "existsT"]
        assert len(m) == 1, (t, d, s, sd, len(m))
        v.append(m[0]["val_acc"] * 100)
    return np.array(v)


def lr(rs, t, d):
    m = [r for r in rs if r["table"] == t and r["dataset"] == d]
    assert len(m) == 1, (t, d, len(m))
    return m[0]["lr_val"] * 100


def main() -> None:
    oj, ot = A3 / "verdict_a3.json", A3 / "verdict_a3.txt"
    assert not oj.exists() and not ot.exists()
    picks = json.loads((HERE / "b0_picks.json").read_text())
    r0b = rows(HERE / "results_0b_val.jsonl", "0b-seeds")
    r3 = rows(A3 / "results_a3_val.jsonl", "3a-tm")
    l3 = rows(A3 / "results_a3_lr.jsonl")
    lw2 = rows(EMB / "w2" / "results_w2_lr.jsonl")
    res, lines, n_pass = {}, [], 0
    for d in SETS:
        s_c4, s_pot = picks[f"{C4}|{d}"]["pick"], picks[f"{POT}|{d}"]["pick"]
        ref, pot = tm(r0b, C4, d, s_c4), tm(r0b, POT, d, s_pot)
        lref, lpot, lref_w2 = lr(l3, C4, d), lr(lw2, POT, d), lr(lw2, C4, d)
        res[d] = {"s_c4": s_c4, "s_potion": s_pot, "tm_ppw": round(float(ref.mean()), 2), "tm_potion": round(float(pot.mean()), 2),
                  "lr_ppw": round(lref, 2), "lr_ppw_w2_row": round(lref_w2, 2), "lr_potion": round(lpot, 2)}
        gains = {}
        for a in ARMS:
            v = tm(r3, a, d, s_c4)
            g_tm, g_lr = float((v - ref).mean()), lr(l3, a, d) - lref
            gains[a] = (g_tm, g_lr)
            gap_tm, gap_lr = float(pot.mean() - ref.mean()), lpot - lref
            res[d][a] = {"tm": round(float(v.mean()), 2), "tm_gain": round(g_tm, 2), "tm_paired": [round(float(x), 2) for x in v - ref],
                         "lr": round(lr(l3, a, d), 2), "lr_gain": round(g_lr, 2),
                         "tm_share_of_gap": round(g_tm / gap_tm, 3) if gap_tm > 0 else None,
                         "lr_share_of_gap": round(g_lr / gap_lr, 3) if gap_lr > 0 else None}
        ok = gains["a3B_d768"][0] >= 0.5 - 1e-9 and gains["a3B_d768"][1] >= 0.5 - 1e-9
        res[d]["pass"] = bool(ok)
        n_pass += ok
        b, a = res[d]["a3B_d768"], res[d]["a3A_d768"]
        lines.append(f"{d} (s {s_c4:g}): a3B TM {b['tm']:.2f} vs ppw {res[d]['tm_ppw']:.2f} ({b['tm_gain']:+.2f}, paired {b['tm_paired']}, "
                     f"share {b['tm_share_of_gap']}), LR {b['lr']:.2f} vs {res[d]['lr_ppw']:.2f} ({b['lr_gain']:+.2f}, share "
                     f"{b['lr_share_of_gap']}) -> {'meets' if ok else 'misses'} the bar; a3A TM {a['tm_gain']:+.2f}, LR {a['lr_gain']:+.2f}; "
                     f"potion TM {res[d]['tm_potion']:.2f} (s {s_pot:g}), LR {res[d]['lr_potion']:.2f}; ppw LR fresh {lref:.2f}, W2 row {lref_w2:.2f}")
    res["verdict"] = "PASS" if n_pass >= 2 else "FAIL"
    res["sets_meeting_bar"] = int(n_pass)
    lines.append(f"VERDICT 3a: {res['verdict']} ({n_pass} of 3 sets at +0.5 with both probes)")
    oj.write_text(json.dumps(res, indent=1))
    ot.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
