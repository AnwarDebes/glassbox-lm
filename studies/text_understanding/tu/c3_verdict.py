#!/usr/bin/env python
"""Step 3c, the verdict (tu/PLAN_3c.md "Bar", fixed before any run; VAL only). Writes tu/c3/verdict_c3.json and .txt
(refuses to overwrite).

The base is tu/c3/base.json (written at the registration, after 3a's verdict): the base table, its feature cache, the
reader's s per set, and where the reference TM rows are.
TM probe: 3c-F's rows (tu/c3/results_c3_val.jsonl, tag "3c-tm", arm "F"), seeds 4-6 at the base's s, minus the
reference rows at the same s and seeds, paired by seed. The reference rows are the base's own rows (base.json "ref_tm")
when c3_feats.py's equivalence check found the rebuilt originals identical to the base cache for that set; otherwise
the rows of the rebuilt originals (tag "3c-ref", arm "orig").
LR probe: 3c-F's row (tu/c3/results_c3_lr.jsonl, arm "F") minus the fresh row on the base cache from the same job
(arm "orig", name "ref"), or on the rebuilt originals (name "ref_rebuilt") when the equivalence check failed for that
set; 3c-C's LR gain is descriptive.
The bar compares the unrounded gains with 0.5 - 1e-9. PASS when 3c-F minus the reference is at least +0.5 with BOTH
probes on at least 2 of the 3 sets; else FAIL. Descriptive: 3c-C's LR gain, the keep counts (filter_meta.json), and the
share of the gap to potion that is closed (TM: potion at its own 0b pick, seeds 4-6; LR: potion's W2 row).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
EMB = HERE.parent
C3 = HERE / "c3"
SETS = ["clinc150", "banking77", "hwu64"]
SEEDS = [4, 5, 6]
POT = "potion-base-32M"


def rows(p, tag=None):
    out = [json.loads(l) for l in open(p)]
    return [r for r in out if tag is None or r.get("tag") == tag]


def tm(rs, t, d, s, arm=None):
    v = []
    for sd in SEEDS:
        m = [r for r in rs if r["table"] == t and r["dataset"] == d and float(r["s"]) == float(s) and r["seed"] == sd
             and r["view"] == "existsT" and (arm is None or r.get("arm") == arm)]
        assert len(m) == 1, (t, d, s, sd, arm, len(m))
        v.append(m[0]["val_acc"] * 100)
    return np.array(v)


def lr(rs, d, arm=None, name=None, table=None):
    m = [r for r in rs if r["dataset"] == d and (arm is None or r.get("arm") == arm) and (name is None or r.get("name") == name)
         and (table is None or r.get("table") == table)]
    assert len(m) == 1, (d, arm, name, table, len(m))
    return m[0]["lr_val"] * 100


def main() -> None:
    oj, ot = C3 / "verdict_c3.json", C3 / "verdict_c3.txt"
    assert not oj.exists() and not ot.exists()
    base = json.loads((C3 / "base.json").read_text())
    fmeta = json.loads((C3 / "filter_meta.json").read_text())
    r3 = rows(C3 / "results_c3_val.jsonl")
    l3 = rows(C3 / "results_c3_lr.jsonl")
    rref = rows(Path(base["ref_tm"]["file"]), base["ref_tm"]["tag"])
    rpot = rows(Path(base["potion_tm"]["file"]), base["potion_tm"]["tag"])
    lw2 = rows(EMB / "w2" / "results_w2_lr.jsonl")
    res, lines, n_pass = {"base": base["base"]}, [], 0
    for d in SETS:
        s = base["s"][d]
        cm = json.loads((Path(base["feats_F"]) / f"{d}__p32tok_K512.meta.json").read_text())
        same = bool(cm["equivalence"]["identical"])
        ref = tm(rref, base["base"], d, s) if same else tm([r for r in r3 if r.get("tag") == "3c-ref"], base["base"], d, s, "orig")
        v = tm([r for r in r3 if r.get("tag") == "3c-tm"], base["base"], d, s, "F")
        pot = tm(rpot, POT, d, base["potion_tm"]["s"][d])
        lref, lF, lC = lr(l3, d, "orig", "ref" if same else "ref_rebuilt"), lr(l3, d, "F"), lr(l3, d, "C")
        lpot = lr(lw2, d, table=POT)
        g_tm, g_lr = float((v - ref).mean()), lF - lref
        gap_tm, gap_lr = float(pot.mean() - ref.mean()), lpot - lref
        ok = g_tm >= 0.5 - 1e-9 and g_lr >= 0.5 - 1e-9
        n_pass += ok
        res[d] = {"s": s, "reference_rows": "base" if same else "rebuilt originals", "n_orig": cm["n_orig"], "n_aug_F": cm["n_aug"],
                  "tm_F": round(float(v.mean()), 2), "tm_ref": round(float(ref.mean()), 2), "tm_gain": round(g_tm, 2),
                  "tm_paired": [round(float(x), 2) for x in v - ref], "lr_F": round(lF, 2), "lr_ref": round(lref, 2),
                  "lr_gain": round(g_lr, 2), "lr_C": round(lC, 2), "lr_gain_C": round(lC - lref, 2),
                  "tm_potion": round(float(pot.mean()), 2), "lr_potion": round(lpot, 2),
                  "tm_share_of_gap": round(g_tm / gap_tm, 3) if gap_tm > 0 else None,
                  "lr_share_of_gap": round(g_lr / gap_lr, 3) if gap_lr > 0 else None,
                  "filter": fmeta["sets"].get(d), "pass": bool(ok)}
        x = res[d]
        lines.append(f"{d} (s {s:g}; +{x['n_aug_F']:,} paraphrases on {x['n_orig']:,}): 3c-F TM {x['tm_F']:.2f} vs {x['tm_ref']:.2f} "
                     f"({x['tm_gain']:+.2f}, paired {x['tm_paired']}, share {x['tm_share_of_gap']}), LR {x['lr_F']:.2f} vs "
                     f"{x['lr_ref']:.2f} ({x['lr_gain']:+.2f}, share {x['lr_share_of_gap']}) -> {'meets' if ok else 'misses'} the bar; "
                     f"3c-C LR {x['lr_gain_C']:+.2f}; potion TM {x['tm_potion']:.2f}, LR {x['lr_potion']:.2f}; reference rows: {x['reference_rows']}")
    res["verdict"] = "PASS" if n_pass >= 2 else "FAIL"
    res["sets_meeting_bar"] = int(n_pass)
    lines.append(f"VERDICT 3c: {res['verdict']} ({n_pass} of 3 sets at +0.5 with both probes)")
    oj.write_text(json.dumps(res, indent=1))
    ot.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
