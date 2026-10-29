#!/usr/bin/env python
"""S2d: the CPU gates of A-Struct v2 on tune-hold (PLAN_part3.md section 3), per co-primary arm, plus what is
reported with no rule. Reads the outputs of S1, S2a, S2b (twin and the 4 families per arm) and S2c.

    chain/part3_cpu_job.sh A tm s2d_gates.py [--out DIR]

- K8 (A/A, equal size: rebuilt(2) minus rebuilt(3), amendment 1) and K9 (every learned arm on shuffled labels)
  must pass, else STOP (both arms). B minus each rebuild is reported with no rule.
- K3 (i): the better of the trees mix and the LR2 mix at least 0.0439 bits per token below B; K3 (ii): the LR1 mix
  at least 0.0291 below B. Point estimates on tune-hold; 95% dialogue-bootstrap intervals are printed beside them.
- An arm that passes K8, K9 and K3 goes on to its CPU TM pilot (K3c); an arm that stops runs no GPU job.
Exit code 0 when at least one arm goes on, 3 when both stop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from astruct import paths
from astruct.data import FIT, Corpus
from astruct.gates import M_025, M_2, M_3, k3, paired_bootstrap
from astruct.readout import PosTable
from astruct.store import load_rows

FAMILIES = ("tree", "lr1", "lr2", "hgb")
NAMES = {"tree": "tree twin", "lr1": "LR1", "lr2": "LR2", "hgb": "boosted trees"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths.OUT))
    ap.add_argument("--s0", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths.check_env()
    R = Path(args.root)
    out = paths.OutDir(Path(args.out or R / "s2d_gates"))
    log = paths.Log(out / "log.txt")
    s1 = json.loads((R / "s1" / "results.json").read_text())
    s2a = json.loads((R / "s2a" / "results.json").read_text())
    tw = json.loads((R / "s2b_twin" / "results.json").read_text())
    aa = json.loads((R / "s2c_aa" / "results.json").read_text())
    hold = load_rows(R / "s2a" / "hold_rows.npz", bits=False)
    c = Corpus.load(Path(args.s0 or R / "s0"))
    # reported, no rule: groups of tune-hold positions by the act-slot set's count in fit, and by the true token's type
    uS, nS = np.unique(c.a["S_hash"][c.replies(FIT)], return_counts=True)
    Sh = c.a["S_hash"][hold["pos_reply"]]
    i = np.minimum(np.searchsorted(uS, Sh), len(uS) - 1)
    s_cnt = np.where(uS[i] == Sh, nS[i], 0)
    x = hold["pos_x"]
    groups = {"act_set_unseen_in_fit": s_cnt == 0, "act_set_seen_once": s_cnt == 1, "act_set_seen_twice_or_more": s_cnt >= 2,
              "true_placeholder": c.is_ph[x], "true_end": x == c.END, "true_punct": c.is_punct[x],
              "true_word": ~c.is_ph[x] & (x != c.END) & ~c.is_punct[x]}

    def by_group(bits_, ref_):
        return {k: {"tokens": int(m.sum()), "minus_B": float((bits_ - ref_)[m].mean()) if m.any() else None} for k, m in groups.items()}
    dup = np.load(R / "s2a" / "dup.npz")["hold"]
    Bz = np.load(R / "s2b_twin" / "bits.npz")
    B, floor = Bz["B_hold"], Bz["floor_hold"]
    g = hold["pos_dlg"]
    pt = PosTable(hold, hold["p_true"])
    oracle = -np.log2(np.where(pt.in_C, pt.pfC, pt.pf_true))
    rep = {"floor": s1["floor"], "picks": s1["picks"], "twin_pick": tw["pick"],
           "tune_hold": {"floor": float(floor.mean()), "B": float(B.mean()), "oracle": float(oracle.mean()),
                         "recall_of_C": s2a["recall_of_C"]["hold"], "tokens": len(B), "dialogues": len(np.unique(g)),
                         "duplicate_share": float(dup.mean())},
           "K8": aa["K8"], "K9": {"count twin": tw["K9"]}, "arms": {},
           "groups_floor_minus_B": by_group(floor, B), "group_tokens": {k: int(m.sum()) for k, m in groups.items()},
           "constant_scorer": tw.get("constant_scorer")}
    lines = [(f"A-Struct v2 CPU gates on tune-hold ({len(B)} tokens, {len(np.unique(g))} dialogues); "
              f"floor {s1['floor']} {floor.mean():.5f}, B = floor + twin {tw['pick']} {B.mean():.5f} bits per token, "
              f"oracle bound {oracle.mean():.5f}, recall of C {s2a['recall_of_C']['hold']:.4f}")]
    lines.append(f"K8 (A/A, equal size): rebuilt(2) minus rebuilt(3) {aa['K8']['mean']:+.5f} [{aa['K8']['lo']:+.5f}, "
                 f"{aa['K8']['hi']:+.5f}] (needs within +-{M_025} and the interval covering 0) -> "
                 + ("pass" if aa["K8"]["pass"] else "STOP"))
    rp = aa["reported_no_rule"]
    for nm, key in (("rebuilt(2)", "B_minus_rebuilt2"), ("rebuilt(3)", "B_minus_rebuilt3")):
        d = rp[key]
        lines.append(f"  reported, no rule: B minus {nm} (5% less data and noise) {d['mean']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}]")
    rep["K8_reported_no_rule"] = rp
    k9_all = tw["K9"]["pass"]
    for arm in ("full", "nf"):
        fam_res = {}
        for fam in FAMILIES:
            d = R / f"s2b_{arm}_{fam}"
            r = json.loads((d / "results.json").read_text())
            hb = np.load(d / "bits.npz")[f"hold_{r['pick']}"]
            boot = paired_bootstrap(hb - B, g)
            boot_nd = paired_bootstrap((hb - B)[~dup], g[~dup]) if (~dup).any() else None
            fam_res[fam] = {"pick": r["pick"], "b": r["configs"][r["pick"]]["b"], "tune_hold_bits": float(hb.mean()),
                            "minus_B": boot, "minus_B_without_duplicates": boot_nd, "K9": r["K9"], "groups": by_group(hb, B)}
            rep["K9"][f"{arm}/{fam}"] = r["K9"]
            k9_all = k9_all and r["K9"]["pass"]
        g3 = k3(fam_res["hgb"]["minus_B"]["mean"], fam_res["lr2"]["minus_B"]["mean"], fam_res["lr1"]["minus_B"]["mean"])
        rep["arms"][arm] = {"families": fam_res, "K3": g3}
        lines.append(f"arm A-{arm}:")
        for fam in FAMILIES:
            x = fam_res[fam]
            lines.append(f"  {NAMES[fam]} ({x['pick']}, b {x['b']}): {x['tune_hold_bits']:.5f}; minus B {x['minus_B']['mean']:+.5f} "
                         f"[{x['minus_B']['lo']:+.5f}, {x['minus_B']['hi']:+.5f}]; K9 " + ("pass" if x["K9"]["pass"] else "FAIL"))
        lines.append(f"  K3 (i) better of trees and LR2 {g3['best_trees_lr2']:+.5f} (needs <= {-M_3}) -> "
                     + ("pass" if g3["i_pass"] else "stop") + f"; K3 (ii) LR1 {g3['d_lr1']:+.5f} (needs <= {-M_2}) -> "
                     + ("pass" if g3["ii_pass"] else "stop"))
    go = {}
    for arm in ("full", "nf"):
        go[arm] = bool(aa["K8"]["pass"] and k9_all and rep["arms"][arm]["K3"]["pass"])
        lines.append(f"A-{arm}: " + ("goes on to its CPU TM pilot (K3c)" if go[arm] else "STOP at the CPU gates (no GPU job)"))
    rep["K9_all_pass"] = bool(k9_all)
    rep["go_to_K3c"] = go
    lines.append("K9 (shuffled labels, every learned arm): " + ("pass" if k9_all else "FAIL: stop and find the leak"))
    rep["code_sha256"] = paths.code_hashes()
    txt = "\n".join(lines)
    log(txt)
    (out / "gates.json").write_text(json.dumps(rep, indent=1))
    (out / "words.txt").write_text(txt + "\n")
    out.done({"utc_end": paths.utc(), "go_to_K3c": go})
    if not any(go.values()):
        sys.exit(3)


if __name__ == "__main__":
    main()
