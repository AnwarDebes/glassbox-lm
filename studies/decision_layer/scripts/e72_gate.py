#!/usr/bin/env python
"""e72 gate G-c4 (read only; plan logs/plan_e72_count_table.md): recompute the gate from the rows.

    e72_gate.py      # prints the checks and the verdict per dataset

Per dataset, the e72 TM row (results/explore/abcd.jsonl, tag e72_<ds>_Bc4_c<n>, arm tm) must reach the potion
view-B row (e60_star_B_c16000 0.6343, e60_abcd_B_c4000 0.8325, arm tm) minus 2.0 points. Also checked: the row ran
with encoder none, view B's parts, R19's settings and seed 1; its static_info names the C4 table (file SHA-256
e1771d9e) for both fields; its bits equal the smoke run's (tag e72_smoke_<ds>); the e72 rows sit after the plan's
freeze time. Reported with no rule: the cost against potion and bge-large dev, n_bits, the best epoch, the train time,
the GPU peak, and the ABCD no-memory twin (e72_abcd_Bc4nomem_c4000) against potion's (e61_abcd_Bnomem_c4000).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "results" / "explore" / "abcd.jsonl"
TABLE_FILE = "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a"
GATE = {"abcd": ("e72_abcd_Bc4_c4000", "e60_abcd_B_c4000", 4000, 0.8398), "star": ("e72_star_Bc4_c16000", "e60_star_B_c16000", 16000, 0.6394)}
MARGIN = 0.02


def main() -> None:
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    tm = {}
    for d in rows:
        if d.get("arm") == "tm" and d.get("tag", "").startswith(("e72_", "e60_", "e61_")):
            tm.setdefault(d["tag"], []).append(d)
    frozen = json.loads((ROOT / "logs" / "plan_e72_count_table.frozen.json").read_text())["frozen_utc"]
    print(f"plan frozen {frozen}")
    for ds, (tag, ref_tag, n, bge) in GATE.items():
        got, ref = tm.get(tag, []), tm.get(ref_tag, [])
        smoke = tm.get(f"e72_smoke_{ds}", [])
        if len(got) != 1 or len(ref) != 1:
            print(f"{ds}: rows {tag} {len(got)}, {ref_tag} {len(ref)}: NOT DECIDED (need exactly one of each)")
            continue
        r, p = got[0], ref[0]
        checks = {
            "encoder none": r["encoder"] == "none",
            "parts view B": tuple(r["parts"]) == ("mem", "words", "xcust", "xrecent"),
            "settings": r["cfg"]["n_clauses"] == n and r["cfg"]["T"] == n and r["cfg"]["s"] == 8.0 and r["cfg"]["epochs"] == 40
            and r["init"] == "half" and r["seed"] == 1 and r["cfg"]["seed"] == 1 and r["lex_k"] == 1000,
            "C4 table in both fields": all(r["static_info"][f]["table_file_sha256"] == TABLE_FILE for f in ("xcust", "xrecent")),
            "bits equal the smoke's": len(smoke) == 1 and smoke[0]["bits_sha256"] == r["bits_sha256"],
            "after the freeze": r["time"] > frozen,
        }
        ok_checks = all(checks.values())
        diff = r["acc"] - p["acc"]
        passed = r["acc"] >= p["acc"] - MARGIN - 1e-12
        print(f"{ds}: e72 {r['acc']:.4f} vs potion {p['acc']:.4f}: {100 * diff:+.2f} points (gate: at least {p['acc'] - MARGIN:.4f}) -> "
              f"{'PASS' if passed and ok_checks else 'FAIL' if ok_checks else 'CHECKS FAILED'}")
        print(f"  checks: {checks}")
        print(f"  no rule: vs bge-large dev {bge:.4f}: {100 * (r['acc'] - bge):+.2f}; n_bits {r['n_bits']} (potion {p['n_bits']}); "
              f"xcust {r['static_info']['xcust']['n_bits']}, xrecent {r['static_info']['xrecent']['n_bits']}; best epoch {r['best_epoch']} "
              f"(potion {p['best_epoch']}); train {r['train_s']} s (potion {p['train_s']}); GPU peak {r['peak_gpu_mb']} MB")
    twin, ptwin = tm.get("e72_abcd_Bc4nomem_c4000", []), tm.get("e61_abcd_Bnomem_c4000", [])
    if len(twin) == 1 and len(ptwin) == 1:
        print(f"no rule: ABCD no-memory twin e72 {twin[0]['acc']:.4f} vs potion {ptwin[0]['acc']:.4f}: {100 * (twin[0]['acc'] - ptwin[0]['acc']):+.2f} points")


if __name__ == "__main__":
    main()
