#!/usr/bin/env python
"""e80's dry check (plan logs/plan_e80_2d_recent.md, frozen 2026-10-20T20:53:40Z; dated note
logs/plan_e80_note_2026-10-25.md): the five dry jobs of logs/jobs_e80_dry.txt (the frozen job lines with --clauses 200
--epochs 2 and tags e80dry_*) and the five reference rows that scripts/e80_check.py reads. Read only, CPU only, no
rule; no accuracy is printed. Written 2026-10-25, before any e80 number. No test file is read.

    e80_dry_check.py     # after logs/jobs_e80_dry.txt; its last line must be "all dry checks passed"

Checks, stopping at the first failure:
- the note's freeze record holds the SHA-256 of the note's text;
- one tm row per dry tag and seed (e80dry_abcd_BX and e80dry_star_BX seeds 1 and 2, e80dry_mwoz_BX seed 1), written
  after the note's freeze;
- the settings: dataset, encoder none, parts view B + xcrecent, lex_k 1000, 200 clauses, T 200, 2 epochs, the cell's
  s, the seed;
- the bits: n_bits 10,361 (STAR), 9,657 (ABCD) and 9,315 (MultiWOZ), the self-test's counts; static_info.xcrecent
  with n_bits 3,134, 2,973 and 2,735 on the C4 table file (e1771d9e...); the two seeds of a dataset share their train
  and dev bits;
- the reference rows: each reference (tag, seed) has exactly one tm row, at its pinned line of
  results/explore/abcd.jsonl, and that line's text (without its newline) hashes to the pinned SHA-256. The file only
  grows, so the lines stay where they are.
Tested by tests/test_e80_dry_check.py (made-up dry rows beside copies of the real reference lines).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTE = "plan_e80_note_2026-10-25"
BN = ["mem", "words", "xcust", "xrecent", "xcrecent"]
TABLE = "e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a"  # data/tables/c4words_S1000_v1/table.npz
DRY = {"abcd": {"tag": "e80dry_abcd_BX", "seeds": (1, 2), "s": 16.0, "n_bits": 9657, "xcrecent": 2973},
       "star": {"tag": "e80dry_star_BX", "seeds": (1, 2), "s": 16.0, "n_bits": 10361, "xcrecent": 3134},
       "multiwoz": {"tag": "e80dry_mwoz_BX", "seeds": (1,), "s": 8.0, "n_bits": 9315, "xcrecent": 2735}}
REFS = {("e74c_star_Bc4_c16000_s16", 1): (86, "f24aaf84ebc100e21d69feb23ccef489d647498a184b8472d26b92469fdd5211"),
        ("e74c_abcd_Bc4_c4000_s16", 1): (87, "2a466ac6e48a0fc3660509f45e73333e1852557264596cf7070072555ec8b3c6"),
        ("e77_mwoz_Bc4_c16000_s8", 1): (106, "e216c044c4aba9987ed2c7763679f768835b7bfc16e7cebaa202d23eb0fcffdb"),
        ("e78_abcd_B_c4000_s16", 2): (109, "df4e529115d73dafc69c05a778e4c4e819eeec9689c592f66d5e6128d2dc1d76"),
        ("e78_star_B_c16000_s16", 2): (112, "5a697fed399f7ff4e69f52ac74672c49da656eb3f167af5d2431ab46d47b33a2")}


def refuse(msg: str) -> None:
    raise SystemExit(f"e80_dry_check: {msg}")


def main() -> None:
    rec, txt = ROOT / "logs" / f"{NOTE}.frozen.json", ROOT / "logs" / f"{NOTE}.md"
    if not rec.exists():
        refuse(f"{rec.name} is missing: the dry run is checked only after the dated note of 2026-10-25 is frozen")
    r = json.loads(rec.read_text())
    if not txt.exists() or hashlib.sha256(txt.read_bytes()).hexdigest() != r["sha256"]:
        refuse(f"logs/{NOTE}.md does not hash to its freeze record")
    frozen = r["frozen_utc"]
    lines = (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines()
    rows: dict = {}
    for n, line in enumerate(lines, 1):
        d = json.loads(line)
        if d.get("arm") == "tm" and d.get("tag"):
            rows.setdefault((d["tag"], d["seed"]), []).append((n, line, d))
    for ds, g in DRY.items():
        got = {}
        for sd in g["seeds"]:
            hits = rows.get((g["tag"], sd), [])
            if len(hits) != 1:
                refuse(f"{g['tag']} seed {sd}: {len(hits)} rows (need exactly 1)")
            d = hits[0][2]
            if d["time"] <= frozen:
                refuse(f"{g['tag']} seed {sd}: written before the note's freeze")
            c = d["cfg"]
            if not (d["dataset"] == ds and d["encoder"] == "none" and list(d["parts"]) == BN and d.get("lex_k") == 1000
                    and (c["n_clauses"], c["T"], c["epochs"], c["s"], c["seed"]) == (200, 200, 2, g["s"], sd)):
                refuse(f"{g['tag']} seed {sd}: settings differ from the dry job line's")
            xc = d.get("static_info", {}).get("xcrecent", {})
            if d.get("n_bits") != g["n_bits"] or xc.get("n_bits") != g["xcrecent"] or xc.get("table_file_sha256") != TABLE:
                refuse(f"{g['tag']} seed {sd}: bits {d.get('n_bits')} with xcrecent {xc.get('n_bits')} (table {str(xc.get('table_file_sha256'))[:8]}),"
                       f" not {g['n_bits']} with {g['xcrecent']} on the C4 table")
            got[sd] = d["bits_sha256"]
            print(f"ok {g['tag']} seed {sd}: settings; {g['n_bits']} bits, xcrecent {g['xcrecent']} on the C4 table", flush=True)
        if len(got) == 2 and got[1] != got[2]:
            refuse(f"{g['tag']}: the two seeds' train and dev bits differ")
        if len(got) == 2:
            print(f"ok {g['tag']}: the two seeds share their train and dev bits", flush=True)
    for (tag, sd), (n, want) in REFS.items():
        hits = rows.get((tag, sd), [])
        if len(hits) != 1 or hits[0][0] != n or hashlib.sha256(hits[0][1].encode()).hexdigest() != want:
            refuse(f"reference {tag} seed {sd}: not exactly one row at line {n} with the pinned SHA-256 ({want[:8]})")
    print(f"ok the {len(REFS)} reference rows: one row each, at their pinned lines, unchanged", flush=True)
    print("all dry checks passed")


if __name__ == "__main__":
    main()
