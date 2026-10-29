"""Licence table from per-version PyPI JSON files. Args: deps_list.txt deps_dir out.tsv"""
import json
import sys
from pathlib import Path

lst, d, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = []
for line in lst.read_text().split():
    name, ver = line.split("==")
    j = json.loads((d / f"{name}-{ver}.json").read_text())
    info = j["info"]
    expr = info.get("license_expression") or ""
    lic = (info.get("license") or "").strip().splitlines()
    lic1 = lic[0][:60] if lic else ""
    cls = "; ".join(c.replace("License :: ", "").replace("OSI Approved :: ", "") for c in info.get("classifiers", []) if c.startswith("License"))
    best = expr or cls or lic1 or "UNKNOWN"
    rows.append((name, ver, best, expr, cls, lic1))
with out.open("w") as f:
    f.write("name\tversion\tlicence (best)\tlicense_expression\tclassifiers\tlicense field (1st line)\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
for r in rows:
    print(f"{r[0]:28s} {r[1]:22s} {r[2][:70]}")
