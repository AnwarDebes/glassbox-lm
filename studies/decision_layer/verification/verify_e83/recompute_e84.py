"""Independent recompute of e84's gates (private; from the plan's text; read only; standard library)."""
import json, re, sys
from pathlib import Path
S, out = Path(sys.argv[1]), Path(sys.argv[2]).read_text()
rows = {}
for l in (S / "results/explore/abcd.jsonl").read_text().splitlines():
    d = json.loads(l)
    if d.get("arm") == "tm":
        rows.setdefault((d.get("tag"), d["seed"]), []).append(d)
def m2(t):
    v = []
    for sd in (1, 2):
        r = rows[(t, sd)]; assert len(r) == 1 and r[0]["cfg"]["neg_sampling"] == "random" and r[0]["cfg"]["n_clauses"] == 16000, (t, sd)
        h = [e["val_acc"] for e in r[0]["history"]]; v.append((r[0]["acc"], sum(h[-10:]) / 10))
    return [(v[0][i] + v[1][i]) / 2 for i in range(2)]
ref, bge, c4 = m2("e83_multiwoz_a3B_c16000_s8_nrand"), m2("e84_multiwoz_bge_c16000_s8_nrand"), m2("e84_multiwoz_C4_c16000_s8_nrand")
g = {"G-ref": [round(100 * (ref[i] - bge[i]), 2) for i in range(2)], "G-C4": [round(100 * (c4[i] - bge[i]), 2) for i in range(2)]}
print(g)
bad = [k for k, (b, l) in g.items() if not re.search(rf"^{k} .*best-dev {re.escape(f'{b:+.2f}')} \(needs >= -1.0\), last-10 {re.escape(f'{l:+.2f}')}: MET, steady", out, re.M)]
print("MATCH" if not bad else f"MISMATCH {bad}")
