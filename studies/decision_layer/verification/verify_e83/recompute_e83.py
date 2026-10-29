"""Independent recompute of e83's numbers (private; written from the plan's text and the note, not from e83_pick.py's
code; read only). Usage: python -I recompute_e83.py <study root> <e83_pick_final.out>"""
import gzip, json, re, sys
from pathlib import Path
import numpy as np

S, out = Path(sys.argv[1]), Path(sys.argv[2]).read_text()
rows = {}
for l in (S / "results/explore/abcd.jsonl").read_text().splitlines():
    d = json.loads(l)
    if d.get("arm") == "tm":
        rows.setdefault((d.get("tag"), d["seed"]), []).append(d)
def one(t, sd):
    v = rows[(t, sd)]; assert len(v) == 1, (t, sd, len(v)); return v[0]
def st(d):
    h = [e["val_acc"] for e in d["history"]]; return d["acc"], h[-1], sum(h[-10:]) / 10
got, bad = {}, []
ref = st(one("e83_multiwoz_a3B_c4000_s8_ref", 2))
for a in ("nrand", "nmix50", "drop50"):
    x = st(one(f"e83_multiwoz_a3B_c4000_s8_{a}", 2))
    got[a] = (round(100 * (x[0] - ref[0]), 2), round(100 * (x[2] - ref[2]), 2))
    m = re.search(rf"^  {a}\s+[\d.]+ / [\d.]+ / [\d.]+\s+against REF: best-dev ([+-][\d.]+), last-10 ([+-][\d.]+)$", out, re.M)
    if not m or (float(m.group(1)), float(m.group(2))) != got[a]: bad.append((a, got[a], m.groups() if m else None))
qual = [a for a, (b, l) in got.items() if b >= 0.5 - 1e-9 and l >= 0.5 - 1e-9]
pick = max(qual, key=lambda a: (got[a][1], -["nrand", "nmix50", "drop50"].index(a)))
def mean2(t, c=None):
    v = [st(one(t, sd)) for sd in (1, 2)]; return [(v[0][i] + v[1][i]) / 2 for i in range(3)]
def diff(t, r):
    n, rr = mean2(t), mean2(r); return [round(100 * (n[i] - rr[i]), 2) for i in range(3)]
c1 = diff(f"e83_multiwoz_a3B_c16000_s8_{pick}", "e82_multiwoz_Bt_c16000_s8")
lr = json.loads((S / "results/explore/dryrun/dry_e82mwoz_lr_mem_a3B/seed0.json").read_text())["dev_acc"]
tm = mean2(f"e83_multiwoz_a3B_c16000_s8_{pick}")[0]
c2 = round(100 * (tm - lr), 2)
c3 = {ds: diff(f"e83_{ds}_a3B_c{c}_s16_{pick}", r) for ds, c, r in (("star", 16000, "e82_star_Bt_c16000_s16"), ("abcd", 4000, "e82_abcd_Bt_c4000_s16"))}
with gzip.open(S / "data/multiwoz/examples_train.json.gz", "rt") as f:
    names = json.load(f)["buttons"]
six = ["general-bye", "general-bye+general-welcome", "Train-Inform", "Train-Inform+Train-OfferBook", "Booking-Book", "Booking-Book+general-reqmore"]
def pooled(a):
    z = np.load(S / f"results/explore/e83_preds/e83_multiwoz_a3B_c4000_s8_{a}_seed2.npz"); y, p = z["y_val"].astype(int), z["preds"][-1].astype(int)
    m = np.isin(y, [names.index(x) for x in six]); return 100 * float((p[m] == y[m]).mean())
p1 = {a: round(pooled(a) - pooled("ref"), 2) for a in ("nrand", "nmix50")}
print("screen", got, "pick", pick, "C1", c1, "C2", c2, "C3", c3, "P1", p1)
E = lambda x: re.escape(f"{x:+.2f}")  # a signed number as a literal in a pattern
checks = [(rf"pick: {pick}$", None),
          (rf"best-dev {E(c1[0])}, last-10 mean {E(c1[2])} \(each needs >= \+0.5\): MET; beside it: final epoch {E(c1[1])}", None),
          (rf"TM minus LR {E(c2)} .*: MET", None),
          (rf"C3 \({pick} on star.*best-dev {E(c3['star'][0])} \(needs >= -0.3\): not met", None),
          (rf"C3 \({pick} on abcd.*best-dev {E(c3['abcd'][0])} \(needs >= -0.3\): not met", None)] + \
         [(rf"P1 \({a}\).*: {E(p1[a])} \(needs >= \+2.0\): not met", None) for a in p1]
for pat, _ in checks:
    if not re.search(pat, out, re.M): bad.append(pat)
print("MATCH" if not bad else f"MISMATCH {bad}")
