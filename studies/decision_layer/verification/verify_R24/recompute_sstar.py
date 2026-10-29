"""Independent s* recompute for R24 from the e85 dev rows (results/explore/abcd.jsonl), by e77's rule as e85's plan
words it: 4k, seed 1, grid [8, 12, 16, 24]; a new s needs at least +0.3 points over s 8; an exact tie goes to the first
of the grid; a moved pick is checked at 16k (not needed when nothing moved). Standard library only."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"e85_' in l]
acc_key = next(k for k in ("dev_acc", "acc", "best_dev_acc") if k in rows[0])
grid = [8, 12, 16, 24]
by = {}
for r in rows:
    t = r["tag"]
    agent = t.split("_")[2]
    s = int(t.split("_s")[-1].split("_")[0])
    assert r["seed"] == 1 and "_c4000_" in t, t
    by.setdefault(agent, {})
    assert s not in by[agent], ("duplicate row", t)
    by[agent][s] = r[acc_key]
print(f"{len(rows)} e85 rows; accuracy key {acc_key}")
for agent, d in sorted(by.items()):
    assert sorted(d) == grid, (agent, sorted(d))
    ref = d[8]
    gains = {s: 100 * (d[s] - ref) for s in grid}
    best = max(grid, key=lambda s: (round(gains[s], 12), -grid.index(s)))
    pick = best if gains[best] >= 0.3 - 1e-9 and best != 8 else 8
    print(agent, " ".join(f"s{s} {d[s]:.4f} ({gains[s]:+.2f})" for s in grid), f"-> 4k pick s {pick}" + ("" if pick == 8 else " (needs the 16k check)"))
