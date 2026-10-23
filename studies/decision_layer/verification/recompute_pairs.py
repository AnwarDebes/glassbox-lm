"""Independent recompute of paired differences for a registered decision-layer test (read only).

Per test action turn, each arm's correctness (pred == y) is averaged over its seeds; a difference A - B is the mean
over action turns; its 95% interval comes from a bootstrap over test dialogues (all turns of a dialogue together),
10,000 resamples with a separate RNG (PCG64, seed 20261019), so it checks the frozen interval (seed 0) without copying it.

    recompute_pairs.py FOLDER A:B[:BAR[:LOWBAR]] ...
      FOLDER  results/confirm/<registration> (relative to the study) or an absolute path
      BAR     the point-estimate bar for A - B in points (e.g. 1.0 or -2.0); LOWBAR, if given, the bar for the lower end
              ('pos' = the interval must lie above 0)

Example (R22 as proposed):
    recompute_pairs.py R22star tm_mem_c4_s16:tm_mem_c4:1.0:pos tm_mem_c4_s16:lr_mem_c4:-1.0:-1.0 \
        tm_mem_c4_s16:tm_mem_bge_s16:-2.0:-2.0
Also checks: labels and dialogue ids equal across all arms and seeds; prints seeds per arm and each arm's accuracy.
"""
import sys
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parents[1]


def load(folder):
    seeds = sorted(int(p.stem[4:]) for p in folder.glob("seed*.npz"))
    zs = {k: np.load(folder / f"seed{k}.npz") for k in seeds}
    return seeds, zs


def boot(d, cid, n=10000, seed=20261019):
    uniq, inv = np.unique(cid, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.Generator(np.random.PCG64(seed))
    parts = []
    for _ in range(2):
        idx = rng.integers(0, len(uniq), size=(n // 2, len(uniq)))
        parts.append(s[idx].sum(1) / c[idx].sum(1))
    bs = np.concatenate(parts)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), len(uniq)


def main():
    folder = Path(sys.argv[1])
    if not folder.is_absolute():
        folder = S / "results" / "confirm" / sys.argv[1]
    specs = [a.split(":") for a in sys.argv[2:]]
    arms = sorted({x for sp in specs for x in sp[:2]})
    corr, meta, problems = {}, None, []
    for a in arms:
        seeds, zs = load(folder / a)
        if not seeds:
            problems.append(f"{a}: no seed files")
            continue
        y0, c0 = zs[seeds[0]]["y"], zs[seeds[0]]["convo_id"]
        for k in seeds:
            if not (np.array_equal(zs[k]["y"], y0) and np.array_equal(zs[k]["convo_id"], c0)):
                problems.append(f"{a} seed {k}: labels or dialogue ids differ")
        if meta is None:
            meta = (y0, c0)
        elif not (np.array_equal(meta[0], y0) and np.array_equal(meta[1], c0)):
            problems.append(f"{a}: labels or dialogue ids differ from the first arm")
        C = np.stack([(zs[k]["pred"] == zs[k]["y"]).astype(float) for k in seeds])
        corr[a] = C.mean(0)
        print(f"{a:22s} seeds {seeds}  acc {100 * C.mean():.2f}  per seed {[round(float(100 * x), 2) for x in C.mean(1)]}")
    for sp in specs:
        a, b = sp[0], sp[1]
        if a not in corr or b not in corr:
            print(f"{a} - {b}: missing arm")
            continue
        m, lo, hi, nd = boot(corr[a] - corr[b], meta[1])
        line = f"{a + ' - ' + b:44s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  ({nd} dialogues, {len(meta[1])} turns)"
        if len(sp) >= 3:
            bar = float(sp[2])
            ok = m >= bar - 1e-9
            line += f"  point >= {bar:+.1f}: {'MET' if ok else 'NOT MET'}"
            if len(sp) >= 4:
                if sp[3] == "pos":
                    line += f"; interval above 0: {'yes' if lo > 0 else 'no'}"
                else:
                    lb = float(sp[3])
                    line += f"; lower end >= {lb:+.1f} (strong): {'yes' if lo >= lb - 1e-9 else 'no'}"
        print(line)
    print("problems:", problems if problems else "none")


if __name__ == "__main__":
    main()
