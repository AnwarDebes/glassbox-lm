"""Independent recompute of R21's pass rules (read only; writes nothing in the repo).

Written from registrations/R21_c4_table_agent.md, not from report_abcd.py:
- Per test action turn, each arm's correctness (pred == y) is averaged over its seeds.
- A difference A - B is the mean over action turns of (A - B); its 95% interval comes from a bootstrap over test
  dialogues (all action turns of a dialogue resampled together), 10,000 resamples. A separate RNG (PCG64, seed 20261019),
  so the interval is a check of the frozen one (seed 0), not a copy.
- R21.1: tm_mem_c4 - tm_mem_static_R19 >= -2.0 points on the point estimate; "strong" if the lower end >= -2.0.
- R21.2: tm_mem_c4 - tm_nomem_c4 >= +5.0 points, with the interval above 0.
- A dataset is met when both hold; the headline needs both datasets.
Also checks: seed counts, labels and dialogue ids equal across arms, cfg of the new arms equal to R19's except the
seed, encoder 'none', parts, registration field, and that every new seed file was written after the freeze.

    recompute_r21.py [--only-refs]     # --only-refs: self-test on the copied references (no R21 test number read)
"""
import json
import sys
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parents[1]
FREEZE = "2026-10-19T05:48:24Z"
SEEDS_EXPECTED = {"tm_mem_c4": 5, "tm_nomem_c4": 5, "lr_mem_c4": 1, "tm_mem_static_R19": 5, "tm_nomem_static_R19": 5,
                  "lr_mem_static_R19": 1, "tm_mem_R18": 5, "tm_nomem_R18": 5, "lr_mem_R18": 1,
                  "tm_mem_R9b": 10, "tm_nomem_R9b": 10, "lr_mem_R9b": 1}
PARTS = {"tm_mem_c4": "mem,words,xcust,xrecent", "tm_nomem_c4": "words,xcust,xrecent", "lr_mem_c4": "mem,words,xcust,xrecent"}
ONLY_REFS = "--only-refs" in sys.argv
NEW = ("tm_mem_c4", "tm_nomem_c4", "lr_mem_c4")


def load(folder):
    seeds = sorted(int(p.stem[4:]) for p in folder.glob("seed*.json"))
    js = {k: json.loads((folder / f"seed{k}.json").read_text()) for k in seeds}
    zs = {k: np.load(folder / f"seed{k}.npz") for k in seeds}
    return seeds, js, zs


def boot(d, cid, n=10000, seed=20261019):
    uniq, inv = np.unique(cid, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.Generator(np.random.PCG64(seed))
    lo_hi = []
    for _ in range(2):  # two halves, so memory stays small
        idx = rng.integers(0, len(uniq), size=(n // 2, len(uniq)))
        lo_hi.append(s[idx].sum(1) / c[idx].sum(1))
    bs = np.concatenate(lo_hi)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), len(uniq)


problems = []
verdicts = {}
for ds, refs, cfg_ref in (("star", ("tm_mem_static_R19", "tm_nomem_static_R19", "lr_mem_static_R19", "tm_mem_R18",
                                    "tm_nomem_R18", "lr_mem_R18"), "tm_mem_static_R19"),
                          ("abcd", ("tm_mem_static_R19", "tm_nomem_static_R19", "lr_mem_static_R19", "tm_mem_R9b",
                                    "tm_nomem_R9b", "lr_mem_R9b"), "tm_mem_static_R19")):
    base = S / "results" / "confirm" / f"R21{ds}"
    arms = list(refs) if ONLY_REFS else list(NEW) + list(refs)
    corr, meta = {}, None
    print(f"\n== R21{ds} ==")
    ref_cfg = load(base / cfg_ref)[1]
    ref_cfg = {k: v for k, v in next(iter(ref_cfg.values()))["cfg"].items() if k != "seed"}
    for a in arms:
        folder = base / a
        if not folder.is_dir():
            problems.append(f"{ds} {a}: folder missing")
            continue
        seeds, js, zs = load(folder)
        if len(seeds) != SEEDS_EXPECTED[a]:
            problems.append(f"{ds} {a}: {len(seeds)} seeds, expected {SEEDS_EXPECTED[a]}")
        if not seeds:
            continue
        y0, c0 = zs[seeds[0]]["y"], zs[seeds[0]]["convo_id"]
        for k in seeds:
            if not (np.array_equal(zs[k]["y"], y0) and np.array_equal(zs[k]["convo_id"], c0)):
                problems.append(f"{ds} {a} seed {k}: labels or dialogue ids differ")
        if meta is None:
            meta = (y0, c0)
        elif not (np.array_equal(meta[0], y0) and np.array_equal(meta[1], c0)):
            problems.append(f"{ds} {a}: labels or dialogue ids differ from the first arm")
        if a in NEW:
            for k in seeds:
                j = js[k]
                if j.get("time", "") <= FREEZE:
                    problems.append(f"{ds} {a} seed {k}: time {j.get('time')} not after the freeze")
                if j.get("encoder") != "none":
                    problems.append(f"{ds} {a} seed {k}: encoder {j.get('encoder')}")
                if j.get("registration") != f"R21{ds}":
                    problems.append(f"{ds} {a} seed {k}: registration {j.get('registration')}")
                parts = j.get("parts")
                parts = ",".join(parts) if isinstance(parts, list) else parts
                if parts != PARTS[a]:
                    problems.append(f"{ds} {a} seed {k}: parts {parts}")
                if a.startswith("tm_"):
                    cfg = {kk: v for kk, v in j["cfg"].items() if kk != "seed"}
                    if cfg != ref_cfg:
                        diff = {kk: (cfg.get(kk), ref_cfg.get(kk)) for kk in set(cfg) | set(ref_cfg) if cfg.get(kk) != ref_cfg.get(kk)}
                        problems.append(f"{ds} {a} seed {k}: cfg differs from R19's: {diff}")
        C = np.stack([(zs[k]["pred"] == zs[k]["y"]).astype(float) for k in seeds])
        corr[a] = C.mean(0)
        print(f"  {a:20s} seeds {seeds}  acc {100 * C.mean():.2f}  per seed {[round(float(100 * x), 2) for x in C.mean(1)]}")
    if meta is None:
        continue
    cid = meta[1]
    pairs = [("tm_mem_static_R19", "tm_nomem_static_R19")]
    pairs += [("tm_mem_R18", "tm_nomem_R18")] if ds == "star" else [("tm_mem_R9b", "tm_nomem_R9b")]
    if not ONLY_REFS:
        pairs = [("tm_mem_c4", "tm_mem_static_R19"), ("tm_mem_c4", "tm_nomem_c4"),
                 ("tm_mem_c4", "tm_mem_R18" if ds == "star" else "tm_mem_R9b"),
                 ("tm_nomem_c4", "tm_nomem_static_R19"), ("lr_mem_c4", "lr_mem_static_R19"),
                 ("tm_mem_c4", "lr_mem_c4")] + pairs
    res = {}
    for a, b in pairs:
        if a in corr and b in corr:
            m, lo, hi, nd = boot(corr[a] - corr[b], cid)
            res[(a, b)] = (m, lo, hi)
            print(f"  {a + ' - ' + b:42s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  ({nd} dialogues, {len(cid)} action turns)")
    if not ONLY_REFS and ("tm_mem_c4", "tm_mem_static_R19") in res and ("tm_mem_c4", "tm_nomem_c4") in res:
        m1, lo1, _ = res[("tm_mem_c4", "tm_mem_static_R19")]
        m2, lo2, _ = res[("tm_mem_c4", "tm_nomem_c4")]
        r1 = m1 >= -2.0 - 1e-9
        r1s = r1 and lo1 >= -2.0 - 1e-9
        r2 = m2 >= 5.0 - 1e-9 and lo2 > 0
        verdicts[ds] = r1 and r2
        print(f"  R21.1 {'MET' if r1 else 'NOT MET'}{' (strong)' if r1s else ''}; R21.2 {'MET' if r2 else 'NOT MET'}"
              f" -> dataset {'MET' if r1 and r2 else 'NOT MET'}")

print("\nproblems:", problems if problems else "none")
if not ONLY_REFS and len(verdicts) == 2:
    print("HEADLINE (both datasets):", "MET" if all(verdicts.values()) else "NOT MET", verdicts)
