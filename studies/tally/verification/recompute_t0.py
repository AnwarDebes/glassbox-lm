"""Independent recompute of the T0 verdict of the tally study, written from the registration text.

    OMP_NUM_THREADS=1 nice -n 19 ../tm_language_model/.venv-tm/bin/python -I verification/recompute_t0.py . \
        --out <json> --report results/<UTC>_t0_report_devtest        # from the study folder

Own code, no import of the study's code. It
- finds the 30 final run folders (coalesced seeds 1-10, flat seeds 1-10, logistic regression of the
  coalesced banks seeds 1-10) and checks: one folder per seed, state finished, the registered settings,
  start after the amendment that fixed them, the same source hashes in every run, the same dev-test
  positions and blocks in every run, finite positive bits;
- recomputes each arm's bits per word on dev-test (each position averaged over the seeds, as the
  registration says), the four paired differences, and a block bootstrap with its own generator
  (PCG64 seed 20261025, 10,000 resamples, percentile 95% interval), plus, reported beside and not
  part of any rule, a two-level bootstrap (seeds, then blocks) and the per-seed differences;
- compares with the report's metrics.json when --report is given.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

SEEDS = list(range(1, 11))
FAMILIES = {
    "coal": (r"^\d{8}T\d{6}Z_t0_final_coal_c2000_s20_seed(\d+)$", dict(machine="coalesced", s=20.0, epochs=9, clauses=2000, mode="final")),
    "flat": (r"^\d{8}T\d{6}Z_t0_final_flat_c100_s5_seed(\d+)$", dict(machine="flat", s=5.0, epochs=1, clauses=100, mode="final")),
    "logreg": (r"^\d{8}T\d{6}Z_t0_logreg_of_final_coal_c2000_s20_seed(\d+)$", None),
}
AMEND = {"coal": "registrations/T0_amendment_coalesced_settings.frozen.json",
         "flat": "registrations/T0_amendment_flat_settings.frozen.json"}
B = 10_000
SP = "devtest"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_runs(study, problems):
    res = os.path.join(study, "results")
    found = {k: {} for k in FAMILIES}
    for name in sorted(os.listdir(res)):
        for fam, (pat, _) in FAMILIES.items():
            m = re.match(pat, name)
            if m:
                seed = int(m.group(1))
                if seed in found[fam]:
                    problems.append(f"two folders for {fam} seed {seed}: {found[fam][seed]} and {name}")
                found[fam][seed] = os.path.join(res, name)
    for fam in FAMILIES:
        if sorted(found[fam]) != SEEDS:
            problems.append(f"{fam}: seeds found {sorted(found[fam])}, expected {SEEDS}")
    return found


def check_runs(study, found, problems, info):
    for fam, (_, want) in FAMILIES.items():
        amend_utc = None
        if fam in AMEND:
            rec = json.load(open(os.path.join(study, AMEND[fam])))
            amend_utc = rec["frozen_utc"]
            info[f"{fam}_amendment_frozen_utc"] = amend_utc
        src_sets = set()
        for seed, f in sorted(found[fam].items()):
            run = json.load(open(os.path.join(f, "run.json")))
            cfg = json.load(open(os.path.join(f, "config.json")))
            if run.get("state") != "finished":
                problems.append(f"{fam} seed {seed}: state {run.get('state')}")
            if amend_utc and not run["start_utc"] > amend_utc:
                problems.append(f"{fam} seed {seed}: start {run['start_utc']} not after amendment {amend_utc}")
            if want:
                for k, v in want.items():
                    if cfg.get(k) != v:
                        problems.append(f"{fam} seed {seed}: config {k}={cfg.get(k)!r}, expected {v!r}")
                if cfg.get("seed") != seed:
                    problems.append(f"{fam} seed {seed}: config seed {cfg.get('seed')}")
            src = json.load(open(os.path.join(f, "source_hashes.json")))
            src_code = {k: v for k, v in src.items() if k.startswith("src/")}
            src_sets.add(json.dumps(src_code, sort_keys=True))
            info.setdefault("runs", []).append(dict(family=fam, seed=seed, folder=os.path.basename(f),
                                                    start=run["start_utc"], end=run.get("end_utc"),
                                                    bits_sha256=sha256(os.path.join(f, "bits.npz"))))
        if fam == "logreg":
            for seed, f in sorted(found[fam].items()):
                src_run = json.load(open(os.path.join(f, "config.json"))).get("source_run")
                want_src = os.path.basename(found["coal"].get(seed, ""))
                if src_run != want_src:
                    problems.append(f"logreg seed {seed}: source_run {src_run!r}, expected {want_src!r}")
        info[f"{fam}_distinct_src_hash_sets"] = len(src_sets)
        if fam in AMEND:
            rec = json.load(open(os.path.join(study, AMEND[fam])))
            frozen = json.dumps({k: v for k, v in rec["src"].items()}, sort_keys=True)
            info[f"{fam}_src_equals_amendment_record"] = (src_sets == {frozen})
            if src_sets != {frozen}:
                problems.append(f"{fam}: run source hashes differ from the amendment freeze record")
        if len(src_sets) != 1:
            problems.append(f"{fam}: {len(src_sets)} different source-hash sets across seeds")


def load_arms(found, problems):
    per = {}
    blocks = None
    n = None
    for fam in ("coal", "flat"):
        for seed in SEEDS:
            z = np.load(os.path.join(found[fam][seed], "bits.npz"))
            b = z[f"blocks__{SP}"]
            if blocks is None:
                blocks, n = b.copy(), len(b)
            elif not np.array_equal(blocks, b):
                problems.append(f"{fam} seed {seed}: dev-test blocks differ from the first run")
            keys = ["learned", "counted"] + ([f"untrained{r}" for r in range(10)] if fam == "coal" else [])
            for k in keys:
                v = z[f"{k}__{SP}"].astype(np.float64)
                if len(v) != n or not np.all(np.isfinite(v)) or v.min() <= 0:
                    problems.append(f"{fam} seed {seed} {k}: length {len(v)} vs {n}, finite {np.all(np.isfinite(v))}, min {v.min()}")
                per[(fam, k, seed)] = v
    for seed in SEEDS:
        z = np.load(os.path.join(found["logreg"][seed], "bits.npz"))
        v = z[f"logreg__{SP}"].astype(np.float64)
        if len(v) != n or not np.all(np.isfinite(v)) or v.min() <= 0:
            problems.append(f"logreg seed {seed}: length {len(v)} vs {n}")
        per[("logreg", "logreg", seed)] = v
    return per, blocks


def block_stats(diff_by_seed, blocks):
    """diff_by_seed: (S, n). Returns per-seed per-block sums (S, G) and block counts (G,)."""
    ub, inv = np.unique(blocks, return_inverse=True)
    G = len(ub)
    sums = np.stack([np.bincount(inv, weights=d, minlength=G) for d in diff_by_seed])
    cnts = np.bincount(inv, minlength=G).astype(np.float64)
    return sums, cnts


def boot_one_level(sums, cnts, rng):
    """Seeds averaged first (as registered), then blocks resampled."""
    s = sums.mean(0)
    G = len(cnts)
    idx = rng.integers(0, G, size=(B, G))
    stats = s[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def boot_two_level(sums, cnts, rng):
    S, G = sums.shape
    out = np.empty(B)
    for i in range(B):
        si = rng.integers(0, S, S)
        gi = rng.integers(0, G, G)
        out[i] = sums[si][:, gi].mean(0).sum() / cnts[gi].sum()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def reading(lo, hi, a, b):
    if hi < 0:
        return f"{a} beats {b}"
    if lo > 0:
        return f"{b} beats {a}"
    return "no difference shown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None, help="run folder of tally.t0.report")
    a = ap.parse_args()
    study = os.path.abspath(a.study)
    problems, info = [], {}
    found = find_runs(study, problems)
    if problems:
        print(json.dumps(dict(problems=problems), indent=1))
        sys.exit(2)
    check_runs(study, found, problems, info)
    per, blocks = load_arms(found, problems)
    n = len(blocks)
    rng = np.random.Generator(np.random.PCG64(20261025))

    def stackseeds(fam, k):
        return np.stack([per[(fam, k, s)] for s in SEEDS])

    arms_by_seed = {
        "learned": stackseeds("coal", "learned"),
        "counted": stackseeds("coal", "counted"),
        "untrained": np.mean([stackseeds("coal", f"untrained{r}") for r in range(10)], 0),
        "logreg": stackseeds("logreg", "logreg"),
        "flat_learned": stackseeds("flat", "learned"),
        "flat_counted": stackseeds("flat", "counted"),
    }
    out = dict(split=SP, seeds=len(SEEDS), positions=n, blocks=int(len(np.unique(blocks))),
               bootstrap=dict(generator="PCG64", seed=20261025, resamples=B, interval="percentile 95%"),
               bpw={k: float(v.mean(0).mean()) for k, v in arms_by_seed.items()},
               bpw_per_seed={k: [float(x) for x in v.mean(1)] for k, v in arms_by_seed.items()})
    out["ppl"] = {k: float(2 ** v) for k, v in out["bpw"].items()}
    pairs = [("counted", "learned"), ("counted", "untrained"), ("counted", "logreg"), ("flat_counted", "flat_learned")]
    for x, y in pairs:
        d = arms_by_seed[x] - arms_by_seed[y]
        sums, cnts = block_stats(d, blocks)
        mean = float(d.mean(0).mean())
        lo, hi = boot_one_level(sums, cnts, rng)
        lo2, hi2 = boot_two_level(sums, cnts, rng)
        per_seed = [float(v) for v in d.mean(1)]
        out[f"{x}_minus_{y}"] = dict(mean=mean, ci95=[lo, hi], reading=reading(lo, hi, x, y),
                                     two_level_ci95=[lo2, hi2], per_seed=per_seed,
                                     seeds_negative=int(sum(v < 0 for v in per_seed)))
    out["checks"] = info
    out["problems"] = problems
    if a.report:
        rep = json.load(open(os.path.join(a.report, "metrics.json")))
        cmp = {}
        for x, y in pairs:
            key = f"{x}_minus_{y}"
            if key not in rep:
                cmp[key] = "missing in report"
                continue
            r, m = rep[key], out[key]
            width = r["ci95"][1] - r["ci95"][0]
            tol = max(0.01, 0.1 * width)
            cmp[key] = dict(mean_report=r["mean"], mean_recompute=m["mean"], mean_equal=abs(r["mean"] - m["mean"]) < 1e-9,
                            ci_report=r["ci95"], ci_recompute=m["ci95"],
                            ci_close=bool(abs(r["ci95"][0] - m["ci95"][0]) <= tol and abs(r["ci95"][1] - m["ci95"][1]) <= tol),
                            reading_report=r["reading"], reading_recompute=m["reading"],
                            reading_equal=r["reading"] == m["reading"])
        bpw_eq = {k: abs(rep["bpw"][k] - out["bpw"][k]) < 1e-9 for k in out["bpw"] if k in rep.get("bpw", {})}
        cmp["bpw_equal"] = bpw_eq
        cmp["seeds_equal"] = rep.get("seeds") == len(SEEDS)
        cmp["positions_equal"] = rep.get("positions") == n
        ok = all(v.get("mean_equal") and v.get("ci_close") and v.get("reading_equal")
                 for k, v in cmp.items() if isinstance(v, dict) and "mean_equal" in v)
        ok = ok and all(bpw_eq.values()) and cmp["seeds_equal"] and cmp["positions_equal"] and not problems
        cmp["verdict"] = "MATCH" if ok else "MISMATCH"
        out["compare_with_report"] = cmp
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in out if k not in ("checks", "bpw_per_seed")}, indent=1))


if __name__ == "__main__":
    main()
