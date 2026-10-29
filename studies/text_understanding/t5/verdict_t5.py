#!/usr/bin/env python
"""Verdict of REGISTRATION_T5.md (with Amendment 1): reads t5/results_t5.jsonl. VAL ONLY.

    python -I verdict_t5.py [--results F] [--done F] [--manifest F]   (the options are for the self-test only)

Written before any registered val row existed. Per dataset (CLINC150, Banking77, HWU64): every view's val accuracy
(points), mean of seeds 1 and 2; G = potionV_K512 - wordsV; the bars of the registration:
  H5a (R1):  R1 - wordsV >= 0.25 G on >= 2 of 3 datasets, AND R1 - ppmiV_K86 >= +0.5 on >= 2 of 3;
             strong: R1 - wordsV >= 0.50 G on >= 2 of 3 (and the count bar).
  H5b (R3a): R3a - wordsV >= 0.25 G on >= 2 of 3, AND R3a - ppmiV_K512 >= +0.5 on >= 2 of 3; strong as H5a.
Reading fixed here, before any row: the two counts of a bar are separate, as registered (they need not hold on
the same datasets; the datasets where both hold are printed too); a dataset with G <= 0 (potion no better than
words) counts as not met for the gap parts and is flagged; the comparisons allow 1e-9 for float rounding; a view
counts on a dataset only when both seeds are there, and a bar with a missing dataset is printed as incomplete; a
repeated (dataset, view, seed) row is ignored (the first is kept) and printed.
Reported, no bar: potionV_K86; every view minus the flat potion existsT val rows of test 2 (results_confirm.jsonl,
view existsT, seeds 1 and 2, val_acc only; those use each dataset's own vocabulary); the bit counts and the 15%
line of Amendment 1 (each twin against its TM-AE view at the same K); share of val tokens without a code; bits on
per val sentence; best epochs; the code-table and script hashes of every row; the TM-AE training
(t5/tmae_done.json) and the R1 report of t5/codes_manifest.json (weights, empty rows, probe neighbours, clauses).
"""
import argparse
import json
from pathlib import Path

import numpy as np

T5 = Path(__file__).resolve().parent
EMB = T5.parent
DATASETS = ("clinc150", "banking77", "hwu64")
SEEDS = (1, 2)
VIEWS = ("wordsV", "potionV_K512", "ppmiV_K512", "ppmiV_K86", "potionV_K86", "R1", "R3a")
TWIN = {"R1": "ppmiV_K86", "R3a": "ppmiV_K512"}  # the count-only twin at the same K
TABLE = {"R1": "R1", "R3a": "R3a", "wordsV": "wordsV", "potionV_K512": "potionV", "potionV_K86": "potionV",
         "ppmiV_K512": "ppmiV", "ppmiV_K86": "ppmiV"}
FROZEN = {"t5_tm.py": "45f9e733575e6273", "dl/tm.py": "f649043f5801605b", "dl/tm_init.py": "557ff69706529738",
          "dl/boolean.py": "b5f5917f390da8a6", "dl/data.py": "0c4f8d22576e817c"}
EPS = 1e-9


def load_rows(path: Path):
    res, dups = {}, []
    if path.exists():
        for x in path.read_text().splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            if r.get("registration") != "REGISTRATION_T5.md" or r.get("smoke"):
                continue
            k = (r["dataset"], r["view"], int(r["seed"]))
            if k in res:
                dups.append(k)
            else:
                res[k] = r
    return res, dups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=T5 / "results_t5.jsonl")
    ap.add_argument("--done", type=Path, default=T5 / "tmae_done.json")
    ap.add_argument("--manifest", type=Path, default=T5 / "codes_manifest.json")
    ap.add_argument("--t2", type=Path, default=EMB / "results_confirm.jsonl")
    args = ap.parse_args()
    res, dups = load_rows(args.results)
    print(f"REGISTRATION_T5.md verdict, VAL accuracy in points, mean of seeds {SEEDS}; {len(res)} rows from {args.results}")
    if dups:
        print(f"  repeated rows ignored (first kept): {dups}")

    def acc(ds, v):
        a = [res[(ds, v, s)]["val_acc"] * 100 for s in SEEDS if (ds, v, s) in res]
        return float(np.mean(a)) if len(a) == len(SEEDS) else None

    t2: dict = {}
    if args.t2.exists():
        for x in args.t2.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r.get("view") == "existsT" and int(r["seed"]) in SEEDS:
                    t2.setdefault(r["dataset"], {})[int(r["seed"])] = r["val_acc"] * 100
    t2m = {ds: float(np.mean([t2[ds][s] for s in SEEDS])) for ds in t2 if all(s in t2[ds] for s in SEEDS)}

    # ---- the rows
    for ds in DATASETS:
        ref = f"{t2m[ds]:.2f}" if ds in t2m else "n/a"
        print(f"\n== {ds} (test 2 flat potion existsT val, seeds 1-2: {ref})")
        for v in VIEWS:
            rr = [res.get((ds, v, s)) for s in SEEDS]
            if not any(rr):
                print(f"  {v:13s} no rows")
                continue
            a = acc(ds, v)
            seeds = " / ".join(f"{r['val_acc'] * 100:.2f}" if r else "-" for r in rr)
            got = [r for r in rr if r]
            nb = sorted({r["n_bits"] for r in got})
            line = (f"  {v:13s} {(f'{a:6.2f}' if a is not None else '  incomplete')} (seeds {seeds}; best epochs "
                    f"{[r.get('best_epoch') for r in got]}); bits {nb if len(nb) > 1 else nb[0]}, on per val sentence "
                    f"{got[0]['bits_on_per_val_sentence']}, val tokens without code {got[0]['val_tokens_without_code_share']:.3f}, "
                    f"val sentences without code {got[0]['val_sentences_without_code']}")
            if a is not None and ds in t2m:
                line += f"; vs test 2 existsT {a - t2m[ds]:+.2f}"
            print(line)
            if len(nb) > 1:
                print(f"    note: the bit count differs between seeds {nb}")

    # ---- the bars
    print("\n== bars (G = potionV_K512 - wordsV)")
    verdict = {}
    for view, twin in TWIN.items():
        g25 = g50 = cnt = 0
        both, missing = [], []
        for ds in DATASETS:
            a, w, p, t = acc(ds, view), acc(ds, "wordsV"), acc(ds, "potionV_K512"), acc(ds, twin)
            if None in (a, w, p, t):
                missing.append(ds)
                continue
            G = p - w
            share = (a - w) / G if G > EPS else None
            h25 = share is not None and share >= 0.25 - EPS
            h50 = share is not None and share >= 0.50 - EPS
            hc = (a - t) >= 0.5 - EPS
            g25, g50, cnt = g25 + h25, g50 + h50, cnt + hc
            if h25 and hc:
                both.append(ds)
            sh = f"{share * 100:.1f}% of G" if share is not None else "G <= 0: FLAGGED, counts as not met"
            print(f"  {view} {ds}: {view} - wordsV {a - w:+.2f} (G {G:.2f}: {sh}); {view} - {twin} {a - t:+.2f}"
                  f" -> gap 25% {'yes' if h25 else 'no'}, gap 50% {'yes' if h50 else 'no'}, count +0.5 {'yes' if hc else 'no'}")
        works, strong = g25 >= 2 and cnt >= 2, g50 >= 2 and cnt >= 2
        name = "H5a" if view == "R1" else "H5b"
        state = "INCOMPLETE (missing: " + ", ".join(missing) + ")" if missing else ""
        verdict[name] = (works, strong, bool(missing))
        print(f"  {name} ({view}): gap >= 25% on {g25} of 3, gap >= 50% on {g50} of 3, beats {twin} by >= 0.5 on {cnt} of 3"
              f" (both on: {', '.join(both) or 'none'}) -> works {'MET' if works else 'NOT MET'}, strong "
              f"{'MET' if strong else 'NOT MET'} {state}".rstrip())
    rep = {ds: f"{acc(ds, 'potionV_K86') - acc(ds, 'R1'):+.2f}" for ds in DATASETS
           if acc(ds, "potionV_K86") is not None and acc(ds, "R1") is not None}
    print(f"  report: potionV_K86 - R1 (same K) {rep or 'n/a'}")
    if any(w and not m for w, _, m in verdict.values()):
        print("  a bar works: by the registration, a NEW registration fixes a test run before any test read")

    # ---- bit counts against the 15% line (Amendment 1)
    print("\n== bit counts (seed 1 rows; Amendment 1: a twin more than 15% away from its TM-AE view is reported)")
    for view, twin in TWIN.items():
        for other in (twin, "potionV_K86" if view == "R1" else "potionV_K512"):
            for ds in DATASETS:
                rv, rt = res.get((ds, view, 1)), res.get((ds, other, 1))
                if rv and rt:
                    d = (rt["n_bits"] - rv["n_bits"]) / rv["n_bits"]
                    tag = "BEYOND 15%" if abs(d) > 0.15 else "within 15%"
                    kind = "twin" if other == twin else "report only"
                    print(f"  {ds} {other} {rt['n_bits']} vs {view} {rv['n_bits']}: {d * 100:+.1f}% ({tag}; {kind})")

    # ---- hashes: every row against the frozen scripts and the manifest's tables
    print("\n== checks")
    man = json.loads(args.manifest.read_text()) if args.manifest.exists() else {"tables": {}}
    bad = []
    for k, r in sorted(res.items()):
        for f, want in FROZEN.items():
            if r["sha256"].get(f) != want:
                bad.append(f"{k} {f} {r['sha256'].get(f)} (frozen {want})")
        tab = man["tables"].get(TABLE[r["view"]])
        if tab is None or r["codes_sha256"] != tab["sha256"][:16]:
            bad.append(f"{k} codes {r['codes_sha256']} (manifest {tab['sha256'][:16] if tab else 'missing'})")
    print(f"  rows with a script or table hash that differs: {len(bad)}")
    for b in bad[:20]:
        print(f"    {b}")
    done = json.loads(args.done.read_text()) if args.done.exists() else None
    if done is None:
        print(f"  {args.done.name}: missing (the TM-AE training has not finished)")
    else:
        src = {t: man["tables"].get(t, {}).get("source", {}).get("ckpt_sha256") for t in ("R1", "R3a")}
        same = all(s == done["last_checkpoint_sha256"] for s in src.values())
        print(f"  TM-AE: {done['reason']}, {done['rounds_done']} of {done['rounds_planned']} rounds, checkpoint "
              f"{done['last_checkpoint']} ({done['last_checkpoint_rounds']} rounds), train {done['train_wall_s'] / 3600:.2f} h "
              f"wall, {done['s_per_update_mean'] * 1000:.2f} ms per update; R1 / R3a tables from this checkpoint: "
              f"{'yes' if same else 'NO ' + json.dumps(src)}")
        print(f"  TM-AE final weight histogram: {done['weight_hist_final']}")
    r1 = man["tables"].get("R1", {}).get("report")
    if r1:
        print(f"  R1 table: {r1['R1_empty_rows']} tokens without a code ({r1['R1_empty_share'] * 100:.1f}%), bits per token "
              f"mean {r1['R1_bits_per_token_mean']} max {r1['R1_bits_per_token_max']}; clauses without a literal "
              f"{r1['clauses_without_literal']}, literals per clause mean {r1['literals_per_clause_mean']} max "
              f"{r1['literals_per_clause_max']}")
        for w, p in r1["probes"].items():
            nb = ", ".join(f"{n} {c}" for n, c in p.get("R3a_neighbours", []))
            jb = ", ".join(f"{n} {c}" for n, c in p.get("R1_jaccard_neighbours", []))
            print(f"  probe {w!r}: R1 bits {p['R1_bits']}; R3a neighbours: {nb or '-'}; R1 Jaccard: {jb or '-'}")
            for c in p.get("top_clauses", [])[:2]:
                print(f"      clause {c['clause']} (weight {c['weight']}, {c['n_literals']} literals): {' AND '.join(c['literals'][:8])}"
                      f"{' ...' if c['n_literals'] > 8 else ''}")


if __name__ == "__main__":
    main()
