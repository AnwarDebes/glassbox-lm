#!/usr/bin/env python
"""M2, the verdict (tu/PLAN_M2.md "Hypotheses"): H1 and H2 on MTOP English's test split, and the lines with no rule.

    python -I -B m2_verdict.py --m2 tu/m2 --matched fresh/mtop_test_matched_v1.json --overlap fresh/mtop_overlap.json

Input: <m2>/results_m2_tm.jsonl (tag "m2", seeds 7-9, s 8), <m2>/results_m2_lr.jsonl, the test predictions they name,
the test labels and IDs (m2_data.load: the registered read of en/test), the frozen matched list and (for the c4_fit
line) its source overlap file. Every number is computed from the saved predictions (each row's test accuracy must
equal its predictions' accuracy rounded to 4 decimals). Points throughout.

tu/m1_verdict.py (d32188aaf86a3c8d) with these changes (one dataset, M2's arms and rules) and nothing else:
(1) one set, mtop_en, so there is no 2-of-3 rule. A hypothesis is MET when its bar holds on the point estimates with
    BOTH probes (TM = the mean over seeds 7-9 of its test accuracy; LR = its one fit), and STRONG when the lower end of
    the registered 95% interval clears the same bar with both probes (M1's definition).
    - H1: a3B - C4 >= +0.5 with TM and with LR.
    - H2, one-sided, no upper bound: a3B - potion >= -0.5 with TM and with LR.
    The claim sentences are written by claim() from the numbers (review of draft 1, design MAJOR 1, M1 addendum item 4):
    H1 met: "a3B - C4 is at least +0.5 points on MTOP English with the TM (+x [l, u]) and with LR (+y [l, u]) (point
    estimates)", plus "the lower ends of the item-level 95% intervals clear +0.5" only when STRONG; H2 met: "a3B is no
    more than 0.5 points below potion-base-32M on MTOP English with the TM and with LR (point estimates; one-sided;
    ...)"; not met: one clause per probe with its interval, "holds" or "below the bar", and "A true difference inside
    the interval is not ruled out". Never "beats", never "within 0.5 points", never "statistically non-inferior".
    After the re-check of draft 2 (2026-10-25): a MET hypothesis that is not STRONG gets the fixed clause "not STRONG: a
    point-estimate result, weak evidence (...)" (MINOR A), and every claim sentence ends with its arms' provenance
    (a3B network-free at run time with bge-base-en-v1.5@a5beb1e3 as teacher at training time; C4 no neural network at
    any stage; potion-base-32M@1e5a03f8 distilled from bge-base-en-v1.5 by its authors, the teacher family a3B shares;
    MINOR B).
(2) arms C4, a3B and potion only. Descriptive (no rule): TM minus LR per arm (the seed-averaged TM correctness minus
    LR's on the same items), with the per-seed differences; every comparison's en/eval difference (best-epoch val
    numbers, optimistic) beside its test difference.
(3) the registered interval: a paired bootstrap over test items, 10,000 resamples, numpy default_rng([20261025, 0]);
    one index draw per resample, shared by every comparison and by seeds 7-9; percentiles 2.5 / 97.5 (M1's method).
(4) beside it, with no rule: a seed-inclusive two-level interval for every comparison with a TM side, numpy
    default_rng([20261025, 1]): per resample one item draw (shared by every comparison), then for each TM
    configuration its 3 seeds drawn with replacement from seeds 7-9 (independently per configuration; one draw per
    configuration per resample, shared by the comparisons that use it). LR (one fit) has no seed level and is printed
    "no seed level". When a hypothesis is STRONG but a TM probe's seed-inclusive lower end misses the bar, the flag
    reads "MET (STRONG on the item-level interval only)" (a fixed qualifier; the outcome is unchanged).
(5) a test or val item whose intent is not in en/train has label -1 and counts as an error for every arm; the counts
    and the number of train intents are reported.
(6) secondary (no rule): every comparison and arm on the test items NOT in the frozen matched list
    (fresh/mtop_test_matched_v1.json: positions = 0-based rows of en/test.txt), with an item-only interval
    (default_rng([20261025, 2])); with --overlap, the same without only the c4_fit pool's items (a3B's teacher-fit
    sentences, the one pool an M2 arm's training touched; default_rng([20261025, 3])). The lists must fit the test
    split (n_test, counts, positions in range, the IDs at the positions; the c4_fit union inside the list and of its
    stated size). A list or overlap file that does not fit marks the lines that use it NOT EVALUABLE, with the reasons,
    and H1 and H2 are written all the same (re-check of draft 2, MINOR E: these files feed no rule, and the 12 test
    reads have happened by then). The "paraphrase_exact" items (55 in the frozen list) get their own line.
(7) rows must carry tag "m2" and dataset "mtop_en"; each arm's rows must name its table (C4 = ppw_S1000_d500_e0.5
    e1771d9eaf4a07f7, a3B = a3B_d768 4a1ed2aed7aa4ce7, potion = potion-base-32M at 1e5a03f8), and the TM seeds and the
    LR fit of an arm must share one feature cache (one kmeans_run_id); the outputs are verdict_m2.json and
    verdict_m2.txt, the text with one per-probe table of the four rule parts.
Refuses to write unless every configuration has exactly seeds 7, 8 and 9, one row each, s 8, no smoke row, and the
same TM settings apart from the seed; every LR arm has exactly one row; and every prediction file has n_test entries
(a missing run leaves H1 and H2 NOT EVALUABLE: no rerun after a test read). Refuses to overwrite. Exit codes: 2
overwrite, 3 rows, 4 predictions (the earlier exit 5 for the matched list or the overlap file is gone, see (6)). Under
the per-probe table the text prints each rule part with its per-seed differences and seed SE, as M1's verdict text did
(MINOR G).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SET = "mtop_en"
SEEDS = (7, 8, 9)
TM_CONFIGS = (("C4", 1), ("a3B", 1), ("potion", 1))
LR_ARMS = ("C4", "a3B", "potion")
N_BOOT, BOOT_SEED, CHUNK = 10_000, 20261025, 250
EPS = 1e-9
MATCHED_V1_SHA256 = "1ea621f8c8a67ff05c76c4b30a595bdc2f650ddfc85586c4bf3d1462b8d98405"
EXPECTED_TABLE = {"C4": ("ppw_S1000_d500_e0.5", "e1771d9eaf4a07f7"), "a3B": ("a3B_d768", "4a1ed2aed7aa4ce7"),
                  "potion": ("potion-base-32M", "minishlab/potion-base-32M@1e5a03f8eeb2c98b928fbbd846f22f816360919f")}
# (name, probe, a, b, bar or None); a TM side is (arm, repeat), an LR side is the arm's name
COMPARISONS = (
    ("H1_tm", "tm", ("a3B", 1), ("C4", 1), 0.5), ("H1_lr", "lr", "a3B", "C4", 0.5),
    ("H2_tm", "tm", ("a3B", 1), ("potion", 1), -0.5), ("H2_lr", "lr", "a3B", "potion", -0.5),
    ("tm_minus_lr_C4", "tm-lr", ("C4", 1), "C4", None),
    ("tm_minus_lr_a3B", "tm-lr", ("a3B", 1), "a3B", None),
    ("tm_minus_lr_potion", "tm-lr", ("potion", 1), "potion", None),
)
HYP = {"H1": ("H1_tm", "H1_lr"), "H2": ("H2_tm", "H2_lr")}
RULE = {"H1": "a3B - C4 >= +0.5 points with TM and with LR",
        "H2": "a3B - potion >= -0.5 points with TM and with LR, one-sided (no upper bound)"}
WEAK = {"H1": ("not STRONG: a point-estimate result, weak evidence (tu/PLAN_M2.md Power: at no true gain H1 is MET in "
               "0.4-10.5% of runs)"),
        "H2": ("not STRONG: a point-estimate result, weak evidence (tu/PLAN_M2.md Power: at a true -1.0 H2 is MET in "
               "0.1-7.8% of runs)")}
TAIL = {"H1": ("Provenance: a3B is network-free at run time, with bge-base-en-v1.5@a5beb1e3 as teacher at training time; "
               "C4 uses no neural network at any stage."),
        "H2": ("Provenance: a3B is network-free at run time, with bge-base-en-v1.5@a5beb1e3 as teacher at training time; "
               "potion-base-32M@1e5a03f8 is distilled from bge-base-en-v1.5 by its authors, so the two share the teacher "
               "family.")}
PROVENANCE = (
    "MTOP English en/test, read once by this registered run (use count 0 -> 1); numbers on MTOP English only, never "
    "pooled with M1's three reused splits. C4 (ppw_S1000_d500_e0.5): no neural network at any stage. a3B (a3B_d768): "
    "network-free at run time, bge-base-en-v1.5@a5beb1e3 as teacher at training time. potion-base-32M@1e5a03f8: the "
    "published static table, distilled from bge-base-en-v1.5 by its authors (the teacher revision they used is not "
    "known). MTEB's English classification tasks include MTOP intent and domain, so MTOP texts in the teacher's or "
    "potion's training data, or model selection on MTOP's test split by their developers, cannot be ruled out; teacher "
    "contamination would inflate H1 (a3B has the teacher, C4 does not) and affects H2 less (both share the teacher).")


def iv(ci) -> str:
    return "no seed level" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def show(n: str, comp: dict) -> str:
    """One comparison as text: the point, the registered interval, the seed-inclusive interval (or "no seed level")
    and, for a TM side, the per-seed differences and the seed SE."""
    c = comp[n]
    s = f"{n} {c['point']:+.2f} {iv(c['ci95'])}"
    s += " (no seed level" if c["ci95_seed_inclusive"] is None else f" (seed-inclusive {iv(c['ci95_seed_inclusive'])}"
    if "seed_se" in c:
        s += f"; per seed {', '.join(f'{v:+.2f}' for v in c['per_seed'])}; seed SE {c['seed_se']:.2f}"
    return s + f"; en/eval {c['val_point_best_epoch']:+.2f})"


def claim(h: str, hyp: dict, comp: dict) -> str:
    """The claim sentence of a hypothesis, written from the numbers (see (1))."""
    tm, lr = HYP[h]
    ct, cl = comp[tm], comp[lr]
    if hyp[h]["met"]:
        if h == "H1":
            s = (f"H1 MET: a3B - C4 is at least +0.5 points on MTOP English with the TM ({ct['point']:+.2f} {iv(ct['ci95'])}) "
                 f"and with LR ({cl['point']:+.2f} {iv(cl['ci95'])}) (point estimates)")
        else:
            s = ("H2 MET: a3B is no more than 0.5 points below potion-base-32M on MTOP English with the TM and with LR "
                 f"(point estimates; one-sided; TM {ct['point']:+.2f} {iv(ct['ci95'])}, LR {cl['point']:+.2f} {iv(cl['ci95'])})")
        if hyp[h]["strong"]:
            s += f"; the lower ends of the item-level 95% intervals clear {ct['bar']:+.1f}"
            if hyp[h]["strong_item_level_only"]:
                s += (f" (on the item-level interval only: the TM's seed-inclusive lower end is "
                      f"{ct['ci95_seed_inclusive'][0]:+.2f})")
        else:
            s += f"; {WEAK[h]}"
        return s + ". " + TAIL[h]
    diff = "a3B - C4" if h == "H1" else "a3B - potion"
    parts = [f"with {name}, {diff} is {c['point']:+.2f} {iv(c['ci95'])}: "
             + ("holds" if c["point"] >= c["bar"] - EPS else "below the bar") for name, c in (("the TM", ct), ("LR", cl))]
    return (f"{h} NOT MET (the registered rule {RULE[h]} was not met): " + "; ".join(parts)
            + ". A true difference inside the interval is not ruled out. " + TAIL[h])


def check_rows(tm: list[dict], lr: list[dict]) -> list[str]:
    errs = []
    if any(r.get("smoke") for r in tm):
        errs.append("a smoke row is in the TM results")
    for r in tm:
        if r.get("tag") != "m2" or r.get("dataset") != SET:
            errs.append(f"TM row with tag {r.get('tag')!r} and dataset {r.get('dataset')!r}: expected 'm2' and {SET!r}")
    for r in lr:
        if r.get("dataset") != SET:
            errs.append(f"LR row {r.get('name')!r} with dataset {r.get('dataset')!r}: expected {SET!r}")
    rows = [r for r in tm if r.get("tag") == "m2" and r.get("dataset") == SET]
    lrs = [r for r in lr if r.get("dataset") == SET]
    cfg0 = None
    for arm, rep in TM_CONFIGS:
        got = [r for r in rows if r["arm"] == arm and r["repeat"] == rep]
        seeds = sorted(r["seed"] for r in got)
        if seeds != list(SEEDS):
            errs.append(f"TM {arm} repeat {rep}: seeds {seeds}, expected {list(SEEDS)}")
        for r in got:
            if r["s"] != 8.0:
                errs.append(f"TM {arm} repeat {rep} seed {r['seed']}: s {r['s']}")
            c = {k: v for k, v in r["cfg"].items() if k != "seed"}
            if cfg0 is None:
                cfg0 = c
            elif c != cfg0:
                errs.append(f"TM {arm} repeat {rep} seed {r['seed']}: TM settings differ")
            if (r.get("table"), r.get("table_sha256")) != EXPECTED_TABLE[arm]:
                errs.append(f"TM {arm} seed {r['seed']}: table {r.get('table')} {r.get('table_sha256')}, expected "
                            f"{EXPECTED_TABLE[arm][0]} {EXPECTED_TABLE[arm][1]}")
    extra = sorted({(r["arm"], r["repeat"]) for r in rows} - set(TM_CONFIGS))
    if extra:
        errs.append(f"TM rows of configurations not in M2: {extra}")
    for arm in LR_ARMS:
        got = [r for r in lrs if r["name"] == arm]
        if len(got) != 1:
            errs.append(f"LR {arm}: {len(got)} rows, expected 1")
        for r in got:
            if r.get("arm") != arm or (r.get("table"), r.get("table_sha256")) != EXPECTED_TABLE[arm]:
                errs.append(f"LR {arm}: arm {r.get('arm')!r}, table {r.get('table')} {r.get('table_sha256')}, expected "
                            f"{arm} {EXPECTED_TABLE[arm][0]} {EXPECTED_TABLE[arm][1]}")
        caches = {r.get("kmeans_run_id") for r in rows if r["arm"] == arm} | {r.get("kmeans_run_id") for r in got}
        if len(caches) > 1:
            errs.append(f"arm {arm}: its TM and LR rows come from {len(caches)} feature caches (kmeans_run_id {sorted(map(str, caches))})")
    return errs


def check_matched(m: dict, n: int, ids: list[str] | None) -> list[str]:
    errs = []
    pos, para = m.get("positions", []), m.get("paraphrase_exact", {}).get("positions", [])
    if m.get("n_test") != n:
        errs.append(f"matched list n_test {m.get('n_test')} != test items {n}")
    if m.get("n_left_out") != len(pos) or m.get("n_kept") != n - len(pos) or len(set(pos)) != len(pos):
        errs.append("matched list counts do not fit its positions")
    if any(not (0 <= p < n) for p in pos + para):
        errs.append("a matched position is outside the test split")
    if ids is None:
        errs.append("the loaded test split has no IDs to check the matched list against")
    elif len(ids) != n:
        errs.append(f"{len(ids)} test IDs for {n} test items")
    elif not errs:
        if [ids[p] for p in pos] != list(m.get("ids", [])):
            errs.append("the IDs at the matched positions differ from the list's IDs")
        if [ids[p] for p in para] != list(m.get("paraphrase_exact", {}).get("ids", [])):
            errs.append("the IDs at the paraphrase_exact positions differ from the list's IDs")
    return errs


def c4_fit_positions(ov: dict, m: dict, n: int) -> tuple[list[int], list[str]]:
    c4 = sorted({int(i) for k in ("exact", "order_free", "one_edit") for i in ov["hits"]["test"]["c4_fit"][k]})
    errs = []
    if ov.get("mtop_sizes", {}).get("test") != n:
        errs.append(f"overlap file test size {ov.get('mtop_sizes', {}).get('test')} != test items {n}")
    if len(c4) != m.get("per_pool_union", {}).get("c4_fit"):
        errs.append(f"c4_fit union {len(c4)} != the matched list's per_pool_union c4_fit {m.get('per_pool_union', {}).get('c4_fit')}")
    if not set(c4) <= set(m.get("positions", [])):
        errs.append("a c4_fit position is not in the matched list")
    return c4, errs


def boot_items(D: dict[str, np.ndarray], stream: int) -> dict[str, tuple[float, float]]:
    """95% percentile intervals (points) of mean(D[name][idx]) over N_BOOT shared index draws (M1's method)."""
    n = len(next(iter(D.values())))
    rng = np.random.default_rng([BOOT_SEED, stream])
    stats = {k: [] for k in D}
    for _ in range(N_BOOT // CHUNK):
        idx = rng.integers(0, n, size=(CHUNK, n))
        for k, v in D.items():
            stats[k].append(v[idx].mean(1))
    out = {}
    for k, v in stats.items():
        s = 100 * np.concatenate(v)
        out[k] = (round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4))
    return out


def boot_two_level(ok_tm: dict, ok_lr: dict, stream: int) -> dict[str, tuple[float, float] | None]:
    """The seed-inclusive interval (no rule): items, then seeds within each TM configuration (see (4)); None for the
    LR-only comparisons, which have no seed level."""
    n = len(next(iter(ok_lr.values())))
    rng = np.random.default_rng([BOOT_SEED, stream])
    stats = {name: [] for name, *_ in COMPARISONS}
    cols = np.arange(CHUNK)[:, None]
    for _ in range(N_BOOT // CHUNK):
        idx = rng.integers(0, n, size=(CHUNK, n))
        acc = {}
        for key in TM_CONFIGS:
            m = ok_tm[key]  # seeds x items
            per = np.stack([m[j][idx].mean(1) for j in range(m.shape[0])])  # seeds x CHUNK
            pick = rng.integers(0, m.shape[0], size=(CHUNK, m.shape[0]))
            acc[key] = per[pick, cols].mean(1)
        for arm in LR_ARMS:
            acc[arm] = ok_lr[arm][idx].mean(1)
        for name, _probe, x, z, _bar in COMPARISONS:
            stats[name].append(acc[x] - acc[z])
    out = {}
    for name, probe, _x, _z, _bar in COMPARISONS:
        if probe == "lr":
            out[name] = None
            continue
        s = 100 * np.concatenate(stats[name])
        out[name] = (round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4))
    return out


def diffs(ok_tm: dict, ok_lr: dict, keep: np.ndarray | None = None):
    """Per comparison: the per-item difference D (seed-averaged for TM), the point, and for TM the per-seed values and
    the paired seed SE."""
    D, point, se, per = {}, {}, {}, {}
    for name, probe, x, z, _bar in COMPARISONS:
        if probe == "lr":
            d = ok_lr[x] - ok_lr[z]
            d = d if keep is None else d[keep]
            D[name] = d
        else:
            diff = ok_tm[x] - (ok_tm[z] if probe == "tm" else ok_lr[z][None, :])  # seeds x items
            diff = diff if keep is None else diff[:, keep]
            D[name] = diff.mean(0)
            per_seed = 100 * diff.mean(1)
            per[name] = [round(float(v), 4) for v in per_seed]
            se[name] = round(float(per_seed.std(ddof=1) / math.sqrt(len(SEEDS))), 4)
        point[name] = round(float(100 * D[name].mean()), 4)
    return D, point, se, per


def val_points(T: dict, L: dict) -> dict[str, float]:
    """Each comparison's en/eval difference (best-epoch val accuracies of the same rows; descriptive, optimistic)."""
    vt = {key: 100 * sum(T[key][s]["val_acc"] for s in SEEDS) / len(SEEDS) for key in TM_CONFIGS}
    vl = {arm: 100 * L[arm]["lr_val"] for arm in LR_ARMS}
    out = {}
    for name, probe, x, z, _bar in COMPARISONS:
        a = vt[x] if probe in ("tm", "tm-lr") else vl[x]
        b = vt[z] if probe == "tm" else vl[z]
        out[name] = round(a - b, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m2", type=Path, required=True)
    ap.add_argument("--matched", type=Path, required=True, help="the frozen matched list (fresh/mtop_test_matched_v1.json)")
    ap.add_argument("--overlap", type=Path, default=None, help="its source overlap file (fresh/mtop_overlap.json), for the c4_fit line")
    ap.add_argument("--labels", default="", help="tests only: FILE.py:function returning a stand-in dataset")
    a = ap.parse_args()
    outj, outt = a.m2 / "verdict_m2.json", a.m2 / "verdict_m2.txt"
    for p in (outj, outt):
        if p.exists():
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 2
    tm = [json.loads(x) for x in (a.m2 / "results_m2_tm.jsonl").read_text().splitlines() if x.strip()]
    lr = [json.loads(x) for x in (a.m2 / "results_m2_lr.jsonl").read_text().splitlines() if x.strip()]
    errs = check_rows(tm, lr)
    if errs:
        print("REFUSED:\n" + "\n".join(errs) + "\nH1 and H2 are NOT EVALUABLE from these rows (tu/PLAN_M2.md: a run that "
              "failed after its test read is not rerun).", file=sys.stderr)
        return 3
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m2_feats import load_ds

    ds = load_ds(SET, a.labels)
    y = np.asarray(ds.labels["test"])
    n = len(y)
    mbytes = a.matched.read_bytes()
    try:
        matched = json.loads(mbytes)
        m_errs = check_matched(matched, n, getattr(ds, "ids", {}).get("test"))
    except (ValueError, TypeError, AttributeError) as e:
        matched, m_errs = {}, [f"the matched list is unreadable: {type(e).__name__}: {e}"]
    c4pos, obytes, o_errs = None, None, []
    if a.overlap is not None:
        obytes = a.overlap.read_bytes()
        try:
            c4pos, o_errs = c4_fit_positions(json.loads(obytes), matched, n)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            o_errs = [f"the overlap file is unreadable: {type(e).__name__}: {e}"]
    T = {key: {r["seed"]: r for r in tm if r["arm"] == key[0] and r["repeat"] == key[1]} for key in TM_CONFIGS}
    L = {arm: next(r for r in lr if r["name"] == arm) for arm in LR_ARMS}
    ok_tm, ok_lr = {}, {}
    for key in TM_CONFIGS:
        preds = [np.load(T[key][s]["preds"]) for s in SEEDS]
        if any(len(p) != n for p in preds):
            print(f"REFUSED: {key[0]}@{key[1]}: a prediction file does not have {n} entries", file=sys.stderr)
            return 4
        m = np.stack([(p == y) for p in preds]).astype(np.float64)
        for j, s in enumerate(SEEDS):
            if round(float(m[j].mean()), 4) != T[key][s]["test_acc"]:
                print(f"REFUSED: {key[0]}@{key[1]} seed {s}: predictions give {m[j].mean():.4f}, row {T[key][s]['test_acc']}",
                      file=sys.stderr)
                return 4
        ok_tm[key] = m
    for arm in LR_ARMS:
        p = np.load(L[arm]["preds"])
        if len(p) != n:
            print(f"REFUSED: LR {arm}: the prediction file does not have {n} entries", file=sys.stderr)
            return 4
        m = (p == y).astype(np.float64)
        if round(float(m.mean()), 4) != L[arm]["lr_test"]:
            print(f"REFUSED: LR {arm}: predictions give {m.mean():.4f}, row {L[arm]['lr_test']}", file=sys.stderr)
            return 4
        ok_lr[arm] = m
    D, point, se, per = diffs(ok_tm, ok_lr)
    ci = boot_items(D, 0)
    ci2 = boot_two_level(ok_tm, ok_lr, 1)
    vp = val_points(T, L)
    comp = {name: {"point": point[name], "ci95": ci[name], "ci95_seed_inclusive": ci2[name],
                   **({"per_seed": per[name], "seed_se": se[name]} if name in se else {}),
                   "val_point_best_epoch": vp[name], **({"bar": bar} if bar is not None else {})}
            for name, _p, _x, _z, bar in COMPARISONS}
    hyp = {}
    for h, parts in HYP.items():
        met = all(point[p] >= comp[p]["bar"] - EPS for p in parts)
        strong = all(ci[p][0] >= comp[p]["bar"] - EPS for p in parts)  # M1's definition, not conditioned on met
        seed_miss = any(ci2[p] is not None and ci2[p][0] < comp[p]["bar"] - EPS for p in parts)
        hyp[h] = {"met": met, "strong": strong, "strong_item_level_only": bool(strong and seed_miss), "rule": RULE[h],
                  "per_probe": {p: {"holds": point[p] >= comp[p]["bar"] - EPS, "lower_clears": ci[p][0] >= comp[p]["bar"] - EPS,
                                    "seed_inclusive_lower_clears": (None if ci2[p] is None else ci2[p][0] >= comp[p]["bar"] - EPS)}
                                for p in parts}}
    for h in HYP:
        hyp[h]["claim"] = claim(h, hyp, comp)
    yv = np.asarray(ds.labels["val"])
    means = {f"{arm}@{rep}": {"test": round(float(100 * ok_tm[(arm, rep)].mean()), 4),
                              "test_per_seed": [round(float(100 * v), 4) for v in ok_tm[(arm, rep)].mean(1)],
                              "test_seed_sd": round(float(100 * ok_tm[(arm, rep)].mean(1).std(ddof=1)), 4),
                              "val": round(100 * sum(T[(arm, rep)][s]["val_acc"] for s in SEEDS) / len(SEEDS), 4),
                              "best_epochs": [T[(arm, rep)][s].get("best_epoch") for s in SEEDS]}
             for arm, rep in TM_CONFIGS}
    lrm = {arm: {"test": round(float(100 * ok_lr[arm].mean()), 4), "val": round(100 * L[arm]["lr_val"], 4),
                 "C": L[arm]["C"], "acc_by_C": L[arm].get("acc_by_C")} for arm in LR_ARMS}

    def subset(keep: np.ndarray, stream: int, rule: str, pools: str) -> dict:
        Dk, pk, _sek, perk = diffs(ok_tm, ok_lr, keep)
        cik = boot_items(Dk, stream)
        return {"rule": rule, "pools": pools, "n_left_out": int((~keep).sum()), "n_kept": int(keep.sum()),
                "comparisons": {k: {"point": pk[k], "ci95_items": cik[k], **({"per_seed": perk[k]} if k in perk else {})}
                                for k in pk},
                "tm": {f"{arm}@{rep}": round(float(100 * ok_tm[(arm, rep)][:, keep].mean()), 4) for arm, rep in TM_CONFIGS},
                "lr": {arm: round(float(100 * ok_lr[arm][keep].mean()), 4) for arm in LR_ARMS}}

    ne = lambda reasons: {"rule": "none (descriptive)", "status": "not evaluable", "reasons": reasons}
    ppu = matched.get("per_pool_union", {})
    pools = (f"c4_fit {ppu.get('c4_fit', '?')} (3a's C4 fit sentences: a3B's teacher-fit input, the one pool an M2 arm's "
             f"training touched); ours_train_val {ppu.get('ours_train_val', '?')} and paraphrases {ppu.get('paraphrases', '?')} "
             "(no M2 arm trained on them); the pools overlap")
    if m_errs:
        secondary, para = ne(m_errs), ne(m_errs)
    else:
        keep = np.ones(n, bool)
        keep[matched["positions"]] = False
        secondary = {**subset(keep, 2, "none (descriptive; not a contamination control for most pools)", pools),
                     "status": "evaluated"}
        pp = np.asarray(matched.get("paraphrase_exact", {}).get("positions", []), dtype=np.int64)
        para = {"rule": "none (descriptive; no M2 arm trains on paraphrases)", "status": "evaluated", "n": int(len(pp)),
                "tm": {f"{arm}@{rep}": (round(float(100 * ok_tm[(arm, rep)][:, pp].mean()), 4) if len(pp) else None)
                       for arm, rep in TM_CONFIGS},
                "lr": {arm: (round(float(100 * ok_lr[arm][pp].mean()), 4) if len(pp) else None) for arm in LR_ARMS}}
    sec_c4 = None
    if a.overlap is not None:
        if m_errs or o_errs:
            sec_c4 = ne(m_errs + o_errs)
        else:
            kc = np.ones(n, bool)
            kc[c4pos] = False
            sec_c4 = {**subset(kc, 3, "none (descriptive)", "c4_fit only (from fresh/mtop_overlap.json)"), "status": "evaluated"}
    counts = {"n_test": n, "n_val": int(len(yv)), "train_intents": len(ds.intents),
              "test_items_intent_not_in_train": int((y < 0).sum()), "val_items_intent_not_in_train": int((yv < 0).sum())}

    def flag(h: str) -> str:
        if hyp[h]["met"]:
            if hyp[h]["strong"]:
                return "MET (STRONG on the item-level interval only)" if hyp[h]["strong_item_level_only"] else "MET (STRONG)"
            return "MET (not strong)"
        return "NOT MET (the interval lower ends clear the bar)" if hyp[h]["strong"] else "NOT MET"

    head = "VERDICT M2 (MTOP English test, first use): " + "; ".join(f"{h} {flag(h)}" for h in HYP)
    res = {"headline": head, "set": SET, "hypotheses": hyp, "comparisons": comp, "tm": means, "lr": lrm, "counts": counts,
           "secondary_without_matched": secondary, "secondary_without_c4_fit": sec_c4, "paraphrase_exact_items": para,
           "bootstrap": {"resamples": N_BOOT, "registered": f"default_rng([{BOOT_SEED}, 0]), items only",
                         "seed_inclusive": f"default_rng([{BOOT_SEED}, 1]), items then seeds within each TM configuration",
                         "secondary": f"default_rng([{BOOT_SEED}, 2]), items only",
                         "secondary_c4_fit": f"default_rng([{BOOT_SEED}, 3]), items only"},
           "matched": {"path": str(a.matched), "sha256": hashlib.sha256(mbytes).hexdigest(),
                       "is_frozen_v1": hashlib.sha256(mbytes).hexdigest() == MATCHED_V1_SHA256},
           "overlap": ({"path": str(a.overlap), "sha256": hashlib.sha256(obytes).hexdigest()} if obytes is not None else None),
           "multiplicity": ("H1 and H2 are separate claims with no multiplicity correction; each stands alone. No combined "
                            "line; no claim rests on any one rule being met."),
           "interval_limit": ("Registered 95% intervals: paired bootstrap over test items (10,000 resamples, "
                              f"default_rng([{BOOT_SEED}, 0])); they leave out the variance between training seeds, the "
                              "k-means and C, so the per-seed differences, the seed SDs and a seed-inclusive two-level "
                              "interval (no rule; LR has no seed level) are given beside every TM number."),
           "provenance": PROVENANCE}
    lines = [head, PROVENANCE, res["multiplicity"], res["interval_limit"],
             f"n_test {n}; train intents {counts['train_intents']}; test items whose intent is not in en/train "
             f"{counts['test_items_intent_not_in_train']} (an error for every arm); val items {counts['val_items_intent_not_in_train']}.",
             "TM (mean of seeds 7-9): " + " / ".join(f"{k} {v['test']:.2f} (per seed {', '.join(f'{x:.2f}' for x in v['test_per_seed'])};"
                                                       f" seed SD {v['test_seed_sd']:.2f}; val {v['val']:.2f})" for k, v in means.items()),
             "LR (C picked on en/eval): " + " / ".join(f"{k} {v['test']:.2f} (C {v['C']:g}; val {v['val']:.2f})" for k, v in lrm.items()),
             "",
             "| Rule | Probe | Point | Bar | Holds | Item-level 95% (registered) | Lower clears | Seed-inclusive 95% (no rule) | Lower clears | en/eval (best epoch) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda b: "-" if b is None else ("yes" if b else "no")
    for h, parts in HYP.items():
        for p in parts:
            c, pp_ = comp[p], hyp[h]["per_probe"][p]
            lines.append(f"| {h} | {'TM' if p.endswith('_tm') else 'LR'} | {c['point']:+.2f} | {c['bar']:+.1f} | {yn(pp_['holds'])} | "
                         f"{iv(c['ci95'])} | {yn(pp_['lower_clears'])} | {iv(c['ci95_seed_inclusive'])} | "
                         f"{yn(pp_['seed_inclusive_lower_clears'])} | {c['val_point_best_epoch']:+.2f} |")
    lines.append("")
    lines.append("  rule parts (per-seed differences and seed SE for TM): " + "; ".join(show(p, comp) for h in HYP for p in HYP[h]))
    for h in HYP:
        lines.append(f"  {hyp[h]['claim']}")
    lines.append("  descriptive (no rule): " + "; ".join(show(nm, comp) for nm, _p, _x, _z, bar in COMPARISONS if bar is None))
    for label, sec in (("secondary (no rule), without the frozen matched list v1", secondary),
                       ("secondary (no rule), without the c4_fit pool only", sec_c4)):
        if sec is None:
            continue
        if sec["status"] == "not evaluable":
            lines.append(f"  {label}: NOT EVALUABLE ({'; '.join(sec['reasons'])}); H1 and H2 above do not use it")
            continue
        lines.append(f"  {label}: {sec['n_left_out']} left out (pools: {sec['pools']}), n {sec['n_kept']}: "
                     + "; ".join(f"{k} {v['point']:+.2f} [{v['ci95_items'][0]:+.2f}, {v['ci95_items'][1]:+.2f}]"
                                 for k, v in sec["comparisons"].items())
                     + "; TM " + " / ".join(f"{k} {v:.2f}" for k, v in sec["tm"].items())
                     + "; LR " + " / ".join(f"{k} {v:.2f}" for k, v in sec["lr"].items()))
    if para["status"] == "not evaluable":
        lines.append(f"  the paraphrase-exact items (no rule): NOT EVALUABLE ({'; '.join(para['reasons'])})")
    elif para["n"]:
        lines.append(f"  the {para['n']} paraphrase-exact items (no rule; no M2 arm trains on paraphrases): TM "
                     + " / ".join(f"{k} {v:.2f}" for k, v in para["tm"].items()) + "; LR "
                     + " / ".join(f"{k} {v:.2f}" for k, v in para["lr"].items()))
    if m_errs or o_errs:
        print("WARNING: secondary lines NOT EVALUABLE (H1 and H2 are written): " + "; ".join(m_errs + o_errs), file=sys.stderr)
    with open(outj, "x") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    with open(outt, "x") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
