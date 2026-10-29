#!/usr/bin/env python
"""Independent recompute of the 3c verdict, written from tu/PLAN_3c.md ("Features and readers", "Bar"), not from
tu/c3_verdict.py (only that script's output keys were read, for --compare).

    OMP_NUM_THREADS=1 nice -n 19 $PY -I recompute_3c.py --base tu/c3/base.json --val tu/c3/results_c3_val.jsonl \
        --lr tu/c3/results_c3_lr.jsonl --w2-lr w2/results_w2_lr.jsonl --out <new file> [--compare tu/c3/verdict_c3.json]

The plan's rules, as read here:
- Reference per set: when <base.json feats_F>/<set>__p32tok_K512.meta.json says equivalence.identical, the base's own
  rows (base.json ref_tm: its tag, table = the base, s = the base's s, seeds 4-6; ref_lr: table = the base); otherwise
  the rebuilt originals (TM tag 3c-ref, arm orig; LR name ref_rebuilt).
- TM gain: the mean over seeds 4-6 of the paired differences, 3c-F (tag 3c-tm, arm F) minus the reference, same seed.
- LR gain: lr_val of F minus the reference's. Every gain is in points (x 100) and unrounded.
- PASS: both gains at least 0.5 - 1e-9 on at least 2 of the 3 sets; otherwise FAIL.
Checks (each one is reported; any failure gives exit code 1): exactly one row per set, seed and arm; every TM row at
the base's s; the F rows were trained on the F features with n_orig originals plus n_aug paraphrases (the F meta); the
code-check row (3c-check, CLINC150, seed 4) equals the base's row; c3_lr.py's 'ref' row equals base.json's ref_lr row.
Descriptive: the per-seed gains, 3c-C's LR gain, potion's TM mean (base.json potion_tm) and LR (W2's potion row), and
the share of each gap that is closed.
--compare: the registered verdict file must give the same verdict, count of sets, reference rows, n_orig and n_aug,
and each of its rounded numbers must be within its rounding (0.005; 0.0005 for the shares) of this recompute's.
The output file is never overwritten. Every input file's sha256 is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

SETS = ("clinc150", "banking77", "hwu64")
SEEDS = (4, 5, 6)
POTION = "potion-base-32M"
BAR = 0.5 - 1e-9


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def failed(self) -> list[dict]:
        return [c for c in self.items if not c["ok"]]


def one(rows: list[dict], what: str, ck: Checks, **cond) -> dict | None:
    m = [r for r in rows if all(r.get(k) == v for k, v in cond.items())]
    ck.add(f"one row: {what}", len(m) == 1, f"{len(m)} rows match {cond}")
    return m[0] if len(m) == 1 else None


def same_path(a, b) -> bool:
    return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()


def per_set(d: str, base: dict, val: list[dict], lr: list[dict], w2: list[dict], ref_tm: list[dict],
            ref_lr: list[dict], pot_tm: list[dict], ck: Checks, files: dict) -> dict | None:
    s = float(base["s"][d])
    mpath = Path(base["feats_F"]) / f"{d}__p32tok_K512.meta.json"
    files[str(mpath)] = sha(mpath)
    meta = json.loads(mpath.read_text())
    identical = bool(meta["equivalence"]["identical"])
    n_orig, n_aug = int(meta["n_orig"]), int(meta["n_aug"])
    F, R = [], []
    for sd in SEEDS:
        f = one(val, f"3c-tm F {d} seed {sd}", ck, tag="3c-tm", arm="F", dataset=d, seed=sd)
        if identical:
            r = one(ref_tm, f"base TM {d} seed {sd}", ck, dataset=d, seed=sd, s=s)
        else:
            r = one(val, f"3c-ref orig {d} seed {sd}", ck, tag="3c-ref", arm="orig", dataset=d, seed=sd)
        if f is None or r is None:
            continue
        ck.add(f"s of the TM rows {d} seed {sd}", float(f["s"]) == s and float(r["s"]) == s,
               f"F {f['s']}, reference {r['s']}, base {s}")
        ck.add(f"F row trained on the F features {d} seed {sd}", same_path(f.get("emb_dir"), base["feats_F"])
               and f.get("n_orig") == n_orig and f.get("n_train") == n_orig + n_aug,
               f"emb_dir {f.get('emb_dir')}, n_orig {f.get('n_orig')}, n_train {f.get('n_train')}; meta {n_orig} + {n_aug}")
        F.append(float(f["val_acc"]))
        R.append(float(r["val_acc"]))
    lF = one(lr, f"LR F {d}", ck, name="F", dataset=d)
    lC = one(lr, f"LR C {d}", ck, name="C", dataset=d)
    lnew = one(lr, f"LR ref {d}", ck, name="ref", dataset=d)
    lbase = one(ref_lr, f"base LR {d}", ck, dataset=d)
    if lnew is not None and lbase is not None:
        ck.add(f"LR code check {d}: c3_lr.py's ref row equals the base's", lnew["lr_val"] == lbase["lr_val"],
               f"{lnew['lr_val']} vs {lbase['lr_val']}")
    lref = lbase if identical else one(lr, f"LR ref_rebuilt {d}", ck, name="ref_rebuilt", dataset=d)
    pot = [one(pot_tm, f"potion TM {d} seed {sd}", ck, dataset=d, seed=sd, s=float(base["potion_tm"]["s"][d]))
           for sd in SEEDS]
    lpot = one(w2, f"potion LR {d}", ck, table=POTION, dataset=d, view="existsT")
    if len(F) != len(SEEDS) or None in (lF, lC, lref, lpot) or None in pot:
        return None
    paired = [100 * (x - y) for x, y in zip(F, R)]
    g_tm = st.mean(paired)
    g_lr = 100 * (float(lF["lr_val"]) - float(lref["lr_val"]))
    g_lr_c = 100 * (float(lC["lr_val"]) - float(lref["lr_val"]))
    tm_ref, tm_f = 100 * st.mean(R), 100 * st.mean(F)
    tm_pot = 100 * st.mean(float(p["val_acc"]) for p in pot)
    lr_ref, lr_pot = 100 * float(lref["lr_val"]), 100 * float(lpot["lr_val"])
    gap_tm, gap_lr = tm_pot - tm_ref, lr_pot - lr_ref
    return {"s": s, "identical": identical, "reference_rows": "base" if identical else "rebuilt originals",
            "n_orig": n_orig, "n_aug_F": n_aug, "tm_F": tm_f, "tm_ref": tm_ref, "tm_gain": g_tm, "tm_paired": paired,
            "lr_F": 100 * float(lF["lr_val"]), "lr_ref": lr_ref, "lr_gain": g_lr, "lr_C": 100 * float(lC["lr_val"]),
            "lr_gain_C": g_lr_c, "tm_potion": tm_pot, "lr_potion": lr_pot,
            "tm_share_of_gap": g_tm / gap_tm if gap_tm > 0 else None,
            "lr_share_of_gap": g_lr / gap_lr if gap_lr > 0 else None,
            "meets_bar": g_tm >= BAR and g_lr >= BAR}


def compare(v: dict, res: dict, ck: Checks) -> None:
    ck.add("compare: verdict", v.get("verdict") == res["verdict"], f"registered {v.get('verdict')}, recompute {res['verdict']}")
    ck.add("compare: sets meeting the bar", v.get("sets_meeting_bar") == res["sets_meeting_bar"],
           f"registered {v.get('sets_meeting_bar')}, recompute {res['sets_meeting_bar']}")
    for d in SETS:
        mine, theirs = res["sets"].get(d), v.get(d)
        if mine is None or not isinstance(theirs, dict):
            ck.add(f"compare {d}: present", False, f"recompute {mine is not None}, registered {isinstance(theirs, dict)}")
            continue
        for k in ("reference_rows", "n_orig", "n_aug_F"):
            ck.add(f"compare {d}: {k}", theirs.get(k) == mine[k], f"registered {theirs.get(k)}, recompute {mine[k]}")
        for k in ("tm_F", "tm_ref", "tm_gain", "lr_F", "lr_ref", "lr_gain", "lr_C", "lr_gain_C", "tm_potion", "lr_potion"):
            t = theirs.get(k)
            ck.add(f"compare {d}: {k}", t is not None and abs(t - mine[k]) <= 0.005 + 1e-9, f"registered {t}, recompute {mine[k]:.6f}")
        tp = theirs.get("tm_paired") or []
        ck.add(f"compare {d}: tm_paired", len(tp) == len(mine["tm_paired"])
               and all(abs(a - b) <= 0.005 + 1e-9 for a, b in zip(tp, mine["tm_paired"])),
               f"registered {tp}, recompute {[round(x, 6) for x in mine['tm_paired']]}")
        for k in ("tm_share_of_gap", "lr_share_of_gap"):
            t, m = theirs.get(k), mine[k]
            ok = (t is None and m is None) or (t is not None and m is not None and abs(t - m) <= 0.0005 + 1e-9)
            ck.add(f"compare {d}: {k}", ok, f"registered {t}, recompute {m}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--lr", type=Path, required=True)
    ap.add_argument("--w2-lr", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path)
    a = ap.parse_args()
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    base = json.loads(a.base.read_text())
    files = {str(p): sha(p) for p in (a.base, a.val, a.lr, a.w2_lr, Path(base["ref_tm"]["file"]),
                                      Path(base["ref_lr"]["file"]), Path(base["potion_tm"]["file"]))}
    ck = Checks()
    val, lr, w2 = jsonl(a.val), jsonl(a.lr), jsonl(a.w2_lr)
    ref_tm = [r for r in jsonl(base["ref_tm"]["file"]) if r.get("tag") == base["ref_tm"]["tag"] and r.get("table") == base["base"]]
    ref_lr = [r for r in jsonl(base["ref_lr"]["file"]) if r.get("table") == base["base"]]
    pot_tm = [r for r in jsonl(base["potion_tm"]["file"]) if r.get("tag") == base["potion_tm"]["tag"] and r.get("table") == POTION]
    s_c = float(base["s"]["clinc150"])
    mine = one(val, "3c-check clinc150 seed 4", ck, tag="3c-check", dataset="clinc150", seed=4)
    theirs = one(ref_tm, "base TM clinc150 seed 4 (code check)", ck, dataset="clinc150", seed=4, s=s_c)
    if mine is not None and theirs is not None:
        ck.add("code check: the 3c-check row equals the base's row", mine["val_acc"] == theirs["val_acc"]
               and float(mine["s"]) == s_c, f"{mine['val_acc']} at s {mine['s']} vs {theirs['val_acc']} at s {s_c}")
    sets = {}
    for d in SETS:
        x = per_set(d, base, val, lr, w2, ref_tm, ref_lr, pot_tm, ck, files)
        if x is not None:
            sets[d] = x
    n_pass = sum(1 for x in sets.values() if x["meets_bar"])
    complete = len(sets) == len(SETS)
    res = {"recompute": "verify_3c/recompute_3c.py", "sha256_self": sha(Path(__file__).resolve()),
           "verdict": ("PASS" if n_pass >= 2 else "FAIL") if complete else "INCOMPLETE",
           "sets_meeting_bar": n_pass, "bar": "both gains >= 0.5 - 1e-9 points on at least 2 of 3 sets", "sets": sets,
           "inputs_sha256": files}
    if a.compare:
        res["inputs_sha256"][str(a.compare)] = sha(a.compare)
        compare(json.loads(a.compare.read_text()), res, ck)
    res["checks"] = ck.items
    res["checks_failed"] = len(ck.failed())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    for d, x in sets.items():
        print(f"{d} (s {x['s']:g}, reference rows: {x['reference_rows']}, +{x['n_aug_F']} on {x['n_orig']}): "
              f"TM {x['tm_F']:.2f} vs {x['tm_ref']:.2f} = {x['tm_gain']:+.4f} (paired {', '.join(f'{p:+.2f}' for p in x['tm_paired'])}); "
              f"LR {x['lr_F']:.2f} vs {x['lr_ref']:.2f} = {x['lr_gain']:+.4f}; 3c-C LR {x['lr_gain_C']:+.2f}; "
              f"potion TM {x['tm_potion']:.2f}, LR {x['lr_potion']:.2f}; meets the bar: {x['meets_bar']}")
    print(f"recompute verdict: {res['verdict']} ({n_pass} of 3 sets); checks: {len(ck.items) - len(ck.failed())} ok, "
          f"{len(ck.failed())} failed")
    for c in ck.failed():
        print(f"FAILED CHECK: {c['check']}: {c['detail']}")
    return 0 if not ck.failed() and complete else 1


if __name__ == "__main__":
    sys.exit(main())
