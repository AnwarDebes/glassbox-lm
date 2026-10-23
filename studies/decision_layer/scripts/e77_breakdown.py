#!/usr/bin/env python
"""Where the text part fails (dev only; CPU; read only; no rule): the dev decisions of saved explore models, broken down.

    e77_breakdown.py --tags e77_mwoz_Bc4_c16000_s16 e77_mwoz_bge_c16000_s8 [--out NAME]

For each tag, the one TM row of results/explore/abcd.jsonl with that tag and its saved model
(results/explore/abcd_models/<tag>_seed<seed>.npz, written by abcd_explore.py --save). The view is rebuilt as the row's
run built it: the C4 view (encoder "none") through dl.count_text.patch, as table_run.py does, or the bge-large view
from its cache (or dl.bitcache's cached dev bits, if their hash is the row's). Two checks before any breakdown: the dev bits' SHA-256 equals the row's, and the CPU decision
(dl.cpu_tm_fast, integer votes) gives exactly the row's dev accuracy.

Breakdowns of dev accuracy (MultiWOZ labels are sorted "Domain-act" sets joined by "+"; on other datasets only the
first three apply):
- first action of a dialogue (k = 0) against later ones;
- per gold label (the 30 most frequent);
- the 15 most frequent confusions (gold -> predicted);
- domain right: the predicted label has the gold label's set of domains (Attraction, Booking, Hotel, Restaurant, Taxi,
  Train, general); act right: it has the gold label's set of acts (Inform, Request, ...), whatever the domain;
- per gold domain set and per gold act.
With two or more tags, each tag's numbers sit side by side, and the first tag minus each other tag is given per
breakdown cell. Writes results/explore/e77_breakdown_<out>.json (refuses to overwrite). Added 2026-10-19 (numbers
on where the text part fails).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dl import count_text, cpu_tm_fast  # noqa: E402


def row_for(tag: str) -> dict:
    got = [d for d in map(json.loads, (ROOT / "results" / "explore" / "abcd.jsonl").read_text().splitlines())
           if d.get("arm") == "tm" and d.get("tag") == tag]
    if len(got) != 1:
        raise SystemExit(f"{tag}: {len(got)} rows (need exactly 1)")
    return got[0]


def doms(label: str) -> tuple:
    return tuple(sorted({p.split("-")[0] for p in label.split("+")})) if "-" in label else (label,)


def acts(label: str) -> tuple:
    return tuple(sorted({p.split("-", 1)[1] for p in label.split("+")})) if "-" in label else (label,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = ROOT / "results" / "explore" / f"e77_breakdown_{args.out or '_'.join(args.tags)}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; not overwriting")
    rows = {t: row_for(t) for t in args.tags}
    ds = {r["dataset"] for r in rows.values()}
    if len(ds) != 1:
        raise SystemExit(f"tags from more than one dataset: {ds}")
    ds = ds.pop()
    tables = {r["static_info"]["xcust"]["table"] for r in rows.values() if r["encoder"] == "none"}
    if len(tables) > 1:
        raise SystemExit(f"more than one table: {tables}")
    if tables:
        count_text.patch(tables.pop())
    from dl.abcd import ABCDView, load_examples

    tr, blist = load_examples("train", ds)
    dv, _ = load_examples("dev", ds)
    y = np.array([r["y"] for r in dv])
    first = np.array([r["k"] == 0 for r in dv])
    gold = [blist[i] for i in y]
    res = {"what": "e77 breakdown of dev decisions (dev only, no rule)", "dataset": ds, "n_dev": len(dv), "tags": {}}
    preds = {}
    for t, r in rows.items():
        X, src = None, "view rebuilt"
        cache = ROOT / "data" / ds / f"bits_{'-'.join(r['parts'])}_dev.npz"  # dl.bitcache's file for the bge-large view
        if r["encoder"] == "bge-large" and r["lex_k"] == 1000 and cache.exists():
            Xc = np.load(cache)["X"]
            if hashlib.sha256(np.ascontiguousarray(Xc, dtype=np.uint8).tobytes()).hexdigest() == r["bits_sha256"]["dev"]:
                X, src = Xc, str(cache.relative_to(ROOT))
        if X is None:  # the bge-large view's centres are drawn on the GPU, so its rebuild needs one
            E = np.load(ROOT / "data" / "emb" / f"{ds}__{r['encoder']}.npz")
            emb = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
            view = ABCDView(parts=tuple(r["parts"]), lex_k=r["lex_k"], seed=0).fit(tr, emb["train"], blist)
            X = view.transform(dv, emb["dev"])
        h = hashlib.sha256(np.ascontiguousarray(X, dtype=np.uint8).tobytes()).hexdigest()
        if h != r["bits_sha256"]["dev"]:
            raise SystemExit(f"{t}: dev bits {h[:8]} != the row's {r['bits_sha256']['dev'][:8]}")
        z = np.load(ROOT / "results" / "explore" / "abcd_models" / f"{t}_seed{r['seed']}.npz")
        if list(z["buttons"]) != blist:
            raise SystemExit(f"{t}: the model's buttons differ")
        p = cpu_tm_fast.class_sums(cpu_tm_fast.pack(z["ta"], z["w"]), X).argmax(1)
        acc = float((p == y).mean())
        if acc != r["acc"]:
            raise SystemExit(f"{t}: CPU accuracy {acc} != the row's {r['acc']}")
        preds[t] = p
        print(f"{t}: bits {h[:8]} ({src}) and accuracy {acc:.4f} equal the row's", flush=True)
    top = [lb for lb, _ in Counter(gold).most_common(30)]
    for t, p in preds.items():
        ok = p == y
        pl = [blist[i] for i in p]
        d = {"acc": float(ok.mean()), "acc_first_action": float(ok[first].mean()), "acc_later_actions": float(ok[~first].mean()),
             "n_first": int(first.sum()), "per_label": {}, "confusions": [], "domain_right": None, "act_right": None,
             "per_domain_set": {}, "per_act": {}}
        for lb in top:
            m = np.array([g == lb for g in gold])
            d["per_label"][lb] = {"n": int(m.sum()), "acc": float(ok[m].mean()), "share_predicted_as": float(np.mean([q == lb for q in pl]))}
        d["confusions"] = [[g, q, n] for (g, q), n in Counter((g, q) for g, q in zip(gold, pl) if g != q).most_common(15)]
        if any("-" in g for g in gold):
            dr = np.array([doms(g) == doms(q) for g, q in zip(gold, pl)])
            ar = np.array([acts(g) == acts(q) for g, q in zip(gold, pl)])
            d["domain_right"], d["act_right"] = float(dr.mean()), float(ar.mean())
            d["domain_right_among_wrong"] = float(dr[~ok].mean()) if (~ok).any() else None
            d["act_right_among_wrong"] = float(ar[~ok].mean()) if (~ok).any() else None
            for key, fn in (("per_domain_set", doms), ("per_act", acts)):
                cats = Counter(c for g in gold for c in ([fn(g)] if key == "per_domain_set" else fn(g)))
                for c, n in cats.most_common(20):
                    m = np.array([(fn(g) == c) if key == "per_domain_set" else (c in fn(g)) for g in gold])
                    d[key]["+".join(c) if isinstance(c, tuple) else c] = {"n": int(m.sum()), "acc": float(ok[m].mean())}
        res["tags"][t] = d
    if len(args.tags) > 1:
        a = args.tags[0]
        res["first_minus_other"] = {}
        for b in args.tags[1:]:
            A, B = res["tags"][a], res["tags"][b]
            diff = {k: round(100 * (A[k] - B[k]), 2) for k in ("acc", "acc_first_action", "acc_later_actions", "domain_right", "act_right")
                    if A.get(k) is not None and B.get(k) is not None}
            for key in ("per_label", "per_domain_set", "per_act"):
                diff[key] = {c: round(100 * (A[key][c]["acc"] - B[key][c]["acc"]), 2) for c in A[key] if c in B[key]}
            both = (preds[a] == y) & (preds[b] == y)
            diff["both_right"] = float(both.mean())
            diff["only_first_right"] = float(((preds[a] == y) & (preds[b] != y)).mean())
            diff["only_other_right"] = float(((preds[a] != y) & (preds[b] == y)).mean())
            res["first_minus_other"][f"{a} - {b}"] = diff
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    for t, d in res["tags"].items():
        print(f"{t}: acc {d['acc']:.4f}, first {d['acc_first_action']:.4f} (n {d['n_first']}), later {d['acc_later_actions']:.4f}"
              + (f", domain right {d['domain_right']:.4f}, act right {d['act_right']:.4f}" if d["domain_right"] is not None else ""))
    print("wrote", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
