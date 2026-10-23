#!/usr/bin/env python
"""e72d (note logs/plan_e72d_guards_c4.md): e70's checks and guards on the agent with the C4-word table.

    verify_policy_c4.py --check encoding               # pre-check (dev rows only)
    verify_policy_c4.py --check properties [--guard]   # the 50 checks per model, raw or guarded
    verify_policy_c4.py --check guardcost              # the guards' cost on dev, open and closed loop

The models are R21's five ABCD tm_mem_c4 models (results/confirm/R21abcd/tm_mem_c4: model_seed<k>.npz and
bit_names.json; view B on the C4 table). Everything else is e70's, imported from scripts/verify_policy.py
unedited: the same 50 checks (lineb_counts.properties), the exact MILP encoding and HiGHS options, the level-B domain
rows (built from these models' bit names), and the guards, with M recomputed per model from its own weights.
The dev bits are built as in the runs: dl.count_text.patch (potion's table cannot load) and the empty "none" cache
(no encoder vector). Dev and proof work only: no test split is read. Rows go to results/explore/verify_e72d.jsonl.
Added 2026-10-19.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

import verify_policy as vp  # noqa: E402
from dl import count_text, cpu_tm_fast  # noqa: E402

SRC = ROOT / "results" / "confirm" / "R21abcd" / "tm_mem_c4"
OUT = ROOT / "results" / "explore" / "verify_e72d.jsonl"
TABLE = "c4words-S1000-v1"


def write(rec: dict) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["encoding", "properties", "guardcost"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--n-votes", type=int, default=500)
    ap.add_argument("--n-winner", type=int, default=50)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--tag", default="e72d")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    meta = count_text.patch(TABLE)
    vp.SRC = SRC  # load_tm reads model_seed<k>.npz from here
    from abcd_closed_loop import closed_loop  # noqa: E402
    from dl.abcd import ABCDView, load_examples, memory_from_rows  # noqa: E402
    from lineb_counts import properties  # noqa: E402

    names = json.loads((SRC / "bit_names.json").read_text())
    rows_dom = vp.domain_rows(names)
    print("domain", vp.domain_rows.stats, flush=True)
    tr, blist = load_examples("train")
    dv, _ = load_examples("dev")
    E = np.load(ROOT / "data" / "emb" / "abcd__none.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, em["train"], blist)
    assert view.names == names, "the view's names differ from the models' bit_names.json"
    Xdv = view.transform(dv, em["dev"])
    base = {"tag": args.tag, "models": str(SRC.relative_to(ROOT)), "table": meta["table"], "table_file_sha256": meta["table_file_sha256"]}
    checks = vp.checks_for(properties(blist), blist)
    assert len(checks) == 50, len(checks)

    if args.check == "encoding":
        Xtr = view.transform(tr, em["train"])
        rec = {**base, "check": "encoding", "domain": vp.domain_rows.stats, "solver": {**vp.SOLVER, "threads": args.threads, **vp.HIGHS_OPTS},
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "domain_holds_train": [int(vp.rows_hold(rows_dom, Xtr).sum()), len(Xtr)],
               "domain_holds_dev": [int(vp.rows_hold(rows_dom, Xdv).sum()), len(Xdv)]}
        rng = np.random.default_rng(0)
        pick = rng.choice(len(Xdv), size=args.n_votes, replace=False)
        by_conv: dict[int, list[int]] = {}
        for i, r in enumerate(dv):
            by_conv.setdefault(r["convo_id"], []).append(i)
        for idx in by_conv.values():
            idx.sort(key=lambda i: dv[i]["k"])
        for seed in args.seeds:
            tm = vp.load_tm(seed)
            S_fast = cpu_tm_fast.class_sums(tm["pack"], Xdv[pick])
            S_enc = np.stack([vp.tm_scores_fixed(tm, Xdv[i]) for i in pick])
            p_closed = closed_loop(lambda Z: cpu_tm_fast.class_sums(tm["pack"], Z), Xdv, dv, blist, vp.N_MEM)
            Xc = Xdv.copy()
            for idx in by_conv.values():
                for j, i in enumerate(idx):
                    Xc[i, : vp.N_MEM] = memory_from_rows([{"prev": [blist[p_closed[q]] for q in idx[:j]]}], blist)[0]
            win = []
            for n_, i in enumerate(pick[: args.n_winner]):
                fixed = {b: int(Xdv[i, b]) for b in range(len(names))}
                s = S_fast[n_]
                own, other = int(s.argmax()), int(np.argsort(-s)[1])
                r1 = vp.build_and_solve(tm, rows_dom, fixed, own, list(range(vp.B_)), 60, fix_by_bounds=True, threads=args.threads)
                r2 = vp.build_and_solve(tm, rows_dom, fixed, other, list(range(vp.B_)), 60, fix_by_bounds=True, threads=args.threads)
                win.append((r1["status"], r2["status"], r1["seconds"], r2["seconds"]))
            rec[f"seed{seed}"] = {
                "votes_equal": int((S_fast == S_enc).all(1).sum()), "votes_rows": len(pick),
                "closed_loop_domain_holds": int(vp.rows_hold(rows_dom, Xc).sum()),
                "winner_own_feasible": sum(a == "feasible" for a, _, _, _ in win),
                "winner_other_infeasible": sum(b == "infeasible" for _, b, _, _ in win),
                "winner_rows": len(win), "winner_max_seconds": max(max(c, d) for _, _, c, d in win)}
            print(seed, json.dumps(rec[f"seed{seed}"]), flush=True)
        write(rec)
        print(json.dumps(rec), flush=True)
        return

    if args.check == "guardcost":
        y = np.array([r["y"] for r in dv])
        rec = {**base, "check": "guardcost", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for seed in args.seeds:
            tm = vp.load_tm(seed)
            g = dict(tm, guards=vp.guards_for(tm, checks, blist))
            plain = lambda Z: cpu_tm_fast.class_sums(tm["pack"], Z)  # noqa: E731
            guarded = lambda Z: vp.guard_sums(g, Z)  # noqa: E731
            r = {"guards_M": [int(-wt) for _, _, wt in g["guards"]]}
            for loop in ("open", "closed"):
                if loop == "open":
                    p0, p1 = plain(Xdv).argmax(1), guarded(Xdv).argmax(1)
                else:
                    p0, p1 = closed_loop(plain, Xdv, dv, blist, vp.N_MEM), closed_loop(guarded, Xdv, dv, blist, vp.N_MEM)
                ch = p0 != p1
                r[loop] = {"acc": float((p0 == y).mean()), "acc_guarded": float((p1 == y).mean()), "changed": int(ch.sum()),
                           "changed_right_before": int((ch & (p0 == y)).sum()), "changed_right_after": int((ch & (p1 == y)).sum())}
            rec[f"seed{seed}"] = r
            print(seed, json.dumps({k: v for k, v in r.items() if k != "guards_M"}), flush=True)
        write(rec)
        return

    # property runs: only after the note is frozen
    if not (ROOT / "logs" / "plan_e72d_guards_c4.frozen.json").exists():
        raise SystemExit("property runs start only after logs/plan_e72d_guards_c4.md is frozen")
    done_ = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            d = json.loads(line)
            done_.add((d.get("tag"), d.get("seed"), d.get("guarded"), d.get("check_id")))
    for seed in args.seeds:
        tm = vp.load_tm(seed)
        if args.guard:
            tm["guards"] = vp.guards_for(tm, checks, blist)
            tm["clauses"] = tm["clauses"] + [[lit] for lit, _, _ in tm["guards"]]
            G = np.zeros((len(tm["guards"]), tm["W"].shape[1]), np.int64)
            for gi, (_, cls, wt) in enumerate(tm["guards"]):
                G[gi, cls] = wt
            tm["W"] = np.concatenate([tm["W"], G], 0)
        for ci, c in enumerate(checks):
            if (args.tag, seed, bool(args.guard), ci) in done_:
                continue  # already in the results file: never run twice
            r = vp.build_and_solve(tm, rows_dom, c["fixed"], c["target"], c["against"], args.time_limit, threads=args.threads)
            rec = {**base, "model": "tm_c4", "seed": seed, "guarded": bool(args.guard), "check_id": ci, "property": c["property"],
                   "button": c["button"], "requires": c["requires"], **{k: v for k, v in r.items() if k != "x"},
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if r["status"] == "feasible":
                x = r["x"]
                s_enc = vp.tm_scores_fixed(tm, x)
                s_fast = vp.guard_sums(tm, x[None])[0]
                win = int(s_enc.argmax())
                cond = (x[: vp.N_MEM].sum() == 0) if c["property"] == "P_first" else (x[blist.index(c["requires"])] == 0)
                ok = bool((s_enc == s_fast).all() and cond and vp.rows_hold(rows_dom, x[None])[0]
                          and (win not in c["against"] if c["property"] == "P_first" else win == c["target"]))
                rec.update({"replay_ok": ok, "replay_winner": blist[win], "decoded": vp.decode(x, names, blist),
                            "nearest_dev": vp.nearest_dev(x, Xdv), "x_on": np.flatnonzero(x).tolist()})
                rec["outcome"] = "counterexample" if ok else "replay failed"
            elif r["status"] == "infeasible":
                rec["outcome"] = "proved"
            elif r["status"] == "time limit":
                rec["outcome"] = "unknown"
            else:
                rec["outcome"] = r["status"]
            write(rec)
            print(json.dumps({k: rec[k] for k in ("seed", "guarded", "check_id", "property", "button", "outcome", "seconds", "nodes")}), flush=True)


if __name__ == "__main__":
    main()
