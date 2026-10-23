#!/usr/bin/env python
"""Line B (plan logs/plan_e70_verify.md): prove, for every input of a constrained bit domain, that a trained ABCD
policy follows properties derived from the written guidelines, or find a counterexample. MILP with scipy's HiGHS.

    verify_policy.py --check encoding --seeds 1          # the checks before any property run (dev rows only)
    verify_policy.py --check properties --seeds 1 [--guard]

The TM (R19abcd tm_mem_static, view B) is encoded exactly:
- clauses as cpu_tm_fast keeps them (no empty clause); fixed bits are evaluated away;
- each remaining clause j is c_j in [0, 1] with c_j <= l for its literals and c_j >= sum(l) - (n_j - 1);
- integer votes s_k = sum_j w_jk c_j + const_k;
- "B wins" is s_B - s_k >= 1 for k < B and s_B - s_k >= 0 for k > B (ties go to the lowest index, as argmax).
Domain (Level B): the bit constraints that every real input satisfies (memory consistency, thermometer chains,
1-2-gram implies its words). A proof holds for every bit vector in that domain, a superset of all real inputs.
Results: one JSON line per check in results/explore/verify_e70.jsonl. Dev only; the test split is not read.
Added 2026-10-19 for Line B.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.optimize._highspy import _core as _highs_core

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl import cpu_tm_fast  # noqa: E402

SRC = ROOT / "results" / "confirm" / "R19abcd" / "tm_mem_static"
OUT = ROOT / "results" / "explore" / "verify_e70.jsonl"
B_ = 30  # buttons
N_MEM = 2 * B_ + 6
CONCEPT = re.compile(r"^some word of the (customer's words so far|last 3 turns) is close to concept (\d+) .* at level (\d+) \(cos >= ([-0-9.]+)\)$")
WORD = re.compile(r'^last 3 turns contain "(.+)"$')
# HiGHS options passed verbatim through scipy (tolerances are HiGHS 1.12 defaults, set explicitly so each row records them)
HIGHS_OPTS = {"mip_feasibility_tolerance": 1e-6, "primal_feasibility_tolerance": 1e-7, "random_seed": 0}
SOLVER = {"scipy": scipy.__version__, "highs": f"{_highs_core.HIGHS_VERSION_MAJOR}.{_highs_core.HIGHS_VERSION_MINOR}.{_highs_core.HIGHS_VERSION_PATCH}"}
warnings.filterwarnings("ignore", message="Unrecognized options detected")


# ---------- the domain: constraints every real input satisfies ----------

def domain_rows(names: list[str]) -> list[tuple[dict[int, float], float, float]]:
    """Linear rows lo <= sum(coef * x) <= hi over the input bits (Level B)."""
    rows = []
    done = list(range(0, B_))
    last = list(range(B_, 2 * B_))
    cnt = list(range(2 * B_, N_MEM))
    assert names[0].startswith("done(") and names[B_].startswith("last action = ") and names[2 * B_] == "at least 1 actions so far"
    for b in range(B_):
        rows.append(({last[b]: 1, done[b]: -1}, -np.inf, 0))  # last = b implies done(b)
        rows.append(({done[b]: 1, cnt[0]: -1}, -np.inf, 0))  # any done implies at least 1 action
    rows.append(({i: 1 for i in last}, -np.inf, 1))  # at most one last
    rows.append(({**{i: 1 for i in last}, cnt[0]: -1}, 0, 0))  # some last iff at least 1 action
    rows.append(({**{i: 1 for i in done}, cnt[0]: -1}, 0, np.inf))  # at least 1 action implies some done
    for k in range(len(cnt) - 1):
        rows.append(({cnt[k + 1]: 1, cnt[k]: -1}, -np.inf, 0))  # the count chain
    # distinct done buttons <= number of actions, while "at least 6" is off
    rows.append(({**{i: 1 for i in done}, **{c: -1 for c in cnt}, cnt[-1]: -1 - B_}, -np.inf, 0))
    # thermometer chains
    chains: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for i, n in enumerate(names):
        m = CONCEPT.match(n)
        if m:
            chains.setdefault((m[1], int(m[2])), []).append((int(m[3]), i))
    n_chain = 0
    for key, lv in chains.items():
        lv.sort()
        for (_, lo_i), (_, hi_i) in zip(lv, lv[1:]):
            rows.append(({hi_i: 1, lo_i: -1}, -np.inf, 0))  # level j+1 implies level j
            n_chain += 1
    # a 1-2-gram implies its words (one CountVectorizer stream, the same text)
    word_idx = {m[1]: i for i, n in enumerate(names) if (m := WORD.match(n))}
    n_word = 0
    for w, i in word_idx.items():
        parts = w.split(" ")
        if len(parts) == 2:
            for p in parts:
                if p in word_idx:
                    rows.append(({i: 1, word_idx[p]: -1}, -np.inf, 0))
                    n_word += 1
    domain_rows.stats = {"rows": len(rows), "thermometer_rows": n_chain, "word_rows": n_word, "concept_chains": len(chains)}
    return rows


def rows_hold(rows, X: np.ndarray) -> np.ndarray:
    """[n] bool: every domain row holds for each bit vector."""
    ok = np.ones(len(X), bool)
    Xf = X.astype(np.int64)
    for coef, lo, hi in rows:
        v = sum(c * Xf[:, i] for i, c in coef.items())
        ok &= (v >= lo) & (v <= hi)
    return ok


# ---------- the TM as linear pieces ----------

def load_tm(seed: int, guards: list | None = None) -> dict:
    z = np.load(SRC / f"model_seed{seed}.npz")
    m = cpu_tm_fast.pack(z["ta"], z["w"])
    clauses = [m["lits"][s:e].tolist() for s, e in zip(m["starts"], list(m["starts"][1:]) + [len(m["lits"])])]
    W = m["w"].copy()
    if guards:  # veto clauses: (literal index over L = [x, 1 - x], class, weight)
        for lit, cls, wt in guards:
            clauses.append([lit])
            row = np.zeros((1, W.shape[1]), np.int64)
            row[0, cls] = wt
            W = np.concatenate([W, row], 0)
    return {"clauses": clauses, "W": W, "F": m["n_features"], "pack": m}


def tm_scores_fixed(tm: dict, x: np.ndarray) -> np.ndarray:
    """Votes with every bit fixed, through the same partial evaluation the MILP uses."""
    F = tm["F"]
    lit_val = np.concatenate([x, 1 - x])
    fire = np.array([all(lit_val[l] for l in c) for c in tm["clauses"]], np.int64)
    return fire @ tm["W"]


# ---------- one MILP: can class B win under the fixed bits? ----------

def build_and_solve(tm: dict, rows_dom, fixed: dict[int, int], target: int, against: list[int], time_limit: float,
                    fix_by_bounds: bool = False, threads: int = 2) -> dict:
    F, W = tm["F"], tm["W"]
    free = [i for i in range(F) if (fix_by_bounds or i not in fixed)]
    col = {i: k for k, i in enumerate(free)}
    const = np.zeros(W.shape[1], np.int64)
    active = []  # (clause index, [(input index, positive?)])
    for j, lits in enumerate(tm["clauses"]):
        keep, dead = [], False
        for l in lits:
            i, pos = (l, True) if l < F else (l - F, False)
            if i in fixed and not fix_by_bounds:
                if (fixed[i] == 1) != pos:
                    dead = True
                    break
                continue  # a true fixed literal
            keep.append((i, pos))
        if dead:
            continue
        if not keep:
            const += W[j]
            continue
        active.append((j, keep))
    nx, nc = len(free), len(active)
    r_i, c_i, v, lo, hi = [], [], [], [], []
    r = 0

    def add(entries, lo_, hi_):
        nonlocal r
        for cc, vv in entries:
            r_i.append(r)
            c_i.append(cc)
            v.append(vv)
        lo.append(lo_)
        hi.append(hi_)
        r += 1

    for a, (j, keep) in enumerate(active):
        cj = nx + a
        n_pos = 0
        for i, pos in keep:  # c_j <= l
            if pos:
                add([(cj, 1), (col[i], -1)], -np.inf, 0)
            else:
                add([(cj, 1), (col[i], 1)], -np.inf, 1)
        # c_j >= sum(l) - (n - 1): c_j - sum(pos x) + sum(neg x) >= 1 - n + (#neg)
        ent = [(cj, 1)] + [(col[i], -1 if pos else 1) for i, pos in keep]
        n_neg = sum(not pos for _, pos in keep)
        add(ent, 1 - len(keep) + n_neg, np.inf)
    for k in against:
        if k == target:
            continue
        d = W[[j for j, _ in active], target] - W[[j for j, _ in active], k]
        theta = 1 if k < target else 0
        ent = [(nx + a, int(d[a])) for a in range(nc) if d[a] != 0]
        add(ent, theta - int(const[target] - const[k]), np.inf)
    n_dom = 0
    for coef, lo_, hi_ in rows_dom:
        ent, shift = [], 0
        for i, cf in coef.items():
            if i in col:
                ent.append((col[i], cf))
            else:
                shift += cf * fixed[i]
        if not ent:
            if not (lo_ <= shift <= hi_):
                return {"status": "fixed bits break the domain", "seconds": 0.0}
            continue
        add(ent, lo_ - shift, hi_ - shift)
        n_dom += 1
    A = sp.csr_matrix((v, (r_i, c_i)), shape=(r, nx + nc))
    lb = np.zeros(nx + nc)
    ub = np.ones(nx + nc)
    if fix_by_bounds:
        for i, val in fixed.items():
            lb[col[i]] = ub[col[i]] = val
    integrality = np.concatenate([np.ones(nx), np.zeros(nc)])
    t0 = time.time()
    res = milp(c=np.zeros(nx + nc), constraints=LinearConstraint(A, np.array(lo, float), np.array(hi, float)),
               integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "disp": False, "presolve": True, "threads": threads, **HIGHS_OPTS})
    out = {"seconds": round(time.time() - t0, 2), "highs_status": int(res.status), "message": str(res.message),
           "n_free_bits": nx, "n_active_clauses": nc, "n_rows": r, "n_domain_rows": n_dom,
           "nodes": int(getattr(res, "mip_node_count", -1) or -1), "solver": {**SOLVER, "time_limit": time_limit, "threads": threads, **HIGHS_OPTS}}
    if res.status == 0 and res.x is not None:
        x = np.zeros(F, np.uint8)
        for i, val in fixed.items():
            x[i] = val
        for i in free:
            if not (fix_by_bounds and i in fixed):
                x[i] = int(round(res.x[col[i]]))
        out["status"] = "feasible"
        out["x"] = x
    elif res.status == 2 or "infeasible" in str(res.message).lower():
        out["status"] = "infeasible"
    elif res.status == 1:
        out["status"] = "time limit"
    else:
        out["status"] = f"other ({res.status})"
    return out


# ---------- the 50 checks, guards, replay ----------

def checks_for(props: dict, blist: list[str]) -> list[dict]:
    """P_first: one check per forbidden button (does it beat all 5 allowed ones with an empty history);
    P_prec(A, B): one check per pair (does B win with done(A) = 0)."""
    idx = {b: i for i, b in enumerate(blist)}
    empty = {i: 0 for i in range(N_MEM)}
    out = []
    for b in props["forbidden_first"]:
        out.append({"property": "P_first", "button": b, "requires": None, "fixed": empty, "target": idx[b],
                    "against": [idx[a] for a in props["allowed_first"]], "guard_lit": ("not", 2 * B_)})
    for a, b in props["prec"]:
        out.append({"property": f"P_prec({a}, {b})", "button": b, "requires": a, "fixed": {idx[a]: 0}, "target": idx[b],
                    "against": [k for k in range(B_) if k != idx[b]], "guard_lit": ("not", idx[a])})
    return out


def guards_for(tm: dict, checks: list[dict], blist: list[str]) -> list[tuple[int, int, int]]:
    """One veto clause per check: its literal (not done(A), or not "at least 1 action") with weight -M on B.
    M = B's positive weights + the absolute weights of pull-up-account + 1: pull-up-account is never vetoed, and
    with the guard on, B's votes stay below pull-up-account's for every input."""
    W, F = tm["W"], tm["F"]
    pua = blist.index("pull-up-account")
    assert all(c["button"] != "pull-up-account" for c in checks)
    out = []
    for c in checks:
        b = c["target"]
        M = int(np.clip(W[:, b], 0, None).sum() + np.abs(W[:, pua]).sum() + 1)
        out.append((F + c["guard_lit"][1], b, -M))
    return out


def guard_sums(tm: dict, X: np.ndarray) -> np.ndarray:
    """Class sums of the (possibly guarded) model for many rows: the packed clauses plus the guard clauses."""
    S = cpu_tm_fast.class_sums(tm["pack"], X)
    for lit, cls, wt in tm.get("guards", []):
        F = tm["F"]
        fires = X[:, lit] if lit < F else 1 - X[:, lit - F]
        S[:, cls] += wt * fires.astype(np.int64)
    return S


def decode(x: np.ndarray, names: list[str], blist: list[str]) -> dict:
    done = [blist[i] for i in range(B_) if x[i]]
    last = [blist[i - B_] for i in range(B_, 2 * B_) if x[i]]
    count = int(x[2 * B_ : N_MEM].sum())
    words = [WORD.match(names[i])[1] for i in range(N_MEM, len(names)) if x[i] and WORD.match(names[i])]
    conc = {}
    for i in range(N_MEM, len(names)):
        if x[i] and (m := CONCEPT.match(names[i])):
            key = ("cust" if m[1].startswith("customer") else "recent", int(m[2]))
            conc[key] = max(conc.get(key, -1), int(m[3]))
    return {"done": done, "last": last, "at_least": count, "words_on": words,
            "concepts_on": {"cust": sum(k[0] == "cust" for k in conc), "recent": sum(k[0] == "recent" for k in conc)}}


def nearest_dev(x: np.ndarray, Xdv: np.ndarray) -> dict:
    """Nearest real dev row by Hamming distance over the text bits: among rows with the same memory bits if any, else all."""
    same = np.flatnonzero((Xdv[:, :N_MEM] == x[None, :N_MEM]).all(1))
    pool = same if len(same) else np.arange(len(Xdv))
    d = (Xdv[pool, N_MEM:] != x[None, N_MEM:]).sum(1)
    j = int(pool[int(d.argmin())])
    return {"row": j, "text_hamming": int(d.min()), "same_memory_rows": int(len(same)), "equal_row": bool(d.min() == 0 and len(same) > 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["encoding", "properties", "guardcost"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--n-votes", type=int, default=500)
    ap.add_argument("--n-winner", type=int, default=50)
    ap.add_argument("--guard", action="store_true", help="the guarded model: one veto clause per check (plan, section Guards)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    names = json.loads((SRC / "bit_names.json").read_text())
    rows_dom = domain_rows(names)
    print("domain", domain_rows.stats, flush=True)
    if args.check == "encoding":
        from dl.abcd import ABCDView, load_examples  # noqa: E402
        from abcd_closed_loop import closed_loop  # noqa: E402

        tr, blist = load_examples("train")
        dv, _ = load_examples("dev")
        E = np.load(ROOT / "data" / "emb" / "abcd__potion-32m.npz")
        em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
        view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, em["train"], blist)
        assert view.names == names
        Xtr, Xdv = view.transform(tr, em["train"]), view.transform(dv, em["dev"])
        rec = {"tag": args.tag, "check": "encoding", "domain": domain_rows.stats, "solver": {**SOLVER, "threads": args.threads, **HIGHS_OPTS},
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rec["domain_holds_train"] = int(rows_hold(rows_dom, Xtr).sum()), len(Xtr)
        rec["domain_holds_dev"] = int(rows_hold(rows_dom, Xdv).sum()), len(Xdv)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(Xdv), size=args.n_votes, replace=False)
        for seed in args.seeds:
            tm = load_tm(seed)
            S_fast = cpu_tm_fast.class_sums(tm["pack"], Xdv[pick])
            S_enc = np.stack([tm_scores_fixed(tm, Xdv[i]) for i in pick])
            p_closed = closed_loop(lambda Z: cpu_tm_fast.class_sums(tm["pack"], Z), Xdv, dv, blist, N_MEM)
            Xc = Xdv.copy()
            by_conv: dict[int, list[int]] = {}
            for i, r in enumerate(dv):
                by_conv.setdefault(r["convo_id"], []).append(i)
            from dl.abcd import memory_from_rows  # noqa: E402
            for idx in by_conv.values():
                idx.sort(key=lambda i: dv[i]["k"])
                for j, i in enumerate(idx):
                    Xc[i, :N_MEM] = memory_from_rows([{"prev": [blist[p_closed[q]] for q in idx[:j]]}], blist)[0]
            win = []
            for i in pick[: args.n_winner]:
                fixed = {b: int(Xdv[i, b]) for b in range(len(names))}
                s = S_fast[list(pick).index(i)]
                own = int(s.argmax())
                other = int(np.argsort(-s)[1])
                r1 = build_and_solve(tm, rows_dom, fixed, own, list(range(B_)), 60, fix_by_bounds=True, threads=args.threads)
                r2 = build_and_solve(tm, rows_dom, fixed, other, list(range(B_)), 60, fix_by_bounds=True, threads=args.threads)
                win.append((r1["status"], r2["status"], r1["seconds"], r2["seconds"]))
            rec[f"seed{seed}"] = {
                "votes_equal": int((S_fast == S_enc).all(1).sum()), "votes_rows": len(pick),
                "closed_loop_domain_holds": int(rows_hold(rows_dom, Xc).sum()),
                "winner_own_feasible": sum(a == "feasible" for a, _, _, _ in win), "winner_other_infeasible": sum(b == "infeasible" for _, b, _, _ in win),
                "winner_rows": len(win), "winner_max_seconds": max(max(c, d) for _, _, c, d in win)}
            print(seed, json.dumps(rec[f"seed{seed}"]), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        return
    if args.check == "guardcost":
        # the guards' accuracy cost on DEV, open loop (logged memory) and closed loop (own presses); no solver
        from dl.abcd import ABCDView, load_examples  # noqa: E402
        from abcd_closed_loop import closed_loop  # noqa: E402
        from lineb_counts import properties  # noqa: E402

        tr, blist = load_examples("train")
        dv, _ = load_examples("dev")
        checks = checks_for(properties(blist), blist)
        E = np.load(ROOT / "data" / "emb" / "abcd__potion-32m.npz")
        em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
        view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, em["train"], blist)
        assert view.names == names
        Xdv = view.transform(dv, em["dev"])
        y = np.array([r["y"] for r in dv])
        rec = {"tag": args.tag, "check": "guardcost", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for seed in args.seeds:
            tm = load_tm(seed)
            g = dict(tm, guards=guards_for(tm, checks, blist))
            plain = lambda Z: cpu_tm_fast.class_sums(tm["pack"], Z)  # noqa: E731
            guarded = lambda Z: guard_sums(g, Z)  # noqa: E731
            r = {}
            for loop in ("open", "closed"):
                if loop == "open":
                    p0, p1 = plain(Xdv).argmax(1), guarded(Xdv).argmax(1)
                else:
                    p0, p1 = closed_loop(plain, Xdv, dv, blist, N_MEM), closed_loop(guarded, Xdv, dv, blist, N_MEM)
                ch = p0 != p1
                r[loop] = {"acc": float((p0 == y).mean()), "acc_guarded": float((p1 == y).mean()), "changed": int(ch.sum()),
                           "changed_right_before": int((ch & (p0 == y)).sum()), "changed_right_after": int((ch & (p1 == y)).sum())}
            rec[f"seed{seed}"] = r
            print(seed, json.dumps(r), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return
    # property runs (plan logs/plan_e70_verify.md; frozen before the first one)
    frozen = ROOT / "logs" / "plan_e70_verify.frozen.json"
    if not frozen.exists():
        raise SystemExit("property runs start only after the plan is frozen (logs/plan_e70_verify.md)")
    from dl.abcd import ABCDView, load_examples  # noqa: E402
    from lineb_counts import properties  # noqa: E402

    tr, blist = load_examples("train")
    dv, _ = load_examples("dev")
    assert all(names[i] == f"done({blist[i]})" for i in range(B_)) and names[2 * B_] == "at least 1 actions so far"
    props = properties(blist)
    checks = checks_for(props, blist)
    assert len(checks) == 50, len(checks)
    E = np.load(ROOT / "data" / "emb" / "abcd__potion-32m.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, em["train"], blist)
    assert view.names == names
    Xdv = view.transform(dv, em["dev"])
    done_ = {(json.loads(l).get("tag"), json.loads(l).get("seed"), json.loads(l).get("guarded"), json.loads(l).get("check_id"))
             for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    for seed in args.seeds:
        tm = load_tm(seed)
        if args.guard:
            tm["guards"] = guards_for(tm, checks, blist)
            tm["clauses"] = tm["clauses"] + [[lit] for lit, _, _ in tm["guards"]]
            G = np.zeros((len(tm["guards"]), tm["W"].shape[1]), np.int64)
            for g, (_, cls, wt) in enumerate(tm["guards"]):
                G[g, cls] = wt
            tm["W"] = np.concatenate([tm["W"], G], 0)
        for ci, c in enumerate(checks):
            key = (args.tag, seed, bool(args.guard), ci)
            if key in done_:
                continue  # already in the results file: never run twice
            r = build_and_solve(tm, rows_dom, c["fixed"], c["target"], c["against"], args.time_limit, threads=args.threads)
            rec = {"tag": args.tag, "model": "tm", "seed": seed, "guarded": bool(args.guard), "check_id": ci, "property": c["property"],
                   "button": c["button"], "requires": c["requires"], **{k: v for k, v in r.items() if k != "x"},
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if r["status"] == "feasible":
                x = r["x"]
                s_enc = tm_scores_fixed(tm, x)
                s_fast = guard_sums(tm, x[None])[0]
                win = int(s_enc.argmax())
                cond = (x[:N_MEM].sum() == 0) if c["property"] == "P_first" else (x[blist.index(c["requires"])] == 0)
                ok = bool((s_enc == s_fast).all() and cond and rows_hold(rows_dom, x[None])[0]
                          and (win not in c["against"] if c["property"] == "P_first" else win == c["target"]))
                rec.update({"replay_ok": ok, "replay_winner": blist[win], "decoded": decode(x, names, blist), "nearest_dev": nearest_dev(x, Xdv),
                            "x_on": np.flatnonzero(x).tolist()})
                rec["outcome"] = "counterexample" if ok else "replay failed"
            elif r["status"] == "infeasible":
                rec["outcome"] = "proved"
            elif r["status"] == "time limit":
                rec["outcome"] = "unknown"
            else:
                rec["outcome"] = r["status"]
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec[k] for k in ("seed", "guarded", "check_id", "property", "button", "outcome", "seconds", "nodes")}), flush=True)


if __name__ == "__main__":
    main()
