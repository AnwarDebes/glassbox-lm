#!/usr/bin/env python
"""Line B (plan logs/plan_e70_verify.md, section "The MLP part"): the same 50 checks for the MLP twins
(scripts/train_mlp_twins.py), with the same domain, solver, options and time limit as scripts/verify_policy.py.

    verify_mlp.py --check encoding --hidden 64 128 512      # the pre-checks before the first MLP property run
    verify_mlp.py --check properties --hidden 128

Encoding (one hidden ReLU layer, eval mode, float32 weights read as float64):
- a_i = W1_i . x + b1_i as a continuous variable with exact bounds L_i, U_i over the 0/1 inputs;
- a unit with U_i <= 0 is dropped (always 0); with L_i >= 0, h_i = a_i; otherwise a binary z_i and
  h_i >= a_i, h_i <= a_i - L_i (1 - z_i), h_i <= U_i z_i, h_i >= 0;
- logits o = W2 h + b2; "B wins" is o_B - o_k >= 1e-6 for every compared k.
Its proofs carry a floating-point caveat (HiGHS tolerances; float32 weights). Written after the plan's freeze:
a recorded deviation from "freeze everything first". Added 2026-10-19 for Line B.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from verify_policy import (B_, HIGHS_OPTS, N_MEM, OUT, SOLVER, SRC, checks_for, decode, domain_rows,  # noqa: E402
                           nearest_dev, rows_hold)

TWINS = ROOT / "results" / "explore" / "mlp_twins"
MARGIN = 1e-6


def load_mlp(h: int, seed: int = 1) -> dict:
    z = np.load(TWINS / f"mlp_h{h}_seed{seed}.npz")
    return {k: z[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2")} | {"h": h, "f32": {k: z[k] for k in ("W1", "b1", "W2", "b2")}}


def logits_np(net: dict, X: np.ndarray) -> np.ndarray:
    return np.maximum(X.astype(np.float64) @ net["W1"].T + net["b1"], 0) @ net["W2"].T + net["b2"]


def logits_torch(net: dict, X: np.ndarray) -> np.ndarray:
    f = net["f32"]
    with torch.no_grad():
        a = torch.as_tensor(X, dtype=torch.float32) @ torch.as_tensor(f["W1"]).T + torch.as_tensor(f["b1"])
        return (torch.relu(a) @ torch.as_tensor(f["W2"]).T + torch.as_tensor(f["b2"])).numpy()


def build_and_solve(net: dict, rows_dom, fixed: dict[int, int], target: int, against: list[int], time_limit: float,
                    threads: int = 2, fix_by_bounds: bool = False) -> dict:
    """The MLP MILP, assembled block by block (sparse): x | a | h | z."""
    W1, b1, W2, b2 = net["W1"], net["b1"], net["W2"], net["b2"]
    H, F = W1.shape
    free = [i for i in range(F) if fix_by_bounds or i not in fixed]
    col = {i: k for k, i in enumerate(free)}
    fx = [i for i in fixed if i not in col]
    a0 = b1 + (W1[:, fx] @ np.array([fixed[i] for i in fx], np.float64) if fx else 0.0)
    Wf = W1[:, free]
    L = a0 + np.minimum(Wf, 0).sum(1)
    U = a0 + np.maximum(Wf, 0).sum(1)
    units = np.flatnonzero(U > 0)
    uns = np.flatnonzero(L[units] < 0)  # positions (within units) of the unstable ones
    stb = np.flatnonzero(L[units] >= 0)
    nx, na, nz = len(free), len(units), len(uns)
    Lu, Uu, a0u = L[units], U[units], a0[units]
    I = sp.identity(na, format="csr")
    blocks, lo, hi = [], [], []
    # a = W x + b:  -W x + a = b
    blocks.append(sp.hstack([sp.csr_matrix(-Wf[units]), I, sp.csr_matrix((na, na)), sp.csr_matrix((na, nz))]))
    lo.append(a0u); hi.append(a0u)
    Pu = sp.csr_matrix((np.ones(nz), (np.arange(nz), uns)), shape=(nz, na))  # picks the unstable units
    Ps = sp.csr_matrix((np.ones(len(stb)), (np.arange(len(stb)), stb)), shape=(len(stb), na))
    Iz = sp.identity(nz, format="csr")
    # unstable: h - a >= 0;  h - a - L z <= -L;  h - U z <= 0
    blocks.append(sp.hstack([sp.csr_matrix((nz, nx)), -Pu, Pu, sp.csr_matrix((nz, nz))])); lo.append(np.zeros(nz)); hi.append(np.full(nz, np.inf))
    blocks.append(sp.hstack([sp.csr_matrix((nz, nx)), -Pu, Pu, sp.diags(-Lu[uns])])); lo.append(np.full(nz, -np.inf)); hi.append(-Lu[uns])
    blocks.append(sp.hstack([sp.csr_matrix((nz, nx)), sp.csr_matrix((nz, na)), Pu, sp.diags(-Uu[uns])])); lo.append(np.full(nz, -np.inf)); hi.append(np.zeros(nz))
    # stable (always active): h - a = 0
    blocks.append(sp.hstack([sp.csr_matrix((len(stb), nx)), -Ps, Ps, sp.csr_matrix((len(stb), nz))])); lo.append(np.zeros(len(stb))); hi.append(np.zeros(len(stb)))
    # winner rows: (W2[B] - W2[k]) h >= margin - (b2[B] - b2[k])
    ks = [k for k in against if k != target]
    D = W2[target, units][None, :] - W2[ks][:, units]
    blocks.append(sp.hstack([sp.csr_matrix((len(ks), nx)), sp.csr_matrix((len(ks), na)), sp.csr_matrix(D), sp.csr_matrix((len(ks), nz))]))
    lo.append(np.array([MARGIN - (b2[target] - b2[k]) for k in ks])); hi.append(np.full(len(ks), np.inf))
    # the domain rows over x
    rr, cc, vv, dlo, dhi = [], [], [], [], []
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
        r = len(dlo)
        for c_, v_ in ent:
            rr.append(r); cc.append(c_); vv.append(v_)
        dlo.append(lo_ - shift); dhi.append(hi_ - shift)
    blocks.append(sp.hstack([sp.csr_matrix((vv, (rr, cc)), shape=(len(dlo), nx)), sp.csr_matrix((len(dlo), 2 * na + nz))]))
    lo.append(np.array(dlo, float)); hi.append(np.array(dhi, float))
    A = sp.vstack(blocks, format="csr")
    lov, hiv = np.concatenate(lo), np.concatenate(hi)
    lb = np.concatenate([np.zeros(nx), Lu, np.zeros(na), np.zeros(nz)])
    ub = np.concatenate([np.ones(nx), Uu, np.maximum(Uu, 0), np.ones(nz)])
    if fix_by_bounds:
        for i, val in fixed.items():
            lb[col[i]] = ub[col[i]] = val
    integ = np.concatenate([np.ones(nx), np.zeros(2 * na), np.ones(nz)])
    t0 = time.time()
    res = milp(c=np.zeros(A.shape[1]), constraints=LinearConstraint(A, lov, hiv), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "disp": False, "presolve": True, "threads": threads, **HIGHS_OPTS})
    out = {"seconds": round(time.time() - t0, 2), "highs_status": int(res.status), "message": str(res.message), "n_free_bits": nx,
           "n_units": na, "n_unstable": nz, "n_rows": int(A.shape[0]), "nnz": int(A.nnz), "nodes": int(getattr(res, "mip_node_count", -1) or -1),
           "solver": {**SOLVER, "time_limit": time_limit, "threads": threads, "margin": MARGIN, **HIGHS_OPTS}}
    if res.status == 0 and res.x is not None:
        x = np.zeros(F, np.uint8)
        for i, val in fixed.items():
            x[i] = val
        for i in free:
            if not (fix_by_bounds and i in fixed):
                x[i] = int(round(res.x[col[i]]))
        out.update(status="feasible", x=x)
    elif res.status == 2 or "infeasible" in str(res.message).lower():
        out["status"] = "infeasible"
    elif res.status == 1:
        out["status"] = "time limit"
    else:
        out["status"] = f"other ({res.status})"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["encoding", "properties"])
    ap.add_argument("--hidden", type=int, nargs="+", default=[128])
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-logits", type=int, default=500)
    ap.add_argument("--n-winner", type=int, default=50)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from dl.abcd import ABCDView, load_examples  # noqa: E402
    from lineb_counts import properties  # noqa: E402

    names = json.loads((SRC / "bit_names.json").read_text())
    rows_dom = domain_rows(names)
    tr, blist = load_examples("train")
    dv, _ = load_examples("dev")
    E = np.load(ROOT / "data" / "emb" / "abcd__potion-32m.npz")
    em = {s: {f: E[f"{s}__{f}"] for f in ("cust", "recent", "full")} for s in ("train", "dev")}
    view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, em["train"], blist)
    assert view.names == names
    Xdv = view.transform(dv, em["dev"])
    if args.check == "encoding":
        rng = np.random.default_rng(0)
        pick = rng.choice(len(Xdv), size=args.n_logits, replace=False)
        rec = {"tag": args.tag, "check": "mlp_encoding", "solver": {**SOLVER, "threads": args.threads, "margin": MARGIN, **HIGHS_OPTS},
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for h in args.hidden:
            net = load_mlp(h)
            Ln, Lt = logits_np(net, Xdv[pick]), logits_torch(net, Xdv[pick])
            win = []
            for i in pick[: args.n_winner]:
                fixed = {b: int(Xdv[i, b]) for b in range(len(names))}
                s = logits_np(net, Xdv[i : i + 1])[0]
                own, other = int(s.argmax()), int(np.argsort(-s)[1])
                r1 = build_and_solve(net, rows_dom, fixed, own, list(range(B_)), 120, args.threads, fix_by_bounds=True)
                r2 = build_and_solve(net, rows_dom, fixed, other, list(range(B_)), 120, args.threads, fix_by_bounds=True)
                win.append((r1["status"], r2["status"], r1["seconds"], r2["seconds"]))
            rec[f"h{h}"] = {"max_abs_logit_diff": float(np.abs(Ln - Lt).max()), "argmax_equal": int((Ln.argmax(1) == Lt.argmax(1)).sum()),
                            "rows": len(pick), "winner_own_feasible": sum(a == "feasible" for a, _, _, _ in win),
                            "winner_other_infeasible": sum(b == "infeasible" for _, b, _, _ in win), "winner_rows": len(win),
                            "winner_max_seconds": max(max(c, d) for _, _, c, d in win)}
            print(h, json.dumps(rec[f"h{h}"]), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return
    pre = [json.loads(l) for l in OUT.read_text().splitlines() if json.loads(l).get("check") == "mlp_encoding" and json.loads(l).get("tag") == "e70_mlp_precheck"]
    if not pre:
        raise SystemExit("MLP property runs start only after the MLP pre-check (tag e70_mlp_precheck)")
    props = properties(blist)
    checks = checks_for(props, blist)
    done_ = {(json.loads(l).get("tag"), json.loads(l).get("model"), json.loads(l).get("check_id")) for l in OUT.read_text().splitlines()}
    for h in args.hidden:
        net = load_mlp(h)
        model = f"mlp_h{h}"
        for ci, c in enumerate(checks):
            if (args.tag, model, ci) in done_:
                continue
            r = build_and_solve(net, rows_dom, c["fixed"], c["target"], c["against"], args.time_limit, args.threads)
            rec = {"tag": args.tag, "model": model, "seed": 1, "guarded": False, "check_id": ci, "property": c["property"],
                   "button": c["button"], "requires": c["requires"], **{k: v for k, v in r.items() if k != "x"},
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if r["status"] == "feasible":
                x = r["x"]
                s_np, s_t = logits_np(net, x[None])[0], logits_torch(net, x[None])[0]
                win = int(s_np.argmax())
                cond = (x[:N_MEM].sum() == 0) if c["property"] == "P_first" else (x[blist.index(c["requires"])] == 0)
                ok = bool(win == int(s_t.argmax()) and cond and rows_hold(rows_dom, x[None])[0]
                          and (win not in c["against"] if c["property"] == "P_first" else win == c["target"]))
                rec.update({"replay_ok": ok, "replay_winner": blist[win], "replay_margin": float(np.sort(s_np)[-1] - np.sort(s_np)[-2]),
                            "decoded": decode(x, names, blist), "nearest_dev": nearest_dev(x, Xdv), "x_on": np.flatnonzero(x).tolist()})
                rec["outcome"] = "counterexample" if ok else "replay failed"
            elif r["status"] == "infeasible":
                rec["outcome"] = "proved"
            elif r["status"] == "time limit":
                rec["outcome"] = "unknown"
            else:
                rec["outcome"] = r["status"]
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec[k] for k in ("model", "check_id", "property", "button", "outcome", "seconds", "nodes")}), flush=True)


if __name__ == "__main__":
    main()
