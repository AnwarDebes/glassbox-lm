"""E2.5: solver checks of the guarded TM (e70's method; verify_policy.build_and_solve imported unedited, scipy HiGHS).
Decision-layer venv, CPU only:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python -I -B scripts/milp_check.py \
        --models models/<tag> --rows data/rows_gt_v1.2.2.jsonl [...] --qbits data/query_bits --out results/<file>.jsonl

Per fold model d4_tm_<suite>.json:
1. Domain: every labelled row's bit vector (all 4 suites, that fold's query bits) satisfies the domain rows
   (e2guard.bits.domain_rows); reported as holds / rows.
2. Encoding pre-check on 500 rows (seeded): the votes of verify_policy.tm_scores_fixed (the MILP's own partial
   evaluation) equal the pure-Python decision's (e2guard.models, the one the guard runs) with bias and veto clauses;
   and with every bit fixed (fix_by_bounds), "the CPU winner wins" is feasible and "the other class wins" infeasible.
3. Properties: for each written-invariant premise (W1: 8, W2: 3) -- fix the premise bits to 1, every other bit free in
   the domain -- can ALLOW win?
   - raw TM (learned clauses + the tau bias, no veto clauses): a counterexample is expected; each feasible answer is
     replayed through the pure-Python decision (premise holds, domain holds, ALLOW wins);
   - guarded TM (with the veto clauses): must be infeasible for every premise ("proved"), by construction.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

E2 = Path(__file__).resolve().parents[1]
DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(E2 / "src"))
sys.path.insert(1, str(DL / "scripts"))

import verify_policy as VP

from e2guard.bits import (
    FACT_BITS,
    QueryBits,
    domain_rows,
    rows_hold,
    veto_premises,
)
from e2guard.models import ALLOW, DENY, GuardedTM

SUITES = ["banking", "slack", "travel", "workspace"]


def tm_dict(model: dict, with_veto: bool) -> dict:
    keep = [j for j, k in enumerate(model["clause_kind"]) if with_veto or k != "veto"]
    return {"clauses": [model["clauses"][j] for j in keep], "W": np.array([model["W"][j] for j in keep], np.int64),
            "F": model["F"]}


def vectors(rows_paths: list[str], qb: QueryBits, names: list[str]) -> np.ndarray:
    col = {n: i for i, n in enumerate(names)}
    X = []
    for p in rows_paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("label") not in ("ALLOW", "DENY") or r["facts"]["kind"] is None:
                continue
            x = np.zeros(len(names), np.uint8)
            for n in set(r["facts_on"]) | qb.on(r["query_sha256"]):
                x[col[n]] = 1
            X.append(x)
    return np.stack(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--qbits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rows", type=int, default=500)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--folds", nargs="*", default=SUITES)
    args = ap.parse_args()
    out = Path(args.out)
    assert not out.exists(), f"{out} exists"
    fo = out.open("w")
    summary = {}

    def emit(rec: dict) -> None:
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fo.write(json.dumps(rec, default=str) + "\n")
        fo.flush()

    for held in args.folds:
        model = json.loads((Path(args.models) / f"d4_tm_{held}.json").read_text())
        names = model["names"]
        qb = QueryBits(Path(args.qbits) / f"fold_{held}.json")
        assert names == FACT_BITS + qb.names
        dom = domain_rows(names)
        g = GuardedTM(model)
        X = vectors(args.rows, qb, names)
        holds = sum(rows_hold(dom, x.tolist()) for x in X)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(X), size=min(args.n_rows, len(X)), replace=False)
        tmg = tm_dict(model, with_veto=True)
        votes_equal = winner_own = winner_other = 0
        t_max = 0.0
        for i in pick:
            x = X[i]
            s_py, _ = g.scores(x.tolist())
            s_enc = VP.tm_scores_fixed(tmg, x)
            votes_equal += int(list(s_enc) == list(s_py))
            own = ALLOW if s_py[ALLOW] > s_py[DENY] else DENY
            other = 1 - own
            fixed = {b: int(x[b]) for b in range(len(names))}
            r1 = VP.build_and_solve(tmg, dom, fixed, own, [other], 60, fix_by_bounds=True, threads=args.threads)
            r2 = VP.build_and_solve(tmg, dom, fixed, other, [own], 60, fix_by_bounds=True, threads=args.threads)
            winner_own += r1["status"] == "feasible"
            winner_other += r2["status"] == "infeasible"
            t_max = max(t_max, r1["seconds"], r2["seconds"])
        pre = {"fold": held, "check": "encoding", "domain_rows": len(dom), "rows_in_domain": int(holds),
               "rows_total": len(X), "votes_equal": votes_equal, "winner_own_feasible": winner_own,
               "winner_other_infeasible": winner_other, "precheck_rows": len(pick), "max_seconds": t_max,
               "solver": VP.SOLVER}
        emit(pre)
        print(json.dumps(pre), flush=True)
        res = {"raw": {}, "guarded": {}}
        col = {n: i for i, n in enumerate(names)}
        for variant in ("raw", "guarded"):
            tm = tm_dict(model, with_veto=(variant == "guarded"))
            for v in veto_premises():
                fixed = {col[b]: 1 for b in v["bits"]}
                r = VP.build_and_solve(tm, dom, fixed, ALLOW, [DENY], args.time_limit, threads=args.threads)
                rec = {"fold": held, "check": "property", "variant": variant, "premise": v["id"], "rule": v["rule"],
                       **{k: val for k, val in r.items() if k != "x"}}
                if r["status"] == "feasible":
                    x = r["x"]
                    d = g.decide(x.tolist(), with_veto=(variant == "guarded"))
                    ok = bool(all(x[col[b]] == 1 for b in v["bits"]) and rows_hold(dom, x.tolist()) and d["allow"])
                    rec.update({"outcome": "counterexample" if ok else "replay failed", "replay_ok": ok,
                                "x_on": [names[i] for i in np.flatnonzero(x)][:80], "margin": d["margin"]})
                elif r["status"] == "infeasible":
                    rec["outcome"] = "proved"
                elif r["status"] == "time limit":
                    rec["outcome"] = "unknown"
                else:
                    rec["outcome"] = r["status"]
                res[variant][v["id"]] = rec["outcome"]
                emit(rec)
        summary[held] = {"encoding": {k: pre[k] for k in ("rows_in_domain", "rows_total", "votes_equal",
                                                          "winner_own_feasible", "winner_other_infeasible",
                                                          "precheck_rows")},
                         "raw": res["raw"], "guarded": res["guarded"]}
        print(held, json.dumps(summary[held]), flush=True)
    emit({"check": "summary", "summary": summary})
    fo.close()


if __name__ == "__main__":
    main()
