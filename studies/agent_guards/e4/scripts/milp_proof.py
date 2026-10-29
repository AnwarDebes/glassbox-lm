"""E4.4 part 2 and the guard proofs: MILP checks of the hard rules on the trained TM, raw and guarded (e70's method).

Solver: verify_policy.build_and_solve (glassbox-lm/studies/decision_layer/scripts/verify_policy.py, imported
unedited; scipy HiGHS). Runs in the decision-layer venv, CPU, 2 solver threads. The output records the sha256 of
verify_policy.py, of this script, of every e4guard module, of the model and of every rows file (review issue 1).

For one exported CoTM (scripts/train_guard.py):
  pre-checks (as e70):
    D  every real row (all row files given) satisfies the domain rows (bits.domain_rows): the domain is a superset;
    E  the MILP encoding's votes (verify_policy.tm_scores_fixed) equal the deployed decision's votes
       (e4guard.models.TMGuard.class_sums) on up to 500 rows, raw and guarded;
    W  on up to 50 rows with every bit fixed, the deployed winner is feasible and the other class infeasible.
  properties, one per veto clause (13; rules E4-R1..E4-R10): "can ALLOW win when the clause's facts hold?", every other bit
  free in the domain. Raw TM: a counterexample is possible (it is replayed through the deployed code and decoded to
  bit names). Guarded TM: must be infeasible (a proof); a feasible answer means a bug.
Run: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python -I -B milp_proof.py --model m.json \
       --rows r1.json r2.json --out results/milp_<tag>.json
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
DL = Path(f"{_REPO_ROOT}/studies/decision_layer")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(DL / "scripts"))

import numpy as np  # noqa: E402
import verify_policy as vp  # noqa: E402  (read-only import; -B keeps it from writing bytecode)

from e4guard import bits as B  # noqa: E402
from e4guard.models import ALLOW, DENY, TMGuard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-votes", type=int, default=500)
    ap.add_argument("--n-winner", type=int, default=50)
    args = ap.parse_args()
    t0 = time.time()
    d = json.loads(Path(args.model).read_text())
    raw, grd = TMGuard(d, guarded=False), TMGuard(d, guarded=True)
    tms = {"raw": raw.as_milp_tm(), "guarded": grd.as_milp_tm()}
    dom = B.domain_rows()
    X = np.concatenate([np.array([r["x"] for r in json.loads(Path(p).read_text())["rows"]], np.uint8)
                        for p in args.rows])
    code = [DL / "scripts/verify_policy.py", Path(__file__)] + sorted((E4 / "e4guard").glob("*.py"))
    rec = {"model": args.model, "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
           "rows_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.rows},
           "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
           "model_meta": {k: d.get("meta", {}).get(k) for k in ("tag", "stand_in", "seed", "cfg")},
           "solver": {**vp.SOLVER, **vp.HIGHS_OPTS, "time_limit": args.time_limit, "threads": args.threads},
           "n_bits": B.N_BITS, "n_domain_rows": len(dom), "n_learned_clauses": raw.n_learned, "M": grd.M,
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # D
    ok = B.rows_hold(dom, X)
    rec["pre_D_domain_holds"] = [int(ok.sum()), int(len(X))]
    bad_rows = np.flatnonzero(~ok)[:5].tolist()
    if bad_rows:
        rec["pre_D_failing_examples"] = [[B.NAMES[i] for i in np.flatnonzero(X[j])] for j in bad_rows]
    # E
    rng = np.random.default_rng(0)
    pick = rng.choice(len(X), size=min(args.n_votes, len(X)), replace=False)
    for kind, tmo in (("raw", raw), ("guarded", grd)):
        S_dep = tmo.class_sums(X[pick])
        S_enc = np.stack([vp.tm_scores_fixed(tms[kind], X[i]) for i in pick])
        rec[f"pre_E_votes_equal_{kind}"] = [int((S_dep == S_enc).all(1).sum()), int(len(pick))]
    # W
    win = []
    for i in pick[: args.n_winner]:
        fixed = {b: int(X[i, b]) for b in range(B.N_BITS)}
        s = grd.class_sums(X[i])[0]
        own = int(np.argmax(s))
        other = 1 - own
        r1 = vp.build_and_solve(tms["guarded"], dom, fixed, own, [0, 1], 60, fix_by_bounds=True, threads=args.threads)
        r2 = vp.build_and_solve(tms["guarded"], dom, fixed, other, [0, 1], 60, fix_by_bounds=True, threads=args.threads)
        win.append((r1["status"], r2["status"]))
    rec["pre_W_own_feasible"] = [sum(a == "feasible" for a, _ in win), len(win)]
    rec["pre_W_other_infeasible"] = [sum(b == "infeasible" for _, b in win), len(win)]
    # properties
    checks = []
    for cid, (rule, lits) in B.VETO.items():
        fixed = {B.IDX[n]: int(p) for n, p in lits}
        for kind in ("raw", "guarded"):
            r = vp.build_and_solve(tms[kind], dom, fixed, ALLOW, [ALLOW, DENY], args.time_limit, threads=args.threads)
            c = {"clause": cid, "rule": rule, "model": kind, "status": r["status"], "seconds": r.get("seconds"),
                 "nodes": r.get("nodes"), "n_free_bits": r.get("n_free_bits"), "n_active_clauses": r.get("n_active_clauses")}
            if r["status"] == "feasible":
                x = r["x"]
                S = (raw if kind == "raw" else grd).class_sums(x)[0]
                c["replay_allow_wins"] = bool(S[ALLOW] >= S[DENY])
                c["replay_domain_holds"] = bool(B.rows_hold(dom, x[None])[0])
                c["replay_clause_fires"] = cid in B.veto_fires(x)
                c["outcome"] = "counterexample" if (c["replay_allow_wins"] and c["replay_domain_holds"]
                                                    and c["replay_clause_fires"]) else "replay failed"
                c["bits_on"] = [B.NAMES[i] for i in np.flatnonzero(x)]
            elif r["status"] == "infeasible":
                c["outcome"] = "proved"
            else:
                c["outcome"] = "unknown" if r["status"] == "time limit" else r["status"]
            checks.append(c)
            print(json.dumps({k: c[k] for k in ("clause", "model", "outcome", "seconds")}), flush=True)
    rec["checks"] = checks
    rec["summary"] = {kind: {o: sum(1 for c in checks if c["model"] == kind and c["outcome"] == o)
                             for o in ("proved", "counterexample", "unknown", "replay failed")} for kind in ("raw", "guarded")}
    rec["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in rec if k not in ("checks",)}, indent=1))
    g = rec["summary"]["guarded"]
    sys.exit(0 if g["proved"] == len(B.VETO) and rec["pre_D_domain_holds"][0] == rec["pre_D_domain_holds"][1] else 1)


if __name__ == "__main__":
    main()
