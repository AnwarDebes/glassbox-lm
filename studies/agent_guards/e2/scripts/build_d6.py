"""Build the D6 rule sets (ActGov-style), one per held-out suite, with the solver check of W1-W2 in every round.
Decision-layer venv (scipy HiGHS through verify_policy.build_and_solve, imported unedited). CPU only:

    STAND-IN (offline): .venv-dl/bin/python -I -B scripts/build_d6.py --proposer scripted --rows ... --qbits ... --out models/<tag>
    GPU slot:           .venv-dl/bin/python -I -B scripts/build_d6.py --proposer qwen --base-url http://127.0.0.1:<port>/v1 ...

Per fold (held-out suite h): the proposer sees only the 3 training suites' benign rows (ALLOW-labelled) and the tool
docs; up to 10 rounds; after each round the rules are parsed (unknown names rejected), solver-checked (for every W1-W2
premise: can any rule fire on a domain vector where the premise holds?) and scored on the training rows (benign calls
allowed, DENY calls blocked). Used rule set: the best that passed the solver check (most benign training calls allowed;
ties: more DENY calls blocked, then the earlier round). If no round passes, the arm is recorded as "no rule set passed"
and the file holds no rules (deny all).
rev2 (review item 13): every fold's file also records `nontrivial`: deny_all (no rules) and allows_state_changing (the
solver finds a fact vector of the domain with kind=read false on which some rule fires; false = the rules allow only
reads), plus the benign state-changing training calls allowed. E2.3 is "not testable" if any fold is deny-all or
allows only reads (report_e2.py). Qwen's requests go through the replay store (--store).
rev3 (registration section 13; the decision 3): a transport error of the rule writer (failures.is_transport)
stops the build stage: <out>/STOPPED.json is written and the script exits with 3 (the job file logs the stop; the stage
is rerun once in full on the same store). Any other request error is caught per round: that round has no rule set and
the error is recorded among its problems. The default store is the campaign's runs/sealed/replay_store.jsonl.
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

from e2guard.bits import FACT_BITS, QueryBits, domain_rows, veto_premises
from e2guard.d6_actgov import (
    QwenProposer,
    ScriptedProposer,
    build_messages,
    parse_rules,
    rule_literals,
    rules_fire,
)
from e2guard.facts import ToolTable
from e2guard.failures import EXIT_TRANSPORT, StageStopped, error_type, is_transport
from e2guard.replay_store import ReplayStore, store_record

SUITES = ["banking", "slack", "travel", "workspace"]


def solver_check(rules: list[list[str]], names: list[str], dom, time_limit: float) -> dict:
    if not rules:
        return {"passed": True, "per_premise": {}, "note": "no rules: nothing can be allowed"}
    tm = {"clauses": [rule_literals(r, names) for r in rules], "W": np.array([[0, 1]] * len(rules), np.int64),
          "F": len(names)}
    col = {n: i for i, n in enumerate(names)}
    per, cex = {}, []
    for v in veto_premises():
        r = VP.build_and_solve(tm, dom, {col[b]: 1 for b in v["bits"]}, 1, [0], time_limit, threads=2)
        per[v["id"]] = r["status"]
        if r["status"] == "feasible":
            on = {names[i] for i in np.flatnonzero(r["x"])}
            fired = [k for k, rule in enumerate(rules) if rules_fire([rule], on)]
            cex.append(f"{v['id']}: rules {fired[:5]} allow a call where {' & '.join(v['bits'])} holds, e.g. facts "
                       f"{sorted(n for n in on if not n.startswith('query'))[:14]}")
    return {"passed": all(s == "infeasible" for s in per.values()), "per_premise": per, "counterexamples": cex}


def nontrivial_check(rules: list[list[str]], names: list[str], dom, on_rows, time_limit: float) -> dict:
    """deny_all: no rules. allows_state_changing: is there a domain vector with kind=read = 0 on which a rule fires
    (MILP, the same encoding as solver_check)? Also the benign state-changing training calls the rules allow."""
    benign_sc = [on for on, lab in on_rows if lab == "ALLOW" and "kind=read" not in on]
    out = {"deny_all": not rules, "benign_state_changing_rows": len(benign_sc),
           "benign_state_changing_allowed": sum(rules_fire(rules, on) for on in benign_sc) if rules else 0}
    if not rules:
        out.update({"allows_state_changing": False, "solver_status": "no rules"})
        return out
    tm = {"clauses": [rule_literals(r, names) for r in rules], "W": np.array([[0, 1]] * len(rules), np.int64),
          "F": len(names)}
    col = {n: i for i, n in enumerate(names)}
    r = VP.build_and_solve(tm, dom, {col["kind=read"]: 0}, 1, [0], time_limit, threads=2)
    out.update({"allows_state_changing": r["status"] == "feasible", "solver_status": r["status"]})
    if r["status"] == "feasible":
        on = {names[i] for i in np.flatnonzero(r["x"])}
        out["example_kind"] = sorted(n for n in on if n.startswith("kind="))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", required=True, choices=["scripted", "qwen"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="qwen3.5-9b-q8_0", help="the model id the server lists (its --alias)")
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--qbits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--folds", nargs="*", default=SUITES)
    ap.add_argument("--store", default=str(E2 / "runs" / "sealed" / "replay_store.jsonl"),
                    help="the campaign's replay store (qwen proposer only)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(args, out)
    except StageStopped as e:
        rec = {**e.record(), "script": "build_d6.py", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "store": store_record(args.store) if args.proposer == "qwen" else None}
        (out / "STOPPED.json").write_text(json.dumps(rec, indent=1))
        print(f"STOPPED: {e}", flush=True)
        sys.exit(EXIT_TRANSPORT)


def propose_round(proposer, msgs: list[dict], where: dict) -> tuple[str | None, str | None]:
    """(text, None) or (None, the error) for a request error that is not a transport error (caught per round);
    a transport error stops the stage (StageStopped)."""
    try:
        return proposer.propose(msgs), None
    except Exception as e:
        if is_transport(e):
            raise StageStopped(where, e) from e
        return None, f"{error_type(e)}: {e}"[:300]


def run(args, out: Path) -> None:
    table = ToolTable()
    rows = [json.loads(l) for p in args.rows for l in Path(p).read_text().splitlines()]
    rows = [r for r in rows if r.get("label") in ("ALLOW", "DENY") and r["facts"]["kind"] is not None]
    for held in args.folds:
        assert not (out / f"d6_rules_{held}.json").exists(), f"d6_rules_{held}.json exists"
        qb = QueryBits(Path(args.qbits) / f"fold_{held}.json")
        names = FACT_BITS + qb.names
        dom = domain_rows(names, table)
        train = [r for r in rows if r["suite"] != held]
        on_rows = [(set(r["facts_on"]) | qb.on(r["query_sha256"]), r["label"]) for r in train]
        proposer = ScriptedProposer() if args.proposer == "scripted" else \
            QwenProposer(args.base_url, model=args.model, store=ReplayStore(args.store),
                         context={"stage": "E2-build", "fold": held})
        rounds, prev, feedback, best = [], None, None, None
        for k in range(1, args.rounds + 1):
            msgs = build_messages(table, train, qb.names, feedback, prev)
            text, req_err = propose_round(proposer, msgs, {"fold": held, "round": k, "what": "d6 rule writer"})
            if req_err is not None:
                rules, problems = [], [f"the request failed ({req_err}); this round has no rule set"]
            else:
                rules, problems = parse_rules(text, names)
            chk = solver_check(rules, names, dom, args.time_limit)
            allowed_benign = sum(rules_fire(rules, on) for on, lab in on_rows if lab == "ALLOW")
            blocked_deny = sum(not rules_fire(rules, on) for on, lab in on_rows if lab == "DENY")
            n_b = sum(lab == "ALLOW" for _, lab in on_rows)
            n_d = len(on_rows) - n_b
            blocked_examples = [sorted(n for n in on if not n.startswith("query"))[:12]
                                for on, lab in on_rows if lab == "ALLOW" and not rules_fire(rules, on)][:5]
            rec = {"round": k, "n_rules": len(rules), "parse_problems": problems, "solver": chk,
                   "benign_allowed": f"{allowed_benign}/{n_b}", "deny_blocked": f"{blocked_deny}/{n_d}", "rules": rules}
            rounds.append(rec)
            print(f"[{held}] round {k}: rules {len(rules)} passed {chk['passed']} benign {allowed_benign}/{n_b} "
                  f"deny {blocked_deny}/{n_d} problems {len(problems)}", flush=True)
            if chk["passed"] and rules:
                cand = (allowed_benign, blocked_deny, -k)
                if best is None or cand > best[0]:
                    best = (cand, rules, chk, k)
            fb = []
            fb += problems
            fb += chk.get("counterexamples", [])
            if blocked_examples:
                fb.append(f"benign training calls your rules still block ({n_b - allowed_benign} of {n_b}), e.g.: "
                          + "; ".join(" & ".join(e) for e in blocked_examples))
            feedback, prev = "\n".join(fb), rules
            if chk["passed"] and allowed_benign == n_b:
                break
        rec_out = {"type": "rules", "names": names, "held_out": held, "proposer": proposer.name,
                   "stand_in": args.proposer == "scripted", "rounds": rounds,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if best is None:
            rec_out.update({"rules": [], "solver_check": {"passed": True, "note": "no rule set passed; deny all"},
                            "chosen_round": None})
        else:
            rec_out.update({"rules": best[1], "solver_check": best[2], "chosen_round": best[3]})
        rec_out["nontrivial"] = nontrivial_check(rec_out["rules"], names, dom, on_rows, args.time_limit)
        if args.proposer == "qwen":
            rec_out["llm_calls"] = proposer.calls
        (out / f"d6_rules_{held}.json").write_text(json.dumps(rec_out, indent=1))
        print(f"[{held}] chosen round {rec_out['chosen_round']} nontrivial {json.dumps(rec_out['nontrivial'])}", flush=True)


if __name__ == "__main__":
    main()
