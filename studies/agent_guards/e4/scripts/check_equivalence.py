"""E4.4, part 1: A1's code gates and A2's veto clauses are the same function (exhaustive, no sampling).

Enumerates every bit vector of the rule-relevant bits: 17 tool values x 10 status values x all 2^13 combinations of
the 13 rule-fact bits (1,392,640 vectors), with the tool-class bits set from the tool (as facts.py computes them).
That is a SUPERSET of the domain (the domain rows that tie rule-fact bits to each other are not imposed), and the
learned-only bits are not read by either side. For every vector it compares rules.rule_ids (the plain code, run
on the decoded fact record) with bits.veto_rules (the clauses). Any difference is printed and the exit code is 1.

Run: .../taubench/.venv/bin/python -I -B check_equivalence.py --out <json>
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, f"{_REPO_ROOT}/studies/agent_guards/e4")

import numpy as np  # noqa: E402

from e4guard import bits as B  # noqa: E402
from e4guard import facts as F  # noqa: E402
from e4guard import rules  # noqa: E402


def class_bits(tool):
    return {"is_write": tool in F.WRITE_TOOLS, "is_order_write": tool in F.ORDER_WRITE_TOOLS,
            "needs_auth": tool not in F.NO_AUTH_NEEDED, "needs_pending": tool in F.NEEDS_PENDING,
            "needs_delivered": tool in F.NEEDS_DELIVERED, "is_item_change": tool in F.ITEM_CHANGE_TOOLS,
            "uses_payment": tool in F.PAYMENT_TOOLS, "needs_oid_confirm": tool in F.NEEDS_OID_CONFIRM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    tools = F.RETAIL_TOOLS + ["__unknown__"]
    n = n_bad = n_deny = 0
    bad, by_rule = [], Counter()
    for tool, status in itertools.product(tools, B.STATUS_VALUES):
        x0 = np.zeros(B.N_BITS, np.uint8)
        x0[B.IDX[f"tool={tool}"]] = 1
        x0[B.IDX[f"status={status}"]] = 1
        for b, val in class_bits(tool).items():
            x0[B.IDX[b]] = int(val)
        for m in range(1 << len(B.RULE_FACT_BITS)):
            x = x0.copy()
            for j, b in enumerate(B.RULE_FACT_BITS):
                x[B.IDX[b]] = (m >> j) & 1
            f = B.decode_rule_facts(x)
            code = rules.rule_ids(f)
            veto = B.veto_rules(x)
            n += 1
            n_deny += bool(code)
            for r in code:
                by_rule[r] += 1
            if code != veto:
                n_bad += 1
                if len(bad) < 20:
                    bad.append({"tool": tool, "status": status, "bits": {b: int(x[B.IDX[b]]) for b in B.RULE_FACT_BITS},
                                "code": code, "veto": veto})
    out = {"vectors": n, "mismatches": n_bad, "code_denies": n_deny, "per_rule_vectors": dict(by_rule),
           "examples_of_mismatch": bad, "n_veto_clauses": len(B.VETO), "seconds": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "examples_of_mismatch"}, indent=1))
    if bad:
        print("MISMATCH examples:", json.dumps(bad[:5], indent=1))
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
