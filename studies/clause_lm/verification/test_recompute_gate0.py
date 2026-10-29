#!/usr/bin/env python
"""Tests of recompute_gate0.py on toy inputs with known answers (numpy only, no study file read).

Run: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \
  studies/decision_layer/.venv-dl/bin/python -I test_recompute_gate0.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_gate0 as r  # noqa: E402

fails = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# 1. thresholds: constant -> min + 1; two-valued -> midpoint; otherwise the median
Xtr = np.array([[0.5, 0.0, 1.0], [0.5, 2.0, 2.0], [0.5, 0.0, 3.0], [0.5, 2.0, 10.0]], dtype=np.float32)
th, kinds = r.thresholds(Xtr)
check("constant column threshold = min + 1", th[0] == np.float32(1.5))
check("two-valued column threshold = midpoint", th[1] == np.float32(1.0))
check("other column threshold = median (2.5)", th[2] == np.float32(2.5))
check("kinds counted", kinds == {"constant": 1, "two_valued": 1, "median": 1})
B = r.booleanize(np.array([[0.5, 1.0, 2.5], [9.0, 0.9, 2.4]], dtype=np.float32), th)
check("bit = value >= threshold (equality fires)", B.tolist() == [[0, 1, 1], [1, 0, 0]])

# 2. clause firing: f = 2 bits, literals [b0, b1, not b0, not b1]; n_states 10 (include iff ta > 10)
n_states = 10
ta = np.array([[11, 0, 0, 0],    # clause 0: b0
               [0, 0, 0, 11],    # clause 1: not b1
               [0, 0, 0, 0],     # clause 2: empty, never fires
               [11, 11, 0, 0],   # clause 3: b0 and b1
               [10, 0, 0, 0]],   # clause 4: ta == n_states is NOT included -> empty
              dtype=np.int32)
bits = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.uint8)
fire = r.clause_fire(ta, n_states, bits)
check("positive literal fires iff bit 1", fire[:, 0].tolist() == [True, True, False])
check("negated literal fires iff bit 0", fire[:, 1].tolist() == [True, False, True])
check("empty clause never fires", not fire[:, 2].any())
check("conjunction", fire[:, 3].tolist() == [False, True, False])
check("ta == n_states is not an include", not fire[:, 4].any())
sup = np.array([True, False, False, False, False])
check("suppressed clause never fires", not r.clause_fire(ta, n_states, bits, sup)[:, 0].any())
try:
    r.clause_fire(ta, n_states, np.zeros((1, 3), dtype=np.uint8))
    check("shape mismatch refused", False)
except ValueError:
    check("shape mismatch refused", True)

# 3. prediction: summed weights of firing clauses, first argmax on ties
weight = np.array([[2, 0, 0], [0, 2, 0], [5, 5, 5], [0, 0, 9], [7, 7, 7]], dtype=np.int32)
pred, ties = r.predict(ta, weight, n_states, bits)
# sample 0 fires clauses 0,1 -> sums [2,2,0] -> tie -> class 0; sample 1 fires 0,3 -> [2,0,9] -> 2;
# sample 2 fires 1 -> [0,2,0] -> 1
check("predictions", pred.tolist() == [0, 2, 1])
check("top ties counted", ties == 1)

# 4. literal statistics
st = r.literal_stats(ta, weight, n_states)
check("clauses with a positive literal", st["clauses_with_positive_literal"] == 2)
check("clauses with a negated literal", st["clauses_with_negated_literal"] == 1)
check("non-empty clauses", st["non_empty_clauses"] == 3)
check("median literals", st["literals_median"] == 1.0)
check("max literals", st["literals_max"] == 2)
check("weight zero share", abs(st["weight_zero_share"] - 6 / 15) < 1e-12)

# 5. signatures at the boundaries
deg = {"clauses_with_positive_literal": 400, "literals_median": 100.0}
hea = {"clauses_with_positive_literal": 2000, "literals_median": 10.0}
check("degenerate at 0.70 / 400 / 100", r.signature(0.70, deg) == "degenerate")
check("not degenerate at 0.7001", r.signature(0.7001, deg) == "neither")
check("healthy at 0.78 / 2000 / 10", r.signature(0.78, hea) == "healthy")
check("structurally healthy but 0.775 is neither", r.signature(0.775, hea) == "neither")
check("healthy needs the structure", r.signature(0.80, {"clauses_with_positive_literal": 1999,
                                                         "literals_median": 1.0}) == "neither")


# 6. verdicts and every outcome row
def runs(a0_acc, a0_pos, sigs):
    out = {"torch_seed42": {"val_acc": a0_acc, "stats": {"clauses_with_positive_literal": a0_pos}}}
    for k, s in zip(("torch_noneg_seed42", "torch_noneg_seed123", "torch_noneg_seed456"), sigs):
        out[k] = {"signature": s}
    return out


v = r.verdicts(runs(0.81, 2954, ["degenerate"] * 3), "NOT RUN")
check("driver MET within 0.01 (0.81)", v["driver_check"] == "MET")
check("row 5: A CONFIRMED, B NOT RUN", v["outcome_row"] == 5 and v["arm_A"] == "CONFIRMED")
check("row 1: driver NOT MET (0.81 but 1999 clauses)",
      r.verdicts(runs(0.81, 1999, ["degenerate"] * 3), "NOT RUN")["outcome_row"] == 1)
check("row 1: driver NOT MET (0.8086 is more than 0.01 away)",
      r.verdicts(runs(0.8086, 2954, ["degenerate"] * 3), "NOT RUN")["outcome_row"] == 1)
check("row 1: driver MISSING", r.verdicts({}, "NOT RUN")["driver_check"] == "MISSING")
check("row 2: A MIXED", r.verdicts(runs(0.8187, 2954, ["neither"] * 3), "NOT RUN")["outcome_row"] == 2)
check("row 2: A MISSING", r.verdicts(runs(0.8187, 2954, ["degenerate"] * 2), "NOT RUN")["arm_A"] == "MISSING")
check("row 2: B MIXED", r.verdicts(runs(0.8187, 2954, ["degenerate"] * 3), "MIXED")["outcome_row"] == 2)
check("row 3: A REFUTED", r.verdicts(runs(0.8187, 2954, ["healthy"] * 3), "NOT RUN")["outcome_row"] == 3)
check("row 3: B REFUTED", r.verdicts(runs(0.8187, 2954, ["degenerate"] * 3), "REFUTED")["outcome_row"] == 3)
check("row 4: both CONFIRMED", r.verdicts(runs(0.8187, 2954, ["degenerate"] * 3), "CONFIRMED")["outcome_row"] == 4)

print(f"{'FAILED' if fails else 'PASSED'}: {len(fails)} failure(s)")
sys.exit(1 if fails else 0)
