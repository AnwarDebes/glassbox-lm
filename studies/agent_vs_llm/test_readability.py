"""Tests of readability_dev.receipts on a tiny hand-made model (no data files).

run: cd studies/agent_vs_llm && CUDA_VISIBLE_DEVICES= ../decision_layer/.venv-dl/bin/python -I \
         test_readability.py
Model: 4 features with negation (8 literals), 4 clauses, 3 classes.
  clause A: x0          w [5, 0, 0]
  clause B: x1 AND NOT x2  w [0, 3, 0]
  clause C: x3          w [1, 2, 0]
  clause D: x1 AND x3   w [0, 0, 4]
Row 1 = [1,1,0,0]: A and B fire; S = [5,3,0]; c 0, r 1; d(A) 5 > 0, d(B) -3: receipt [A].
Row 2 = [0,1,0,1]: B, C, D fire; S = [1,5,4]; c 1, r 2; d(B) 3, d(C) 2, d(D) -4: ranked B, C; prefix B alone gives
  [0,3,0] -> class 1: receipt [B].
Row 3 = [1,0,0,1]: A, C fire; S = [6,2,0]; c 0, r 1; d(A) 5, d(C) -1: receipt [A].
Row 4 (size 2): a second model (3 classes) where the strongest clause alone points elsewhere:
  E: x0 w [2, 0, 3]; F: x1 w [2, 0, 0]; G: x2 w [0, 3, 0]; row [1,1,1,0]: S = [4,3,3]; c 0, r 1 (lowest index);
  d(E) 2, d(F) 2, d(G) -3; ranked E, F (tie: lower index); E alone gives [2,0,3] -> 2; E+F gives [4,0,3] -> 0: [E, F].
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import readability_dev as RD  # noqa: E402
from dl import cpu_tm_fast  # noqa: E402

OUT = []


def check(name, ok):
    OUT.append(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(OUT[-1])


def model(includes, w, f=4):
    ta = np.full((len(includes), 2 * f), 100, dtype=np.int32)
    for k, lits in enumerate(includes):
        for j in lits:
            ta[k, j] = 300
    m = cpu_tm_fast.pack(ta, np.array(w))
    m["counts"] = np.diff(np.concatenate([m["starts"], [len(m["lits"])]]))
    return m


m = model([[0], [1, 4 + 2], [3], [1, 3]], [[5, 0, 0], [0, 3, 0], [1, 2, 0], [0, 0, 4]])
X = np.array([[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=np.uint8)
R = RD.receipts(m, X)
check("row 1: decision 0, runner-up 1, 2 fired, receipt [A]",
      (R[0]["c"], R[0]["r"], R[0]["n_fired"], list(R[0]["receipt"])) == (0, 1, 2, [0]))
check("row 2: decision 1, runner-up 2, 3 fired, receipt [B]",
      (R[1]["c"], R[1]["r"], R[1]["n_fired"], list(R[1]["receipt"])) == (1, 2, 3, [1]))
check("row 3: decision 0, receipt [A]", (R[2]["c"], list(R[2]["receipt"])) == (0, [0]))
check("clause words: B is 'x1 AND NOT x2'",
      RD.clause_words(m, 1, ["x0", "x1", "x2", "x3"]) == ["x1", "NOT x2"])
check("CPU class sums agree with the decisions", list(cpu_tm_fast.class_sums(m, X).argmax(1)) == [0, 1, 0])
m2 = model([[0], [1], [2]], [[2, 0, 3], [2, 0, 0], [0, 3, 0]])
R2 = RD.receipts(m2, np.array([[1, 1, 1, 0]], dtype=np.uint8))
check("row 4: decision 0, runner-up 1, receipt [E, F] (size 2)",
      (R2[0]["c"], R2[0]["r"], list(R2[0]["receipt"])) == (0, 1, [0, 1]))
R3 = RD.receipts(m2, np.array([[1, 1, 1, 0]], dtype=np.uint8), max_k=1)
check("with at most 1 clause, row 4 has no receipt", R3[0]["receipt"] is None)
n_fail = sum(x.startswith("FAIL") for x in OUT)
print(f"{len(OUT) - n_fail} passed, {n_fail} failed")
print("PASSED" if n_fail == 0 else "FAILED")
sys.exit(1 if n_fail else 0)
