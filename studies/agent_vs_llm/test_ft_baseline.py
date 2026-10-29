"""Checks of ft_baseline.py without a GPU: the text rendering and the loader (which must refuse the test split).

usage: studies/decision_layer/.venv-dl/bin/python -I test_ft_baseline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_baseline as F  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


r = {"prev": [f"a{i}" for i in range(15)], "recent": "user: hi", "cust": "x" * 900 + "END"}
t = F.render(r)
check("keeps the last 10 actions in order", "actions: a5 ; a6" in t and "a14 | recent" in t and "a4 " not in t)
check("recent comes before the customer text", t.index("recent:") < t.index("customer:"))
check("customer text keeps its tail", t.endswith("END") and len(t.split("customer: ")[1]) == F.CUST_CHARS)
check("no previous action renders as none", F.render({"prev": [], "recent": "", "cust": ""}).startswith("actions: none |"))
try:
    F.load("star", "test")
    check("the test split is refused", False)
except ValueError:
    check("the test split is refused", True)
x, y, n, h = F.load("abcd", "dev", limit=5)
check("loader gives texts, labels, label count and a hash", len(x) == 5 and len(y) == 5 and n == 30 and len(h) == 64)
check("labels are in range", all(0 <= v < n for v in y))
x2, y2, n2, _ = F.load("star", "train", limit=3)
check("STAR has 187 labels", n2 == 187 and len(x2) == 3)
print("FAILED" if fails else "PASSED", f"({fails} failed)")
sys.exit(1 if fails else 0)
