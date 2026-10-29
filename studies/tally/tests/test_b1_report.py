"""B1 report: censored runs cannot decide the dual-state line; registered runs need fixed caps.

Run: PYTHONPATH=src python tests/test_b1_report.py

Synthetic run tables (no B1 run is read or started) for the dual-state trial at M = 128:
- dual and marginal all censored at the cap: the reduction is 0 by construction, which must not
  read as "kill" (no claim);
- marginal censored, dual not: the reduction is biased down, so "kill" is blocked;
- dual partly censored with a large reduction: biased up, so "success" is blocked;
- dual censored, marginal not, reduction far below 0: "kill" stands (the bias cannot create it);
- no censoring: success and kill as before.
Second part: tally.b1.run refuses a registered run (--label reg) while CAPS_FROM, the run id of
the timing run that fixed the caps, is not set; nothing is trained and no run folder is written.
"""
import os
import sys
import tempfile

import numpy as np

from tally.b1 import report

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


CAP = 2_000_000


def table(dual, marginal, excess=0.3):
    """runs dict for M = 128: dual and marginal samples per seed (CAP means censored)."""
    runs = {}
    for arm, xs in (("dual", dual), ("marginal", marginal)):
        for s, x in zip(report.SEEDS, xs):
            runs[(arm, report.PRIMARY_M, s)] = (f"/none/{arm}_{s}", {
                "samples_to_target": int(x), "censored": bool(x >= CAP), "final_excess": excess})
    return runs


def verdict(dual, marginal):
    return report.analyse(table(dual, marginal), "test")["verdict"]["dual"]


def main():
    n = len(report.SEEDS)
    v = verdict([CAP] * n, [CAP] * n)
    check("dual and marginal all censored: no claim (not kill)", v["verdict"] == "no claim", v)
    v = verdict([1_950_000] * n, [CAP] * n)
    check("marginal censored: kill blocked", v["verdict"] == "no claim", v)
    v = verdict([200_000] * (n - 2) + [CAP] * 2, [1_000_000] * n)
    check("dual partly censored, reduction 0.44 on paper: success blocked", v["verdict"] == "no claim", v)
    v = verdict([CAP] * n, [1_000_000] * n)
    check("dual censored, marginal not, reduction -1: kill stands", v["verdict"] == "kill", v)
    v = verdict([500_000] * n, [1_000_000] * n)
    check("no censoring, reduction 0.5: success", v["verdict"] == "success", v)
    v = verdict([980_000] * n, [1_000_000] * n)
    check("no censoring, reduction 0.02: kill", v["verdict"] == "kill", v)
    v = verdict([CAP] * n, [CAP] * n)
    check("the report counts censored runs of both arms",
          v.get("censored_dual") == n and v.get("censored_marginal") == n, v)

    from tally.b1 import run as b1run
    if not hasattr(b1run, "CAPS_FROM"):
        check("run.py names the timing run that fixed the caps (CAPS_FROM)", False,
              "missing; a registered run would start with caps that no timing run fixed")
    else:
        from tally import runlog
        with tempfile.TemporaryDirectory() as tmp:
            runlog.RESULTS = tmp
            saved = b1run.CAPS_FROM
            b1run.CAPS_FROM = None
            sys.argv = ["run", "--arm", "marginal", "--M", "8", "--seed", "0", "--label", "reg",
                        "--registration", os.path.join(tmp, "B1.md")]
            try:
                b1run.main()
                stopped, msg = False, ""
            except SystemExit as e:
                stopped, msg = True, str(e.code)
            b1run.CAPS_FROM = saved
            check("registered run refused while CAPS_FROM is not set, before any run folder",
                  stopped and "CAPS_FROM" in msg and not os.listdir(tmp), f"{msg!r} {os.listdir(tmp)}")
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
