"""PLAN_logsize_dev.md: exit 0 only if the gate passed (logsize_agent.py on ABCD's whole split reproduced
runs/agent_abcd_c4_tm_s1's dev accuracy exactly); every queued AGENT(N) line runs this first. Standard library only.

usage: python3.12 -I logsize_gate.py [--result runs/logsize/agent_abcd_whole/result.json]
"""
import argparse
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))


def passed(path):
    if not os.path.exists(path):
        return False, "gate not run"
    chk = json.load(open(path)).get("dev_check") or {}
    return bool(chk.get("equal")), f"rerun {chk.get('got')} vs expected {chk.get('expected')}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=os.path.join(H, "runs/logsize/agent_abcd_whole/result.json"))
    a = ap.parse_args(argv)
    ok, why = passed(a.result)
    print(f"logsize gate {'passed' if ok else 'NOT passed'}: {why}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
