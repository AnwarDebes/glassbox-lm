"""Bit vectors of EVERY executed call (reads and writes) of tau2 results files, for the MILP domain pre-check
("every real input satisfies the domain rows"). No labels, no LLM. --results takes results.json files or arm folders.
This is a declared read of every A0/A1 call, test tasks included (bits only; no reward, no label): registration
revision 2, section "What I knew".
Run: offline_env.sh .../taubench/.venv/bin/python -I -B dump_call_bits.py --results a.json [b.json ...] --out bits.json
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

from e4guard import bits as B  # noqa: E402
from e4guard.facts import events_from_message_dicts, replay_facts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from build_rows_from_results import load_results

    rows, c = [], Counter()
    for p in a.results:
        d, _ = load_results(Path(p))
        for sim in d["simulations"]:
            for _, e, f, _ in replay_facts(events_from_message_dicts(sim["messages"])):
                rows.append({"source": Path(p).name, "task": sim["task_id"], "tool": e["name"], "x": B.encode(f).tolist()})
                c[Path(p).name] += 1
    Path(a.out).write_text(json.dumps({"names": B.NAMES, "rows": rows}))
    print(json.dumps({"calls": len(rows), "by_file": dict(c)}, indent=1))


if __name__ == "__main__":
    main()
