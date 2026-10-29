"""Rows from AgentDojo trace folders (TraceLogger JSON). AgentDojo venv:

    ../agentdojo/.venv/bin/python -I -B scripts/traces_to_rows.py --source standin_trace --out data/rows_standin.jsonl \
        DIR [DIR ...]

--source names where the traces come from: 'd0_trace' for Qwen's D0 runs (GPU stage E2-1), 'standin_trace' for the
offline scripted replays (code tests only; never a finding). Only traces of benchmark v1.2.2 user tasks are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2 / "src"))

from agentdojo.task_suite.load_suites import get_suites

from e2guard.facts import FactExtractor, ToolTable
from e2guard.trace_rows import load_trace, rows_from_trace, trace_files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", required=True, choices=["d0_trace", "standin_trace"])
    ap.add_argument("--version", default="v1.2.2")
    args = ap.parse_args()
    out = Path(args.out)
    assert not out.exists(), f"{out} exists"
    suites = get_suites(args.version)
    x = FactExtractor(ToolTable())
    t0 = time.time()
    tot: Counter = Counter()
    labels: Counter = Counter()
    files = [p for d in args.dirs for p in trace_files(Path(d))]
    with out.open("w") as fo:
        for p in files:
            tr = load_trace(p)
            if tr.get("benchmark_version") != args.version or tr.get("suite_name") not in suites:
                tot["skipped_version_or_suite"] += 1
                continue
            rows, st = rows_from_trace(tr, suites[tr["suite_name"]], x, args.source)
            if "skipped" in st:
                tot["skipped_not_user_task"] += 1
                continue
            tot["traces"] += 1
            for k, v in st.items():
                tot[k] += v
            for r in rows:
                labels[(r["suite"], r["label"])] += 1
                fo.write(json.dumps(r, default=str) + "\n")
    meta = {"source": args.source, "dirs": args.dirs, "files": len(files), "counts": dict(tot),
            "labels": {f"{s}:{l}": n for (s, l), n in sorted(labels.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))},
            "rows_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "seconds": round(time.time() - t0, 1),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
