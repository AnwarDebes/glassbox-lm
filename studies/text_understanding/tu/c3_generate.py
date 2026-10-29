#!/usr/bin/env python
"""Step 3c, the generator client (tu/PLAN_3c.md "Generator"): training texts -> a local llama-server -> paraphrases.

    python -I c3_generate.py --port 18081 --out tu/c3/gen.jsonl --workers 8              # the GPU job: all training texts
    python -I c3_generate.py --port 18082 --out tu/c3/pilot/gen_pilot.jsonl --pilot 20   # the CPU pilot

Reads ONLY dl.data.load(name).texts["train"] and labels["train"] of CLINC150, Banking77 and HWU64 (Banking77's carved
val is not in "train"); no val, test or out-of-scope text is sent. Server: 127.0.0.1 only, /v1/chat/completions, the
prompt and decoding of c3_common.py, thinking off, a fixed seed per request. --pilot N takes the first N training
indices of each set in a numpy default_rng(20261023) permutation; otherwise every training index in order. One JSON
row per source text, appended as it completes (restart-safe: a (set, index) already in the file is skipped; the file
is never rewritten). Prints a progress line every 500 rows (tokens per second over all slots); with --max-hours
it stops after 1,000 rows when the projected total is longer (the plan's 4-hour rule).
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, f"{_REPO_ROOT}/studies/decision_layer/src")
import c3_common as C  # noqa: E402
from dl.data import load  # noqa: E402


def ask(port: int, b: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--sets", default=",".join(C.SETS))
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop after 1,000 rows if the projected total is over this")
    args = ap.parse_args()
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["set"], r["index"]))
    jobs = []
    for name in args.sets.split(","):
        ds = load(name)
        tr, ytr = ds.texts["train"], ds.labels["train"]
        idx = np.random.default_rng(20261023).permutation(len(tr))[: args.pilot] if args.pilot else np.arange(len(tr))
        jobs += [(name, int(i), tr[i], ds.intents[int(ytr[i])]) for i in idx if (name, int(i)) not in done]
    print(json.dumps({"to_do": len(jobs), "already_done": len(done), "workers": args.workers}), flush=True)
    lock = threading.Lock()
    t0, n_done, n_tok = time.time(), 0, 0

    def one(job):
        name, i, text, intent = job
        t1 = time.time()
        r = ask(args.port, C.body(name, i, text, intent))
        raw = r["choices"][0]["message"].get("content") or ""
        lines, why = C.parse(raw, text)
        return {"set": name, "index": i, "label": intent, "text": text, "raw": raw, "paraphrases": lines, "parse": why,
                "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage"), "seconds": round(time.time() - t1, 2)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(args.out, "a") as f:
        futs = [ex.submit(one, j) for j in jobs]
        for fu in as_completed(futs):
            row = fu.result()
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
                n_tok += (row.get("usage") or {}).get("completion_tokens", 0)
                if n_done % 500 == 0 or n_done == len(jobs):
                    el = time.time() - t0
                    proj = el / n_done * len(jobs) / 3600
                    print(json.dumps({"done": n_done, "of": len(jobs), "seconds": round(el, 1),
                                      "new_tokens_per_s": round(n_tok / max(el, 1e-9), 1),
                                      "projected_hours": round(proj, 2)}), flush=True)
                    if args.max_hours and n_done >= 1000 and proj > args.max_hours:
                        print(json.dumps({"STOP": f"projected {proj:.2f} h > {args.max_hours} h"}), flush=True)
                        for g in futs:
                            g.cancel()
                        break


if __name__ == "__main__":
    main()
