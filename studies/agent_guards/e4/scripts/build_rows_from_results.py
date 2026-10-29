"""Write rows for the learned part from tau2 results.json files (our A0/A1 runs, after the GPU stage). CPU, no LLM.

Row = one EXECUTED write of a trajectory (it reached the environment; denied calls never do), with the facts the guard
saw (facts.py, calls in execution order), labelled DENY (1) iff it is not one of the task's gold actions with matching
arguments (lists as multisets), from the task definitions embedded in the results file (v1.0.1 for our runs). Only
tasks of --split are kept (registered training: the 74 train tasks).

Revision 2 (review issue 8, decision 6): the learned part decides only writes on which no veto clause fires, so it is
trained, cross-validated and gated on exactly those rows:
  - VETO-CLEAN: a write on which a veto clause fires (bits.veto_fires) is left out of the training rows (counted);
  - DEDUPLICATED across A0 and A1: the files are read in the order given (A0 first, then A1); a row whose
    (task, tool, canonical arguments, bit vector) equals a row already kept is left out (counted). A1 replays A0
    until its first denial, so without this every task with no denial would count twice.
  - the 10-row minimum of each class (train_guard.py --min-class) is counted on these kept rows.
--all-out writes EVERY executed write of the split with its flags (veto_clean, duplicate), for the out-of-fold go-bar
on A1's executed train-task writes (scripts/gobar.py) and for the reports.
--results takes results.json files or arm folders (an arm folder's _rerun folder replaces its tasks).
Run: offline_env.sh .../taubench/.venv/bin/python -I -B build_rows_from_results.py --results runs/<c>/A0_base \
       runs/<c>/A1_base --sources qwen_A0 qwen_A1 --split train --split-file <split_tasks.json> --out rows.json \
       [--all-out all.json]
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))

from e4guard import bits as B  # noqa: E402
from e4guard import rules  # noqa: E402
from e4guard.facts import WRITE_TOOLS, canon_args, events_from_message_dicts, replay_facts  # noqa: E402


def norm(v):
    return sorted(str(x) for x in v) if isinstance(v, list) else str(v)


def gold_match(name, args, gold):
    for a in gold:
        ga = a.get("arguments") or {}
        if a.get("name") == name and set(ga) == set(args or {}) and all(norm(ga[k]) == norm(args[k]) for k in ga):
            return True
    return False


def load_results(path: Path) -> tuple[dict, dict]:
    """A results.json file, or an arm folder (its results.json merged with its _rerun folder, e4guard/runs.py)."""
    from e4guard.runs import load_arm

    if path.is_dir():
        arm = load_arm(path)
        files = [path / "results.json"] + ([Path(str(path) + "_rerun") / "results.json"] if arm["rerun_tasks"] else [])
        res = {"tasks": list(arm["tasks"].values()), "simulations": list(arm["sims"].values())}
    else:
        files = [path]
        res = json.loads(path.read_text())
    return res, {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def write_rows(results: dict, source: str, split: str, split_of: dict) -> list[dict]:
    """Every executed write of every simulation of `split` in one tau2 results dict, with its flags."""
    tasks = {t["id"]: t for t in results["tasks"]}
    out = []
    for sim in results["simulations"]:
        if split != "base" and split_of.get(sim["task_id"]) != split:
            continue
        gold = ((tasks.get(sim["task_id"]) or {}).get("evaluation_criteria") or {}).get("actions") or []
        for _, e, f, res in replay_facts(events_from_message_dicts(sim["messages"])):
            if e["name"] not in WRITE_TOOLS:
                continue
            x = B.encode(f)
            veto = B.veto_fires(x)
            out.append({"source": source, "task": sim["task_id"], "split": split_of.get(sim["task_id"]),
                        "trial": sim.get("trial"), "tool": e["name"],
                        "args": json.dumps(canon_args(e.get("args") or {}), sort_keys=True, default=str),
                        "label": 0 if gold_match(e["name"], e.get("args"), gold) else 1, "x": x.tolist(),
                        "code_rules": rules.rule_ids(f), "veto_clauses": veto, "veto_clean": not veto,
                        "strict_r3a": rules.strict_r3a_fires(f), "tool_error": bool((res or {}).get("error"))})
    return out


def select_training_rows(all_rows: list[dict]) -> tuple[list[dict], Counter]:
    """Veto-clean, deduplicated rows in input order (the first copy is kept). Marks every row's `duplicate` flag."""
    seen, kept, c = set(), [], Counter()
    for r in all_rows:
        key = (str(r["task"]), r["tool"], r["args"], "".join(map(str, r["x"])))
        r["duplicate"] = key in seen
        c[f"{r['source']}:executed_writes"] += 1
        if not r["veto_clean"]:
            c[f"{r['source']}:vetoed"] += 1
            continue
        if r["duplicate"]:
            c[f"{r['source']}:duplicate"] += 1
            continue
        seen.add(key)
        kept.append(r)
        c[f"{r['source']}:kept_{'ALLOW' if r['label'] == 0 else 'DENY'}"] += 1
    c["kept"] = len(kept)
    c["kept_ALLOW"] = sum(1 for r in kept if r["label"] == 0)
    c["kept_DENY"] = sum(1 for r in kept if r["label"] == 1)
    return kept, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test", "base"])
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-out", default=None)
    ap.add_argument("--stand-in", action="store_true")
    a = ap.parse_args()
    assert len(a.results) == len(a.sources)
    sp = json.loads(Path(a.split_file).read_text())
    split_of = {t: s for s in ("train", "test") for t in sp[s]}
    all_rows, inputs = [], {}
    for path, src in zip(a.results, a.sources):
        res, hashes = load_results(Path(path))
        inputs.update(hashes)
        all_rows += write_rows(res, src, a.split, split_of)
    kept, c = select_training_rows(all_rows)
    meta = {"names": B.NAMES, "stand_in": bool(a.stand_in), "inputs_sha256": inputs, "split": a.split,
            "rule": "veto-clean executed writes, deduplicated by (task, tool, canonical args, bits); first copy kept",
            "counts": dict(c)}
    Path(a.out).write_text(json.dumps({**meta, "rows": kept}))
    if a.all_out:
        Path(a.all_out).write_text(json.dumps({**meta, "rows": all_rows}))
    print(json.dumps({"rows_kept": len(kept), "executed_writes": len(all_rows), "counts": dict(c)}, indent=1))


if __name__ == "__main__":
    main()
