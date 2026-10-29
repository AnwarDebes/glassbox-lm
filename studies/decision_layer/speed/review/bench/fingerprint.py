#!/usr/bin/env python
"""Result fingerprints for the no-loss speed benchmarks (read-only on every input; writes only --out).

A fingerprint splits a run's record into
  - "result":    every field that an optimisation must leave byte-identical (compared exactly, no tolerance);
  - "timing":    fields that are expected to change (train_s, wall times, history 't', solver seconds, clock stamps);
  - "resources": fields that may change and are reported, never compared as results (peak_gpu_mb).
and adds "result_sha256" = SHA-256 of the canonical JSON (sorted keys, no spaces) of "result".

Kinds:
  tm    one TM row of an abcd_explore.py results file (arm "tm", given tag and seed), plus the saved model file
        (ta, w, cfg, names, buttons: SHA-256 of the raw array bytes with dtype and shape). The tag itself is not a result
        field (benchmark reruns use a prefixed tag), every other row field is, except the timing / resource fields.
  milp  the MILP check rows of a verify_policy*.py results file (one tag): per (seed, guarded, check_id) every field
        except "seconds" and "time" (outcome, status, HiGHS status and message, nodes, sizes, counterexample x_on,
        decoded, nearest_dev, replay fields, solver settings). Rows without check_id (encoding, guardcost) are kept
        whole minus "time" / "seconds" under "other".

    python -I -B fingerprint.py tm   --rows R.jsonl --tag T --seed 1 [--model M.npz] [--tag-prefix sb_] --out F.json
    python -I -B fingerprint.py milp --rows R.jsonl --tag e72e [--guarded yes|no|all] [--seeds 1 2] --out F.json

Exit codes: 0 written; 2 input problem (no row, or several rows for one key whose result fields differ).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

TM_TIMING = ("train_s", "time")
TM_RESOURCES = ("peak_gpu_mb",)
TM_NOT_RESULT = ("tag",)
MILP_TIMING = ("seconds", "time")


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True)


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{i}: not JSON ({e})")
    return rows


def array_fp(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    if a.dtype.kind in "OU" or a.dtype.kind == "S":
        data = canon(a.tolist()).encode()
    else:
        data = a.tobytes()
    return {"dtype": str(a.dtype), "shape": list(a.shape), "sha256": hashlib.sha256(data).hexdigest()}


def model_fp(path: Path) -> dict:
    """SHA-256 of each array in the model file (raw bytes + dtype + shape); the zip container's own bytes (compression,
    member timestamps) are not compared."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: array_fp(z[k]) for k in sorted(z.files)}
    with zipfile.ZipFile(path) as zf:
        out["_members"] = sorted(zf.namelist())
    return out


def tm_fp(rows_path: Path, tag: str, seed: int, model: Path | None, tag_prefix: str) -> dict:
    want = tag_prefix + tag
    hits = [r for r in read_rows(rows_path) if r.get("tag") == want and r.get("seed") == seed and r.get("arm") == "tm"]
    if not hits:
        print(f"no tm row with tag {want!r} and seed {seed} in {rows_path}", file=sys.stderr)
        raise SystemExit(2)
    fps = []
    for r in hits:
        result, timing, resources = {}, {}, {}
        for k, v in r.items():
            if k in TM_NOT_RESULT:
                continue
            if k in TM_TIMING:
                timing[k] = v
            elif k in TM_RESOURCES:
                resources[k] = v
            elif k == "history":
                result["history"] = [{kk: vv for kk, vv in h.items() if kk != "t"} for h in v]
                timing["history_t"] = [h.get("t") for h in v]
            else:
                result[k] = v
        fps.append((result, timing, resources))
    results = {canon(f[0]) for f in fps}
    if len(results) > 1:
        print(f"{len(hits)} rows for tag {want!r} seed {seed} disagree on result fields", file=sys.stderr)
        raise SystemExit(2)
    result, timing, resources = fps[-1]
    out = {"kind": "tm", "key": {"tag": tag, "seed": seed, "tag_prefix": tag_prefix}, "n_rows_same_key": len(hits),
           "result": result, "timing": timing, "resources": resources,
           "source": {"rows": str(rows_path)}}
    if model is not None:
        out["result"]["model"] = model_fp(model)
        out["source"]["model"] = str(model)
        out["source"]["model_file_sha256"] = sha_file(model)
    out["result_sha256"] = sha_text(canon(out["result"]))
    return out


def milp_fp(rows_path: Path, tag: str, guarded: str, seeds: list[int] | None) -> dict:
    rows = [r for r in read_rows(rows_path) if r.get("tag") == tag]
    if not rows:
        print(f"no rows with tag {tag!r} in {rows_path}", file=sys.stderr)
        raise SystemExit(2)
    checks, other, timing = {}, [], {}
    for r in rows:
        if "check_id" not in r:
            other.append({k: v for k, v in r.items() if k not in MILP_TIMING})
            continue
        if guarded != "all" and bool(r.get("guarded")) != (guarded == "yes"):
            continue
        if seeds and r.get("seed") not in seeds:
            continue
        key = f"seed{r['seed']}|guarded{int(bool(r.get('guarded')))}|check{r['check_id']:03d}"
        res = {k: v for k, v in r.items() if k not in MILP_TIMING and k != "tag"}
        if key in checks and canon(checks[key]) != canon(res):
            print(f"two rows for {key} disagree", file=sys.stderr)
            raise SystemExit(2)
        checks[key] = res
        timing[key] = r.get("seconds")
    if not checks:
        print("no check rows selected", file=sys.stderr)
        raise SystemExit(2)
    outcomes: dict[str, int] = {}
    for c in checks.values():
        k = f"guarded{int(bool(c.get('guarded')))}:{c.get('outcome')}"
        outcomes[k] = outcomes.get(k, 0) + 1
    out = {"kind": "milp", "key": {"tag": tag, "guarded": guarded, "seeds": seeds},
           "result": {"checks": checks, "outcome_counts": outcomes},
           "timing": {"seconds_per_check": timing, "seconds_total": round(sum(v or 0 for v in timing.values()), 3),
                      "seconds_max": max((v or 0) for v in timing.values())},
           "resources": {}, "source": {"rows": str(rows_path)}, "other_rows": other}
    out["result_sha256"] = sha_text(canon(out["result"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    t = sub.add_parser("tm")
    t.add_argument("--rows", required=True, type=Path)
    t.add_argument("--tag", required=True)
    t.add_argument("--seed", required=True, type=int)
    t.add_argument("--model", type=Path)
    t.add_argument("--tag-prefix", default="")
    t.add_argument("--out", required=True, type=Path)
    m = sub.add_parser("milp")
    m.add_argument("--rows", required=True, type=Path)
    m.add_argument("--tag", required=True)
    m.add_argument("--guarded", default="all", choices=["yes", "no", "all"])
    m.add_argument("--seeds", type=int, nargs="*")
    m.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.kind == "tm":
        fp = tm_fp(a.rows, a.tag, a.seed, a.model, a.tag_prefix)
    else:
        fp = milp_fp(a.rows, a.tag, a.guarded, a.seeds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    tmp.write_text(json.dumps(fp, indent=1, sort_keys=True) + "\n")
    tmp.replace(a.out)
    print(f"{a.out} result_sha256 {fp['result_sha256']}")


if __name__ == "__main__":
    main()
