"""Smoke check of determinism (GPU via the shared llama-server; NOT RUN by the preparer against a real server): send stored
requests again, bypassing the replay store, and compare the replies (text and tool calls) with the stored ones.

DESIGN.md 1.7: the llama-server README warns that results are not guaranteed bit-identical across batch sizes, so the
smoke reports how often a repeated request gives a different reply. Revision 2 (review smoke item 4): at least
--n-agent 50 agent and --n-user 50 user requests, drawn by a seeded sample of the smoke store's rows of each role and
sent in one seeded SHUFFLED order (agent and user mixed, so no request follows its own predecessor). Reported: the
per-request difference rate, per role and pooled, with its exact one-sided 95% upper bound (Clopper-Pearson; 0 of n
gives 1 - 0.05^(1/n)). No rule reads it (review decision 1: reported whatever it is); pairing between arms rests on the
replay store, not on server determinism. Revision 3: the server identity is checked when the call starts (full) and
when it ends (light: pid, /props, argv; e4_run.identity_end_check); anything but "same" at the end exits 2 after the
output is written.
Run: e4 python -I -B repeat_check.py --campaign <smoke campaign> --base http://127.0.0.1:18082/v1 \
       --expect-server pins/server_expected.json --out <json>
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))


def binom_cdf(k: int, n: int, p: float) -> float:
    from math import comb

    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def upper95(k: int, n: int) -> float | None:
    """Exact one-sided 95% upper bound for a rate with k events in n trials (Clopper-Pearson)."""
    if n == 0:
        return None
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def view(resp: dict):
    m = resp["choices"][0]["message"]
    return {"content": m.get("content"),
            "tool_calls": [(t["function"]["name"], t["function"]["arguments"]) for t in (m.get("tool_calls") or [])]}


def pick(rows_by_role: dict, n_agent: int, n_user: int, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for role, n in (("agent", n_agent), ("user", n_user)):
        rows = sorted(rows_by_role.get(role, []))
        rng.shuffle(rows)
        out += [(role, *r) for r in rows[:n]]
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--expect-server", required=True)
    ap.add_argument("--n-agent", type=int, default=50)
    ap.add_argument("--n-user", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from e4_run import die, identity_end_check, local_only_env, server_health, server_identity

    server_health(a.base)
    local_only_env(a.base)          # found by the dry run of 2026-10-21 06:01Z: without it a set proxy blocks 127.0.0.1
    pins = json.loads(Path(a.expect_server).read_text())
    ident, diff = server_identity(a.base, pins)
    if diff:
        die(2, "the server is not the pinned one: " + "; ".join(diff))
    import litellm

    db = sqlite3.connect(str(E4 / "runs" / a.campaign / "replay.sqlite"))
    by = {}
    for key, role, req_s, resp_s in db.execute("SELECT key, role, request, response FROM calls"):
        by.setdefault(role, []).append((key, req_s, resp_s))
    todo = pick(by, a.n_agent, a.n_user, a.seed)
    res = {"agent": [0, 0], "user": [0, 0]}
    diffs = []
    for role, key, req_s, resp_s in todo:
        req = json.loads(req_s)
        r = litellm.completion(model=req["model"], messages=req["messages"], tools=req["tools"],
                               tool_choice=req["tool_choice"], api_base=a.base, api_key="sk-local-no-key", **req["args"])
        old, new = view(json.loads(resp_s)), view(r.to_dict())
        res[role][0] += 1
        if old != new:
            res[role][1] += 1
            diffs.append({"role": role, "key": key, "stored": old, "again": new})
    out = {"order": "seeded shuffle of both roles", "seed": a.seed, "available": {k: len(v) for k, v in by.items()},
           "per_role": {}, "server_identity": ident}
    for role, (n, d) in res.items():
        out["per_role"][role] = {"n": n, "different": d, "rate": d / n if n else None, "upper95": upper95(d, n)}
    n, d = sum(v[0] for v in res.values()), sum(v[1] for v in res.values())
    out["pooled"] = {"n": n, "different": d, "rate": d / n if n else None, "upper95": upper95(d, n)}
    out["enough"] = res["agent"][0] >= a.n_agent and res["user"][0] >= a.n_user
    out["examples"] = diffs[:5]
    end, kind = identity_end_check(a.base, pins, ident)
    out["server_identity_end"] = end
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("examples", "server_identity", "server_identity_end")},
                     indent=1))
    if kind != "same":
        die(2, f"the server identity changed during the repeat check ({kind}): " + "; ".join(end["differences"]))


if __name__ == "__main__":
    main()
