"""E4-0 smoke report (CPU; no LLM): every smoke measure of the review (the pre-freeze review,
E4 "smoke_must_measure" items 0-11), the stop/go gates and the GPU budget with its cut order, all fixed in
REGISTRATION_E4.md revision 2 BEFORE the smoke runs (revision 4: the smoke's extension rule, S0).

BLIND: this report never prints or writes a task reward, a pass rate or a judged score. Rewards are read only to test
that a replay-only re-run reproduces them (the output is a count of equal tasks). The smoke may change only the budget
(by the cut order below) and stop/go; nobody looks at the smoke's rewards before the freeze.

Smoke campaign layout (run_e4_gpu.sh STAGE=smoke, its own campaign SMOKE_<campaign> and store):
  A0_smoke/ A1_smoke/              A0 then A1 (A1 replays A0) on the smoke's train tasks, record mode, --blind
  STOCK_smoke/                     tau2's stock llm_agent, replay_only on that store (A0 equals stock)
  A0_smoke_replay/ A1_smoke_replay/ replay_only re-runs of A0 and A1 (determinism of the whole pipeline)
  repeat_check.json                >= 50 agent + >= 50 user stored requests sent again, shuffled (reported, no gate)
  judge_A0_smoke.json judge_A1_smoke.json   the judge's health on the smoke's train judge tasks (--health-only)
  confirmations.json               every write proposal of A0: the agent's last text-only message, the user's reply,
                                   the yes-matcher's verdict (for the label check); confirmations_labels.json holds
                                   the labels {"labeller": <who>, "labels": {"<id>": true|false}} (true = an explicit
                                   yes to the listed action); <who> is "a person" or "judged by an AI model, not a
                                   person" (revision 3: the label file records who labelled)
Gates (STOP if any fails; PENDING while S8 waits for its labels or S9 for the slot; GO only when every gate passes):
  S0 design: its own campaign (SMOKE_<campaign>, its own store); revision 4 (round-3 review m6), the EXTENSION RULE:
     the smoke's tasks are the first k train tasks in split_tasks.json's order, k = 10 when A0's label candidates on
     the first 10 are at least 10 (S8's minimum), else the smallest k > 10 whose first k tasks give at least 10, and
     at most 20 (k = 20 when even 20 give fewer: S8 then reads "not measured", NO-GO); A1 ran on the same tasks.
     run_e4_gpu.sh adds the tasks one at a time (--count-candidates, --next-task); S0 recomputes k from A0's
     trajectories and passes only when the tasks run are exactly that prefix;
  S1 server identity: every manifest of every smoke arm (manifest.json and the kept manifest.<k>.json of a resumed or
     extended call) carries an identity record with no difference (e4_run.py refuses otherwise) and an end-of-call
     record of kind "same" (revision 3), as do the repeat check and both judge-health outputs; /props total_slots 1
     and n_ctx per slot 65536;
  S2 ends: at least 80% of A0's simulations (8 of 10) end by user or agent stop; 0 infra ends in A0 and A1;
  S3 tool-call parse rate >= 0.90 over the agent's replies (parsed tool calls / (tool calls + tool calls written as
     text));
  S4 context: max prompt tokens + 2048 < n_ctx per slot, and 0 context-overflow errors;
  S5 the replay-only re-runs: 0 store misses; identical trajectories and identical DB rewards on every task;
  S6 STOCK replay-only: 0 store misses and every simulation ends normally;
  S7 live consistency on A0 and A1: code == veto on every verdict; live facts == recomputed facts on every call;
  S8 the yes-matcher's label check (revision 3; re-review item 4): fewer than 10 candidates = "not measured" = NO-GO
     (S8 cannot pass without its check); no label file, no valid labeller or any candidate without a label = PENDING;
     else precision >= 0.90 and recall >= 0.80 over ALL candidates, otherwise FAIL;
  S9 the budget fits the slot after the cut order (below); otherwise STOP and a decision is made before the freeze. The slot is set
     at the freeze (registration 11.3-11.4): the smoke itself runs without it (S9 PENDING).
Budget (DESIGN 2.0's formula: episodes x calls per episode x seconds per call / 60), from the smoke:
  c = mean LLM calls (agent model calls + user turns) per A0 smoke simulation; t = mean seconds per stored call
  (agent and user); f1 = A1's new calls / A0's calls; tJ = mean seconds per judge call (10 s if none);
  fixed now: fA = 0.30 (new-call share of each test-task replay arm), judge calls 40 + 40 + 11 per test arm.
  minutes = (114 c t + 114 c f1 t + sum over test arms of 40 c fA t + judge calls x tJ) / 60.
  Cut order when over the slot: cut A3t first, then A2 seeds 2-3; never A0, A1, A2 seed 1 or A3; still over: STOP.
--min-tasks, --min-candidates (other than 10) and --max-tasks (other than 20) are dry-run hooks (refused unless the
campaign starts with SMOKE_DRYRUN_).
Run: offline_env.sh .../taubench/.venv/bin/python -I -B smoke_report.py --campaign SMOKE_<c> [--slot-minutes N] \
       --out runs/SMOKE_<c>/smoke_report.json
     ... smoke_report.py --campaign SMOKE_<c> --count-candidates     (prints A0's label-candidate count; no file)
     ... smoke_report.py --campaign SMOKE_<c> --next-task <ids ...>  (prints the next train id in split order)
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import hashlib
import json
import sqlite3
import statistics as stx
import sys
from collections import Counter
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
sys.path.insert(1, str(E4 / "scripts"))

from e4guard import bits as B  # noqa: E402
from e4guard import rules  # noqa: E402
from e4guard.facts import (RETAIL_TOOLS, WRITE_TOOLS, canon_args, events_from_message_dicts, is_explicit_yes,  # noqa: E402
                           replay_facts)
from e4guard.runs import db_reward, end_class, guard_events, load_arm, termination, trajectory_sha256  # noqa: E402

N_CTX, MAX_TOKENS, TIMEOUT_S = 65536, 2048, 1800.0
FA, JUDGE_DEFAULT_S = 0.30, 10.0
S8_MIN_CANDIDATES, S0_MIN_TASKS, S0_MAX_TASKS = 10, 10, 20
LABELLERS = ("a person", "judged by an AI model, not a person")
DRY_LABELLER = "dry-run test labels: the matcher's own verdicts, not a check"
TEST_ARMS_ORDER = ["A2s1", "A3", "A2s2", "A2s3", "A3t"]
CUT_ORDER = [["A3t"], ["A3t", "A2s2", "A2s3"]]


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def dist(v):
    return {"n": len(v), "median": stx.median(v) if v else None, "p95": pct(v, 0.95), "max": max(v) if v else None,
            "mean": stx.mean(v) if v else None}


def budget(c, t, f1, tj, slot=None, fa=FA) -> dict:
    """GPU minutes by arm and the cut decision (pure; tests/test_smoke_report.py)."""
    per = {"A0": 114 * c * t, "A1": 114 * c * f1 * t}
    for a in TEST_ARMS_ORDER:
        per[a] = 40 * c * fa * t
    judge = {"A0": 40 * tj, "A1": 40 * tj, **{a: 11 * tj for a in TEST_ARMS_ORDER}}
    mins = {a: (per[a] + judge[a]) / 60 for a in per}
    full = sum(mins.values())
    plan = [("full", [], full)]
    for cut in CUT_ORDER:
        plan.append(("cut " + "+".join(cut), cut, full - sum(mins[a] for a in cut)))
    out = {"inputs": {"c": c, "t": t, "f1": f1, "tJ": tj, "fA": fa}, "minutes_by_arm": mins, "minutes_full": full,
           "plans": [{"plan": p, "cut": cut, "minutes": m} for p, cut, m in plan], "slot_minutes": slot}
    if slot is None:
        out.update({"decision": "pending (no slot given)", "cut": None, "fits": None})
        return out
    for p, cut, m in plan:
        if m <= slot:
            out.update({"decision": p, "cut": cut, "fits": True, "minutes_planned": m})
            return out
    out.update({"decision": "STOP: over the slot after every cut; a decision is made before the freeze",
                "cut": None, "fits": False})
    return out


def tool_health(store_rows) -> dict:
    """Agent replies in the store: parse rate and the other tool-call health counts."""
    c = Counter()
    for role, resp_s in store_rows:
        if role != "agent":
            continue
        c["replies"] += 1
        m = (json.loads(resp_s).get("choices") or [{}])[0].get("message") or {}
        content = m.get("content") or ""
        tcs = m.get("tool_calls") or []
        rc = m.get("reasoning_content") or ""
        if not content.strip() and not tcs:
            c["empty_replies"] += 1
        if rc.strip() or "<think>" in content:
            c["reasoning_text"] += 1
        if content.strip() and tcs:
            c["text_and_tool_call"] += 1
        if len(tcs) > 1:
            c["multi_call_messages"] += 1
        if "<tool_call" in content or "<function=" in content:
            c["tool_calls_written_as_text"] += 1
        for tc in tcs:
            c["tool_calls"] += 1
            fn = tc.get("function") or {}
            ok = True
            try:
                args = json.loads(fn.get("arguments") or "")
                ok = isinstance(args, dict)
            except (json.JSONDecodeError, TypeError):
                ok = False
            if not ok:
                c["unparsed_arguments"] += 1
            if fn.get("name") not in RETAIL_TOOLS:
                c["unknown_tool_names"] += 1
            c["parsed_tool_calls"] += int(ok and fn.get("name") in RETAIL_TOOLS)
    den = c["tool_calls"] + c["tool_calls_written_as_text"]
    return {**dict(c), "parse_rate": (c["parsed_tool_calls"] / den) if den else None}


def confirmation_candidates(arm: dict, events: dict) -> list[dict]:
    """Every write proposal of the arm (A0: all executed): the agent's last text-only message and the user's latest
    message, with the matcher's verdict. Built from the trajectory (calls in order) and the live verdicts."""
    out = []
    for t, sim in sorted(arm["sims"].items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        last_agent, last_user = None, None
        evs = events_from_message_dicts(sim.get("messages") or [])
        for e in evs:
            if e["kind"] == "agent_text" and not e.get("with_calls"):
                last_agent = e["text"]
            elif e["kind"] == "user_text":
                last_user = e["text"]
            elif e["kind"] == "agent_call" and e["name"] in WRITE_TOOLS:
                cid = hashlib.sha256(f"{t}|{e['name']}|{last_agent}|{last_user}".encode()).hexdigest()[:16]
                out.append({"id": cid, "task": t, "tool": e["name"], "agent_listing": last_agent,
                            "user_reply": last_user})
    seen, uniq = set(), []
    for c in out:
        if c["id"] not in seen:
            seen.add(c["id"])
            uniq.append(c)
    for c in uniq:
        c["matcher_yes"] = bool(is_explicit_yes(c["user_reply"]))
    return uniq


def train_order() -> list[str]:
    """The train ids in split_tasks.json's order (the smoke's first 10 and its extension come from it)."""
    return [str(t) for t in json.loads(Path(SPLIT).read_text())["train"]]


def next_smoke_task(current: list, order: list) -> str | None:
    """The next train id in split order that the smoke does not hold yet (None when there is none)."""
    cur = {str(t) for t in current}
    return next((t for t in order if t not in cur), None)


def extension_k(per_task: list, k0: int = S0_MIN_TASKS, min_c: int = S8_MIN_CANDIDATES,
                k_max: int = S0_MAX_TASKS) -> int | None:
    """The smoke's size by the extension rule (revision 4; round-3 review m6): k0 tasks; while the first k tasks give
    fewer than min_c label candidates and k < k_max, one more train task in split order. per_task: the candidates of
    each task run, in split order. None when fewer tasks ran than the rule needs."""
    k = k0
    while True:
        if k > len(per_task):
            return None
        if sum(per_task[:k]) >= min_c or k >= k_max:
            return k
        k += 1


def matcher_scores(cands: list[dict], labels_doc: dict | None, min_candidates: int = S8_MIN_CANDIDATES,
                   allow_dry_labeller: bool = False) -> dict:
    """S8 (revision 3; re-review item 4; the decision 6). "pass" is True, False (STOP) or None (PENDING):
    - fewer than min_candidates candidates: "not measured" = NO-GO (False): S8 cannot pass without its check;
    - no label file, a file that names no valid labeller, or any candidate without a label: PENDING (None);
    - else precision >= 0.90 and recall >= 0.80 over every candidate: pass (True); otherwise FAIL (False)."""
    n = len(cands)
    r = {"candidates": n, "min_candidates": min_candidates}
    if n < min_candidates:
        r.update({"gate": f"not measured (only {n} candidates; S8 needs {min_candidates}): NO-GO", "pass": False})
        return r
    if not labels_doc:
        r.update({"gate": "PENDING (no label file)", "pass": None})
        return r
    ok_l = LABELLERS + ((DRY_LABELLER,) if allow_dry_labeller else ())
    labeller = labels_doc.get("labeller") if isinstance(labels_doc, dict) else None
    labels = labels_doc.get("labels") if isinstance(labels_doc, dict) else None
    if labeller not in ok_l or not isinstance(labels, dict):
        r.update({"gate": f"PENDING (the label file must name the labeller, one of {list(ok_l)}, and hold 'labels')",
                  "pass": None})
        return r
    r["labeller"] = labeller
    missing = [c["id"] for c in cands if c["id"] not in labels]
    if missing:
        r.update({"gate": f"PENDING ({len(missing)} of {n} candidates have no label)", "pass": None,
                  "unlabelled": missing[:20]})
        return r
    lab = [(c["matcher_yes"], bool(labels[c["id"]])) for c in cands]
    tp = sum(1 for m, y in lab if m and y)
    fp = sum(1 for m, y in lab if m and not y)
    fn = sum(1 for m, y in lab if not m and y)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    ok = prec is not None and prec >= 0.90 and rec is not None and rec >= 0.80
    r.update({"labelled": len(lab), "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
              "gate": "pass" if ok else "FAIL", "pass": bool(ok)})
    return r


def rule_firing(events: dict) -> dict:
    """A0's write proposals (monitor mode) by tool: the enforced rules, revision 1's strict E4-R3a (monitor only),
    E4-R3b alone, every rule."""
    by_tool = {}
    for t, ev in events.items():
        for e in ev:
            if e["event"] != "verdict" or not e.get("is_write"):
                continue
            c = by_tool.setdefault(e["call"]["name"], Counter())
            c["proposals"] += 1
            c["any_rule"] += int(bool(e["code_rules"]))
            c["strict_r3a_monitor"] += int(bool((e.get("monitor") or {}).get("strict_r3a")))
            c["E4-R3a_text_literal"] += int("E4-R3a" in (e.get("veto_clauses") or []))
            c["E4-R3b"] += int("E4-R3b" in (e.get("veto_clauses") or []))
            for r in e["code_rules"]:
                c[r] += 1
    return {k: dict(v) for k, v in sorted(by_tool.items())}


def a1_behaviour(a0: dict, a1: dict, ev0: dict, ev1: dict) -> dict:
    den, rec, fb, extra = Counter(), 0, 0, []
    n_denied_writes = 0
    for t, ev in ev1.items():
        executed = set()
        sim = a1["sims"].get(t)
        if sim is not None:
            for _, e, f, _ in replay_facts(events_from_message_dicts(sim.get("messages") or [])):
                executed.add((e["name"], json.dumps(canon_args(e.get("args") or {}), sort_keys=True, default=str)))
        for e in ev:
            if e["event"] == "verdict" and not e["allow"]:
                for r in e["code_rules"]:
                    den[r] += 1
                if e.get("is_write"):
                    n_denied_writes += 1
                    key = (e["call"]["name"], json.dumps(canon_args(e["call"].get("args") or {}), sort_keys=True,
                                                         default=str))
                    rec += int(key in executed)
            elif e["event"] == "fallback":
                fb += 1
        calls0 = sum(1 for e in ev0.get(t, []) if e["event"] == "model_call")
        calls1 = sum(1 for e in ev if e["event"] == "model_call")
        extra.append(calls1 - calls0)
    return {"denials_by_rule": dict(den), "denied_writes": n_denied_writes,
            "denied_writes_executed_later": rec, "fallbacks": fb, "extra_agent_calls_per_task": dist(extra)}


def main():
    from e4guard.episodes import quiet_logs

    quiet_logs("WARNING")              # tau2 logs every replayed tool call at DEBUG otherwise
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--slot-minutes", type=float, default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--min-tasks", type=int, default=S0_MIN_TASKS)
    ap.add_argument("--min-candidates", type=int, default=S8_MIN_CANDIDATES)
    ap.add_argument("--max-tasks", type=int, default=S0_MAX_TASKS)
    ap.add_argument("--count-candidates", action="store_true", help="print A0_smoke's label-candidate count and stop")
    ap.add_argument("--next-task", nargs="*", default=None, help="print the next train id in split order not given")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dry = a.campaign.startswith("SMOKE_DRYRUN_")
    if not dry and (a.min_tasks != S0_MIN_TASKS or a.min_candidates != S8_MIN_CANDIDATES
                    or a.max_tasks != S0_MAX_TASKS or a.labels):
        sys.exit("REFUSING: --min-tasks, --min-candidates, --max-tasks and --labels are dry-run hooks "
                 "(SMOKE_DRYRUN_ campaigns)")
    camp = E4 / "runs" / a.campaign
    if a.next_task is not None:
        print(next_smoke_task(a.next_task, train_order()) or "")
        return
    if a.count_candidates:
        a0 = load_arm(camp / "A0_smoke")
        print(len(confirmation_candidates(a0, {})) if a0 else 0)
        return
    if not a.out:
        sys.exit("REFUSING: --out is required for the report")
    rep = {"campaign": a.campaign, "blind": True, "gates": {}, "reported": {}}
    g, rp = rep["gates"], rep["reported"]
    arms = {n: load_arm(camp / n) for n in ("A0_smoke", "A1_smoke", "STOCK_smoke", "A0_smoke_replay", "A1_smoke_replay")}
    summ = {n: (json.loads((camp / n / "run_summary.json").read_text()) if (camp / n / "run_summary.json").exists()
                else None) for n in arms}
    man = {n: (json.loads((camp / n / "manifest.json").read_text()) if (camp / n / "manifest.json").exists() else None)
           for n in arms}
    man_all = {n: [json.loads(m.read_text()) for m in sorted((camp / n).glob("manifest*.json"))]
               if (camp / n).exists() else [] for n in arms}       # every call's record (revision 4: extended calls)
    ev = {n: guard_events(camp / n) for n in arms}
    # 11 smoke design and the extension rule (revision 4; round-3 review m6)
    n_tasks = len(arms["A0_smoke"]["sims"]) if arms["A0_smoke"] else 0
    own = a.campaign.startswith("SMOKE_")
    order = train_order()
    run_tasks = (sorted(arms["A0_smoke"]["sims"], key=lambda t: order.index(t) if t in order else 10 ** 6)
                 if arms["A0_smoke"] else [])
    cands = confirmation_candidates(arms["A0_smoke"], ev["A0_smoke"]) if arms["A0_smoke"] else []
    per = Counter(c["task"] for c in cands)
    per_task = [per.get(t, 0) for t in run_tasks]
    k_rule = extension_k(per_task, a.min_tasks, a.min_candidates, a.max_tasks)
    prefix = run_tasks == order[:len(run_tasks)]
    same_a1 = bool(arms["A1_smoke"]) and sorted(arms["A1_smoke"]["sims"]) == sorted(run_tasks)
    g["S0_design"] = {"pass": bool(own and prefix and same_a1 and k_rule is not None and n_tasks == k_rule
                                   and a.min_tasks <= n_tasks <= a.max_tasks),
                      "own_campaign": own, "n_tasks": n_tasks, "min_tasks": a.min_tasks, "max_tasks": a.max_tasks,
                      "tasks": run_tasks, "split_order_prefix": prefix, "A1_same_tasks": same_a1,
                      "extension": {"rule": "k0 tasks; one more train task in split order while the first k give "
                                            "fewer than min_candidates label candidates and k < max_tasks",
                                    "k0": a.min_tasks, "min_candidates": a.min_candidates, "max_tasks": a.max_tasks,
                                    "candidates_per_task": per_task, "k_by_rule": k_rule,
                                    "tasks_added": max(0, n_tasks - a.min_tasks)}}
    # S1 server identity (item 0): every call's start and end record (revision 4: every manifest of every arm)
    ids = {n: (m or {}).get("server_identity") for n, m in man.items()}
    ends = {n: [((m or {}).get("server_identity_end") or {}).get("kind") for m in man_all[n]] or [None]
            for n in man_all}
    for extra in ("repeat_check.json", "judge_A0_smoke.json", "judge_A1_smoke.json"):     # revision 3: end checks
        q = camp / extra
        d = json.loads(q.read_text()) if q.exists() else {}
        ids[extra] = d.get("server_identity")
        ends[extra] = [(d.get("server_identity_end") or {}).get("kind")]
    starts = [m.get("server_identity") for n in man_all for m in man_all[n]] + \
        [ids[x] for x in ("repeat_check.json", "judge_A0_smoke.json", "judge_A1_smoke.json")]
    ok1 = all(man_all[n] for n in man_all) and all(i is not None and not i.get("differences") for i in starts)
    ok_end = all(k == "same" for v in ends.values() for k in v)
    p0 = (ids.get("A0_smoke") or {}).get("props") or {}
    g["S1_server_identity"] = {"pass": bool(ok1 and ok_end and p0.get("total_slots") == 1
                                            and p0.get("n_ctx_per_slot") == N_CTX),
                               "start_checks_clean": ok1, "end_checks": ends,
                               "calls_per_arm": {n: len(v) for n, v in man_all.items()},
                               "props": p0, "gguf_sha256": (ids.get("A0_smoke") or {}).get("gguf_sha256"),
                               "argv": ((ids.get("A0_smoke") or {}).get("processes") or [{}])[0].get("argv")}
    # store rows (items 1, 3)
    db = sqlite3.connect(str(camp / "replay.sqlite"))
    rows = db.execute("SELECT role, ms, response, arm FROM calls").fetchall()
    sec = {r: [ms / 1000 for role, ms, _, _ in rows if role == r] for r in ("agent", "user", "judge", "other")}
    tok = {r: {"prompt": [], "completion": []} for r in ("agent", "user", "judge", "other")}
    for role, _, resp_s, _ in rows:
        u = json.loads(resp_s).get("usage") or {}
        tok.setdefault(role, {"prompt": [], "completion": []})
        tok[role]["prompt"].append(u.get("prompt_tokens") or 0)
        tok[role]["completion"].append(u.get("completion_tokens") or 0)
    rp["seconds_per_call"] = {r: dist(v) for r, v in sec.items()}
    rp["tokens_per_call"] = {r: {k: dist(v) for k, v in d.items()} for r, d in tok.items()}
    max_prompt = max([x for d in tok.values() for x in d["prompt"]] or [0])
    overflow = 0
    for n in ("A0_smoke", "A1_smoke"):
        for s in (arms[n] or {"sims": {}})["sims"].values():
            if end_class(s) == "infra" and "context" in json.dumps(s.get("info") or {}).lower():
                overflow += 1
    g["S4_context"] = {"pass": bool(max_prompt + MAX_TOKENS < N_CTX and overflow == 0), "max_prompt_tokens": max_prompt,
                       "max_tokens": MAX_TOKENS, "n_ctx_per_slot": N_CTX, "overflow_errors": overflow}
    th = tool_health([(role, resp) for role, _, resp, _ in rows])
    rp["tool_call_health"] = th
    g["S3_parse_rate"] = {"pass": bool(th["parse_rate"] is not None and th["parse_rate"] >= 0.90),
                          "parse_rate": th["parse_rate"]}
    # item 2 per simulation
    per_sim = {}
    for n in ("A0_smoke", "A1_smoke"):
        if not summ[n]:
            continue
        per_sim[n] = [{k: r.get(k) for k in ("task", "termination", "duration_s", "agent_model_calls", "user_turns",
                                             "tau2_attempts")} for r in summ[n]["rows"]]
        for r in per_sim[n]:
            r["over_timeout"] = bool((r["duration_s"] or 0) >= TIMEOUT_S)
            r["tau2_reruns"] = (r["tau2_attempts"] or 1) - 1
    rp["per_simulation"] = per_sim
    # S2 ends
    e0 = Counter(end_class(s) for s in arms["A0_smoke"]["sims"].values()) if arms["A0_smoke"] else Counter()
    e1 = Counter(end_class(s) for s in arms["A1_smoke"]["sims"].values()) if arms["A1_smoke"] else Counter()
    stops0 = sum(1 for s in (arms["A0_smoke"] or {"sims": {}})["sims"].values()
                 if termination(s) in ("user_stop", "agent_stop"))
    g["S2_ends"] = {"pass": bool(n_tasks and stops0 >= 0.8 * n_tasks and e0.get("infra", 0) == 0
                                 and e1.get("infra", 0) == 0),
                    "A0_user_or_agent_stop": stops0, "A0_tasks": n_tasks, "A0_ends": dict(e0), "A1_ends": dict(e1)}
    # S5 replay-only re-runs (item 4) and the repeat check
    s5 = {}
    for orig, rr in (("A0_smoke", "A0_smoke_replay"), ("A1_smoke", "A1_smoke_replay")):
        if not arms[orig] or not arms[rr]:
            s5[rr] = {"pass": False, "why": "missing"}
            continue
        ts = sorted(arms[orig]["sims"])
        same_t = sum(1 for t in ts if t in arms[rr]["sims"]
                     and trajectory_sha256(arms[orig]["sims"][t]) == trajectory_sha256(arms[rr]["sims"][t]))
        same_r = sum(1 for t in ts if t in arms[rr]["sims"]
                     and db_reward(arms[orig]["sims"][t]) == db_reward(arms[rr]["sims"][t]))   # equality only
        miss = (summ[rr] or {}).get("store", {}).get("replay_miss")
        s5[rr] = {"tasks": len(ts), "identical_trajectories": same_t, "identical_db_rewards": same_r,
                  "store_misses": miss, "pass": bool(same_t == len(ts) and same_r == len(ts) and miss == 0)}
    g["S5_replay_only_rerun"] = {"pass": all(v["pass"] for v in s5.values()), **s5}
    rc = camp / "repeat_check.json"
    rp["repeat_check"] = ({k: v for k, v in json.loads(rc.read_text()).items() if k in ("per_role", "pooled", "enough",
                                                                                          "available")}
                          if rc.exists() else None)
    # S6 STOCK (item 5)
    st = arms["STOCK_smoke"]
    g["S6_A0_equals_stock"] = {"pass": bool(st and (summ["STOCK_smoke"] or {}).get("store", {}).get("replay_miss") == 0
                                            and all(end_class(s) == "normal" for s in st["sims"].values())),
                               "store": (summ["STOCK_smoke"] or {}).get("store")}
    # S7 live consistency (item 10)
    from summarize_e4 import live_consistency

    lc = {n: live_consistency(arms[n], ev[n]) for n in ("A0_smoke", "A1_smoke") if arms[n]}
    g["S7_live_consistency"] = {"pass": bool(lc and all(v["ok"] for v in lc.values())),
                                **{n: {k: v[k] for k in ("verdicts", "code_neq_veto", "executed_calls",
                                                         "fact_mismatches", "tasks_with_sequence_mismatch")}
                                   for n, v in lc.items()}}
    # item 6: rule firing on A0's writes; S8 the yes-matcher
    rp["rule_firing_A0_writes_by_tool"] = rule_firing(ev["A0_smoke"])
    (camp / "confirmations.json").write_text(json.dumps(cands, indent=1))
    lp = Path(a.labels) if a.labels else camp / "confirmations_labels.json"
    labels = json.loads(lp.read_text()) if lp.exists() else None
    ms = matcher_scores(cands, labels, a.min_candidates, allow_dry_labeller=dry)
    g["S8_yes_matcher"] = ms
    # item 7: A1 on the same tasks
    if arms["A0_smoke"] and arms["A1_smoke"]:
        rp["A1_behaviour"] = a1_behaviour(arms["A0_smoke"], arms["A1_smoke"], ev["A0_smoke"], ev["A1_smoke"])
        from summarize_e4 import db_hash

        ch = 0
        for t, s1 in arms["A1_smoke"]["sims"].items():
            s0 = arms["A0_smoke"]["sims"].get(t)
            task = arms["A1_smoke"]["tasks"].get(t)
            if s0 is not None and task is not None and db_hash(s1, task) != db_hash(s0, task):
                ch += 1
        rp["A1_behaviour"]["tasks_whose_final_db_differs_from_A0"] = ch
    # item 8: training-row yield
    from build_rows_from_results import select_training_rows, write_rows

    sp = json.loads(Path(SPLIT).read_text())
    split_of = {t: s for s in ("train", "test") for t in sp[s]}
    allr = []
    for n, src in (("A0_smoke", "qwen_A0"), ("A1_smoke", "qwen_A1")):
        if arms[n]:
            allr += write_rows({"tasks": list(arms[n]["tasks"].values()), "simulations": list(arms[n]["sims"].values())},
                               src, "train", split_of)
    kept, cnt = select_training_rows(allr)
    w0 = [r for r in allr if r["source"] == "qwen_A0"]
    rp["training_row_yield"] = {
        "A0_executed_writes_per_task": (len(w0) / n_tasks) if n_tasks else None,
        "A0_share_non_gold": (sum(r["label"] for r in w0) / len(w0)) if w0 else None,
        "A0_share_veto_clean": (sum(r["veto_clean"] for r in w0) / len(w0)) if w0 else None,
        "kept_rows_A0_A1": cnt.get("kept", 0), "kept_DENY": cnt.get("kept_DENY", 0),
        "projected_veto_clean_DENY_rows_74_tasks": (cnt.get("kept_DENY", 0) / n_tasks * 74) if n_tasks else None,
        "minimum": 10, "note": "a projection only; the binding test is stage train_guard's real count"}
    # item 9: judge health
    jh = {}
    for n in ("A0", "A1"):
        p = camp / f"judge_{n}_smoke.json"
        if p.exists():
            jh[n] = json.loads(p.read_text()).get("health")
    rp["judge_health"] = jh
    # budget (item 1) and S9
    calls = [(r.get("agent_model_calls") or 0) + (r.get("user_turns") or 0) for r in per_sim.get("A0_smoke", [])]
    c_bar = stx.mean(calls) if calls else 0.0
    stored = [ms / 1000 for role, ms, _, _ in rows if role in ("agent", "user")]
    t_bar = stx.mean(stored) if stored else 0.0
    n0 = sum(1 for _, _, _, arm in rows if arm == "A0_smoke")
    n1 = sum(1 for _, _, _, arm in rows if arm == "A1_smoke")
    f1 = (n1 / n0) if n0 else 1.0
    js = [x for h in jh.values() if h for x in (h.get("seconds_per_call") or [])]
    tj = stx.mean(js) if js else JUDGE_DEFAULT_S
    bud = budget(c_bar, t_bar, f1, tj, a.slot_minutes)
    rep["budget"] = bud
    g["S9_budget"] = {"pass": bud["fits"], "decision": bud["decision"], "cut": bud["cut"]}
    # decision
    hard = [k for k, v in g.items() if v.get("pass") is False]
    pending = [k for k, v in g.items() if v.get("pass") is None]
    rep["decision"] = ("STOP: " + ", ".join(hard)) if hard else (("PENDING: " + ", ".join(pending)) if pending else "GO")
    rep["no_rewards_reported"] = True
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"decision": rep["decision"], "gates": {k: v.get("pass") for k, v in g.items()},
                      "budget": {k: bud[k] for k in ("minutes_full", "decision", "cut")}}, indent=1, default=str))


SPLIT = (f"{_REPO_ROOT}/studies/agent_guards/taubench/data_untrusted/tau2-bench/extracted/"
         "tau2-bench-fc0055dc4e0a316c3f83133267fbd6faaa770992/data/tau2/domains/retail/split_tasks.json")

if __name__ == "__main__":
    main()
