"""A FAKE local OpenAI-compatible server for the E4 dry run (no model, no GPU; it is NOT llama-server and never stands
in for a result). It runs as its own process so the driver's server-identity check (argv, executable, /props, the
"GGUF" at model_path) is exercised end to end against a pins file written for the stub.

Replies are scripted from the task (tau2 retail tasks + e4guard/scripted.py plans):
  user simulator requests (system prompt with <scenario>): the task is found by its scenario text; the first reply is
    the plan's first user message; a reply to a listing (CONFIRM_MARK) is "Yes, please go ahead."; the final text
    (FINAL_MARK) gets "###STOP###"; anything else "OK, please continue.";
  agent requests (system prompt with <policy>; the task is found from the conversation's first user message): the
    k-th agent reply of the conversation is the k-th step of the task's plan (k = assistant messages in the request minus the greeting): a tool call or a text; the plan is the gold
    plan for task ids = 0 mod 3, a plausible WRONG write (no rule broken; DENY label) for 1 mod 3 and the
    "no confirmation" violation (E4-R3 in A1) for 2 mod 3 (falling back to the gold plan when a variant is missing);
  judge requests (system prompt with "expected outcomes"): all met when there is one assertion; an EMPTY results list
    otherwise (so the count-mismatch judge error is exercised).
GET /health, /v1/models, /props; POST /stub/health {"status": ...} sets the health answer; GET /stub/stats.
Revision 5: POST /stub/delay {"seconds": x} waits x seconds before every chat reply (the dry run's slot-end stop needs a
run that is still going when the operator's TERM arrives). Revision 6: POST /stub/props {...} overrides those /props
fields ({} restores them): the dry run's server that is not the pinned one at a run's next start (round-5 review Major 1).
Run: .../taubench/.venv/bin/python -I -B fake_llama_server.py --port P --model-path <file> --stub-tag <text>
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))

import argparse
import ast
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

E4 = Path(f"{_REPO_ROOT}/studies/agent_guards/e4")
sys.path.insert(0, str(E4))
TEMPLATE = "{# fake chat template of the E4 dry-run stub #}"
LOCK = threading.Lock()
STATE = {"health": {"status": "ok"}, "delay": 0.0, "props": {}, "posts": 0, "by_role": {}, "plans": {}, "tasks": None, "scen": None,
         "first_to_task": {}}


def tasks():
    if STATE["tasks"] is None:
        from tau2.runner.helpers import load_tasks

        ts = load_tasks("retail", "base")
        STATE["tasks"] = {str(t.id): t for t in ts}
        STATE["scen"] = {str(t.user_scenario): str(t.id) for t in ts}
    return STATE["tasks"]


def plan_of(tid: str) -> dict:
    if tid not in STATE["plans"]:
        from e4guard.scripted import gold_plan, plausible_wrong_writes, violation_plans

        t = tasks()[tid]
        g = gold_plan(t)
        info = {"plan": g["plan"], "first": g["first_message"], "variant": "gold"}
        k = int(tid) % 3
        if k == 1:
            w = plausible_wrong_writes(t)
            if w:
                info = {"plan": w[0]["plan"], "first": w[0]["first_message"], "variant": w[0]["name"]}
        elif k == 2:
            v = [p for p in violation_plans(t) if p["name"] == "R3_no_confirmation"]
            if v:
                info = {"plan": v[0]["plan"], "first": v[0]["first_message"], "variant": v[0]["name"]}
        STATE["plans"][tid] = info
        STATE["first_to_task"][info["first"]] = tid
    return STATE["plans"][tid]


def task_of_user_request(sysm: str) -> str | None:
    tasks()
    a, b = sysm.find("<scenario>\n"), sysm.find("\n</scenario>")
    if a < 0 or b < 0:
        return None
    return STATE["scen"].get(sysm[a + len("<scenario>\n"):b])


def reply_user(msgs, sysm):
    from e4guard.scripted import CONFIRM_MARK, FINAL_MARK

    tid = task_of_user_request(sysm)
    STATE["current"] = tid
    own = [m for m in msgs if m.get("role") == "assistant"]          # the user's own earlier messages
    if not own:
        return {"role": "assistant", "content": plan_of(tid)["first"] if tid else "Hi, I need help."}
    last = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")
    if FINAL_MARK in last or len(own) >= 30:
        return {"role": "assistant", "content": "Thanks, that's all. ###STOP###"}
    if CONFIRM_MARK in last:
        return {"role": "assistant", "content": "Yes, please go ahead."}
    return {"role": "assistant", "content": "OK, please continue."}


def reply_agent(msgs):
    """The task is found from the conversation's first user message (the stub wrote it), so a replayed user turn
    that never reached the stub cannot confuse it; the last user-simulator request is the fallback."""
    from e4guard.scripted import FINAL_MARK

    first = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
    tid = STATE["first_to_task"].get(first, STATE.get("current"))
    k = sum(1 for m in msgs if m.get("role") == "assistant") - 1     # minus the greeting
    plan = plan_of(tid)["plan"] if tid else []
    if k >= len(plan):
        return {"role": "assistant", "content": f"{FINAL_MARK} That is everything I can do. Goodbye."}
    st = plan[k]
    if st[0] == "text":
        return {"role": "assistant", "content": st[1]}
    if st[0] == "bundle":
        st = st[1][0]
    return {"role": "assistant", "content": None, "tool_calls": [{"id": f"call_{tid}_{k}", "type": "function",
            "function": {"name": st[1], "arguments": json.dumps(st[2])}}]}


def reply_judge(msgs):
    up = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
    tail = up.split("expectedOutcomes:", 1)[-1].strip()
    try:
        items = ast.literal_eval(tail)
    except (ValueError, SyntaxError):
        items = []
    if len(items) == 1:
        res = [{"expectedOutcome": items[0], "reasoning": "stub", "metExpectation": True}]
    else:
        res = []
    return {"role": "assistant", "content": json.dumps({"results": res})}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            h = STATE["health"]
            return self._send(h, 200 if h.get("status") == "ok" else 503)
        if self.path.rstrip("/") == "/v1/models":
            return self._send({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
        if self.path == "/props":
            return self._send({"default_generation_settings": {"n_ctx": 65536, "params": {}}, "total_slots": 1,
                               "model_alias": "fake-model", "model_path": ARGS.model_path, "chat_template": TEMPLATE,
                               "build_info": "b1-4c3c581", "is_sleeping": False, **STATE["props"]})
        if self.path == "/stub/stats":
            return self._send({"posts": STATE["posts"], "by_role": STATE["by_role"]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode() or "{}")
        if self.path == "/stub/health":
            STATE["health"] = body
            return self._send({"ok": True})
        if self.path == "/stub/delay":
            STATE["delay"] = float(body.get("seconds") or 0.0)
            return self._send({"ok": True})
        if self.path == "/stub/props":
            STATE["props"] = dict(body)
            return self._send({"ok": True})
        if STATE["delay"] > 0:
            time.sleep(STATE["delay"])
        with LOCK:
            STATE["posts"] += 1
            msgs = body.get("messages", [])
            sysm = next((m.get("content") or "" for m in msgs if m.get("role") == "system"), "")
            if "expected outcomes" in sysm:
                role, msg = "judge", reply_judge(msgs)
            elif "<policy>" in sysm:
                role, msg = "agent", reply_agent(msgs)
            else:
                role, msg = "user", reply_user(msgs, sysm)
            STATE["by_role"][role] = STATE["by_role"].get(role, 0) + 1
        pt = max(1, len(json.dumps(msgs)) // 4)
        self._send({"id": f"stub-{STATE['posts']}", "object": "chat.completion", "created": int(time.time()),
                    "model": "fake-model",
                    "choices": [{"index": 0, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop",
                                 "message": msg}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": 5, "total_tokens": pt + 5}})


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stub-tag", default="e4-dry-run")
    ARGS = ap.parse_args()
    from e4guard.episodes import quiet_logs

    quiet_logs("ERROR")
    tasks()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), H)
    srv.serve_forever()


ARGS = None
if __name__ == "__main__":
    main()
