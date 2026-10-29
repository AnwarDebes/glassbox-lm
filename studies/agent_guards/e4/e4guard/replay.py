"""Paired replay store for E4 (DESIGN.md 1.7): sha256(request) -> the server's reply, shared by all arms.

An arm sends the same requests as the arm before it until its first different guard verdict, so those replies come
from the store; later requests go to the server and are stored too. This pairs the arms and saves GPU time.

It wraps the `completion` name that tau2's `generate` calls (tau2.utils.llm_utils.completion, imported there from
litellm), so the agent, the user simulator and the NL judge all pass through it. The key covers what determines the
reply (model, messages, tools, tool_choice, temperature, seed, max_tokens, top_p, response_format, extra_body and
any other sampling argument) and leaves out transport-only arguments (api_base, api_key, timeout, num_retries), so a
port change does not break pairing. Modes: "record" (miss -> server, store), "replay_only" (miss -> ReplayMiss).
Each row also keeps who asked (the tau2 call site cannot be seen here, so the role is guessed from the system prompt:
"agent" if it holds the retail policy's <policy> tag, "user" if it holds the user guidelines, "judge" if it holds the
NL judge's "expected outcomes", else "other").
The llama-server README warns that results are not guaranteed bit-identical across batch sizes; the smoke step
measures how often a repeated request gives a different reply (scripts/repeat_check.py).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone

TRANSPORT_ONLY = {"api_base", "base_url", "api_key", "timeout", "num_retries", "request_timeout", "max_retries"}


class ReplayMiss(RuntimeError):
    pass


def request_key(model, messages, tools, tool_choice, kwargs) -> tuple[str, str]:
    body = {"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice,
            "args": {k: v for k, v in sorted(kwargs.items()) if k not in TRANSPORT_ONLY}}
    s = json.dumps(body, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest(), s


def guess_role(messages) -> str:
    """The role column only (never part of the key): agent, user, judge (tau2's NL-assertion judge) or other."""
    sysm = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
    if "<policy>" in sysm:
        return "agent"
    if "User Simulation Guidelines" in sysm or "<scenario>" in sysm:
        return "user"
    if "expected outcomes" in sysm:
        return "judge"
    return "other"


class ReplayStore:
    def __init__(self, path: str, mode: str = "record"):
        assert mode in ("record", "replay_only")
        self.path, self.mode = path, mode
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, role TEXT, request TEXT, response TEXT, "
                        "utc TEXT, ms REAL, arm TEXT)")
        self.db.commit()
        self.stats = {"hit": 0, "miss": 0, "stored": 0, "replay_miss": 0}
        self.arm = None

    def get(self, key):
        with self.lock:
            row = self.db.execute("SELECT response FROM calls WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, role, request_s, response_d, ms):
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?,?,?)",
                            (key, role, request_s, json.dumps(response_d, default=str),
                             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), ms, self.arm))
            self.db.commit()
            self.stats["stored"] += 1

    def wrap(self, real_completion):
        from litellm import ModelResponse

        def completion(model=None, messages=None, tools=None, tool_choice=None, **kwargs):
            key, s = request_key(model, messages, tools, tool_choice, kwargs)
            hit = self.get(key)
            if hit is not None:
                self.stats["hit"] += 1
                return ModelResponse(**hit)
            if self.mode == "replay_only":
                self.stats["replay_miss"] += 1          # counted, then raised (tau2 re-runs, then INFRASTRUCTURE_ERROR)
                raise ReplayMiss(key)
            self.stats["miss"] += 1
            t0 = time.perf_counter()
            resp = real_completion(model=model, messages=messages, tools=tools, tool_choice=tool_choice, **kwargs)
            self.put(key, guess_role(messages or []), s, resp.to_dict(), round(1000 * (time.perf_counter() - t0), 1))
            return resp

        return completion


def install(store: ReplayStore):
    """Route tau2's completion calls through the store (module-global rebinding; threads share it)."""
    import tau2.utils.llm_utils as lu

    real = lu.completion
    lu.completion = store.wrap(real)
    return real
