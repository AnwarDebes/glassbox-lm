#!/usr/bin/env python
"""Tool-call smoke test for a local llama-server (127.0.0.1 only; standard library only).

    python -I tool_smoke.py PORT OUT.jsonl

Sends the 50 prompts of tool_prompts.json to http://127.0.0.1:PORT/v1/chat/completions, each with all 10 tools,
temperature 0, thinking off (chat_template_kwargs enable_thinking false), at most 256 new tokens. Per prompt it
records: one tool call or not, the function name, whether the arguments parse as JSON, the required arguments present,
and the expected values (strings: substring, case-insensitive; numbers: equal). Writes one row per prompt to OUT.jsonl
(refuses to overwrite) and prints the summary.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ask(port: int, prompt: str, tools: list) -> dict:
    body = {"messages": [{"role": "system", "content": "You are a helpful assistant. Call a tool when one fits the request."},
                         {"role": "user", "content": prompt}],
            "tools": [{"type": "function", "function": t} for t in tools], "tool_choice": "auto",
            "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def value_ok(got, want) -> bool:
    if isinstance(want, (int, float)):
        try:
            return float(got) == float(want)
        except (TypeError, ValueError):
            return False
    return isinstance(got, str) and str(want).lower() in got.lower()


def main() -> None:
    port, out = int(sys.argv[1]), Path(sys.argv[2])
    assert not out.exists(), f"{out} exists"
    spec = json.loads((HERE / "tool_prompts.json").read_text())
    tools = spec["tools"]
    req = {t["name"]: t["parameters"].get("required", []) for t in tools}
    n = {"one_call": 0, "name": 0, "json": 0, "required": 0, "values": 0}
    rows = []
    t0 = time.time()
    for prompt, want, checks in spec["prompts"]:
        t1 = time.time()
        r = ask(port, prompt, tools)
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        row = {"prompt": prompt, "expect": want, "n_calls": len(calls), "content": msg.get("content"), "seconds": round(time.time() - t1, 2)}
        if len(calls) == 1:
            n["one_call"] += 1
            f = calls[0]["function"]
            row["name"] = f["name"]
            row["arguments"] = f.get("arguments")
            try:
                args = json.loads(f["arguments"]) if isinstance(f["arguments"], str) else f["arguments"]
                row["json_ok"] = isinstance(args, dict)
            except json.JSONDecodeError:
                args, row["json_ok"] = None, False
            if f["name"] == want:
                n["name"] += 1
            if row["json_ok"]:
                n["json"] += 1
                row["required_ok"] = all(k in args for k in req.get(f["name"], []))
                row["values_ok"] = f["name"] == want and all(value_ok(args.get(k), v) for k, v in checks)
                n["required"] += row["required_ok"]
                n["values"] += row["values_ok"]
        rows.append(row)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summ = {"prompts": len(rows), **n, "seconds": round(time.time() - t0, 1),
            "usage_last": r.get("usage"), "timings_last": r.get("timings")}
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
